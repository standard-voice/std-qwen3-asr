# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Qwen3-ASR Standard ASR engine (thin HTTP/WS adapter over a vLLM backend).

``Qwen3ASR`` subclasses :class:`~standard_asr.EngineBase` and implements only the
two engine hooks -- ``_transcribe`` (batch) and ``_start_transcription``
(streaming). The standard layer gives us audio negotiation/conversion, parameter
gating, language-axis resolution, the CLI, the web server, and the compliance
suite for free.

Adapter, not fork (see ``VERIFICATION.md`` STEP 1): this engine is a pure client.
It connects to a vLLM OpenAI-compatible server (the user's required backend) or
the DashScope Flash cloud service, selected by config. It never imports vLLM,
torch, or Qwen's inference code -- those run on the remote host. This keeps the
plugin's dependency + license footprint tiny (Apache-2.0 adapter) and isolated
from the model runtime (spec G.4.2).

Lazy purity (spec IC.9): ``__init__`` captures config only -- no network, no
client construction. The backend client is created on first use in
``_ensure_model_loaded`` (it binds an ``httpx.AsyncClient``; streaming clients are
created per session on the session's event loop, per the sync-bridge contract
ST §6.5).
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from standard_asr import (
    RuntimeParams,
    TranscriptionResult,
)
from standard_asr.audio_format import AudioFormat
from standard_asr.capabilities import DeclaredCapabilities
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    PreparedAudio,
)
from standard_asr.exceptions import ConfigError, TranscriptionError
from standard_asr.language import effective_language
from standard_asr.results import Diagnostic
from standard_asr.runtime_params import ProviderParams
from standard_asr.streaming import TranscriptionSession

from ._audio import float32_to_pcm16, wrap_pcm16_wav
from .backends.base import Backend, BatchRequest, BatchResult, StreamRequest
from .backends.dashscope import DEFAULT_DASHSCOPE_BASE_URL, DashScopeBackend
from .backends.vllm import DEFAULT_VLLM_BASE_URL, VLLMBackend
from .capabilities import QWEN3_ASR_CAPABILITIES
from .config import Qwen3ASRConfig
from .languages import from_backend_language, to_dashscope_code
from .params import Qwen3ASRParams
from .properties import (
    Qwen3ASR06BProperties,
    Qwen3ASR17BProperties,
    Qwen3ASRProperties,
)

#: Default DashScope model id for the hosted Flash preset.
_DASHSCOPE_FLASH_MODEL = "qwen3-asr-flash"


class Qwen3ASR(EngineBase):
    """Standard ASR adapter for the ``qwen3-asr/flash`` preset (hosted DashScope).

    This is the canonical preset. The open-weight vLLM presets subclass it and
    override only ``backend_model`` + ``properties`` (spec IC.7: a preset is a
    distinct entry point / class, not an init ``model`` field).

    Args:
        **kwargs: Configuration overrides for :class:`Qwen3ASRConfig`.
    """

    #: The remote model id this preset requests from its backend. Overridden per
    #: preset. For the Flash preset this is the DashScope service id; for the
    #: vLLM presets it is the HuggingFace checkpoint id ``vllm serve`` loaded.
    backend_model: ClassVar[str] = _DASHSCOPE_FLASH_MODEL

    #: Which backend this preset defaults to when config does not override it.
    default_backend: ClassVar[str] = "dashscope"

    properties: ClassVar[BaseProperties] = Qwen3ASRProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = QWEN3_ASR_CAPABILITIES
    provider_params_type: ClassVar[type[ProviderParams] | None] = Qwen3ASRParams
    config_type: ClassVar[type[BaseConfig[str]] | None] = Qwen3ASRConfig

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; no network).

        Config is built via ``from_env`` (spec IC.4): unset fields fall back to
        ``STANDARD_ASR_QWEN3_ASR__*`` environment variables, with explicit
        ``kwargs`` winning. ``api_key`` is wrapped in ``SecretStr`` by
        construction -- never handled as plaintext.

        When ``backend`` is not explicitly chosen, the preset's
        :attr:`default_backend` is applied so the hosted preset defaults to
        DashScope and the open-weight presets default to vLLM.

        Args:
            **kwargs: Configuration overrides.
        """
        explicit = dict(kwargs)
        explicit.setdefault("backend", type(self).default_backend)
        self.config = Qwen3ASRConfig.from_env("qwen3-asr", **explicit)
        self._backend: Backend | None = None

    # ------------------------------------------------------------------ #
    # Backend lifecycle
    # ------------------------------------------------------------------ #
    def _resolved_base_url(self, config: Qwen3ASRConfig) -> str:
        """Resolve the backend base URL (config override or backend default).

        Args:
            config: The engine config.

        Returns:
            The effective base URL.
        """
        if config.base_url is not None:
            return config.base_url
        if config.backend == "dashscope":
            return DEFAULT_DASHSCOPE_BASE_URL
        return DEFAULT_VLLM_BASE_URL

    def _build_backend(self) -> Backend:
        """Construct the active backend client from config.

        Returns:
            A backend transport implementing :class:`Backend`.

        Raises:
            ConfigError: If a required credential is missing for the chosen
                backend.
        """
        config = cast(Qwen3ASRConfig, self.config)
        api_key = config.api_key.get_secret_value() if config.api_key is not None else None
        base_url = self._resolved_base_url(config)

        if config.backend == "dashscope":
            if not api_key:
                raise ConfigError(
                    "The DashScope backend requires an api_key. Set it explicitly "
                    "or via STANDARD_ASR_QWEN3_ASR__API_KEY / DASHSCOPE_API_KEY."
                )
            return DashScopeBackend(
                base_url=base_url,
                model=type(self).backend_model,
                api_key=api_key,
                connect_timeout=config.connect_timeout,
                read_timeout=config.read_timeout,
                verify_tls=config.verify_tls,
            )
        return VLLMBackend(
            base_url=base_url,
            model=type(self).backend_model,
            api_key=api_key,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            stream_transport=config.stream_transport,
            verify_tls=config.verify_tls,
        )

    def _ensure_model_loaded(self) -> None:
        """Create the backend client lazily (spec IC.9: not in ``__init__``).

        Raises:
            ConfigError: If a required credential is missing.
        """
        if self._backend is None:
            self._backend = self._build_backend()

    def prepare(self) -> None:
        """Construct the backend client without transcribing (warm-up hook).

        Raises:
            ConfigError: If a required credential is missing.
        """
        self._ensure_model_loaded()

    # ------------------------------------------------------------------ #
    # Shared parameter mapping
    # ------------------------------------------------------------------ #
    def _resolve_backend_language(self, request_language: str | None) -> str | None:
        """Resolve the request language to a backend language token.

        Applies the standard ``effective_language`` resolution (request override
        vs ``default_language``), translates a concrete BCP-47 tag to the
        backend's ISO code, and returns ``None`` for auto-detect (or for a tag
        outside the supported inventory -- omit and let the engine auto-detect
        rather than send an invalid code).

        Args:
            request_language: The per-request ``language`` (BCP-47 / ``"auto"`` /
                ``None``).

        Returns:
            A backend ISO language code, or ``None`` to auto-detect.
        """
        config = cast(Qwen3ASRConfig, self.config)
        resolved = effective_language(
            request_language,
            config.default_language,
            has_language_axis=self.properties.has_language_axis,
            runtime_override_supported=True,
        )
        if not resolved or resolved == "auto":
            return None
        return to_dashscope_code(resolved)

    def _provider(self, params: RuntimeParams) -> Qwen3ASRParams:
        """Return the engine-specific params (or defaults when unset).

        Args:
            params: The gated runtime params.

        Returns:
            A :class:`Qwen3ASRParams` -- the caller's, or a default instance.
        """
        provider = params.provider_params
        if provider is None:
            return Qwen3ASRParams()
        return cast(Qwen3ASRParams, provider)

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def _encode_audio(self, prepared: PreparedAudio) -> tuple[bytes, str | None]:
        """Turn negotiated audio into encoded bytes + a container hint.

        The engine accepts ``ENCODED_FILE`` / ``ENCODED_BYTES`` / ``ARRAY``. For
        an array, the standard layer already encoded it to canonical WAV when it
        delivered ``ENCODED_BYTES`` -- but if an array reaches us directly we
        encode it ourselves to WAV.

        Args:
            prepared: Engine-ready audio in one of the accepted shapes.

        Returns:
            ``(audio_bytes, container_hint)``.

        Raises:
            TranscriptionError: If no usable payload is present.
        """
        if prepared.data is not None:
            return prepared.data, prepared.container
        if prepared.path is not None:
            with open(prepared.path, "rb") as fh:
                return fh.read(), prepared.container
        if prepared.array is not None:
            pcm = float32_to_pcm16(prepared.array)
            rate = prepared.sample_rate or self.properties.native_sample_rate
            return wrap_pcm16_wav(pcm, sample_rate=rate), "wav"
        raise TranscriptionError(  # pragma: no cover - negotiation guarantees a payload
            "No audio payload present after negotiation."
        )

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Transcribe negotiated audio via the backend (batch).

        Args:
            prepared: Engine-ready audio.
            params: Gated runtime parameters.

        Returns:
            A Standard ASR transcription result.

        Raises:
            TranscriptionError: On any backend/transport failure (spec Runtime
                R7: a portable error type across engines, native cause preserved).
        """
        self._ensure_model_loaded()
        backend = cast(Backend, self._backend)
        provider = self._provider(params)
        audio_bytes, container = self._encode_audio(prepared)
        request = BatchRequest(
            audio=audio_bytes,
            container=container,
            language=self._resolve_backend_language(params.language),
            prompt=params.prompt,  # Qwen "context" via the portable prompt channel.
            temperature=provider.temperature,
            top_p=provider.top_p,
            max_completion_tokens=provider.max_completion_tokens,
            enable_itn=provider.enable_itn,
            emotion=provider.emotion,
        )
        try:
            result = _run_batch(backend, request)
        except Exception as exc:  # noqa: BLE001 - normalized to the standard contract
            raise TranscriptionError(
                f"Qwen3-ASR transcription failed: {type(exc).__name__}."
            ) from exc
        return _to_result(result)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        """Construct the streaming session (the base already gated everything).

        The base ``start_transcription`` template ran input exclusion, language
        config validation, the fail-closed wire-format check, parameter gating
        (incl. ``provider_params`` swap-safety), and language resolution before
        calling us. We just build the session with the frozen params.

        Both streaming shapes are supported and routed through the backend's
        incremental stream:

        * **Incremental input** (``audio_format=...``): the session feeds the
          backend from the fed PCM queue. This is the primary, vLLM-Realtime path.
        * **Whole-input + streaming output** (``audio=...``, spec §7.3): the base
          negotiated the whole input into ``prepared_audio``; we convert it to
          PCM16 once and the session feeds the backend from that fixed buffer.

        Args:
            gated_params: The gated, frozen runtime parameters (spec R5).
            audio_format: The incremental wire format (encoding/rate/channels), or
                ``None`` for the whole-input path.
            prepared_audio: Negotiated whole input for the streaming-output path,
                or ``None`` for the incremental path.

        Returns:
            A :class:`~std_qwen3_asr.session.Qwen3ASRSession`.
        """
        from .session import Qwen3ASRSession

        self._ensure_model_loaded()
        backend = cast(Backend, self._backend)
        provider = self._provider(gated_params)
        # The wire rate is the engine's required rate (the base wire-format guard
        # already enforced this); fall back to native if no format was given.
        sample_rate = (
            audio_format.sample_rate
            if audio_format is not None
            else self.properties.native_sample_rate
        )
        whole_input_pcm: bytes | None = None
        if prepared_audio is not None:
            whole_input_pcm, sample_rate = self._prepared_to_pcm16(prepared_audio)
        request = StreamRequest(
            language=self._resolve_backend_language(gated_params.language),
            prompt=gated_params.prompt,
            temperature=provider.temperature,
            sample_rate=sample_rate,
            enable_itn=provider.enable_itn,
        )
        return Qwen3ASRSession(backend=backend, request=request, whole_input_pcm=whole_input_pcm)

    def _prepared_to_pcm16(self, prepared: PreparedAudio) -> tuple[bytes, int]:
        """Convert negotiated whole-input audio to raw PCM16 + its sample rate.

        Args:
            prepared: The negotiated whole input (array; or encoded bytes/path
                that we decode via the standard loader).

        Returns:
            ``(pcm16_bytes, sample_rate)``.

        Raises:
            TranscriptionError: If the audio cannot be decoded to PCM.
        """
        native = self.properties.native_sample_rate
        samples: np.ndarray[Any, np.dtype[np.float32]]
        if prepared.array is not None:
            samples = prepared.array
            rate = prepared.sample_rate or native
        else:
            # Decode encoded bytes/path to a float32 mono array at the native rate
            # via the standard loader (it resamples to ``target_sr`` for us).
            from standard_asr.utils.audio_loader import (
                load_audio_from_bytes,
                load_audio_from_path,
            )

            if prepared.data is None and prepared.path is None:  # pragma: no cover - defensive
                raise TranscriptionError("No audio payload present for streaming whole input.")
            try:
                if prepared.data is not None:
                    samples = load_audio_from_bytes(prepared.data, target_sample_rate=native)
                else:
                    assert prepared.path is not None  # narrowed above
                    samples = load_audio_from_path(prepared.path, target_sample_rate=native)
            except Exception as exc:  # noqa: BLE001
                raise TranscriptionError(
                    f"Failed to decode whole-input audio for streaming: {type(exc).__name__}."
                ) from exc
            rate = native
        return float32_to_pcm16(samples), int(rate)


def _to_result(result: BatchResult) -> TranscriptionResult:
    """Map a normalized backend result to a ``TranscriptionResult``.

    Args:
        result: A :class:`~std_qwen3_asr.backends.base.BatchResult`.

    Returns:
        The Standard ASR result, with engine-specific fields (emotion, raw) in
        ``extra`` (never in standardized ``metadata``; spec TR.1).
    """
    extra: dict[str, Any] = dict(result.raw)
    if result.emotion:
        extra["emotion"] = result.emotion
    diagnostics: list[Diagnostic] = []
    return TranscriptionResult(
        text=result.text,
        detected_language=from_backend_language(result.detected_language),
        duration=result.duration,
        diagnostics=diagnostics,
        extra=extra,
    )


def _run_batch(backend: Backend, request: BatchRequest) -> BatchResult:
    """Run a batch transcription synchronously on a dedicated event loop.

    The batch path is synchronous (``transcribe``), but the backend client is
    async-first. We must not touch an ambient running loop (the sync-bridge
    contract, ST §6.5, and general safety), so we run on a fresh event loop owned
    by this call. The backend's HTTP client is bound to that loop, so we close it
    *inside* the same loop before tearing the loop down -- otherwise its sockets
    would outlive the loop and leak (a ResourceWarning). The backend recreates a
    fresh client on the next call. ``transcribe_async`` (from the base) gives true
    async callers a non-blocking path that reuses the caller's loop instead.

    Args:
        backend: The active backend.
        request: The normalized batch request.

    Returns:
        The normalized batch result.
    """
    import asyncio

    async def _run() -> BatchResult:
        try:
            return await backend.transcribe(request)
        finally:
            await backend.aclose()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# --------------------------------------------------------------------------- #
# Open-weight presets (vLLM). Each preset is its own entry point / class
# (spec IC.7). A preset overrides only the backend model id, default backend,
# and properties; everything else is inherited.
# --------------------------------------------------------------------------- #
class Qwen3ASR17B(Qwen3ASR):
    """The ``qwen3-asr/1.7b`` preset (open weights, served by self-hosted vLLM)."""

    backend_model: ClassVar[str] = "Qwen/Qwen3-ASR-1.7B"
    default_backend: ClassVar[str] = "vllm"
    properties: ClassVar[BaseProperties] = Qwen3ASR17BProperties()


class Qwen3ASR06B(Qwen3ASR):
    """The ``qwen3-asr/0.6b`` preset (open weights, served by self-hosted vLLM)."""

    backend_model: ClassVar[str] = "Qwen/Qwen3-ASR-0.6B"
    default_backend: ClassVar[str] = "vllm"
    properties: ClassVar[BaseProperties] = Qwen3ASR06BProperties()
