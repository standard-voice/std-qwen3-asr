# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init configuration for the Qwen3-ASR engine.

The config captures *where* to reach the backend and *how* to authenticate, plus
a few deployment knobs. Per the standard's init/runtime boundary (spec IC.7),
per-request decoding behavior lives in :mod:`std_qwen3_asr.params`
(``ProviderParams``), not here.

Security (spec IC.3, "Security by default"): ``api_key`` is a ``SecretStr``
(inherited from :class:`CredentialsConfigMixin`) so it is never logged or echoed
as plaintext. ``base_url`` is validated to be HTTP(S) at construction; for the
public DashScope endpoints we additionally require HTTPS, while a local vLLM
``base_url`` may be plain ``http://`` (loopback/LAN). SSRF protection for the
*audio fetch* path (when the backend pulls a URL we forward) is the standard
layer's job via ``allow_private_urls`` (spec AI R5); this validation is about the
*control-plane* endpoint we connect to.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from standard_asr.engine import BaseConfig, CredentialsConfigMixin, LanguageConfigMixin
from standard_asr.exceptions import ConfigError

#: Backend transport selector. ``"vllm"`` speaks the OpenAI-compatible
#: ``/v1/audio/transcriptions`` (batch) + SSE/WebSocket (streaming) protocol of a
#: self-hosted vLLM server. ``"dashscope"`` speaks the hosted Alibaba Cloud
#: chat-completions protocol for the ``qwen3-asr-flash`` service.
Backend = Literal["vllm", "dashscope"]

#: Streaming transport for the vLLM backend. ``"realtime"`` uses the WebSocket
#: Realtime API (``/v1/realtime``), which is the path Qwen3-ASR genuinely streams
#: through. ``"sse"`` uses ``stream=true`` on the transcription endpoint
#: (the generic vLLM mechanism). See README for the trade-offs.
StreamTransport = Literal["realtime", "sse"]


class Qwen3ASRConfig(LanguageConfigMixin, CredentialsConfigMixin, BaseConfig[Literal["qwen3-asr"]]):
    """Connection + deployment configuration for the Qwen3-ASR backend.

    Args:
        engine: Discriminator value for the engine (always ``"qwen3-asr"``).
        backend: Which backend protocol to speak (``"vllm"`` or ``"dashscope"``).
        api_key: Secret API key / token (inherited; ``SecretStr``). Required for
            DashScope; optional for an unauthenticated local vLLM server.
        base_url: Base URL of the backend (inherited). For vLLM this is the
            OpenAI-compatible root, e.g. ``http://localhost:8000/v1``. For
            DashScope it defaults to the international compatible-mode endpoint.
        region: Optional routing hint (inherited; unused for vLLM).
        org_id: Optional org id (inherited; unused).
        default_language: Default language (BCP-47 or ``"auto"``; inherited).
            Defaults to ``"auto"`` -- Qwen3-ASR's natural multilingual
            auto-detection (the engine reports the detected language back). The
            engine exposes a language axis (``selectable_languages``), so the
            standard requires this to be set (spec IC.6 / LANG R1).
        default_candidate_languages: Default candidate-language list (inherited;
            unused -- Qwen does not consume a candidate list).
        connect_timeout: TCP/handshake connect timeout in seconds.
        read_timeout: Per-read timeout in seconds for the batch request and for
            each streaming chunk (streaming uses it as the inter-chunk idle
            bound).
        stream_transport: For the vLLM backend, which streaming transport to use
            (``"realtime"`` WebSocket or ``"sse"``).
        realtime_segment_seconds: Advisory segment length (seconds) reported by
            the vLLM Realtime endpoint. vLLM hardcodes 5.0s today (see findings);
            this is documentation-only and does not change wire behavior.
        verify_tls: Whether to verify TLS certificates (default ``True``; set
            ``False`` only for a self-signed local dev server, an explicit
            opt-out of the secure default).
    """

    engine: Literal["qwen3-asr"] = "qwen3-asr"
    # Default to Qwen's multilingual auto-detection. Required because the engine
    # declares a language axis (spec IC.6 / LANG R1); "auto" is in
    # selectable_languages.
    default_language: str | None = Field(
        default="auto", description="Default language (BCP-47 or 'auto')."
    )
    backend: Backend = Field(
        default="vllm",
        description="Backend protocol: 'vllm' (self-hosted) or 'dashscope' (cloud Flash).",
    )
    connect_timeout: float = Field(default=10.0, gt=0.0, description="Connect timeout in seconds.")
    read_timeout: float = Field(
        default=60.0, gt=0.0, description="Read timeout in seconds (per chunk when streaming)."
    )
    stream_transport: StreamTransport = Field(
        default="realtime",
        description="vLLM streaming transport: 'realtime' (WebSocket) or 'sse'.",
    )
    realtime_segment_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Advisory vLLM Realtime segment length in seconds (doc-only).",
    )
    verify_tls: bool = Field(
        default=True, description="Verify TLS certs (set False only for local dev)."
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        """Validate the control-plane base URL scheme and shape.

        Args:
            value: The configured base URL, or ``None`` (use the backend default).

        Returns:
            The base URL with any trailing slash trimmed, or ``None``.

        Raises:
            ConfigError: If the URL is malformed or uses a non-HTTP(S) scheme.
        """
        if value is None:
            return None
        trimmed = value.rstrip("/")
        parsed = urlparse(trimmed)
        if parsed.scheme not in ("http", "https"):
            raise ConfigError(
                f"base_url must be an http(s) URL; got scheme {parsed.scheme!r} in {value!r}."
            )
        if not parsed.netloc:
            raise ConfigError(f"base_url must include a host; got {value!r}.")
        return trimmed

    @model_validator(mode="after")
    def _validate_backend_combination(self) -> Qwen3ASRConfig:
        """Enforce backend-specific invariants.

        DashScope is a public cloud endpoint, so a non-loopback ``base_url`` MUST
        be HTTPS (no plaintext credentials over the public internet). Plain
        ``http://`` is allowed only for loopback hosts (``localhost`` /
        ``127.0.0.1`` / ``::1``), where traffic never leaves the machine -- this
        keeps the secure default while permitting a local mock/proxy in front of
        DashScope (and integration tests). A local vLLM server may always use
        plain ``http://``.

        Returns:
            The validated config instance.

        Raises:
            ConfigError: If a non-loopback DashScope ``base_url`` is not HTTPS.
        """
        if self.backend == "dashscope" and self.base_url is not None:
            parsed = urlparse(self.base_url)
            host = (parsed.hostname or "").lower()
            is_loopback = host in {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme != "https" and not is_loopback:
                raise ConfigError(
                    "DashScope base_url MUST use https:// for non-loopback hosts (it carries "
                    f"credentials over the public internet); got {self.base_url!r}."
                )
        return self
