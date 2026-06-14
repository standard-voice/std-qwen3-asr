# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""DashScope (Alibaba Cloud) backend for the hosted ``qwen3-asr-flash`` service.

This backend speaks the OpenAI-compatible *chat-completions* protocol that
DashScope exposes for Qwen3-ASR-Flash (the ``compatible-mode/v1`` endpoint). The
audio is sent inline as a base64 data URI in an ``input_audio`` content part, and
Qwen-specific options ride in ``asr_options``.

Batch response: the transcript is in ``choices[0].message.content``; the detected
language + emotion are in ``choices[0].message.annotations[0]``
(``{type: "audio_info", language, emotion}``).

Streaming: ``stream=true`` yields standard chat-completion SSE delta chunks
(``choices[0].delta.content`` fragments). This is *token streaming of the
transcript*, append-only -- mapped to the same ``partial`` accumulation +
``stable_until=0`` model as the vLLM streaming path.

Reachability note: DashScope is a paid cloud service and requires an API key.
This backend is implemented and unit-tested against a local fake, but live
verification needs a real ``DASHSCOPE_API_KEY`` (see VERIFICATION.md).
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from .._audio import wrap_pcm16_wav
from .._json import as_object, first_object, get_float, get_list, get_object, get_str, loads_object
from .base import BatchRequest, BatchResult, StreamRequest, TranscriptDelta

#: International compatible-mode endpoint root (OpenAI SDK ``base_url``). The CN
#: root (``https://dashscope.aliyuncs.com/compatible-mode/v1``) is selectable via
#: config ``base_url``.
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

#: Container -> MIME for the inline data URI.
_MIME_BY_CONTAINER: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


def _data_uri(audio: bytes, container: str | None) -> str:
    """Build a base64 ``data:`` URI for the audio payload.

    Args:
        audio: Encoded audio bytes.
        container: Best-effort container hint.

    Returns:
        A ``data:<mime>;base64,<...>`` URI.
    """
    mime = _MIME_BY_CONTAINER.get((container or "wav").lower(), "audio/wav")
    encoded = base64.b64encode(audio).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(prompt: str | None, data_uri: str) -> list[dict[str, Any]]:
    """Assemble the chat ``messages`` array (context + audio).

    Args:
        prompt: Free-text context (Qwen ``context``), placed in a system message,
            or ``None`` to omit it.
        data_uri: The inline audio data URI.

    Returns:
        The ``messages`` list.
    """
    messages: list[dict[str, Any]] = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": data_uri}}],
        }
    )
    return messages


def _asr_options(request: BatchRequest | StreamRequest) -> dict[str, Any]:
    """Build the ``asr_options`` block from a request.

    Args:
        request: A batch or streaming request.

    Returns:
        The ``asr_options`` mapping (omitting ``language`` when auto-detecting).
    """
    options: dict[str, Any] = {"enable_itn": request.enable_itn}
    if request.language:
        options["language"] = request.language
    return options


class DashScopeBackend:
    """Async DashScope OpenAI-compatible backend for ``qwen3-asr-flash``.

    Args:
        base_url: Compatible-mode root; defaults to the international endpoint.
        model: The DashScope model id (e.g. ``"qwen3-asr-flash"``).
        api_key: The DashScope API key (required).
        connect_timeout: Connect timeout in seconds.
        read_timeout: Read timeout in seconds.
        verify_tls: Whether to verify TLS certificates.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        connect_timeout: float,
        read_timeout: float,
        verify_tls: bool,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._verify_tls = verify_tls
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        """Build request headers with bearer auth.

        Returns:
            A header mapping.

        Raises:
            ValueError: If no API key is configured (DashScope requires one).
        """
        if not self._api_key:
            raise ValueError(
                "DashScope backend requires an api_key. Set it in config or via "
                "STANDARD_ASR_QWEN3_ASR__API_KEY / DASHSCOPE_API_KEY."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _http(self) -> httpx.AsyncClient:
        """Return the lazily-created pooled HTTP client.

        Returns:
            The shared :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, verify=self._verify_tls)
        return self._client

    async def transcribe(self, request: BatchRequest) -> BatchResult:
        """Transcribe via the chat-completions endpoint.

        Args:
            request: The normalized batch request.

        Returns:
            The normalized result.

        Raises:
            httpx.HTTPStatusError: On a non-2xx response.
            httpx.HTTPError: On a transport failure.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(
                request.prompt, _data_uri(request.audio, request.container)
            ),
            "stream": False,
            "temperature": request.temperature,
            "asr_options": _asr_options(request),
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.max_completion_tokens is not None:
            body["max_completion_tokens"] = request.max_completion_tokens
        response = await self._http().post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        response.raise_for_status()
        return _parse_chat_completion(response.json())

    async def stream(
        self,
        request: StreamRequest,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptDelta]:
        """Stream via ``stream=true`` chat-completions SSE.

        DashScope streaming requires the whole audio up front (the Flash service
        is not an incremental-input realtime API), so we collect the fed PCM
        frames, wrap them as WAV, and stream the transcript deltas back.

        Args:
            request: The normalized streaming request.
            audio_source: An async iterator of raw PCM16 frames.

        Yields:
            Incremental transcript deltas (one per SSE delta chunk).
        """
        frames: list[bytes] = []
        async for frame in audio_source:
            frames.append(frame)
        wav = wrap_pcm16_wav(b"".join(frames), sample_rate=request.sample_rate)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(request.prompt, _data_uri(wav, "wav")),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": request.temperature,
            "asr_options": _asr_options(request),
        }
        async with self._http().stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                delta = _parse_chat_sse_line(line)
                if delta is not None:
                    yield delta

    async def aclose(self) -> None:
        """Close the pooled HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_chat_completion(payload: dict[str, object]) -> BatchResult:
    """Normalize a DashScope chat-completion response.

    Args:
        payload: The decoded JSON body.

    Returns:
        A normalized :class:`BatchResult` (transcript + annotations).
    """
    text = ""
    language: str | None = None
    emotion: str | None = None
    choices = get_list(payload, "choices")
    if choices:
        message = get_object(first_object(choices), "message")
        text = _coerce_content(message.get("content"))
        for raw_ann in get_list(message, "annotations"):
            ann = as_object(raw_ann)
            if ann.get("type") == "audio_info":
                language = get_str(ann, "language") or language
                emotion = get_str(ann, "emotion") or emotion
    usage = get_object(payload, "usage")
    return BatchResult(
        text=text,
        detected_language=language,
        emotion=emotion,
        duration=get_float(usage, "seconds"),
    )


def _coerce_content(content: object) -> str:
    """Coerce a chat ``content`` field (string or list parts) to plain text.

    Args:
        content: The ``message.content`` value (a string, or a list of parts).

    Returns:
        The transcript text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in cast("list[object]", content):
            text = as_object(part).get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _parse_chat_sse_line(line: str) -> TranscriptDelta | None:
    """Parse one chat-completions SSE line into a transcript delta.

    Args:
        line: A raw line from the SSE stream.

    Returns:
        A :class:`TranscriptDelta` for a content delta, or ``None`` otherwise.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    chunk = loads_object(data)
    choices = get_list(chunk, "choices")
    if not choices:
        return None
    delta = get_object(first_object(choices), "delta")
    content = _coerce_content(delta.get("content"))
    if not content:
        return None
    return TranscriptDelta(text=content)
