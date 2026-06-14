# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""vLLM OpenAI-compatible backend for Qwen3-ASR.

This backend talks to a self-hosted vLLM server (``vllm serve Qwen/Qwen3-ASR-*``)
over its OpenAI-compatible HTTP API. It NEVER imports vLLM or torch -- vLLM is
the *remote runtime*, reached purely over the wire (license/dependency isolation,
spec G.4.2).

Wire surfaces used:

* **Batch** -- ``POST {base_url}/audio/transcriptions``, ``multipart/form-data``
  (the OpenAI Whisper-style file-upload shape). Response: ``{"text": "..."}``
  (``response_format=json``).
* **Streaming** -- two transports, selected by config:
    * ``realtime`` (default): the WebSocket Realtime API at ``{ws_base}/realtime``.
      Client sends ``session.update`` + ``input_audio_buffer.append`` (base64
      PCM16) frames and ``input_audio_buffer.commit``; the server emits
      ``transcription.delta`` (append text) and ``transcription.done`` events.
      This is the path Qwen3-ASR genuinely streams through.
    * ``sse``: ``stream=true`` on the transcription endpoint. The server emits
      Server-Sent Events whose JSON chunks are ``transcription.chunk`` objects
      with ``choices[0].delta.content`` text fragments, ending with a
      ``finish_reason="stop"`` chunk.

Both streaming transports are append-only token streams -- the deltas only grow
the transcript -- which is why the adapter declares ``word_stability=false`` and
reports ``stable_until=0`` (spec ST §4.2; Qwen3-ASR streaming is the spec's named
example).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

from .._audio import wrap_pcm16_wav
from .._json import first_object, get_float, get_list, get_object, get_str, loads_object
from .base import BatchRequest, BatchResult, StreamRequest, TranscriptDelta

#: Default OpenAI-compatible root for a local vLLM server.
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

#: MIME types for the small set of containers we hand to the multipart upload.
_MIME_BY_CONTAINER: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
}


def _mime_and_name(container: str | None) -> tuple[str, str]:
    """Pick a MIME type and filename for the upload from a container hint.

    Args:
        container: Best-effort container hint (e.g. ``"wav"``), or ``None``.

    Returns:
        A ``(mime, filename)`` pair. Unknown/None containers fall back to WAV
        (the standard layer encodes arrays to canonical WAV).
    """
    key = (container or "wav").lower()
    mime = _MIME_BY_CONTAINER.get(key, "application/octet-stream")
    ext = key if key in _MIME_BY_CONTAINER else "bin"
    return mime, f"audio.{ext}"


def _http_to_ws(base_url: str) -> str:
    """Convert an http(s) base URL to its ws(s) equivalent.

    Args:
        base_url: The OpenAI-compatible HTTP root (``http(s)://host/v1``).

    Returns:
        The WebSocket root (``ws(s)://host/v1``).
    """
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :]
    return base_url  # pragma: no cover - base_url is validated to be http(s)


class VLLMBackend:
    """Async vLLM OpenAI-compatible backend.

    Args:
        base_url: OpenAI-compatible root (e.g. ``http://localhost:8000/v1``).
        model: The vLLM model id to request (e.g. ``"Qwen/Qwen3-ASR-1.7B"``).
        api_key: Optional bearer token (vLLM ``--api-key``); ``None`` for an
            unauthenticated server.
        connect_timeout: Connect timeout in seconds.
        read_timeout: Read timeout in seconds (per chunk when streaming).
        stream_transport: ``"realtime"`` (WebSocket) or ``"sse"``.
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
        stream_transport: str,
        verify_tls: bool,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._stream_transport = stream_transport
        self._verify_tls = verify_tls
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        """Build request headers, including auth when an API key is set.

        Returns:
            A header mapping.
        """
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _http(self) -> httpx.AsyncClient:
        """Return the lazily-created pooled HTTP client.

        Returns:
            The shared :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, verify=self._verify_tls)
        return self._client

    async def transcribe(self, request: BatchRequest) -> BatchResult:
        """Transcribe via the multipart ``/audio/transcriptions`` endpoint.

        Args:
            request: The normalized batch request.

        Returns:
            The normalized result.

        Raises:
            httpx.HTTPStatusError: On a non-2xx response.
            httpx.HTTPError: On a transport failure.
        """
        mime, filename = _mime_and_name(request.container)
        data: dict[str, str] = {"model": self._model, "response_format": "json"}
        if request.language:
            data["language"] = request.language
        if request.prompt:
            data["prompt"] = request.prompt
        data["temperature"] = str(request.temperature)
        # vLLM extra/sampling params travel as extra form fields (the SDK puts
        # them in extra_body; on the raw multipart wire they are plain fields).
        if request.top_p is not None:
            data["top_p"] = str(request.top_p)
        if request.max_completion_tokens is not None:
            data["max_completion_tokens"] = str(request.max_completion_tokens)
        # ITN is a Qwen-specific knob; forward via vllm_xargs (string map).
        if request.enable_itn:
            data["vllm_xargs"] = json.dumps({"enable_itn": True})

        files = {"file": (filename, request.audio, mime)}
        response = await self._http().post(
            f"{self._base_url}/audio/transcriptions",
            headers=self._headers(),
            data=data,
            files=files,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return _parse_transcription_json(payload)

    async def stream(
        self,
        request: StreamRequest,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptDelta]:
        """Open a streaming transcription and yield transcript deltas.

        Args:
            request: The normalized streaming request.
            audio_source: An async iterator of raw PCM16 frames.

        Yields:
            Incremental transcript deltas.
        """
        if self._stream_transport == "sse":
            async for delta in self._stream_sse(request, audio_source):
                yield delta
        else:
            async for delta in self._stream_realtime(request, audio_source):
                yield delta

    async def _stream_realtime(
        self,
        request: StreamRequest,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptDelta]:
        """Stream via the WebSocket Realtime API (``/realtime``).

        Args:
            request: The normalized streaming request.
            audio_source: An async iterator of raw PCM16 frames.

        Yields:
            Incremental transcript deltas (one per ``transcription.delta`` event).
        """
        ws_url = f"{_http_to_ws(self._base_url)}/realtime"
        extra_headers = self._headers()
        async with websockets.connect(
            ws_url,
            additional_headers=extra_headers,
            open_timeout=self._timeout.connect,
            max_size=None,
        ) as ws:
            await ws.send(json.dumps(_realtime_session_update(request, self._model)))

            async def _pump() -> None:
                """Feed audio frames then commit the input buffer."""
                async for frame in audio_source:
                    encoded = base64.b64encode(frame).decode("ascii")
                    await ws.send(
                        json.dumps({"type": "input_audio_buffer.append", "audio": encoded})
                    )
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # Drive the pump concurrently with reception so the session is
            # full-duplex (spec ST §3.3). The pump task is awaited/cancelled in
            # the finally block to avoid leaks.
            pump_task = asyncio.ensure_future(_pump())
            try:
                async for raw in ws:
                    message = loads_object(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    mtype = message.get("type")
                    if mtype == "transcription.delta":
                        yield TranscriptDelta(text=get_str(message, "delta") or "")
                    elif mtype == "transcription.done":
                        break
                    elif mtype == "error":
                        # Surface a backend-side error as an exception the engine
                        # wraps into an ``engine_error`` event.
                        raise RuntimeError(f"vLLM realtime error: {message.get('error')}")
            finally:
                # Best-effort cleanup of the audio pump. If the stream ended
                # before all audio was fed (error / server close), cancel it; the
                # await drains the cancellation (and any send error the pump hit
                # racing the close). Purely defensive teardown.
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump_task

    async def _stream_sse(
        self,
        request: StreamRequest,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptDelta]:
        """Stream via SSE on the transcription endpoint (``stream=true``).

        The SSE path uploads the audio collected from ``audio_source`` as one
        request body, then streams ``transcription.chunk`` deltas. (vLLM's SSE
        transcription is request/response with streamed output rather than
        incremental input; we collect the fed frames into one WAV-less raw
        upload. For genuine low-latency incremental input prefer ``realtime``.)

        Args:
            request: The normalized streaming request.
            audio_source: An async iterator of raw PCM16 frames.

        Yields:
            Incremental transcript deltas (one per SSE delta chunk).
        """
        frames: list[bytes] = []
        async for frame in audio_source:
            frames.append(frame)
        pcm = b"".join(frames)
        wav = wrap_pcm16_wav(pcm, sample_rate=request.sample_rate)

        data: dict[str, str] = {
            "model": self._model,
            "response_format": "json",
            "stream": "true",
            "temperature": str(request.temperature),
        }
        if request.language:
            data["language"] = request.language
        if request.prompt:
            data["prompt"] = request.prompt
        if request.enable_itn:
            data["vllm_xargs"] = json.dumps({"enable_itn": True})
        files = {"file": ("audio.wav", wav, "audio/wav")}

        async with self._http().stream(
            "POST",
            f"{self._base_url}/audio/transcriptions",
            headers=self._headers(),
            data=data,
            files=files,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                delta = _parse_sse_line(line)
                if delta is not None:
                    yield delta

    async def aclose(self) -> None:
        """Close the pooled HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _realtime_session_update(request: StreamRequest, model: str) -> dict[str, Any]:
    """Build the Realtime ``session.update`` handshake frame.

    Args:
        request: The normalized streaming request.
        model: The vLLM model id.

    Returns:
        A JSON-serializable handshake dict.
    """
    session: dict[str, Any] = {
        "model": model,
        "input_audio_format": "pcm16",
        "temperature": request.temperature,
    }
    if request.language:
        session["language"] = request.language
    if request.prompt:
        # Qwen "context" biasing -> the system/instruction slot of the session.
        session["instructions"] = request.prompt
    if request.enable_itn:
        session["enable_itn"] = True
    return {"type": "session.update", "session": session}


def _parse_transcription_json(payload: dict[str, object]) -> BatchResult:
    """Normalize a vLLM transcription JSON response.

    Args:
        payload: The decoded JSON body. The minimal shape is ``{"text": "..."}``;
            ``verbose_json`` adds ``language``/``duration``/``segments``.

    Returns:
        A normalized :class:`BatchResult`.
    """
    raw: dict[str, object] = {}
    task = payload.get("task")
    if task is not None:
        raw["task"] = task
    return BatchResult(
        text=get_str(payload, "text") or "",
        detected_language=get_str(payload, "language"),
        duration=get_float(payload, "duration"),
        raw=raw,
    )


def _parse_sse_line(line: str) -> TranscriptDelta | None:
    """Parse one SSE line into a transcript delta, if it carries content.

    Args:
        line: A raw line from the SSE stream.

    Returns:
        A :class:`TranscriptDelta` for a content chunk, or ``None`` for blank
        lines, the ``[DONE]`` sentinel, and chunks without a text delta.
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
    content = get_str(delta, "content")
    if not content:
        return None
    return TranscriptDelta(text=content)
