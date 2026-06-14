# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend protocol + normalized request/result shapes.

A *backend* is the thin transport that turns a normalized request into HTTP/WS
traffic against a remote Qwen3-ASR deployment and normalizes the response back.
Keeping the wire details behind this interface means the engine's event mapping
and parameter gating never depend on which backend is active.

The shapes here are intentionally minimal -- they carry exactly what the wire
needs and what the result mapping consumes, nothing more.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def _empty_raw() -> dict[str, object]:
    """Return a fresh empty mapping for the ``raw`` field default.

    A typed factory (vs ``default_factory=dict``) keeps pyright strict happy:
    ``dict`` alone infers ``dict[Unknown, Unknown]``.

    Returns:
        An empty ``dict[str, object]``.
    """
    return {}


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """A normalized one-shot transcription request.

    Args:
        audio: Encoded audio bytes (e.g. WAV/MP3) ready to upload.
        container: Best-effort container/format hint (e.g. ``"wav"``,
            ``"mp3"``), used to pick a MIME type and filename for the upload.
        language: Resolved backend language token, or ``None`` for auto-detect.
            For DashScope this is an ISO code; for vLLM an ISO code too (vLLM's
            ``language`` form field). ``None`` => omit the field.
        prompt: Free-text biasing context (Qwen ``context``), or ``None``.
        temperature: Decode sampling temperature.
        top_p: Optional nucleus sampling parameter.
        max_completion_tokens: Optional generated-token cap.
        enable_itn: Whether to request inverse text normalization.
        emotion: Whether to request emotion annotation.
    """

    audio: bytes
    container: str | None
    language: str | None
    prompt: str | None
    temperature: float
    top_p: float | None
    max_completion_tokens: int | None
    enable_itn: bool
    emotion: bool


@dataclass(frozen=True, slots=True)
class BatchResult:
    """A normalized transcription response.

    Args:
        text: The full transcript text.
        detected_language: Backend-reported language string (ISO code or English
            name; the engine maps it to BCP-47), or ``None``.
        emotion: Backend-reported emotion label, or ``None``.
        duration: Billed/processed audio duration in seconds, if reported.
        raw: Small, non-sensitive extra fields to surface in ``result.extra``.
    """

    text: str
    detected_language: str | None = None
    emotion: str | None = None
    duration: float | None = None
    raw: dict[str, object] = field(default_factory=_empty_raw)


@dataclass(frozen=True, slots=True)
class StreamRequest:
    """A normalized incremental-streaming request (parameters only; audio is fed).

    Args:
        language: Resolved backend language token, or ``None`` for auto-detect.
        prompt: Free-text biasing context (Qwen ``context``), or ``None``.
        temperature: Decode sampling temperature.
        sample_rate: Wire sample rate in Hz (always the engine's required rate).
        enable_itn: Whether to request inverse text normalization.
    """

    language: str | None
    prompt: str | None
    temperature: float
    sample_rate: int
    enable_itn: bool


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    """One incremental transcript update from a streaming backend.

    vLLM streaming is append-only: each delta carries a *new text fragment* to
    append to the running transcript (it is NOT the cumulative text). The session
    accumulates these into the cumulative ``text`` the standard protocol requires
    (spec §4.3).

    Args:
        text: The new text fragment to append (may be empty for a keepalive).
        detected_language: Backend-reported language, if the chunk carries it.
    """

    text: str
    detected_language: str | None = None


@runtime_checkable
class Backend(Protocol):
    """Transport that performs batch and streaming transcription.

    Implementations are async-first. The streaming method is an async generator
    that connects, feeds audio from ``audio_source``, and yields
    :class:`TranscriptDelta` objects until the remote signals end-of-stream.
    """

    async def transcribe(self, request: BatchRequest) -> BatchResult:
        """Perform a one-shot transcription.

        Args:
            request: The normalized batch request.

        Returns:
            The normalized result.

        Raises:
            Exception: Any transport/backend failure; the engine wraps it as a
                portable ``TranscriptionError`` (spec Runtime R7).
        """
        ...

    def stream(
        self,
        request: StreamRequest,
        audio_source: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptDelta]:
        """Open a streaming transcription and yield transcript deltas.

        Args:
            request: The normalized streaming request (frozen params).
            audio_source: An async iterator of raw PCM frames to feed.

        Returns:
            An async iterator of incremental transcript deltas.
        """
        ...

    async def aclose(self) -> None:
        """Release any pooled transport resources."""
        ...
