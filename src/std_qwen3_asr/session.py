# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming session: maps Qwen3-ASR backend deltas to Standard ASR events.

This is the core of the streaming integration. The base
:class:`~standard_asr.runtime.streaming.TranscriptionSession` owns the pump,
backpressure, deadlines, lifecycle enforcement, the sync bridge, and result
reduction; we implement only :meth:`_produce` (async).

Event mapping (the design rationale -- see ``docs/STANDARD_ASR_FINDINGS.md``):

* Qwen3-ASR streams **append-only token deltas** (vLLM Realtime
  ``transcription.delta`` / SSE ``transcription.chunk``; DashScope chat SSE
  deltas). Each delta is a *fragment* to append.
* We **accumulate** fragments into the cumulative segment text and emit a
  ``partial`` carrying the full current text (spec §4.3 forbids delta-on-the-wire;
  cumulative/replace is mandatory). When the stream ends we emit one ``final``.
* **One deterministic segment** ``seg-0`` (spec §3.4): the backend gives no
  segment ids and the stream is one continuous text flow, so a fixed id makes the
  id sequence reproducible across runs of the same audio.
* **``stable_until=0`` on every event.** The streaming wire carries no per-token
  timestamps or right-context, so we cannot honestly freeze any prefix. Spec
  ST §4.2 names exactly this case ("Qwen3-ASR streaming ... MUST report
  stable_until=0; word_stability MUST be declared false"). We declare
  ``word_stability=false`` in capabilities to match.
* **``audio_processed_until`` is omitted** -- the backend reports no audio cursor,
  and the spec forbids fabricating one (§4.4: "MUST NOT carry a fabricated
  audio_processed_until ... no reliable cursor => omit the field"). The base's
  liveness backstop is anchored on audio consumption, not on this field, so
  omitting it is safe.
* **No ``supersede``** (``re_segments=false``): append-only never retracts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from standard_asr.runtime.streaming import TranscriptionEvent, TranscriptionSession

from .backends.base import Backend, StreamRequest
from .languages import from_backend_language

#: The single deterministic segment id for a Qwen3-ASR streaming session.
_SEGMENT_ID = "seg-0"


class Qwen3ASRSession(TranscriptionSession):
    """A streaming session backed by a Qwen3-ASR vLLM/DashScope backend.

    The session reads fed PCM frames via :meth:`audio_chunks`, forwards them to
    the backend's streaming transport, accumulates the returned token-delta
    fragments into cumulative text, and yields ``partial`` events (then one
    ``final``).

    Args:
        backend: The active backend transport.
        request: The frozen streaming request (params resolved at session start;
            spec Runtime R5: frozen for the whole session).
        whole_input_pcm: Raw PCM16 bytes for the whole-input streaming-output
            path (``start_transcription(audio=...)``, spec §7.3). When set, the
            session feeds the backend from this fixed buffer instead of the
            incremental ``audio_chunks()`` queue. ``None`` for the incremental
            ``audio_format`` path.
        **session_kwargs: Forwarded to :class:`TranscriptionSession` (deadlines,
            buffer bounds, ``strict_lifecycle``).
    """

    def __init__(
        self,
        *,
        backend: Backend,
        request: StreamRequest,
        whole_input_pcm: bytes | None = None,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        self._backend = backend
        self._request = request
        self._whole_input_pcm = whole_input_pcm

    async def _audio_source(self) -> AsyncIterator[bytes]:
        """Yield the audio frames to feed the backend.

        For the incremental path this delegates to :meth:`audio_chunks` (the fed
        PCM queue). For the whole-input path it yields the fixed PCM buffer as a
        single frame (and consumes nothing from the queue, since no incremental
        audio is fed in that mode).

        Yields:
            Raw PCM16 frames.
        """
        if self._whole_input_pcm is not None:
            yield self._whole_input_pcm
            return
        async for chunk in self.audio_chunks():
            yield chunk

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Yield events mapped from the backend's streaming output.

        Returning normally ends the stream; the base appends ``done``. A backend
        exception escaping here is wrapped by the base into an ``engine_error``
        event (spec §6.2), the streaming analogue of the batch ``TranscriptionError``.

        Yields:
            ``partial`` events with growing cumulative text, then one ``final``.
        """
        cumulative = ""
        detected_language: str | None = None

        async for delta in self._backend.stream(self._request, self._audio_source()):
            if delta.detected_language and detected_language is None:
                # Map the raw backend language (ISO code / English name) to BCP-47;
                # an unmappable value stays None (events require a valid BCP-47
                # ``detected_language``; fabricating one is forbidden).
                detected_language = from_backend_language(delta.detected_language)
            if not delta.text:
                # Empty keepalive fragment: nothing to show, no event. (We do not
                # synthesize a progress heartbeat with a fabricated cursor.)
                continue
            cumulative += delta.text
            yield TranscriptionEvent.partial(
                _SEGMENT_ID,
                cumulative,
                # No right-context / timestamps on the wire => no frozen prefix.
                stable_until=0,
                **_lang_kwargs(detected_language),
            )

        # Always close the segment with one final carrying the complete text.
        # Even when no delta arrived (silent audio), the empty final gives the
        # segment a defined terminal so the reduced result is well-formed.
        yield TranscriptionEvent.final(
            _SEGMENT_ID,
            cumulative,
            stable_until=0,
            **_lang_kwargs(detected_language),
        )


def _lang_kwargs(detected_language: str | None) -> dict[str, str]:
    """Build the optional ``detected_language`` kwarg for an event.

    The mapped BCP-47 language is only attached once the backend reports it.
    Passing ``detected_language=None`` is fine, but only including the key when
    present keeps events minimal and avoids re-asserting ``None`` repeatedly.

    Args:
        detected_language: The mapped BCP-47 language, or ``None``.

    Returns:
        ``{"detected_language": tag}`` if present, else ``{}``.
    """
    return {} if detected_language is None else {"detected_language": detected_language}
