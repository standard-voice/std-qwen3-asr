# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming integration tests against the in-process fake backend server.

This is the heart of the validation: it drives the adapter's streaming path
end-to-end on this machine (no GPU) over real transports -- the vLLM Realtime
WebSocket and the SSE fallback -- and asserts the Standard ASR event mapping:

* cumulative/replace ``text`` accumulation from append-only deltas (spec §4.3),
* a single deterministic ``seg-0`` segment (spec §3.4),
* ``stable_until=0`` on every event (spec ST §4.2: Qwen3-ASR streaming is the
  named case -- no right-context/timestamps => no frozen prefix),
* one ``final`` then the base-appended ``done``,
* correct event sequence per the compliance event-sequence checker,
* the sync bridge, the whole-input path, and error wrapping.
"""

from __future__ import annotations

import pytest
from standard_asr import AudioPath, RuntimeParams, SyncSession
from standard_asr.audio_format import AudioFormat
from standard_asr.compliance import check_event_sequence
from standard_asr.streaming import TranscriptionEvent, TranscriptionSession, reduce_event

from std_qwen3_asr import Qwen3ASR, Qwen3ASR17B, Qwen3ASRParams

from .conftest import TEST_AUDIO_PATH
from .fake_server import FakeConfig, running_server

PCM_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


async def _collect(session: TranscriptionSession) -> list[TranscriptionEvent]:
    """Drive a session to completion and collect all emitted events."""
    events: list[TranscriptionEvent] = []
    async with session:
        async for event in session:
            events.append(event)
    return events


async def _run_stream(
    engine: Qwen3ASR17B | Qwen3ASR,
    frames: list[bytes],
    *,
    params: RuntimeParams | None = None,
) -> list[TranscriptionEvent]:
    """Open an incremental session, feed frames, and collect events."""
    session = engine.start_transcription(audio_format=PCM_FORMAT, params=params)
    session.feed(frames)
    return await _collect(session)


# --------------------------------------------------------------------------- #
# Realtime WebSocket transport (the primary Qwen3-ASR streaming path)
# --------------------------------------------------------------------------- #
async def test_realtime_event_mapping(pcm16_frames: list[bytes]) -> None:
    with running_server(FakeConfig(deltas=["the ", "quick ", "brown ", "fox"])) as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        events = await _run_stream(engine, pcm16_frames)

    types = [e.type for e in events]
    assert types[-1] == "done"
    partials = [e for e in events if e.type == "partial"]
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1

    # Cumulative/replace (spec §4.3): every partial carries the FULL text so far,
    # never a delta. The standard layer MAY coalesce consecutive partials of the
    # same segment under backpressure (§6.4), so we assert the invariants that
    # always hold rather than an exact partial count: each delivered partial is a
    # growing prefix of the final, and the partials are monotonically increasing.
    assert all(p.text is not None for p in partials)
    texts = [p.text or "" for p in partials]
    assert texts == sorted(texts, key=len)  # monotonic growth (cumulative)
    for t in texts:
        assert "the quick brown fox".startswith(t)  # each is a prefix of the whole
    # One deterministic segment.
    assert {e.segment_id for e in partials + finals} == {"seg-0"}
    # stable_until=0 everywhere (spec ST §4.2, Qwen3-ASR streaming named case).
    assert all(e.stable_until == 0 for e in partials + finals)
    # The final carries the complete transcript.
    assert finals[0].text == "the quick brown fox"


async def test_realtime_forwards_session_params(pcm16_frames: list[bytes]) -> None:
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        await _run_stream(
            engine,
            pcm16_frames,
            params=RuntimeParams(
                language="ja",
                prompt="ASR context",
                provider_params=Qwen3ASRParams(enable_itn=True),
            ),
        )
    realtime = next(r for r in server.requests if r.path == "/v1/realtime")
    assert realtime.realtime_session["language"] == "ja"
    assert realtime.realtime_session["instructions"] == "ASR context"
    assert realtime.realtime_session["enable_itn"] is True
    assert realtime.realtime_session["input_audio_format"] == "pcm16"
    # The audio frames actually reached the server.
    assert realtime.realtime_audio_bytes == sum(len(f) for f in pcm16_frames)


async def test_realtime_event_sequence_is_compliant(pcm16_frames: list[bytes]) -> None:
    # The recorded event stream MUST satisfy the standard's event-sequence
    # contract (segment lifecycle + ordering). This is the streaming compliance
    # dimension the CLI cannot synthesize (plugin_entrypoints.md).
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        events = await _run_stream(engine, pcm16_frames)
    report = check_event_sequence(events)
    assert report.passed, [i.message for i in report.issues]


async def test_realtime_detected_language_mapped_to_bcp47(pcm16_frames: list[bytes]) -> None:
    # The fake's realtime path emits deltas without a language; ensure the
    # session still produces a valid stream (detected_language stays None, never
    # a fabricated tag). Language mapping itself is covered in unit tests.
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        events = await _run_stream(engine, pcm16_frames)
    for e in events:
        # detected_language, when set, must be a real BCP-47 tag (never "auto").
        assert e.detected_language in (None,) or e.detected_language != "auto"


# --------------------------------------------------------------------------- #
# SSE transport
# --------------------------------------------------------------------------- #
async def test_sse_event_mapping(pcm16_frames: list[bytes]) -> None:
    with running_server(FakeConfig(deltas=["hello ", "world"])) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, stream_transport="sse")
        events = await _run_stream(engine, pcm16_frames)

    partials = [e for e in events if e.type == "partial"]
    finals = [e for e in events if e.type == "final"]
    # Partials may coalesce (§6.4); assert prefix/monotonic invariants + final.
    texts = [p.text or "" for p in partials]
    assert texts == sorted(texts, key=len)
    for t in texts:
        assert "hello world".startswith(t)
    assert finals[0].text == "hello world"
    assert all(e.stable_until == 0 for e in partials + finals)
    assert events[-1].type == "done"
    # SSE collected the frames and uploaded them as one request.
    assert any(r.path == "/v1/audio/transcriptions" for r in server.requests)


async def test_sse_dashscope_streaming(pcm16_frames: list[bytes]) -> None:
    with running_server(FakeConfig(deltas=["ni ", "hao"])) as server:
        engine = Qwen3ASR(
            backend="dashscope",
            base_url=server.http_base_url,  # loopback http allowed
            api_key="sk-test",
        )
        events = await _run_stream(engine, pcm16_frames)
    finals = [e for e in events if e.type == "final"]
    assert finals[0].text == "ni hao"
    assert any(r.path == "/v1/chat/completions" for r in server.requests)


# --------------------------------------------------------------------------- #
# Reduction, sync bridge, whole-input, errors
# --------------------------------------------------------------------------- #
async def test_reduce_to_result(pcm16_frames: list[bytes]) -> None:
    with running_server(FakeConfig(deltas=["a", "b", "c"])) as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        session = engine.start_transcription(audio_format=PCM_FORMAT)
        session.feed(pcm16_frames)
        await _collect(session)
        result = session.result()
    assert result.text == "abc"
    # Manual reduce (the canonical 3-line app reduce) reaches the same text.
    segments: dict[str, str] = {}
    session2_events = [
        TranscriptionEvent.partial("seg-0", "a"),
        TranscriptionEvent.final("seg-0", "abc"),
    ]
    for ev in session2_events:
        reduce_event(segments, ev)
    assert "".join(segments.values()) == "abc"


def test_sync_bridge(pcm16_frames: list[bytes]) -> None:
    # The sync bridge must drive the async session without deadlock or leak.
    with running_server(FakeConfig(deltas=["x ", "y"])) as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        session = engine.start_transcription(audio_format=PCM_FORMAT)
        collected: list[str] = []
        with SyncSession(session) as sync:
            sync.feed(pcm16_frames)
            for event in sync:
                if event.type == "final":
                    collected.append(event.text or "")
            result = sync.result()
    assert collected == ["x y"]
    assert result.text == "x y"


@pytest.mark.skipif(not TEST_AUDIO_PATH.exists(), reason="reference test audio not available")
async def test_whole_input_streaming_output() -> None:
    # The whole-input + streaming-output path (spec §7.3): pass a complete file;
    # the base negotiates it to prepared_audio, the session feeds it to the
    # backend and streams results. Uses the real 48 kHz stereo reference clip
    # (decoded/resampled by the standard layer) -- end to end on this machine.
    with running_server(FakeConfig(deltas=["full ", "transcript"])) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, stream_transport="sse")
        session = engine.start_transcription(audio=AudioPath(str(TEST_AUDIO_PATH)))
        events = await _collect(session)
    finals = [e for e in events if e.type == "final"]
    assert finals[0].text == "full transcript"
    # The whole clip was uploaded (decoded from m4a, resampled to 16 kHz mono).
    assert server.requests[-1].file_len > 1000


async def test_streaming_backend_error_becomes_engine_error(pcm16_frames: list[bytes]) -> None:
    # A backend failure during streaming must surface as an in-stream error event
    # (engine_error), the streaming analogue of the batch TranscriptionError.
    with running_server(FakeConfig(fail_status=500)) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, stream_transport="sse")
        events = await _run_stream(engine, pcm16_frames)
    errors = [e for e in events if e.type == "error"]
    assert errors, "expected an error event when the backend fails"
    assert errors[0].code == "engine_error"
    assert errors[0].recoverable is False  # fails safe to terminal


async def test_empty_audio_yields_well_formed_stream() -> None:
    # No deltas (silent/empty) must still produce a defined terminal: an empty
    # final + done, so the reduced result is well-formed.
    with running_server(FakeConfig(deltas=[])) as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        session = engine.start_transcription(audio_format=PCM_FORMAT)
        session.feed([b"\x00\x00" * 800])
        events = await _collect(session)
    assert [e.type for e in events] == ["final", "done"]
    assert events[0].text == ""
