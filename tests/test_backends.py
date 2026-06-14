# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct backend tests: request building, response parsing, error paths.

These exercise the backend transports and their pure parse helpers directly
(rather than only through the engine), covering the wire-shape edge cases:
optional sampling params, SSE sentinels, the Realtime error event, DashScope's
list-form content + annotations, and the missing-api-key guard.
"""

from __future__ import annotations

import asyncio

import pytest

from std_qwen3_asr.backends.base import BatchRequest, StreamRequest
from std_qwen3_asr.backends.dashscope import (
    DashScopeBackend,
    _coerce_content,
    _parse_chat_completion,
    _parse_chat_sse_line,
)
from std_qwen3_asr.backends.vllm import (
    VLLMBackend,
    _http_to_ws,
    _mime_and_name,
    _parse_sse_line,
    _parse_transcription_json,
    _realtime_session_update,
)

from .fake_server import FakeConfig, running_server


def _batch_request(**overrides: object) -> BatchRequest:
    """Build a BatchRequest with sensible defaults for tests."""
    base: dict[str, object] = {
        "audio": b"\x00\x00",
        "container": "wav",
        "language": None,
        "prompt": None,
        "temperature": 0.0,
        "top_p": None,
        "max_completion_tokens": None,
        "enable_itn": False,
        "emotion": False,
    }
    base.update(overrides)
    return BatchRequest(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# vLLM parse helpers
# --------------------------------------------------------------------------- #
def test_parse_transcription_json_minimal_and_verbose() -> None:
    assert _parse_transcription_json({"text": "hi"}).text == "hi"
    verbose = _parse_transcription_json(
        {"text": "hi", "language": "en", "duration": 2.5, "task": "transcribe"}
    )
    assert verbose.detected_language == "en"
    assert verbose.duration == 2.5
    assert verbose.raw["task"] == "transcribe"


def test_parse_sse_line_variants() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": comment") is None
    assert _parse_sse_line("data: [DONE]") is None
    assert _parse_sse_line('data: {"choices": []}') is None  # no delta
    assert _parse_sse_line('data: {"choices": [{"delta": {}}]}') is None  # empty delta
    delta = _parse_sse_line('data: {"choices": [{"delta": {"content": "x"}}]}')
    assert delta is not None and delta.text == "x"


def test_http_to_ws_conversion() -> None:
    assert _http_to_ws("http://h:8000/v1") == "ws://h:8000/v1"
    assert _http_to_ws("https://h/v1") == "wss://h/v1"


def test_mime_and_name() -> None:
    assert _mime_and_name("mp3") == ("audio/mpeg", "audio.mp3")
    assert _mime_and_name(None) == ("audio/wav", "audio.wav")
    # Unknown container -> octet-stream + .bin.
    assert _mime_and_name("xyz") == ("application/octet-stream", "audio.bin")


def test_vllm_sse_forwards_language_prompt_itn() -> None:
    with running_server(FakeConfig(deltas=["a", "b"])) as server:
        backend = VLLMBackend(
            base_url=server.http_base_url,
            model="m",
            api_key=None,
            connect_timeout=2.0,
            read_timeout=2.0,
            stream_transport="sse",
            verify_tls=True,
        )

        async def _src():  # type: ignore[no-untyped-def]
            yield b"\x00\x00" * 100

        async def _go() -> list[str]:
            out: list[str] = []
            async for d in backend.stream(
                StreamRequest(
                    language="en", prompt="ctx", temperature=0.0, sample_rate=16000, enable_itn=True
                ),
                _src(),
            ):
                out.append(d.text)
            await backend.aclose()
            return out

        deltas = asyncio.run(_go())
    assert deltas == ["a", "b"]
    form = server.requests[-1].form
    assert form["language"] == "en"
    assert form["prompt"] == "ctx"
    assert "enable_itn" in form["vllm_xargs"]
    assert form["stream"] == "true"


def test_realtime_session_update_includes_optional_fields() -> None:
    req = StreamRequest(
        language="en", prompt="ctx", temperature=0.3, sample_rate=16000, enable_itn=True
    )
    frame = _realtime_session_update(req, "Qwen/Qwen3-ASR-1.7B")
    session = frame["session"]
    assert session["language"] == "en"
    assert session["instructions"] == "ctx"
    assert session["enable_itn"] is True
    assert session["temperature"] == 0.3


# --------------------------------------------------------------------------- #
# DashScope parse helpers
# --------------------------------------------------------------------------- #
def test_coerce_content_string_and_list() -> None:
    assert _coerce_content("plain") == "plain"
    assert _coerce_content([{"text": "a"}, {"text": "b"}, {"image": "x"}]) == "ab"
    assert _coerce_content(42) == ""


def test_parse_chat_completion_with_annotations() -> None:
    payload: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "content": [{"text": "hello"}],
                    "annotations": [
                        {"type": "other"},
                        {"type": "audio_info", "language": "zh", "emotion": "happy"},
                    ],
                }
            }
        ],
        "usage": {"seconds": 3},
    }
    result = _parse_chat_completion(payload)
    assert result.text == "hello"
    assert result.detected_language == "zh"
    assert result.emotion == "happy"
    assert result.duration == 3.0


def test_parse_chat_completion_empty() -> None:
    assert _parse_chat_completion({}).text == ""


def test_parse_chat_sse_line() -> None:
    assert _parse_chat_sse_line("") is None
    assert _parse_chat_sse_line("data: [DONE]") is None
    assert _parse_chat_sse_line('data: {"choices": []}') is None
    # A delta with empty content yields no transcript delta.
    assert _parse_chat_sse_line('data: {"choices": [{"delta": {"content": ""}}]}') is None
    delta = _parse_chat_sse_line('data: {"choices": [{"delta": {"content": "z"}}]}')
    assert delta is not None and delta.text == "z"


# --------------------------------------------------------------------------- #
# DashScope backend behavior
# --------------------------------------------------------------------------- #
def test_dashscope_aclose_without_client_is_noop() -> None:
    # aclose() before any request created a client must be a safe no-op.
    backend = DashScopeBackend(
        base_url="https://x/v1",
        model="m",
        api_key="k",
        connect_timeout=1.0,
        read_timeout=1.0,
        verify_tls=True,
    )
    asyncio.run(backend.aclose())  # no client yet -> no-op


def test_dashscope_client_reused() -> None:
    with running_server() as server:
        backend = DashScopeBackend(
            base_url=server.http_base_url,
            model="m",
            api_key="k",
            connect_timeout=2.0,
            read_timeout=2.0,
            verify_tls=False,
        )

        async def _go() -> None:
            await backend.transcribe(_batch_request())
            c1 = backend._client  # pyright: ignore[reportPrivateUsage]
            await backend.transcribe(_batch_request())
            assert backend._client is c1  # pyright: ignore[reportPrivateUsage]
            await backend.aclose()

        asyncio.run(_go())


def test_dashscope_missing_api_key_raises() -> None:
    backend = DashScopeBackend(
        base_url="https://x/v1",
        model="qwen3-asr-flash",
        api_key=None,
        connect_timeout=1.0,
        read_timeout=1.0,
        verify_tls=True,
    )

    async def _go() -> None:
        await backend.transcribe(_batch_request())

    with pytest.raises(ValueError, match="api_key"):
        asyncio.run(_go())


def test_dashscope_forwards_sampling_params() -> None:
    with running_server() as server:
        backend = DashScopeBackend(
            base_url=server.http_base_url,
            model="qwen3-asr-flash",
            api_key="k",
            connect_timeout=2.0,
            read_timeout=2.0,
            verify_tls=False,
        )

        async def _go() -> None:
            await backend.transcribe(
                _batch_request(top_p=0.5, max_completion_tokens=128, language="en", enable_itn=True)
            )
            await backend.aclose()

        asyncio.run(_go())
    body = server.requests[-1].json_body
    assert body["top_p"] == 0.5
    assert body["max_completion_tokens"] == 128
    assert body["asr_options"]["language"] == "en"
    assert body["asr_options"]["enable_itn"] is True


# --------------------------------------------------------------------------- #
# vLLM backend behavior
# --------------------------------------------------------------------------- #
def test_vllm_client_is_reused_across_calls() -> None:
    # Two transcribe calls on the same backend (no aclose between) must reuse the
    # pooled httpx client (covers the "client already exists" branch).
    with running_server(FakeConfig(transcript="reuse")) as server:
        backend = VLLMBackend(
            base_url=server.http_base_url,
            model="m",
            api_key=None,
            connect_timeout=2.0,
            read_timeout=2.0,
            stream_transport="realtime",
            verify_tls=True,
        )

        async def _go() -> None:
            r1 = await backend.transcribe(_batch_request())
            client1 = backend._client  # pyright: ignore[reportPrivateUsage]
            r2 = await backend.transcribe(_batch_request())
            client2 = backend._client  # pyright: ignore[reportPrivateUsage]
            assert r1.text == r2.text == "reuse"
            assert client1 is client2  # same pooled client
            await backend.aclose()

        asyncio.run(_go())


def test_vllm_forwards_sampling_params() -> None:
    with running_server() as server:
        backend = VLLMBackend(
            base_url=server.http_base_url,
            model="Qwen/Qwen3-ASR-1.7B",
            api_key=None,
            connect_timeout=2.0,
            read_timeout=2.0,
            stream_transport="realtime",
            verify_tls=True,
        )

        async def _go() -> None:
            await backend.transcribe(
                _batch_request(top_p=0.8, max_completion_tokens=64, language="en", prompt="p")
            )
            await backend.aclose()

        asyncio.run(_go())
    form = server.requests[-1].form
    assert form["top_p"] == "0.8"
    assert form["max_completion_tokens"] == "64"
    assert form["language"] == "en"
    assert form["prompt"] == "p"


def _run_ws_server(handler: object, port: int) -> tuple[object, object, object]:
    """Start a one-off websockets server in a thread; return (thread, loop, stop)."""
    import threading

    from websockets.asyncio.server import serve

    ready = threading.Event()
    box: dict[str, object] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop = loop.create_future()
        box["loop"] = loop
        box["stop"] = stop

        async def _serve() -> None:
            async with serve(handler, "127.0.0.1", port):  # type: ignore[arg-type]
                ready.set()
                import contextlib

                with contextlib.suppress(asyncio.CancelledError):
                    await stop

        loop.run_until_complete(_serve())
        loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=5)
    return t, box["loop"], box["stop"]


def test_vllm_realtime_server_closes_without_done() -> None:
    # The realtime server sends deltas then closes the socket WITHOUT a
    # transcription.done. The backend's receive loop must end on socket close
    # (not hang), and the audio pump task is cancelled cleanly.
    import json

    from .fake_server import _free_port

    port = _free_port()

    async def _handler(ws: object) -> None:
        # Read just the session.update + a few audio frames, then send a delta and
        # an ignored message and close abruptly WITHOUT consuming all audio or
        # sending transcription.done. This leaves the client's audio pump still
        # running (it has many frames to feed) so closing the socket cancels it
        # mid-send -- exercising the defensive pump cleanup.
        seen = 0
        async for _raw in ws:  # type: ignore[attr-defined]
            seen += 1
            if seen >= 3:
                break
        await ws.send(json.dumps({"type": "session.updated"}))  # type: ignore[attr-defined]  # ignored type
        await ws.send(json.dumps({"type": "transcription.delta", "delta": "partial"}))  # type: ignore[attr-defined]
        # Close without 'transcription.done' (server-side disconnect mid-stream).

    thread, loop, stop = _run_ws_server(_handler, port)
    backend = VLLMBackend(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="m",
        api_key=None,
        connect_timeout=2.0,
        read_timeout=2.0,
        stream_transport="realtime",
        verify_tls=True,
    )

    async def _src():  # type: ignore[no-untyped-def]
        # Many frames so the pump is still feeding when the server closes.
        for _ in range(500):
            yield b"\x00\x00" * 800

    async def _go() -> list[str]:
        out: list[str] = []
        async for d in backend.stream(
            StreamRequest(
                language=None, prompt=None, temperature=0.0, sample_rate=16000, enable_itn=False
            ),
            _src(),
        ):
            out.append(d.text)
        await backend.aclose()
        return out

    try:
        deltas = asyncio.run(_go())
    finally:
        loop.call_soon_threadsafe(lambda: stop.cancel() if not stop.done() else None)  # type: ignore[union-attr]
        thread.join(timeout=5)  # type: ignore[union-attr]
    # The stream terminates cleanly on the server's abrupt close (no hang); the
    # delta may or may not have raced through before the close. The point is the
    # pump was cancelled mid-flight without leaking or deadlocking.
    assert deltas in ([], ["partial"])


def test_vllm_realtime_error_event_propagates() -> None:
    # If the realtime server emits an error event, the backend raises so the
    # engine wraps it as engine_error. Drive the backend directly against a tiny
    # WS server that sends an error frame after the commit.
    import json

    from .fake_server import _free_port

    port = _free_port()

    async def _handler(ws: object) -> None:
        async for raw in ws:  # type: ignore[attr-defined]
            msg = json.loads(raw if isinstance(raw, str) else raw.decode())
            if msg.get("type") == "input_audio_buffer.commit":
                break
        await ws.send(json.dumps({"type": "error", "error": {"message": "boom"}}))  # type: ignore[attr-defined]

    thread, loop, stop = _run_ws_server(_handler, port)
    backend = VLLMBackend(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="m",
        api_key=None,
        connect_timeout=2.0,
        read_timeout=2.0,
        stream_transport="realtime",
        verify_tls=True,
    )

    async def _src():  # type: ignore[no-untyped-def]
        yield b"\x00\x00" * 50

    async def _go() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in backend.stream(
                StreamRequest(
                    language=None, prompt=None, temperature=0.0, sample_rate=16000, enable_itn=False
                ),
                _src(),
            ):
                pass
        await backend.aclose()

    try:
        asyncio.run(_go())
    finally:
        loop.call_soon_threadsafe(lambda: stop.cancel() if not stop.done() else None)  # type: ignore[union-attr]
        thread.join(timeout=5)  # type: ignore[union-attr]
