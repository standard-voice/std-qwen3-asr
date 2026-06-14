# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""A fake Qwen3-ASR backend server for integration tests (no GPU, no network).

This stands in for a real vLLM OpenAI-compatible server (and the DashScope
chat-completions surface) so the adapter's wire mapping, event semantics, and
session lifecycle can be exercised end-to-end ON THIS MACHINE -- the M5 Max has
no CUDA GPU, so a real vLLM server cannot run here (an expected validation-phase
blocker; see VERIFICATION.md). The fake reproduces the representative response
shapes from the vendor docs:

* ``POST /v1/audio/transcriptions`` (multipart) -- vLLM batch; returns
  ``{"text": ...}`` (or SSE ``transcription.chunk`` deltas when ``stream=true``).
* ``WS /v1/realtime`` -- vLLM Realtime; consumes ``input_audio_buffer.append`` /
  ``.commit`` and emits ``transcription.delta`` + ``transcription.done``.
* ``POST /v1/chat/completions`` -- DashScope; returns the chat shape with
  ``annotations`` (or SSE deltas when ``stream=true``).

It is intentionally simple: it echoes a fixed transcript (optionally split into
deltas for streaming) and reflects back the language / itn / prompt it received so
tests can assert the adapter sent them correctly. A real model is not involved.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from websockets.asyncio.server import ServerConnection, serve

#: The canned transcript the fake returns, and its per-delta fragmentation.
DEFAULT_TRANSCRIPT = "the quick brown fox"
DEFAULT_DELTAS = ["the ", "quick ", "brown ", "fox"]


@dataclass
class CapturedRequest:
    """What the fake observed for one request (for test assertions).

    Args:
        path: The request path.
        form: Captured multipart form fields (transcription endpoint).
        json_body: Captured JSON body (chat-completions endpoint).
        file_bytes: The uploaded file's byte length (multipart).
        realtime_session: The Realtime ``session.update`` payload, if any.
        realtime_audio_bytes: Total decoded audio bytes received over Realtime.
    """

    path: str
    form: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] = field(default_factory=dict)
    file_len: int = 0
    realtime_session: dict[str, Any] = field(default_factory=dict)
    realtime_audio_bytes: int = 0


@dataclass
class FakeConfig:
    """Tunable behavior for the fake server.

    Args:
        transcript: The full transcript to return.
        deltas: The streaming fragments to emit (concatenate to ``transcript``).
        language: The language to report (ISO code for chat annotations / vLLM).
        emotion: The emotion to report in chat annotations.
        fail_status: If set, every request returns this HTTP status (error path).
        require_bearer: If set, requests MUST carry this exact bearer token.
    """

    transcript: str = DEFAULT_TRANSCRIPT
    deltas: list[str] = field(default_factory=lambda: list(DEFAULT_DELTAS))
    language: str = "en"
    emotion: str = "neutral"
    fail_status: int | None = None
    require_bearer: str | None = None


class FakeBackendServer:
    """A real uvicorn + websockets server emulating the Qwen3-ASR backends.

    Runs on a random free port in a background thread so the adapter connects to
    a genuine ``http://127.0.0.1:<port>`` URL over real sockets (HTTP, SSE, and
    WebSocket), exercising the true transport paths.

    Args:
        config: Behavior configuration.
    """

    def __init__(self, config: FakeConfig | None = None) -> None:
        self.config = config or FakeConfig()
        self.requests: list[CapturedRequest] = []
        self._http_port = _free_port()
        self._ws_port = _free_port()
        self._app = self._build_app()
        self._uvicorn: uvicorn.Server | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop: asyncio.Future[None] | None = None

    # -- URLs ----------------------------------------------------------- #
    @property
    def http_base_url(self) -> str:
        """OpenAI-compatible HTTP root the adapter should target."""
        return f"http://127.0.0.1:{self._http_port}/v1"

    @property
    def realtime_base_url(self) -> str:
        """HTTP root whose ws:// form hosts the Realtime endpoint."""
        return f"http://127.0.0.1:{self._ws_port}/v1"

    # -- bearer check --------------------------------------------------- #
    def _auth_ok(self, authorization: str | None) -> bool:
        """Return whether the Authorization header satisfies the config."""
        if self.config.require_bearer is None:
            return True
        return authorization == f"Bearer {self.config.require_bearer}"

    # -- HTTP app ------------------------------------------------------- #
    def _build_app(self) -> FastAPI:
        """Build the FastAPI app for the HTTP/SSE surfaces."""
        app = FastAPI()

        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request) -> Any:
            form = await request.form()
            try:
                fields = {k: v for k, v in form.items() if isinstance(v, str)}
                upload = form.get("file")
                file_len = 0
                if upload is not None and hasattr(upload, "read"):
                    file_len = len(await upload.read())  # type: ignore[union-attr]
            finally:
                # Close the spooled UploadFile temp files to avoid a leak warning.
                await form.close()
            self.requests.append(
                CapturedRequest(path="/v1/audio/transcriptions", form=fields, file_len=file_len)
            )
            if self.config.fail_status is not None:
                return JSONResponse({"error": "forced"}, status_code=self.config.fail_status)
            if not self._auth_ok(request.headers.get("authorization")):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            if fields.get("stream") == "true":
                return StreamingResponse(
                    self._sse_transcription_chunks(), media_type="text/event-stream"
                )
            return JSONResponse({"text": self.config.transcript})

        @app.post("/v1/chat/completions")
        async def chat(request: Request) -> Any:
            body: dict[str, Any] = await request.json()
            self.requests.append(CapturedRequest(path="/v1/chat/completions", json_body=body))
            if self.config.fail_status is not None:
                return JSONResponse({"error": "forced"}, status_code=self.config.fail_status)
            if not self._auth_ok(request.headers.get("authorization")):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            if body.get("stream"):
                return StreamingResponse(self._sse_chat_chunks(), media_type="text/event-stream")
            return JSONResponse(self._chat_completion_body())

        return app

    def _chat_completion_body(self) -> dict[str, Any]:
        """Build a non-streaming chat-completion response body."""
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": self.config.transcript,
                        "annotations": [
                            {
                                "type": "audio_info",
                                "language": self.config.language,
                                "emotion": self.config.emotion,
                            }
                        ],
                    },
                }
            ],
            "usage": {"seconds": 1, "total_tokens": 10},
        }

    async def _sse_transcription_chunks(self) -> AsyncIterator[bytes]:
        """Yield vLLM ``transcription.chunk`` SSE events."""
        for fragment in self.config.deltas:
            chunk = {
                "id": "trsc-1",
                "object": "transcription.chunk",
                "choices": [{"delta": {"content": fragment}}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        final = {"id": "trsc-1", "object": "transcription.chunk", "choices": []}
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _sse_chat_chunks(self) -> AsyncIterator[bytes]:
        """Yield DashScope chat-completion SSE delta events."""
        for fragment in self.config.deltas:
            chunk = {"choices": [{"delta": {"content": fragment}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    # -- Realtime WS ---------------------------------------------------- #
    async def _realtime_handler(self, ws: ServerConnection) -> None:
        """Handle one Realtime WebSocket connection."""
        captured = CapturedRequest(path="/v1/realtime")
        self.requests.append(captured)
        committed = False
        async for raw in ws:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            msg: dict[str, Any] = json.loads(text)
            mtype = msg.get("type")
            if mtype == "session.update":
                captured.realtime_session = msg.get("session", {})
            elif mtype == "input_audio_buffer.append":
                import base64

                captured.realtime_audio_bytes += len(base64.b64decode(msg.get("audio", "")))
            elif mtype == "input_audio_buffer.commit":
                committed = True
                break
        if not committed:  # pragma: no cover - clients always commit in tests
            return
        for fragment in self.config.deltas:
            await ws.send(json.dumps({"type": "transcription.delta", "delta": fragment}))
        await ws.send(json.dumps({"type": "transcription.done"}))

    # -- lifecycle ------------------------------------------------------ #
    def __enter__(self) -> FakeBackendServer:
        """Start the HTTP and WS servers and wait until both are ready."""
        self._start_http()
        self._start_ws()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop both servers and join their threads."""
        self._stop_ws()
        self._stop_http()

    def _start_http(self) -> None:
        # ws="none": this uvicorn instance serves only HTTP/SSE; the Realtime
        # WebSocket runs on a separate ``websockets.asyncio.server``. Disabling
        # uvicorn's WS support also avoids importing the deprecated
        # ``websockets.legacy`` module (which would trip warnings-as-errors).
        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self._http_port,
            log_level="warning",
            ws="none",
        )
        self._uvicorn = uvicorn.Server(config)
        self._http_thread = threading.Thread(target=self._uvicorn.run, daemon=True)
        self._http_thread.start()
        # Poll uvicorn's own readiness flag (more reliable than a TCP probe,
        # which can race the bind on a just-released ephemeral port).
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._uvicorn.started:
                return
            time.sleep(0.02)
        raise RuntimeError("fake HTTP server did not start in time")  # pragma: no cover

    def _stop_http(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)

    def _start_ws(self) -> None:
        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ws_loop = loop
            self._ws_stop = loop.create_future()

            async def _serve() -> None:
                async with serve(self._realtime_handler, "127.0.0.1", self._ws_port):
                    ready.set()
                    assert self._ws_stop is not None
                    # Wait for the stop signal; swallow the cancellation used to
                    # unblock it so the thread exits cleanly (no unhandled error).
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._ws_stop

            try:
                loop.run_until_complete(_serve())
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()
        ready.wait(timeout=5)

    def _stop_ws(self) -> None:
        if self._ws_loop is not None and self._ws_stop is not None:
            loop = self._ws_loop
            stop = self._ws_stop

            def _cancel() -> None:
                if not stop.done():
                    stop.cancel()

            loop.call_soon_threadsafe(_cancel)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=5)


def _free_port() -> int:
    """Find and return a free localhost TCP port."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def running_server(config: FakeConfig | None = None) -> Generator[FakeBackendServer]:
    """Context manager yielding a started :class:`FakeBackendServer`.

    Args:
        config: Optional behavior configuration.

    Yields:
        The running fake server.
    """
    with FakeBackendServer(config) as server:
        yield server


# Silence the unused-import lint for websockets (used via serve/ServerConnection).
_ = websockets
