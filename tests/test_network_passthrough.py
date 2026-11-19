# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend requests honor the proxy variables of the environment.

This plugin downloads no model weights, so ``HF_ENDPOINT`` does not apply.
Its HTTP requests go to a vLLM server or to DashScope through httpx, which
reads ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, and ``NO_PROXY``;
Standard ASR neither sets nor changes them. This test does not cover the
realtime stream, which uses websockets. It runs a real transcription in a
subprocess, without the parent's proxy variables, against host names that
never resolve. A local proxy records each request and answers 404, so no
HTTP request leaves the machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

#: Variables the subprocess must not inherit, in upper or lower case: they
#: would change where the requests go.
_INHERITED = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"})

_TRANSCRIBE = """
import sys
import numpy as np
from standard_asr import AudioArray
from std_qwen3_asr import Qwen3ASR, Qwen3ASR17B
engine = (
    Qwen3ASR17B(base_url=sys.argv[2])
    if sys.argv[1] == "vllm"
    else Qwen3ASR(api_key="sk-test", base_url=sys.argv[2])
)
try:
    engine.transcribe(AudioArray(np.zeros(1600, dtype=np.float32), 16000))
except Exception as error:
    # The chain of causes, outermost first, shows why the request failed.
    link, names = error, []
    while link is not None and len(names) < 12:
        names.append(type(link).__name__)
        link = link.__cause__ or link.__context__
    print(" > ".join(names))
"""


@pytest.fixture
def proxy() -> Iterator[tuple[str, list[str]]]:
    """Run a proxy on 127.0.0.1 that records request lines and answers 404."""
    seen: list[str] = []

    class Recorder(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            seen.append(self.requestline)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = do_CONNECT = _answer

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def _transcribe(backend: str, base_url: str, settings: dict[str, str]) -> str:
    """Run a transcription in a subprocess; return the class names in its failure's cause chain."""
    env = {key: value for key, value in os.environ.items() if key.upper() not in _INHERITED}
    env.update(settings)
    result = subprocess.run(
        [sys.executable, "-c", _TRANSCRIBE, backend, base_url],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Every case ends in a failed transcription, because no server answers with a transcript.
    assert result.stdout.strip(), result
    return result.stdout


def test_backend_requests_go_where_the_environment_points(
    proxy: tuple[str, list[str]],
) -> None:
    url, seen = proxy

    # A request to a plain-HTTP vLLM server goes to the proxy in HTTP_PROXY.
    _transcribe("vllm", "http://vllm.invalid:8000/v1", {"HTTP_PROXY": url})
    assert any(
        line.startswith("POST http://vllm.invalid:8000/v1/audio/transcriptions ") for line in seen
    ), seen

    # A request to DashScope, over HTTPS, goes to the proxy in HTTPS_PROXY.
    seen.clear()
    _transcribe("dashscope", "https://dashscope.invalid/compatible-mode/v1", {"HTTPS_PROXY": url})
    assert any(line.startswith("CONNECT dashscope.invalid:443 ") for line in seen), seen

    # NO_PROXY sends the request around the proxy: it goes straight to
    # vllm.invalid, whose name never resolves.
    seen.clear()
    chain = _transcribe(
        "vllm", "http://vllm.invalid:8000/v1", {"HTTP_PROXY": url, "NO_PROXY": "vllm.invalid"}
    )
    assert seen == []
    assert "gaierror" in chain, chain
