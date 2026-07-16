# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Batch integration tests against the in-process fake backend server.

These run end-to-end on this machine (no GPU): the adapter connects to a real
``http://127.0.0.1`` fake server over real sockets and we assert both the
resulting ``TranscriptionResult`` AND what the adapter put on the wire (so the
request mapping -- language code, prompt->context, itn -- is verified, not just
the happy-path text).
"""

from __future__ import annotations

import numpy as np
import pytest
from standard_asr import AudioArray, RuntimeParams
from standard_asr.contract.exceptions import ConfigError, TranscriptionError

from std_qwen3_asr import Qwen3ASR, Qwen3ASR17B, Qwen3ASRParams

from .conftest import TEST_AUDIO_PATH
from .fake_server import FakeConfig, running_server


def test_vllm_batch_array(silence_array: np.ndarray) -> None:
    with running_server(FakeConfig(transcript="hello world", language="en")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        result = engine.transcribe(AudioArray(silence_array, 16000))
    assert result.text == "hello world"
    # vLLM's response_format=json returns only {"text": ...} with NO language
    # field, so detected_language is honestly None on this path (the DashScope
    # chat path DOES carry it -- see test_dashscope_batch_chat_shape).
    assert result.detected_language is None
    # The adapter uploaded a multipart file to the transcription endpoint.
    req = server.requests[-1]
    assert req.path == "/v1/audio/transcriptions"
    assert req.form["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert req.file_len > 0


def test_vllm_batch_maps_language_and_prompt(silence_array: np.ndarray) -> None:
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        engine.transcribe(
            AudioArray(silence_array, 16000),
            params=RuntimeParams(
                language="zh-CN",
                prompt="量子计算",  # Qwen context biasing via the portable prompt channel
                provider_params=Qwen3ASRParams(enable_itn=True),
            ),
        )
    form = server.requests[-1].form
    assert form["language"] == "zh"  # zh-CN -> zh ISO code
    assert form["prompt"] == "量子计算"
    assert "enable_itn" in form["vllm_xargs"]


def test_vllm_batch_auto_language_omits_field(silence_array: np.ndarray) -> None:
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        # default_language is "auto" -> no language field sent.
        engine.transcribe(AudioArray(silence_array, 16000))
    assert "language" not in server.requests[-1].form


def test_dashscope_batch_chat_shape(silence_array: np.ndarray) -> None:
    with running_server(FakeConfig(transcript="ni hao", language="zh", emotion="happy")) as server:
        engine = Qwen3ASR(backend="dashscope", base_url=server.http_base_url, api_key="sk-test")
        result = engine.transcribe(
            AudioArray(silence_array, 16000),
            params=RuntimeParams(prompt="context here"),
        )
    assert result.text == "ni hao"
    assert result.detected_language == "zh"
    # Emotion is engine-specific -> surfaced in extra, never standardized metadata.
    assert result.extra["emotion"] == "happy"
    assert result.metadata == {}
    # The adapter sent a chat-completions body with input_audio + system context.
    body = server.requests[-1].json_body
    assert body["model"] == "qwen3-asr-flash"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "context here"
    assert body["messages"][-1]["content"][0]["type"] == "input_audio"
    assert body["asr_options"]["enable_itn"] is False


def test_dashscope_requires_api_key(silence_array: np.ndarray) -> None:
    engine = Qwen3ASR(backend="dashscope", base_url="https://dashscope.example.com/v1")
    # ConfigError from the lazy backend build propagates before the transcribe
    # try/except (a caller-fixable error, not a runtime TranscriptionError).
    with pytest.raises(ConfigError, match="api_key"):
        engine.transcribe(AudioArray(silence_array, 16000))


def test_batch_backend_error_wrapped(silence_array: np.ndarray) -> None:
    # A backend 500 must surface as a portable TranscriptionError (spec R7).
    with running_server(FakeConfig(fail_status=500)) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        with pytest.raises(TranscriptionError):
            engine.transcribe(AudioArray(silence_array, 16000))


def test_bearer_token_forwarded(silence_array: np.ndarray) -> None:
    with running_server(FakeConfig(require_bearer="tok-123")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, api_key="tok-123")
        result = engine.transcribe(AudioArray(silence_array, 16000))
    assert result.text  # 200 means the bearer matched


def test_bearer_token_mismatch_fails(silence_array: np.ndarray) -> None:
    with running_server(FakeConfig(require_bearer="right")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, api_key="wrong")
        with pytest.raises(TranscriptionError):
            engine.transcribe(AudioArray(silence_array, 16000))


@pytest.mark.skipif(not TEST_AUDIO_PATH.exists(), reason="reference test audio not available")
def test_real_audio_negotiation_through_fake(silence_array: np.ndarray) -> None:
    # Exercise the FULL audio-negotiation path on the real reference clip
    # (48 kHz stereo m4a): the standard layer decodes + resamples + downmixes to
    # 16 kHz mono and hands the adapter encoded bytes, which it uploads. The fake
    # returns a canned transcript -- this proves the adapter's audio handling and
    # upload work on a real file on THIS machine (no GPU / no real model).
    with running_server(FakeConfig(transcript="reference transcript")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        result = engine.transcribe(str(TEST_AUDIO_PATH))
    assert result.text == "reference transcript"
    assert server.requests[-1].file_len > 1000  # a real ~57s clip was uploaded
