# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine + session path coverage: input shapes, defaults, prepare, stream edges."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from standard_asr import AudioArray, AudioBytes
from standard_asr.audio_format import AudioFormat
from standard_asr.exceptions import TranscriptionError

from std_qwen3_asr import Qwen3ASR, Qwen3ASR17B, Qwen3ASRConfig
from std_qwen3_asr._audio import float32_to_pcm16, wrap_pcm16_wav
from std_qwen3_asr.backends.base import StreamRequest, TranscriptDelta
from std_qwen3_asr.session import Qwen3ASRSession

from .conftest import TEST_AUDIO_PATH
from .fake_server import FakeConfig, running_server

PCM_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


# --------------------------------------------------------------------------- #
# Default base_url resolution + prepare()
# --------------------------------------------------------------------------- #
def test_default_base_url_per_backend() -> None:
    vllm = Qwen3ASR17B()
    vllm_cfg = cast(Qwen3ASRConfig, vllm.config)
    assert vllm._resolved_base_url(vllm_cfg).endswith(":8000/v1")  # pyright: ignore[reportPrivateUsage]
    flash = Qwen3ASR(api_key="k")  # dashscope default
    flash_cfg = cast(Qwen3ASRConfig, flash.config)
    assert "dashscope" in flash._resolved_base_url(flash_cfg)  # pyright: ignore[reportPrivateUsage]


def test_dashscope_default_base_url_is_built() -> None:
    # No base_url + dashscope backend -> the backend is built with the default
    # international endpoint (exercises the lazy build + default resolution).
    engine = Qwen3ASR(api_key="k")  # flash preset, dashscope default, no base_url
    assert engine.config.base_url is None  # validator returned None for unset
    engine.prepare()
    assert engine._backend is not None  # pyright: ignore[reportPrivateUsage]


def test_prepare_builds_backend() -> None:
    engine = Qwen3ASR17B(base_url="http://localhost:8000/v1")
    assert engine._backend is None  # pyright: ignore[reportPrivateUsage]
    engine.prepare()
    backend = engine._backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    # A second ensure/prepare reuses the existing backend (no rebuild).
    engine.prepare()
    assert engine._backend is backend  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- #
# Batch input shapes (encoded bytes, encoded file)
# --------------------------------------------------------------------------- #
def test_batch_encoded_bytes_input() -> None:
    # Build a tiny valid WAV and feed it as AudioBytes (ENCODED_BYTES passthrough).
    pcm = float32_to_pcm16(np.zeros(1600, dtype=np.float32))
    wav = wrap_pcm16_wav(pcm, sample_rate=16000)
    with running_server(FakeConfig(transcript="from bytes")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        result = engine.transcribe(AudioBytes(wav, container="wav"))
    assert result.text == "from bytes"
    assert server.requests[-1].file_len == len(wav)


def test_batch_encoded_file_input(tmp_path: object) -> None:
    from pathlib import Path

    pcm = float32_to_pcm16(np.zeros(1600, dtype=np.float32))
    wav = wrap_pcm16_wav(pcm, sample_rate=16000)
    p = Path(str(tmp_path)) / "a.wav"
    p.write_bytes(wav)
    with running_server(FakeConfig(transcript="from file")) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url)
        result = engine.transcribe(str(p))
    assert result.text == "from file"
    assert server.requests[-1].file_len == len(wav)


# --------------------------------------------------------------------------- #
# Streaming whole-input with an in-memory array (no file decode path)
# --------------------------------------------------------------------------- #
async def test_streaming_whole_input_array(silence_array: np.ndarray) -> None:
    with running_server(FakeConfig(deltas=["arr ", "input"])) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, stream_transport="sse")
        session = engine.start_transcription(audio=AudioArray(silence_array, 16000))
        events: list[object] = []
        async with session:
            async for ev in session:
                events.append(ev)
        result = session.result()
    assert result.text == "arr input"


async def test_streaming_whole_input_encoded_bytes() -> None:
    # Whole-input streaming where the input is ENCODED bytes (the engine decodes
    # them to PCM via the standard loader -- exercises the decode branch).
    pcm = float32_to_pcm16(np.zeros(8000, dtype=np.float32))
    wav = wrap_pcm16_wav(pcm, sample_rate=16000)
    with running_server(FakeConfig(deltas=["bytes ", "stream"])) as server:
        engine = Qwen3ASR17B(base_url=server.http_base_url, stream_transport="sse")
        session = engine.start_transcription(audio=AudioBytes(wav, container="wav"))
        async with session:
            async for _ in session:
                pass
        result = session.result()
    assert result.text == "bytes stream"


def test_streaming_whole_input_undecodable_bytes_raises() -> None:
    # Garbage "encoded" bytes that the loader cannot decode -> the engine wraps
    # the failure as a TranscriptionError at session construction.
    engine = Qwen3ASR17B(base_url="http://localhost:8000/v1", stream_transport="sse")
    with pytest.raises(TranscriptionError, match="decode"):
        engine.start_transcription(audio=AudioBytes(b"not audio at all", container="wav"))


# --------------------------------------------------------------------------- #
# Session detected_language mapping + empty-delta skip (direct)
# --------------------------------------------------------------------------- #
class _ScriptedBackend:
    """A backend whose stream yields a fixed list of deltas (incl. empties)."""

    def __init__(self, deltas: list[TranscriptDelta]) -> None:
        self._deltas = deltas

    async def transcribe(self, request: object) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    async def stream(self, request: object, audio_source: object):  # type: ignore[no-untyped-def]
        # Drain the audio source so the session's pump completes.
        async for _ in audio_source:  # type: ignore[attr-defined]
            pass
        for d in self._deltas:
            yield d

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None


async def test_session_maps_language_and_skips_empty_delta() -> None:
    backend = _ScriptedBackend(
        [
            TranscriptDelta(
                text="", detected_language="English"
            ),  # empty -> skipped, lang captured
            TranscriptDelta(text="hello"),
            TranscriptDelta(text=" world"),
        ]
    )
    request = StreamRequest(
        language=None, prompt=None, temperature=0.0, sample_rate=16000, enable_itn=False
    )
    session = Qwen3ASRSession(backend=backend, request=request)  # type: ignore[arg-type]
    session.feed([b"\x00\x00" * 100])
    events = []
    async with session:
        async for ev in session:
            events.append(ev)

    partials = [e for e in events if e.type == "partial"]
    finals = [e for e in events if e.type == "final"]
    # Empty leading delta produced no partial; cumulative text grows over the rest.
    assert finals[0].text == "hello world"
    # The English name was mapped to BCP-47 'en' and attached.
    assert finals[0].detected_language == "en"
    assert all(e.detected_language == "en" for e in partials if e.detected_language)


def test_test_audio_reference_exists() -> None:
    # Document, as a test, that the reference clip the suite relies on is present.
    assert TEST_AUDIO_PATH.exists(), f"missing reference audio at {TEST_AUDIO_PATH}"
