# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures and helpers for the std-qwen3-asr test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

#: The reference English test clip shipped with the standard_asr repo (~57s).
TEST_AUDIO_PATH = (
    Path(__file__).resolve().parents[2]
    / "standard_asr"
    / "reference"
    / "standard_asr_test_audio_english.m4a"
)


@pytest.fixture
def pcm16_frames() -> list[bytes]:
    """Return a few small PCM16 mono frames (16 kHz) for streaming tests.

    Returns:
        A list of raw PCM16 byte chunks.
    """
    rng = np.random.default_rng(0)
    chunks: list[bytes] = []
    for _ in range(3):
        samples = (rng.standard_normal(1600) * 0.1).astype(np.float32)
        pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        chunks.append(pcm)
    return chunks


@pytest.fixture
def silence_array() -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Return one second of 16 kHz float32 silence as an audio array.

    Returns:
        A float32 silence waveform.
    """
    return np.zeros(16000, dtype=np.float32)
