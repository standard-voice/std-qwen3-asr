# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Small audio helpers shared by the engine and backends.

These are deliberately tiny and dependency-light: a WAV-wrapper for raw PCM16 and
a float32->PCM16 quantizer that follows the spec's canonical quantization
convention (clip to [-1, 1], round-half, little-endian int16) so the bytes match
what other Standard ASR implementations produce (spec AI R4 quantization note).
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def float32_to_pcm16(samples: NDArray[np.float32]) -> bytes:
    """Quantize a float32 mono waveform to little-endian signed PCM16 bytes.

    Follows the spec's canonical convention: clip to ``[-1.0, 1.0]``, then
    ``round_half(sample * 32767)`` to the nearest int16 (NOT truncation), written
    little-endian (spec AI R4). Non-finite samples are sanitized to ``0`` before
    the cast (int16 cannot represent NaN).

    Args:
        samples: A float32 waveform array.

    Returns:
        The PCM16 byte string.
    """
    arr: NDArray[np.float32] = np.asarray(samples, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped: NDArray[np.float32] = arr.clip(-1.0, 1.0)
    quantized: NDArray[np.int16] = np.round(clipped * 32767.0).astype("<i2")
    return quantized.tobytes()


def wrap_pcm16_wav(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw PCM16 bytes in a minimal WAV (RIFF) container.

    Args:
        pcm: Raw little-endian signed 16-bit PCM samples.
        sample_rate: Sample rate in Hz.
        channels: Channel count (mono for v1 streaming).

    Returns:
        WAV-encoded bytes (44-byte header + PCM data).
    """
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_size = len(pcm)
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + pcm
