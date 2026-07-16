# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real vLLM streaming demo (requires a CUDA host running vLLM + Qwen3-ASR).

This points the adapter at a real vLLM server and streams the reference clip.
It CANNOT run on a machine without a CUDA GPU (vLLM is CUDA-first). On a GPU
host, first launch the server::

    pip install "vllm[audio]"
    vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
    # (or: qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000)

Then, with this package installed, run::

    STD_QWEN3_VLLM_URL=http://<gpu-host>:8000/v1 \
        uv run python examples/stream_against_vllm.py /path/to/audio.wav

The script feeds 16 kHz mono PCM16 frames over the vLLM Realtime WebSocket and
prints Standard ASR events. See VERIFICATION.md for the full reproduction.
"""

from __future__ import annotations

import asyncio
import os
import sys
import wave

from standard_asr.audio.format import AudioFormat

from std_qwen3_asr import Qwen3ASR17B

PCM_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def _read_pcm16_frames(path: str, frame_ms: int = 100) -> list[bytes]:
    """Read a 16 kHz mono PCM16 WAV into fixed-size frames.

    Args:
        path: Path to a 16 kHz mono 16-bit PCM WAV file.
        frame_ms: Frame size in milliseconds.

    Returns:
        A list of raw PCM16 frames.

    Raises:
        ValueError: If the WAV is not 16 kHz mono 16-bit PCM.
    """
    with wave.open(path, "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(
                "Expected 16 kHz mono 16-bit PCM WAV. Convert with: "
                f"ffmpeg -i {path} -ar 16000 -ac 1 -c:a pcm_s16le out.wav"
            )
        frames_per_chunk = int(16000 * frame_ms / 1000)
        out: list[bytes] = []
        while True:
            chunk = wav.readframes(frames_per_chunk)
            if not chunk:
                break
            out.append(chunk)
    return out


async def main(path: str) -> None:
    """Stream a WAV file against a real vLLM server and print events."""
    base_url = os.environ.get("STD_QWEN3_VLLM_URL", "http://localhost:8000/v1")
    engine = Qwen3ASR17B(base_url=base_url, stream_transport="realtime")
    session = engine.start_transcription(audio_format=PCM_FORMAT)
    session.feed(_read_pcm16_frames(path))
    async with session:
        async for event in session:
            if event.type in ("partial", "final"):
                print(f"{event.type:8} stable_until={event.stable_until} {event.text!r}")
            elif event.type == "error":
                print(f"ERROR code={event.code} recoverable={event.recoverable}")
    print("final transcript:", repr(session.result().text))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/stream_against_vllm.py <16k-mono-pcm16.wav>")
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1]))
