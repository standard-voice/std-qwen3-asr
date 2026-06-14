# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo: stream against the bundled mock vLLM server (no GPU needed).

This is the GPU-free way to see the adapter's full streaming path on this
machine. It starts the in-process fake vLLM Realtime server (from the test
suite), opens a Standard ASR streaming session pointed at it, feeds synthetic
PCM frames, and prints the Standard ASR events as they arrive.

Run::

    uv run python examples/stream_against_mock.py

For the REAL vLLM path (requires a CUDA host running ``vllm serve
Qwen/Qwen3-ASR-1.7B``), see ``examples/stream_against_vllm.py`` and
``VERIFICATION.md``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from standard_asr.audio_format import AudioFormat

# Make the test-suite fake server importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_server import FakeConfig, running_server  # noqa: E402

from std_qwen3_asr import Qwen3ASR17B  # noqa: E402

PCM_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def _frames() -> list[bytes]:
    """Build a few seconds of synthetic 16 kHz PCM16 frames."""
    rng = np.random.default_rng(0)
    out: list[bytes] = []
    for _ in range(5):
        samples = (rng.standard_normal(1600) * 0.05).astype(np.float32)
        out.append(np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return out


async def main() -> None:
    """Stream against the mock server and print every Standard ASR event."""
    with running_server(
        FakeConfig(deltas=["Hello ", "from ", "Qwen3-", "ASR ", "streaming."])
    ) as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        session = engine.start_transcription(audio_format=PCM_FORMAT)
        session.feed(_frames())
        print("--- streaming events ---")
        async with session:
            async for event in session:
                if event.type in ("partial", "final"):
                    print(
                        f"{event.type:8} seg={event.segment_id} "
                        f"stable_until={event.stable_until} text={event.text!r}"
                    )
                else:
                    print(f"{event.type:8} {event}")
        print("--- reduced result ---")
        print(repr(session.result().text))


if __name__ == "__main__":
    asyncio.run(main())
