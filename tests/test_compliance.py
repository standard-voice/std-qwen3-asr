# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR compliance checks wired into the test suite.

The five compliance dimensions (plugin_entrypoints.md): entry-point metadata,
streaming parameter gating, the async->sync bridge, the event-sequence contract,
and provider_params swap-safety. ``check_event_sequence`` is exercised against a
real recorded stream in ``test_integration_streaming.py``; the rest live here.
"""

from __future__ import annotations

from standard_asr.audio.format import AudioFormat
from standard_asr.compliance import (
    check_entrypoints,
    check_provider_params_swap_safety,
    check_streaming_param_gating,
    check_sync_bridge,
)

from std_qwen3_asr import Qwen3ASR17B

from .fake_server import running_server

PCM_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def test_entrypoints_compliant() -> None:
    report = check_entrypoints()
    assert report.passed, [i.message for i in report.issues]


def test_streaming_param_gating_compliant() -> None:
    # The open-weight preset constructs with no required credentials (vLLM may be
    # unauthenticated), so it can be probed without secrets.
    report = check_streaming_param_gating(Qwen3ASR17B())
    assert report.passed, [i.message for i in report.issues]


def test_provider_params_swap_safety() -> None:
    report = check_provider_params_swap_safety(Qwen3ASR17B())
    assert report.passed, [i.message for i in report.issues]


def test_sync_bridge_no_deadlock() -> None:
    # Open a real session against the fake realtime server and assert the bridge
    # terminates without deadlock or a leaked thread.
    with running_server() as server:
        engine = Qwen3ASR17B(base_url=server.realtime_base_url, stream_transport="realtime")
        report = check_sync_bridge(lambda: engine.start_transcription(audio_format=PCM_FORMAT))
    assert report.passed, [i.message for i in report.issues]
