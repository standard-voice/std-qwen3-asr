# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Artifact-lifecycle declaration and report behavior (protocol 1.1).

Every preset is a client for a separately operated service (DashScope or a
deployed vLLM server), so the whole engine family declares
``NO_ARTIFACT_ACQUISITION`` and reports a not-applicable artifact status.
"""

from __future__ import annotations

import pytest
from standard_asr import ARTIFACTS_NOT_APPLICABLE
from standard_asr.engine import NO_ARTIFACT_ACQUISITION

from std_qwen3_asr import Qwen3ASR, Qwen3ASR06B, Qwen3ASR17B

PRESETS = [Qwen3ASR, Qwen3ASR17B, Qwen3ASR06B]


@pytest.mark.parametrize("preset", PRESETS)
def test_declared_metadata_is_no_acquisition(preset: type[Qwen3ASR]) -> None:
    # Class-level read, no instantiation: discovery and the metadata endpoint
    # read this without constructing a client or resolving credentials.
    assert preset.declared_metadata.artifacts == NO_ARTIFACT_ACQUISITION


@pytest.mark.parametrize("preset", PRESETS)
def test_protocol_version_is_1_1(preset: type[Qwen3ASR]) -> None:
    assert preset.properties.protocol_version == "1.1.0"


def test_artifact_status_reports_not_applicable() -> None:
    # The open-weight preset constructs without credentials (vLLM may be
    # unauthenticated), so status can run without secrets.
    report = Qwen3ASR17B().artifact_status()
    assert report.applicable is False
    assert report.requirements == ()
    assert report.readiness == ARTIFACTS_NOT_APPLICABLE


def test_acquire_artifacts_is_an_idempotent_no_op() -> None:
    engine = Qwen3ASR17B()
    first = engine.acquire_artifacts()
    second = engine.acquire_artifacts()
    assert first.readiness == ARTIFACTS_NOT_APPLICABLE
    assert second == first
