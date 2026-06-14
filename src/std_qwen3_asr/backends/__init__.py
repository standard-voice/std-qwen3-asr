# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend transports for the Qwen3-ASR adapter.

Each backend speaks one remote protocol and normalizes it to the small internal
shapes in :mod:`std_qwen3_asr.backends.base`. The engine selects a backend from
config; the rest of the adapter (event mapping, gating) is backend-agnostic.
"""

from .base import (
    Backend,
    BatchRequest,
    BatchResult,
    StreamRequest,
    TranscriptDelta,
)

__all__ = [
    "Backend",
    "BatchRequest",
    "BatchResult",
    "StreamRequest",
    "TranscriptDelta",
]
