# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR plugin for Qwen3-ASR over a vLLM (OpenAI-compatible) backend.

A thin client adapter: it connects to a self-hosted vLLM server serving a
Qwen3-ASR checkpoint, or to the hosted DashScope ``qwen3-asr-flash`` service, and
exposes both batch and streaming transcription through the Standard ASR protocol.
The package never imports vLLM, torch, or Qwen's inference code -- those run on
the remote host (dependency + license isolation, spec G.4.2).

Public surface: the engine classes, config, properties, params, capabilities, and
the entry-point factories. Engine *selection* is by preset (entry point), not by
constructing these directly, but they are exported for typing and advanced use.
"""

from .capabilities import QWEN3_ASR_CAPABILITIES
from .config import Backend, Qwen3ASRConfig, StreamTransport
from .engine import Qwen3ASR, Qwen3ASR06B, Qwen3ASR17B
from .entrypoint import create_0_6b, create_1_7b, create_flash
from .params import Qwen3ASRParams
from .properties import (
    Qwen3ASR06BProperties,
    Qwen3ASR17BProperties,
    Qwen3ASRProperties,
)
from .session import Qwen3ASRSession

__all__ = [
    "QWEN3_ASR_CAPABILITIES",
    "Backend",
    "Qwen3ASR",
    "Qwen3ASR06B",
    "Qwen3ASR06BProperties",
    "Qwen3ASR17B",
    "Qwen3ASR17BProperties",
    "Qwen3ASRConfig",
    "Qwen3ASRParams",
    "Qwen3ASRProperties",
    "Qwen3ASRSession",
    "StreamTransport",
    "create_0_6b",
    "create_1_7b",
    "create_flash",
]
