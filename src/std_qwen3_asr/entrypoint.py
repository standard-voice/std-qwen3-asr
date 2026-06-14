# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factories registered under ``standard_asr.models``.

Each factory returns a CONCRETE engine class (annotated as such) so the registry
can read class-level metadata (``properties``, ``declared_capabilities``,
``provider_params_type``) WITHOUT instantiating or authenticating the engine
(spec §3.1 / adapting_engine.md). Annotating ``-> StandardASR`` would break
instantiation-free discovery; the compliance suite enforces the concrete return.
"""

from __future__ import annotations

from typing import Any

from .engine import Qwen3ASR, Qwen3ASR06B, Qwen3ASR17B


def create_flash(**kwargs: Any) -> Qwen3ASR:
    """Create the ``qwen3-asr/flash`` preset (hosted DashScope Flash service).

    Args:
        **kwargs: Optional config overrides forwarded to :class:`Qwen3ASR`.

    Returns:
        A configured :class:`Qwen3ASR` instance (DashScope backend by default).
    """
    return Qwen3ASR(**kwargs)


def create_1_7b(**kwargs: Any) -> Qwen3ASR17B:
    """Create the ``qwen3-asr/1.7b`` preset (open weights via self-hosted vLLM).

    Args:
        **kwargs: Optional config overrides forwarded to :class:`Qwen3ASR17B`.

    Returns:
        A configured :class:`Qwen3ASR17B` instance (vLLM backend by default).
    """
    return Qwen3ASR17B(**kwargs)


def create_0_6b(**kwargs: Any) -> Qwen3ASR06B:
    """Create the ``qwen3-asr/0.6b`` preset (open weights via self-hosted vLLM).

    Args:
        **kwargs: Optional config overrides forwarded to :class:`Qwen3ASR06B`.

    Returns:
        A configured :class:`Qwen3ASR06B` instance (vLLM backend by default).
    """
    return Qwen3ASR06B(**kwargs)
