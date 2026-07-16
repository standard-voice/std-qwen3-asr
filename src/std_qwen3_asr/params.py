# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine-specific runtime parameters for Qwen3-ASR (the escape hatch).

Per spec Runtime §3.1/§3.2, the *portable* request set (``language``,
``prompt``, ``phrase_hints``, ``word_timestamps``, ``candidate_languages``) is
gated for us by the standard layer; engine-private knobs that have no portable
equivalent live here, in a typed ``ProviderParams`` subclass with
``extra="forbid"`` (so a typo or a swapped-engine params object is rejected --
spec Runtime R3 swap-safety).

**Mapping note (spec §5.3):** Qwen3-ASR's headline "context" biasing feature is
reached through the *portable* ``prompt`` channel, NOT a knob here -- the spec's
canonical example maps Standard ASR ``prompt`` -> Qwen ``context``. Putting it
here too would create two truths for one feature. So ``context`` is deliberately
absent; use ``RuntimeParams.prompt``.

This class must be a *distinct terminal type* (never the bare ``ProviderParams``)
so swap-safety has something exact to check against (spec Runtime §3.2).
"""

from __future__ import annotations

from pydantic import Field
from standard_asr.contract.params import ProviderParams


class Qwen3ASRParams(ProviderParams):
    """Non-portable decoding knobs for Qwen3-ASR.

    Args:
        enable_itn: Inverse text normalization (e.g. "one hundred" -> "100",
            spoken dates/numbers -> written form). Qwen supports this for
            Chinese and English only; for other languages it is a no-op upstream.
            Maps to DashScope ``asr_options.enable_itn`` and is forwarded to vLLM
            via ``vllm_xargs`` when set.
        temperature: Sampling temperature for the underlying LLM decode. ``0.0``
            is greedy/deterministic (recommended for transcription). Forwarded as
            an OpenAI sampling param.
        top_p: Nucleus sampling parameter, forwarded when set.
        max_completion_tokens: Hard cap on generated tokens (a safety bound for
            pathological audio); forwarded when set.
        emotion: Request Qwen's per-utterance emotion annotation when the backend
            supports it. The detected emotion (if any) is surfaced in the
            result's ``extra`` channel (engine-specific, spec TR.1), never in
            standardized ``metadata``.
    """

    enable_itn: bool = Field(
        default=False,
        description="Inverse text normalization (Chinese/English only upstream).",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Decode sampling temperature; 0.0 = greedy (recommended).",
    )
    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling top_p; forwarded when set.",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Optional hard cap on generated tokens.",
    )
    emotion: bool = Field(
        default=False,
        description="Request per-utterance emotion annotation (surfaced in result.extra).",
    )
