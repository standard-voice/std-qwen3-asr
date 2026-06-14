# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Declared capabilities for the Qwen3-ASR backend (fail-closed).

Capabilities are a fail-closed promise: we declare ONLY what we genuinely
deliver; anything omitted is treated by applications as unsupported (spec
Capabilities R1). The declarations below are deliberately conservative and are
identical across presets (the hosted Flash service and the open-weight vLLM
builds expose the same behavior through the same wire shapes).

Batch (``transcribe``):

* ``language.runtime_override`` -- yes. A per-request BCP-47 ``language`` is
  mapped to the backend's language field (DashScope ``asr_options.language`` /
  vLLM ``language``). NOTE: vLLM language forcing is currently unreliable
  upstream (see findings); we still gate-accept the parameter (the standard
  contract) and forward it, documenting it as best-effort on the vLLM backend.
* ``guidance.prompt`` -- yes, this is Qwen3-ASR's signature "context" biasing
  feature reached through the portable ``prompt`` channel (spec §5.3). Qwen
  documents a 10,000-token context budget; we declare a conservative
  ``max_tokens`` well under it (the standard counts tokens with a script-aware
  approximation that under-counts BPE, so headroom is required -- spec §3.3).
* ``word_timestamps`` -- NOT declared (omitted => unsupported). The hosted Flash
  REST response returns text only (no word/segment timestamps), and Qwen3-ASR's
  streaming path explicitly does not return timestamps. Declaring even
  ``segment`` granularity would be a false promise on this backend.
* ``phrase_hints`` -- NOT declared. Qwen biases via free-text ``context``
  (mapped from ``prompt``), not a structured term-boost list; there is no honest
  ``phrase_hints`` channel, so we omit it (an app can pass terms inside
  ``prompt``).

Streaming (``start_transcription``):

* ``streaming_input`` + ``streaming_output`` -- yes (vLLM streams incrementally).
* ``emits_partials`` -- yes (we accumulate token deltas into a growing
  cumulative ``partial`` per spec §4.3).
* ``re_segments`` -- no. vLLM token streaming is append-only; we never retract
  or merge/split a finalized segment, so we never emit ``supersede``.
* ``word_stability`` -- **false**. The streaming wire carries no per-token
  timestamps or right-context, so we report ``stable_until=0`` for every event.
  The spec names Qwen3-ASR streaming as the canonical case for this (ST §4.2):
  "engines with no right_context or timestamp info (e.g. Qwen3-ASR streaming)
  MUST report stable_until=0; word_stability MUST be declared false."
* ``reconnect`` -- ``unsupported``. The vLLM Realtime endpoint is stateful with
  hardcoded segmentation and offers no loss-free resume contract; we declare
  ``unsupported`` rather than over-promise ``seamless``/``lossy``.
* ``finality_level`` -- ``final`` (default). We do not guarantee a
  post-processing-immutable ``closed`` terminal, so we conservatively declare
  ``final`` (spec §5.3 "default conservative").
* ``timestamps`` -- ``none`` (no streaming timestamps).
"""

from __future__ import annotations

from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PromptCap,
    PromptConstraints,
    ReconnectCap,
    StreamingCapabilities,
    StreamingGuidanceCaps,
)

#: Conservative prompt/context token budget. Qwen documents ~10,000 tokens for
#: ``context``; the standard's script-aware approximation under-counts BPE for
#: Latin text, so we declare well under the hard cap to keep headroom (spec
#: §3.3 max_tokens guidance).
_PROMPT_MAX_TOKENS = 2000

QWEN3_ASR_CAPABILITIES = DeclaredCapabilities(
    # Top-level orthogonal streaming axes (siblings of the batch/streaming
    # subtrees, spec ST §3.2). BOTH are genuinely true: the vLLM Realtime path
    # ingests audio incrementally (streaming_input) AND emits transcript deltas
    # before all input is consumed (streaming_output). Default is False
    # (fail-closed), so they MUST be set explicitly.
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        guidance=GuidanceCaps(
            prompt=PromptCap(
                supported=True,
                constraints=PromptConstraints(max_tokens=_PROMPT_MAX_TOKENS),
            ),
        ),
    ),
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        guidance=StreamingGuidanceCaps(
            prompt=PromptCap(
                supported=True,
                constraints=PromptConstraints(max_tokens=_PROMPT_MAX_TOKENS),
            ),
        ),
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=False),
        word_stability=FlagCap(supported=False),
        reconnect=ReconnectCap(mode="unsupported"),
        finality_level=FinalityCap(mode="final"),
        # timestamps defaults to mode="none" -- no streaming timestamps.
    ),
)
