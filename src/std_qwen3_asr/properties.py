# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static identity and I/O boundaries for the Qwen3-ASR presets (Properties).

Properties are the engine's fixed, machine-readable identity (spec G.1.3): what
audio shapes it accepts, its sample-rate boundaries, the streaming wire encoding
whitelist, size/duration limits, and the BCP-47 languages it can select/detect.
They are read off the *class* without instantiation (discovery, ``show``,
REST ``GET /v1/capabilities``), so they live as ``ClassVar`` instances.

Audio shape rationale:

* ``ENCODED_FILE`` / ``ENCODED_BYTES`` -- both vLLM (``/v1/audio/transcriptions``,
  multipart file upload) and DashScope (base64 data-URI in the request body)
  consume *encoded* audio, not raw arrays. Declaring both lets the standard layer
  hand us an in-memory upload without a temp file, and lets a path passthrough
  when the app already has a file.
* ``ARRAY`` -- declared so an app that already holds a decoded waveform doesn't
  pay an encode->decode round trip on its side; the standard layer encodes the
  array to a canonical 16 kHz mono WAV for us (spec AI R4) before we upload.

Sample rate: Qwen3-ASR runs natively at 16 kHz. We accept only 16 kHz
(``accepted_sample_rates=[16000]``); the standard layer resamples other rates to
it on the *batch* path. For the *streaming* wire we set
``required_input_sample_rate=16000`` because the vLLM Realtime endpoint hard
requires PCM16 @ 16 kHz and v1 does not resample streaming frames (spec AI R7) --
so an off-rate stream fails loudly at session establishment rather than being
mistranscribed.

Limits: the hosted DashScope *sync* path caps a single request at 10 MB / ~3
minutes; we record ``max_file_size`` conservatively at the DashScope sync bound.
Self-hosted vLLM has a configurable, generally higher bound, but the smaller
documented limit is the safe portable promise (an app can always chunk).
"""

from __future__ import annotations

from typing import Literal

from standard_asr.engine import BaseProperties, InputKind, SampleRateRange

from .languages import DETECTABLE_LANGUAGES, SELECTABLE_LANGUAGES

#: DashScope sync per-request payload cap (10 MB). The safe cross-backend
#: promise; self-hosted vLLM is usually higher but apps can always chunk.
_DASHSCOPE_SYNC_MAX_BYTES = 10 * 1024 * 1024
#: DashScope sync per-request duration cap (~3 minutes). Documented as the
#: load-bearing constraint of the hosted service.
_DASHSCOPE_SYNC_MAX_SECONDS = 180.0


class Qwen3ASRProperties(BaseProperties):
    """Static metadata for the canonical ``qwen3-asr/flash`` preset.

    ``model_name`` MUST equal the entry-point preset key's model component so
    ``properties.model_id`` matches the registered key (compliance-enforced). The
    base describes the hosted Flash preset; the open-weight presets subclass this
    and override ``model_name`` (+ ``description``) only.
    """

    engine_id: str = "qwen3-asr"
    model_name: str = "flash"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {
        InputKind.ENCODED_FILE,
        InputKind.ENCODED_BYTES,
        InputKind.ARRAY,
    }
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    # vLLM Realtime streaming hard-requires PCM16 @ 16 kHz and v1 does not
    # resample streaming wire frames -- pin the streaming wire rate so an off-rate
    # session fails loudly (spec AI R7 / ST §3.1).
    required_input_sample_rate: int | None = 16000
    # Canonical streaming wire encoding (16-bit signed LE PCM). The vLLM Realtime
    # API ingests exactly this (``(audio*32767).astype(int16).tobytes()``).
    wire_encodings: list[str] | None = ["pcm_s16le"]
    max_file_size: int | None = _DASHSCOPE_SYNC_MAX_BYTES
    max_audio_duration: float | None = _DASHSCOPE_SYNC_MAX_SECONDS
    selectable_languages: list[str] = SELECTABLE_LANGUAGES
    detectable_languages: list[str] = DETECTABLE_LANGUAGES
    description: str | None = (
        "Qwen3-ASR-Flash (hosted DashScope) via the OpenAI-compatible backend."
    )


class Qwen3ASR17BProperties(Qwen3ASRProperties):
    """Static metadata for the ``qwen3-asr/1.7b`` open-weight preset (vLLM)."""

    model_name: str = "1.7b"
    # Self-hosted vLLM has no fixed 10 MB / 3-min sync cap; leave size/duration
    # limits unset (None) so the standard layer does not reject larger uploads
    # that a local server can handle. Apps that target DashScope should use the
    # 'flash' preset, which carries the hosted caps.
    max_file_size: int | None = None
    max_audio_duration: float | None = None
    description: str | None = (
        "Qwen3-ASR 1.7B open weights (Apache-2.0) served by a self-hosted vLLM server."
    )


class Qwen3ASR06BProperties(Qwen3ASRProperties):
    """Static metadata for the ``qwen3-asr/0.6b`` open-weight preset (vLLM)."""

    model_name: str = "0.6b"
    max_file_size: int | None = None
    max_audio_duration: float | None = None
    description: str | None = (
        "Qwen3-ASR 0.6B open weights (Apache-2.0) served by a self-hosted vLLM server."
    )
