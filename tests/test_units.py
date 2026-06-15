# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure helper modules (no network, no server)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from std_qwen3_asr import (
    QWEN3_ASR_CAPABILITIES,
    Qwen3ASR,
    Qwen3ASR06B,
    Qwen3ASR17B,
    Qwen3ASRConfig,
    Qwen3ASRParams,
)
from std_qwen3_asr._audio import float32_to_pcm16, wrap_pcm16_wav
from std_qwen3_asr._json import (
    as_object,
    first_object,
    get_float,
    get_list,
    get_object,
    get_str,
    loads_object,
)
from std_qwen3_asr.languages import (
    DETECTABLE_LANGUAGES,
    SELECTABLE_LANGUAGES,
    from_backend_language,
    to_dashscope_code,
    to_qwen_name,
)


# --------------------------------------------------------------------------- #
# languages
# --------------------------------------------------------------------------- #
def test_selectable_languages_include_auto_and_are_unique() -> None:
    assert "auto" in SELECTABLE_LANGUAGES
    assert "en" in SELECTABLE_LANGUAGES
    assert len(SELECTABLE_LANGUAGES) == len(set(SELECTABLE_LANGUAGES))
    # auto is not a detectable language (it is a directive, not a result).
    assert "auto" not in DETECTABLE_LANGUAGES


@pytest.mark.parametrize(
    ("bcp47", "code", "name"),
    [
        ("en", "en", "English"),
        ("en-US", "en", "English"),
        ("zh", "zh", "Chinese"),
        ("zh-CN", "zh", "Chinese"),
        ("yue", "yue", "Cantonese"),
        ("ja", "ja", "Japanese"),
    ],
)
def test_language_mapping_roundtrip(bcp47: str, code: str, name: str) -> None:
    assert to_dashscope_code(bcp47) == code
    assert to_qwen_name(bcp47) == name


def test_unsupported_language_maps_to_none() -> None:
    # A real BCP-47 tag Qwen does not support -> None (omit, auto-detect).
    assert to_dashscope_code("sw") is None  # Swahili: not in Qwen's inventory
    assert to_qwen_name("sw") is None


@pytest.mark.parametrize(
    ("backend_value", "expected"),
    [
        ("en", "en"),
        ("zh", "zh"),
        ("English", "en"),
        ("Chinese", "zh"),
        ("Chinese,English", "zh"),  # mixed -> dominant first
        ("Cantonese", "yue"),
        (None, None),
        ("", None),
        ("Klingon", None),  # unknown -> None, never a fabricated tag
        ("  ", None),
    ],
)
def test_from_backend_language(backend_value: str | None, expected: str | None) -> None:
    assert from_backend_language(backend_value) == expected


# --------------------------------------------------------------------------- #
# _audio
# --------------------------------------------------------------------------- #
def test_float32_to_pcm16_canonical_quantization() -> None:
    # 1.0 -> 32767, -1.0 -> -32767 (round-half * 32767, not 32768).
    samples = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_pcm16(samples)
    decoded = np.frombuffer(pcm, dtype="<i2")
    assert list(decoded) == [0, 32767, -32767]


def test_float32_to_pcm16_clips_and_sanitizes() -> None:
    samples = np.array([2.0, -2.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    decoded = np.frombuffer(float32_to_pcm16(samples), dtype="<i2")
    # clipped to +-1 then quantized; NaN->0, +inf->1, -inf->-1.
    assert list(decoded) == [32767, -32767, 0, 32767, -32767]


def test_wrap_pcm16_wav_header() -> None:
    pcm = b"\x00\x00\x01\x00"
    wav = wrap_pcm16_wav(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav.endswith(pcm)
    # data chunk size is the PCM length.
    assert wav[40:44] == len(pcm).to_bytes(4, "little")


# --------------------------------------------------------------------------- #
# _json
# --------------------------------------------------------------------------- #
def test_json_helpers_happy_path() -> None:
    obj = loads_object('{"a": "x", "n": 2, "lst": [{"k": 1}], "o": {"p": 3}}')
    assert get_str(obj, "a") == "x"
    assert get_float(obj, "n") == 2.0
    assert get_list(obj, "lst") == [{"k": 1}]
    assert get_object(obj, "o") == {"p": 3}
    assert first_object(get_list(obj, "lst")) == {"k": 1}


def test_json_helpers_defensive() -> None:
    assert loads_object("not json") == {}
    assert loads_object("[1,2,3]") == {}  # top-level non-object
    assert get_str({}, "missing") is None
    assert get_str({"x": ""}, "x") is None  # empty string -> None
    assert get_float({"x": True}, "x") is None  # bool is not a duration
    assert get_float({"x": "nope"}, "x") is None
    assert get_list({"x": 5}, "x") == []
    assert get_object({"x": 5}, "x") == {}
    assert as_object(5) == {}
    assert first_object([]) == {}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults() -> None:
    cfg = Qwen3ASRConfig.from_env("qwen3-asr")
    assert cfg.backend == "vllm"
    assert cfg.default_language == "auto"
    assert cfg.stream_transport == "realtime"
    assert cfg.verify_tls is True


def test_config_base_url_trims_trailing_slash() -> None:
    cfg = Qwen3ASRConfig.from_env("qwen3-asr", base_url="http://localhost:8000/v1/")
    assert cfg.base_url == "http://localhost:8000/v1"


def test_config_base_url_none_passthrough() -> None:
    # Explicit None goes through the validator unchanged (use backend default).
    cfg = Qwen3ASRConfig.from_env("qwen3-asr", base_url=None)
    assert cfg.base_url is None


def test_config_rejects_non_http_base_url() -> None:
    # Construct directly to assert the validator's specific message: from_env wraps
    # validation failures into a ConfigError whose offending value is redacted, so
    # only direct construction surfaces the raw pydantic ValidationError.
    with pytest.raises(ValidationError, match="http"):
        Qwen3ASRConfig(base_url="ftp://example.com")


def test_config_rejects_hostless_base_url() -> None:
    with pytest.raises(ValidationError, match="host"):
        Qwen3ASRConfig(base_url="http://")


def test_config_dashscope_requires_https() -> None:
    with pytest.raises(ValidationError, match="https"):
        Qwen3ASRConfig(backend="dashscope", base_url="http://dashscope.example.com/v1")


def test_config_dashscope_https_ok() -> None:
    cfg = Qwen3ASRConfig.from_env(
        "qwen3-asr", backend="dashscope", base_url="https://dashscope.example.com/v1"
    )
    assert cfg.base_url == "https://dashscope.example.com/v1"


def test_config_api_key_is_secret() -> None:
    cfg = Qwen3ASRConfig.from_env("qwen3-asr", api_key="sk-secret")
    # SecretStr never reveals the value in str/repr.
    assert "sk-secret" not in repr(cfg)
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "sk-secret"


def test_config_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_QWEN3_ASR__API_KEY", "env-key")
    monkeypatch.setenv("STANDARD_ASR_QWEN3_ASR__BASE_URL", "http://host:9000/v1")
    cfg = Qwen3ASRConfig.from_env("qwen3-asr")
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "env-key"
    assert cfg.base_url == "http://host:9000/v1"


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #
def test_params_defaults_and_validation() -> None:
    p = Qwen3ASRParams()
    assert p.enable_itn is False
    assert p.temperature == 0.0
    assert p.emotion is False


def test_params_reject_unknown_key() -> None:
    # extra="forbid" on ProviderParams -> a typo is rejected (swap-safety).
    with pytest.raises(ValidationError):
        Qwen3ASRParams(context="oops")  # type: ignore[call-arg]


def test_params_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        Qwen3ASRParams(temperature=5.0)


# --------------------------------------------------------------------------- #
# properties / capabilities / presets
# --------------------------------------------------------------------------- #
def test_preset_model_ids_match_entrypoint_keys() -> None:
    assert Qwen3ASR.properties.model_id == "qwen3-asr/flash"
    assert Qwen3ASR17B.properties.model_id == "qwen3-asr/1.7b"
    assert Qwen3ASR06B.properties.model_id == "qwen3-asr/0.6b"


def test_streaming_capabilities_declared_honestly() -> None:
    caps = QWEN3_ASR_CAPABILITIES
    assert caps.supports("streaming_input")
    assert caps.supports("streaming_output")
    assert caps.supports("streaming.emits_partials")
    # The spec-named stable_until=0 case: word_stability MUST be false.
    assert not caps.supports("streaming.word_stability")
    # Append-only: no supersede.
    assert not caps.supports("streaming.re_segments")
    # No streaming timestamps / honest reconnect.
    assert not caps.supports("streaming.timestamps")


def test_batch_capabilities_prompt_supported_no_phrase_hints() -> None:
    caps = QWEN3_ASR_CAPABILITIES
    assert caps.supports("batch.guidance.prompt")
    assert caps.supports("batch.language.runtime_override")
    # phrase_hints and word_timestamps are NOT declared (fail-closed).
    assert not caps.supports("batch.guidance.phrase_hints")
    assert not caps.supports("batch.word_timestamps")


def test_properties_streaming_wire_is_pcm16_16k_mono() -> None:
    props = Qwen3ASR.properties
    assert props.wire_encodings == ["pcm_s16le"]
    assert props.required_input_sample_rate == 16000
    assert props.native_sample_rate == 16000


def test_flash_preset_carries_dashscope_size_limits() -> None:
    assert Qwen3ASR.properties.max_file_size == 10 * 1024 * 1024
    assert Qwen3ASR.properties.max_audio_duration == 180.0
    # Open-weight presets have no fixed sync cap.
    assert Qwen3ASR17B.properties.max_file_size is None
    assert Qwen3ASR17B.properties.max_audio_duration is None


def test_init_is_pure_no_network() -> None:
    # __init__ must not construct a backend / open a connection (spec IC.9).
    engine = Qwen3ASR17B()
    assert engine._backend is None  # pyright: ignore[reportPrivateUsage]


def test_default_backend_per_preset() -> None:
    assert Qwen3ASR().config.backend == "dashscope"  # flash -> cloud
    assert Qwen3ASR17B().config.backend == "vllm"  # open weights -> vllm
    assert Qwen3ASR06B().config.backend == "vllm"
