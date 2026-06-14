# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR language inventory and BCP-47 <-> Qwen mapping.

Qwen3-ASR exposes two different language-naming surfaces depending on the
backend:

* The **DashScope cloud** API (``asr_options.language``) takes ISO-639 codes
  (``"zh"``, ``"en"``, ``"yue"`` ...).
* The **open-weight / vLLM** code path validates against English *name* strings
  (``"Chinese"``, ``"English"``, ``"Cantonese"`` ...) -- there is no ISO mapping
  in the upstream code.

Standard ASR speaks BCP-47 (``selectable_languages`` / ``language`` /
``detected_language`` are BCP-47, spec G.1.3). This module is the single source
of truth that translates a resolved BCP-47 tag into whichever shape the active
backend needs, and translates a backend-reported language back into BCP-47 for
``detected_language``. Keeping it in one table avoids the "same value means
different things on different backends" trap (spec Runtime §6).
"""

from __future__ import annotations

from standard_asr.language import normalize_bcp47

#: BCP-47 primary subtag -> (DashScope ISO code, open-weight English name).
#:
#: Covers the 30 language varieties Qwen3-ASR advertises. ``yue`` (Cantonese)
#: is a distinct BCP-47 subtag from ``zh`` (Mandarin/standard Chinese) and Qwen
#: treats them separately, so both are listed. We key on the *primary* subtag
#: (``normalize_bcp47(tag).split("-")[0]``) so region-qualified tags such as
#: ``en-US`` / ``zh-CN`` resolve to the same engine language.
_LANGUAGE_TABLE: dict[str, tuple[str, str]] = {
    "zh": ("zh", "Chinese"),
    "yue": ("yue", "Cantonese"),
    "en": ("en", "English"),
    "ar": ("ar", "Arabic"),
    "de": ("de", "German"),
    "fr": ("fr", "French"),
    "es": ("es", "Spanish"),
    "pt": ("pt", "Portuguese"),
    "id": ("id", "Indonesian"),
    "it": ("it", "Italian"),
    "ko": ("ko", "Korean"),
    "ru": ("ru", "Russian"),
    "th": ("th", "Thai"),
    "vi": ("vi", "Vietnamese"),
    "ja": ("ja", "Japanese"),
    "tr": ("tr", "Turkish"),
    "hi": ("hi", "Hindi"),
    "ms": ("ms", "Malay"),
    "nl": ("nl", "Dutch"),
    "sv": ("sv", "Swedish"),
    "da": ("da", "Danish"),
    "fi": ("fi", "Finnish"),
    "pl": ("pl", "Polish"),
    "cs": ("cs", "Czech"),
    "fil": ("fil", "Filipino"),
    "fa": ("fa", "Persian"),
    "el": ("el", "Greek"),
    "ro": ("ro", "Romanian"),
    "hu": ("hu", "Hungarian"),
    "mk": ("mk", "Macedonian"),
}

#: Reverse map: English name (lower-cased) -> BCP-47 primary subtag. Built once.
_NAME_TO_BCP47: dict[str, str] = {
    name.lower(): bcp47 for bcp47, (_code, name) in _LANGUAGE_TABLE.items()
}
#: Reverse map: ISO code (lower-cased) -> BCP-47 primary subtag. The ISO codes
#: are nearly identical to the BCP-47 subtags, but keeping an explicit reverse
#: map keeps the mapping honest (e.g. should a code ever diverge).
_CODE_TO_BCP47: dict[str, str] = {
    code.lower(): bcp47 for bcp47, (code, _name) in _LANGUAGE_TABLE.items()
}

#: BCP-47 tags Standard ASR advertises as selectable for this engine, plus the
#: reserved ``"auto"`` directive (multilingual auto-detection = ``language=None``
#: upstream). This is what goes into ``Properties.selectable_languages``.
SELECTABLE_LANGUAGES: list[str] = ["auto", *_LANGUAGE_TABLE.keys()]

#: BCP-47 tags the engine can *detect* (everything it supports; Qwen reports the
#: detected language in its response annotations).
DETECTABLE_LANGUAGES: list[str] = list(_LANGUAGE_TABLE.keys())


def _primary_subtag(bcp47_tag: str) -> str:
    """Return the lower-cased primary subtag of a BCP-47 tag.

    Args:
        bcp47_tag: A BCP-47 language tag (e.g. ``"en-US"``).

    Returns:
        The lower-cased primary subtag (e.g. ``"en"``).
    """
    return normalize_bcp47(bcp47_tag).split("-", maxsplit=1)[0].lower()


def to_dashscope_code(bcp47_tag: str) -> str | None:
    """Translate a BCP-47 tag to the DashScope ``asr_options.language`` code.

    Args:
        bcp47_tag: A resolved BCP-47 tag (never ``"auto"`` -- callers resolve
            auto-detection to ``language=None`` before reaching the wire).

    Returns:
        The DashScope ISO code, or ``None`` if the tag is not in the supported
        inventory (the caller should then omit the language field and let the
        engine auto-detect rather than send an invalid code).
    """
    entry = _LANGUAGE_TABLE.get(_primary_subtag(bcp47_tag))
    return entry[0] if entry is not None else None


def to_qwen_name(bcp47_tag: str) -> str | None:
    """Translate a BCP-47 tag to the open-weight/vLLM English language name.

    Args:
        bcp47_tag: A resolved BCP-47 tag.

    Returns:
        The English language name (e.g. ``"Chinese"``), or ``None`` if the tag
        is unsupported.
    """
    entry = _LANGUAGE_TABLE.get(_primary_subtag(bcp47_tag))
    return entry[1] if entry is not None else None


def from_backend_language(value: str | None) -> str | None:
    """Translate a backend-reported language into a BCP-47 tag.

    Qwen reports the detected language either as an ISO code (DashScope
    annotations: ``"zh"``) or, in mixed-language audio, as a comma-joined name
    list (open-weight: ``"Chinese,English"``). We take the FIRST reported
    language as the dominant one and map it to BCP-47, since ``detected_language``
    is a single tag (spec TR.1). An unrecognized value yields ``None`` rather than
    a fabricated tag (``validate_detected_language`` rejects ``"auto"`` and
    non-BCP-47 strings, so honesty beats guessing).

    Args:
        value: The language string from the backend response, or ``None``.

    Returns:
        A BCP-47 primary subtag, or ``None`` if it cannot be mapped.
    """
    if not value:
        return None
    first = value.split(",", maxsplit=1)[0].strip()
    if not first:
        return None
    lowered = first.lower()
    # Try ISO code first (DashScope), then English name (open-weight).
    return _CODE_TO_BCP47.get(lowered) or _NAME_TO_BCP47.get(lowered)
