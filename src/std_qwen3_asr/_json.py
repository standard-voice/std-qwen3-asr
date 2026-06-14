# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed helpers for reading untyped JSON from backend responses.

``json.loads`` returns ``Any``, which under pyright strict spreads "partially
unknown" types through every downstream access. These helpers normalize a decoded
payload to ``dict[str, object]`` and pull typed leaves out of it, so the backend
parsers stay strict-clean and explicit about what shape they expect.
"""

from __future__ import annotations

import json
from typing import cast


def loads_object(text: str) -> dict[str, object]:
    """Parse a JSON string expected to be an object.

    Args:
        text: The JSON text.

    Returns:
        The decoded object, or an empty dict if the text is not a JSON object.
    """
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(decoded, dict):
        return cast("dict[str, object]", decoded)
    return {}


def get_str(obj: dict[str, object], key: str) -> str | None:
    """Return ``obj[key]`` coerced to ``str`` when present and string-like.

    Args:
        obj: The mapping.
        key: The key to read.

    Returns:
        The string value, or ``None`` if absent/empty/non-string.
    """
    value = obj.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def get_float(obj: dict[str, object], key: str) -> float | None:
    """Return ``obj[key]`` coerced to ``float`` when numeric.

    Args:
        obj: The mapping.
        key: The key to read.

    Returns:
        The float value, or ``None`` if absent/non-numeric.
    """
    value = obj.get(key)
    if isinstance(value, bool):  # bool is an int subclass; not a duration
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def get_list(obj: dict[str, object], key: str) -> list[object]:
    """Return ``obj[key]`` as a list when present and list-like.

    Args:
        obj: The mapping.
        key: The key to read.

    Returns:
        The list, or an empty list.
    """
    value = obj.get(key)
    if isinstance(value, list):
        return cast("list[object]", value)
    return []


def get_object(obj: dict[str, object], key: str) -> dict[str, object]:
    """Return ``obj[key]`` as an object when present and dict-like.

    Args:
        obj: The mapping.
        key: The key to read.

    Returns:
        The object, or an empty dict.
    """
    return as_object(obj.get(key))


def as_object(value: object) -> dict[str, object]:
    """Coerce an arbitrary value to an object mapping.

    Args:
        value: Any value (e.g. a list element).

    Returns:
        The value as a dict, or an empty dict if it is not a mapping.
    """
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return {}


def first_object(items: list[object]) -> dict[str, object]:
    """Return the first list element coerced to an object mapping.

    Args:
        items: A list of arbitrary values.

    Returns:
        The first element as a dict, or an empty dict if absent/non-mapping.
    """
    if not items:
        return {}
    return as_object(items[0])
