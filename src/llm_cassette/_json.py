"""Strict JSON helpers shared by matching, redaction, serialization and usage parsing."""

from __future__ import annotations

import json
from typing import NoReturn, TypeAlias, cast

JSONValue: TypeAlias = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
"""Any value that :func:`json.loads` can produce."""

JSONDocument: TypeAlias = dict[str, JSONValue] | list[JSONValue]
"""A top-level JSON object or array."""

_DOCUMENT_START = frozenset(b"{[")
_WHITESPACE = b" \t\r\n"


def _reject_constant(name: str) -> NoReturn:
    msg = f"non-standard JSON constant {name!r}"
    raise ValueError(msg)


def parse_document(data: bytes) -> JSONDocument | None:
    """Parse ``data`` as a JSON object or array.

    Returns ``None`` for anything else: empty bodies, invalid UTF-8, invalid JSON, JSON scalars
    and the non-standard ``NaN``/``Infinity`` constants that :mod:`json` accepts by default.
    """
    stripped = data.lstrip(_WHITESPACE)
    if not stripped or stripped[0] not in _DOCUMENT_START:
        return None
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if isinstance(value, dict | list):
        return cast("JSONDocument", value)
    return None  # pragma: no cover - guarded by the first-byte check above


def canonical(value: JSONValue) -> str:
    """Serialize ``value`` compactly with sorted keys, so equal documents give equal strings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty(value: JSONValue) -> str:
    """Serialize ``value`` with sorted keys and indentation, for human-readable diffs."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)


def sort_keys(value: JSONValue) -> JSONValue:
    """Return a copy of ``value`` whose objects have their keys in sorted order."""
    if isinstance(value, dict):
        return {key: sort_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_keys(item) for item in value]
    return value
