"""Strict JSON helpers shared by matching, redaction, serialization and usage parsing."""

from __future__ import annotations

import json
import math
from typing import NoReturn, TypeAlias, cast

JSONValue: TypeAlias = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
"""Any value that :func:`json.loads` can produce."""

JSONDocument: TypeAlias = dict[str, JSONValue] | list[JSONValue]
"""A top-level JSON object or array."""

_DOCUMENT_START = frozenset(b"{[")
_WHITESPACE = b" \t\r\n"
# Deeper documents are treated as opaque bytes: the recursive helpers below (and json.dumps)
# would otherwise hit the interpreter's recursion limit. Real API payloads are far shallower.
MAX_DEPTH = 200


def _reject_constant(name: str) -> NoReturn:
    msg = f"non-standard JSON constant {name!r}"
    raise ValueError(msg)


def _finite_float(text: str) -> float:
    # A literal such as 1e400 overflows to infinity, which no cassette format can store.
    value = float(text)
    if not math.isfinite(value):
        msg = f"JSON number {text} does not fit in a float"
        raise ValueError(msg)
    return value


def _loads(data: bytes) -> object:
    return json.loads(
        data.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float
    )


def _depth(value: JSONValue) -> int:
    deepest, stack = 0, [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict | list):
            deepest = max(deepest, depth)
            children = node.values() if isinstance(node, dict) else node
            stack.extend((child, depth + 1) for child in children)
    return deepest


def parse_document(data: bytes) -> JSONDocument | None:
    """Parse ``data`` as a JSON object or array.

    Returns ``None`` for anything else: empty bodies, invalid UTF-8, invalid JSON, JSON scalars,
    the non-standard ``NaN``/``Infinity`` constants that :mod:`json` accepts by default, numbers
    too large for a float (``1e400``), and documents nested more than :data:`MAX_DEPTH` levels
    deep.
    """
    stripped = data.lstrip(_WHITESPACE)
    if not stripped or stripped[0] not in _DOCUMENT_START:
        return None
    try:
        value = _loads(data)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(value, dict | list):  # pragma: no cover - guarded by the first-byte check
        return None
    document = cast("JSONDocument", value)
    # Counting brackets (in C) bounds the depth from above, so the walk is rarely needed.
    if stripped.count(b"[") + stripped.count(b"{") <= MAX_DEPTH:
        return document
    return document if _depth(document) <= MAX_DEPTH else None


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
