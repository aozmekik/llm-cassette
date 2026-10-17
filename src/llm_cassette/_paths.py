"""Dotted JSON paths such as ``metadata.user_id`` or ``messages.*.content``.

A path is a sequence of segments separated by ``.``. A segment selects an object key; a
non-negative integer segment selects a list index; ``*`` selects every key of an object or every
item of a list. Paths that do not exist in a document are ignored, so one configuration can be
applied to many request shapes.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_cassette._json import JSONValue

WILDCARD = "*"


class _Deleted(enum.Enum):
    TOKEN = enum.auto()


_DELETED = _Deleted.TOKEN

_Leaf = Callable[[JSONValue], "JSONValue | _Deleted"]


@dataclass(frozen=True, slots=True)
class JsonPath:
    """A parsed dotted path."""

    text: str
    segments: tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> JsonPath:
        """Parse ``text``; raises :class:`ValueError` if it is empty or has an empty segment."""
        segments = tuple(text.split("."))
        if any(not segment for segment in segments):
            msg = f"invalid JSON path {text!r}: segments must be non-empty"
            raise ValueError(msg)
        return cls(text, segments)


def parse_paths(paths: Sequence[str], *, argument: str) -> tuple[JsonPath, ...]:
    """Parse a user-supplied sequence of paths, rejecting a bare string by mistake."""
    if isinstance(paths, str):
        msg = f"{argument} must be a sequence of paths, not a single string: use [{paths!r}]"
        raise TypeError(msg)
    return tuple(JsonPath.parse(path) for path in paths)


def _list_indexes(head: str, length: int) -> range | tuple[int, ...]:
    if head == WILDCARD:
        return range(length)
    if head.isascii() and head.isdigit() and int(head) < length:
        return (int(head),)
    return ()


def _rewrite(node: JSONValue, segments: tuple[str, ...], leaf: _Leaf) -> JSONValue | _Deleted:
    if not segments:
        return leaf(node)
    head, rest = segments[0], segments[1:]
    if isinstance(node, dict):
        keys = list(node) if head == WILDCARD else [head] if head in node else []
        if not keys:
            return node
        new_object = dict(node)
        for key in keys:
            new_value = _rewrite(node[key], rest, leaf)
            if isinstance(new_value, _Deleted):
                del new_object[key]
            else:
                new_object[key] = new_value
        return new_object
    if isinstance(node, list):
        indexes = _list_indexes(head, len(node))
        if not indexes:
            return node
        new_list: list[JSONValue] = []
        for index, item in enumerate(node):
            new_item = _rewrite(item, rest, leaf) if index in indexes else item
            if not isinstance(new_item, _Deleted):
                new_list.append(new_item)
        return new_list
    return node


def _apply(value: JSONValue, path: JsonPath, leaf: _Leaf) -> JSONValue:
    result = _rewrite(value, path.segments, leaf)
    # Paths always have at least one segment, so the root itself is never the leaf.
    assert not isinstance(result, _Deleted)  # noqa: S101
    return result


def replace_paths(
    value: JSONValue, paths: Sequence[JsonPath], replacement: JSONValue
) -> tuple[JSONValue, bool]:
    """Replace every value selected by ``paths`` with ``replacement``.

    Returns the new document (the input is never mutated) and whether anything was replaced.
    """
    replaced = False

    def leaf(_: JSONValue) -> JSONValue:
        nonlocal replaced
        replaced = True
        return replacement

    for path in paths:
        value = _apply(value, path, leaf)
    return value, replaced


def delete_paths(value: JSONValue, paths: Sequence[JsonPath]) -> JSONValue:
    """Remove every key or list item selected by ``paths``; the input is never mutated."""

    def leaf(_: JSONValue) -> _Deleted:
        return _DELETED

    for path in paths:
        value = _apply(value, path, leaf)
    return value
