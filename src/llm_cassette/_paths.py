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


def _selects_index(head: str, index: int) -> bool:
    return head == WILDCARD or (head.isascii() and head.isdigit() and int(head) == index)


def _rewrite(
    node: JSONValue, suffixes: Sequence[tuple[str, ...]], leaf: _Leaf
) -> JSONValue | _Deleted:
    """Apply every path suffix to ``node`` at once, so all of them see the original document.

    Applying paths one after another would let an earlier deletion shift the list indexes that a
    later path refers to: ``messages.0`` and ``messages.1`` must remove the first two messages.
    """
    if any(not suffix for suffix in suffixes):
        return leaf(node)
    if isinstance(node, dict):
        new_object: dict[str, JSONValue] = {}
        changed = False
        for key, child in node.items():
            rests = [tuple(rest) for head, *rest in suffixes if head in {key, WILDCARD}]
            new_child = _rewrite(child, rests, leaf) if rests else child
            changed = changed or new_child is not child
            if not isinstance(new_child, _Deleted):
                new_object[key] = new_child
        return new_object if changed else node
    if isinstance(node, list):
        new_list: list[JSONValue] = []
        changed = False
        for index, item in enumerate(node):
            rests = [tuple(rest) for head, *rest in suffixes if _selects_index(head, index)]
            new_item = _rewrite(item, rests, leaf) if rests else item
            changed = changed or new_item is not item
            if not isinstance(new_item, _Deleted):
                new_list.append(new_item)
        return new_list if changed else node
    return node


def _apply(value: JSONValue, paths: Sequence[JsonPath], leaf: _Leaf) -> JSONValue:
    if not paths:
        return value
    result = _rewrite(value, [path.segments for path in paths], leaf)
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

    return _apply(value, paths, leaf), replaced


def delete_paths(value: JSONValue, paths: Sequence[JsonPath]) -> JSONValue:
    """Remove every key or list item selected by ``paths``; the input is never mutated.

    Every path is resolved against the original document, so list indexes refer to the items
    as they were before anything was deleted.
    """

    def leaf(_: JSONValue) -> _Deleted:
        return _DELETED

    return _apply(value, paths, leaf)
