from __future__ import annotations

import copy
import json
import random
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_cassette._json import MAX_DEPTH, JSONValue, canonical, parse_document, sort_keys
from llm_cassette._paths import JsonPath, delete_paths, parse_paths, replace_paths

json_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=25,
)
json_documents = st.dictionaries(st.text(max_size=8), json_values, max_size=5) | st.lists(
    json_values, max_size=5
)


# ------------------------------------------------------------------------------- parse_document


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b'{"a": 1}', {"a": 1}),
        (b"  \n[1, 2]", [1, 2]),
        (b"", None),
        (b"   ", None),
        (b"42", None),
        (b'"text"', None),
        (b"{not json", None),
        (b'{"a": NaN}', None),
        (b'{"a": Infinity}', None),
        (b"\xff\xfe{}", None),
        (b"[" * 100_000 + b"]" * 100_000, None),
    ],
    ids=[
        "object",
        "array",
        "empty",
        "blank",
        "scalar",
        "string",
        "invalid",
        "nan",
        "inf",
        "utf8",
        "deep",
    ],
)
def test_parse_document(data: bytes, expected: JSONValue) -> None:
    assert parse_document(data) == expected


def test_depth_limit_is_inclusive() -> None:
    assert parse_document(b"[" * MAX_DEPTH + b"]" * MAX_DEPTH) is not None
    assert parse_document(b"[" * (MAX_DEPTH + 1) + b"]" * (MAX_DEPTH + 1)) is None
    assert parse_document(b'{"a": ' * MAX_DEPTH + b"1" + b"}" * MAX_DEPTH) is not None


@given(json_documents, st.randoms())
def test_canonical_ignores_key_order_and_whitespace(
    document: JSONValue, rnd: random.Random
) -> None:
    def shuffled(value: JSONValue) -> JSONValue:
        if isinstance(value, dict):
            items = list(value.items())
            rnd.shuffle(items)
            return {key: shuffled(item) for key, item in items}
        if isinstance(value, list):
            return [shuffled(item) for item in value]
        return value

    reencoded = json.dumps(shuffled(document), indent=rnd.choice([None, 1, 4]))
    reparsed = parse_document(reencoded.encode())
    assert reparsed is not None
    assert canonical(reparsed) == canonical(document)


def test_canonical_distinguishes_booleans_from_numbers() -> None:
    assert canonical({"stream": True}) != canonical({"stream": 1})
    assert canonical([1]) != canonical([1.0])


def test_sort_keys_is_recursive() -> None:
    assert list(sort_keys({"b": {"d": 1, "c": 2}, "a": [{"z": 0, "y": 1}]})) == ["a", "b"]  # type: ignore[arg-type]
    nested = sort_keys({"b": {"d": 1, "c": 2}, "a": [{"z": 0, "y": 1}]})
    assert json.dumps(nested) == '{"a": [{"y": 1, "z": 0}], "b": {"c": 2, "d": 1}}'


# ------------------------------------------------------------------------------------- paths


def test_parse_path() -> None:
    assert JsonPath.parse("messages.*.content").segments == ("messages", "*", "content")


@pytest.mark.parametrize("bad", ["", ".", "a..b", "a.", ".a"])
def test_parse_path_rejects_empty_segments(bad: str) -> None:
    with pytest.raises(ValueError, match="segments must be non-empty"):
        JsonPath.parse(bad)


def test_parse_paths_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        parse_paths("metadata.user_id", argument="ignore_paths")


def _paths(*texts: str) -> tuple[JsonPath, ...]:
    return tuple(JsonPath.parse(text) for text in texts)


DOCUMENT: JSONValue = {
    "model": "gpt-4o-mini",
    "metadata": {"user_id": "u-123", "trace": "t"},
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}},
    ],
}


def replaced(*paths: str) -> Any:
    result, _ = replace_paths(DOCUMENT, _paths(*paths), "X")
    return result


def test_replace_nested_key() -> None:
    assert replace_paths(DOCUMENT, _paths("metadata.user_id"), "X")[1]
    assert replaced("metadata.user_id")["metadata"] == {"user_id": "X", "trace": "t"}


def test_replace_with_wildcards_and_indexes() -> None:
    assert [m["content"] for m in replaced("messages.*.content")["messages"]] == ["X", "X"]
    assert [m["role"] for m in replaced("messages.1.role")["messages"]] == ["system", "X"]
    assert replaced("metadata.*")["metadata"] == {"user_id": "X", "trace": "X"}


@pytest.mark.parametrize(
    "path",
    ["missing", "metadata.missing", "messages.7.role", "messages.x", "model.deeper", "messages.¹"],
)
def test_missing_paths_change_nothing(path: str) -> None:
    result, changed = replace_paths(DOCUMENT, _paths(path), "X")
    assert not changed
    assert result == DOCUMENT


def test_delete_keys_and_list_items() -> None:
    result = delete_paths(DOCUMENT, _paths("metadata", "messages.*.cache_control", "messages.0"))
    assert result == {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def test_delete_every_list_item() -> None:
    assert delete_paths({"a": [1, 2, 3]}, _paths("a.*")) == {"a": []}


@given(json_documents, st.lists(st.sampled_from(["a", "*", "0", "a.b", "*.*", "1.*"]), max_size=3))
def test_rewrites_never_mutate_their_input(document: JSONValue, texts: list[str]) -> None:
    original = copy.deepcopy(document)
    delete_paths(document, _paths(*texts))
    replace_paths(document, _paths(*texts), "X")
    assert document == original
