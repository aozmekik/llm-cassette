from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_cassette._sse import ServerSentEvent, parse_sse


def test_events_and_types() -> None:
    text = 'event: message_start\ndata: {"a":1}\n\ndata: plain\n\n'
    assert parse_sse(text) == [
        ServerSentEvent("message_start", '{"a":1}'),
        ServerSentEvent("message", "plain"),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("data: a\ndata: b\n\n", ["a\nb"]),
        ("data:no-space\n\n", ["no-space"]),
        ("data:  two spaces\n\n", [" two spaces"]),
        ("data\n\n", [""]),
        ("data:\n\n", [""]),
        (": a comment\ndata: x\n\n", ["x"]),
        ("id: 1\nretry: 10\nunknown: z\ndata: x\n\n", ["x"]),
        ("event: ping\n\n", []),
        ("data: x\r\n\r\n", ["x"]),
        ("data: x\r\rdata: y\r\r", ["x", "y"]),
        ("﻿data: bom\n\n", ["bom"]),
        ("data: complete\n\ndata: cut off", ["complete"]),
        ("data: no blank line\n", []),
        ("", []),
    ],
    ids=[
        "multiline",
        "no-space",
        "one-space-stripped",
        "field-only",
        "empty-data",
        "comment",
        "ignored-fields",
        "no-data",
        "crlf",
        "cr",
        "bom",
        "unterminated",
        "undispatched",
        "empty",
    ],
)
def test_specification_rules(text: str, expected: list[str]) -> None:
    assert [event.data for event in parse_sse(text)] == expected


def test_event_type_resets_after_dispatch() -> None:
    events = parse_sse("event: custom\ndata: 1\n\ndata: 2\n\n")
    assert [event.event for event in events] == ["custom", "message"]


line_text = st.text(
    st.characters(exclude_characters="\r\n", exclude_categories=["Cs"]), max_size=20
)


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["", "message_delta", "ping"]),
            st.lists(line_text, min_size=1, max_size=3),
        ),
        max_size=6,
    ),
    st.sampled_from(["\n", "\r\n", "\r"]),
)
def test_line_endings_do_not_matter(events: list[tuple[str, list[str]]], newline: str) -> None:
    def encode(eol: str) -> str:
        out = []
        for name, data in events:
            if name:
                out.append(f"event: {name}{eol}")
            out.extend(f"data: {line}{eol}" for line in data)
            out.append(eol)
        return "".join(out)

    parsed = parse_sse(encode(newline))
    assert parsed == parse_sse(encode("\n"))
    assert [event.data for event in parsed] == ["\n".join(data) for _, data in events]
