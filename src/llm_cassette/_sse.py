"""A small parser for ``text/event-stream`` bodies.

Implements the event-stream interpretation rules of the WHATWG HTML Living Standard
(section 9.2, "Server-sent events"): lines end in CRLF, LF or CR; lines starting with ``:`` are
comments; a field name is followed by an optional ``:`` and one optional space; ``data`` lines
accumulate; a blank line dispatches the event; an event without data is not dispatched; and an
unterminated event at the end of the stream is discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """One dispatched event: its type (``message`` by default) and its data."""

    event: str
    data: str


def parse_sse(text: str) -> list[ServerSentEvent]:
    """Parse a complete event-stream ``text`` into the events a browser would dispatch."""
    text = text.removeprefix("﻿")
    lines = _LINE_BREAK.split(text)
    # The final element is either "" (the text ended with a line break) or an unterminated line,
    # which the specification discards.
    lines.pop()
    events: list[ServerSentEvent] = []
    event_type = ""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                events.append(ServerSentEvent(event_type or "message", "\n".join(data)))
            event_type, data = "", []
            continue
        if line.startswith(":"):
            continue
        name, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if name == "event":
            event_type = value
        elif name == "data":
            data.append(value)
        # "id" and "retry" do not affect the dispatched data; unknown fields are ignored.
    return events
