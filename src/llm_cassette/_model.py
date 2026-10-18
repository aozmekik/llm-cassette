"""Immutable, HTTP-library-agnostic records of requests, responses and interactions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from llm_cassette._json import JSONDocument, parse_document

Headers: TypeAlias = tuple[tuple[str, str], ...]
"""Header pairs with lower-case names, sorted by name; repeated headers keep their order.

Values hold the raw header bytes decoded as ISO-8859-1 (latin-1), the convention WSGI uses
(PEP 3333): every byte maps to one character, so any value round-trips exactly. For the usual
ASCII values this is simply the text of the header.
"""


def header_bytes(value: str) -> bytes:
    """The wire bytes of a header value stored under the :data:`Headers` convention.

    A value with characters beyond U+00FF cannot have come from the wire; it was typed into a
    cassette by hand, and is sent as UTF-8.
    """
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        return value.encode("utf-8")


def header_text(raw: bytes) -> str:
    """Store the wire bytes of a header value under the :data:`Headers` convention."""
    return raw.decode("latin-1")


def normalize_headers(items: Iterable[tuple[str, str]]) -> Headers:
    """Lower-case header names and sort the pairs by name.

    The sort is stable, so repeated headers such as ``set-cookie`` keep their relative order.
    """
    return tuple(sorted(((name.lower(), value) for name, value in items), key=lambda h: h[0]))


def _first_header(headers: Headers, name: str) -> str | None:
    wanted = name.lower()
    return next((value for key, value in headers if key == wanted), None)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """An HTTP request as the cassette sees it: after redaction, independent of the HTTP library.

    Attributes:
        method: Upper-case HTTP method.
        url: Absolute URL, including the query string.
        headers: Header pairs, see :data:`Headers`.
        body: Raw request body. JSON bodies loaded from a cassette are in canonical form
            (sorted keys, compact separators), not the exact bytes that were sent.
    """

    method: str
    url: str
    headers: Headers = ()
    body: bytes = b""

    def header(self, name: str) -> str | None:
        """Return the first value of header ``name`` (case-insensitive), or ``None``."""
        return _first_header(self.headers, name)

    def json(self) -> JSONDocument | None:
        """Return the body parsed as a JSON object or array, or ``None`` if it is not one."""
        return parse_document(self.body)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One piece of a response body, exactly as the client received it.

    Attributes:
        data: The decoded bytes of this chunk (``content-encoding`` already removed).
        delay_ms: Milliseconds between the previous chunk (or the response headers) and this one.
    """

    data: bytes
    delay_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RecordedResponse:
    """An HTTP response stored chunk by chunk.

    Attributes:
        status: HTTP status code.
        headers: Header pairs, see :data:`Headers`. ``content-encoding`` and ``content-length``
            are dropped when the body was decoded during recording.
        chunks: The body, split exactly where the upstream split it.
        delay_ms: Milliseconds between sending the request and receiving the response headers.
        complete: Whether the client read the body to its end while it was recorded. It is
            ``False`` when the client closed the response early (``openai``'s sync client stops
            at ``data: [DONE]``) or the recording was interrupted (Ctrl-C, cancellation). A
            replayed incomplete response serves the recorded chunks and then fails with a read
            error if the client reads further, because the rest of the body was never recorded.
    """

    status: int
    headers: Headers = ()
    chunks: tuple[Chunk, ...] = ()
    delay_ms: float = 0.0
    complete: bool = True

    @property
    def body(self) -> bytes:
        """The full response body: all chunks joined."""
        return b"".join(chunk.data for chunk in self.chunks)

    def header(self, name: str) -> str | None:
        """Return the first value of header ``name`` (case-insensitive), or ``None``."""
        return _first_header(self.headers, name)


@dataclass(frozen=True, slots=True)
class Interaction:
    """A recorded request together with the response it received."""

    request: RecordedRequest
    response: RecordedResponse
