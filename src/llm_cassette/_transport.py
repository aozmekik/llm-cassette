"""Library-agnostic plumbing shared by the httpx and httpx2 adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from llm_cassette._model import (
    Interaction,
    RecordedRequest,
    header_bytes,
    header_text,
    normalize_headers,
)

# Headers that describe the wire encoding of the body. The adapters hand the client (and the
# cassette) the decoded body, so these would be wrong once the body has been decoded.
_WIRE_ENCODING_HEADERS = frozenset({"content-encoding", "content-length"})
_IDENTITY_ENCODINGS = frozenset({"", "identity"})
_PASSTHROUGH_EXTENSIONS = ("http_version", "reason_phrase")
# Request headers that describe the connection or the exact body bytes rather than the request.
# Stored JSON bodies are canonicalized, so a recorded content-length would be wrong anyway.
_CONNECTION_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-connection",
        "te",
        "transfer-encoding",
        "upgrade",
    }
)


def text_headers(raw: Iterable[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    """Convert raw header pairs to the :data:`~llm_cassette.Headers` convention, losslessly.

    Taking the raw bytes matters: httpx decodes header values with a guessed codec, and
    re-encoding the decoded text as ASCII fails for a UTF-8 value such as a filename.
    """
    return [(header_text(name), header_text(value)) for name, value in raw]


def wire_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    """Convert header pairs under the :data:`~llm_cassette.Headers` convention to wire bytes."""
    return [(header_bytes(name), header_bytes(value)) for name, value in headers]


def snapshot_request(
    method: str, url: str, headers: Iterable[tuple[str, str]], body: bytes
) -> RecordedRequest:
    """Build a :class:`RecordedRequest` from the parts of a library-specific request.

    Connection-level headers (``host``, ``content-length``, ``connection``...) are left out.
    """
    kept = ((name, value) for name, value in headers if name.lower() not in _CONNECTION_HEADERS)
    return RecordedRequest(
        method=method.upper(), url=url, headers=normalize_headers(kept), body=body
    )


def decoded_body_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return response headers suitable for a body that has already been decoded.

    When the upstream used a ``content-encoding`` such as gzip, the adapters read the body through
    the library's decoder, so ``content-encoding`` and ``content-length`` are dropped.
    """
    items = [(name.lower(), value) for name, value in headers]
    encoded = any(
        name == "content-encoding" and value.strip().lower() not in _IDENTITY_ENCODINGS
        for name, value in items
    )
    if not encoded:
        return items
    return [(name, value) for name, value in items if name not in _WIRE_ENCODING_HEADERS]


def passthrough_extensions(extensions: Mapping[str, object]) -> dict[str, object]:
    """Keep only the response extensions that describe the response, not the connection."""
    return {key: extensions[key] for key in _PASSTHROUGH_EXTENSIONS if key in extensions}


class Stopwatch:
    """Measures the milliseconds between consecutive events with an injectable clock."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Start timing now, according to ``clock`` (seconds, monotonic)."""
        self._clock = clock
        self._last = clock()

    def lap_ms(self) -> float:
        """Milliseconds since the previous lap (or since the stopwatch started)."""
        now = self._clock()
        elapsed = max(0.0, now - self._last) * 1000.0
        self._last = now
        return elapsed


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """What to send back for a recorded interaction, with delays already scaled to seconds.

    Attributes:
        status: The HTTP status code.
        headers: The response headers as wire bytes.
        headers_delay: Seconds to wait before returning the response.
        chunks: Body chunks, each with the seconds to wait before it.
        truncated: ``None`` for a complete recording; otherwise the message of the read error
            raised when the client reads past the recorded chunks.
    """

    status: int
    headers: list[tuple[bytes, bytes]]
    headers_delay: float
    chunks: tuple[tuple[bytes, float], ...]
    truncated: str | None

    @classmethod
    def build(
        cls, interaction: Interaction, timing_scale: float, *, method: str, url: str
    ) -> ReplayPlan:
        """Plan the replay of ``interaction``, multiplying recorded delays by ``timing_scale``."""
        response = interaction.response
        factor = timing_scale / 1000.0
        truncated = None
        if not response.complete:
            truncated = (
                f"llm-cassette: the recorded response to {method} {url} ends after "
                f"{len(response.chunks)} chunk(s) ({len(response.body)} bytes) because the "
                "client stopped reading it while it was recorded (an early close, Ctrl-C or a "
                "cancellation), so the rest of the body was never recorded; re-record the "
                "cassette (mode 'record' or --llm-record-mode=record)"
            )
        return cls(
            status=response.status,
            headers=wire_headers(response.headers),
            headers_delay=response.delay_ms * factor,
            chunks=tuple((chunk.data, chunk.delay_ms * factor) for chunk in response.chunks),
            truncated=truncated,
        )
