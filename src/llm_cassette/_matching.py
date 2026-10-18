"""Deciding whether a live request matches a recorded one, and explaining why it does not."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TypeAlias
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_cassette._json import canonical, parse_document, pretty
from llm_cassette._model import Interaction, RecordedRequest
from llm_cassette._paths import JsonPath, delete_paths, parse_paths

Matcher: TypeAlias = Callable[[RecordedRequest, RecordedRequest], bool]
"""A custom matcher: called as ``matcher(live, recorded)`` and returns whether they match.

Both requests have already been redacted, so a matcher never sees a secret. A matcher is only
called for recorded requests that pass the built-in comparisons, but it should still expect any
body: :meth:`RecordedRequest.json` returns ``None`` for a body that is not a JSON document.
"""

_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_DIFF_LINES = 40
_MAX_SIMILARITY_CHARS = 20_000


def _names(values: Sequence[str], *, argument: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{argument} must be a sequence of names, not a single string: use [{values!r}]"
        raise TypeError(msg)
    return tuple(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class Matching:
    """How live requests are compared with recorded ones.

    By default a request matches when its method, URL and body are equal. URLs are compared
    after normalization (lower-case scheme and host, default port dropped, query parameters
    sorted). Bodies that are JSON objects or arrays are compared structurally, so key order and
    whitespace never matter; other bodies are compared byte for byte.

    Attributes:
        method: Compare HTTP methods.
        url: Compare normalized URLs.
        body: Compare bodies.
        headers: Names of request headers whose values must also be equal.
        ignore_paths: Dotted JSON paths removed from both bodies before comparing, e.g.
            ``"metadata.user_id"``, ``"stream_options"`` or ``"messages.*.cache_control"``.
        ignore_query: Query parameter names removed from both URLs before comparing.
        custom: Extra :data:`Matcher` callables; all of them must return ``True``. They are
            only called for recorded requests that already pass the built-in comparisons.
    """

    method: bool = True
    url: bool = True
    body: bool = True
    headers: Sequence[str] = ()
    ignore_paths: Sequence[str] = ()
    ignore_query: Sequence[str] = ()
    custom: Sequence[Matcher] = ()

    def __post_init__(self) -> None:
        """Validate the configuration and freeze every sequence into a tuple."""
        headers = tuple(name.lower() for name in _names(self.headers, argument="headers"))
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "ignore_paths", _names(self.ignore_paths, argument="ignore_paths"))
        object.__setattr__(self, "ignore_query", _names(self.ignore_query, argument="ignore_query"))
        object.__setattr__(self, "custom", tuple(self.custom))
        _compiled(tuple(self.ignore_paths))  # validate early: raises ValueError on a bad path

    def matches(self, live: RecordedRequest, recorded: RecordedRequest) -> bool:
        """Return whether ``live`` matches ``recorded`` under this configuration."""
        return not mismatches(
            self, live, request_key(self, live), recorded, request_key(self, recorded)
        )


@cache
def _compiled(paths: tuple[str, ...]) -> tuple[JsonPath, ...]:
    return parse_paths(paths, argument="ignore_paths")


def _ignored(matching: Matching) -> tuple[JsonPath, ...]:
    return _compiled(tuple(matching.ignore_paths))


@dataclass(frozen=True, slots=True)
class RequestKey:
    """The normalized parts of a request that built-in matching compares."""

    method: str
    url: str
    body: str | bytes
    headers: tuple[tuple[str, tuple[str, ...]], ...]


def normalize_url(url: str, ignore_query: Sequence[str] = ()) -> str:
    """Normalize ``url`` for comparison.

    Lower-cases the scheme and host, drops user info, the fragment and the default port, uses
    ``/`` for an empty path and sorts the query parameters, dropping those in ``ignore_query``.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        return url
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"
    ignored = set(ignore_query)
    query = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in ignored
    )
    return urlunsplit((scheme, netloc, parts.path or "/", urlencode(query), ""))


def _normalized_body(body: bytes, paths: Sequence[JsonPath]) -> str | bytes:
    document = parse_document(body)
    if document is None:
        return body
    return canonical(delete_paths(document, paths))


def request_key(matching: Matching, request: RecordedRequest) -> RequestKey:
    """Compute the comparison key of ``request`` under ``matching``."""
    return RequestKey(
        method=request.method.upper(),
        url=normalize_url(request.url, matching.ignore_query),
        body=_normalized_body(request.body, _ignored(matching)),
        headers=tuple(
            (name, tuple(value for key, value in request.headers if key == name))
            for name in matching.headers
        ),
    )


def index_key(matching: Matching, key: RequestKey) -> tuple[object, ...]:
    """The parts of ``key`` that built-in matching compares, usable as a dictionary key.

    Two requests pass every built-in comparison exactly when their index keys are equal.
    """
    return (
        key.method if matching.method else None,
        key.url if matching.url else None,
        key.body if matching.body else None,
        key.headers,
    )


def _matcher_name(matcher: Matcher) -> str:
    return getattr(matcher, "__qualname__", None) or repr(matcher)


def builtin_mismatches(
    matching: Matching, live_key: RequestKey, recorded_key: RequestKey
) -> list[str]:
    """List the built-in aspects (method, URL, body, headers) in which two keys differ."""
    reasons: list[str] = []
    if matching.method and live_key.method != recorded_key.method:
        reasons.append("method")
    if matching.url and live_key.url != recorded_key.url:
        reasons.append("url")
    if matching.body and live_key.body != recorded_key.body:
        reasons.append("body")
    for (name, live_values), (_, recorded_values) in zip(
        live_key.headers, recorded_key.headers, strict=True
    ):
        if live_values != recorded_values:
            reasons.append(f"header {name}")
    return reasons


def custom_mismatches(
    matching: Matching, live: RecordedRequest, recorded: RecordedRequest
) -> list[str]:
    """Name the first custom matcher that rejects a pair that passed the built-in comparisons.

    Matchers run in order and stop at the first rejection, so a later matcher can rely on the
    earlier ones. Exceptions raised by a matcher propagate: they are bugs, not mismatches.
    """
    rejecting = next((m for m in matching.custom if not m(live, recorded)), None)
    return [] if rejecting is None else [f"custom matcher {_matcher_name(rejecting)}"]


def mismatches(
    matching: Matching,
    live: RecordedRequest,
    live_key: RequestKey,
    recorded: RecordedRequest,
    recorded_key: RequestKey,
) -> list[str]:
    """List the aspects in which ``live`` differs from ``recorded``; empty means they match.

    Custom matchers only run when every built-in comparison passes, so they never see a pair
    with a different method or URL (or body, unless body matching is off).
    """
    return builtin_mismatches(matching, live_key, recorded_key) or custom_mismatches(
        matching, live, recorded
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A recorded interaction considered while matching, with the reasons it did not match."""

    index: int
    interaction: Interaction
    reasons: list[str]


def _body_lines(body: bytes, paths: Sequence[JsonPath]) -> list[str]:
    document = parse_document(body)
    if document is not None:
        return pretty(delete_paths(document, paths)).splitlines()
    if not body:
        return ["<empty body>"]
    try:
        return body.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"<{len(body)} bytes of binary data>"]


def _similarity(a: list[str], b: list[str]) -> float:
    left = "\n".join(a)[:_MAX_SIMILARITY_CHARS]
    right = "\n".join(b)[:_MAX_SIMILARITY_CHARS]
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _closest(
    matching: Matching, live: RecordedRequest, candidates: Sequence[Candidate]
) -> Candidate:
    fewest = min(len(candidate.reasons) for candidate in candidates)
    tied = [candidate for candidate in candidates if len(candidate.reasons) == fewest]
    if len(tied) == 1:
        return tied[0]
    live_lines = _body_lines(live.body, _ignored(matching))
    return max(
        tied,
        key=lambda c: _similarity(
            _body_lines(c.interaction.request.body, _ignored(matching)),
            live_lines,
        ),
    )


def _truncate(lines: list[str]) -> list[str]:
    if len(lines) <= _MAX_DIFF_LINES:
        return lines
    hidden = len(lines) - _MAX_DIFF_LINES
    return [*lines[:_MAX_DIFF_LINES], f"... ({hidden} more diff lines)"]


def _explain(matching: Matching, live: RecordedRequest, closest: Candidate) -> list[str]:
    recorded = closest.interaction.request
    lines = [
        f"closest recorded interaction: #{closest.index}, differs in: " + ", ".join(closest.reasons)
    ]
    if "method" in closest.reasons:
        lines.append(f"  method: recorded {recorded.method}, live {live.method}")
    if "url" in closest.reasons:
        lines.append(f"  url: - {normalize_url(recorded.url, matching.ignore_query)}")
        lines.append(f"       + {normalize_url(live.url, matching.ignore_query)}")
    lines.extend(
        f"  header {name}: recorded {recorded.header(name)!r}, live {live.header(name)!r}"
        for name in matching.headers
        if f"header {name}" in closest.reasons
    )
    if "body" in closest.reasons:
        diff = difflib.unified_diff(
            _body_lines(recorded.body, _ignored(matching)),
            _body_lines(live.body, _ignored(matching)),
            fromfile=f"recorded #{closest.index} body",
            tofile="live request body",
            lineterm="",
            n=2,
        )
        lines.extend(_truncate(list(diff)))
    return lines


def describe_unmatched(
    matching: Matching,
    live: RecordedRequest,
    candidates: Sequence[Candidate],
    *,
    cassette: str,
    exhausted: int,
) -> str:
    """Build the message of an :class:`~llm_cassette.UnmatchedRequestError`.

    Args:
        matching: The configuration that rejected the request.
        live: The (redacted) live request.
        candidates: Every recorded interaction with the reasons it did not match.
        cassette: A short description of the cassette (path and mode).
        exhausted: How many interactions matched but had already been played.
    """
    lines = [f"no recorded interaction matches {live.method} {live.url}", f"cassette: {cassette}"]
    if exhausted:
        lines.append(
            f"{exhausted} matching interaction(s) were already played; "
            "pass allow_repeats=True to replay them again, or re-record the cassette"
        )
    elif not candidates:
        lines.append("the cassette has no recorded interactions")
    else:
        lines.extend(_explain(matching, live, _closest(matching, live, candidates)))
        lines.append(
            "hint: re-record the cassette (mode 'record' or --llm-record-mode=record), or ignore "
            "volatile fields with Matching(ignore_paths=[...])"
        )
    return "\n".join(lines)
