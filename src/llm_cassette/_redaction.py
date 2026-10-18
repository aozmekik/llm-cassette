"""Removing secrets from interactions before they are stored, compared or written."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from functools import cache
from urllib.parse import (
    parse_qsl,
    quote,
    quote_plus,
    unquote_plus,
    urlencode,
    urlsplit,
    urlunsplit,
)

from llm_cassette._json import JSONValue, canonical, is_json_scalar, parse_document
from llm_cassette._model import (
    Chunk,
    Headers,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    header_bytes,
)
from llm_cassette._paths import JsonPath, parse_paths, replace_paths

DEFAULT_REDACTED_HEADERS: frozenset[str] = frozenset(
    {
        "anthropic-organization-id",
        "authorization",
        "cookie",
        "openai-organization",
        "openai-project",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
)
"""Header names that are always redacted by default (compared case-insensitively)."""

DEFAULT_SENSITIVE_SUBSTRINGS: tuple[str, ...] = ("key", "token", "secret")
"""Header and query-parameter names containing any of these are redacted by default."""

# Headers whose values are URLs (``link`` holds a list of ``<URL>`` targets). Their user info and
# sensitive query parameters are redacted like those of the request URL: a redirect to a
# presigned download link carries its credentials in the query string.
_URL_HEADERS = frozenset({"content-location", "link", "location", "referer"})
_LINK_TARGET = re.compile(r"<([^>]*)>")
# A JSON string literal, escapes included.
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
# JSON text nests JSON strings in practice (tool-call arguments are a JSON document encoded as a
# string), so string literals are decoded up to this many levels deep when looking for secrets.
_MAX_ESCAPE_LEVELS = 4


def _verbatim(text: str) -> str:
    return text


def _json_escaped(text: str) -> str:
    return json.dumps(text)[1:-1]


def _json_escaped_unicode(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _percent_encoded(text: str) -> str:
    return quote(text, safe="")


def _path_encoded(text: str) -> str:
    return quote(text)


# The forms in which a literal secret commonly reaches the wire: as is, inside a JSON string
# (with json.dumps' default \uXXXX escapes or without them), and percent-encoded in a URL path,
# a query string or a form body.
_ENCODINGS: tuple[Callable[[str], str], ...] = (
    _verbatim,
    _json_escaped,
    _json_escaped_unicode,
    _percent_encoded,
    _path_encoded,
    quote_plus,
)


@cache
def _compiled(paths: tuple[str, ...]) -> tuple[JsonPath, ...]:
    return parse_paths(paths, argument="body_paths")


def _lowered(values: Iterable[str], *, argument: str) -> frozenset[str]:
    if isinstance(values, str):
        msg = f"{argument} must be a collection of names, not a single string: use {{{values!r}}}"
        raise TypeError(msg)
    return frozenset(value.lower() for value in values)


def _strings(values: Sequence[str], *, argument: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{argument} must be a sequence of strings, not a single string: use [{values!r}]"
        raise TypeError(msg)
    return tuple(values)


def _replacements(secrets: Sequence[str], placeholder: str) -> tuple[tuple[str, str], ...]:
    """Pairs of (encoded secret, placeholder encoded the same way), longest secret form first.

    Longer forms go first, so a secret that contains another secret is replaced whole.
    """
    pairs: dict[str, str] = {}
    for secret in secrets:
        for encode in _ENCODINGS:
            pairs.setdefault(encode(secret), encode(placeholder))
    return tuple(sorted(pairs.items(), key=lambda pair: len(pair[0]), reverse=True))


def _is_json_media_type(content_type: str | None) -> bool:
    media_type = (content_type or "").partition(";")[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _adjust_content_length(headers: Headers, delta: int) -> Headers:
    def adjusted(name: str, value: str) -> str:
        number = value.strip()
        if name == "content-length" and number.isascii() and number.isdigit():
            return str(max(0, int(number) + delta))
        return value

    return tuple((name, adjusted(name, value)) for name, value in headers)


def _redact_userinfo(netloc: str, placeholder: str) -> str:
    userinfo, at, host = netloc.rpartition("@")
    if not at:
        return netloc
    username, colon, password = userinfo.partition(":")
    redacted = (placeholder if username else "") + colon + (placeholder if password else "")
    return f"{redacted}@{host}"


def _decoded_views(text: str, levels: int = _MAX_ESCAPE_LEVELS) -> Iterator[str]:
    """``text`` itself, its percent-decoding, and the decoded value of its JSON strings."""
    yield text
    if "%" in text or "+" in text:
        yield unquote_plus(text)
    if levels and "\\" in text:
        for match in _JSON_STRING.finditer(text):
            try:
                decoded = json.loads(match.group())
            except ValueError:
                continue
            yield from _decoded_views(decoded, levels - 1)


def _headers_blob(headers: Headers) -> bytes:
    return b"\n".join(name.encode() + b": " + header_bytes(value) for name, value in headers)


@dataclass(frozen=True, slots=True, kw_only=True)
class Redaction:
    """What to remove from interactions before they are stored.

    Redaction runs on every request before it is matched or recorded and on every interaction
    before the cassette exposes or writes it, so nothing unredacted reaches disk.

    Attributes:
        headers: Header names whose values are replaced (case-insensitive). Applies to request and
            response headers. Passing a set replaces the defaults; to extend them, pass
            ``DEFAULT_REDACTED_HEADERS | {"x-tenant-id"}``.
        substrings: Header and query-parameter names containing any of these substrings are
            redacted too. The default also catches rate-limit headers such as
            ``x-ratelimit-remaining-tokens``; pass a narrower tuple to keep them.
        query_params: Query-parameter names whose values are replaced (case-insensitive). Query
            parameters are redacted in the request URL and in URL-valued headers (``location``,
            ``content-location``, ``link``, ``referer``), as is any user info (``user:pass@``).
        body_paths: Dotted JSON paths (see :class:`~llm_cassette.Matching`) replaced in JSON
            request bodies and in non-streaming JSON response bodies. A response whose
            content type is JSON but whose body does not parse (for example one the client
            closed half-way) is replaced by the placeholder as a whole, because the paths cannot
            be located in it.
        secrets: Literal values (for example the API key itself) replaced anywhere in URLs,
            headers and bodies, including their JSON-escaped and percent-encoded forms. Before a
            cassette is written, :meth:`~llm_cassette.Cassette.save` also looks for them in
            decoded JSON strings, in percent-decoded text and in the serialized file; if one
            survived (for instance split across two streamed chunks), it refuses to write and
            raises :class:`~llm_cassette.SecretLeakError`.
        drop_headers: Header names, or :mod:`fnmatch` patterns such as ``"x-stainless-*"``, removed
            from requests and responses altogether (case-insensitive). Dropping volatile headers
            such as ``user-agent`` keeps re-recorded cassettes from changing with the SDK, OS or
            Python version.
        placeholder: The replacement value; printable ASCII, since it is also a header value.
    """

    headers: AbstractSet[str] = DEFAULT_REDACTED_HEADERS
    substrings: Sequence[str] = DEFAULT_SENSITIVE_SUBSTRINGS
    query_params: AbstractSet[str] = frozenset()
    body_paths: Sequence[str] = ()
    secrets: Sequence[str] = ()
    drop_headers: Sequence[str] = ()
    placeholder: str = "REDACTED"
    _text_replacements: tuple[tuple[str, str], ...] = field(init=False, repr=False, compare=False)
    _byte_replacements: tuple[tuple[bytes, bytes], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the configuration and normalize names to lower case."""
        object.__setattr__(self, "headers", _lowered(self.headers, argument="headers"))
        object.__setattr__(
            self, "query_params", _lowered(self.query_params, argument="query_params")
        )
        substrings = tuple(s.lower() for s in _strings(self.substrings, argument="substrings"))
        if any(not substring for substring in substrings):
            msg = "substrings must be non-empty (an empty substring would redact every header)"
            raise ValueError(msg)
        object.__setattr__(self, "substrings", substrings)
        object.__setattr__(self, "body_paths", _strings(self.body_paths, argument="body_paths"))
        _compiled(tuple(self.body_paths))  # validate early: raises ValueError on a bad path
        drop = tuple(p.lower() for p in _strings(self.drop_headers, argument="drop_headers"))
        object.__setattr__(self, "drop_headers", drop)
        if not (self.placeholder.isascii() and self.placeholder.isprintable()):
            msg = "the placeholder must be printable ASCII, so that it is a valid header value"
            raise ValueError(msg)
        secrets = _strings(self.secrets, argument="secrets")
        if any(not secret for secret in secrets):
            msg = "secrets must be non-empty strings"
            raise ValueError(msg)
        if any(secret in self.placeholder for secret in secrets):
            msg = "the placeholder must not contain a secret"
            raise ValueError(msg)
        object.__setattr__(self, "secrets", secrets)
        text = _replacements(secrets, self.placeholder)
        object.__setattr__(self, "_text_replacements", text)
        object.__setattr__(
            self, "_byte_replacements", tuple((a.encode(), b.encode()) for a, b in text)
        )

    def is_sensitive_header(self, name: str) -> bool:
        """Return whether the value of header ``name`` is replaced under this configuration."""
        lowered = name.lower()
        return lowered in self.headers or any(s in lowered for s in self.substrings)

    def is_sensitive_query_param(self, name: str) -> bool:
        """Return whether query parameter ``name`` is redacted under this configuration."""
        lowered = name.lower()
        return lowered in self.query_params or any(s in lowered for s in self.substrings)

    def is_dropped_header(self, name: str) -> bool:
        """Return whether header ``name`` is removed altogether under this configuration."""
        lowered = name.lower()
        return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in self.drop_headers)

    def redact_request(self, request: RecordedRequest) -> RecordedRequest:
        """Return a copy of ``request`` with every sensitive value replaced."""
        body = request.body
        document = self._redact_document(body)
        if document is not None:
            body = canonical(document).encode()
        return replace(
            request,
            url=self._redact_url(request.url),
            headers=self._redact_headers(request.headers),
            body=self._replace_secrets_bytes(body),
        )

    def redact_response(self, response: RecordedResponse) -> RecordedResponse:
        """Return a copy of ``response`` with every sensitive value replaced.

        Body paths apply only when the whole body is one JSON document, which is then stored as
        a single chunk; streamed (SSE) bodies keep their chunks, and only literal secrets are
        replaced in them. When the body changes, ``content-length`` is adjusted to match.
        """
        body = response.body
        chunks = response.chunks
        rewritten = self._redact_response_body(body, response.header("content-type"))
        if rewritten is not None:
            chunks = (Chunk(rewritten, sum(chunk.delay_ms for chunk in chunks)),)
        if self.secrets:
            chunks = tuple(
                replace(chunk, data=self._replace_secrets_bytes(chunk.data)) for chunk in chunks
            )
        headers = self._redact_headers(response.headers)
        redacted_length = sum(len(chunk.data) for chunk in chunks)
        if redacted_length != len(body):
            headers = _adjust_content_length(headers, redacted_length - len(body))
        return replace(response, headers=headers, chunks=chunks)

    def redact(self, interaction: Interaction) -> Interaction:
        """Return ``interaction`` with both its request and response redacted."""
        return Interaction(
            request=self.redact_request(interaction.request),
            response=self.redact_response(interaction.response),
        )

    def find_secret(
        self, interactions: Sequence[Interaction], *, serialized: str | None = None
    ) -> str | None:
        """Return where a configured secret appears in ``interactions``, or ``None``.

        Each field is searched as is, percent-decoded, and with the escapes of its JSON strings
        decoded; ``serialized``, the cassette text about to be written, is searched too. The
        result names a location such as ``interactions[2].response.body``, never the secret.
        """
        if not self.secrets:
            return None
        for index, interaction in enumerate(interactions):
            request, response = interaction.request, interaction.response
            fields: list[tuple[str, bytes]] = [
                ("request.url", request.url.encode()),
                ("request.headers", _headers_blob(request.headers)),
                ("request.body", request.body),
                ("response.headers", _headers_blob(response.headers)),
                ("response.body", response.body),
            ]
            for name, blob in fields:
                if self._exposes_secret(blob.decode("utf-8", errors="replace")):
                    return f"interactions[{index}].{name}"
        if serialized is not None and self._exposes_secret(serialized):
            return "the serialized cassette"
        return None

    def _exposes_secret(self, text: str) -> bool:
        return any(secret in view for view in _decoded_views(text) for secret in self.secrets)

    def _redact_document(self, body: bytes) -> JSONValue | None:
        """The JSON ``body`` with body paths and escaped secrets replaced, if that changes it."""
        paths = _compiled(tuple(self.body_paths))
        # Without a backslash, JSON strings hold no escapes, so the plain replacement of the
        # secrets' encoded forms already finds every occurrence.
        scrub = bool(self.secrets) and b"\\" in body
        if not paths and not scrub:
            return None
        document = parse_document(body)
        if document is None:
            return None
        rewritten, replaced = replace_paths(document, paths, self.placeholder)
        if scrub:
            rewritten, scrubbed = self._scrub(rewritten)
            replaced = replaced or scrubbed
        return rewritten if replaced else None

    def _redact_response_body(self, body: bytes, content_type: str | None) -> bytes | None:
        rewritten = self._redact_document(body)
        if rewritten is not None:
            return json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")).encode()
        if (
            self.body_paths
            and body.strip()
            and _is_json_media_type(content_type)
            and parse_document(body) is None
            and not is_json_scalar(body)
        ):
            # The paths cannot be located in a JSON body that does not parse, but the values
            # they name may still be in it: store none of it.
            return self.placeholder.encode()
        return None

    def _scrub(self, value: JSONValue) -> tuple[JSONValue, bool]:
        """Replace the secrets inside the strings (keys included) of a JSON document."""
        found = False

        def text(string: str) -> str:
            nonlocal found
            replaced = self._replace_secrets_text(string)
            found = found or replaced != string
            return replaced

        def walk(node: JSONValue) -> JSONValue:
            if isinstance(node, str):
                return text(node)
            if isinstance(node, list):
                return [walk(item) for item in node]
            if isinstance(node, dict):
                return {text(key): walk(item) for key, item in node.items()}
            return node

        return walk(value), found

    def _replace_secrets_text(self, text: str) -> str:
        for form, replacement in self._text_replacements:
            text = text.replace(form, replacement)
        return text

    def _replace_secrets_bytes(self, data: bytes) -> bytes:
        for form, replacement in self._byte_replacements:
            data = data.replace(form, replacement)
        return data

    def _replace_secrets_header(self, value: str) -> str:
        try:
            raw = value.encode("latin-1")
        except UnicodeEncodeError:  # typed into a cassette by hand: real text, not wire bytes
            return self._replace_secrets_text(value)
        return self._replace_secrets_bytes(raw).decode("latin-1")

    def _redact_headers(self, headers: Headers) -> Headers:
        return tuple(
            (name, self._redact_header(name, value))
            for name, value in headers
            if not self.is_dropped_header(name)
        )

    def _redact_header(self, name: str, value: str) -> str:
        if self.is_sensitive_header(name):
            return self.placeholder
        lowered = name.lower()
        if lowered == "link":
            value = _LINK_TARGET.sub(lambda m: f"<{self._redact_url(m.group(1))}>", value)
        elif lowered in _URL_HEADERS:
            value = self._redact_url(value)
        return self._replace_secrets_header(value)

    def _redact_url(self, url: str) -> str:
        try:
            parts = urlsplit(url)
        except ValueError:  # not a URL after all (a malformed location header, say)
            return self._replace_secrets_text(url)
        netloc = _redact_userinfo(parts.netloc, quote(self.placeholder, safe=""))
        query = parts.query
        pairs = parse_qsl(query, keep_blank_values=True)
        if any(self.is_sensitive_query_param(name) for name, _ in pairs):
            query = urlencode(
                [
                    (name, self.placeholder if self.is_sensitive_query_param(name) else value)
                    for name, value in pairs
                ]
            )
        rebuilt = urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
        return self._replace_secrets_text(
            rebuilt if (netloc, query) != (parts.netloc, parts.query) else url
        )
