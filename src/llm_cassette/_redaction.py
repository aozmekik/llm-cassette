"""Removing secrets from interactions before they are stored, compared or written."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from functools import cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_cassette._json import canonical, parse_document
from llm_cassette._model import Chunk, Headers, Interaction, RecordedRequest, RecordedResponse
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


@dataclass(frozen=True, slots=True)
class Redaction:
    """What to remove from interactions before they are stored.

    Redaction runs on every request before it is matched or recorded and on every interaction
    before the cassette exposes or writes it, so nothing unredacted reaches disk.

    Attributes:
        headers: Header names whose values are replaced (case-insensitive). Applies to request and
            response headers.
        substrings: Header and query-parameter names containing any of these substrings are
            redacted too. The default also catches rate-limit headers such as
            ``x-ratelimit-remaining-tokens``; pass a narrower tuple to keep them.
        query_params: Query-parameter names whose values are replaced (case-insensitive).
        body_paths: Dotted JSON paths (see :class:`~llm_cassette.Matching`) replaced in JSON
            request bodies and in non-streaming JSON response bodies.
        secrets: Literal values (for example the API key itself) replaced anywhere in URLs,
            headers and bodies. If one still appears in an interaction, for instance split
            across two streamed chunks, :meth:`~llm_cassette.Cassette.save` refuses to write and
            raises :class:`~llm_cassette.SecretLeakError`.
        placeholder: The replacement value.
    """

    headers: AbstractSet[str] = DEFAULT_REDACTED_HEADERS
    substrings: Sequence[str] = DEFAULT_SENSITIVE_SUBSTRINGS
    query_params: AbstractSet[str] = frozenset()
    body_paths: Sequence[str] = ()
    secrets: Sequence[str] = ()
    placeholder: str = "REDACTED"

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
        secrets = _strings(self.secrets, argument="secrets")
        if any(not secret for secret in secrets):
            msg = "secrets must be non-empty strings"
            raise ValueError(msg)
        if any(secret in self.placeholder for secret in secrets):
            msg = "the placeholder must not contain a secret"
            raise ValueError(msg)
        if any(char in self.placeholder for char in "\r\n\0"):
            msg = "the placeholder must be a valid header value (no CR, LF or NUL)"
            raise ValueError(msg)
        object.__setattr__(self, "secrets", secrets)

    def is_sensitive_header(self, name: str) -> bool:
        """Return whether header ``name`` is redacted under this configuration."""
        lowered = name.lower()
        return lowered in self.headers or any(s in lowered for s in self.substrings)

    def is_sensitive_query_param(self, name: str) -> bool:
        """Return whether query parameter ``name`` is redacted under this configuration."""
        lowered = name.lower()
        return lowered in self.query_params or any(s in lowered for s in self.substrings)

    def redact_request(self, request: RecordedRequest) -> RecordedRequest:
        """Return a copy of ``request`` with every sensitive value replaced."""
        body = request.body
        paths = _compiled(tuple(self.body_paths))
        if paths and (document := parse_document(body)) is not None:
            redacted, changed = replace_paths(document, paths, self.placeholder)
            if changed:
                body = canonical(redacted).encode()
        return replace(
            request,
            url=self._redact_url(request.url),
            headers=self._redact_headers(request.headers),
            body=self._replace_secrets_bytes(body),
        )

    def redact_response(self, response: RecordedResponse) -> RecordedResponse:
        """Return a copy of ``response`` with every sensitive value replaced.

        Body paths apply only when the whole body is one JSON document; streamed (SSE) bodies
        are left chunk-for-chunk intact apart from secret replacement.
        """
        chunks = response.chunks
        paths = _compiled(tuple(self.body_paths))
        if paths and (document := parse_document(response.body)) is not None:
            redacted, changed = replace_paths(document, paths, self.placeholder)
            if changed:
                data = json.dumps(redacted, ensure_ascii=False, separators=(",", ":")).encode()
                chunks = (Chunk(data, sum(chunk.delay_ms for chunk in chunks)),)
        return replace(
            response,
            headers=self._redact_headers(response.headers),
            chunks=tuple(
                replace(chunk, data=self._replace_secrets_bytes(chunk.data)) for chunk in chunks
            ),
        )

    def redact(self, interaction: Interaction) -> Interaction:
        """Return ``interaction`` with both its request and response redacted."""
        return Interaction(
            request=self.redact_request(interaction.request),
            response=self.redact_response(interaction.response),
        )

    def find_secret(self, interactions: Sequence[Interaction]) -> str | None:
        """Return where a configured secret appears in ``interactions``, or ``None``.

        The result names a location such as ``interactions[2].response.body``, never the secret.
        """
        if not self.secrets:
            return None
        encoded = [secret.encode() for secret in self.secrets]
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
                if any(secret in blob for secret in encoded):
                    return f"interactions[{index}].{name}"
        return None

    def _replace_secrets(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, self.placeholder)
        return text

    def _replace_secrets_bytes(self, data: bytes) -> bytes:
        placeholder = self.placeholder.encode()
        for secret in self.secrets:
            data = data.replace(secret.encode(), placeholder)
        return data

    def _redact_headers(self, headers: Headers) -> Headers:
        return tuple(
            (
                name,
                self.placeholder
                if self.is_sensitive_header(name)
                else self._replace_secrets(value),
            )
            for name, value in headers
        )

    def _redact_url(self, url: str) -> str:
        parts = urlsplit(url)
        netloc = parts.netloc
        if parts.password is not None:
            netloc = netloc.replace(f":{parts.password}@", f":{self.placeholder}@", 1)
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
        return self._replace_secrets(
            rebuilt if (netloc, query) != (parts.netloc, parts.query) else url
        )


def _headers_blob(headers: Headers) -> bytes:
    return "\n".join(f"{name}: {value}" for name, value in headers).encode()
