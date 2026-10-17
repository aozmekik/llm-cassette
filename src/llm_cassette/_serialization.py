"""Reading and writing cassette files (YAML or JSON, schema version 1).

The on-disk document is::

    version: 1
    interactions:
      - request:
          method: POST
          url: https://api.openai.com/v1/chat/completions
          headers: {name: value | [values]}      # sorted by name
          body: {json: ...} | {text: ...} | {base64: ...}   # omitted when empty
        response:
          status: 200
          headers: {name: value | [values]}
          delay_ms: 312.5                         # request sent -> headers received
          chunks:
            - delay_ms: 0.4                       # since the previous chunk
              text: "..."                         # or base64: "..."

Output is deterministic: mapping keys appear in a fixed order, headers and JSON request bodies
are sorted, and YAML is always produced by PyYAML's pure-Python emitter (libyaml's C emitter
makes slightly different style choices, which would make files differ between machines).
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, cast

import yaml

from llm_cassette._errors import CassetteFormatError
from llm_cassette._json import JSONValue, canonical, parse_document, sort_keys
from llm_cassette._model import Chunk, Headers, Interaction, RecordedRequest, RecordedResponse

SCHEMA_VERSION = 1

Format: TypeAlias = Literal["yaml", "json"]

_Document: TypeAlias = dict[str, object]

# Characters PyYAML treats as line breaks (YAML 1.1) or strips: in plain, single-quoted or
# literal scalars the loader would fold them into "\n" or a space, so such strings are always
# written double-quoted, where they are escaped.
_NEEDS_ESCAPES = frozenset("\r\x85\u2028\u2029\ufeff")

_BODY_KINDS = frozenset({"json", "text", "base64"})
_CHUNK_KINDS = frozenset({"text", "base64"})
_CHUNK_EXTRA = frozenset({"delay_ms"})

_Loader: type[yaml.CSafeLoader | yaml.SafeLoader]
try:
    _Loader = yaml.CSafeLoader
except AttributeError:  # pragma: no cover - depends on how PyYAML was built
    _Loader = yaml.SafeLoader


class _Dumper(yaml.SafeDumper):
    """A safe dumper that indents block sequences and never wraps long lines."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # noqa: FBT001, FBT002
        """Indent sequences inside mappings, which reads better in diffs."""
        del indentless
        super().increase_indent(flow, indentless=False)


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    # Multi-line strings (SSE chunks, pretty-printed JSON, prompts) read best as literal blocks.
    # PyYAML itself falls back to a quoted style when a literal block cannot hold the value.
    style: str | None = None
    if _NEEDS_ESCAPES.intersection(value):
        style = '"'
    elif "\n" in value:
        style = "|"
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _represent_str)


# --------------------------------------------------------------------------------------- dumping


def _dump_headers(headers: Headers) -> dict[str, str | list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, value in headers:
        grouped.setdefault(name, []).append(value)
    return {
        name: values[0] if len(values) == 1 else values for name, values in sorted(grouped.items())
    }


def _dump_bytes(data: bytes) -> dict[str, object]:
    try:
        return {"text": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"base64": base64.b64encode(data).decode("ascii")}


def _dump_request_body(body: bytes) -> dict[str, object]:
    document = parse_document(body)
    if document is not None:
        return {"json": sort_keys(document)}
    return _dump_bytes(body)


def _delay(value: float) -> float:
    return round(value, 1)


def _dump_interaction(interaction: Interaction) -> _Document:
    request, response = interaction.request, interaction.response
    request_doc: _Document = {
        "method": request.method,
        "url": request.url,
        "headers": _dump_headers(request.headers),
    }
    if request.body:
        request_doc["body"] = _dump_request_body(request.body)
    response_doc: _Document = {
        "status": response.status,
        "headers": _dump_headers(response.headers),
        "delay_ms": _delay(response.delay_ms),
        "chunks": [
            {"delay_ms": _delay(chunk.delay_ms), **_dump_bytes(chunk.data)}
            for chunk in response.chunks
        ],
    }
    return {"request": request_doc, "response": response_doc}


def dumps(interactions: Sequence[Interaction], fmt: Format) -> str:
    """Serialize ``interactions`` as a cassette document in format ``fmt``."""
    document = {
        "version": SCHEMA_VERSION,
        "interactions": [_dump_interaction(interaction) for interaction in interactions],
    }
    if fmt == "json":
        return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    return yaml.dump(
        document,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        indent=2,
        width=2**31 - 1,
    )


# --------------------------------------------------------------------------------------- loading


class _Reader:
    """Validates the loosely-typed parsed document, naming the failing location in errors."""

    def __init__(self, source: str) -> None:
        self._source = source

    def fail(self, where: str, message: str) -> CassetteFormatError:
        return CassetteFormatError(f"{self._source}: {where}: {message}")

    def mapping(self, value: object, where: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise self.fail(where, f"expected a mapping with string keys, got {_kind(value)}")
        return cast("Mapping[str, object]", value)

    def sequence(self, value: object, where: str) -> Sequence[object]:
        if not isinstance(value, list):
            raise self.fail(where, f"expected a list, got {_kind(value)}")
        return cast("list[object]", value)

    def string(self, value: object, where: str) -> str:
        if not isinstance(value, str):
            raise self.fail(where, f"expected a string, got {_kind(value)}")
        return value

    def status(self, value: object, where: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 100 <= value <= 999:  # noqa: PLR2004
            raise self.fail(where, f"expected an HTTP status code, got {value!r}")
        return value

    def delay(self, value: object, where: str) -> float:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise self.fail(where, f"expected a non-negative number of milliseconds, got {value!r}")
        return float(value)

    def known_keys(self, value: Mapping[str, object], allowed: set[str], where: str) -> None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise self.fail(where, f"unknown key(s) {', '.join(unknown)}")

    def headers(self, value: object, where: str) -> Headers:
        pairs: list[tuple[str, str]] = []
        for name, entry in self.mapping(value, where).items():
            values = entry if isinstance(entry, list) else [entry]
            pairs.extend((name, self.string(v, f"{where}.{name}")) for v in values)
        return tuple(sorted(((n.lower(), v) for n, v in pairs), key=lambda pair: pair[0]))

    def json_value(self, value: object, where: str) -> JSONValue:
        """Check that ``value`` holds only JSON types (YAML can also produce dates, sets...)."""
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise self.fail(where, f"expected a finite number, got {value!r}")
            return value
        if isinstance(value, list):
            return [self.json_value(item, f"{where}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, dict):
            return {
                self.string(key, f"{where} key"): self.json_value(item, f"{where}.{key}")
                for key, item in value.items()
            }
        raise self.fail(where, f"expected a JSON value, got {_kind(value)}")

    def data(
        self, value: object, where: str, *, kinds: frozenset[str], extra: frozenset[str]
    ) -> bytes:
        body = self.mapping(value, where)
        present = sorted(kinds.intersection(body))
        if len(present) != 1 or set(body) - kinds - extra:
            raise self.fail(where, f"expected exactly one of {' or '.join(sorted(kinds))}")
        kind = present[0]
        if kind == "json":
            document = self.json_value(body["json"], f"{where}.json")
            if not isinstance(document, dict | list):
                location = f"{where}.json"
                raise self.fail(location, "expected a JSON object or array")
            return canonical(document).encode()
        text = self.string(body[kind], f"{where}.{kind}")
        if kind == "text":
            return text.encode("utf-8")
        try:
            return base64.b64decode(text, validate=True)
        except binascii.Error as error:
            location = f"{where}.base64"
            raise self.fail(location, f"invalid base64: {error}") from None

    def chunk(self, value: object, where: str) -> Chunk:
        return Chunk(
            data=self.data(value, where, kinds=_CHUNK_KINDS, extra=_CHUNK_EXTRA),
            delay_ms=self.delay(
                self.mapping(value, where).get("delay_ms", 0.0), f"{where}.delay_ms"
            ),
        )

    def interaction(self, value: object, where: str) -> Interaction:
        entry = self.mapping(value, where)
        self.known_keys(entry, {"request", "response"}, where)
        request = self.mapping(entry.get("request"), f"{where}.request")
        self.known_keys(request, {"method", "url", "headers", "body"}, f"{where}.request")
        response = self.mapping(entry.get("response"), f"{where}.response")
        self.known_keys(response, {"status", "headers", "delay_ms", "chunks"}, f"{where}.response")
        chunks = tuple(
            self.chunk(chunk, f"{where}.response.chunks[{i}]")
            for i, chunk in enumerate(
                self.sequence(response.get("chunks", []), f"{where}.response.chunks")
            )
        )
        return Interaction(
            request=RecordedRequest(
                method=self.string(request.get("method"), f"{where}.request.method").upper(),
                url=self.string(request.get("url"), f"{where}.request.url"),
                headers=self.headers(request.get("headers", {}), f"{where}.request.headers"),
                body=self.data(
                    request["body"], f"{where}.request.body", kinds=_BODY_KINDS, extra=frozenset()
                )
                if "body" in request
                else b"",
            ),
            response=RecordedResponse(
                status=self.status(response.get("status"), f"{where}.response.status"),
                headers=self.headers(response.get("headers", {}), f"{where}.response.headers"),
                chunks=chunks,
                delay_ms=self.delay(response.get("delay_ms", 0.0), f"{where}.response.delay_ms"),
            ),
        )


def _kind(value: object) -> str:
    return "nothing" if value is None else type(value).__name__


def loads(text: str, fmt: Format, *, source: str) -> list[Interaction]:
    """Parse a cassette document; ``source`` names the file in error messages.

    Raises:
        CassetteFormatError: If the document is not valid YAML/JSON, has an unsupported schema
            version, or does not have the expected shape.
    """
    reader = _Reader(source)
    try:
        parsed: object = json.loads(text) if fmt == "json" else yaml.load(text, Loader=_Loader)  # noqa: S506 - a safe loader
    except (ValueError, yaml.YAMLError) as error:
        msg = f"{source}: not a valid {fmt.upper()} document: {error}"
        raise CassetteFormatError(msg) from None
    document = reader.mapping(parsed, "document")
    version = document.get("version")
    if version != SCHEMA_VERSION:
        msg = (
            f"unsupported cassette schema version {version!r} "
            f"(this llm-cassette reads version {SCHEMA_VERSION})"
        )
        where = "version"
        raise reader.fail(where, msg)
    reader.known_keys(document, {"version", "interactions"}, "document")
    entries = reader.sequence(document.get("interactions", []), "interactions")
    return [reader.interaction(entry, f"interactions[{i}]") for i, entry in enumerate(entries)]
