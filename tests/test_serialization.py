from __future__ import annotations

import dataclasses
import json
import random
import textwrap

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_cassette import (
    CassetteFormatError,
    Chunk,
    Interaction,
    RecordedRequest,
    RecordedResponse,
)
from llm_cassette._json import JSONValue, canonical
from llm_cassette._serialization import Format, dumps, loads
from tests.test_json_and_paths import json_documents

FORMATS: list[Format] = ["yaml", "json"]


def interaction(
    body: bytes = b'{"model":"m"}',
    chunks: tuple[Chunk, ...] = (Chunk(b"ok", 1.5),),
    headers: tuple[tuple[str, str], ...] = (("content-type", "application/json"),),
) -> Interaction:
    return Interaction(
        RecordedRequest("POST", "https://api.example.com/v1/x", headers, body),
        RecordedResponse(200, (("content-type", "text/plain"),), chunks, delay_ms=12.0),
    )


def roundtrip(interactions: list[Interaction], fmt: Format) -> list[Interaction]:
    return loads(dumps(interactions, fmt), fmt, source="test")


GOLDEN = textwrap.dedent(
    """\
    version: 1
    interactions:
      - request:
          method: POST
          url: https://api.example.com/v1/x
          headers:
            content-type: application/json
            x-multi:
              - a
              - b
          body:
            json:
              messages:
                - content: |-
                    çok satırlı
                    mesaj
                  role: user
              model: m
              temperature: 0.7
        response:
          status: 200
          headers:
            content-type: text/plain
          delay_ms: 12.0
          chunks:
            - delay_ms: 1.2
              text: |+
                data: {"a":1}

            - delay_ms: 0.0
              base64: //4=
    """
)


def test_yaml_output_is_readable_and_stable() -> None:
    body = json.dumps(
        {
            "temperature": 0.7,
            "model": "m",
            "messages": [{"role": "user", "content": "çok satırlı\nmesaj"}],
        }
    ).encode()
    item = interaction(
        body=body,
        headers=(("content-type", "application/json"), ("x-multi", "a"), ("x-multi", "b")),
        chunks=(Chunk(b'data: {"a":1}\n\n', 1.25), Chunk(b"\xff\xfe", 0.0)),
    )
    assert dumps([item], "yaml").replace("...\n", "") == GOLDEN


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_preserves_every_field(fmt: Format) -> None:
    items = [
        interaction(),
        interaction(body=b"", chunks=()),
        interaction(body=b"plain text body"),
        interaction(body=b"\x00\xff binary", chunks=(Chunk(b"\x00\x01", 0.5), Chunk(b"tail", 2.5))),
        interaction(headers=(("set-cookie", "a=1"), ("set-cookie", "b=2"))),
    ]
    assert roundtrip(items, fmt) == items


@pytest.mark.parametrize("fmt", FORMATS)
def test_output_does_not_depend_on_input_order(fmt: Format) -> None:
    headers = [("b-header", "1"), ("a-header", "2"), ("c-header", "3")]
    body = {"z": 1, "a": {"y": 2, "b": 3}}
    first = interaction(headers=tuple(headers), body=json.dumps(body).encode())
    random.Random(0).shuffle(headers)
    reordered = {"a": {"b": 3, "y": 2}, "z": 1}
    second = interaction(
        headers=tuple(sorted(headers)), body=json.dumps(reordered, indent=4).encode()
    )
    assert dumps([first], fmt) == dumps([second], fmt)


def test_json_format_is_plain_json() -> None:
    document = json.loads(dumps([interaction()], "json"))
    assert document["version"] == 1
    assert document["interactions"][0]["request"]["body"] == {"json": {"model": "m"}}


@given(json_documents)
def test_any_json_request_body_survives_yaml(document: JSONValue) -> None:
    item = interaction(body=canonical(document).encode())
    (loaded,) = roundtrip([item], "yaml")
    loaded_body = json.loads(loaded.request.body)
    assert canonical(loaded_body) == canonical(document)


@given(st.lists(st.text(), max_size=5))
def test_any_text_chunk_survives_yaml_byte_for_byte(texts: list[str]) -> None:
    chunks = tuple(Chunk(text.encode("utf-8", "surrogatepass"), 0.0) for text in texts)
    if any(b"\xed" in chunk.data for chunk in chunks):  # lone surrogates are not valid UTF-8
        return
    (loaded,) = roundtrip([interaction(chunks=chunks)], "yaml")
    assert loaded.response.chunks == chunks


@pytest.mark.parametrize("fmt", FORMATS)
@given(st.lists(st.binary(max_size=64), max_size=5))
def test_any_binary_chunk_survives(fmt: Format, blobs: list[bytes]) -> None:
    chunks = tuple(Chunk(blob, 0.0) for blob in blobs)
    (loaded,) = roundtrip([interaction(chunks=chunks)], fmt)
    assert loaded.response.chunks == chunks


@pytest.mark.parametrize(
    "text", ["a\r\nb", "a\x85b\n", "line sep\n", "﻿bom\n", "trailing \nspace", "  indented\nx"]
)
def test_awkward_multiline_strings_survive(text: str) -> None:
    chunks = (Chunk(text.encode(), 0.0),)
    (loaded,) = roundtrip([interaction(chunks=chunks)], "yaml")
    assert loaded.response.chunks == chunks


@pytest.mark.parametrize("fmt", FORMATS)
def test_incomplete_responses_are_marked(fmt: Format) -> None:
    complete = interaction()
    incomplete = Interaction(
        complete.request, dataclasses.replace(complete.response, complete=False)
    )
    text = dumps([complete, incomplete], fmt)
    assert text.count("complete") == 1, "only the exception is written"
    assert roundtrip([complete, incomplete], fmt) == [complete, incomplete]


def test_incomplete_is_written_before_the_chunks() -> None:
    item = interaction()
    text = dumps(
        [Interaction(item.request, dataclasses.replace(item.response, complete=False))], "yaml"
    )
    assert "      delay_ms: 12.0\n      complete: false\n      chunks:\n" in text


@pytest.mark.parametrize("fmt", FORMATS)
def test_header_bytes_survive_as_latin1_text(fmt: Format) -> None:
    value = 'attachment; filename="résumé.pdf"'.encode().decode("latin-1")
    item = interaction(headers=(("content-disposition", value),))
    assert roundtrip([item], fmt) == [item]


def test_an_overflowing_json_number_is_stored_as_text() -> None:
    body = b'{"temperature": 1e400}'
    (loaded,) = roundtrip([interaction(body=body)], "yaml")
    assert loaded.request.body == body
    assert "text: '{\"temperature\": 1e400}'" in dumps([interaction(body=body)], "yaml")


def test_delays_are_rounded_to_a_tenth_of_a_millisecond() -> None:
    item = interaction(chunks=(Chunk(b"x", 1.23456),))
    (loaded,) = roundtrip([item], "yaml")
    assert loaded.response.chunks[0].delay_ms == 1.2


ERRORS = [
    ("not: [valid", "yaml", "not a valid YAML document"),
    ("{", "json", "not a valid JSON document"),
    ("- a list", "yaml", "document: expected a mapping"),
    ("interactions: []", "yaml", "unsupported cassette schema version None"),
    ("version: 2\ninteractions: []", "yaml", "unsupported cassette schema version 2"),
    ("version: 1.0\ninteractions: []", "yaml", "unsupported cassette schema version 1.0"),
    ("version: true\ninteractions: []", "yaml", "unsupported cassette schema version True"),
    ("version: 1\nextra: 1", "yaml", "unknown key"),
    ("version: 1\ninteractions: {}", "yaml", "interactions: expected a list"),
    ("version: 1\ninteractions: [1]", "yaml", r"interactions\[0\]: expected a mapping"),
]


@pytest.mark.parametrize(("text", "fmt", "match"), ERRORS)
def test_malformed_documents_are_rejected(text: str, fmt: Format, match: str) -> None:
    with pytest.raises(CassetteFormatError, match=match):
        loads(text, fmt, source="c.yaml")


def _with(path: str, value: object) -> str:
    document = json.loads(dumps([interaction()], "json"))
    target = document["interactions"][0]
    *parents, last = path.split(".")
    for key in parents:
        target = target[key] if not key.isdigit() else target[int(key)]
    if value is KeyError:
        del target[last]
    else:
        target[last] = value
    return json.dumps(document)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("response.status", "200", "expected an HTTP status code"),
        ("response.status", True, "expected an HTTP status code"),
        ("response.status", 42, "expected an HTTP status code"),
        ("response.delay_ms", -1, "non-negative number"),
        ("response.delay_ms", "fast", "non-negative number"),
        ("response.chunks", "x", "expected a list"),
        ("response.chunks.0.delay_ms", None, "non-negative number"),
        ("response.chunks.0.text", 1, "expected a string"),
        ("response.chunks.0.base64", "AA==", "exactly one of base64 or text"),
        ("response.extra", 1, "unknown key"),
        ("response.complete", "no", "expected true or false"),
        ("response.complete", 0, "expected true or false"),
        ("request.method", None, "expected a string, got nothing"),
        ("request.headers", ["a"], "expected a mapping"),
        ("request.headers", {"a": 1}, "expected a string"),
        ("request.body", {"json": 1}, "expected a JSON object or array"),
        ("request.body", {"json": {}, "text": ""}, "exactly one of"),
        ("request.body", {"base64": "!!"}, "invalid base64"),
        ("request.body", {"json": {"x": 1e999}}, "finite number"),
        ("request", KeyError, "expected a mapping with string keys, got nothing"),
    ],
)
def test_invalid_fields_name_their_location(path: str, value: object, match: str) -> None:
    with pytest.raises(CassetteFormatError, match=match) as raised:
        loads(_with(path, value), "json", source="c.json")
    assert str(raised.value).startswith("c.json: interactions[0]")


def test_yaml_only_types_are_rejected_inside_json_bodies() -> None:
    text = dumps([interaction()], "yaml").replace("model: m", "model: 2024-01-01")
    with pytest.raises(CassetteFormatError, match="expected a JSON value, got date"):
        loads(text, "yaml", source="c.yaml")


def test_header_names_are_lower_cased_on_load() -> None:
    text = dumps([interaction()], "yaml").replace(
        "content-type: application/json", "Content-Type: x"
    )
    (loaded,) = loads(text, "yaml", source="c.yaml")
    assert loaded.request.headers == (("content-type", "x"),)


def test_missing_optional_fields_default_sensibly() -> None:
    text = (
        "version: 1\n"
        "interactions:\n"
        "  - request: {method: get, url: 'https://x'}\n"
        "    response: {status: 204}\n"
    )
    (loaded,) = loads(text, "yaml", source="c.yaml")
    assert loaded == Interaction(RecordedRequest("GET", "https://x"), RecordedResponse(204))
