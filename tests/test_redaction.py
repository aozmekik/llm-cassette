from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote, quote_plus, unquote, unquote_plus

import httpx2
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from llm_cassette import (
    DEFAULT_REDACTED_HEADERS,
    Cassette,
    Chunk,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    Redaction,
    SecretLeakError,
)
from llm_cassette.clients import anthropic_client, openai_client
from llm_cassette.httpx2 import CassetteTransport
from tests.fakes import FAKE_ANTHROPIC_KEY, FAKE_OPENAI_KEY, FAKE_ORG, FakeUpstream

R = "REDACTED"


def request(**overrides: object) -> RecordedRequest:
    defaults: dict[str, object] = {
        "method": "POST",
        "url": "https://api.example.com/v1/x",
        "headers": (),
        "body": b"",
    }
    return RecordedRequest(**{**defaults, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize("name", sorted(DEFAULT_REDACTED_HEADERS))
def test_default_headers_are_redacted(name: str) -> None:
    redacted = Redaction().redact_request(request(headers=((name, "secret-value"),)))
    assert redacted.headers == ((name, R),)


@pytest.mark.parametrize(
    "name", ["x-goog-api-key", "api-key", "x-auth-token", "x-client-secret", "X-API-KEY"]
)
def test_headers_with_sensitive_substrings_are_redacted(name: str) -> None:
    assert Redaction().is_sensitive_header(name)


@pytest.mark.parametrize("name", ["content-type", "user-agent", "x-request-id", "keep-alive"])
def test_ordinary_headers_are_kept(name: str) -> None:
    assert not Redaction().is_sensitive_header(name)
    assert Redaction().redact_request(request(headers=((name, "v"),))).headers == ((name, "v"),)


def test_substrings_can_be_narrowed_to_keep_rate_limit_headers() -> None:
    narrow = Redaction(substrings=("key", "secret"))
    assert not narrow.is_sensitive_header("x-ratelimit-remaining-tokens")
    assert Redaction().is_sensitive_header("x-ratelimit-remaining-tokens")


@given(
    st.from_regex(r"[a-z\-]{0,8}(key|token|secret)[a-z\-]{0,8}", fullmatch=True),
    st.text(st.characters(codec="ascii", exclude_categories=["Cc"]), min_size=1),
)
def test_any_sensitive_header_name_is_redacted(name: str, value: str) -> None:
    redacted = Redaction().redact_request(request(headers=((name, value),)))
    assert redacted.headers == ((name, R),)


def test_query_parameters_and_userinfo_are_redacted() -> None:
    redaction = Redaction(query_params={"Sig"})
    url = "https://user:hunter2@api.example.com/v1/x?key=AIza-FAKE&sig=abc&model=m&access_token=t"
    redacted = redaction.redact_request(request(url=url)).url
    assert redacted == (
        "https://REDACTED:REDACTED@api.example.com/v1/x"
        "?key=REDACTED&sig=REDACTED&model=m&access_token=REDACTED"
    )


@pytest.mark.parametrize(
    ("netloc", "expected"),
    [
        ("sk-live-USERNAMETOKEN@api.example.com", "REDACTED@api.example.com"),
        ("user:pw@api.example.com:8443", "REDACTED:REDACTED@api.example.com:8443"),
        (":pw@api.example.com", ":REDACTED@api.example.com"),
        ("user:@api.example.com", "REDACTED:@api.example.com"),
        ("api.example.com", "api.example.com"),
    ],
)
def test_all_of_the_user_info_is_redacted(netloc: str, expected: str) -> None:
    redacted = Redaction().redact_request(request(url=f"https://{netloc}/v1/x")).url
    assert redacted == f"https://{expected}/v1/x"


def test_a_placeholder_is_percent_encoded_in_user_info() -> None:
    redaction = Redaction(placeholder="<redacted>")
    redacted = redaction.redact_request(request(url="https://token@api.example.com/")).url
    assert redacted == "https://%3Credacted%3E@api.example.com/"


def test_url_valued_response_headers_are_redacted() -> None:
    presigned = (
        "https://bucket.s3.amazonaws.com/x?X-Amz-Security-Token=TOPSECRETSTS&X-Amz-Expires=60"
    )
    response = RecordedResponse(
        302,
        (
            ("content-location", "/v1/files/1?api_key=FAKEKEY"),
            ("link", '<https://api.example.com/p?page=2&token=T1>; rel="next", <https://u@h/>'),
            ("location", presigned),
            ("x-other", "https://api.example.com/?token=kept-in-an-ordinary-header"),
        ),
    )
    headers = dict(Redaction().redact_response(response).headers)
    assert headers["location"] == (
        "https://bucket.s3.amazonaws.com/x?X-Amz-Security-Token=REDACTED&X-Amz-Expires=60"
    )
    assert headers["content-location"] == "/v1/files/1?api_key=REDACTED"
    assert headers["link"] == (
        '<https://api.example.com/p?page=2&token=REDACTED>; rel="next", <https://REDACTED@h/>'
    )
    assert headers["x-other"].endswith("kept-in-an-ordinary-header")


def test_a_malformed_url_header_is_left_alone() -> None:
    response = RecordedResponse(302, (("location", "http://[::1/broken"),))
    assert Redaction().redact_response(response).headers == (("location", "http://[::1/broken"),)


def test_the_referer_request_header_is_redacted() -> None:
    headers = (("referer", "https://app.example.com/?session_token=abc"),)
    redacted = Redaction().redact_request(request(headers=headers))
    assert redacted.headers == (("referer", "https://app.example.com/?session_token=REDACTED"),)


def test_urls_without_sensitive_parts_are_untouched() -> None:
    url = "https://api.example.com/v1/x?b=%20&a=1"
    assert Redaction().redact_request(request(url=url)).url == url


def test_other_query_parameters_keep_the_encoding_the_client_chose() -> None:
    url = "https://api.example.com/v1/x?b=%20&a=1+2&&api%5Fkey=k1&c=%7e&token&d=x=y"
    assert Redaction().redact_request(request(url=url)).url == (
        "https://api.example.com/v1/x?b=%20&a=1+2&&api%5Fkey=REDACTED&c=%7e&token=REDACTED&d=x=y"
    )


B64_SECRET = "Zm9vYmFy+YmF6/cXV4=="  # base64 text: holds +, / and =


@pytest.mark.parametrize(
    "encoded",
    [
        B64_SECRET,  # as typed into the URL; httpx sends it like this
        quote(B64_SECRET),
        quote(B64_SECRET, safe=""),
        quote_plus(B64_SECRET),
        "Zm9vYmFy+YmF6%2FcXV4%3D%3D",  # + kept, / and = escaped
        "Zm9vYmFy%2bYmF6%2fcXV4%3d%3d",  # lower-case hex
        "%5Am9vYmFy+YmF6/cXV4==",  # an unreserved character escaped
    ],
)
@pytest.mark.parametrize(
    ("sensitive", "stored"), [("", ""), ("api_key=k&", f"api_key={R}&")], ids=["alone", "api-key"]
)
def test_a_secret_in_a_url_is_replaced_however_it_was_percent_encoded(
    encoded: str, sensitive: str, stored: str
) -> None:
    redaction = Redaction(secrets=[B64_SECRET])
    url = f"https://api.example.com/v1/{encoded}/x?{sensitive}q={encoded}#{encoded}"
    redacted = redaction.redact_request(request(url=url))
    assert redacted.url == f"https://api.example.com/v1/{R}/x?{stored}q={R}#{R}"
    assert redaction.find_secret([Interaction(redacted, RecordedResponse(200))]) is None


def test_a_secret_ending_in_a_percent_sign_takes_its_whole_escape() -> None:
    redaction = Redaction(secrets=["0000000%"])
    url = f"https://api.example.com/{quote('0000000%')}?q={quote_plus('0000000%')}&r=0000000%"
    assert (
        redaction.redact_request(request(url=url)).url == f"https://api.example.com/{R}?q={R}&r={R}"
    )


@pytest.mark.parametrize(
    ("secret", "sent"),
    [
        ("a+b c/d=e-0001", "a+b%20c/d=e-0001"),  # a space is escaped, + / = are not
        ("pässwörd+1/2", "p%C3%A4ssw%C3%B6rd+1/2"),  # so are non-ASCII characters
        ("100%-sure-0001", "100%-sure-0001"),  # a % that starts no escape is kept
    ],
)
def test_secrets_in_raw_urls_are_replaced_in_the_form_httpx_sends(
    tmp_path: Path, secret: str, sent: str
) -> None:
    wire: list[str] = []

    def upstream(incoming: httpx2.Request) -> httpx2.Response:
        wire.append(str(incoming.url))
        return httpx2.Response(200, json={})

    path = tmp_path / "c.yaml"
    with Cassette(path, mode="record", redaction=Redaction(secrets=[secret])) as cassette:
        client = httpx2.Client(
            transport=CassetteTransport(cassette, httpx2.MockTransport(upstream))
        )
        client.get(f"https://x.example.com/v1/{secret}?q={secret}")
        client.get(f"https://x.example.com/v1/x?api_key=k&q={secret}")
    assert wire == [
        f"https://x.example.com/v1/{sent}?q={sent}",
        f"https://x.example.com/v1/x?api_key=k&q={sent}",
    ]
    text = path.read_text(encoding="utf-8")
    assert "url: https://x.example.com/v1/REDACTED?q=REDACTED" in text
    assert "url: https://x.example.com/v1/x?api_key=REDACTED&q=REDACTED" in text
    for view in (text, unquote(text), unquote_plus(text)):
        assert secret not in view


def test_request_body_paths() -> None:
    body = json.dumps({"model": "m", "metadata": {"user_id": "u-1"}, "user": "alice"}).encode()
    redacted = Redaction(body_paths=["metadata.user_id", "user"]).redact_request(request(body=body))
    assert json.loads(redacted.body) == {"model": "m", "metadata": {"user_id": R}, "user": R}


def test_body_paths_leave_non_json_and_unmatched_bodies_untouched() -> None:
    redaction = Redaction(body_paths=["user"])
    assert redaction.redact_request(request(body=b"user=alice")).body == b"user=alice"
    raw = b'{"model":  "m"}'
    assert redaction.redact_request(request(body=raw)).body == raw


def test_response_body_paths_apply_to_json_but_not_streams() -> None:
    redaction = Redaction(body_paths=["user"])
    response = RecordedResponse(
        200,
        (("content-type", "application/json"),),
        (Chunk(b'{"user": "al', 1.0), Chunk(b'ice"}', 2.0)),
    )
    redacted = redaction.redact_response(response)
    assert redacted.chunks == (Chunk(b'{"user":"REDACTED"}', 3.0),)
    stream = RecordedResponse(200, (), (Chunk(b'data: {"user": "alice"}\n\n'),))
    assert redaction.redact_response(stream) == stream


JSON_TYPE = (("content-type", "application/json; charset=utf-8"),)


def test_a_json_response_that_does_not_parse_is_replaced_whole() -> None:
    """A truncated body may hold the configured fields, which cannot be located in it."""
    truncated = RecordedResponse(
        200,
        JSON_TYPE,
        (Chunk(b'{"user": {"email": "alice@example.com", "na', 1.0),),
        complete=False,
    )
    redacted = Redaction(body_paths=["user.email"]).redact_response(truncated)
    assert redacted.chunks == (Chunk(b"REDACTED", 1.0),)
    assert not redacted.complete


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        (JSON_TYPE, b'"just a string"'),
        (JSON_TYPE, b""),
        (JSON_TYPE, b"  \n"),
        ((("content-type", "text/plain"),), b"user=alice"),
        ((("content-type", "application/problem+xml"),), b"<user>alice</user>"),
    ],
    ids=["json-scalar", "empty", "blank", "text", "xml"],
)
def test_bodies_body_paths_cannot_apply_to_are_kept(
    headers: tuple[tuple[str, str], ...], body: bytes
) -> None:
    response = RecordedResponse(200, headers, (Chunk(body),) if body else ())
    assert Redaction(body_paths=["user"]).redact_response(response) == response


def test_vendor_json_media_types_count_as_json() -> None:
    response = RecordedResponse(
        200, (("content-type", "application/vnd.api+json"),), (Chunk(b"{broken"),)
    )
    assert Redaction(body_paths=["x"]).redact_response(response).body == b"REDACTED"


def test_content_length_follows_a_redacted_body() -> None:
    body = json.dumps({"user": {"email": "alice@example.com"}, "id": 1}, indent=2).encode()
    response = RecordedResponse(
        200, (*JSON_TYPE, ("content-length", str(len(body)))), (Chunk(body),)
    )
    redacted = Redaction(body_paths=["user.email"]).redact_response(response)
    assert redacted.header("content-length") == str(len(redacted.body))
    assert len(redacted.body) < len(body)

    secret = "sk-very-secret-0001"
    echo = RecordedResponse(
        200, (("content-length", "29"),), (Chunk(f"you sent {secret}!".encode()),)
    )
    redacted = Redaction(secrets=[secret]).redact_response(echo)
    assert redacted.body == b"you sent REDACTED!"
    assert redacted.header("content-length") == "18"


def test_content_length_of_an_incomplete_body_keeps_the_missing_part() -> None:
    secret = "sk-very-secret-0001"
    partial = RecordedResponse(
        200,
        (("content-length", "1000"),),
        (Chunk(f"echo {secret}".encode()),),
        complete=False,
    )
    redacted = Redaction(secrets=[secret]).redact_response(partial)
    assert redacted.header("content-length") == str(1000 - len(secret) + len("REDACTED"))


def test_an_unparseable_content_length_is_left_alone() -> None:
    secret = "sk-very-secret-0001"
    response = RecordedResponse(200, (("content-length", "n/a"),), (Chunk(secret.encode()),))
    assert Redaction(secrets=[secret]).redact_response(response).header("content-length") == "n/a"


def test_secrets_are_replaced_everywhere() -> None:
    secret = "sk-very-secret-0001"
    redaction = Redaction(secrets=[secret], substrings=("nothing-matches",), headers=frozenset())
    interaction = Interaction(
        request(
            url=f"https://api.example.com/v1/x?k={secret}",
            headers=(("x-custom", f"Bearer {secret}"),),
            body=f'{{"echo": "{secret}"}}'.encode(),
        ),
        RecordedResponse(200, (("x-echo", secret),), (Chunk(f"you sent {secret}".encode()),)),
    )
    redacted = redaction.redact(interaction)
    assert secret not in repr(redacted)
    assert redaction.find_secret([redacted]) is None
    assert redacted.response.chunks[0].data == b"you sent REDACTED"


SECRET = 'pässwörd-"quoted"-\\-123456'


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"password": SECRET}).encode(),  # \u00e4 escapes, the json.dumps default
        json.dumps({"password": SECRET}, ensure_ascii=False).encode(),  # only \" and \\ escaped
        json.dumps({"tool": {"arguments": json.dumps({"password": SECRET})}}).encode(),
        json.dumps({"password": SECRET}).replace("/", "\\/").replace("p", "\\u0070").encode(),
    ],
    ids=["ascii-escapes", "unicode", "nested-json-string", "unusual-escapes"],
)
def test_json_escaped_secrets_in_request_bodies_are_replaced(body: bytes) -> None:
    redaction = Redaction(secrets=[SECRET])
    redacted = redaction.redact_request(request(body=body))
    decoded = json.dumps(json.loads(redacted.body), ensure_ascii=False)
    assert "pässwörd" not in decoded
    assert "pässwörd" not in redacted.body.decode()
    assert redaction.find_secret([Interaction(redacted, RecordedResponse(200))]) is None


def test_escaped_secrets_never_reach_the_file(tmp_path: Path) -> None:
    """Record through httpx2 with json= and form data; grep the cassette for the secret."""
    secret = 'Zm9v+YmFy/YmF6=-pässwörd"x'

    def echo(incoming: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"echo": incoming.content.decode()})

    path = tmp_path / "c.yaml"
    with Cassette(path, mode="record", redaction=Redaction(secrets=[secret])) as cassette:
        client = httpx2.Client(transport=CassetteTransport(cassette, httpx2.MockTransport(echo)))
        client.post("https://x.example.com/", json={"password": secret})
        client.post("https://x.example.com/", params={"sig": secret}, data={"assertion": secret})
        client.get(f"https://x.example.com/files/{quote(secret, safe='')}")
        client.get(f"https://x.example.com/files/{secret}?api_key=k&q={secret}")  # httpx escapes
    text = path.read_text(encoding="utf-8")
    for form in (secret, quote_plus(secret), quote(secret, safe=""), json.dumps(secret)[1:-1]):
        assert form not in text
    for view in (unquote(text), unquote_plus(text)):
        assert secret not in view
    assert "url: https://x.example.com/files/REDACTED?api_key=REDACTED&q=REDACTED" in text
    assert text.count("REDACTED") >= 9


@pytest.mark.parametrize(
    "form",
    [
        re.sub(r"%[0-9A-F]{2}", lambda m: m.group().lower(), quote_plus(B64_SECRET)),
        "Zm9vYmFy+YmF6%2FcXV4%3D%3D",  # + read as itself, not as a space
        quote('{"k": "Zm9vYmFy\\u002BYmF6/cXV4=="}'),  # a JSON escape inside percent-encoding
    ],
    ids=["lower-case-hex", "literal-plus", "percent-encoded-json"],
)
def test_find_secret_decodes_bodies_redaction_leaves_alone(form: str) -> None:
    """Outside URLs, only the common encodings of a secret are replaced; the check decodes."""
    redaction = Redaction(secrets=[B64_SECRET])
    leaked = Interaction(request(body=f"q={form}".encode()), RecordedResponse(200))
    assert redaction.redact(leaked).request.body == f"q={form}".encode()
    assert redaction.find_secret([redaction.redact(leaked)]) == "interactions[0].request.body"


def test_find_secret_looks_inside_decoded_views() -> None:
    secret = "Zm9v+YmFy/YmF6="
    redaction = Redaction(secrets=[secret])
    split = '{"a": "Zm9v\\u002BYmFy", "b": "/YmF6="}'
    stream = RecordedResponse(200, (), (Chunk(b'data: {"echo": "Zm9v\\u002BYmFy/YmF6="}\n\n'),))
    assert (
        redaction.find_secret([Interaction(request(), stream)]) == "interactions[0].response.body"
    )
    assert (
        redaction.find_secret([Interaction(request(body=split.encode()), RecordedResponse(200))])
        is None
    )
    assert redaction.find_secret([], serialized=f"text: {secret}\n") == "the serialized cassette"
    assert redaction.find_secret([], serialized="text: nothing\n") is None


def test_escaped_secrets_are_found_anywhere_in_a_document() -> None:
    secret = "pässwörd-123456"
    body = json.dumps({"list": [1, None, {"deep": [f"x {secret}"]}], secret: True}).encode()
    redacted = Redaction(secrets=[secret]).redact_request(request(body=body))
    assert json.loads(redacted.body) == {"list": [1, None, {"deep": ["x REDACTED"]}], R: True}


def test_text_that_only_looks_like_json_strings_is_searched_as_is() -> None:
    redaction = Redaction(secrets=["sk-very-secret-0001"])
    invalid_escape = b'not json: "\\q" and "sk-very-secret-0001"'
    leaked = Interaction(request(body=invalid_escape), RecordedResponse(200))
    assert redaction.find_secret([leaked]) == "interactions[0].request.body"
    clean = Interaction(request(body=b'"\\q" "\\u00e4"'), RecordedResponse(200))
    assert redaction.find_secret([clean]) is None


def test_a_secret_inside_a_longer_secret_is_replaced_whole() -> None:
    redaction = Redaction(secrets=["sk-short-01", "sk-short-0123456789"])
    redacted = redaction.redact_request(request(body=b"key=sk-short-0123456789"))
    assert redacted.body == b"key=REDACTED"


def test_non_ascii_secrets_in_header_values_are_replaced() -> None:
    secret = "pässwörd-123456"
    stored = f"Basic {secret}".encode().decode("latin-1")  # how wire bytes are stored
    redaction = Redaction(secrets=[secret], substrings=("nothing",), headers=frozenset())
    redacted = redaction.redact_request(request(headers=(("x-custom", stored),)))
    assert redacted.headers == (("x-custom", "Basic REDACTED"),)
    leaked = Interaction(request(headers=(("x-custom", stored),)), RecordedResponse(200))
    assert redaction.find_secret([leaked]) == "interactions[0].request.headers"
    typed = redaction.redact_request(request(headers=(("x-note", f"şeker {secret}"),)))
    assert typed.headers == (("x-note", "şeker REDACTED"),), "hand-typed text works too"


def test_find_secret_catches_a_secret_split_across_chunks() -> None:
    secret = "sk-very-secret-0001"
    redaction = Redaction(secrets=[secret])
    response = RecordedResponse(200, (), (Chunk(b"echo: sk-very-"), Chunk(b"secret-0001")))
    redacted = redaction.redact(Interaction(request(), response))
    assert redaction.find_secret([redacted]) == "interactions[0].response.body"
    assert Redaction().find_secret([redacted]) is None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"secrets": [""]}, ValueError, "non-empty strings"),
        ({"secrets": ["RED"]}, ValueError, "must not contain a secret"),
        ({"secrets": ["%3Cred"], "placeholder": "<red>"}, ValueError, "encoded forms"),
        ({"placeholder": "a\nb"}, ValueError, "valid header value"),
        ({"placeholder": "REDACTED\t"}, ValueError, "printable ASCII"),
        ({"placeholder": "gizlendi—"}, ValueError, "printable ASCII"),
        ({"drop_headers": "user-agent"}, TypeError, "not a single string"),
        ({"substrings": [""]}, ValueError, "redact every header"),
        ({"headers": "authorization"}, TypeError, "not a single string"),
        ({"query_params": "key"}, TypeError, "not a single string"),
        ({"secrets": "sk-abc"}, TypeError, "not a single string"),
        ({"body_paths": ["a..b"]}, ValueError, "invalid JSON path"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Redaction(**kwargs)  # type: ignore[arg-type]


def test_dropped_headers_disappear_from_requests_and_responses() -> None:
    redaction = Redaction(drop_headers=["X-Stainless-*", "user-agent", "date"])
    assert redaction.drop_headers == ("x-stainless-*", "user-agent", "date")
    interaction = Interaction(
        request(
            headers=(
                ("authorization", "Bearer sk-x"),
                ("user-agent", "OpenAI/Python 3.34.0"),
                ("x-stainless-os", "MacOS"),
                ("X-Stainless-Runtime-Version", "3.14.5"),
            )
        ),
        RecordedResponse(200, (("content-type", "application/json"), ("date", "Sat, 18 Oct"))),
    )
    redacted = redaction.redact(interaction)
    assert redacted.request.headers == (("authorization", R),)
    assert redacted.response.headers == (("content-type", "application/json"),)
    assert redaction.is_dropped_header("X-STAINLESS-ARCH")
    assert not Redaction().is_dropped_header("user-agent"), "nothing is dropped by default"


def test_the_default_headers_can_be_extended() -> None:
    redaction = Redaction(headers=DEFAULT_REDACTED_HEADERS | {"X-Tenant-Id"})
    assert redaction.is_sensitive_header("x-tenant-id")
    assert redaction.is_sensitive_header("set-cookie")


secret_texts = st.text(st.characters(codec="utf-8", exclude_categories=["Cs"]), min_size=8)


def _carrying(*, path: str, form: str, json_text: str, json_ascii: str) -> Interaction:
    """An interaction with the given texts in its URL path and query and in both JSON bodies."""
    body = f'{{"user": "u", "echo": "{json_text}", "echo_ascii": "{json_ascii}"}}'.encode()
    return Interaction(
        request(url=f"https://user:pw@api.example.com/{path}?sig={form}", body=body),
        RecordedResponse(200, JSON_TYPE, (Chunk(body),)),
    )


@given(secret_texts)
def test_redaction_is_idempotent_and_leaves_no_secret(secret: str) -> None:
    assume(secret not in R)  # Redaction rejects a secret that the placeholder contains
    redaction = Redaction(secrets=[secret], body_paths=["user"])
    # A perfect redaction leaves the fixed text with placeholders where the secret was. A secret
    # that occurs in that text anyway (in a header name, or across the host and a placeholder as
    # in "com/REDA") cannot be told apart from it, and find_secret reports it on purpose (see the
    # next test), so the property is about every other secret.
    perfect = Redaction(body_paths=["user"]).redact(
        _carrying(path=R, form=R, json_text=R, json_ascii=R)
    )
    assume(redaction.find_secret([perfect]) is None)
    interaction = _carrying(
        path=quote(secret),
        form=quote_plus(secret),
        json_text=json.dumps(secret, ensure_ascii=False)[1:-1],
        json_ascii=json.dumps(secret)[1:-1],
    )
    once = redaction.redact(interaction)
    assert redaction.redact(once) == once
    assert redaction.find_secret([once]) is None
    assert once.request.url == perfect.request.url


#: Characters a query value can hold unescaped without changing the URL's structure.
_LITERAL_IN_QUERY = frozenset("-._~!$'()*+,;=:@/?")


@given(secret_texts, st.data())
def test_a_secret_in_a_url_never_survives_in_any_mix_of_escapes(
    secret: str, data: st.DataObject
) -> None:
    """Each character of the secret is sent as itself or escaped, in either hex case."""
    assume(secret not in R)
    redaction = Redaction(secrets=[secret])
    sensitive = data.draw(st.sampled_from(["", "api_key=k&"]), label="sensitive")
    template = "https://api.example.com/v1/{}/x?" + sensitive + "q={}"
    perfect = template.format(R, R).replace("=k&", f"={R}&")
    assume(not any(secret in view for view in (perfect, unquote(perfect), unquote_plus(perfect))))

    def encoded() -> str:
        sent = []
        for char in secret:
            literal = char.isalnum() or char in _LITERAL_IN_QUERY
            if literal and data.draw(st.booleans()):
                sent.append(char)
            elif char == " " and data.draw(st.booleans()):
                sent.append("+")
            else:
                escapes = quote(char, safe="")
                sent.append(escapes.lower() if data.draw(st.booleans()) else escapes)
        return "".join(sent)

    url = template.format(encoded(), encoded())
    once = redaction.redact_request(request(url=url))
    for view in (once.url, unquote(once.url), unquote_plus(once.url)):
        assert secret not in view
    assert redaction.find_secret([Interaction(once, RecordedResponse(200))]) is None
    assert redaction.redact_request(once) == once


@pytest.mark.parametrize(
    ("secret", "where"),
    [
        ("content-", "interactions[0].response.headers"),  # part of the header name content-type
        ("com/REDA", "interactions[0].request.url"),  # the host followed by a placeholder
    ],
)
def test_a_secret_that_matches_text_redaction_keeps_is_reported(secret: str, where: str) -> None:
    """The leak check cannot tell such a secret from the text, so it errs on the side of safety."""
    redaction = Redaction(secrets=[secret])
    interaction = _carrying(path=quote(secret), form="x", json_text="x", json_ascii="x")
    assert redaction.find_secret([redaction.redact(interaction)]) == where


def test_header_names_are_case_insensitive() -> None:
    redaction = Redaction(headers={"X-Tenant"})
    assert redaction.headers == frozenset({"x-tenant"})
    assert redaction.is_sensitive_header("X-TENANT")


def test_saving_refuses_to_write_a_surviving_secret(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    secret = "sk-very-secret-0001"
    cassette = Cassette(path, mode="record", redaction=Redaction(secrets=[secret]))
    recording = cassette.resolve(request())
    assert not isinstance(recording, Interaction)
    recording.start(200, [], delay_ms=0)
    recording.add_chunk(b"echo: sk-very-", delay_ms=0)
    recording.add_chunk(b"secret-0001", delay_ms=0)
    with pytest.raises(SecretLeakError, match=r"interactions\[0\]\.response\.body"):
        cassette.save()
    assert not path.exists()


def test_fake_keys_never_reach_disk(tmp_path: Path) -> None:
    """Record real SDK traffic carrying fake credentials and grep the cassette for them."""
    upstream = FakeUpstream()
    path = tmp_path / "c.yaml"
    with Cassette(path, mode="record", redaction=Redaction(body_paths=["metadata.user_id"])) as c:
        openai = openai_client(
            c,
            transport=upstream.transport(),
            api_key=FAKE_OPENAI_KEY,
            organization=FAKE_ORG,
            project="proj_FAKEPROJECT",
        )
        openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_id": "user-FAKE-PII"},
        )
        anthropic = anthropic_client(c, transport=upstream.transport(), api_key=FAKE_ANTHROPIC_KEY)
        anthropic.messages.create(
            model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
    text = path.read_text(encoding="utf-8")
    for leaked in (
        FAKE_OPENAI_KEY,
        FAKE_ANTHROPIC_KEY,
        FAKE_ORG,
        "proj_FAKEPROJECT",
        "FAKECOOKIE",
        "user-FAKE-PII",
        "00000000-fake",
    ):
        assert leaked not in text
    assert text.count("REDACTED") >= 8
    assert upstream.requests[0].headers["authorization"] == f"Bearer {FAKE_OPENAI_KEY}"
