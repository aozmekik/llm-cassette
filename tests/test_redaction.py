from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
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
        "https://user:REDACTED@api.example.com/v1/x"
        "?key=REDACTED&sig=REDACTED&model=m&access_token=REDACTED"
    )


def test_urls_without_sensitive_parts_are_untouched() -> None:
    url = "https://api.example.com/v1/x?b=%20&a=1"
    assert Redaction().redact_request(request(url=url)).url == url


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
        ({"placeholder": "a\nb"}, ValueError, "valid header value"),
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
