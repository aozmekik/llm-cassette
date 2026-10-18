from __future__ import annotations

import json

import pytest

from llm_cassette import Interaction, Matching, RecordedRequest, RecordedResponse
from llm_cassette._matching import (
    Candidate,
    describe_unmatched,
    mismatches,
    normalize_url,
    request_key,
)

URL = "https://api.openai.com/v1/chat/completions"


def post(body: object, *, url: str = URL, method: str = "POST", **headers: str) -> RecordedRequest:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return RecordedRequest(method=method, url=url, headers=tuple(headers.items()), body=raw)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://API.Example.COM/v1/x", "https://api.example.com/v1/x"),
        ("https://api.example.com:443/v1", "https://api.example.com/v1"),
        ("http://api.example.com:80/v1", "http://api.example.com/v1"),
        ("https://api.example.com:8443/v1", "https://api.example.com:8443/v1"),
        ("https://api.example.com", "https://api.example.com/"),
        ("https://api.example.com/x?b=2&a=1&a=0", "https://api.example.com/x?a=0&a=1&b=2"),
        ("https://api.example.com/x?a=", "https://api.example.com/x?a="),
        ("https://api.example.com/x#fragment", "https://api.example.com/x"),
        ("https://user:pw@api.example.com/x", "https://api.example.com/x"),
        ("http://[::1]:8000/x", "http://[::1]:8000/x"),
        ("http://example.com:notaport/x", "http://example.com:notaport/x"),
    ],
)
def test_normalize_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


def test_normalize_url_drops_ignored_query_parameters() -> None:
    url = "https://example.com/v1?api-version=2024&trace=abc"
    assert normalize_url(url, ["trace"]) == "https://example.com/v1?api-version=2024"


def test_json_bodies_match_structurally() -> None:
    recorded = post({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    live = RecordedRequest(
        "post", URL, (), b'{\n  "messages": [{"content": "hi", "role": "user"}],\n  "model": "m"\n}'
    )
    assert Matching().matches(live, recorded)


def test_json_values_keep_their_types() -> None:
    assert not Matching().matches(post({"stream": True}), post({"stream": 1}))


def test_non_json_bodies_match_byte_for_byte() -> None:
    assert Matching().matches(post(b"a=1"), post(b"a=1"))
    assert not Matching().matches(post(b"a=1"), post(b"a=1 "))
    assert not Matching().matches(post(b'{"a": 1}'), post(b"{'a': 1}"))


def test_each_aspect_can_be_disabled() -> None:
    a, b = post({"x": 1}), post({"x": 2}, url=URL + "?v=2", method="PUT")
    assert mismatches(Matching(), a, request_key(Matching(), a), b, request_key(Matching(), b)) == [
        "method",
        "url",
        "body",
    ]
    assert Matching(method=False, url=False, body=False).matches(a, b)


def test_ignore_paths_remove_volatile_fields() -> None:
    a = post(
        {"model": "m", "metadata": {"user_id": "1"}, "stream_options": {"include_usage": True}}
    )
    b = post({"model": "m", "metadata": {"user_id": "2"}})
    assert not Matching().matches(a, b)
    assert Matching(ignore_paths=["metadata.user_id", "stream_options"]).matches(a, b)


def test_ignore_query() -> None:
    a = post({}, url=URL + "?trace=1&v=1")
    b = post({}, url=URL + "?v=1&trace=2")
    assert not Matching().matches(a, b)
    assert Matching(ignore_query=["trace"]).matches(a, b)


def test_header_matching_is_opt_in_and_case_insensitive() -> None:
    a = post({}, **{"anthropic-beta": "x"})
    b = post({}, **{"anthropic-beta": "y"})
    assert Matching().matches(a, b)
    matching = Matching(headers=["Anthropic-Beta"])
    assert matching.headers == ("anthropic-beta",)
    assert not matching.matches(a, b)
    assert mismatches(matching, a, request_key(matching, a), b, request_key(matching, b)) == [
        "header anthropic-beta"
    ]


def test_custom_matchers_see_both_requests() -> None:
    seen: list[tuple[str, str]] = []

    def same_model(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        seen.append((live.url, recorded.url))
        live_body, recorded_body = live.json(), recorded.json()
        assert isinstance(live_body, dict)
        assert isinstance(recorded_body, dict)
        return live_body["model"] == recorded_body["model"]

    matching = Matching(body=False, custom=[same_model])
    assert matching.matches(post({"model": "a", "x": 1}), post({"model": "a", "x": 2}))
    assert not matching.matches(post({"model": "a"}), post({"model": "b"}))
    assert seen == [(URL, URL), (URL, URL)]


def test_custom_matchers_only_run_when_the_built_in_aspects_match() -> None:
    calls: list[str] = []

    def spy(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        del live
        calls.append(recorded.method)
        return True

    matching = Matching(custom=[spy])
    assert not matching.matches(post({}), post({}, method="GET"))
    assert calls == []
    assert matching.matches(post({}), post({}))
    assert calls == ["POST"]


def test_matching_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        Matching(False)  # type: ignore[call-arg]  # noqa: FBT003


@pytest.mark.parametrize("argument", ["headers", "ignore_paths", "ignore_query"])
def test_a_bare_string_is_rejected(argument: str) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        Matching(**{argument: "metadata.user_id"})  # type: ignore[arg-type]


def test_invalid_ignore_path_is_rejected_early() -> None:
    with pytest.raises(ValueError, match="invalid JSON path"):
        Matching(ignore_paths=["a..b"])


def test_matching_is_hashable_and_comparable() -> None:
    assert Matching(ignore_paths=["a"]) == Matching(ignore_paths=("a",))
    assert hash(Matching(ignore_paths=["a"])) == hash(Matching(ignore_paths=("a",)))


# ------------------------------------------------------------------------------ explanations


def _candidate(
    index: int, request: RecordedRequest, live: RecordedRequest, matching: Matching
) -> Candidate:
    reasons = mismatches(
        matching, live, request_key(matching, live), request, request_key(matching, request)
    )
    return Candidate(index, Interaction(request, RecordedResponse(200)), reasons)


def test_unmatched_message_shows_a_body_diff_against_the_closest_request() -> None:
    matching = Matching()
    live = post({"model": "m", "messages": [{"role": "user", "content": "Say goodbye"}]})
    recorded = [
        post({"model": "m"}, url=URL.replace("chat/completions", "embeddings")),
        post({"model": "m", "messages": [{"role": "user", "content": "Say hello"}]}),
    ]
    candidates = [_candidate(i, r, live, matching) for i, r in enumerate(recorded)]
    message = describe_unmatched(
        matching, live, candidates, cassette="c.yaml (mode=none)", exhausted=0
    )
    assert message.splitlines()[:3] == [
        f"no recorded interaction matches POST {URL}",
        "cassette: c.yaml (mode=none)",
        "closest recorded interaction: #1, differs in: body",
    ]
    assert '-      "content": "Say hello",' in message
    assert '+      "content": "Say goodbye",' in message
    assert "hint: re-record" in message


def test_ties_are_broken_by_body_similarity() -> None:
    matching = Matching()
    live = post({"prompt": "the quick brown fox jumps"})
    recorded = [
        post({"prompt": "something else entirely"}),
        post({"prompt": "the quick brown fox"}),
    ]
    candidates = [_candidate(i, r, live, matching) for i, r in enumerate(recorded)]
    message = describe_unmatched(matching, live, candidates, cassette="c", exhausted=0)
    assert "closest recorded interaction: #1" in message


def test_unmatched_message_explains_method_url_and_header_differences() -> None:
    matching = Matching(headers=["x-tenant"])
    live = post({"a": 1}, url=URL + "?v=2", method="PUT", **{"x-tenant": "b"})
    candidates = [_candidate(0, post({"a": 1}, **{"x-tenant": "a"}), live, matching)]
    message = describe_unmatched(matching, live, candidates, cassette="c", exhausted=0)
    assert "differs in: method, url, header x-tenant" in message
    assert "method: recorded POST, live PUT" in message
    assert f"url: - {URL}" in message
    assert f"+ {URL}?v=2" in message
    assert "header x-tenant: recorded 'a', live 'b'" in message


def test_long_diffs_are_truncated() -> None:
    matching = Matching()
    live = post({"items": list(range(200))})
    candidates = [_candidate(0, post({"items": list(range(1000, 1200))}), live, matching)]
    message = describe_unmatched(matching, live, candidates, cassette="c", exhausted=0)
    assert "more diff lines)" in message
    assert len(message.splitlines()) < 60


def test_binary_and_empty_bodies_are_described() -> None:
    matching = Matching()
    live = post(b"\xff\x00\x01")
    candidates = [_candidate(0, post(b""), live, matching)]
    message = describe_unmatched(matching, live, candidates, cassette="c", exhausted=0)
    assert "-<empty body>" in message
    assert "+<3 bytes of binary data>" in message


def test_plain_text_bodies_are_diffed_line_by_line() -> None:
    matching = Matching()
    live = post(b"line one\nline two")
    candidates = [_candidate(0, post(b"line one\nline 2"), live, matching)]
    message = describe_unmatched(matching, live, candidates, cassette="c", exhausted=0)
    assert "-line 2" in message
    assert "+line two" in message


def test_exhausted_and_empty_cassettes_have_specific_messages() -> None:
    live = post({})
    exhausted = describe_unmatched(Matching(), live, [], cassette="c", exhausted=2)
    assert "2 matching interaction(s) were already played" in exhausted
    assert "allow_repeats=True" in exhausted
    empty = describe_unmatched(Matching(), live, [], cassette="c", exhausted=0)
    assert "the cassette has no recorded interactions" in empty


def test_equal_headers_match() -> None:
    matching = Matching(headers=["anthropic-version"])
    a = post({}, **{"anthropic-version": "2023-06-01"})
    assert matching.matches(a, post({}, **{"anthropic-version": "2023-06-01"}))
    assert not matching.matches(a, post({}))
