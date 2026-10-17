"""Record through a fake OpenAI upstream, then replay with the real ``openai`` SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

import llm_cassette.httpx
from llm_cassette import Cassette, Matching, UnmatchedRequestError
from llm_cassette.clients import async_openai_client, openai_client
from llm_cassette.usage import usage_of
from tests.fakes import FAKE_OPENAI_KEY, HTTPX, FakeUpstream

MESSAGES: list[Any] = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "Bugün hava nasıl?"},
]


def chat(client: openai.OpenAI, **kwargs: Any) -> Any:
    return client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, **kwargs)


@pytest.mark.parametrize("gzip_json", [False, True], ids=["identity", "gzip"])
def test_chat_completion(cassette_path: Path, gzip_json: bool) -> None:  # noqa: FBT001
    upstream = FakeUpstream(gzip_json=gzip_json)
    with Cassette(cassette_path, mode="once") as cassette:
        live = chat(
            openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)
        )
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = chat(openai_client(cassette, transport=upstream.transport()))
    assert replayed.model_dump() == live.model_dump()
    assert replayed.choices[0].message.content == "Merhaba! You said: Bugün hava nasıl?"
    assert replayed.usage is not None
    assert replayed.usage.prompt_tokens == 11
    assert upstream.calls == 1


def test_streamed_chat_with_include_usage(cassette_path: Path, upstream: FakeUpstream) -> None:
    options: dict[str, Any] = {"stream": True, "stream_options": {"include_usage": True}}
    with Cassette(cassette_path, mode="record") as cassette:
        client = openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)
        live = [chunk.model_dump() for chunk in chat(client, **options)]
        (recorded,) = cassette.interactions
    assert len(recorded.response.chunks) > 10, "the fake split the stream into many chunks"

    with Cassette(cassette_path, mode="none") as cassette:
        replayed = [chunk.model_dump() for chunk in chat(openai_client(cassette), **options)]
        (played,) = cassette.played
    assert replayed == live
    assert [c.data for c in played.response.chunks] == [c.data for c in recorded.response.chunks]
    text = "".join(c["choices"][0]["delta"]["content"] or "" for c in replayed if c["choices"])
    assert text == "Merhaba! You said: Bugün hava nasıl?"
    assert replayed[-1]["usage"]["completion_tokens"] == 6
    usage = usage_of(played)
    assert usage is not None
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (7, 4, 6)


@pytest.mark.anyio
async def test_async_chat_and_stream(cassette_path: Path, upstream: FakeUpstream) -> None:
    async def run(client: openai.AsyncOpenAI) -> tuple[Any, list[Any]]:
        completion = await client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)
        stream = await client.chat.completions.create(
            model="gpt-4o-mini", messages=MESSAGES, stream=True
        )
        chunks = [chunk.model_dump() async for chunk in stream]
        return completion.model_dump(), chunks

    with Cassette(cassette_path, mode="record") as cassette:
        live = await run(
            async_openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)
        )
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = await run(async_openai_client(cassette))
    assert replayed == live
    assert upstream.calls == 2


def test_responses_api(cassette_path: Path, upstream: FakeUpstream) -> None:
    def run(client: openai.OpenAI) -> tuple[str, str]:
        response = client.responses.create(model="gpt-4o-mini", input="Kısaca anlat")
        with client.responses.stream(model="gpt-4o-mini", input="Akış") as stream:
            for _ in stream:
                pass
            final = stream.get_final_response()
        return response.output_text, final.output_text

    with Cassette(cassette_path, mode="record") as cassette:
        live = run(openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY))
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = run(openai_client(cassette))
        usages = [usage_of(i) for i in cassette.played]
    assert replayed == live == ("Merhaba! You said: Kısaca anlat", "Merhaba! You said: Akış")
    assert [(u.input_tokens, u.cache_read_tokens) for u in usages if u] == [(16, 5), (16, 5)]


def test_error_responses_replay_as_sdk_errors(cassette_path: Path) -> None:
    upstream = FakeUpstream(fail_with=429)
    with Cassette(cassette_path, mode="record") as cassette:
        client = openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)
        with pytest.raises(openai.RateLimitError):
            chat(client)
    with Cassette(cassette_path, mode="none") as cassette, pytest.raises(openai.RateLimitError):
        chat(openai_client(cassette))
    assert upstream.calls == 1, "max_retries=0: the 429 was not retried"


def test_an_unmatched_request_surfaces_through_the_sdk(
    cassette_path: Path, upstream: FakeUpstream
) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        chat(openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY))
    with Cassette(cassette_path, mode="none") as cassette:
        client = openai_client(cassette)
        # Recent openai releases let the cassette's error through; older 3.x releases wrap it in
        # APIConnectionError with the original as the cause.
        with pytest.raises((UnmatchedRequestError, openai.APIConnectionError)) as raised:
            client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        error = (
            raised.value
            if isinstance(raised.value, UnmatchedRequestError)
            else raised.value.__cause__
        )
        assert isinstance(error, UnmatchedRequestError)
        assert '+  "model": "gpt-4o"' in str(error)
        assert cassette.unmatched == (error,), "max_retries=0: the SDK did not retry"


def test_ignore_paths_tolerate_volatile_fields(cassette_path: Path, upstream: FakeUpstream) -> None:
    matching = Matching(ignore_paths=["metadata.user_id", "stream_options"])
    with Cassette(cassette_path, mode="record") as cassette:
        client = openai_client(cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)
        list(chat(client, stream=True, metadata={"user_id": "run-1"}))
    with Cassette(cassette_path, mode="none", matching=matching) as cassette:
        stream = chat(
            openai_client(cassette),
            stream=True,
            metadata={"user_id": "run-2"},
            stream_options={"include_usage": False},
        )
        assert len(list(stream)) > 3


def test_openai_with_a_legacy_httpx_client(cassette_path: Path, upstream: FakeUpstream) -> None:
    """openai>=3 still accepts an httpx (0.x) client, wired through llm_cassette.httpx."""

    def client(cassette: Cassette) -> openai.OpenAI:
        transport = llm_cassette.httpx.CassetteTransport(cassette, upstream.transport(HTTPX))
        http_client: Any = httpx.Client(transport=transport)
        return openai.OpenAI(api_key=FAKE_OPENAI_KEY, http_client=http_client, max_retries=0)

    with Cassette(cassette_path, mode="record") as cassette:
        live = chat(client(cassette))
        live_stream = [c.model_dump() for c in chat(client(cassette), stream=True)]
    with Cassette(cassette_path, mode="none") as cassette:
        assert chat(client(cassette)).model_dump() == live.model_dump()
        assert [c.model_dump() for c in chat(client(cassette), stream=True)] == live_stream
