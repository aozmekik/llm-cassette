"""Record through a fake Anthropic upstream, then replay with the real ``anthropic`` SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
import pytest

from llm_cassette import Cassette, UnmatchedRequestError
from llm_cassette.clients import anthropic_client, async_anthropic_client
from llm_cassette.usage import usage_of
from tests.fakes import FAKE_ANTHROPIC_KEY, FakeUpstream

MODEL = "claude-sonnet-4-6"
MESSAGES: list[Any] = [{"role": "user", "content": "Selam, nasılsın?"}]
EXPECTED = "Merhaba! You said: Selam, nasılsın?"


def test_message(cassette_path: Path, upstream: FakeUpstream) -> None:
    with Cassette(cassette_path, mode="once") as cassette:
        client = anthropic_client(
            cassette, transport=upstream.transport(), api_key=FAKE_ANTHROPIC_KEY
        )
        live = client.messages.create(model=MODEL, max_tokens=64, messages=MESSAGES)
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = anthropic_client(cassette).messages.create(
            model=MODEL, max_tokens=64, messages=MESSAGES
        )
        (played,) = cassette.played
    assert replayed.model_dump() == live.model_dump()
    assert replayed.content[0].type == "text"
    assert replayed.content[0].text == EXPECTED
    usage = usage_of(played)
    assert usage is not None
    assert (
        usage.input_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.output_tokens,
    ) == (25, 11, 3, 5)


def test_streamed_message_events(cassette_path: Path, upstream: FakeUpstream) -> None:
    def run(client: anthropic.Anthropic) -> tuple[list[str], str, int]:
        with client.messages.stream(model=MODEL, max_tokens=64, messages=MESSAGES) as stream:
            kinds: list[str] = [event.type for event in stream]
            final = stream.get_final_message()
        return (
            kinds,
            "".join(b.text for b in final.content if b.type == "text"),
            final.usage.output_tokens,
        )

    with Cassette(cassette_path, mode="record") as cassette:
        live = run(
            anthropic_client(cassette, transport=upstream.transport(), api_key=FAKE_ANTHROPIC_KEY)
        )
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = run(anthropic_client(cassette))
        (played,) = cassette.played
    assert replayed == live
    kinds, text, output_tokens = replayed
    assert text == EXPECTED
    assert output_tokens == 5
    for kind in (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ):
        assert kind in kinds
    usage = usage_of(played)
    assert usage is not None
    assert usage.output_tokens == 5, "message_delta's cumulative count replaces message_start's"


@pytest.mark.anyio
async def test_async_message_and_raw_stream(cassette_path: Path, upstream: FakeUpstream) -> None:
    async def run(client: anthropic.AsyncAnthropic) -> tuple[str, list[str]]:
        message = await client.messages.create(model=MODEL, max_tokens=64, messages=MESSAGES)
        stream = await client.messages.create(
            model=MODEL, max_tokens=64, messages=MESSAGES, stream=True
        )
        events: list[str] = [event.type async for event in stream]
        block = message.content[0]
        assert block.type == "text"
        return block.text, events

    with Cassette(cassette_path, mode="record") as cassette:
        live = await run(
            async_anthropic_client(
                cassette, transport=upstream.transport(), api_key=FAKE_ANTHROPIC_KEY
            )
        )
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = await run(async_anthropic_client(cassette))
    assert replayed == live
    assert replayed[1][0] == "message_start"
    assert replayed[1][-1] == "message_stop"
    assert upstream.calls == 2


def test_missing_credentials_are_reported_by_the_fake(
    cassette_path: Path, upstream: FakeUpstream
) -> None:
    """The fake upstream behaves like the real API when the key is missing."""
    with Cassette(cassette_path, mode="record") as cassette:
        client = anthropic_client(
            cassette,
            transport=upstream.transport(),
            api_key=FAKE_ANTHROPIC_KEY,
            default_headers={"x-api-key": anthropic.Omit()},
        )
        with pytest.raises(anthropic.AuthenticationError):
            client.messages.create(model=MODEL, max_tokens=8, messages=MESSAGES)


def test_an_unmatched_request_is_wrapped_by_the_sdk(cassette_path: Path) -> None:
    """The anthropic SDK wraps transport errors in APIConnectionError, keeping ours as the cause."""
    with Cassette(cassette_path, mode="none") as cassette:
        client = anthropic_client(cassette)
        with pytest.raises(anthropic.APIConnectionError) as raised:
            client.messages.create(model=MODEL, max_tokens=8, messages=MESSAGES)
        assert isinstance(raised.value.__cause__, UnmatchedRequestError)
        assert len(cassette.unmatched) == 1, "max_retries=0: the SDK did not retry"
