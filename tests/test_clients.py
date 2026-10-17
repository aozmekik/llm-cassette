from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from llm_cassette import Cassette
from llm_cassette.clients import (
    REPLAY_API_KEY,
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    openai_client,
)
from llm_cassette.httpx2 import AsyncCassetteTransport, CassetteTransport

Factory = Callable[..., Any]
FACTORIES: list[tuple[Factory, type[Any], type[Any]]] = [
    (openai_client, openai.OpenAI, CassetteTransport),
    (async_openai_client, openai.AsyncOpenAI, AsyncCassetteTransport),
    (anthropic_client, anthropic.Anthropic, CassetteTransport),
    (async_anthropic_client, anthropic.AsyncAnthropic, AsyncCassetteTransport),
]
IDS = ["openai", "async_openai", "anthropic", "async_anthropic"]


@pytest.fixture
def replaying(cassette_path: Path) -> Cassette:
    cassette_path.parent.mkdir(parents=True)
    cassette_path.write_text("version: 1\ninteractions: []\n")
    return Cassette(cassette_path, mode="none")


@pytest.mark.parametrize(("factory", "kind", "transport"), FACTORIES, ids=IDS)
def test_helpers_wire_the_sdk_client_to_the_cassette(
    factory: Factory, kind: type[Any], transport: type[Any], replaying: Cassette
) -> None:
    client = factory(replaying)
    assert isinstance(client, kind)
    assert client.max_retries == 0
    assert client.api_key == REPLAY_API_KEY
    http_client = client._client
    assert isinstance(http_client, httpx2.Client | httpx2.AsyncClient)
    assert http_client.follow_redirects
    wired: Any = http_client._transport
    assert isinstance(wired, transport)
    assert wired.cassette is replaying


@pytest.mark.parametrize(("factory", "kind", "transport"), FACTORIES, ids=IDS)
def test_keyword_arguments_are_forwarded(
    factory: Factory, kind: type[Any], transport: type[Any], replaying: Cassette
) -> None:
    del kind, transport
    client = factory(
        replaying, api_key="sk-explicit", max_retries=3, base_url="https://proxy.example.com/v1"
    )
    assert (client.api_key, client.max_retries) == ("sk-explicit", 3)
    assert str(client.base_url) == "https://proxy.example.com/v1/"


@pytest.mark.parametrize(("factory", "kind", "transport"), FACTORIES, ids=IDS)
def test_http_client_cannot_be_overridden(
    factory: Factory, kind: type[Any], transport: type[Any], replaying: Cassette
) -> None:
    del kind, transport
    with pytest.raises(TypeError, match="pass transport= to change the upstream"):
        factory(replaying, http_client=httpx2.Client())


def test_recording_needs_a_real_key(cassette_path: Path) -> None:
    recording = Cassette(cassette_path, mode="record")
    with pytest.raises(openai.OpenAIError, match="api_key"):
        openai_client(recording)
    with pytest.raises(TypeError, match="api_key"):
        anthropic_client(recording).messages.create(model="m", max_tokens=1, messages=[])


def test_recording_reads_the_key_from_the_environment(
    cassette_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert openai_client(Cassette(cassette_path, mode="record")).api_key == "sk-from-env"
