from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from llm_cassette import Chunk, Interaction, RecordedRequest, RecordedResponse
from llm_cassette.usage import Price, PriceTable, Usage, usage_of
from tests.fakes import sse

JSON = (("content-type", "application/json"),)
EVENT_STREAM = (("content-type", "text/event-stream; charset=utf-8"),)


def interaction(
    body: bytes, headers: tuple[tuple[str, str], ...] = JSON, model: str | None = "requested"
) -> Interaction:
    request_body = json.dumps({"model": model} if model else {"input": "x"}).encode()
    request = RecordedRequest("POST", "https://api.example.com/v1/x", (), request_body)
    # Split the body so parsing must not depend on chunk boundaries.
    chunks = tuple(Chunk(body[i : i + 5]) for i in range(0, len(body), 5))
    return Interaction(request, RecordedResponse(200, headers, chunks))


def streamed(events: list[tuple[str | None, Any]]) -> Interaction:
    return interaction(sse(events), EVENT_STREAM)


def test_openai_chat_completion() -> None:
    body = {
        "object": "chat.completion",
        "model": "gpt-4o-mini-2024-07-18",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
    }
    usage = usage_of(interaction(json.dumps(body).encode()))
    assert usage == Usage(
        provider="openai",
        model="gpt-4o-mini-2024-07-18",
        requested_model="requested",
        input_tokens=36,
        cache_read_tokens=64,
        output_tokens=20,
    )
    assert usage.total_input_tokens == 100


def test_openai_stream_with_include_usage() -> None:
    chunk = {
        "object": "chat.completion.chunk",
        "model": "gpt-x",
        "choices": [{"delta": {"content": "hi"}}],
        "usage": None,
    }
    final = {
        "object": "chat.completion.chunk",
        "model": "gpt-x",
        "choices": [],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
    }
    usage = usage_of(streamed([(None, chunk), (None, chunk), (None, final), (None, "[DONE]")]))
    assert usage is not None
    assert (usage.provider, usage.model, usage.input_tokens, usage.output_tokens) == (
        "openai",
        "gpt-x",
        9,
        3,
    )


def test_openai_stream_without_usage_reports_nothing() -> None:
    chunk = {
        "object": "chat.completion.chunk",
        "model": "gpt-x",
        "choices": [{"delta": {"content": "hi"}}],
    }
    assert usage_of(streamed([(None, chunk), (None, "[DONE]")])) is None


def test_openai_responses() -> None:
    response = {
        "object": "response",
        "model": "gpt-r",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 10},
        },
    }
    usage = usage_of(interaction(json.dumps(response).encode()))
    assert usage is not None
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (40, 10, 7)


@pytest.mark.parametrize(
    "terminal", ["response.completed", "response.incomplete", "response.failed"]
)
def test_openai_responses_stream(terminal: str) -> None:
    response = {
        "object": "response",
        "model": "gpt-r",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    events: list[tuple[str | None, Any]] = [
        ("response.created", {"type": "response.created", "response": {**response, "usage": None}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "x"}),
        (terminal, {"type": terminal, "response": response}),
    ]
    usage = usage_of(streamed(events))
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (5, 2)


def test_anthropic_message() -> None:
    message = {
        "type": "message",
        "model": "claude-x",
        "usage": {
            "input_tokens": 12,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 30,
            "output_tokens": 8,
        },
    }
    usage = usage_of(interaction(json.dumps(message).encode()))
    assert usage == Usage("anthropic", "claude-x", "requested", 12, 100, 30, 8)
    assert usage.total_input_tokens == 142


def test_anthropic_stream_applies_cumulative_message_delta_counts() -> None:
    start = {
        "type": "message",
        "model": "claude-x",
        "usage": {"input_tokens": 12, "cache_read_input_tokens": 4, "output_tokens": 1},
    }
    events: list[tuple[str | None, Any]] = [
        ("message_start", {"type": "message_start", "message": start}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"text": "hi"}}),
        ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 5}}),
        (
            "message_delta",
            {"type": "message_delta", "usage": {"output_tokens": 9, "input_tokens": 13}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    usage = usage_of(streamed(events))
    assert usage is not None
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (13, 4, 9)


def test_message_delta_without_start_still_counts() -> None:
    usage = usage_of(
        streamed(
            [
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "usage": {
                            "output_tokens": 3,
                            "cache_read_input_tokens": 2,
                            "cache_creation_input_tokens": 1,
                        },
                    },
                )
            ]
        )
    )
    assert usage is not None
    assert (
        usage.provider,
        usage.model,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    ) == ("anthropic", "requested", 3, 2, 1)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[1, 2]",
        json.dumps({"object": "list", "data": []}).encode(),
        json.dumps({"object": "chat.completion", "usage": None}).encode(),
        json.dumps({"object": "response", "usage": "n/a"}).encode(),
        json.dumps({"type": "message", "usage": 3}).encode(),
    ],
)
def test_responses_without_usage(body: bytes) -> None:
    assert usage_of(interaction(body)) is None


def test_stream_payloads_that_are_not_objects_are_skipped() -> None:
    events: list[tuple[str | None, Any]] = [
        (None, "[DONE]"),
        ("message_start", {"type": "message_start", "message": "oops"}),
        ("response.completed", {"type": "response.completed", "response": None}),
    ]
    assert usage_of(streamed(events)) is None


def test_bad_counts_are_treated_as_zero() -> None:
    body = {
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": -5,
            "completion_tokens": True,
            "prompt_tokens_details": {"cached_tokens": "4"},
        },
    }
    usage = usage_of(interaction(json.dumps(body).encode()))
    assert usage is not None
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (0, 0, 0)
    assert usage.model == "requested", "falls back to the requested model"


def test_requests_without_a_model() -> None:
    body = {"object": "chat.completion", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    usage = usage_of(interaction(json.dumps(body).encode(), model=None))
    assert usage is not None
    assert (usage.model, usage.requested_model) == (None, None)


# --------------------------------------------------------------------------------- prices


def test_price_cost_uses_every_bucket() -> None:
    price = Price(
        input=Decimal(2), output=Decimal(8), cache_read=Decimal("0.5"), cache_write=Decimal("2.5")
    )
    usage = Usage(
        "anthropic",
        "m",
        input_tokens=1000,
        cache_read_tokens=2000,
        cache_write_tokens=400,
        output_tokens=500,
    )
    assert price.cost(usage) == Decimal("0.008")  # (2000 + 1000 + 1000 + 4000) / 1e6


def test_cache_tokens_default_to_the_input_price() -> None:
    usage = Usage(
        "openai",
        "m",
        input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert Price(input=Decimal(1), output=Decimal(0)).cost(usage) == Decimal(3)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prices.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_and_lookup(tmp_path: Path) -> None:
    table = PriceTable.load(
        write(
            tmp_path,
            """
            [models."gpt-4o-mini"]
            input = 1.10
            output = 4

            [models."gpt-4o*"]
            input = 9
            output = 9

            [models."claude-*"]
            input = 3
            output = 15
            cache_read = 0.3
            cache_write = 3.75
            """,
        )
    )
    assert len(table) == 3
    assert table.lookup("gpt-4o-mini") == Price(Decimal("1.10"), Decimal(4))
    assert table.lookup("gpt-4o-mini-2024-07-18", "gpt-4o-mini") == Price(
        Decimal("1.10"), Decimal(4)
    ), "an exact match on any candidate beats a pattern"
    assert table.lookup("gpt-4o-2024-08-06") == Price(Decimal(9), Decimal(9))
    claude = table.lookup("claude-sonnet-4-6")
    assert claude is not None
    assert claude.cache_write == Decimal("3.75")
    assert table.lookup("o3", None, "") is None
    usage = Usage("openai", "gpt-4o-mini-2024-07-18", "gpt-4o-mini", input_tokens=1_000_000)
    assert table.cost(usage) == Decimal("1.10")
    assert table.cost(Usage("openai", "unknown")) is None


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("models = [", "invalid TOML"),
        ("[other]\nx = 1", "expected only a \\[models\\] table"),
        ("[models]\nm = 1", "at least input and output"),
        ('[models."m"]\ninput = 1', "at least input and output"),
        ('[models."m"]\ninput = 1\noutput = 1\nbatch = 1', "unknown key"),
        ('[models."m"]\ninput = -1\noutput = 1', "non-negative"),
        ('[models."m"]\ninput = "1"\noutput = 1', "expected a number"),
        ('[models."m"]\ninput = true\noutput = 1', "expected a number"),
        ('[models."m"]\ninput = nan\noutput = 1', "non-negative"),
    ],
)
def test_invalid_price_tables(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PriceTable.load(write(tmp_path, text))


def test_message_delta_without_output_tokens_keeps_earlier_counts() -> None:
    start = {
        "type": "message",
        "model": "claude-x",
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    events: list[tuple[str | None, Any]] = [
        ("message_start", {"type": "message_start", "message": start}),
        ("message_delta", {"type": "message_delta", "usage": {"server_tool_use": {}}}),
    ]
    usage = usage_of(streamed(events))
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (2, 1)
