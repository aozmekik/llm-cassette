"""Fixtures for the demo suite.

``fake_upstream`` stands in for api.openai.com and api.anthropic.com so the demo can record
without credentials; its responses are synthetic. In a real suite you would not pass
``transport=`` to the client helpers, and recording would talk to the real APIs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest

from llm_cassette import Matching, Redaction
from llm_cassette.pytest_plugin import CassetteConfig


class _Stream(httpx2.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _sse(payloads: list[dict[str, Any]], *, done: bool) -> list[bytes]:
    events = [f"data: {json.dumps(p, ensure_ascii=False)}\n\n".encode() for p in payloads]
    return [*events, b"data: [DONE]\n\n"] if done else events


def _openai(body: dict[str, Any]) -> httpx2.Response:
    prompt = body["messages"][-1]["content"]
    answer = f"Echo: {prompt}"
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    base = {"id": "chatcmpl-demo", "created": 1_760_000_000, "model": "gpt-4o-mini-2024-07-18"}
    if not body.get("stream"):
        message = {"role": "assistant", "content": answer}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        payload = {**base, "object": "chat.completion", "choices": [choice], "usage": usage}
        return httpx2.Response(200, json=payload)
    deltas = [{"content": piece} for piece in answer.split(" ")]
    chunks = [
        {
            **base,
            "object": "chat.completion.chunk",
            "usage": None,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
    headers = {"content-type": "text/event-stream"}
    return httpx2.Response(200, headers=headers, stream=_Stream(_sse(chunks, done=True)))


def _anthropic(body: dict[str, Any]) -> httpx2.Response:
    prompt = body["messages"][-1]["content"]
    message = {
        "id": "msg_demo",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": f"Echo: {prompt}"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "cache_read_input_tokens": 0, "output_tokens": 4},
    }
    return httpx2.Response(200, json=message, headers={"request-id": "req_demo"})


def _handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    if request.url.path == "/v1/chat/completions":
        return _openai(body)
    if request.url.path == "/v1/messages":
        return _anthropic(body)
    return httpx2.Response(404)


@pytest.fixture
def fake_upstream() -> httpx2.MockTransport:
    """A synthetic stand-in for the provider APIs (see the module docstring)."""
    return httpx2.MockTransport(_handler)


@pytest.fixture
def llm_cassette_config() -> CassetteConfig:
    """Project-wide cassette defaults: ignore per-run metadata, scrub the user field."""
    return {
        "matching": Matching(ignore_paths=["metadata.run_id"]),
        "redaction": Redaction(body_paths=["user"]),
    }
