"""Fake OpenAI and Anthropic upstreams for tests: no test ever talks to a real API.

The payloads follow the shapes documented in the OpenAI API reference (Chat Completions,
including ``chat.completion.chunk`` streaming with ``stream_options.include_usage``, and the
Responses API) and the Anthropic Messages API reference (including the streaming events
``message_start``, ``content_block_start``, ``ping``, ``content_block_delta``,
``content_block_stop``, ``message_delta`` and ``message_stop``). Values are synthetic.

Streams are deliberately split at awkward byte offsets (inside JSON, inside multi-byte UTF-8
characters, several events per chunk) so tests prove that chunk boundaries survive recording.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import httpx
import httpx2

import llm_cassette.httpx
import llm_cassette.httpx2

CREATED = 1_760_000_000
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
FAKE_OPENAI_KEY = "sk-proj-FAKE0123456789abcdefFAKE"
FAKE_ANTHROPIC_KEY = "sk-ant-api03-FAKE0123456789abcdefFAKE"
FAKE_ORG = "org-FAKEORGANIZATION"


@dataclass(frozen=True)
class HttpLib:
    """One of the two HTTP libraries under test, with its llm-cassette adapter module."""

    name: str
    http: ModuleType
    adapter: ModuleType

    def stream(self, chunks: list[bytes]) -> Any:
        """A response stream (sync and async) yielding exactly ``chunks``."""
        sync_base: type = self.http.SyncByteStream
        async_base: type = self.http.AsyncByteStream

        class ChunkStream(sync_base, async_base):  # type: ignore[misc]
            def __iter__(self) -> Iterator[bytes]:
                yield from chunks

            async def __aiter__(self) -> AsyncIterator[bytes]:
                for chunk in chunks:
                    yield chunk

        return ChunkStream()


HTTPX = HttpLib("httpx", httpx, llm_cassette.httpx)
HTTPX2 = HttpLib("httpx2", httpx2, llm_cassette.httpx2)
LIBS = (HTTPX, HTTPX2)


@dataclass(frozen=True)
class FakeRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class FakeReply:
    status: int
    headers: list[tuple[str, str]]
    chunks: list[bytes]


def split(data: bytes, sizes: list[int]) -> list[bytes]:
    """Split ``data`` into pieces of the given sizes, cycling through ``sizes``."""
    pieces: list[bytes] = []
    offset, index = 0, 0
    while offset < len(data):
        size = sizes[index % len(sizes)]
        pieces.append(data[offset : offset + size])
        offset += size
        index += 1
    return pieces


def sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    """Encode events as an event stream; a ``str`` payload is sent verbatim (e.g. ``[DONE]``)."""
    out = []
    for name, payload in events:
        data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
        prefix = f"event: {name}\n" if name else ""
        out.append(f"{prefix}data: {data}\n\n")
    return "".join(out).encode()


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return " ".join(part.get("text", "") for part in content)
    return ""


def reply_text(prompt: str) -> str:
    """The deterministic answer every fake gives: it includes non-ASCII text on purpose."""
    return f"Merhaba! You said: {prompt}"


def _words(text: str) -> list[str]:
    words = text.split(" ")
    return [word if i == 0 else f" {word}" for i, word in enumerate(words)]


@dataclass
class FakeUpstream:
    """A fake API host serving OpenAI and Anthropic endpoints.

    Attributes:
        requests: Every request received, in order.
        gzip_json: Compress non-streaming JSON responses, like the real APIs often do.
        chunk_sizes: Byte sizes used to split streamed bodies into network chunks.
        fail_with: If set, every request gets this status and an error body.
    """

    requests: list[FakeRequest] = field(default_factory=list)
    gzip_json: bool = False
    chunk_sizes: list[int] = field(default_factory=lambda: [7, 61, 3, 140])
    fail_with: int | None = None

    @property
    def calls(self) -> int:
        return len(self.requests)

    def transport(self, lib: HttpLib = HTTPX2) -> Any:
        """A ``MockTransport`` of ``lib`` backed by this fake (works for sync and async)."""

        def handler(request: Any) -> Any:
            fake = FakeRequest(
                method=request.method,
                path=request.url.path,
                headers={k.lower(): v for k, v in request.headers.multi_items()},
                body=request.content,
            )
            reply = self.handle(fake)
            return lib.http.Response(
                reply.status, headers=reply.headers, stream=lib.stream(reply.chunks)
            )

        return lib.http.MockTransport(handler)

    def handle(self, request: FakeRequest) -> FakeReply:
        self.requests.append(request)
        routes: dict[tuple[str, str], Callable[[FakeRequest], FakeReply]] = {
            ("POST", "/v1/chat/completions"): self._openai_chat,
            ("POST", "/v1/responses"): self._openai_responses,
            ("POST", "/v1/messages"): self._anthropic_messages,
        }
        route = routes.get((request.method, request.path))
        if route is None:
            return self._json(404, {"error": {"message": "not found", "type": "invalid_request"}})
        if self.fail_with is not None:
            return self._json(self.fail_with, {"error": {"message": "fake failure"}})
        return route(request)

    # ----------------------------------------------------------------------------- helpers

    def _json(
        self, status: int, payload: dict[str, Any], headers: list[tuple[str, str]] | None = None
    ) -> FakeReply:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode()
        all_headers = [("content-type", "application/json"), *(headers or [])]
        if self.gzip_json:
            body = gzip.compress(body)
            all_headers.append(("content-encoding", "gzip"))
        all_headers.append(("content-length", str(len(body))))
        return FakeReply(status, all_headers, [body])

    def _stream(self, body: bytes, headers: list[tuple[str, str]]) -> FakeReply:
        return FakeReply(
            200,
            [("content-type", "text/event-stream; charset=utf-8"), *headers],
            split(body, self.chunk_sizes),
        )

    # ------------------------------------------------------------------------------ openai

    def _openai_headers(self) -> list[tuple[str, str]]:
        number = self.calls
        return [
            ("openai-organization", FAKE_ORG),
            ("openai-processing-ms", "321"),
            ("openai-version", "2020-10-01"),
            ("x-request-id", f"req_fake{number:04d}"),
            ("x-ratelimit-remaining-tokens", "149990"),
            ("set-cookie", "__cf_bm=FAKECOOKIE; path=/; HttpOnly"),
        ]

    def _openai_unauthorized(self, request: FakeRequest) -> FakeReply | None:
        if request.headers.get("authorization", "").startswith("Bearer "):
            return None
        error = {
            "message": "You didn't provide an API key.",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
        return self._json(401, {"error": error})

    def _openai_chat(self, request: FakeRequest) -> FakeReply:
        if (denied := self._openai_unauthorized(request)) is not None:
            return denied
        body = request.json()
        text = reply_text(_last_user_text(body["messages"]))
        prompt_tokens, completion_tokens = 9 + len(body["messages"]), len(_words(text))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
        }
        base = {
            "id": f"chatcmpl-fake{self.calls:04d}",
            "created": CREATED,
            "model": OPENAI_MODEL,
            "service_tier": "default",
            "system_fingerprint": "fp_fake",
        }
        if not body.get("stream"):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "refusal": None,
                "annotations": [],
            }
            choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": "stop"}
            payload = {**base, "object": "chat.completion", "choices": [choice], "usage": usage}
            return self._json(200, payload, self._openai_headers())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            payload = {**base, "object": "chat.completion.chunk", "choices": [choice]}
            if include_usage:
                payload["usage"] = None
            return payload

        events: list[tuple[str | None, dict[str, Any] | str]] = [
            (None, chunk({"role": "assistant", "content": "", "refusal": None}, None))
        ]
        events += [(None, chunk({"content": word}, None)) for word in _words(text)]
        events.append((None, chunk({}, "stop")))
        if include_usage:
            events.append(
                (None, {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
            )
        events.append((None, "[DONE]"))
        return self._stream(sse(events), self._openai_headers())

    def _openai_responses(self, request: FakeRequest) -> FakeReply:
        if (denied := self._openai_unauthorized(request)) is not None:
            return denied
        body = request.json()
        prompt = body["input"] if isinstance(body["input"], str) else _last_user_text(body["input"])
        text = reply_text(prompt)
        output_text = {"type": "output_text", "text": text, "annotations": []}
        message = {
            "type": "message",
            "id": "msg_fake0001",
            "status": "completed",
            "role": "assistant",
            "content": [output_text],
        }
        usage = {
            "input_tokens": 21,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens": len(_words(text)),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 21 + len(_words(text)),
        }
        response = {
            "id": f"resp_fake{self.calls:04d}",
            "object": "response",
            "created_at": CREATED,
            "status": "completed",
            "model": OPENAI_MODEL,
            "output": [message],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage,
        }
        if not body.get("stream"):
            return self._json(200, response, self._openai_headers())
        in_progress = {**response, "status": "in_progress", "output": [], "usage": None}
        item = {**message, "status": "in_progress", "content": []}
        part = {**output_text, "text": ""}
        where = {"item_id": "msg_fake0001", "output_index": 0, "content_index": 0}
        steps: list[dict[str, Any]] = [
            {"type": "response.created", "response": in_progress},
            {"type": "response.in_progress", "response": in_progress},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.content_part.added", **where, "part": part},
            *(
                {"type": "response.output_text.delta", **where, "delta": word, "logprobs": []}
                for word in _words(text)
            ),
            {"type": "response.output_text.done", **where, "text": text, "logprobs": []},
            {"type": "response.content_part.done", **where, "part": output_text},
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": response},
        ]
        events: list[tuple[str | None, dict[str, Any] | str]] = [
            (step["type"], {**step, "sequence_number": number}) for number, step in enumerate(steps)
        ]
        return self._stream(sse(events), self._openai_headers())

    # --------------------------------------------------------------------------- anthropic

    def _anthropic_headers(self) -> list[tuple[str, str]]:
        return [
            ("request-id", f"req_fake{self.calls:04d}"),
            ("anthropic-organization-id", "00000000-fake-0000-0000-000000000000"),
            ("anthropic-ratelimit-tokens-remaining", "79000"),
        ]

    def _anthropic_messages(self, request: FakeRequest) -> FakeReply:
        if "x-api-key" not in request.headers or "anthropic-version" not in request.headers:
            error = {"type": "authentication_error", "message": "x-api-key header is required"}
            return self._json(401, {"type": "error", "error": error})
        body = request.json()
        text = reply_text(_last_user_text(body["messages"]))
        usage = {
            "input_tokens": 25,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 11,
            "output_tokens": len(_words(text)),
            "service_tier": "standard",
        }
        message = {
            "id": f"msg_fake{self.calls:04d}",
            "type": "message",
            "role": "assistant",
            "model": ANTHROPIC_MODEL,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
        if not body.get("stream"):
            return self._json(200, message, self._anthropic_headers())
        start = {
            **message,
            "content": [],
            "stop_reason": None,
            "usage": {**usage, "output_tokens": 1},
        }
        events: list[tuple[str | None, dict[str, Any] | str]] = [
            ("message_start", {"type": "message_start", "message": start}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            ("ping", {"type": "ping"}),
        ]
        events += [
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": word},
                },
            )
            for word in _words(text)
        ]
        events += [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": len(_words(text))},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        return self._stream(sse(events), self._anthropic_headers())
