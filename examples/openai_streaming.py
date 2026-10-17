"""Record and replay a streamed OpenAI chat completion with the official ``openai`` SDK.

Run it with ``uv run python examples/openai_streaming.py``. The upstream is a fake that speaks
the Chat Completions streaming format (``chat.completion.chunk`` events, a final usage chunk
because of ``stream_options={"include_usage": True}``, then ``[DONE]``). Its data is synthetic.
To record against the real API instead, drop ``transport=`` and set ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2

from llm_cassette import Cassette, Matching
from llm_cassette.clients import openai_client
from llm_cassette.usage import usage_of

WORDS = ["Kısa", " ve", " net", "."]


def _chunk(delta: dict[str, Any], *, finish: str | None = None) -> dict[str, Any]:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return {
        "id": "chatcmpl-example",
        "object": "chat.completion.chunk",
        "created": 1_760_000_000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [choice],
        "usage": None,
    }


def _events() -> Iterator[bytes]:
    usage = {"prompt_tokens": 14, "completion_tokens": 4, "total_tokens": 18}
    payloads = [_chunk({"role": "assistant", "content": ""})]
    payloads += [_chunk({"content": word}) for word in WORDS]
    payloads.append(_chunk({}, finish="stop"))
    payloads.append({**_chunk({}), "choices": [], "usage": usage})
    for payload in payloads:
        yield f"data: {json.dumps(payload)}\n\n".encode()
    yield b"data: [DONE]\n\n"


class _Stream(httpx2.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield from _events()


def fake_openai(request: httpx2.Request) -> httpx2.Response:
    assert request.headers["authorization"].startswith("Bearer ")
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Stream())


def ask(cassette: Cassette) -> str:
    client = openai_client(
        cassette, transport=httpx2.MockTransport(fake_openai), api_key="sk-example-fake"
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Bir cümleyle özetle."}],
        stream=True,
        stream_options={"include_usage": True},
        metadata={"run_id": "differs-every-run"},
    )
    return "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)


def main() -> None:
    matching = Matching(ignore_paths=["metadata.run_id"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "openai_streaming.yaml"
        with Cassette(path, mode="record", matching=matching) as cassette:
            print("recorded:", ask(cassette))
        with Cassette(path, mode="none", matching=matching) as cassette:
            print("replayed:", ask(cassette))
            (interaction,) = cassette.played
        usage = usage_of(interaction)
        assert usage is not None
        print(
            f"usage from the recorded stream: model={usage.model} "
            f"input={usage.total_input_tokens} output={usage.output_tokens}"
        )
        print(f"{len(interaction.response.chunks)} chunks replayed with their original boundaries")


if __name__ == "__main__":
    main()
