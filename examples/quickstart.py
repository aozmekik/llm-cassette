"""Record a streamed response once, then replay it with the same chunk boundaries.

Run it with ``uv run python examples/quickstart.py``. No network access or API key is needed:
the "upstream" is an ``httpx2.MockTransport`` that streams a short, synthetic event stream.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import httpx2

from llm_cassette import Cassette
from llm_cassette.httpx2 import CassetteTransport

URL = "https://api.example.com/v1/chat/completions"
CHUNKS = [
    b'data: {"delta": "Mer',
    b'haba"}\n\n',
    b'data: {"delta": " d\xc3',
    b'\xbcnya"}\n\ndata: [DONE]\n\n',
]


class _Stream(httpx2.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield from CHUNKS


def fake_upstream(request: httpx2.Request) -> httpx2.Response:
    print(f"  upstream called: {request.method} {request.url}")
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Stream())


def stream_once(cassette: Cassette) -> list[bytes]:
    transport = CassetteTransport(cassette, httpx2.MockTransport(fake_upstream))
    with httpx2.Client(transport=transport) as client:
        request = {"model": "gpt-4o-mini", "stream": True}
        headers = {"authorization": "Bearer sk-example-not-a-real-key"}
        with client.stream("POST", URL, json=request, headers=headers) as response:
            return list(response.iter_bytes())


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "quickstart.yaml"

        print("1. mode='once' with no cassette file: the request is recorded")
        with Cassette(path, mode="once") as cassette:
            recorded = stream_once(cassette)

        print("2. mode='none': the response comes from the cassette")
        with Cassette(path, mode="none") as cassette:
            replayed = stream_once(cassette)

        assert replayed == recorded == CHUNKS
        print(f"   same {len(replayed)} chunks, split at the same bytes: {replayed}")
        print(f"3. the cassette ({path.name}):\n")
        print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
