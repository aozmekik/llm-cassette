"""Measure what replaying from a cassette costs. All data is synthetic.

Run with ``uv run python benchmarks/replay_overhead.py`` (``--quick`` for a smoke run).

Table 1 sends the same request through an ``httpx2.Client`` two ways:

* ``mock``: straight to an ``httpx2.MockTransport``, the cheapest possible upstream;
* ``replay``: through ``CassetteTransport`` in ``none`` mode, answered from a loaded cassette.

The difference is llm-cassette's per-request overhead: snapshotting and redacting the request,
matching it, and streaming the recorded chunks back.

Table 2 times ``Cassette.resolve`` alone against cassettes of growing size, where the matching
interaction is the last one, to show the cost of the linear matching scan.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import httpx2

import llm_cassette
from llm_cassette import Cassette, Interaction, RecordedRequest, Recording
from llm_cassette.httpx2 import CassetteTransport

URL = "https://api.openai.com/v1/chat/completions"


def _request(content: str) -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}


REQUEST = _request("x" * 2_000)


def _sse(count: int) -> list[bytes]:
    event = {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "token"}}]}
    return [f"data: {json.dumps(event)}\n\n".encode()] * count + [b"data: [DONE]\n\n"]


JSON_BODY = [json.dumps({"object": "chat.completion", "choices": [], "pad": "y" * 2_000}).encode()]


class _Stream(httpx2.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _mock(chunks: list[bytes]) -> httpx2.MockTransport:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_Stream(chunks))

    return httpx2.MockTransport(handler)


def _send(client: httpx2.Client) -> None:
    with client.stream("POST", URL, json=REQUEST) as response:
        for _ in response.iter_bytes():
            pass


def _per_call_us(run: Callable[[], object], iterations: int, repeats: int) -> list[float]:
    """Microseconds per call for each repeat, after one discarded warm-up repeat."""
    results = []
    for repeat in range(repeats + 1):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run()
        if repeat:
            results.append((time.perf_counter_ns() - start) / 1_000 / iterations)
    return results


def _cell(samples: list[float]) -> str:
    return f"{statistics.median(samples):.0f} ({min(samples):.0f}-{max(samples):.0f})"


def end_to_end(directory: Path, iterations: int, repeats: int) -> None:
    """Print table 1: a request through a mock transport versus through a replaying cassette."""
    print("| response | mock, us/request | replay, us/request | overhead, us |")
    print("|---|---|---|---|")
    for label, chunks in [("2 KB JSON, 1 chunk", JSON_BODY), ("SSE, 201 chunks", _sse(200))]:
        path = directory / f"{len(chunks)}.yaml"
        with Cassette(path, mode="record") as cassette:
            _send(httpx2.Client(transport=CassetteTransport(cassette, _mock(chunks))))
        mock_client = httpx2.Client(transport=_mock(chunks))
        cassette = Cassette(path, mode="none", allow_repeats=True)
        replay_client = httpx2.Client(transport=CassetteTransport(cassette))
        mock = _per_call_us(partial(_send, mock_client), iterations, repeats)
        replay = _per_call_us(partial(_send, replay_client), iterations, repeats)
        overhead = statistics.median(replay) - statistics.median(mock)
        print(f"| {label} | {_cell(mock)} | {_cell(replay)} | {overhead:.0f} |")


def matching_scan(directory: Path, iterations: int, repeats: int) -> None:
    """Print table 2: ``Cassette.resolve`` against cassettes of growing size."""
    print("| recorded interactions | Cassette.resolve, us/call |")
    print("|---|---|")
    for size in (1, 10, 100, 1000):
        path = directory / f"scan-{size}.yaml"
        with Cassette(path, mode="record") as cassette:
            for index in range(size):
                recording = cassette.resolve(_snapshot(_request(f"prompt {index}")))
                assert isinstance(recording, Recording)
                recording.start(200, [], delay_ms=0)
        replay = Cassette(path, mode="none", allow_repeats=True)
        last = _snapshot(_request(f"prompt {size - 1}"))
        assert isinstance(replay.resolve(last), Interaction)
        samples = _per_call_us(partial(replay.resolve, last), iterations, repeats)
        print(f"| {size} | {_cell(samples)} |")


def _snapshot(body: dict[str, Any]) -> RecordedRequest:
    headers = (("authorization", "Bearer sk-bench"), ("content-type", "application/json"))
    return RecordedRequest("POST", URL, headers, json.dumps(body).encode())


def main() -> None:
    """Parse the arguments and print both tables."""
    parser = argparse.ArgumentParser(description="llm-cassette replay overhead")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="tiny run, for CI smoke tests")
    args = parser.parse_args()
    iterations, repeats = (20, 1) if args.quick else (args.iterations, args.repeats)

    print(
        f"Python {platform.python_version()} ({platform.machine()}, {platform.system()}), "
        f"httpx2 {httpx2.__version__}, llm-cassette {llm_cassette.__version__}; "
        f"{iterations} calls x {repeats} repeats, median (min-max)\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        end_to_end(Path(directory), iterations, repeats)
        print()
        matching_scan(Path(directory), iterations, repeats)


if __name__ == "__main__":
    main()
