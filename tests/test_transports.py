"""The httpx and httpx2 adapters, exercised with the same tests through both libraries."""

from __future__ import annotations

import gzip
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from llm_cassette import Cassette, UnmatchedRequestError
from tests.fakes import FakeUpstream, HttpLib

URL = "https://api.example.com/v1/chat/completions"
CHUNKS = [
    b'data: {"a": 1}\n\n',
    b'data: {"b"',
    b": 2}\n\n",
    "data: ş".encode()[:-1],
    "ş".encode()[1:] + b"\n\n",
]


class FakeClock:
    """A monotonic clock that advances by a fixed step every time it is read."""

    def __init__(self, step: float) -> None:
        self.now = 100.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def streaming_upstream(
    lib: HttpLib, chunks: list[bytes] = CHUNKS, **headers: str
) -> tuple[Any, list[Any]]:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return lib.http.Response(
            200,
            headers={"content-type": "text/event-stream", **headers},
            stream=lib.stream(chunks),
        )

    return lib.http.MockTransport(handler), seen


def exploding_upstream(lib: HttpLib) -> Any:
    def handler(_: Any) -> Any:
        msg = "the upstream must not be called while replaying"
        raise AssertionError(msg)

    return lib.http.MockTransport(handler)


def sync_client(lib: HttpLib, cassette: Cassette, upstream: Any, **kwargs: Any) -> Any:
    return lib.http.Client(transport=lib.adapter.CassetteTransport(cassette, upstream, **kwargs))


def async_client(lib: HttpLib, cassette: Cassette, upstream: Any, **kwargs: Any) -> Any:
    return lib.http.AsyncClient(
        transport=lib.adapter.AsyncCassetteTransport(cassette, upstream, **kwargs)
    )


def test_both_adapters_expose_the_same_api(lib: HttpLib) -> None:
    assert lib.adapter.__all__ == ["AsyncCassetteTransport", "CassetteTransport"]
    assert issubclass(lib.adapter.CassetteTransport, lib.http.BaseTransport)
    assert issubclass(lib.adapter.AsyncCassetteTransport, lib.http.AsyncBaseTransport)


def test_record_then_replay(lib: HttpLib, cassette_path: Path) -> None:
    upstream, seen = streaming_upstream(lib)
    with (
        Cassette(cassette_path, mode="once") as cassette,
        sync_client(lib, cassette, upstream) as client,
    ):
        with client.stream("POST", URL, json={"model": "m", "stream": True}) as response:
            recorded = list(response.iter_bytes())
        assert response.status_code == 200
    assert recorded == CHUNKS
    assert len(seen) == 1

    with Cassette(cassette_path, mode="none") as cassette:
        client = sync_client(lib, cassette, exploding_upstream(lib))
        with client.stream("POST", URL, json={"stream": True, "model": "m"}) as response:
            replayed = list(response.iter_bytes())
        assert response.headers["content-type"] == "text/event-stream"
    assert replayed == CHUNKS, "chunk boundaries survive, even inside a UTF-8 character"


@pytest.mark.anyio
async def test_record_then_replay_async(lib: HttpLib, cassette_path: Path) -> None:
    upstream, seen = streaming_upstream(lib)
    with Cassette(cassette_path, mode="once") as cassette:
        async with (
            async_client(lib, cassette, upstream) as client,
            client.stream("POST", URL, json={"q": 1}) as response,
        ):
            recorded = [chunk async for chunk in response.aiter_bytes()]
    assert recorded == CHUNKS
    assert len(seen) == 1

    with Cassette(cassette_path, mode="none") as cassette:
        async with async_client(lib, cassette, exploding_upstream(lib)) as client:
            response = await client.post(URL, json={"q": 1})
            assert response.content == b"".join(CHUNKS)
            with pytest.raises(UnmatchedRequestError, match="already played"):
                await client.post(URL, json={"q": 1})


def test_gzip_bodies_are_stored_decoded(lib: HttpLib, cassette_path: Path) -> None:
    payload = {"text": "Merhaba dünya"}
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False).encode())

    def handler(_: Any) -> Any:
        headers = {
            "content-type": "application/json",
            "content-encoding": "gzip",
            "content-length": str(len(compressed)),
        }
        return lib.http.Response(200, headers=headers, stream=lib.stream([compressed]))

    with (
        Cassette(cassette_path, mode="record") as cassette,
        sync_client(lib, cassette, lib.http.MockTransport(handler)) as client,
    ):
        live = client.get(URL)
        assert live.json() == payload
        assert "content-encoding" not in live.headers
    text = cassette_path.read_text(encoding="utf-8")
    assert "Merhaba dünya" in text
    assert "content-encoding" not in text
    assert "content-length" not in text

    with Cassette(cassette_path, mode="none") as cassette:
        assert sync_client(lib, cassette, exploding_upstream(lib)).get(URL).json() == payload


def test_connection_headers_are_not_recorded(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib)
    with Cassette(cassette_path, mode="record") as cassette:
        sync_client(lib, cassette, upstream).post(URL, json={"a": 1}, headers={"x-custom": "kept"})
        (interaction,) = cassette.interactions
    names = {name for name, _ in interaction.request.headers}
    assert "x-custom" in names
    assert names.isdisjoint({"host", "content-length", "connection"})


def test_timing_is_recorded_and_replayed_with_a_scale(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"a", b"b", b"c"])
    with Cassette(cassette_path, mode="record") as cassette:
        client = sync_client(lib, cassette, upstream, clock=FakeClock(step=0.010))
        assert client.get(URL).content == b"abc"
        (interaction,) = cassette.interactions
    assert interaction.response.delay_ms == pytest.approx(10.0)
    assert [chunk.delay_ms for chunk in interaction.response.chunks] == pytest.approx([10.0] * 3)

    sleeps: list[float] = []
    with Cassette(cassette_path, mode="none", timing_scale=0.5) as cassette:
        assert sync_client(lib, cassette, None, sleep=sleeps.append).get(URL).content == b"abc"
    assert sleeps == pytest.approx([0.005] * 4)

    sleeps.clear()
    with Cassette(cassette_path, mode="none") as cassette:
        sync_client(lib, cassette, None, sleep=sleeps.append).get(URL)
    assert sleeps == [], "the default timing_scale of 0 never sleeps"


@pytest.mark.anyio
async def test_async_timing(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"a", b"b"])
    with Cassette(cassette_path, mode="record") as cassette:
        client = async_client(lib, cassette, upstream, clock=FakeClock(step=0.004))
        assert (await client.get(URL)).content == b"ab"

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with Cassette(cassette_path, mode="none", timing_scale=1.0) as cassette:
        client = async_client(lib, cassette, None, sleep=fake_sleep)
        assert (await client.get(URL)).content == b"ab"
    assert sleeps == pytest.approx([0.004] * 3)


@pytest.mark.anyio
async def test_real_timing_with_the_default_async_sleep(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"a", b"b"])
    with Cassette(cassette_path, mode="record") as cassette:
        await async_client(lib, cassette, upstream, clock=FakeClock(step=0.002)).get(URL)
    with Cassette(cassette_path, mode="none", timing_scale=1.0) as cassette:
        client = async_client(lib, cassette, None)
        with anyio.fail_after(5):
            assert (await client.get(URL)).content == b"ab"


def test_upstream_errors_propagate_and_record_nothing(lib: HttpLib, cassette_path: Path) -> None:
    def handler(request: Any) -> Any:
        msg = "connection refused"
        raise lib.http.ConnectError(msg, request=request)

    with Cassette(cassette_path, mode="record") as cassette:
        client = sync_client(lib, cassette, lib.http.MockTransport(handler))
        with pytest.raises(lib.http.ConnectError):
            client.get(URL)
        assert cassette.interactions == ()
    assert not cassette_path.exists()


class _FailingStream:
    """Yields one chunk, then fails like a dropped connection."""

    def __init__(self, error: type[Exception]) -> None:
        self.error = error

    def __iter__(self) -> Iterator[bytes]:
        yield b"partial"
        msg = "connection reset"
        raise self.error(msg)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"partial"
        msg = "connection reset"
        raise self.error(msg)


def _failing_upstream(lib: HttpLib) -> Any:
    base: type = type("S", (_FailingStream, lib.http.SyncByteStream, lib.http.AsyncByteStream), {})

    def handler(_: Any) -> Any:
        return lib.http.Response(200, stream=base(lib.http.ReadError))

    return lib.http.MockTransport(handler)


def test_a_body_that_fails_midway_is_not_recorded(lib: HttpLib, cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        with pytest.raises(lib.http.ReadError):
            sync_client(lib, cassette, _failing_upstream(lib)).get(URL)
        assert cassette.interactions == ()


@pytest.mark.anyio
async def test_a_body_that_fails_midway_is_not_recorded_async(
    lib: HttpLib, cassette_path: Path
) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        with pytest.raises(lib.http.ReadError):
            await async_client(lib, cassette, _failing_upstream(lib)).get(URL)
        assert cassette.interactions == ()


def test_closing_a_stream_early_records_what_was_read(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"one", b"two", b"three"])
    with Cassette(cassette_path, mode="record") as cassette:
        client = sync_client(lib, cassette, upstream)
        with client.stream("GET", URL) as response:
            iterator = response.iter_bytes()
            assert next(iterator) == b"one"
        assert response.is_closed
        (interaction,) = cassette.interactions
    assert [chunk.data for chunk in interaction.response.chunks] == [b"one"]


@pytest.mark.anyio
async def test_closing_an_async_stream_early_records_what_was_read(
    lib: HttpLib, cassette_path: Path
) -> None:
    upstream, _ = streaming_upstream(lib, [b"one", b"two", b"three"])
    with Cassette(cassette_path, mode="record") as cassette:
        async with (
            async_client(lib, cassette, upstream) as client,
            client.stream("GET", URL) as response,
        ):
            async for chunk in response.aiter_bytes():
                assert chunk == b"one"
                break
        (interaction,) = cassette.interactions
    assert [chunk.data for chunk in interaction.response.chunks] == [b"one"]


def test_unmatched_requests_raise_through_the_client(lib: HttpLib, cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="none") as cassette:
        client = sync_client(lib, cassette, exploding_upstream(lib))
        with pytest.raises(UnmatchedRequestError, match="no recorded interaction matches GET"):
            client.get(URL)
        assert len(cassette.unmatched) == 1


def test_transports_close_what_they_wrap(lib: HttpLib, cassette_path: Path) -> None:
    closed: list[str] = []

    class Upstream(lib.http.BaseTransport):  # type: ignore[name-defined, misc]
        def close(self) -> None:
            closed.append("sync")

    class AsyncUpstream(lib.http.AsyncBaseTransport):  # type: ignore[name-defined, misc]
        async def aclose(self) -> None:
            closed.append("async")

    cassette = Cassette(cassette_path, mode="none")
    transport = lib.adapter.CassetteTransport(cassette, Upstream())
    assert transport.cassette is cassette
    transport.close()
    async_transport = lib.adapter.AsyncCassetteTransport(cassette, AsyncUpstream())
    assert async_transport.cassette is cassette
    anyio.run(async_transport.aclose)
    assert closed == ["sync", "async"]
    # Without a wrapped transport there is nothing to close, and replaying never creates one.
    lib.adapter.CassetteTransport(cassette).close()
    anyio.run(lib.adapter.AsyncCassetteTransport(cassette).aclose)


def test_a_default_upstream_is_created_lazily_when_recording(
    lib: HttpLib, cassette_path: Path
) -> None:
    transport = lib.adapter.CassetteTransport(Cassette(cassette_path, mode="record"))
    assert transport._transport is None
    assert isinstance(transport._upstream(), lib.http.HTTPTransport)
    transport.close()
    async_transport = lib.adapter.AsyncCassetteTransport(Cassette(cassette_path, mode="record"))
    assert isinstance(async_transport._upstream(), lib.http.AsyncHTTPTransport)
    anyio.run(async_transport.aclose)


@pytest.mark.anyio
async def test_concurrent_async_requests_get_their_own_answers(
    cassette_path: Path, upstream: FakeUpstream, lib: HttpLib
) -> None:
    prompts = [f"prompt {n}" for n in range(20)]

    async def ask(client: Any, prompt: str) -> str:
        body = {"model": "m", "messages": [{"role": "user", "content": prompt}]}
        response = await client.post(URL, json=body, headers={"authorization": "Bearer sk-fake"})
        content: str = response.json()["choices"][0]["message"]["content"]
        return content

    async def ask_all(client: Any) -> list[str]:
        answers: dict[str, str] = {}

        async def one(prompt: str) -> None:
            answers[prompt] = await ask(client, prompt)

        async with anyio.create_task_group() as group:
            for prompt in prompts:
                group.start_soon(one, prompt)
        return [answers[prompt] for prompt in prompts]

    with Cassette(cassette_path, mode="record") as cassette:
        recorded = await ask_all(async_client(lib, cassette, upstream.transport(lib)))
    with Cassette(cassette_path, mode="none") as cassette:
        replayed = await ask_all(async_client(lib, cassette, exploding_upstream(lib)))
    assert recorded == replayed == [f"Merhaba! You said: {p}" for p in prompts]
    assert upstream.calls == 20


def test_streams_are_single_pass_like_network_streams(lib: HttpLib, cassette_path: Path) -> None:
    """Iterating ``response.stream`` again (SDKs drain it) continues instead of failing."""
    upstream, _ = streaming_upstream(lib, [b"a", b"b"])
    with Cassette(cassette_path, mode="record") as cassette:
        with sync_client(lib, cassette, upstream).stream("GET", URL) as response:
            assert list(response.iter_bytes()) == [b"a", b"b"]
            assert list(response.stream) == []
        assert len(cassette.interactions) == 1, "a second pass must not abandon the recording"
    with (
        Cassette(cassette_path, mode="none") as cassette,
        sync_client(lib, cassette, None).stream("GET", URL) as response,
    ):
        assert list(response.iter_bytes()) == [b"a", b"b"]
        assert list(response.stream) == []


@pytest.mark.anyio
async def test_async_streams_are_single_pass(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"a", b"b"])
    with Cassette(cassette_path, mode="record") as cassette:
        async with async_client(lib, cassette, upstream).stream("GET", URL) as response:
            assert [c async for c in response.aiter_bytes()] == [b"a", b"b"]
            assert [c async for c in response.stream] == []
        assert len(cassette.interactions) == 1
    with Cassette(cassette_path, mode="none") as cassette:
        async with async_client(lib, cassette, None).stream("GET", URL) as response:
            assert [c async for c in response.aiter_bytes()] == [b"a", b"b"]
            assert [c async for c in response.stream] == []


@pytest.mark.anyio
async def test_async_upstream_errors_record_nothing(lib: HttpLib, cassette_path: Path) -> None:
    def handler(request: Any) -> Any:
        msg = "connection refused"
        raise lib.http.ConnectError(msg, request=request)

    with Cassette(cassette_path, mode="record") as cassette:
        with pytest.raises(lib.http.ConnectError):
            await async_client(lib, cassette, lib.http.MockTransport(handler)).get(URL)
        assert cassette.interactions == ()


def test_a_stream_closed_before_reading_records_an_empty_body(
    lib: HttpLib, cassette_path: Path
) -> None:
    upstream, _ = streaming_upstream(lib, [b"never read"])
    with Cassette(cassette_path, mode="record") as cassette:
        with sync_client(lib, cassette, upstream).stream("GET", URL) as response:
            assert response.status_code == 200
        (interaction,) = cassette.interactions
    assert interaction.response.chunks == ()


@pytest.mark.anyio
async def test_an_async_stream_closed_before_reading(lib: HttpLib, cassette_path: Path) -> None:
    upstream, _ = streaming_upstream(lib, [b"never read"])
    with Cassette(cassette_path, mode="record") as cassette:
        async with async_client(lib, cassette, upstream).stream("GET", URL) as response:
            assert response.status_code == 200
        (interaction,) = cassette.interactions
    assert interaction.response.chunks == ()
