"""Cassette transports for `httpx <https://www.python-httpx.org>`_ (0.x).

Use this module with HTTP clients built on the original httpx, including ``openai`` before 3.0,
``anthropic`` before 1.0, and ``openai>=3``, which also accepts an ``httpx.Client``::

    import httpx
    from llm_cassette import Cassette
    from llm_cassette.httpx import CassetteTransport

    cassette = Cassette("tests/cassettes/chat.yaml")
    client = httpx.Client(transport=CassetteTransport(cassette))

This module mirrors :mod:`llm_cassette.httpx2` line for line; the two are kept separate so each
type-checks against its own library.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Iterator

import anyio
import httpx

from llm_cassette._cassette import Cassette, Recording
from llm_cassette._model import Interaction
from llm_cassette._transport import (
    ReplayPlan,
    Stopwatch,
    decoded_body_headers,
    passthrough_extensions,
    snapshot_request,
    text_headers,
    wire_headers,
)

__all__ = ["AsyncCassetteTransport", "CassetteTransport"]


# Every stream below is single-pass, like a network stream: iterating it again continues where
# the previous iteration stopped. SDKs rely on this when they drain a response after the last
# event (openai's async client does), so a second pass must neither fail nor start over.


class _ReplayStream(httpx.SyncByteStream):
    def __init__(
        self, plan: ReplayPlan, sleep: Callable[[float], None], request: httpx.Request
    ) -> None:
        self._chunks = iter(plan.chunks)
        self._truncated = plan.truncated
        self._sleep = sleep
        self._request = request

    def __iter__(self) -> Iterator[bytes]:
        for data, delay in self._chunks:
            if delay > 0:
                self._sleep(delay)
            yield data
        if self._truncated is not None:
            raise httpx.ReadError(self._truncated, request=self._request)


class _AsyncReplayStream(httpx.AsyncByteStream):
    def __init__(
        self,
        plan: ReplayPlan,
        sleep: Callable[[float], Awaitable[None]],
        request: httpx.Request,
    ) -> None:
        self._chunks = iter(plan.chunks)
        self._truncated = plan.truncated
        self._sleep = sleep
        self._request = request

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        for data, delay in self._chunks:
            if delay > 0:
                await self._sleep(delay)
            yield data
        if self._truncated is not None:
            raise httpx.ReadError(self._truncated, request=self._request)


# A recording is finished only when the upstream body is exhausted while the client reads it. A
# client that closes the response first (or is interrupted by Ctrl-C or a cancellation) leaves
# it incomplete, so it replays with a read error past the recorded end instead of passing for a
# whole response. An exception from the upstream itself abandons the recording.


class _RecordingStream(httpx.SyncByteStream):
    def __init__(self, upstream: httpx.Response, recording: Recording, watch: Stopwatch) -> None:
        self._upstream = upstream
        self._recording = recording
        self._watch = watch
        self._chunks: Iterator[bytes] | None = None
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        if self._chunks is None:
            self._chunks = self._upstream.iter_bytes()
        try:
            for chunk in self._chunks:
                self._recording.add_chunk(chunk, delay_ms=self._watch.lap_ms())
                yield chunk
        except Exception:
            # A body that failed half-way would replay as a silently truncated success.
            self._recording.abandon()
            raise
        if not self._closed:
            self._recording.finish()

    def close(self) -> None:
        self._closed = True
        if isinstance(self._chunks, Generator):
            self._chunks.close()
        self._upstream.close()


class _AsyncRecordingStream(httpx.AsyncByteStream):
    def __init__(self, upstream: httpx.Response, recording: Recording, watch: Stopwatch) -> None:
        self._upstream = upstream
        self._recording = recording
        self._watch = watch
        self._chunks: AsyncIterator[bytes] | None = None
        self._closed = False

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        if self._chunks is None:
            self._chunks = self._upstream.aiter_bytes()
        try:
            async for chunk in self._chunks:
                self._recording.add_chunk(chunk, delay_ms=self._watch.lap_ms())
                yield chunk
        except Exception:
            self._recording.abandon()
            raise
        if not self._closed:
            self._recording.finish()

    async def aclose(self) -> None:
        self._closed = True
        if isinstance(self._chunks, AsyncGenerator):
            await self._chunks.aclose()
        await self._upstream.aclose()


class CassetteTransport(httpx.BaseTransport):
    """An httpx transport that replays requests from a cassette or records them into it.

    In recording mode, requests are forwarded to ``transport`` (an ``httpx.HTTPTransport`` by
    default, created on first use) and the decoded response is recorded chunk by chunk while it
    streams to the client. Otherwise responses come from the cassette and ``transport`` is never
    used. Closing this transport closes the wrapped one.

    Args:
        cassette: Where interactions are read from and recorded into.
        transport: The real transport to record through.
        clock: Monotonic clock in seconds, used to measure recorded delays.
        sleep: Called with the (scaled) delay in seconds before replayed chunks.
    """

    def __init__(
        self,
        cassette: Cassette,
        transport: httpx.BaseTransport | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wrap ``transport`` (or a default ``httpx.HTTPTransport``) with ``cassette``."""
        self._cassette = cassette
        self._transport = transport
        self._clock = clock
        self._sleep = sleep

    @property
    def cassette(self) -> Cassette:
        """The cassette this transport reads from and records into."""
        return self._cassette

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Replay a recorded response for ``request``, or forward and record it."""
        outcome = self._cassette.resolve(
            snapshot_request(
                request.method, str(request.url), text_headers(request.headers.raw), request.read()
            )
        )
        if isinstance(outcome, Interaction):
            plan = ReplayPlan.build(
                outcome, self._cassette.timing_scale, method=request.method, url=str(request.url)
            )
            if plan.headers_delay > 0:
                self._sleep(plan.headers_delay)
            return httpx.Response(
                plan.status,
                headers=plan.headers,
                stream=_ReplayStream(plan, self._sleep, request),
            )
        watch = Stopwatch(self._clock)
        try:
            upstream = self._upstream().handle_request(request)
        except BaseException:
            outcome.abandon()
            raise
        headers = decoded_body_headers(text_headers(upstream.headers.raw))
        outcome.start(upstream.status_code, headers, delay_ms=watch.lap_ms())
        return httpx.Response(
            upstream.status_code,
            headers=wire_headers(headers),
            stream=_RecordingStream(upstream, outcome, watch),
            extensions=passthrough_extensions(upstream.extensions),
        )

    def close(self) -> None:
        """Close the wrapped transport, if one was created or given."""
        if self._transport is not None:
            self._transport.close()

    def _upstream(self) -> httpx.BaseTransport:
        if self._transport is None:
            self._transport = httpx.HTTPTransport()
        return self._transport


class AsyncCassetteTransport(httpx.AsyncBaseTransport):
    """The asynchronous counterpart of :class:`CassetteTransport`.

    Args:
        cassette: Where interactions are read from and recorded into.
        transport: The real transport to record through; defaults to
            ``httpx.AsyncHTTPTransport``, created on first use.
        clock: Monotonic clock in seconds, used to measure recorded delays.
        sleep: Awaited with the (scaled) delay in seconds before replayed chunks; defaults to
            :func:`anyio.sleep`, so asyncio and trio both work.
    """

    def __init__(
        self,
        cassette: Cassette,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        """Wrap ``transport`` (or a default ``httpx.AsyncHTTPTransport``) with ``cassette``."""
        self._cassette = cassette
        self._transport = transport
        self._clock = clock
        self._sleep = sleep

    @property
    def cassette(self) -> Cassette:
        """The cassette this transport reads from and records into."""
        return self._cassette

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Replay a recorded response for ``request``, or forward and record it."""
        outcome = self._cassette.resolve(
            snapshot_request(
                request.method,
                str(request.url),
                text_headers(request.headers.raw),
                await request.aread(),
            )
        )
        if isinstance(outcome, Interaction):
            plan = ReplayPlan.build(
                outcome, self._cassette.timing_scale, method=request.method, url=str(request.url)
            )
            if plan.headers_delay > 0:
                await self._sleep(plan.headers_delay)
            return httpx.Response(
                plan.status,
                headers=plan.headers,
                stream=_AsyncReplayStream(plan, self._sleep, request),
            )
        watch = Stopwatch(self._clock)
        try:
            upstream = await self._upstream().handle_async_request(request)
        except BaseException:
            outcome.abandon()
            raise
        headers = decoded_body_headers(text_headers(upstream.headers.raw))
        outcome.start(upstream.status_code, headers, delay_ms=watch.lap_ms())
        return httpx.Response(
            upstream.status_code,
            headers=wire_headers(headers),
            stream=_AsyncRecordingStream(upstream, outcome, watch),
            extensions=passthrough_extensions(upstream.extensions),
        )

    async def aclose(self) -> None:
        """Close the wrapped transport, if one was created or given."""
        if self._transport is not None:
            await self._transport.aclose()

    def _upstream(self) -> httpx.AsyncBaseTransport:
        if self._transport is None:
            self._transport = httpx.AsyncHTTPTransport()
        return self._transport
