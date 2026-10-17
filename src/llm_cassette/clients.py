"""Official OpenAI and Anthropic SDK clients wired to a cassette.

Example::

    from llm_cassette.clients import openai_client

    def test_haiku(llm_cassette):
        client = openai_client(llm_cassette)
        client.chat.completions.create(model="gpt-4o-mini", messages=[...])

The helpers need ``openai>=3`` and ``anthropic>=1``, which are built on httpx2. For older SDKs,
build the client yourself around :class:`llm_cassette.httpx.CassetteTransport`.

Every helper:

* sets ``max_retries=0`` unless you pass it: an unmatched request is not worth retrying, and the
  SDK's backoff would slow the failure down;
* passes a placeholder ``api_key`` when the cassette replays, so replaying needs no credentials
  (when recording, the SDK reads ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` as usual);
* forwards every other keyword argument to the SDK client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx2

from llm_cassette._cassette import Cassette
from llm_cassette.httpx2 import AsyncCassetteTransport, CassetteTransport

if TYPE_CHECKING:
    import anthropic
    import openai

__all__ = [
    "REPLAY_API_KEY",
    "anthropic_client",
    "async_anthropic_client",
    "async_openai_client",
    "openai_client",
]

REPLAY_API_KEY = "llm-cassette-replay"
"""The ``api_key`` the helpers pass while replaying; it never leaves the process."""


def _options(cassette: Cassette, kwargs: dict[str, Any]) -> dict[str, Any]:
    if "http_client" in kwargs:
        msg = "the client is wired to the cassette already; pass transport= to change the upstream"
        raise TypeError(msg)
    options = dict(kwargs)
    options.setdefault("max_retries", 0)
    if not cassette.is_recording:
        options.setdefault("api_key", REPLAY_API_KEY)
    return options


def _sync_http_client(cassette: Cassette, transport: httpx2.BaseTransport | None) -> httpx2.Client:
    return httpx2.Client(transport=CassetteTransport(cassette, transport), follow_redirects=True)


def _async_http_client(
    cassette: Cassette, transport: httpx2.AsyncBaseTransport | None
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=AsyncCassetteTransport(cassette, transport), follow_redirects=True
    )


def openai_client(
    cassette: Cassette, *, transport: httpx2.BaseTransport | None = None, **kwargs: Any
) -> openai.OpenAI:
    """Return an ``openai.OpenAI`` client that replays from or records into ``cassette``.

    Args:
        cassette: The cassette to use.
        transport: The upstream transport used when recording (the network by default).
        **kwargs: Forwarded to ``openai.OpenAI``.
    """
    import openai  # noqa: PLC0415 - optional dependency

    return openai.OpenAI(
        http_client=_sync_http_client(cassette, transport), **_options(cassette, kwargs)
    )


def async_openai_client(
    cassette: Cassette, *, transport: httpx2.AsyncBaseTransport | None = None, **kwargs: Any
) -> openai.AsyncOpenAI:
    """Return an ``openai.AsyncOpenAI`` client that replays from or records into ``cassette``.

    Args:
        cassette: The cassette to use.
        transport: The upstream transport used when recording (the network by default).
        **kwargs: Forwarded to ``openai.AsyncOpenAI``.
    """
    import openai  # noqa: PLC0415 - optional dependency

    return openai.AsyncOpenAI(
        http_client=_async_http_client(cassette, transport), **_options(cassette, kwargs)
    )


def anthropic_client(
    cassette: Cassette, *, transport: httpx2.BaseTransport | None = None, **kwargs: Any
) -> anthropic.Anthropic:
    """Return an ``anthropic.Anthropic`` client that replays from or records into ``cassette``.

    Args:
        cassette: The cassette to use.
        transport: The upstream transport used when recording (the network by default).
        **kwargs: Forwarded to ``anthropic.Anthropic``.
    """
    import anthropic  # noqa: PLC0415 - optional dependency

    return anthropic.Anthropic(
        http_client=_sync_http_client(cassette, transport), **_options(cassette, kwargs)
    )


def async_anthropic_client(
    cassette: Cassette, *, transport: httpx2.AsyncBaseTransport | None = None, **kwargs: Any
) -> anthropic.AsyncAnthropic:
    """Return an ``anthropic.AsyncAnthropic`` client that replays from or records into ``cassette``.

    Args:
        cassette: The cassette to use.
        transport: The upstream transport used when recording (the network by default).
        **kwargs: Forwarded to ``anthropic.AsyncAnthropic``.
    """
    import anthropic  # noqa: PLC0415 - optional dependency

    return anthropic.AsyncAnthropic(
        http_client=_async_http_client(cassette, transport), **_options(cassette, kwargs)
    )
