"""Token usage parsed from recorded LLM responses, with optional cost estimates.

Usage is read from the response payload, so it works for any endpoint that speaks one of these
shapes (including OpenAI-compatible proxies):

* OpenAI Chat Completions: the ``usage`` object of ``chat.completion`` and of the final
  ``chat.completion.chunk`` when streaming with ``stream_options={"include_usage": True}``.
* OpenAI Responses: ``response.usage``, and the ``response.completed`` (or ``.incomplete`` /
  ``.failed``) event when streaming.
* Anthropic Messages: ``message.usage``, and when streaming the ``message_start`` usage updated
  by the cumulative counts in ``message_delta`` events.

llm-cassette ships no prices. Cost estimates come only from a :class:`PriceTable` you supply.
"""

from __future__ import annotations

import fnmatch
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from llm_cassette._json import JSONValue, parse_document
from llm_cassette._model import Interaction, RecordedResponse
from llm_cassette._sse import parse_sse

__all__ = ["Price", "PriceTable", "Usage", "usage_of"]

_MILLION = Decimal(1_000_000)
_RESPONSES_TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)

Provider = Literal["openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts of one API call, normalized across providers.

    Providers count cached input differently (OpenAI's ``prompt_tokens`` include cached tokens,
    Anthropic's ``input_tokens`` exclude them), so the counts are split into disjoint buckets
    that can each be priced.

    Attributes:
        provider: ``"openai"`` or ``"anthropic"``, from the payload shape.
        model: The model the response reports, or the requested model if it reports none.
        requested_model: The ``model`` field of the request body, if any.
        input_tokens: Input tokens billed at the base input rate (cache reads/writes excluded).
        cache_read_tokens: Input tokens served from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache (Anthropic).
        output_tokens: Output tokens, including any reasoning tokens.
    """

    provider: Provider
    model: str | None
    requested_model: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """All input tokens: base, cache reads and cache writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def _count(source: Mapping[str, JSONValue], key: str) -> int:
    value = source.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _object(value: JSONValue) -> Mapping[str, JSONValue] | None:
    return value if isinstance(value, dict) else None


def _string(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


@dataclass
class _Accumulator:
    provider: Provider | None = None
    model: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def _set(
        self,
        provider: Provider,
        model: JSONValue,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.provider = provider
        self.model = _string(model) or self.model
        self.input_tokens = input_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.output_tokens = output_tokens

    def _openai_chat(
        self, payload: Mapping[str, JSONValue], usage: Mapping[str, JSONValue]
    ) -> None:
        details = _object(usage.get("prompt_tokens_details")) or {}
        cached = _count(details, "cached_tokens")
        self._set(
            "openai",
            payload.get("model"),
            input_tokens=max(0, _count(usage, "prompt_tokens") - cached),
            cache_read_tokens=cached,
            output_tokens=_count(usage, "completion_tokens"),
        )

    def _openai_response(self, response: Mapping[str, JSONValue]) -> None:
        usage = _object(response.get("usage"))
        if usage is None:
            return
        details = _object(usage.get("input_tokens_details")) or {}
        cached = _count(details, "cached_tokens")
        self._set(
            "openai",
            response.get("model"),
            input_tokens=max(0, _count(usage, "input_tokens") - cached),
            cache_read_tokens=cached,
            output_tokens=_count(usage, "output_tokens"),
        )

    def _anthropic(self, message: Mapping[str, JSONValue]) -> None:
        usage = _object(message.get("usage"))
        if usage is None:
            return
        self._set(
            "anthropic",
            message.get("model"),
            input_tokens=_count(usage, "input_tokens"),
            cache_read_tokens=_count(usage, "cache_read_input_tokens"),
            cache_write_tokens=_count(usage, "cache_creation_input_tokens"),
            output_tokens=_count(usage, "output_tokens"),
        )

    def _anthropic_delta(self, usage: Mapping[str, JSONValue]) -> None:
        # message_delta counts are cumulative and override the message_start values. A field
        # that is missing or null keeps its earlier value, as the anthropic SDK's accumulator does.
        self.provider = self.provider or "anthropic"
        if usage.get("input_tokens") is not None:
            self.input_tokens = _count(usage, "input_tokens")
        if usage.get("cache_read_input_tokens") is not None:
            self.cache_read_tokens = _count(usage, "cache_read_input_tokens")
        if usage.get("cache_creation_input_tokens") is not None:
            self.cache_write_tokens = _count(usage, "cache_creation_input_tokens")
        if usage.get("output_tokens") is not None:
            self.output_tokens = _count(usage, "output_tokens")

    def feed(self, payload: Mapping[str, JSONValue]) -> None:
        kind, event = payload.get("object"), payload.get("type")
        usage = _object(payload.get("usage"))
        if kind in {"chat.completion", "chat.completion.chunk"} and usage is not None:
            self._openai_chat(payload, usage)
        elif kind == "response":
            self._openai_response(payload)
        elif event in _RESPONSES_TERMINAL_EVENTS:
            response = _object(payload.get("response"))
            if response is not None:
                self._openai_response(response)
        elif event == "message":
            self._anthropic(payload)
        elif event == "message_start":
            message = _object(payload.get("message"))
            if message is not None:
                self._anthropic(message)
        elif event == "message_delta" and usage is not None:
            self._anthropic_delta(usage)


def _payloads(response: RecordedResponse) -> Iterable[Mapping[str, JSONValue]]:
    body = response.body
    content_type = (response.header("content-type") or "").lower()
    if "text/event-stream" in content_type:
        for event in parse_sse(body.decode("utf-8", errors="replace")):
            document = parse_document(event.data.encode("utf-8"))
            if isinstance(document, dict):
                yield document
        return
    document = parse_document(body)
    if isinstance(document, dict):
        yield document


def usage_of(interaction: Interaction) -> Usage | None:
    """Parse the token usage of ``interaction``, or return ``None`` if it reports none."""
    accumulator = _Accumulator()
    for payload in _payloads(interaction.response):
        accumulator.feed(payload)
    if accumulator.provider is None:
        return None
    request_body = interaction.request.json()
    requested = _string(request_body.get("model")) if isinstance(request_body, dict) else None
    return Usage(
        provider=accumulator.provider,
        model=accumulator.model or requested,
        requested_model=requested,
        input_tokens=accumulator.input_tokens,
        cache_read_tokens=accumulator.cache_read_tokens,
        cache_write_tokens=accumulator.cache_write_tokens,
        output_tokens=accumulator.output_tokens,
    )


# ------------------------------------------------------------------------------------- pricing


@dataclass(frozen=True, slots=True)
class Price:
    """Prices of one model in USD per million tokens.

    Attributes:
        input: Base input tokens.
        output: Output tokens.
        cache_read: Cache reads; ``None`` prices them at ``input``.
        cache_write: Cache writes; ``None`` prices them at ``input``.
    """

    input: Decimal
    output: Decimal
    cache_read: Decimal | None = None
    cache_write: Decimal | None = None

    def cost(self, usage: Usage) -> Decimal:
        """The estimated cost of ``usage`` in USD."""
        cache_read = self.input if self.cache_read is None else self.cache_read
        cache_write = self.input if self.cache_write is None else self.cache_write
        total = (
            usage.input_tokens * self.input
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
            + usage.output_tokens * self.output
        )
        return total / _MILLION


_PRICE_KEYS = frozenset({"input", "output", "cache_read", "cache_write"})


def _decimal(value: object, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        msg = f"{where}: expected a number, got {value!r}"
        raise ValueError(msg)  # noqa: TRY004 - a malformed file is a value error for callers
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        msg = f"{where}: expected a non-negative number, got {value!r}"
        raise ValueError(msg)
    return number


class PriceTable:
    """Model prices supplied by you, used to estimate the cost of recorded usage.

    Keys are exact model names or :mod:`fnmatch` patterns such as ``"claude-sonnet-4*"``. A
    lookup tries an exact match first and then the patterns in table order, for the reported
    model and then the requested model.
    """

    def __init__(self, prices: Mapping[str, Price]) -> None:
        """Create a table from a mapping of model name (or pattern) to :class:`Price`."""
        self._prices = dict(prices)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> PriceTable:
        """Load a TOML price table.

        The file has one table per model, with prices in USD per million tokens::

            [models."gpt-4o-mini"]
            input = 1.00
            output = 2.00
            cache_read = 0.50     # optional, defaults to input

        Raises:
            ValueError: If the file is not valid TOML or does not have this shape.
        """
        source = Path(path)
        try:
            document = tomllib.loads(source.read_text(encoding="utf-8"), parse_float=Decimal)
        except tomllib.TOMLDecodeError as error:
            msg = f"{source}: invalid TOML: {error}"
            raise ValueError(msg) from None
        models = document.get("models")
        if not isinstance(models, dict) or set(document) - {"models"}:
            msg = f"{source}: expected only a [models] table"
            raise ValueError(msg)
        prices: dict[str, Price] = {}
        for name, entry in models.items():
            where = f"{source}: models.{name}"
            if not isinstance(entry, dict) or not {"input", "output"} <= set(entry):
                msg = f"{where}: expected a table with at least input and output"
                raise ValueError(msg)
            unknown = sorted(set(entry) - _PRICE_KEYS)
            if unknown:
                msg = f"{where}: unknown key(s) {', '.join(unknown)}"
                raise ValueError(msg)
            prices[name] = Price(
                input=_decimal(entry["input"], f"{where}.input"),
                output=_decimal(entry["output"], f"{where}.output"),
                cache_read=_decimal(entry["cache_read"], f"{where}.cache_read")
                if "cache_read" in entry
                else None,
                cache_write=_decimal(entry["cache_write"], f"{where}.cache_write")
                if "cache_write" in entry
                else None,
            )
        return cls(prices)

    def __len__(self) -> int:
        """The number of entries in the table."""
        return len(self._prices)

    def lookup(self, *models: str | None) -> Price | None:
        """Return the price of the first of ``models`` that the table knows, or ``None``."""
        candidates = [model for model in models if model]
        for model in candidates:
            if model in self._prices:
                return self._prices[model]
        for model in candidates:
            for pattern, price in self._prices.items():
                if fnmatch.fnmatchcase(model, pattern):
                    return price
        return None

    def cost(self, usage: Usage) -> Decimal | None:
        """The estimated cost of ``usage`` in USD, or ``None`` if its model has no price."""
        price = self.lookup(usage.model, usage.requested_model)
        return None if price is None else price.cost(usage)
