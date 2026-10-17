"""The pytest plugin: the ``llm_cassette`` marker and fixture, options and the usage report.

Installing llm-cassette registers this module through the ``pytest11`` entry point.

Options, from lowest to highest precedence:

1. ini options ``llm_record_mode``, ``llm_cassette_dir``, ``llm_cassette_format`` and
   ``llm_secret_env``;
2. an ``llm_cassette_config`` fixture override returning a :class:`CassetteConfig`;
3. ``@pytest.mark.llm_cassette(...)`` markers (module, then class, then function);
4. ``--llm-record-mode`` on the command line, which always wins so that CI can force ``none``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import pytest

from llm_cassette._cassette import Cassette, RecordMode
from llm_cassette._matching import Matching
from llm_cassette._model import Interaction
from llm_cassette._redaction import Redaction
from llm_cassette.usage import PriceTable, Usage, usage_of

__all__ = ["CassetteConfig"]


class CassetteConfig(TypedDict, total=False):
    """Defaults for every ``llm_cassette`` fixture, returned by an ``llm_cassette_config`` override.

    Example::

        @pytest.fixture
        def llm_cassette_config() -> CassetteConfig:
            return {"matching": Matching(ignore_paths=["metadata.user_id"])}
    """

    mode: RecordMode | str
    matching: Matching
    redaction: Redaction
    timing_scale: float
    allow_repeats: bool


_CONFIG_KEYS = frozenset(CassetteConfig.__optional_keys__)
_MARKER_KEYS = _CONFIG_KEYS | {"name", "ignore_paths"}
_SUFFIXES = {"yaml": ".yaml", "json": ".json"}
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.\-\[\]]+")
_MAX_FILENAME = 120
_MIN_SECRET_LENGTH = 8

_PHASE_REPORTS = pytest.StashKey[dict[str, pytest.TestReport]]()
_T = TypeVar("_T")  # PEP 695 syntax would need Python 3.12


# ------------------------------------------------------------------------------ configuration


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the command-line and ini options."""
    group = parser.getgroup("llm-cassette", "record and replay LLM API calls")
    group.addoption(
        "--llm-record-mode",
        choices=[mode.value for mode in RecordMode],
        default=None,
        help="record mode for every llm_cassette fixture; overrides markers and ini options",
    )
    group.addoption(
        "--llm-usage-report",
        action="store_true",
        default=False,
        help="print a per-test token usage summary at the end of the session",
    )
    group.addoption(
        "--llm-price-table",
        metavar="PATH",
        default=None,
        help="TOML file of model prices (USD per million tokens) for cost estimates in the "
        "usage report; implies --llm-usage-report",
    )
    parser.addini(
        "llm_record_mode", "default record mode: record, once, replay or none", default="once"
    )
    parser.addini(
        "llm_cassette_dir",
        "cassette directory, relative to each test file unless absolute",
        default="cassettes",
    )
    parser.addini("llm_cassette_format", "cassette file format: yaml or json", default="yaml")
    parser.addini(
        "llm_secret_env",
        "environment variables whose values must never be written to a cassette",
        type="linelist",
        default=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    )


@dataclass
class _UsageLog:
    enabled: bool
    prices: PriceTable | None
    tests: dict[str, list[Usage]] = field(default_factory=dict)

    def add(self, nodeid: str, interactions: Iterable[Interaction]) -> None:
        usages = [usage for i in interactions if (usage := usage_of(i)) is not None]
        if usages:
            self.tests.setdefault(nodeid, []).extend(usages)


_USAGE_LOG = pytest.StashKey[_UsageLog]()


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker, validate ini options and load the price table."""
    config.addinivalue_line(
        "markers",
        "llm_cassette(name=None, *, mode=None, matching=None, redaction=None, ignore_paths=(), "
        "timing_scale=None, allow_repeats=None): configure the llm_cassette fixture",
    )
    mode = config.getini("llm_record_mode")
    if mode not in {m.value for m in RecordMode}:
        msg = f"llm_record_mode must be one of record, once, replay, none; got {mode!r}"
        raise pytest.UsageError(msg)
    fmt = config.getini("llm_cassette_format")
    if fmt not in _SUFFIXES:
        msg = f"llm_cassette_format must be yaml or json; got {fmt!r}"
        raise pytest.UsageError(msg)
    price_path = config.getoption("llm_price_table")
    prices = None
    if price_path is not None:
        try:
            prices = PriceTable.load(price_path)
        except (OSError, ValueError) as error:
            msg = f"--llm-price-table: {error}"
            raise pytest.UsageError(msg) from None
    enabled = bool(config.getoption("llm_usage_report")) or prices is not None
    config.stash[_USAGE_LOG] = _UsageLog(enabled=enabled, prices=prices)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Remember each phase's report so the fixture can tell whether the test body passed."""
    del call
    report = yield
    item.stash.setdefault(_PHASE_REPORTS, {})[report.when] = report
    return report


# ------------------------------------------------------------------------------------ fixture


@dataclass(frozen=True, slots=True)
class _Options:
    path: Path
    mode: RecordMode
    matching: Matching
    redaction: Redaction
    timing_scale: float
    allow_repeats: bool


def _marker_options(item: pytest.Item) -> dict[str, Any]:
    """Merge every ``llm_cassette`` marker on ``item``, the closest one winning."""
    merged: dict[str, Any] = {}
    for marker in reversed(list(item.iter_markers("llm_cassette"))):
        if len(marker.args) > 1:
            pytest.fail(
                "llm_cassette accepts at most one positional argument (name)", pytrace=False
            )
        options = dict(marker.kwargs)
        if marker.args:
            if "name" in options:
                pytest.fail(
                    "llm_cassette got the name both positionally and by keyword", pytrace=False
                )
            options["name"] = marker.args[0]
        unknown = sorted(set(options) - _MARKER_KEYS)
        if unknown:
            allowed = ", ".join(sorted(_MARKER_KEYS))
            pytest.fail(
                f"llm_cassette got unknown option(s) {', '.join(unknown)}; allowed: {allowed}",
                pytrace=False,
            )
        merged.update(options)
    return merged


def _expect(value: object, kind: type[_T], what: str) -> _T:
    if not isinstance(value, kind):
        pytest.fail(f"{what} must be a {kind.__name__}, got {type(value).__name__}", pytrace=False)
    return value


def _safe_filename(name: str) -> str:
    safe = _UNSAFE_FILENAME_CHARS.sub("_", name).strip("._") or "cassette"
    if len(safe) <= _MAX_FILENAME:
        return safe
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"{safe[:_MAX_FILENAME]}-{digest}"


def cassette_path(item: pytest.Item, *, name: str | None, directory: str, fmt: str) -> Path:
    """Where the cassette of ``item`` lives.

    ``<test file dir>/<directory>/<test module>/<name>.<fmt>``, where ``name`` defaults to the
    test's node id without the module, e.g. ``TestChat.test_stream[gpt-4o-mini]``.
    """
    module = item.path
    base = Path(directory)
    if not base.is_absolute():
        base = module.parent / base
    stem = name if name is not None else item.nodeid.split("::", 1)[-1].replace("::", ".")
    return base / module.stem / f"{_safe_filename(stem)}{_SUFFIXES[fmt]}"


def _secrets_from_env(config: pytest.Config) -> tuple[str, ...]:
    values = (os.environ.get(name, "") for name in config.getini("llm_secret_env"))
    return tuple(value for value in values if len(value) >= _MIN_SECRET_LENGTH)


def _resolve(request: pytest.FixtureRequest, defaults: CassetteConfig) -> _Options:
    config = request.config
    unknown = sorted(set(defaults) - _CONFIG_KEYS)
    if unknown:
        pytest.fail(
            f"llm_cassette_config returned unknown key(s) {', '.join(unknown)}", pytrace=False
        )
    marker = _marker_options(request.node)
    combined: dict[str, Any] = {**defaults, **marker}

    mode = (
        config.getoption("llm_record_mode")
        or combined.get("mode")
        or config.getini("llm_record_mode")
    )
    matching = _expect(combined.get("matching", Matching()), Matching, "matching")
    if "ignore_paths" in marker:
        extra = marker["ignore_paths"]
        if isinstance(extra, str):
            pytest.fail("ignore_paths must be a list of paths, not a string", pytrace=False)
        matching = replace(matching, ignore_paths=(*matching.ignore_paths, *extra))
    redaction = _expect(combined.get("redaction", Redaction()), Redaction, "redaction")
    env_secrets = [s for s in _secrets_from_env(config) if s not in redaction.secrets]
    if env_secrets:
        redaction = replace(redaction, secrets=(*redaction.secrets, *env_secrets))
    name = combined.get("name")
    if name is not None:
        name = _expect(name, str, "name")
    try:
        record_mode = RecordMode(mode)
    except ValueError:
        pytest.fail(f"unknown record mode {mode!r}", pytrace=False)
    return _Options(
        path=cassette_path(
            request.node,
            name=name,
            directory=config.getini("llm_cassette_dir"),
            fmt=config.getini("llm_cassette_format"),
        ),
        mode=record_mode,
        matching=matching,
        redaction=redaction,
        timing_scale=float(combined.get("timing_scale", 0.0)),
        allow_repeats=bool(combined.get("allow_repeats", False)),
    )


@pytest.fixture
def llm_cassette_config() -> CassetteConfig:
    """Project-wide defaults for ``llm_cassette``; override this fixture in a conftest.py."""
    return {}


@pytest.fixture
def llm_cassette(
    request: pytest.FixtureRequest, llm_cassette_config: CassetteConfig
) -> Iterator[Cassette]:
    """A :class:`~llm_cassette.Cassette` for the current test, saved when the test finishes.

    The cassette path is derived from the test's node id (see :func:`cassette_path`). If the
    code under test swallowed an :class:`~llm_cassette.UnmatchedRequestError` (SDKs wrap
    transport errors), the test still fails at teardown.
    """
    options = _resolve(request, llm_cassette_config)
    cassette = Cassette(
        options.path,
        mode=options.mode,
        matching=options.matching,
        redaction=options.redaction,
        timing_scale=options.timing_scale,
        allow_repeats=options.allow_repeats,
    )
    yield cassette
    cassette.save()
    request.config.stash[_USAGE_LOG].add(request.node.nodeid, cassette.played)
    call_report = request.node.stash.get(_PHASE_REPORTS, {}).get("call")
    if cassette.unmatched and call_report is not None and call_report.passed:
        count = len(cassette.unmatched)
        pytest.fail(
            f"{count} request(s) did not match the cassette, although the test passed (the code "
            f"under test probably caught the error):\n\n{cassette.unmatched[0]}",
            pytrace=False,
        )


# ------------------------------------------------------------------------------- usage report


def _cell_cost(cost: Decimal | None) -> str:
    return "n/a" if cost is None else f"{cost:.6f}"


def _summarize(
    label: str, usages: list[Usage], prices: PriceTable | None
) -> tuple[list[str], set[str]]:
    models = sorted({usage.model or "?" for usage in usages})
    row = [
        label,
        ", ".join(models),
        str(len(usages)),
        str(sum(u.total_input_tokens for u in usages)),
        str(sum(u.cache_read_tokens for u in usages)),
        str(sum(u.cache_write_tokens for u in usages)),
        str(sum(u.output_tokens for u in usages)),
    ]
    unpriced: set[str] = set()
    if prices is not None:
        costs = [prices.cost(usage) for usage in usages]
        unpriced = {u.model or "?" for u, c in zip(usages, costs, strict=True) if c is None}
        known = [cost for cost in costs if cost is not None]
        row.append(_cell_cost(sum(known, Decimal(0)) if not unpriced else None))
    return row, unpriced


def render_usage_report(tests: dict[str, list[Usage]], prices: PriceTable | None) -> list[str]:
    """Render the per-test usage table printed by ``--llm-usage-report``."""
    if not tests:
        return ["no LLM usage was recorded or replayed in this session"]
    header = ["test", "model", "calls", "input", "cache read", "cache write", "output"]
    if prices is not None:
        header.append("cost USD")
    rows: list[list[str]] = []
    unpriced: set[str] = set()
    for nodeid in sorted(tests):
        row, missing = _summarize(nodeid, tests[nodeid], prices)
        rows.append(row)
        unpriced |= missing
    total, _ = _summarize("total", [u for usages in tests.values() for u in usages], prices)
    total[1] = ""
    widths = [max(len(r[i]) for r in [header, *rows, total]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        left = [cells[i].ljust(widths[i]) for i in range(2)]
        right = [cells[i].rjust(widths[i]) for i in range(2, len(cells))]
        return "  ".join(left + right).rstrip()

    lines = [line(header), *(line(row) for row in rows), line(total)]
    lines.append(
        "input = all input tokens (cache reads and writes included); token counts come from the "
        "recorded responses"
    )
    if unpriced:
        lines.append(f"n/a: no price in --llm-price-table for {', '.join(sorted(unpriced))}")
    return lines


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    """Print the usage report when ``--llm-usage-report`` or ``--llm-price-table`` is given."""
    log = config.stash.get(_USAGE_LOG, None)
    if log is None or not log.enabled:
        return
    terminalreporter.write_sep("=", "llm-cassette usage")
    for text in render_usage_report(log.tests, log.prices):
        terminalreporter.write_line(text)
