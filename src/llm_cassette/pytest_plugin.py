"""The pytest plugin: the ``llm_cassette`` marker and fixture, options and the usage report.

Installing llm-cassette registers this module through the ``pytest11`` entry point.

Options, from lowest to highest precedence:

1. ini options ``llm_record_mode``, ``llm_cassette_dir``, ``llm_cassette_format`` and
   ``llm_secret_env``;
2. an ``llm_cassette_config`` fixture override returning a :class:`CassetteConfig`;
3. ``@pytest.mark.llm_cassette(...)`` markers (module, then class, then function); an option
   given as ``None`` counts as not given;
4. ``--llm-record-mode`` on the command line, which always wins so that CI can force ``none``.

Under pytest-xdist, each worker sends the usage it saw to the controller, which prints one
report for the whole session.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
import os
import re
from collections.abc import Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict, TypeVar, cast

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
_MARKER_KEYS = _CONFIG_KEYS | {"name", "ignore_paths", "allow_unmatched"}
_SUFFIXES = {"yaml": ".yaml", "json": ".json"}
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.\-\[\]]+")
_MAX_FILENAME = 120
_DIGEST_LENGTH = 8
_MIN_SECRET_LENGTH = 8
_WORKER_USAGE = "llm_cassette_usage"  # key in pytest-xdist's workeroutput

_PHASE_REPORTS = pytest.StashKey[dict[str, pytest.TestReport]]()
# Which test owns each cassette file in this session, keyed by the case-folded path.
_CASSETTE_OWNERS = pytest.StashKey[dict[str, tuple[str, Path]]]()
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

    def export(self) -> dict[str, list[dict[str, object]]]:
        """The log as plain data, which pytest-xdist can send from a worker to the controller."""
        return {
            nodeid: [dataclasses.asdict(usage) for usage in usages]
            for nodeid, usages in self.tests.items()
        }

    def merge(self, exported: Mapping[str, list[dict[str, Any]]]) -> None:
        """Add the usage that a worker exported."""
        for nodeid, usages in exported.items():
            self.tests.setdefault(nodeid, []).extend(Usage(**usage) for usage in usages)


_USAGE_LOG = pytest.StashKey[_UsageLog]()


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker, validate ini options and load the price table."""
    config.addinivalue_line(
        "markers",
        "llm_cassette(name=None, *, mode=None, matching=None, redaction=None, ignore_paths=None, "
        "timing_scale=None, allow_repeats=None, allow_unmatched=None): configure the "
        "llm_cassette fixture",
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


def pytest_sessionfinish(session: pytest.Session) -> None:
    """On a pytest-xdist worker, hand the usage this worker saw to the controller."""
    workeroutput: dict[str, object] | None = getattr(session.config, "workeroutput", None)
    log = session.config.stash.get(_USAGE_LOG, None)
    if workeroutput is not None and log is not None:
        workeroutput[_WORKER_USAGE] = log.export()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: object, error: object) -> None:
    """On the pytest-xdist controller, merge the usage a finished worker sent."""
    del error
    workeroutput: dict[str, object] = getattr(node, "workeroutput", {})
    # pop: xdist calls this hook a second time for a worker that went down with an error.
    exported = workeroutput.pop(_WORKER_USAGE, None)
    config: pytest.Config | None = getattr(node, "config", None)
    log = config.stash.get(_USAGE_LOG, None) if config is not None else None
    if isinstance(exported, dict) and log is not None:
        log.merge(cast("dict[str, list[dict[str, Any]]]", exported))


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
    allow_unmatched: bool


def _marker_options(item: pytest.Item) -> dict[str, Any]:
    """Merge every ``llm_cassette`` marker on ``item``, the closest one winning."""
    merged: dict[str, Any] = {}
    for marker in reversed(list(item.iter_markers("llm_cassette"))):
        if len(marker.args) > 1:
            pytest.fail(
                "llm_cassette accepts at most one positional argument (name)", pytrace=False
            )
        options = {key: value for key, value in marker.kwargs.items() if value is not None}
        if marker.args and marker.args[0] is not None:
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


def _real_number(value: object, what: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        pytest.fail(f"{what} must be a finite number >= 0, got {value!r}", pytrace=False)
    return float(value)


def _safe_filename(name: str) -> str:
    """A portable file name for ``name``, distinct for distinct names.

    Each run of characters outside ``[A-Za-z0-9_.-[]]`` becomes one ``_``, and leading and
    trailing ``.`` and ``_`` are dropped. Whenever that changes the name (or it is too long), a
    short hash of the original is appended, so ``What is 2+2?`` and ``What is 2*2?`` do not end
    up in the same file.
    """
    safe = _UNSAFE_FILENAME_CHARS.sub("_", name).strip("._") or "cassette"
    if safe == name and len(safe) <= _MAX_FILENAME:
        return safe
    digest = hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{safe[:_MAX_FILENAME]}-{digest[:_DIGEST_LENGTH]}"


def cassette_path(item: pytest.Item, *, name: str | None, directory: str, fmt: str) -> Path:
    """Where the cassette of ``item`` lives.

    ``<test file dir>/<directory>/<test module>/<name>.<fmt>``, where ``name`` defaults to the
    test's node id without the module, e.g. ``TestChat.test_stream[gpt-4o-mini]``. An absolute
    ``directory`` mirrors the test file's location below the rootdir, so ``tests/unit/test_x.py``
    and ``tests/e2e/test_x.py`` keep separate folders.
    """
    module = item.path
    base = Path(directory)
    if not base.is_absolute():
        base = module.parent / base
    else:
        # A test file outside the rootdir keeps the bare module folder; _claim catches clashes.
        with contextlib.suppress(ValueError):
            base /= module.parent.relative_to(item.config.rootpath)
    stem = name if name is not None else item.nodeid.split("::", 1)[-1].replace("::", ".")
    return base / module.stem / f"{_safe_filename(stem)}{_SUFFIXES[fmt]}"


def _claim(config: pytest.Config, path: Path, nodeid: str) -> None:
    """Fail unless ``path`` belongs to the test ``nodeid`` alone in this session.

    Two tests sharing a cassette would overwrite each other's recordings (``record``) or replay
    the other test's traffic (``once``). Paths are compared case-insensitively, because the
    cassettes are committed and checked out on case-insensitive file systems too.
    """
    owners = config.stash.setdefault(_CASSETTE_OWNERS, {})
    owner, owned = owners.setdefault(str(path).casefold(), (nodeid, path))
    if owner != nodeid:
        spelled = "" if owned == path else f" (as {owned.name})"
        pytest.fail(
            f"{nodeid} cannot use the cassette {path}: {owner} already uses it{spelled}. A "
            "cassette holds the traffic of one test, so give one of them its own name with "
            "@pytest.mark.llm_cassette(name=...); file names are compared case-insensitively.",
            pytrace=False,
        )


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
    combined: dict[str, Any] = {
        **{key: value for key, value in defaults.items() if value is not None},
        **marker,
    }

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
        timing_scale=_real_number(combined.get("timing_scale", 0.0), "timing_scale"),
        allow_repeats=_expect(combined.get("allow_repeats", False), bool, "allow_repeats"),
        allow_unmatched=_expect(marker.get("allow_unmatched", False), bool, "allow_unmatched"),
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

    The cassette path is derived from the test's node id (see :func:`cassette_path`) and must not
    be shared with another test. If the code under test swallowed an
    :class:`~llm_cassette.UnmatchedRequestError` (SDKs wrap transport errors), the test still
    fails at teardown, unless its marker says ``allow_unmatched=True``.
    """
    options = _resolve(request, llm_cassette_config)
    _claim(request.config, options.path, request.node.nodeid)
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
    if (
        cassette.unmatched
        and not options.allow_unmatched
        and call_report is not None
        and call_report.passed
    ):
        count = len(cassette.unmatched)
        pytest.fail(
            f"{count} request(s) did not match the cassette, although the test passed (the code "
            "under test probably caught the error; mark the test with "
            "@pytest.mark.llm_cassette(allow_unmatched=True) if that is what it checks):"
            f"\n\n{cassette.unmatched[0]}",
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
