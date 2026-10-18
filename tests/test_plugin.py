"""The pytest plugin, exercised end to end in throwaway pytest sessions (``pytester``)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# In-process pytester sessions drop every module they import, so the next session imports (and
# assertion-rewrites) it again. Importing the heavy ones here keeps them loaded once.
import hypothesis  # noqa: F401
import openai  # noqa: F401
import pytest

import llm_cassette.clients  # noqa: F401
import tests.fakes  # noqa: F401
from llm_cassette import Cassette
from llm_cassette.pytest_plugin import _safe_filename, render_usage_report
from llm_cassette.usage import PriceTable, Usage

CONFTEST = """
import pytest
from llm_cassette.clients import openai_client
from tests.fakes import FAKE_OPENAI_KEY, FakeUpstream


@pytest.fixture
def upstream():
    return FakeUpstream()


@pytest.fixture
def ask(llm_cassette, upstream):
    client = openai_client(llm_cassette, transport=upstream.transport(), api_key=FAKE_OPENAI_KEY)

    def ask(prompt, **extra):
        completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], **extra
        )
        return completion.choices[0].message.content

    return ask
"""

CHAT_TEST = """
def test_chat(ask, upstream, llm_cassette):
    assert ask("hi") == "Merhaba! You said: hi"
    assert upstream.calls == (1 if llm_cassette.is_recording else 0)
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST)
    return pytester


def test_once_records_then_replays(project: pytest.Pytester) -> None:
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(passed=1)
    cassette = project.path / "cassettes" / "test_demo" / "test_chat.yaml"
    assert cassette.exists()
    assert len(Cassette(cassette, mode="none").interactions) == 1
    project.runpytest("--llm-record-mode=none").assert_outcomes(passed=1)
    project.runpytest("--llm-record-mode=replay").assert_outcomes(passed=1)


def test_record_mode_rewrites(project: pytest.Pytester) -> None:
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(passed=1)
    cassette = project.path / "cassettes" / "test_demo" / "test_chat.yaml"

    def without_delays() -> list[str]:
        return [line for line in cassette.read_text().splitlines() if "delay_ms" not in line]

    first = without_delays()
    project.runpytest("--llm-record-mode=record").assert_outcomes(passed=1)
    assert without_delays() == first, "same traffic, same file apart from the measured delays"


def test_cassette_names_follow_the_node_id(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_names="""
        import pytest

        class TestGroup:
            @pytest.mark.parametrize("prompt", ["What is 2+2?", "What is 2*2?", "gpt-4o-mini"])
            def test_param(self, ask, prompt):
                assert ask(prompt).endswith(prompt)

        @pytest.mark.llm_cassette("custom_name")
        def test_named(ask):
            ask("named")
        """
    )
    project.runpytest().assert_outcomes(passed=4)
    project.runpytest("--llm-record-mode=none").assert_outcomes(passed=4)
    names = sorted(p.name for p in (project.path / "cassettes" / "test_names").iterdir())
    assert re.fullmatch(r"TestGroup\.test_param\[What_is_2_2_\]-[0-9a-f]{8}\.yaml", names[0])
    assert re.fullmatch(r"TestGroup\.test_param\[What_is_2_2_\]-[0-9a-f]{8}\.yaml", names[1])
    assert names[0] != names[1], "names that only sanitize alike get different files"
    assert names[2] == "TestGroup.test_param[gpt-4o-mini].yaml", "safe names are kept as they are"
    assert names[3] == "custom_name.yaml"


def test_two_tests_cannot_share_a_cassette(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_names="""
        import pytest

        @pytest.mark.llm_cassette("shared")
        def test_one(ask):
            ask("one")

        @pytest.mark.llm_cassette(name="shared")
        def test_two(ask):
            ask("two")
        """
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            (
                "*test_names.py::test_two cannot use the cassette *shared.yaml: "
                "test_names.py::test_one already uses it. A cassette holds the traffic of one test*"
            )
        ]
    )


def test_cassette_names_are_compared_case_insensitively(project: pytest.Pytester) -> None:
    """test_A and test_a would share one file on macOS and Windows checkouts."""
    project.makepyfile(
        test_names="""
        def test_A(ask):
            ask("upper")

        def test_a(ask):
            ask("lower")
        """
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*test_a cannot use the cassette *test_A already uses it (as*"])


def test_a_rerun_of_the_same_test_keeps_its_cassette(project: pytest.Pytester) -> None:
    """A test may claim its own cassette again, as when pytest-rerunfailures reruns it."""
    project.makepyfile(
        test_names="""
        import pytest
        from llm_cassette.pytest_plugin import _claim

        def test_twice(request, llm_cassette):
            _claim(request.config, llm_cassette.path, request.node.nodeid)
        """
    )
    project.runpytest().assert_outcomes(passed=1)


def test_an_absolute_cassette_dir_mirrors_the_test_tree(
    project: pytest.Pytester, tmp_path_factory: pytest.TempPathFactory
) -> None:
    target = tmp_path_factory.mktemp("recordings")
    project.makeini(f"[pytest]\nllm_cassette_dir = {target}\n")
    for folder in ("unit", "e2e"):
        package = project.mkpydir(f"suite_{folder}")
        (package / "test_chat.py").write_text(
            f"def test_chat(ask):\n    assert ask('{folder}').endswith('{folder}')\n"
        )
    project.runpytest().assert_outcomes(passed=2)
    assert (target / "suite_unit" / "test_chat" / "test_chat.yaml").exists()
    assert (target / "suite_e2e" / "test_chat" / "test_chat.yaml").exists()
    project.runpytest("--llm-record-mode=none").assert_outcomes(passed=2)


def test_cli_mode_overrides_markers(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_demo="""
        import pytest

        @pytest.mark.llm_cassette(mode="none")
        def test_chat(ask):
            ask("hi")
        """
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*UnmatchedRequestError: no recorded interaction matches POST*"])
    project.runpytest("--llm-record-mode=once").assert_outcomes(passed=1)
    project.runpytest().assert_outcomes(passed=1)


def test_replay_mode_needs_the_cassette(project: pytest.Pytester) -> None:
    project.makepyfile(test_demo=CHAT_TEST)
    result = project.runpytest("--llm-record-mode=replay")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*CassetteNotFoundError: cassette *test_chat.yaml does not exist*"]
    )


def test_markers_merge_from_module_to_function(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_demo="""
        import pytest
        from llm_cassette import Matching

        pytestmark = pytest.mark.llm_cassette(matching=Matching(ignore_paths=["temperature"]))

        @pytest.mark.llm_cassette(ignore_paths=["metadata"])
        def test_chat(ask, llm_cassette):
            assert llm_cassette.matching.ignore_paths == ("temperature", "metadata")
            ask("hi", temperature=0.1 if llm_cassette.is_recording else 0.9,
                metadata={"run": "1" if llm_cassette.is_recording else "2"})
        """
    )
    project.runpytest().assert_outcomes(passed=1)
    project.runpytest("--llm-record-mode=none").assert_outcomes(passed=1)


def test_config_fixture_sets_project_defaults(project: pytest.Pytester) -> None:
    project.makeconftest(
        CONFTEST
        + """
from llm_cassette import Redaction


@pytest.fixture
def llm_cassette_config():
    return {"redaction": Redaction(body_paths=["messages.*.content"]), "timing_scale": 0.0}
"""
    )
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(passed=1)
    text = (project.path / "cassettes" / "test_demo" / "test_chat.yaml").read_text()
    assert "content: REDACTED" in text
    project.runpytest("--llm-record-mode=none").assert_outcomes(passed=1)


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ('"a", "b"', "at most one positional argument"),
        ('"a", name="b"', "both positionally and by keyword"),
        ("record_mode='none'", "unknown option(s) record_mode"),
        ("matching={'body': False}", "matching must be a Matching, got dict"),
        ("redaction=1", "redaction must be a Redaction, got int"),
        ("name=3", "name must be a str, got int"),
        ("ignore_paths='metadata'", "ignore_paths must be a list of paths"),
        ("mode='sometimes'", "unknown record mode 'sometimes'"),
    ],
)
def test_marker_misuse_is_reported(project: pytest.Pytester, marker: str, message: str) -> None:
    project.makepyfile(
        test_demo=f"""
        import pytest

        @pytest.mark.llm_cassette({marker})
        def test_chat(llm_cassette):
            pass
        """
    )
    result = project.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([f"*{message}*"])


def test_unknown_config_keys_are_reported(project: pytest.Pytester) -> None:
    project.makeconftest(
        CONFTEST + "\n@pytest.fixture\ndef llm_cassette_config():\n    return {'modes': 'x'}\n"
    )
    project.makepyfile(test_demo=CHAT_TEST)
    result = project.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*llm_cassette_config returned unknown key(s) modes*"])


def test_ini_options(project: pytest.Pytester) -> None:
    project.makeini(
        "[pytest]\n"
        "llm_cassette_dir = recordings\n"
        "llm_cassette_format = json\n"
        "llm_record_mode = none\n"
    )
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(failed=1)
    project.runpytest("--llm-record-mode=once").assert_outcomes(passed=1)
    assert (project.path / "recordings" / "test_demo" / "test_chat.json").exists()


@pytest.mark.parametrize(
    ("ini", "message"),
    [
        ("llm_record_mode = always", "llm_record_mode must be one of"),
        ("llm_cassette_format = toml", "llm_cassette_format must be yaml or json"),
    ],
)
def test_invalid_ini_options_are_usage_errors(
    pytester: pytest.Pytester, ini: str, message: str
) -> None:
    pytester.makeini(f"[pytest]\n{ini}\n")
    pytester.makepyfile(test_demo="def test_nothing(): pass")
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{message}*"])


def test_an_absolute_cassette_dir_for_a_top_level_test_file(
    project: pytest.Pytester, tmp_path_factory: pytest.TempPathFactory
) -> None:
    target = tmp_path_factory.mktemp("elsewhere")
    project.makeini(f"[pytest]\nllm_cassette_dir = {target}\n")
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(passed=1)
    assert (target / "test_demo" / "test_chat.yaml").exists()


def test_a_swallowed_unmatched_request_still_fails_the_test(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_demo="""
        import pytest

        @pytest.mark.llm_cassette(mode="none")
        def test_chat(ask):
            try:
                ask("hi")
            except Exception:
                pass  # the code under test hides the failure
        """
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*1 request(s) did not match the cassette, although the test passed*",
            "*no recorded interaction matches POST*",
        ]
    )


def test_secrets_from_the_environment_never_reach_disk(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-env-FAKE-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    pytester.makepyfile(
        test_echo="""
        import os
        import httpx2
        from llm_cassette.httpx2 import CassetteTransport

        def echo(request):
            return httpx2.Response(200, text="you sent " + request.headers["x-custom"])

        def test_echo(llm_cassette):
            transport = CassetteTransport(llm_cassette, httpx2.MockTransport(echo))
            client = httpx2.Client(transport=transport)
            key = os.environ["OPENAI_API_KEY"]
            assert client.get("https://example.com", headers={"x-custom": key}).text.endswith(key)
        """
    )
    pytester.runpytest().assert_outcomes(passed=1)
    text = (pytester.path / "cassettes" / "test_echo" / "test_echo.yaml").read_text()
    assert secret not in text
    assert "you sent REDACTED" in text


# --------------------------------------------------------------------------- usage report


def test_usage_report(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_demo="""
        def test_chat(ask):
            ask("hi")
            ask("again")

        def test_no_llm(llm_cassette):
            pass
        """
    )
    result = project.runpytest("--llm-usage-report")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(
        [
            "*= llm-cassette usage =*",
            "test * model * calls * input * cache read * cache write * output",
            "test_demo.py::test_chat * gpt-4o-mini-2024-07-18 * 2 * 20 * 8 * 0 * 8",
            "total * 2 * 20 * 8 * 0 * 8",
        ]
    )
    assert "test_no_llm" not in result.stdout.str()


def test_usage_report_with_prices(project: pytest.Pytester) -> None:
    prices = project.makefile(
        ".toml",
        prices='[models."gpt-4o-mini"]\ninput = 1.0\noutput = 2.0\ncache_read = 0.5\n',
    )
    project.makepyfile(test_demo=CHAT_TEST)
    project.runpytest().assert_outcomes(passed=1)
    result = project.runpytest("--llm-record-mode=none", f"--llm-price-table={prices}")
    result.assert_outcomes(passed=1)
    # 6 uncached input, 4 cached, 4 output tokens: (6 * 1.0 + 4 * 0.5 + 4 * 2.0) / 1e6
    result.stdout.fnmatch_lines(
        ["*cost USD", "test_demo.py::test_chat * 0.000016", "total * 0.000016"]
    )


def test_invalid_price_table_is_a_usage_error(pytester: pytest.Pytester) -> None:
    prices = pytester.makefile(".toml", prices="[models]\nbroken = 1\n")
    result = pytester.runpytest(f"--llm-price-table={prices}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--llm-price-table: *at least input and output*"])


def test_report_without_usage(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_demo="def test_nothing(): pass")
    result = pytester.runpytest("--llm-usage-report")
    result.stdout.fnmatch_lines(["no LLM usage was recorded or replayed in this session"])


def test_render_usage_report_marks_unpriced_models(tmp_path: Path) -> None:
    prices_file = tmp_path / "p.toml"
    prices_file.write_text('[models."known"]\ninput = 1\noutput = 1\n')
    tests = {
        "t.py::a": [
            Usage("openai", "known", input_tokens=10),
            Usage("anthropic", "other", output_tokens=5),
        ],
        "t.py::b": [Usage("openai", "known", input_tokens=1_000_000)],
    }
    lines = render_usage_report(tests, PriceTable.load(prices_file))
    assert lines[1].split() == ["t.py::a", "known,", "other", "2", "10", "0", "0", "5", "n/a"]
    assert lines[2].split()[-1] == "1.000000"
    assert lines[3].split()[0] == "total"
    assert lines[3].split()[-1] == "n/a"
    assert lines[-1] == "n/a: no price in --llm-price-table for other"


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_chat", "test_chat"),
        ("TestChat.test_stream[gpt-4o-mini-0.5]", "TestChat.test_stream[gpt-4o-mini-0.5]"),
        ("a/b\\c:d", f"a_b_c_d-{_digest('a/b' + chr(92) + 'c:d')}"),
        ("test_x[What is 2+2?]", f"test_x[What_is_2_2_]-{_digest('test_x[What is 2+2?]')}"),
        ("...", f"cassette-{_digest('...')}"),
        ("_private", f"private-{_digest('_private')}"),
    ],
)
def test_safe_filenames(name: str, expected: str) -> None:
    assert _safe_filename(name) == expected


def test_long_names_are_shortened_deterministically() -> None:
    name = "test_x[" + "a" * 300 + "]"
    shortened = _safe_filename(name)
    assert shortened == f"{name[:120]}-{_digest(name)}"
    assert shortened == _safe_filename(name)
    other = "test_x[" + "a" * 299 + "b]"
    assert _safe_filename(other) != shortened, "same prefix, different file"
