"""The Python examples in README.md are valid and free of pyflakes errors.

Readers copy these snippets, so a leftover duplicate import or an undefined name is a real bug.
The snippets are not executed (some need an API key or the network); they are compiled and
checked with ruff's pyflakes rules (``F``), which need no project configuration.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

# A fenced block, possibly indented inside a list item; the closing fence has the same indent.
_PYTHON_FENCE = re.compile(
    r"^(?P<indent>[ ]*)```python\n(?P<body>.*?)^(?P=indent)```$", re.MULTILINE | re.DOTALL
)


def _python_blocks() -> list[tuple[int, str]]:
    """Return ``(line number of the opening fence, dedented source)`` for each Python block."""
    text = README.read_text(encoding="utf-8")
    return [
        (text.count("\n", 0, match.start()) + 1, textwrap.dedent(match["body"]))
        for match in _PYTHON_FENCE.finditer(text)
    ]


BLOCKS = _python_blocks()


def test_every_python_fence_is_extracted() -> None:
    assert len(BLOCKS) == README.read_text(encoding="utf-8").count("```python")
    assert BLOCKS


@pytest.mark.parametrize(("line", "source"), BLOCKS, ids=[f"line-{line}" for line, _ in BLOCKS])
def test_python_block_is_valid_and_pyflakes_clean(line: int, source: str) -> None:
    compile(source, f"README.md:{line}", "exec")
    result = subprocess.run(  # noqa: S603  # fixed argv; the snippet is passed on stdin
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "F",
            "--target-version",
            "py311",
            "--output-format",
            "concise",
            "--stdin-filename",
            f"README_line_{line}.py",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"README.md block at line {line}:\n{result.stdout}"
