from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import LIBS, FakeUpstream, HttpLib


@pytest.fixture(params=LIBS, ids=[lib.name for lib in LIBS])
def lib(request: pytest.FixtureRequest) -> HttpLib:
    """Run the test once with httpx and once with httpx2."""
    value: HttpLib = request.param
    return value


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def cassette_path(tmp_path: Path) -> Path:
    return tmp_path / "cassettes" / "test.yaml"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no test can pick up a developer's real API keys or base URLs."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
