# Contributing

Thanks for taking the time to contribute. Bug reports, questions and pull requests are welcome
on [GitHub](https://github.com/aozmekik/llm-cassette/issues).

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and a local virtual environment in `.venv`.

```sh
git clone https://github.com/aozmekik/llm-cassette
cd llm-cassette
uv sync                 # creates .venv with the package (editable) and the dev group
```

Without uv, any Python 3.11+ works with pip 25.1 or newer:

```sh
python3 -m venv .venv
.venv/bin/pip install -e . --group dev
```

## Checks

These are the commands CI runs; please run them before opening a pull request.

```sh
uv run ruff format --check .      # formatting (uv run ruff format . to fix)
uv run ruff check .               # lint, with every rule enabled (see pyproject.toml)
uv run mypy                       # mypy --strict over src, tests, examples and benchmarks
uv run coverage run -m pytest     # the test suite
uv run coverage report            # fails below 97% line and branch coverage
```

The examples and the benchmark smoke run:

```sh
uv run python examples/quickstart.py
uv run python examples/openai_streaming.py
uv run pytest examples/pytest_demo --llm-record-mode=none --llm-usage-report \
  --llm-price-table examples/pytest_demo/prices.toml
uv run python benchmarks/replay_overhead.py --quick
```

CI also runs the suite against the lowest supported dependency versions:

```sh
uv venv .venv-lowest --python 3.11
uv pip install --python .venv-lowest --resolution lowest-direct -e . --group dev
.venv-lowest/bin/python -m pytest
```

## Guidelines

- Tests never call a real API. Use the fake upstreams in `tests/fakes.py`, which follow the
  documented OpenAI and Anthropic wire formats; extend them when you need a new endpoint.
- Changes to `llm_cassette/httpx.py` and `llm_cassette/httpx2.py` must be mirrored in the other
  module (see [ADR 2](docs/adr/0002-adapters-for-httpx-and-httpx2.md)); the transport tests run
  against both.
- A change to the cassette file format needs a new schema version and a migration note.
- Record significant design decisions as a new ADR in `docs/adr/`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- Add an entry to `CHANGELOG.md` under "Unreleased" for user-visible changes.
