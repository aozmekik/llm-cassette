# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, minor
releases may contain breaking changes.

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added

- `Cassette` with the record modes `record`, `once`, `replay` and `none`; thread-safe, saved
  atomically, and saved only when something new was recorded.
- `CassetteTransport` and `AsyncCassetteTransport` for httpx2 (`llm_cassette.httpx2`) and httpx
  (`llm_cassette.httpx`). Responses are recorded chunk by chunk with their timing and replayed with
  the original chunk boundaries, instantly by default or with scaled timing (`timing_scale`).
  Header values are kept as raw bytes, so non-ASCII values record and replay unchanged. Without
  an explicit upstream, recording honours the proxy environment variables.
- Incomplete recordings: a body the client stopped reading (early close, Ctrl-C, cancellation)
  is stored with `complete: false`, and replaying it raises a read error past the recorded end
  instead of ending silently. Adapters mark a body complete with `Recording.finish()`.
- Matching on method, normalized URL and canonical JSON body, with `ignore_paths`,
  `ignore_query`, header matching and custom matchers (called only for requests that pass the
  built-in comparisons); unmatched requests raise `UnmatchedRequestError` with a diff against the
  closest recorded request.
- Redaction of sensitive headers (by name and by `key`/`token`/`secret` substrings), query
  parameters and user info in the request URL and in URL-valued headers, JSON body paths, and
  literal secret values in their plain, JSON-escaped and percent-encoded forms, applied before
  anything is matched, exposed or written; `content-length` follows a redacted body;
  `drop_headers` removes volatile headers; `SecretLeakError` if a secret survives, checked on
  decoded views and on the serialized file.
- Cassette file format version 1 in YAML (default) or JSON, with deterministic output.
- pytest plugin: `llm_cassette` fixture, `@pytest.mark.llm_cassette` marker,
  `llm_cassette_config` fixture for project defaults, `--llm-record-mode`, ini options, cassette
  paths derived from the test node id (collision-free, one cassette per test, checked at setup),
  and teardown failures for swallowed unmatched requests (`allow_unmatched=True` opts out).
- `llm_cassette.clients` helpers that build `openai` and `anthropic` clients (sync and async)
  wired to a cassette.
- `llm_cassette.usage`: token usage parsed from OpenAI Chat Completions, OpenAI Responses and
  Anthropic Messages responses, streamed or not, plus `PriceTable` cost estimates from a
  user-supplied TOML file; `--llm-usage-report` and `--llm-price-table` print a per-test summary,
  aggregated across pytest-xdist workers.

[Unreleased]: https://github.com/aozmekik/llm-cassette/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aozmekik/llm-cassette/releases/tag/v0.1.0
