# 3. Record modes never mix recording and replaying, and unmatched requests always fail the test

- Status: accepted
- Date: 2026-10-18

## Context

VCR-style libraries offer modes such as "record new requests, replay known ones" (often called
`new_episodes`). In an LLM test suite that mode has a sharp edge: a prompt edited by one character
no longer matches, so the request silently goes to the real API, spends money (if a key happens
to be set) and appends to the cassette instead of failing.

Two more facts shape how failures surface:

- SDKs often treat transport exceptions as connection errors. `anthropic` (checked: 1.0.0 and
  1.15.0) wraps any exception in `APIConnectionError`, which its retry policy retries; `openai`
  3.0.0 wraps it too, while `openai` 3.34.0 lets unknown exceptions through. A test that catches
  broad exceptions can therefore swallow an unmatched request entirely.
- With the SDK defaults (two retries with exponential backoff), an unmatched request also makes
  the test slow to fail.

## Decision

- A cassette decides at construction whether it records or replays, and keeps that for its whole
  lifetime. The modes are:

  | mode     | file missing          | request matches | request does not match |
  |----------|-----------------------|-----------------|------------------------|
  | `record` | record everything     | record (fresh)  | record (fresh)         |
  | `once`   | record everything     | replay          | `UnmatchedRequestError` |
  | `replay` | `CassetteNotFoundError` | replay        | `UnmatchedRequestError` |
  | `none`   | treated as empty      | replay          | `UnmatchedRequestError` |

  `once` is the default for local development, `none` is meant for CI, and `replay` additionally
  fails fast when a cassette was never committed.
- `--llm-record-mode` on the command line overrides markers and configuration, so CI can force
  `none` even for a test whose marker says `record`.
- Every `UnmatchedRequestError` is also stored on the cassette (`Cassette.unmatched`). The pytest
  fixture fails the test at teardown if one was raised but the test still passed, unless the
  test's marker sets `allow_unmatched=True` because checking the miss path is its purpose.
- The error message contains a diff against the closest recorded request.
- The SDK client helpers default to `max_retries=0`.

## Consequences

- No replay mode can reach the network, whatever the request looks like.
- Adding a request to an existing test means re-recording that test's cassette with `record`.
  This is more work than appending, but the resulting cassette always reflects one coherent run.
  An explicit append mode is on the roadmap if real use shows it is needed.
- A swallowed unmatched request still fails the test, with the diff in the report.
- `once` keeps whatever the first run recorded, error responses included, so a first run that hit
  a 401 or a 429 replays it until the cassette is re-recorded. Recording error responses is
  deliberate (tests of error handling need them); the README warns about this case.
- Users who construct SDK clients themselves keep the SDK's retry defaults; the README explains
  why the helpers lower them.
