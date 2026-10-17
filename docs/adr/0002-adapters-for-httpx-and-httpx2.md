# 2. Separate adapters for httpx and httpx2 over a library-agnostic core

- Status: accepted
- Date: 2026-10-18

## Context

`openai` 3.0 and `anthropic` 1.0 moved from `httpx` to `httpx2`, a maintained continuation of
httpx published under a new import name (checked against the `requires_dist` metadata of both
packages on PyPI). The two libraries have the same API but distinct classes, and the client
checks them: `httpx2.Client` asserts that a transport returns a response whose stream is an
`httpx2.SyncByteStream`, and the type checkers expect an `httpx2.BaseTransport`. Meanwhile many
other clients, and every earlier SDK release, are still built on `httpx`; `openai>=3` accepts an
`httpx.Client` too, while `anthropic>=1` only accepts `httpx2`.

Options considered:

1. **Support only httpx2.** Smallest code, but excludes every httpx-based client.
2. **One duck-typed adapter** that inspects the incoming request's module and builds the
   response class from it. One copy of the code, but the transport cannot subclass both
   libraries' `BaseTransport` without depending on both, so `mypy --strict` users would need
   casts at every call site, and the class-picking logic is easy to get subtly wrong.
3. **Generate one adapter from the other** at build time. No duplication in the repository, but
   a code generator for ~260 lines is more machinery than the duplication it removes.
4. **Two mirrored adapter modules** over a shared core.

## Decision

Use option 4. Everything that does not depend on the HTTP library lives in the core:
`RecordedRequest`, `Interaction`, matching, redaction, the cassette file format, and the
`Cassette.resolve` / `Recording` protocol that adapters use. `llm_cassette.httpx` and
`llm_cassette.httpx2` are mirror images that only translate between their library's
request/response objects and the core, and each subclasses its own library's transport classes.

## Consequences

- Each adapter type-checks against its own library, and users get correct types with no casts.
- The duplication is deliberate and bounded: the two modules differ only in the imported library
  and their docstrings. The transport tests run every case against both libraries
  (`tests/test_transports.py` is parametrized over them) and assert that their public APIs match.
- A bug fix in one adapter must be applied to the other; the shared tests catch a missed one.
- The core never imports an HTTP library, so adding an adapter for another client means
  implementing two methods on top of `Cassette.resolve` and `Recording`.
