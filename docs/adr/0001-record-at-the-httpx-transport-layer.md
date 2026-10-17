# 1. Record at the httpx transport layer

- Status: accepted
- Date: 2026-10-18

## Context

Tests for code that calls LLM APIs need the real wire responses (streaming included) without
calling the API on every run: calls are slow, cost money, need credentials and are not
deterministic. The official `openai` and `anthropic` Python SDKs both send every request through
an httpx-style client and accept a custom one via `http_client=`.

The interception point decides what the recording contains and what can go wrong:

1. **Mock the SDK methods** (`unittest.mock` on `client.chat.completions.create`). Simple, but the
   test then asserts against hand-written objects, not against what the API returned, and it
   breaks whenever the SDK changes internal call paths. Streaming has to be faked by hand.
2. **Patch the HTTP library globally**, as VCR-style libraries do for `requests`, `urllib3` or
   `httpx`. Works with clients you cannot configure, but it depends on private internals of the
   HTTP library, affects every request in the process (including unrelated ones), and is fragile
   across library releases.
3. **Run a recording proxy** and point `base_url` at it. Language-agnostic and sees real
   traffic, but needs a separate process, port management in CI, and TLS handling for anything
   that does not support a plain-HTTP base URL.
4. **Wrap the transport** that the httpx client already delegates to, and pass the client to the
   SDK explicitly.

## Decision

Use option 4. `CassetteTransport` and `AsyncCassetteTransport` implement the public httpx
transport interface (`handle_request` / `handle_async_request`). When recording they forward to a
real transport and copy the decoded response chunk by chunk as it streams to the client; when
replaying they build the response from the cassette. Helpers in `llm_cassette.clients` create
SDK clients that are already wired, so tests do not repeat the plumbing.

## Consequences

- No global state and no private APIs: only the documented transport interface is used, and
  only clients that were given the transport are affected.
- The cassette contains exactly what the SDK sent and received after serialization, retries and
  redirects, so SDK upgrades that change the wire format show up as cassette diffs.
- Streaming fidelity comes for free: the transport sees the same byte chunks the SDK's SSE
  decoder sees, so they can be replayed with the same boundaries.
- Code under test has to accept an injected client (or a factory). Applications that construct
  SDK clients deep inside library code need a seam; global patching would not.
- Any other httpx-based client (not just the two SDKs) works, but clients built on `requests` or
  `aiohttp` do not. The library-agnostic `Cassette.resolve` / `Recording` API is the extension
  point for such adapters (see [ADR 2](0002-adapters-for-httpx-and-httpx2.md)).
