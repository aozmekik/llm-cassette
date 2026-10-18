# llm-cassette

Record LLM API calls once, replay them in every test run: a VCR for the `openai` and
`anthropic` Python SDKs (and any other httpx-based client), working at the httpx transport layer
and keeping streamed responses chunk for chunk.

[![CI](https://github.com/aozmekik/llm-cassette/actions/workflows/ci.yml/badge.svg)](https://github.com/aozmekik/llm-cassette/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status: early (v0.1.0).** The API may change before 1.0. Not yet published to PyPI.

## Why

Tests that call an LLM API are slow, cost money, need credentials in CI and are not
deterministic. Mocking the SDK by hand gives fast tests that assert against responses nobody
received, and hand-written streams rarely match what the API really sends.

llm-cassette records the real HTTP exchange the first time a test runs, stores it as a readable
YAML file next to the test, and replays it on every later run, without network access or keys.
Because it records below the SDK, a streamed response replays with the exact bytes and chunk
boundaries the SDK's SSE parser saw, and an SDK upgrade that changes the wire format shows up
as a cassette diff.

## Features

- **Transports, not monkeypatching**: `CassetteTransport` / `AsyncCassetteTransport` wrap a real
  httpx2 or httpx transport. Nothing global is patched.
- **Streaming fidelity**: responses are recorded chunk by chunk while they stream to the client,
  and replayed with the original chunk boundaries, instantly by default or with the recorded
  timing scaled by `timing_scale`. A body the client stopped reading (an early `break`, Ctrl-C,
  a timeout) is marked incomplete and fails loudly if a replay reads past the recorded end.
- **Four record modes**: `record`, `once`, `replay`, `none`. Replay modes never touch the network.
- **Matching** on method, normalized URL and canonical JSON body, with ignore paths
  (`metadata.user_id`, `messages.*.cache_control`), ignored query parameters, header matching and
  custom matchers. A miss raises `UnmatchedRequestError` with a diff against the closest recording.
- **Redaction before anything is stored**: `authorization`, `x-api-key`, `openai-organization`,
  `openai-project`, `cookie`, `set-cookie` and any header or query parameter whose name contains
  `key`, `token` or `secret` (in the request URL and in `location`-style headers), URL user
  info, JSON body paths, and literal secret values, including their JSON-escaped and
  percent-encoded forms. A final check refuses to write a cassette in which a secret survived.
- **Readable, deterministic cassettes**: one YAML (or JSON) file per test, schema-versioned, sorted
  keys and headers, decoded bodies, multi-line strings as literal blocks.
- **pytest plugin**: `llm_cassette` fixture, `@pytest.mark.llm_cassette(...)`,
  `--llm-record-mode`, one cassette per test with names derived from the test id, and helpers
  that return `openai` / `anthropic` clients (sync and async) wired to the cassette.
- **Usage report**: `--llm-usage-report` prints per-test token counts parsed from OpenAI Chat
  Completions, OpenAI Responses and Anthropic Messages responses (streamed or not), also under
  `pytest-xdist`, and `--llm-price-table prices.toml` adds cost estimates from prices you supply.
- Typed (`py.typed`, checked with `mypy --strict`), thread-safe, no dependencies beyond PyYAML
  and anyio.

## Install

Not yet published to PyPI. Install from GitHub:

```sh
pip install "git+https://github.com/aozmekik/llm-cassette"
```

Requires Python 3.11 or newer. The adapters use whichever HTTP library you already have; the
`httpx` and `httpx2` extras pull one in explicitly
(`pip install "llm-cassette[httpx2] @ git+https://github.com/aozmekik/llm-cassette"`). The SDK
helpers need `openai>=3` and/or `anthropic>=1`, which depend on `httpx2`. The pytest plugin
needs pytest 8.3 or newer.

## Quick start

```python
# tests/test_chat.py
from llm_cassette.clients import openai_client


def test_haiku(llm_cassette):
    client = openai_client(llm_cassette)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about tests"}],
    )
    assert completion.choices[0].message.content
```

```sh
OPENAI_API_KEY=sk-... pytest tests/test_chat.py   # first run: records tests/cassettes/test_chat/test_haiku.yaml
pytest tests/test_chat.py                         # later runs: replays it, no key needed
pytest --llm-record-mode=none                     # in CI: fail on anything not recorded
```

Commit the `cassettes/` directory with your tests. [`examples/pytest_demo`](examples/pytest_demo)
is a complete, runnable suite; its cassettes were recorded against a fake upstream, so their
content is synthetic.

## Usage

### Record modes

| mode | cassette file missing | request matches | request does not match |
|---|---|---|---|
| `record` | record everything | record (fresh) | record (fresh) |
| `once` (default) | record everything | replay | `UnmatchedRequestError` |
| `replay` | `CassetteNotFoundError` | replay | `UnmatchedRequestError` |
| `none` | treated as empty | replay | `UnmatchedRequestError` |

A cassette records or replays for its whole lifetime; it never mixes the two, so no replay mode
can reach the network ([ADR 3](docs/adr/0003-record-modes-and-failure-semantics.md)). To add a
request to an existing test, re-record its cassette with `record`.

`once` keeps whatever the first run recorded, error responses included: if that run hit a 401
(wrong key), a 429 or a 5xx, every later run replays it. Delete the cassette or re-record it with
`--llm-record-mode=record` after fixing the cause.

### Configuring cassettes in pytest

Settings are applied from lowest to highest precedence:

1. ini options in `pytest.ini` / `pyproject.toml`:

   | option | default | meaning |
   |---|---|---|
   | `llm_record_mode` | `once` | default record mode |
   | `llm_cassette_dir` | `cassettes` | directory, relative to each test file unless absolute |
   | `llm_cassette_format` | `yaml` | `yaml` or `json` |
   | `llm_secret_env` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | env vars whose values (8+ characters) are treated as secrets |

2. an `llm_cassette_config` fixture with project-wide defaults:

   ```python
   # conftest.py
   import pytest
   from llm_cassette import Matching, Redaction
   from llm_cassette.pytest_plugin import CassetteConfig


   @pytest.fixture
   def llm_cassette_config() -> CassetteConfig:
       return {
           "matching": Matching(ignore_paths=["metadata.user_id", "stream_options"]),
           "redaction": Redaction(body_paths=["user"]),
       }
   ```

3. `@pytest.mark.llm_cassette(...)` on a module (`pytestmark`), class or test; the closest one
   wins per option, and an option given as `None` counts as not given. It accepts `name` (also
   positionally: a file name to use instead of the one derived from the test id), `mode`,
   `matching`, `redaction`, `ignore_paths` (added to the matching's), `timing_scale`,
   `allow_repeats` and `allow_unmatched` (see [Matching](#matching)).

4. `--llm-record-mode` on the command line, which overrides everything, so CI can force `none`.

The cassette of `tests/test_chat.py::TestStream::test_tokens[gpt-4o-mini]` is
`tests/cassettes/test_chat/TestStream.test_tokens[gpt-4o-mini].yaml`. Characters other than
`A-Z a-z 0-9 _ . - [ ]` become `_`, and when that changes a name a short hash of the original is
appended, so the ids `What is 2+2?` and `What is 2*2?` get separate files
(`test_x[What_is_2_2_]-<hash>.yaml`). With an absolute `llm_cassette_dir`, the folders mirror the
test files' paths below the rootdir.

A cassette belongs to one test. Two tests that would use the same file in one session, through a
shared `name` or names that differ only in case (`test_A`, `test_a`; macOS and Windows file
systems ignore case), fail at setup: in `record` mode the second would overwrite the first,
and in `once` mode it would replay the other test's traffic.

The client helpers in `llm_cassette.clients` (`openai_client`, `async_openai_client`,
`anthropic_client`, `async_anthropic_client`) forward keyword arguments to the SDK. They set
`max_retries=0` unless you pass it (an unmatched request is not worth retrying), and while
replaying they pass a placeholder API key so no credentials are needed. `transport=` replaces the
upstream used when recording.

### Matching

A live request matches a recorded one when the method, the normalized URL (lower-case scheme and
host, default port dropped, query sorted) and the body are equal. JSON bodies are compared
structurally, so key order and whitespace never matter, but `true` and `1` do. Identical requests
replay in recording order.

```python
from llm_cassette import Matching

from llm_cassette import Matching, RecordedRequest


def same_model(live: RecordedRequest, recorded: RecordedRequest) -> bool:
    a, b = live.json(), recorded.json()  # None unless the body is a JSON object or array
    return isinstance(a, dict) and isinstance(b, dict) and a.get("model") == b.get("model")


Matching(
    ignore_paths=["metadata.user_id", "messages.*.cache_control"],  # removed from both bodies
    ignore_query=["trace_id"],
    headers=["anthropic-beta"],  # these header values must match too
    custom=[same_model],
    body=False,  # any built-in comparison can be switched off
)
```

Custom matchers only see recorded requests that already pass the built-in comparisons, and an
exception in one propagates as the bug it is. All paths are resolved against the original body,
so `ignore_paths=["messages.0", "messages.1"]` ignores the first two messages.

When nothing matches, the error shows the closest recorded request (real output):

```text
no recorded interaction matches POST https://api.openai.com/v1/chat/completions
cassette: /project/tests/cassettes/test_chat/test_haiku.yaml (mode=none, 1 recorded, 0 played)
closest recorded interaction: #0, differs in: body
--- recorded #0 body
+++ live request body
@@ -6,5 +6,5 @@
     },
     {
-      "content": "Write a haiku about tests",
+      "content": "Write a haiku about cassettes",
       "role": "user"
     }
hint: re-record the cassette (mode 'record' or --llm-record-mode=record), or ignore volatile fields with Matching(ignore_paths=[...])
```

SDKs may wrap this error: `anthropic` raises `APIConnectionError` with it as `__cause__`. Every
unmatched request is also kept in `Cassette.unmatched`, and the pytest fixture fails a test that
passed despite one. A test that checks the miss path on purpose opts out with
`@pytest.mark.llm_cassette(allow_unmatched=True)`.

### Redaction

```python
import os

from llm_cassette import DEFAULT_REDACTED_HEADERS, Redaction

Redaction(
    headers=DEFAULT_REDACTED_HEADERS | {"x-tenant-id"},  # a plain set replaces the defaults
    substrings=("key", "secret"),  # default ("key", "token", "secret")
    query_params={"sig"},
    body_paths=["user", "metadata.email"],
    secrets=[os.environ["MY_GATEWAY_TOKEN"]],
    drop_headers=["user-agent", "x-stainless-*"],  # removed entirely; fnmatch patterns
    placeholder="REDACTED",
)
```

Redaction runs on every live request before it is matched, and on every interaction before the
cassette exposes or writes it:

- Query parameters and user info (`https://TOKEN@host`, `user:pass@`) are redacted in the request
  URL and in the URL-valued headers `location`, `content-location`, `link` and `referer`, so a
  redirect to a presigned download link does not keep its token.
- Body paths apply to JSON request bodies and to non-streaming JSON responses. A JSON response
  that does not parse (for example one the client closed half-way) is replaced by the placeholder
  as a whole, because the paths cannot be located in it.
- `secrets` are replaced as is, JSON-escaped (`\u00e4`, `\"`) and percent-encoded (`%2B`, `+`).
  Before writing, the cassette also searches decoded JSON strings, percent-decoded text and the
  serialized file, and raises `SecretLeakError` if a secret is still there (for example split
  across two streamed chunks). The pytest plugin adds the values of `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY` (configurable with `llm_secret_env`).
- When redaction changes a response body, `content-length` is adjusted to match.
- `drop_headers` keeps headers that change with the SDK, OS or Python version out of the
  cassette, so re-recording on another machine gives a clean diff. Nothing is dropped by default.

Note that the default `token` substring also redacts rate-limit headers such as
`x-ratelimit-remaining-tokens`; pass narrower `substrings` to keep them.

### Streaming and timing

Recorded delays are replayed multiplied by `timing_scale`: `0` (the default) replays instantly,
`1` reproduces the recorded pacing, which is useful for testing timeouts or progress UIs. The
transports take `clock=` and `sleep=` arguments, so timing can be tested with fakes.

A response counts as complete when the client read its body to the end while it was recorded.
When the client stopped first (it closed the response early, or Ctrl-C or a cancellation such as
`anyio.fail_after` interrupted it), the cassette stores what was read with `complete: false`
([ADR 5](docs/adr/0005-incomplete-recordings.md)). Replaying it serves the same chunks; a client
that stops at the same point behaves exactly as it did, and one that reads further gets an
`httpx.ReadError` that says to re-record, instead of a body that silently ends early.
`openai`'s sync client stops reading at `data: [DONE]`, so its streamed chat completions are
recorded as incomplete and replay identically; the async client and `anthropic` read to the end.

### Without pytest

```python
import httpx2
from llm_cassette import Cassette
from llm_cassette.httpx2 import CassetteTransport  # llm_cassette.httpx for httpx 0.x

with Cassette("cassettes/chat.yaml", mode="once") as cassette:  # saved on exit
    client = httpx2.Client(transport=CassetteTransport(cassette))
    client.post("https://api.openai.com/v1/chat/completions", json={...})
```

`CassetteTransport(cassette, transport)` wraps `transport` and closes it when closed. Without
one, recording sends requests the way a default `httpx2.Client()` would, honouring
`HTTPS_PROXY`, `NO_PROXY` and the other proxy variables, like the SDKs' own clients. `openai>=3`
also accepts an `httpx.Client`, so `llm_cassette.httpx` works with it as well as with older SDKs
and other httpx-based clients. To support another HTTP client, call `Cassette.resolve(request)`
from an adapter: it returns either the `Interaction` to replay or a `Recording` to fill in
(`start`, `add_chunk`, then `finish` once the body was read to the end, or `abandon`).

### Usage report and cost estimates

```sh
pytest --llm-usage-report --llm-price-table prices.toml
```

```toml
# prices.toml: USD per million tokens. llm-cassette ships no prices; copy them from your provider.
[models."gpt-4o-mini*"]   # exact names or fnmatch patterns
input = 1.00
output = 4.00
cache_read = 0.50         # optional; cache_read and cache_write default to the input price
```

Output of the demo suite (`uv run pytest examples/pytest_demo --llm-record-mode=none
--llm-price-table examples/pytest_demo/prices.toml`; synthetic data and illustrative prices):

```text
============================== llm-cassette usage ==============================
test                                                   model                   calls  input  cache read  cache write  output  cost USD
examples/pytest_demo/test_demo.py::test_anthropic      claude-sonnet-4-6           1     20           0            0       4  0.000080
examples/pytest_demo/test_demo.py::test_openai_chat    gpt-4o-mini-2024-07-18      1     12           0            0       3  0.000024
examples/pytest_demo/test_demo.py::test_openai_stream  gpt-4o-mini-2024-07-18      1     12           0            0       3  0.000024
total                                                                              3     44           0            0      10  0.000128
input = all input tokens (cache reads and writes included); token counts come from the recorded responses
```

Under `pytest-xdist`, each worker sends the usage it saw to the controller, which prints one
report for the whole session.

Providers count cached input differently (OpenAI's `prompt_tokens` include cached tokens,
Anthropic's `input_tokens` exclude them), so `llm_cassette.usage.Usage` splits input into
disjoint buckets that can each be priced. Streamed OpenAI chat completions only carry usage when
requested with `stream_options={"include_usage": True}`.

### What a cassette looks like

An excerpt of [`test_openai_stream.yaml`](examples/pytest_demo/cassettes/test_demo/test_openai_stream.yaml)
(synthetic data; the demo drops `user-agent` and `x-stainless-*` headers):

```yaml
version: 1
interactions:
  - request:
      method: POST
      url: https://api.openai.com/v1/chat/completions
      headers:
        accept: application/json
        accept-encoding: gzip, deflate, zstd
        authorization: REDACTED
        content-type: application/json
      body:
        json:
          messages:
            - content: akış testi
              role: user
          model: gpt-4o-mini
          stream: true
          stream_options:
            include_usage: true
    response:
      status: 200
      headers:
        content-type: text/event-stream
      delay_ms: 0.1                 # request sent -> response headers
      complete: false               # openai's sync client stops reading at [DONE]
      chunks:
        - delay_ms: 0.1             # since the previous chunk
          text: |+
            data: {"id": "chatcmpl-demo", ..., "choices": [{"index": 0, "delta": {"content": "Echo:"}, "finish_reason": null}]}

        # ...
        - delay_ms: 0.0
          text: |+
            data: [DONE]
```

Chunks that are not valid UTF-8 on their own (a multi-byte character split across two network
reads) are stored as `base64:` instead of `text:`, so every byte is preserved. Header values are
stored as their raw bytes read as ISO-8859-1 (the WSGI convention), so a UTF-8 filename in
`content-disposition` round-trips exactly, although it reads as `rÃ©sumÃ©.pdf` in the file.

## How it works

```mermaid
sequenceDiagram
    participant SDK as openai / anthropic SDK
    participant T as CassetteTransport
    participant C as Cassette
    participant U as real transport
    SDK->>T: handle_request(request)
    T->>C: resolve(redacted snapshot)
    alt replaying
        C-->>T: recorded Interaction
        T-->>SDK: Response streaming the recorded chunks
    else recording
        C-->>T: Recording
        T->>U: handle_request(request)
        U-->>T: Response
        T-->>SDK: Response; each decoded chunk is copied into the Recording as it streams
    end
```

Design notes and trade-offs, with the alternatives considered, are in the ADRs:

1. [Record at the httpx transport layer](docs/adr/0001-record-at-the-httpx-transport-layer.md),
   rather than mocking SDK methods, patching HTTP libraries globally or running a proxy.
2. [Separate adapters for httpx and httpx2](docs/adr/0002-adapters-for-httpx-and-httpx2.md):
   `openai>=3` and `anthropic>=1` moved to httpx2, whose classes are distinct from httpx's.
3. [Record modes and failure semantics](docs/adr/0003-record-modes-and-failure-semantics.md):
   no mode mixes recording and replaying, and unmatched requests always fail the test.
4. [The cassette format](docs/adr/0004-cassette-format.md): decoded, chunk-preserving and
   deterministic.
5. [Incomplete recordings](docs/adr/0005-incomplete-recordings.md): a body the client did not
   finish reading is stored as such, rather than dropped, drained or passed off as whole.
6. [One cassette per test](docs/adr/0006-one-cassette-per-test.md): collision-free file names and
   a session-wide ownership check, rather than shared cassettes.

Replaying looks requests up in an index keyed by the built-in match aspects, so a hit costs the
same for one or a thousand recorded interactions; the full comparison only runs to explain a miss.

## Compatibility

| | tested with |
|---|---|
| Python | 3.11.16, 3.12.10, 3.13.2 and 3.14.5 locally (macOS); CI runs 3.11 to 3.14 on Ubuntu and 3.14 on macOS |
| SDKs | `openai` 3.0.0 and 3.34.0, `anthropic` 1.0.0 and 1.15.0 |
| HTTP | `httpx2` 2.7.0 and 2.13.1, `httpx` 0.27.0 and 0.28.1 |

The lowest versions come from installing with `uv pip install --resolution lowest-direct -e .
--group dev` and running the full suite (331 tests passed on Python 3.11.16 and 3.12.10); CI
repeats this on Python 3.11. Windows is not tested yet.

## Testing

The suite has 331 tests and 100% line and branch coverage of `src/` (1460 statements, 362
branches), from `uv run coverage run -m pytest && uv run coverage report` in a fresh clone on an
Apple M1 Pro with 16 GB, macOS 26, with Python 3.14.5 and again with 3.11.16; the suite also
passed on 3.12.10 and 3.13.2. No test calls a real API: fake
upstreams built on `httpx2.MockTransport` follow the documented OpenAI Chat Completions
(including `include_usage` streaming), OpenAI Responses and Anthropic Messages (including the
`message_start`, `content_block_start`, `content_block_delta`, `message_delta` and `message_stop`
events) wire formats, split their streams at awkward byte offsets, and check credentials. The
tests record through them and replay with the real `openai` and `anthropic` clients, sync and
async. Property-based tests (Hypothesis) round-trip arbitrary JSON, text and bytes through the
cassette format and check the SSE parser and the path rewriter.

## Benchmarks

`uv run python benchmarks/replay_overhead.py` (1000 calls x 7 repeats after a warm-up repeat;
median, min-max in parentheses; synthetic payloads). Apple M1 Pro, 16 GB, macOS 26, Python
3.14.5, httpx2 2.13.1. Other heavy jobs were running on the machine (load average about 32 on 8
cores), so treat the numbers as rough: two earlier runs (one on Python 3.11.16) measured an
overhead of 46-51 us for the JSON case and 64-94 us for the SSE case.

Per request through an `httpx2.Client`, compared with an `httpx2.MockTransport` that builds the
same response in memory:

| response | mock, us/request | replay, us/request | overhead, us |
|---|---|---|---|
| 2 KB JSON, 1 chunk | 48 (47-51) | 108 (100-114) | 60 |
| SSE, 201 chunks | 128 (112-194) | 174 (171-183) | 46 |

`Cassette.resolve` alone (redact, normalize and look up a 2 KB JSON request), with the matching
interaction recorded last:

| recorded interactions | us/call |
|---|---|
| 1 | 14 (13-14) |
| 10 | 14 (14-18) |
| 100 | 14 (13-15) |
| 1000 | 14 (14-15) |

In other words, replaying adds tens of microseconds per request, which is negligible next to a
real API round trip.

## Limitations

- Only httpx and httpx2 clients are supported; clients built on `requests` or `aiohttp` need an
  adapter (the `Cassette.resolve` / `Recording` API is the extension point).
- Bodies that are not JSON (multipart uploads, form data) are matched byte for byte, so random
  multipart boundaries never match; use `Matching(body=False)` plus a custom matcher.
- A response whose body fails half-way through recording (the upstream raises) is dropped rather
  than stored truncated; errors are not recorded, only responses. A body the client stops
  reading is stored with `complete: false`, see [Streaming and timing](#streaming-and-timing).
- In `once` mode, a first recording of an error response (401, 429, 5xx) is replayed from then
  on; re-record with `--llm-record-mode=record` once the cause is fixed.
- Body-path redaction does not apply to streamed (SSE) responses; use `secrets=` for literal
  values there.
- Recorded timing includes the time the client took to consume each chunk.
- Replayed responses report HTTP/1.1 whatever protocol was recorded (the protocol version is not
  stored).
- Under `pytest-xdist`, the check that no two tests share a cassette runs within each worker.
- Windows is not tested in CI yet.

## Roadmap

- An explicit append mode (record new requests into an existing cassette) if real use shows the
  need ([ADR 3](docs/adr/0003-record-modes-and-failure-semantics.md) explains why it is not the
  default).
- A redaction hook for streamed bodies.
- A CLI to inspect cassettes and summarize their usage outside pytest.
- Windows in the CI matrix.
- Publishing to PyPI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the checks CI runs. The
changelog is in [CHANGELOG.md](CHANGELOG.md).

## References and inspirations

- The record/replay idea comes from Ruby's [VCR](https://github.com/vcr/vcr) and its Python
  counterparts [vcrpy](https://github.com/kevin1024/vcrpy) and
  [pytest-recording](https://github.com/kiwicom/pytest-recording); llm-cassette is an independent
  implementation focused on LLM streaming.
- [httpx transports](https://www.python-httpx.org/advanced/transports/) and
  [httpx2](https://github.com/pydantic/httpx2).
- The SSE parser follows the event stream interpretation rules of the
  [WHATWG HTML Living Standard, "Server-sent events"](https://html.spec.whatwg.org/multipage/server-sent-events.html).
- Payload shapes: the OpenAI API reference for
  [Chat Completions streaming](https://platform.openai.com/docs/api-reference/chat-streaming) and
  [Responses](https://platform.openai.com/docs/api-reference/responses), and Anthropic's
  [Messages streaming](https://docs.anthropic.com/en/docs/build-with-claude/streaming) docs.
- Hop-by-hop request headers are left out of recordings following
  [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110) section 7.6.1; `host` and `content-length`
  are left out too, because they are derived from the URL and the (canonicalized) body.

## License

[MIT](LICENSE)

Built by [Ahmed Semih Özmekik](https://github.com/aozmekik) at [Fergani](https://fergani.tr).
