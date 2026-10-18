# 4. A deterministic, decoded, chunk-preserving cassette format

- Status: accepted
- Date: 2026-10-18

## Context

Cassettes are committed to git, read in code review, and occasionally edited by hand. They must
also replay streamed responses exactly, keep binary bodies intact, and never contain secrets.

Questions and the options considered:

- **Encoding of the file**: YAML (readable, multi-line strings without escaping) or JSON
  (ubiquitous, strict). Both are supported; YAML is the default.
- **Response bodies**: the raw wire bytes (compressed bodies become opaque base64), the parsed
  JSON (readable, but loses byte and chunk fidelity and cannot hold event streams), or the decoded
  bytes as a list of chunks.
- **Request bodies**: the exact bytes (key order and whitespace produce noisy diffs) or the parsed
  JSON in canonical form.
- **Files**: one file per test, or one per module or session (fewer files, but merge conflicts
  and unrelated diffs whenever one test changes).

## Decision

- One file per test, with a top-level `version: 1` schema version. Loading rejects unknown
  versions and unknown keys with an error that names the offending location.
- Responses are stored as the list of chunks the client received, after `content-encoding` was
  decoded, each chunk with its delay in milliseconds rounded to 0.1. When the body was encoded
  (gzip, br, zstd), `content-encoding` and `content-length` are dropped, because they describe
  bytes the cassette does not hold; an identity-encoded body keeps its `content-length`, which
  redaction adjusts when it changes the body. A chunk that is valid UTF-8 is stored as text,
  anything else as base64, per chunk, so a multi-byte character split across two network reads
  stays exact. A response whose body the client did not read to the end carries
  `complete: false` ([ADR 5](0005-incomplete-recordings.md)).
- Header values are stored as their raw bytes decoded as ISO-8859-1, the convention WSGI uses:
  every byte maps to one character, so a UTF-8 or Latin-1 value round-trips exactly, while ASCII
  values (nearly all of them) read as themselves. Text beyond U+00FF, which can only come from a
  hand edit, is sent as UTF-8.
- JSON request bodies (objects and arrays) are stored as structured data with sorted keys; other
  bodies as text or base64. Connection-level request headers (`host`, `content-length`,
  `connection`, ...) are not stored.
- Output is deterministic: fixed key order, headers sorted by name, YAML always written by
  PyYAML's pure-Python emitter (libyaml's C emitter makes different style choices, so files would
  differ between machines), no line wrapping, and multi-line strings as literal blocks.
- Strings containing U+0085, U+2028 or U+2029 are always double-quoted: PyYAML's loader folds
  those characters into line breaks or spaces in any other scalar style. Property tests round-trip
  arbitrary text, bytes and JSON documents through the format.
- Redaction runs before an interaction is matched, exposed or written, and a final scan refuses to
  write a file that still contains a configured secret, searching each field and the serialized
  file as is, percent-decoded and with JSON string escapes decoded (nested up to four levels). It
  recognizes those encodings only, not, say, a base64-encoded secret.
- JSON numbers that do not fit in a float (`1e400`) make a body opaque text rather than JSON, so
  a cassette that can be written can always be read back.

## Consequences

- Diffs are small and readable, but re-recording is not byte-stable: besides the measured delays,
  it changes volatile request fields that matching ignores (a `metadata.run_id`, say) and volatile
  headers such as `date`, `user-agent` and the SDKs' `x-stainless-*` headers (OS, architecture,
  Python and SDK versions). `Redaction(drop_headers=[...])` removes such headers before they are
  stored; nothing is dropped by default, so the cassette shows what the SDK really sent.
- Stored request bodies are canonical, not the exact bytes sent. Matching compares canonical forms
  anyway; a custom matcher that needs exact bytes cannot get them from a cassette.
- Chunks split inside a multi-byte character appear as base64, which is exact but not readable.
- Timing values change on every re-recording, which is visible in diffs.
- Hand edits are possible, and a malformed edit fails at load time with a precise message.
