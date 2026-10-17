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
  decoded (so `content-encoding` and `content-length` are dropped), each chunk with its delay in
  milliseconds rounded to 0.1. A chunk that is valid UTF-8 is stored as text, anything else as
  base64, per chunk, so a multi-byte character split across two network reads stays exact.
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
  write a file that still contains a configured secret.

## Consequences

- Diffs are small and readable: re-recording the same traffic changes only the measured delays.
- Stored request bodies are canonical, not the exact bytes sent. Matching compares canonical forms
  anyway; a custom matcher that needs exact bytes cannot get them from a cassette.
- Chunks split inside a multi-byte character appear as base64, which is exact but not readable.
- Timing values change on every re-recording, which is visible in diffs.
- Hand edits are possible, and a malformed edit fails at load time with a precise message.
