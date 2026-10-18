# 5. Store a body the client did not finish reading as incomplete

- Status: accepted
- Date: 2026-10-18

## Context

The transports record a response while it streams to the client, so the recording ends when the
client stops reading. That can happen in three ways:

1. the upstream body ends (the normal case);
2. the upstream fails half-way (a reset connection): the adapters abandon the recording, because
   errors are not recorded;
3. the client stops first: it breaks out of a stream early, never reads the body, is interrupted
   by Ctrl-C, or is cancelled (`anyio.fail_after`, an asyncio timeout).

Before this decision, case 3 kept the chunks read so far and saved them as an ordinary
response. Replaying it served a body that ended early with no error, so a recording made by a
run that failed (a timeout, Ctrl-C) could make the next run pass, and code that later read more
of the stream silently got less. Two SDK behaviours matter here:

- `openai`'s sync client stops reading at `data: [DONE]` and closes the response, so the
  transport never sees the end of the body even though nothing else follows in practice.
- `openai`'s async client (3.34.0; 3.0.0 stops at `[DONE]` too) drains the body after `[DONE]`
  for at most 0.1 s (inside `anyio.move_on_after`), and ignores transport errors while doing so;
  `anthropic` reads to the end.

Options considered:

1. **Drain the rest of the upstream body on close**, as recording libraries that read the whole
   response up front effectively do. The cassette gets the full body, but closing a long LLM
   stream early would wait for (and pay for) the rest of the generation, and Ctrl-C or a
   cancellation would block on network reads.
2. **Abandon the recording** in case 3. Simple, but `openai`'s sync streams would never be
   recorded, and an interrupted run would replay as an unmatched request with a confusing diff.
3. **Abandon only on Ctrl-C and cancellation.** Cancellation cannot be told apart from the SDK's
   own bounded drain, so valid recordings would be lost.
4. **Record how the body ended.** Keep the chunks, mark the response `complete: false`, and make
   replay fail only when a client reads past the recorded end.

## Decision

Use option 4. A `Recording` is complete only when the adapter sees the upstream body end while
the client reads it (`Recording.finish()`). Anything else that stops the body, apart from an
upstream exception, leaves it incomplete, and the cassette stores `complete: false` on that
response (written only when false, so complete responses look as before). Replaying an
incomplete response serves the recorded chunks and then raises the library's `ReadError` with a
message that names the request and says to re-record. A transport error, rather than an
llm-cassette exception, is what a real connection would give and what SDKs already handle: the
async `openai` drain ignores it, and code that needs the rest of the body fails loudly.

Version 1 of the cassette schema had not been released when the field was added, so it stays
version 1; readers reject unknown keys, so a future field will need a new version.

## Consequences

- A truncated recording can no longer pass for a whole response, and a failed recording run does
  not turn into a passing replay.
- A client that stops at the same point during replay as during recording behaves exactly as it
  did, so early-exit code paths stay testable.
- Streamed chat completions recorded with `openai`'s sync client carry `complete: false`. That is
  accurate (the client never read to the end) and harmless, but visible in every such cassette,
  so the README explains it.
- Closing early never blocks and never costs extra tokens, but the cassette lacks whatever the
  client did not read; re-recording is the way to get it.
- Adapters for other HTTP clients must call `Recording.finish()` when the body ends.
