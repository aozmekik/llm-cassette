# 6. One cassette per test, with collision-free names and an ownership check

- Status: accepted
- Date: 2026-10-18

## Context

The pytest plugin derives each test's cassette path from its node id
(`<test dir>/cassettes/<module>/<name>.yaml`). The name has to be a portable file name, so
characters outside `[A-Za-z0-9_.-[]]` are replaced. Replacing alone maps different tests to one
file, and LLM suites make that likely: they parametrize over prompts, and the ids `What is 2+2?`
and `What is 2*2?` both become `test_x[What_is_2_2_]`. On case-insensitive file systems (macOS,
Windows), where cassettes are committed and checked out, `test_A` and `test_a` share a file too.
An absolute `llm_cassette_dir` used to put `tests/unit/test_chat.py` and `tests/e2e/test_chat.py`
in the same folder, and the `name` option invited sharing one cassette between tests on purpose.

Sharing breaks in both directions. In `once` mode the first test records and the second replays
the first test's traffic and fails as unmatched. In `record` mode the second test overwrites the
first, so the next `none` run fails although re-recording looked green.

Options considered:

1. **Document the pitfalls** and keep sharing. Cheap, but the failure is silent until CI.
2. **Append within a session** when tests share a file. Makes sharing work for `record`, but the
   cassette no longer reflects one test, and the outcome depends on which tests were selected.
3. **Disambiguate automatically** when a collision is seen (a numeric suffix). The path would
   depend on test order and selection (`-k`), so a cassette recorded in one run would not be
   found in another.
4. **Make names collision-free by construction, and reject what is left.** Append a short hash
   of the original name whenever sanitizing changes it, mirror the test tree under an absolute
   directory, and fail at setup when two node ids still resolve to the same path, compared
   case-insensitively.

## Decision

Use option 4. `_safe_filename` keeps safe names unchanged (`TestChat.test_stream[gpt-4o-mini]`)
and appends the first 8 hex digits of the SHA-256 of the original name when it had to change one
(`test_x[What_is_2_2_]-1a2b3c4d`), or when the name is longer than 120 characters. The fixture
records which test owns each case-folded path in the session's stash and fails the second test
with a message naming both and suggesting `@pytest.mark.llm_cassette(name=...)`. The same test
claiming its own path again (a rerun) is allowed.

## Consequences

- Derived names cannot collide except through case, which is reported at setup, before any
  request is made.
- Cassette paths depend only on the test's own node id, never on which other tests ran.
- `name` gives a test a stable, readable file name; it can no longer share a cassette. Tests that
  want the same recorded traffic can construct a `Cassette` themselves.
- Names that needed sanitizing carry a hash suffix, which is less readable but stable.
- Under pytest-xdist each worker checks its own tests, so a collision between tests that run on
  different workers is only reported when they happen to run on the same one.
