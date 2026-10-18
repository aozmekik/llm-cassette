from __future__ import annotations

import json
import os
import stat
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from llm_cassette import (
    Cassette,
    CassetteFormatError,
    CassetteNotFoundError,
    Interaction,
    Matching,
    RecordedRequest,
    RecordedResponse,
    Recording,
    RecordMode,
    UnmatchedRequestError,
)

URL = "https://api.example.com/v1/chat"


def req(prompt: str = "hi", *, url: str = URL) -> RecordedRequest:
    return RecordedRequest(
        "POST", url, (("authorization", "Bearer sk-x"),), json.dumps({"p": prompt}).encode()
    )


def record(
    cassette: Cassette, request: RecordedRequest, body: bytes = b"ok", status: int = 200
) -> None:
    recording = cassette.resolve(request)
    assert isinstance(recording, Recording)
    recording.start(status, [("content-type", "text/plain")], delay_ms=1.0)
    recording.add_chunk(body, delay_ms=2.0)


def bodies(cassette: Cassette, *requests: RecordedRequest) -> list[bytes]:
    out = []
    for request in requests:
        played = cassette.resolve(request)
        assert isinstance(played, Interaction)
        out.append(played.response.body)
    return out


@pytest.fixture
def recorded(cassette_path: Path) -> Path:
    """A cassette file with two answers to "hi" (in order) and one to "bye"."""
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req("hi"), b"first")
        record(cassette, req("bye"), b"bye!")
        record(cassette, req("hi"), b"second")
    return cassette_path


# ----------------------------------------------------------------------------------- modes


def test_record_writes_a_new_cassette(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode="record")
    assert cassette.is_recording
    record(cassette, req())
    assert not cassette_path.exists()
    assert cassette.save()
    assert cassette_path.exists()
    assert not cassette.save(), "nothing changed since the last save"


def test_record_replaces_previous_contents(recorded: Path) -> None:
    with Cassette(recorded, mode="record") as cassette:
        assert cassette.interactions == ()
        record(cassette, req("new"), b"new")
    assert [i.response.body for i in Cassette(recorded, mode="none").interactions] == [b"new"]


def test_record_without_requests_empties_an_existing_cassette(recorded: Path) -> None:
    with Cassette(recorded, mode="record"):
        pass
    assert Cassette(recorded, mode="none").interactions == ()


def test_recording_nothing_creates_no_file(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="once"):
        pass
    assert not cassette_path.exists()


def test_once_records_when_missing_then_replays(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="once") as first:
        assert first.is_recording
        record(first, req())
    second = Cassette(cassette_path, mode="once")
    assert not second.is_recording
    assert bodies(second, req()) == [b"ok"]
    with pytest.raises(UnmatchedRequestError):
        second.resolve(req("something new"))
    assert not second.save(), "a replaying cassette never writes"


def test_replay_requires_the_file(cassette_path: Path) -> None:
    with pytest.raises(CassetteNotFoundError, match="does not exist") as raised:
        Cassette(cassette_path, mode="replay")
    assert raised.value.path == cassette_path


def test_replay_and_none_never_record(recorded: Path) -> None:
    for mode in ("replay", "none"):
        cassette = Cassette(recorded, mode=mode)
        assert not cassette.is_recording
        assert bodies(cassette, req("bye")) == [b"bye!"]
        with pytest.raises(UnmatchedRequestError):
            cassette.resolve(req("unknown"))


def test_none_treats_a_missing_file_as_empty(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode=RecordMode.NONE)
    with pytest.raises(UnmatchedRequestError, match="file does not exist") as raised:
        cassette.resolve(req())
    assert "the cassette has no recorded interactions" in str(raised.value)
    assert cassette.save() is False
    assert not cassette_path.exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "rewind"}, "is not a valid RecordMode"),
        ({"timing_scale": -1}, "timing_scale"),
        ({"timing_scale": float("nan")}, "timing_scale"),
        ({"timing_scale": float("inf")}, "timing_scale"),
    ],
)
def test_invalid_arguments(cassette_path: Path, kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Cassette(cassette_path, **kwargs)  # type: ignore[arg-type]


def test_the_suffix_selects_the_format(tmp_path: Path) -> None:
    for name in ("c.yaml", "c.YML", "c.json"):
        with Cassette(tmp_path / name, mode="record") as cassette:
            record(cassette, req())
    assert (tmp_path / "c.json").read_text().startswith("{")
    assert (tmp_path / "c.YML").read_text().startswith("version: 1")
    with pytest.raises(ValueError, match=r"use a \.yaml, \.yml or \.json suffix"):
        Cassette(tmp_path / "c.txt")


def test_a_corrupt_cassette_is_reported(cassette_path: Path) -> None:
    cassette_path.parent.mkdir(parents=True)
    cassette_path.write_text("version: 7\ninteractions: []\n")
    with pytest.raises(CassetteFormatError, match="unsupported cassette schema version 7"):
        Cassette(cassette_path, mode="none")
    # Record mode never reads the old file, so it can repair a corrupt cassette.
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req())
    assert len(Cassette(cassette_path, mode="none").interactions) == 1


# ---------------------------------------------------------------------------------- replay


def test_identical_requests_replay_in_recording_order(recorded: Path) -> None:
    cassette = Cassette(recorded, mode="none")
    assert bodies(cassette, req("hi"), req("bye"), req("hi")) == [b"first", b"bye!", b"second"]
    with pytest.raises(
        UnmatchedRequestError, match=r"2 matching interaction\(s\) were already played"
    ):
        cassette.resolve(req("hi"))


def test_allow_repeats_replays_the_last_match(recorded: Path) -> None:
    cassette = Cassette(recorded, mode="none", allow_repeats=True)
    assert bodies(cassette, req("hi"), req("hi"), req("hi"), req("hi")) == [
        b"first",
        b"second",
        b"second",
        b"second",
    ]


def test_played_and_unmatched_are_tracked(recorded: Path) -> None:
    cassette = Cassette(recorded, mode="none")
    bodies(cassette, req("bye"), req("hi"))
    with pytest.raises(UnmatchedRequestError) as raised:
        cassette.resolve(req("nope"))
    assert [i.response.body for i in cassette.played] == [b"bye!", b"first"]
    assert cassette.unmatched == (raised.value,)
    assert raised.value.request.body == b'{"p": "nope"}'
    assert "2 played" in str(raised.value)


def test_matching_configuration_is_used(recorded: Path) -> None:
    cassette = Cassette(recorded, mode="none", matching=Matching(body=False))
    assert bodies(cassette, req("anything")) == [b"first"]
    assert cassette.matching == Matching(body=False)


def test_interactions_are_redacted(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode="record")
    record(cassette, req())
    (interaction,) = cassette.interactions
    assert interaction.request.headers == (("authorization", "REDACTED"),)
    assert cassette.played == cassette.interactions


# -------------------------------------------------------------------------------- recording


def test_abandoned_and_unstarted_recordings_are_not_saved(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        abandoned = cassette.resolve(req("failed"))
        assert isinstance(abandoned, Recording)
        abandoned.start(200, [], delay_ms=0)
        abandoned.abandon()
        unstarted = cassette.resolve(req("pending"))
        assert isinstance(unstarted, Recording)
        assert unstarted.snapshot() is None
        assert unstarted.request.url == URL
        record(cassette, req("kept"))
    (only,) = Cassette(cassette_path, mode="none").interactions
    assert only.request.json() == {"p": "kept"}


def test_unfinished_recordings_are_saved_as_incomplete(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        finished = cassette.resolve(req("finished"))
        assert isinstance(finished, Recording)
        finished.start(200, [], delay_ms=0)
        finished.add_chunk(b"all", delay_ms=0)
        finished.finish()
        record(cassette, req("unfinished"))
    first, second = Cassette(cassette_path, mode="none").interactions
    assert first.response.complete
    assert not second.response.complete
    assert second.response.body == b"ok"


def test_chunks_arriving_after_a_save_are_saved_next_time(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode="record")
    recording = cassette.resolve(req())
    assert isinstance(recording, Recording)
    recording.start(200, [], delay_ms=0)
    recording.add_chunk(b"a", delay_ms=0)
    assert cassette.save()
    recording.add_chunk(b"b", delay_ms=0)
    assert cassette.save()
    (saved,) = Cassette(cassette_path, mode="none").interactions
    assert saved.response.body == b"ab"


def test_saving_is_atomic(cassette_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req(), b"original")
    before = cassette_path.read_bytes()

    def explode(*_: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    cassette = Cassette(cassette_path, mode="record")
    record(cassette, req(), b"replacement")
    monkeypatch.setattr(Path, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        cassette.save()
    assert cassette_path.read_bytes() == before
    assert sorted(p.name for p in cassette_path.parent.iterdir()) == [cassette_path.name]


@pytest.fixture
def umask() -> Iterator[Callable[[int], int]]:
    original = os.umask(0o022)
    os.umask(original)
    yield os.umask
    os.umask(original)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_new_files_follow_the_umask(cassette_path: Path, umask: Callable[[int], int]) -> None:
    umask(0o022)
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req())
    assert stat.S_IMODE(cassette_path.stat().st_mode) == 0o644
    cassette_path.chmod(0o664)
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req("again"))
    assert stat.S_IMODE(cassette_path.stat().st_mode) == 0o664, "an existing mode is kept"

    private = cassette_path.with_name("private.yaml")
    umask(0o077)  # cassettes can hold prompts and customer data
    with Cassette(private, mode="record") as cassette:
        record(cassette, req())
    assert stat.S_IMODE(private.stat().st_mode) == 0o600


def test_saving_flushes_the_file_to_disk(
    cassette_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req())
    assert len(synced) == 1


def test_context_manager_saves_even_when_the_block_raises(cassette_path: Path) -> None:
    def fail_while_recording() -> None:
        with Cassette(cassette_path, mode="record") as cassette:
            record(cassette, req())
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail_while_recording()
    assert cassette_path.exists()


def test_repr_does_not_leak_contents(recorded: Path) -> None:
    text = repr(Cassette(recorded, mode="none"))
    assert text == f"Cassette({str(recorded)!r}, mode='none', recording=False, interactions=3)"


# ------------------------------------------------------------------------------ concurrency


def test_concurrent_replay_serves_each_interaction_once(cassette_path: Path) -> None:
    copies = 64
    with Cassette(cassette_path, mode="record") as cassette:
        for index in range(copies):
            record(cassette, req("same"), str(index).encode())
    cassette = Cassette(cassette_path, mode="none")
    barrier = threading.Barrier(16)
    served: list[bytes] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        for _ in range(copies // 16):
            body = bodies(cassette, req("same"))[0]
            with lock:
                served.append(body)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(served, key=int) == [str(i).encode() for i in range(copies)]


def test_concurrent_recording_keeps_every_interaction(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode="record")
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        barrier.wait()
        for i in range(25):
            record(cassette, req(f"{n}-{i}"), f"{n}-{i}".encode())
            if i % 10 == 0:
                cassette.save()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cassette.save()
    saved = Cassette(cassette_path, mode="none").interactions
    assert sorted(i.response.body for i in saved) == sorted(
        f"{n}-{i}".encode() for n in range(8) for i in range(25)
    )
    for interaction in saved:
        assert interaction.request.json() == {"p": interaction.response.body.decode()}


def test_response_helpers() -> None:
    response = RecordedResponse(200, (("content-type", "a"), ("x", "1"), ("x", "2")))
    assert response.header("X") == "1"
    assert response.header("missing") is None
    assert req().header("Authorization") == "Bearer sk-x"


def test_configuration_is_exposed(cassette_path: Path) -> None:
    cassette = Cassette(cassette_path, mode="none", timing_scale=2)
    assert cassette.path == cassette_path
    assert cassette.mode is RecordMode.NONE
    assert cassette.timing_scale == 2.0
    assert cassette.redaction.placeholder == "REDACTED"


def test_custom_matchers_pick_among_built_in_matches(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        for tenant in ("a", "b"):
            recording = cassette.resolve(
                RecordedRequest("POST", URL, (("x-tenant", tenant),), b'{"p": "hi"}')
            )
            assert isinstance(recording, Recording)
            recording.start(200, [], delay_ms=0)
            recording.add_chunk(tenant.encode(), delay_ms=0)
        record(cassette, req("other"))
    calls: list[str | None] = []

    def same_tenant(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        calls.append(recorded.header("x-tenant"))
        return live.header("x-tenant") == recorded.header("x-tenant")

    cassette = Cassette(cassette_path, mode="none", matching=Matching(custom=[same_tenant]))
    live = RecordedRequest("POST", URL, (("x-tenant", "b"),), b'{"p": "hi"}')
    assert bodies(cassette, live) == [b"b"]
    assert calls == ["a", "b"], "only interactions with an equal method, URL and body are tried"
    with pytest.raises(UnmatchedRequestError, match="1 matching interaction"):
        cassette.resolve(live)


def test_custom_matchers_never_see_requests_that_differ_in_a_built_in_aspect(
    cassette_path: Path,
) -> None:
    """The README's model matcher used to crash on a recorded GET while explaining a miss."""
    with Cassette(cassette_path, mode="record") as cassette:
        for request in (
            RecordedRequest("GET", "https://api.example.com/v1/models"),
            RecordedRequest("POST", URL, (), b'{"model": "a", "p": "hi"}'),
        ):
            recording = cassette.resolve(request)
            assert isinstance(recording, Recording)
            recording.start(200, [], delay_ms=0)
            recording.finish()
    seen: list[str] = []

    def same_model(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        seen.append(recorded.method)
        a, b = live.json(), recorded.json()
        return isinstance(a, dict) and isinstance(b, dict) and a.get("model") == b.get("model")

    cassette = Cassette(cassette_path, mode="none", matching=Matching(custom=[same_model]))
    with pytest.raises(UnmatchedRequestError, match="differs in: body"):
        cassette.resolve(RecordedRequest("POST", URL, (), b'{"model": "a", "p": "other"}'))
    assert seen == [], "no recorded request passed the built-in comparisons"
    assert len(cassette.unmatched) == 1


def test_a_custom_matcher_rejection_is_explained(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req())
    calls = 0

    def never(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        nonlocal calls
        calls += 1
        return live is recorded

    cassette = Cassette(cassette_path, mode="none", matching=Matching(custom=[never]))
    with pytest.raises(UnmatchedRequestError) as raised:
        cassette.resolve(req())
    assert "differs in: custom matcher " in str(raised.value)
    assert "never" in str(raised.value)
    assert calls == 1, "explaining the miss reuses the matcher's verdict"


def test_an_exception_in_a_custom_matcher_propagates(cassette_path: Path) -> None:
    with Cassette(cassette_path, mode="record") as cassette:
        record(cassette, req())

    def broken(live: RecordedRequest, recorded: RecordedRequest) -> bool:
        del live, recorded
        msg = "bug in the matcher"
        raise RuntimeError(msg)

    cassette = Cassette(cassette_path, mode="none", matching=Matching(custom=[broken]))
    with pytest.raises(RuntimeError, match="bug in the matcher"):
        cassette.resolve(req())
