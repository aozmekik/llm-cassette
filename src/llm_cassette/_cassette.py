"""The :class:`Cassette`: a file of recorded interactions plus the rules for using it."""

from __future__ import annotations

import math
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

from llm_cassette._errors import CassetteNotFoundError, SecretLeakError, UnmatchedRequestError
from llm_cassette._matching import (
    Candidate,
    Matching,
    RequestKey,
    describe_unmatched,
    index_key,
    mismatches,
    request_key,
)
from llm_cassette._model import (
    Chunk,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    normalize_headers,
)
from llm_cassette._redaction import Redaction
from llm_cassette._serialization import Format, dumps, loads

_FORMATS: dict[str, Format] = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}
_NEW_FILE_MODE = 0o644


class RecordMode(StrEnum):
    """When a cassette forwards requests to the real upstream.

    ============  ========================  ===============  =======================
    mode          cassette file missing     request matches  request does not match
    ============  ========================  ===============  =======================
    ``record``    record everything         record (fresh)   record (fresh)
    ``once``      record everything         replay           UnmatchedRequestError
    ``replay``    CassetteNotFoundError     replay           UnmatchedRequestError
    ``none``      treated as empty          replay           UnmatchedRequestError
    ============  ========================  ===============  =======================

    A cassette either records for its whole lifetime or replays for its whole lifetime; it never
    mixes the two. ``record`` discards the previous contents of the file. Only recording
    cassettes write to disk.
    """

    RECORD = "record"
    """Always call the upstream and rewrite the cassette with this session's interactions."""

    ONCE = "once"
    """Record when the cassette file does not exist yet; otherwise behave like ``replay``."""

    REPLAY = "replay"
    """Never call the upstream; the cassette file must exist."""

    NONE = "none"
    """Never call the upstream or write; a missing file is an empty cassette (use in CI)."""


class Recording:
    """An interaction being recorded: returned by :meth:`Cassette.resolve` in recording mode.

    HTTP adapters fill it in as the upstream response arrives: :meth:`start` with the status and
    headers, :meth:`add_chunk` for every body chunk, then :meth:`finish` once the body has been
    read to its end, or :meth:`abandon` if the upstream failed. A recording that is never
    finished is saved as incomplete (see :attr:`RecordedResponse.complete`). You only use this
    class directly when writing an adapter for another HTTP client; instances are created by the
    cassette.
    """

    def __init__(
        self, request: RecordedRequest, *, lock: threading.RLock, on_change: Callable[[], None]
    ) -> None:
        """Create a recording for ``request``; called by :class:`Cassette` only."""
        self._request = request
        self._lock = lock
        self._on_change = on_change
        self._response: RecordedResponse | None = None
        self._chunks: list[Chunk] = []
        self._complete = False
        self._abandoned = False

    @property
    def request(self) -> RecordedRequest:
        """The (redacted) request being recorded."""
        return self._request

    def start(self, status: int, headers: Iterable[tuple[str, str]], *, delay_ms: float) -> None:
        """Record the response status and headers, received ``delay_ms`` after sending."""
        with self._lock:
            self._response = RecordedResponse(
                status=status, headers=normalize_headers(headers), delay_ms=delay_ms
            )
            self._on_change()

    def add_chunk(self, data: bytes, *, delay_ms: float) -> None:
        """Append a body chunk received ``delay_ms`` after the previous event."""
        with self._lock:
            self._chunks.append(Chunk(data=data, delay_ms=delay_ms))
            self._on_change()

    def finish(self) -> None:
        """Record that the client read the body to its end, so nothing of it is missing."""
        with self._lock:
            self._complete = True
            self._on_change()

    def abandon(self) -> None:
        """Drop this recording, for example because the upstream request failed."""
        with self._lock:
            self._abandoned = True
            self._on_change()

    def snapshot(self) -> Interaction | None:
        """The interaction recorded so far, or ``None`` before the response headers arrive."""
        with self._lock:
            if self._abandoned or self._response is None:
                return None
            response = RecordedResponse(
                status=self._response.status,
                headers=self._response.headers,
                chunks=tuple(self._chunks),
                delay_ms=self._response.delay_ms,
                complete=self._complete,
            )
            return Interaction(request=self._request, response=response)


def _format_for(path: Path) -> Format:
    try:
        return _FORMATS[path.suffix.lower()]
    except KeyError:
        msg = f"cannot infer the cassette format of {path}: use a .yaml, .yml or .json suffix"
        raise ValueError(msg) from None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file readable by its owner only; keep an existing cassette's mode, and
    # give a new one the usual 0o644 rather than leaving it private.
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else _NEW_FILE_MODE
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(temporary).chmod(mode)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class Cassette:
    """Recorded HTTP interactions stored in one file, and the rules for replaying them.

    Example::

        with Cassette("tests/cassettes/hello.yaml", mode="once") as cassette:
            client = httpx2.Client(transport=CassetteTransport(cassette))
            client.get("https://example.com")

    The cassette is thread-safe: one cassette may serve several clients and threads at once.
    Identical requests are answered by identical recorded interactions in recording order.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        mode: RecordMode | str = RecordMode.ONCE,
        matching: Matching | None = None,
        redaction: Redaction | None = None,
        timing_scale: float = 0.0,
        allow_repeats: bool = False,
    ) -> None:
        """Open the cassette at ``path`` (loading it unless it will be re-recorded).

        Args:
            path: Cassette file; ``.yaml``/``.yml`` selects YAML and ``.json`` selects JSON.
            mode: A :class:`RecordMode` or its string value.
            matching: How live requests are matched against recorded ones.
            redaction: What to strip before anything is stored.
            timing_scale: Multiplier for recorded delays during replay: ``0`` (the default)
                replays instantly, ``1`` reproduces the recorded timing.
            allow_repeats: Let a recorded interaction answer more than one live request once
                every matching interaction has been played.

        Raises:
            CassetteNotFoundError: In ``replay`` mode, if ``path`` does not exist.
            CassetteFormatError: If the file is malformed or has an unsupported schema version.
            ValueError: For an unknown mode, file suffix or a negative ``timing_scale``.
        """
        self._path = Path(path)
        self._format = _format_for(self._path)
        self._mode = RecordMode(mode)
        if not math.isfinite(timing_scale) or timing_scale < 0:
            msg = f"timing_scale must be a finite number >= 0, got {timing_scale!r}"
            raise ValueError(msg)
        self._timing_scale = float(timing_scale)
        self._matching = matching if matching is not None else Matching()
        self._redaction = redaction if redaction is not None else Redaction()
        self._allow_repeats = allow_repeats

        exists = self._path.exists()
        if self._mode is RecordMode.REPLAY and not exists:
            raise CassetteNotFoundError(self._path)
        self._file_existed = exists
        self._recording = self._mode is RecordMode.RECORD or (
            self._mode is RecordMode.ONCE and not exists
        )
        self._replayable: tuple[Interaction, ...] = ()
        if exists and not self._recording:
            text = self._path.read_text(encoding="utf-8")
            self._replayable = tuple(loads(text, self._format, source=str(self._path)))
        self._keys: list[RequestKey] | None = None
        self._index: dict[tuple[object, ...], list[int]] | None = None
        self._recordings: list[Recording] = []
        self._played: list[Interaction | Recording] = []
        self._used: set[int] = set()
        self._unmatched: list[UnmatchedRequestError] = []
        self._lock = threading.RLock()
        # A `record` session must rewrite an existing file even if it makes no requests.
        self._version = 1 if self._mode is RecordMode.RECORD and exists else 0
        self._saved_version = 0

    def __repr__(self) -> str:
        """Show the path, mode and size, never the contents."""
        return (
            f"Cassette({str(self._path)!r}, mode={self._mode.value!r}, "
            f"recording={self._recording}, interactions={len(self.interactions)})"
        )

    def __enter__(self) -> Self:
        """Return the cassette itself; :meth:`save` runs on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Save the cassette, even when the block raised (what was recorded is still valid)."""
        self.save()

    @property
    def path(self) -> Path:
        """The cassette file."""
        return self._path

    @property
    def mode(self) -> RecordMode:
        """The record mode this cassette was opened with."""
        return self._mode

    @property
    def is_recording(self) -> bool:
        """Whether requests go to the upstream (``record``, or ``once`` with no file yet)."""
        return self._recording

    @property
    def timing_scale(self) -> float:
        """Multiplier applied to recorded delays during replay."""
        return self._timing_scale

    @property
    def matching(self) -> Matching:
        """The matching configuration."""
        return self._matching

    @property
    def redaction(self) -> Redaction:
        """The redaction configuration."""
        return self._redaction

    @property
    def interactions(self) -> tuple[Interaction, ...]:
        """Every interaction in the cassette, redacted, in recording order.

        While recording, this includes interactions whose responses are still streaming.
        """
        with self._lock:
            if not self._recording:
                return self._replayable
            snapshots = (recording.snapshot() for recording in self._recordings)
            return tuple(self._redaction.redact(i) for i in snapshots if i is not None)

    @property
    def played(self) -> tuple[Interaction, ...]:
        """The interactions served during this session (replayed or recorded), in order."""
        with self._lock:
            served: list[Interaction] = []
            for entry in self._played:
                interaction = entry.snapshot() if isinstance(entry, Recording) else entry
                if interaction is not None:
                    served.append(self._redaction.redact(interaction))
            return tuple(served)

    @property
    def unmatched(self) -> tuple[UnmatchedRequestError, ...]:
        """Every :class:`UnmatchedRequestError` raised by this cassette.

        SDKs often wrap transport errors (and may retry them), so this is the reliable place to
        check for unmatched requests; the pytest plugin fails the test if any are left here.
        """
        with self._lock:
            return tuple(self._unmatched)

    def resolve(self, request: RecordedRequest) -> Interaction | Recording:
        """Decide how to answer a live request.

        This is the hook HTTP adapters call. ``request`` is redacted first, so secrets never
        reach the matcher or the cassette.

        Returns:
            The recorded :class:`Interaction` to replay, or a :class:`Recording` to fill in while
            forwarding the request upstream (only when :attr:`is_recording`).

        Raises:
            UnmatchedRequestError: When replaying and no unused recorded interaction matches.
        """
        live = self._redaction.redact_request(request)
        with self._lock:
            if self._recording:
                recording = Recording(live, lock=self._lock, on_change=self._changed)
                self._recordings.append(recording)
                self._played.append(recording)
                self._changed()
                return recording
            return self._replay(live)

    def _replay(self, live: RecordedRequest) -> Interaction:
        if self._index is None:
            self._keys = [request_key(self._matching, i.request) for i in self._replayable]
            self._index = {}
            for position, key in enumerate(self._keys):
                self._index.setdefault(index_key(self._matching, key), []).append(position)
        live_key = request_key(self._matching, live)
        # The index narrows the search to interactions whose built-in aspects are equal, so the
        # cost of a hit does not grow with the cassette; custom matchers only see that bucket.
        bucket = self._index.get(index_key(self._matching, live_key), [])
        matching_indexes = [
            position
            for position in bucket
            if all(
                match(live, self._replayable[position].request) for match in self._matching.custom
            )
        ]
        unused = next((p for p in matching_indexes if p not in self._used), None)
        if unused is not None:
            self._used.add(unused)
        elif matching_indexes and self._allow_repeats:
            unused = matching_indexes[-1]
        if unused is not None:
            interaction = self._replayable[unused]
            self._played.append(interaction)
            return interaction
        raise self._unmatched_error(live, live_key, exhausted=len(matching_indexes))

    def _unmatched_error(
        self, live: RecordedRequest, live_key: RequestKey, *, exhausted: int
    ) -> UnmatchedRequestError:
        keys = self._keys or []
        candidates = [
            Candidate(position, interaction, reasons)
            for position, (interaction, key) in enumerate(zip(self._replayable, keys, strict=True))
            if (reasons := mismatches(self._matching, live, live_key, interaction.request, key))
        ]
        description = f"{self._path} (mode={self._mode.value}, "
        description += (
            f"{len(self._replayable)} recorded, {len(self._used)} played)"
            if self._file_existed
            else "file does not exist)"
        )
        message = describe_unmatched(
            self._matching, live, candidates, cassette=description, exhausted=exhausted
        )
        error = UnmatchedRequestError(message, request=live)
        self._unmatched.append(error)
        return error

    def _changed(self) -> None:
        with self._lock:
            self._version += 1

    def save(self) -> bool:
        """Write the cassette if this session recorded anything new.

        Replaying cassettes never write. The file is replaced atomically, so an interrupted
        save never leaves a half-written cassette behind.

        Returns:
            Whether the file was written.

        Raises:
            SecretLeakError: If a configured secret survived redaction; nothing is written.
        """
        with self._lock:
            if not self._recording or self._version == self._saved_version:
                return False
            version = self._version
            interactions = self.interactions
            if not interactions and not self._file_existed:
                self._saved_version = version
                return False
            location = self._redaction.find_secret(interactions)
            if location is not None:
                msg = (
                    f"refusing to write {self._path}: a configured secret appears in {location} "
                    "after redaction"
                )
                raise SecretLeakError(msg)
            _atomic_write(self._path, dumps(interactions, self._format))
            self._file_existed = True
            self._saved_version = version
            return True
