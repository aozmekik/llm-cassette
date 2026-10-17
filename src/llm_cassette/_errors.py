"""Exceptions raised by llm-cassette."""

from __future__ import annotations

from pathlib import Path

from llm_cassette._model import RecordedRequest


class CassetteError(Exception):
    """Base class for every error raised by llm-cassette."""


class CassetteNotFoundError(CassetteError):
    """The cassette file does not exist and the record mode does not allow creating it."""

    def __init__(self, path: Path) -> None:
        """Create the error for the missing ``path``."""
        self.path = path
        super().__init__(
            f"cassette {path} does not exist; record it first "
            "(mode 'once' or 'record', or --llm-record-mode=once with the pytest plugin)"
        )


class CassetteFormatError(CassetteError, ValueError):
    """A cassette file is malformed or uses an unsupported schema version."""


class SecretLeakError(CassetteError):
    """A configured secret value survived redaction, so the cassette was not written."""


class UnmatchedRequestError(CassetteError):
    """A live request matched no recorded interaction in a mode that cannot record.

    The message includes a diff against the closest recorded request.

    Attributes:
        request: The live request (after redaction) that did not match.
    """

    def __init__(self, message: str, *, request: RecordedRequest) -> None:
        """Create the error with its full, human-readable ``message``."""
        self.request = request
        super().__init__(message)
