"""Record and replay LLM API calls at the httpx transport layer.

The core is independent of any HTTP library; adapters live in :mod:`llm_cassette.httpx2`
(``openai>=3``, ``anthropic>=1``) and :mod:`llm_cassette.httpx`. SDK client helpers live in
:mod:`llm_cassette.clients`, token usage in :mod:`llm_cassette.usage`, and the pytest plugin in
:mod:`llm_cassette.pytest_plugin`.
"""

from llm_cassette._cassette import Cassette, Recording, RecordMode
from llm_cassette._errors import (
    CassetteError,
    CassetteFormatError,
    CassetteNotFoundError,
    SecretLeakError,
    UnmatchedRequestError,
)
from llm_cassette._matching import Matcher, Matching
from llm_cassette._model import Chunk, Headers, Interaction, RecordedRequest, RecordedResponse
from llm_cassette._redaction import (
    DEFAULT_REDACTED_HEADERS,
    DEFAULT_SENSITIVE_SUBSTRINGS,
    Redaction,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_REDACTED_HEADERS",
    "DEFAULT_SENSITIVE_SUBSTRINGS",
    "Cassette",
    "CassetteError",
    "CassetteFormatError",
    "CassetteNotFoundError",
    "Chunk",
    "Headers",
    "Interaction",
    "Matcher",
    "Matching",
    "RecordMode",
    "RecordedRequest",
    "RecordedResponse",
    "Recording",
    "Redaction",
    "SecretLeakError",
    "UnmatchedRequestError",
    "__version__",
]
