"""A demo suite: run it with ``uv run pytest examples/pytest_demo --llm-record-mode=none``.

The cassettes under ``cassettes/`` were recorded against the synthetic ``fake_upstream``
fixture. Delete them (or pass ``--llm-record-mode=record``) to record them again.
"""

from __future__ import annotations

import uuid

import httpx2
import pytest

from llm_cassette import Cassette
from llm_cassette.clients import anthropic_client, openai_client

# The upstream is fake, so is the key. With a real upstream, leave api_key out: the SDK reads
# OPENAI_API_KEY / ANTHROPIC_API_KEY when recording, and replaying needs no key at all.
FAKE_KEY = "sk-demo-not-a-real-key"


def test_openai_chat(llm_cassette: Cassette, fake_upstream: httpx2.MockTransport) -> None:
    client = openai_client(llm_cassette, transport=fake_upstream, api_key=FAKE_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Merhaba"}],
        metadata={"run_id": str(uuid.uuid4())},  # ignored by matching, see conftest.py
        user="someone@example.com",  # redacted before it is written, see conftest.py
    )
    assert completion.choices[0].message.content == "Echo: Merhaba"


@pytest.mark.llm_cassette(timing_scale=0.0)
def test_openai_stream(llm_cassette: Cassette, fake_upstream: httpx2.MockTransport) -> None:
    client = openai_client(llm_cassette, transport=fake_upstream, api_key=FAKE_KEY)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "akış testi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text = " ".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == "Echo: akış testi"


@pytest.mark.llm_cassette("anthropic_message")
def test_anthropic(llm_cassette: Cassette, fake_upstream: httpx2.MockTransport) -> None:
    client = anthropic_client(llm_cassette, transport=fake_upstream, api_key=FAKE_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "Selam"}],
    )
    block = message.content[0]
    assert block.type == "text"
    assert block.text == "Echo: Selam"
