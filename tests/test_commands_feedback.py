from types import SimpleNamespace

import pytest

from gregg_limper import commands


class FakeMessage(SimpleNamespace):
    pass


@pytest.mark.parametrize(
    "content",
    [
        "Opted in to RAG. Backfill queued.",
        "Already opted in.",
        "Backfill complete. Ingested 3 messages.",
        "Opted out and data purged from RAG.",
        "You are opted in.",
        "You are not opted in.",
        "Available commands: help, rag_opt_in",
        "Initiating lobotomy sequence...",
    ],
)
def test_is_command_feedback_matches_registered_messages(content):
    bot_user = SimpleNamespace(id=42, bot=True)
    message = FakeMessage(
        content=content,
        author=SimpleNamespace(id=42, bot=True),
    )

    assert commands.is_command_feedback(message, bot_user=bot_user)


def test_is_command_feedback_rejects_non_bot_messages():
    bot_user = SimpleNamespace(id=42, bot=True)
    message = FakeMessage(
        content="Available commands: help",
        author=SimpleNamespace(id=99, bot=False),
    )

    assert not commands.is_command_feedback(message, bot_user=bot_user)
