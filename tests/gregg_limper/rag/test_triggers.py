from types import SimpleNamespace

from gregg_limper.memory.rag.triggers import (
    TriggerSet,
    build_trigger_set,
    emoji_matches_trigger,
    message_has_trigger_reaction,
)


class FakeEmoji:
    def __init__(self, name: str | None = None, emoji_id: int | None = None):
        self.name = name
        self.id = emoji_id

    def __str__(self) -> str:
        if self.name is None or self.id is None:
            return super().__str__()
        return f"<:{self.name}:{self.id}>"


def test_build_trigger_set_parses_mixed_entries():
    triggers = build_trigger_set(
        ["🧠", "<:brain:12345>", "sparkle:67890", "98765", ":thumbsup:", "rocket"]
    )

    assert isinstance(triggers, TriggerSet)
    assert triggers.unicode_emojis == {"🧠"}
    assert triggers.custom_ids == {12345, 67890, 98765}
    assert triggers.custom_names == {"brain", "sparkle", "thumbsup", "rocket"}


def test_emoji_matches_trigger_unicode_and_custom():
    triggers = build_trigger_set(["🧠", "brain:12345", ":thumbsup:"])

    assert emoji_matches_trigger("🧠", triggers)
    assert not emoji_matches_trigger("💡", triggers)

    assert emoji_matches_trigger(FakeEmoji("brain", 12345), triggers)
    assert emoji_matches_trigger(FakeEmoji("thumbsup", None), triggers)
    assert not emoji_matches_trigger(FakeEmoji("wave", 4321), triggers)


def test_message_has_trigger_reaction_detects_any_match():
    triggers = build_trigger_set(["🧠"])

    msg = SimpleNamespace(
        reactions=[
            SimpleNamespace(emoji="💡"),
            SimpleNamespace(emoji="🧠"),
        ]
    )

    assert message_has_trigger_reaction(msg, triggers=triggers)

