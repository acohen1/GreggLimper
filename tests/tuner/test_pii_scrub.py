from types import SimpleNamespace

from tuner.runner import _scrub_conversations
from tuner.util.alias import AliasGenerator


def test_scrub_conversations_aliases_content():
    author = SimpleNamespace(id=1, display_name="Alex", name="Alex")
    mention = SimpleNamespace(id=2, display_name="Mahik", name="Mahik")
    message = SimpleNamespace(
        author=author,
        clean_content="hello <@2>",
        content="hello <@2>",
        mentions=[mention],
    )
    convo = SimpleNamespace(messages=[message])

    alias_gen = AliasGenerator()
    alias_map = _scrub_conversations([convo], alias_gen)

    assert getattr(message, "_pii_author_alias") != "Alex"
    assert "<@2>" not in getattr(message, "_pii_clean_content")
    assert getattr(message, "_pii_mentions_data")[0]["display_name"] != "Mahik"
    assert alias_map[getattr(author, "id")] == author.display_name
    assert alias_map[getattr(mention, "id")] == mention.display_name
