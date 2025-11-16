import asyncio

import pytest

from gregg_limper.tools import ToolContext, get_registered_tool_specs
from gregg_limper.tools.executor import execute_tool


def test_registry_exposes_default_tool():
    names = {spec.name for spec in get_registered_tool_specs()}
    assert "retrieve_context" in names


def test_execute_tool_uses_rag(monkeypatch):
    async def fake_vector_search(guild_id, channel_id, query, k):
        assert guild_id == 42
        assert channel_id == 13
        assert query == "pizza"
        assert k == 2
        return [
            {"content": "Cheese pizza", "author_id": 1, "message_id": 100},
            {"content": "Pepperoni pizza", "author_id": 2, "message_id": 101},
        ]

    monkeypatch.setattr("gregg_limper.tools.handlers.rag.rag.vector_search", fake_vector_search)

    result = asyncio.run(
        execute_tool(
            "retrieve_context",
            {"query": "pizza", "k": 2},
            context=ToolContext(guild_id=42, channel_id=13, message_id=7),
        )
    )

    assert "Cheese pizza" in result.content
    assert "Pepperoni pizza" in result.content
