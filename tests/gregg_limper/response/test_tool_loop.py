import asyncio

from gregg_limper.config import core
from gregg_limper.response import _run_with_tools
from gregg_limper.tools import ToolContext, ToolResult, ToolSpec


class DummyFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class DummyCall:
    def __init__(self, name: str, arguments: str, call_id: str):
        self.id = call_id
        self.function = DummyFunction(name, arguments)


class DummyMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class DummyChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "stop"


class DummyResponse:
    def __init__(self, message):
        self.choices = [DummyChoice(message)]


class DummyToolSpec(ToolSpec):
    def to_openai(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def test_run_with_tools_executes_loop(monkeypatch):
    specs = [DummyToolSpec(name="dummy", description="test", parameters={"type": "object", "properties": {}})]

    sequence = [
        DummyResponse(DummyMessage("", [DummyCall("dummy", "{}", "call-1")])),
        DummyResponse(DummyMessage("Final answer")),
    ]

    async def fake_chat_full(messages, model, tools):
        assert model == core.MSG_MODEL_ID
        return sequence.pop(0)

    async def fake_execute_tool(name, arguments, context):
        assert name == "dummy"
        return ToolResult(content="tool-output")

    monkeypatch.setattr("gregg_limper.clients.oai.chat_full", fake_chat_full)
    monkeypatch.setattr("gregg_limper.response.execute_tool", fake_execute_tool)

    text, conversation = asyncio.run(
        _run_with_tools(
            messages=[{"role": "system", "content": "sys"}],
            tool_specs=specs,
            context=ToolContext(guild_id=1, channel_id=2, message_id=3),
        )
    )

    assert text == "Final answer"
    roles = [msg["role"] for msg in conversation]
    assert "tool" in roles
