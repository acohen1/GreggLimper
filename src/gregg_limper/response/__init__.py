"""Entry-point helpers for generating replies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import discord

from gregg_limper.clients import oai, ollama
from gregg_limper.config import core, local_llm
from gregg_limper.tools import (
    ToolContext,
    ToolExecutionError,
    get_registered_tool_specs,
)
from gregg_limper.tools.executor import execute_tool

from .pipeline import build_prompt_payload

logger = logging.getLogger(__name__)


async def handle(message: discord.Message) -> str:
    """Generate a reply using the prompt pipeline."""

    payload = await build_prompt_payload(message)

    tool_specs = get_registered_tool_specs()
    use_tools = bool(tool_specs) and not local_llm.USE_LOCAL

    messages = list(payload.messages)

    if local_llm.USE_LOCAL or not use_tools:
        result_text = (
            await ollama.chat(messages, model=local_llm.LOCAL_MODEL_ID)
            if local_llm.USE_LOCAL
            else await oai.chat(messages, model=core.MSG_MODEL_ID)
        )
        _write_debug_payload(payload, messages)
        return result_text

    result_text, final_messages = await _run_with_tools(
        messages=messages,
        tool_specs=tool_specs,
        context=ToolContext(
            guild_id=getattr(message.guild, "id", None),
            channel_id=getattr(message.channel, "id", None),
            message_id=getattr(message, "id", None),
        ),
    )
    _write_debug_payload(payload, final_messages)
    return result_text


def _write_debug_payload(payload, final_messages: Iterable[dict[str, str]]) -> None:
    _write_debug_file("debug_history.md", payload.history.messages)
    _write_debug_file("debug_context.md", payload.context.user_profiles)
    _write_debug_file("debug_messages.json", list(final_messages), json_dump=True)


async def _run_with_tools(*, messages, tool_specs, context: ToolContext) -> tuple[str, list[dict[str, str]]]:
    openai_tools = [spec.to_openai() for spec in tool_specs]
    conversation = list(messages)
    max_iters = 5

    cached_tool_results: dict[tuple[str, str], str] = {}

    for _ in range(max_iters):
        resp = await oai.chat_full(
            conversation,
            model=core.MSG_MODEL_ID,
            tools=openai_tools,
        )
        choice = resp.choices[0]
        message = choice.message
        assistant_content = message.content or ""
        tool_calls = getattr(message, "tool_calls", None) or []

        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }

        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": getattr(call, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(call.function, "name", None),
                        "arguments": getattr(call.function, "arguments", "{}"),
                    },
                }
                for call in tool_calls
            ]

        conversation.append(assistant_entry)

        if not tool_calls:
            return assistant_content.strip(), conversation

        for call in tool_calls:
            name = getattr(call.function, "name", None)
            arguments = getattr(call.function, "arguments", "{}")
            if not name:
                continue
            logger.info(
                "Executing tool call id=%s name=%s arguments=%s",
                getattr(call, "id", None),
                name,
                arguments,
            )
            cache_key = (name, arguments)
            if cache_key in cached_tool_results:
                content = cached_tool_results[cache_key]
            else:
                try:
                    result = await execute_tool(name, arguments, context=context)
                    content = result.content
                except ToolExecutionError as exc:
                    content = f"Tool '{name}' failed: {exc}"
                cached_tool_results[cache_key] = content
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", None),
                    "name": name,
                    "content": content,
                }
            )

    raise RuntimeError("Exceeded maximum tool call iterations")


def _write_debug_file(
    filename: str, data, json_dump: bool = False
) -> None:  # pragma: no cover - debug helper
    try:
        base = Path("data/runtime")
        base.mkdir(parents=True, exist_ok=True)
        path = base / filename
        if json_dump:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            with path.open("w", encoding="utf-8") as handle:
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            role = item.get("role", "?")
                            content = item.get("content", "")
                            handle.write(f"{role}: {content}\n\n")
                        else:
                            handle.write(f"{item}\n")
                else:
                    handle.write(str(data))
    except Exception as exc:
        logger.debug("Failed to write %s: %s", filename, exc)


__all__ = ["handle", "_run_with_tools"]
