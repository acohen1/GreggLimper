"""
Pipeline step for initial response generation.
"""
from __future__ import annotations

import logging
from typing import Any

from gregg_limper.clients import oai, ollama
from gregg_limper.config import core, local_llm
from gregg_limper.response.engine import PipelineContext, PipelineStep
from gregg_limper.tools import ToolExecutionError, get_registered_tool_specs
from gregg_limper.tools.executor import execute_tool
import json

logger = logging.getLogger(__name__)


class GenerationStep(PipelineStep):
    """
    Generates the initial response from the model, handling tool calls if necessary.
    """
    
    async def run(self, context: PipelineContext) -> PipelineContext:
        # Initialize messages from payload if empty
        if not context.messages:
            context.messages = list(context.payload.messages)

        tool_specs = get_registered_tool_specs()
        use_tools = bool(tool_specs) and not local_llm.USE_LOCAL

        if local_llm.USE_LOCAL or not use_tools:
            result_text = (
                await ollama.chat(context.messages, model=local_llm.LOCAL_MODEL_ID)
                if local_llm.USE_LOCAL
                else await oai.chat(context.messages, model=core.MSG_MODEL_ID)
            )
            context.response_text = result_text
            context.messages.append({"role": "assistant", "content": result_text})
            return context

        # Inline tool-calling loop using the main model
        openai_tools = [spec.to_openai() for spec in tool_specs]
        conversation = list(context.messages)
        executed_tools_log: list[dict[str, Any]] = []
        cached_tool_results: dict[tuple[str, str], Any] = {}
        max_iters = 5
        last_assistant_content = ""

        for _ in range(max_iters):
            resp = await oai.chat_full(
                conversation,
                model=core.MSG_MODEL_ID,
                tools=openai_tools,
            )
            choice = resp.choices[0]
            message = choice.message
            assistant_content = message.content or ""
            last_assistant_content = assistant_content
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
                context.response_text = assistant_content.strip()
                context.messages = conversation
                context.step_metadata = {"tools_executed": executed_tools_log}
                return context

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
                cache_key = (name, _canonical_args(arguments))
                if cache_key in cached_tool_results:
                    content = cached_tool_results[cache_key]
                    response_fragment = None
                else:
                    try:
                        result = await execute_tool(name, arguments, context=context.tool_context)
                        content = result.tool_content
                        response_fragment = result.response_fragment
                    except ToolExecutionError as exc:
                        content = f"Tool '{name}' failed: {exc}"
                        response_fragment = None
                    cached_tool_results[cache_key] = content

                executed_tools_log.append({"name": name, "args": arguments})

                if response_fragment:
                    context.response_fragments.append(response_fragment)

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(call, "id", None),
                        "name": name,
                        "content": content,
                    }
                )

        # Fallback: return the last assistant content to avoid hard failure
        context.response_text = (last_assistant_content or "").strip()
        context.messages = conversation
        context.step_metadata = {
            "tools_executed": executed_tools_log,
            "tool_loop_exhausted": True,
        }
        return context


def _canonical_args(arguments: Any) -> str:
    """Normalize arguments for caching purposes."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return json.dumps(parsed, sort_keys=True)
        except Exception:
            return arguments.strip()
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True)
    return str(arguments)
