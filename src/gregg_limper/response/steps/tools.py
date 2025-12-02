"""
Pipeline step for decoupled tool execution.
"""
from __future__ import annotations

import logging
from typing import Any

from gregg_limper.clients import oai
from gregg_limper.config import core
from gregg_limper.response.engine import PipelineContext, PipelineStep
from gregg_limper.tools import ToolExecutionError, get_registered_tool_specs
from gregg_limper.tools.executor import execute_tool

logger = logging.getLogger(__name__)


class ToolExecutionStep(PipelineStep):
    """
    Analyzes the conversation context using a smart model to decide on and execute tools.
    Results are injected into the context for the subsequent generation step.
    """
    
    async def run(self, context: PipelineContext) -> PipelineContext:
        tool_specs = get_registered_tool_specs()
        if not tool_specs:
            return context
            
        # If no tool check model is configured, we skip this step
        # (or we could fallback to the old behavior, but the goal is decoupling)
        if not core.TOOL_CHECK_MODEL_ID:
            logger.warning("No TOOL_CHECK_MODEL_ID configured, skipping decoupled tool execution.")
            return context

        logger.debug("Running decoupled tool execution check.")
        
        # We use the smart model to check for tool usage
        # We pass the current messages (history + context)
        # We do NOT pass the system prompt of the persona, but maybe a system prompt for the tool checker?
        # For now, let's just use the messages as is, but maybe prepend a system instruction
        
        tool_check_messages = [
            {"role": "system", "content": "You are a helpful assistant. Analyze the conversation and use tools if necessary to answer the user's request."}
        ] + context.messages
        
        openai_tools = [spec.to_openai() for spec in tool_specs]
        
        try:
            # We allow a few iterations of tool calls (e.g. tool A -> tool B)
            # But we do NOT generate the final response here.
            # We only execute tools and capture the results.
            
            # We maintain a local conversation list for the tool checker
            checker_conversation = list(tool_check_messages)
            
            max_iters = 5
            tools_executed = False
            
            for _ in range(max_iters):
                resp = await oai.chat_full(
                    checker_conversation,
                    model=core.TOOL_CHECK_MODEL_ID,
                    tools=openai_tools,
                )
                choice = resp.choices[0]
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None) or []
                
                if not tool_calls:
                    # No more tools needed
                    break
                    
                tools_executed = True
                
                # Append assistant message with tool calls to checker history
                assistant_entry = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ]
                }
                checker_conversation.append(assistant_entry)
                
                # Execute tools
                for call in tool_calls:
                    name = call.function.name
                    arguments = call.function.arguments
                    
                    logger.info("Decoupled tool execution: %s(%s)", name, arguments)
                    
                    try:
                        result = await execute_tool(name, arguments, context=context.tool_context)
                        content = result.context_content
                        if result.response_content:
                            context.response_fragments.append(result.response_content)
                    except ToolExecutionError as exc:
                        content = f"Tool '{name}' failed: {exc}"
                        
                    # Append result to checker history
                    checker_conversation.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": content
                    })
                    
                    # INJECT RESULT INTO MAIN CONTEXT
                    # We inject this as a SYSTEM message so the finetune sees it as authoritative context
                    # "Tool 'name' returned: content"
                    context.messages.append({
                        "role": "system",
                        "content": f"Tool '{name}' returned:\n{content}"
                    })
            
            if tools_executed:
                logger.info("Decoupled tool execution complete. Results injected.")
                
        except Exception as e:
            logger.error("Error during decoupled tool execution: %s", e)
            
        return context
