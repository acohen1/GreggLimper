"""
Logic for multi-pass response accumulation.
"""
from __future__ import annotations

import logging
from typing import Callable, Awaitable

from gregg_limper.clients import oai
from gregg_limper.config import core
from gregg_limper.response.pipeline import PromptPayload

logger = logging.getLogger(__name__)


async def check_completeness(response_text: str, history_messages: list[dict[str, str]]) -> bool:
    """
    Uses a lightweight model to determine if the response is complete.
    Returns True if complete, False if it needs more detail.
    """
    if not core.DETAIL_CHECK_MODEL_ID:
        return True

    return await oai.check_completeness(response_text, history_messages, core.DETAIL_CHECK_MODEL_ID)


async def accumulate_response(
    initial_response: str,
    payload: PromptPayload,
    runner: Callable[[list[dict[str, str]]], Awaitable[tuple[str, list[dict[str, str]]]]]
) -> str:
    """
    Iteratively accumulates the response until it is deemed complete.
    
    Args:
        initial_response: The text generated in the first pass.
        payload: The original prompt payload (history, context, etc.).
        runner: A async function that takes a list of messages and returns (response_text, final_messages).
                This should be `_run_with_tools` or a wrapper around the simple chat client.
    """
    current_response = initial_response
    
    # If no check model is configured, just return the initial response
    if not core.DETAIL_CHECK_MODEL_ID:
        return current_response

    loops = 0
    max_loops = core.DETAIL_CHECK_MAX_LOOPS
    
    # We need a mutable list of messages to append to
    # We start with the original messages from the payload
    # NOTE: payload.messages includes system prompt, tool specs, context, and history.
    # We shouldn't modify payload.messages in place if it's used elsewhere, but here we are consuming it.
    # We also need the history messages for the classifier context
    history_messages = payload.history.messages

    while loops < max_loops:
        is_complete = await check_completeness(current_response, history_messages)
        if is_complete:
            break
            
        loops += 1
        logger.info("Response incomplete, starting accumulation loop %d/%d", loops, max_loops)
        
        # Prepare for the next pass
        # We start fresh with the original payload messages each time
        current_messages = list(payload.messages)
        
        # We append the current response as an ASSISTANT message
        current_messages.append({"role": "assistant", "content": current_response})
        
        # Add a system instruction to encourage rewriting
        # We append this as a user message to guide the model
        current_messages.append({"role": "user", "content": oai.get_refinement_instruction()})
        
        # Run the pipeline again
        try:
            new_chunk, _ = await runner(current_messages)
            if new_chunk:
                # REWRITE STRATEGY: The new chunk IS the new response.
                # We do not append. We replace.
                current_response = new_chunk
            else:
                logger.info("Accumulation yielded empty response, stopping.")
                break
        except Exception as e:
            logger.error("Error during accumulation loop: %s", e)
            break
            
    return current_response
