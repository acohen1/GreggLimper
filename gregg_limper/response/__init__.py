"""
responses/
==========
This package defines post-caching behaviors that trigger when the bot is mentioned
or otherwise flagged to respond. These actions do not bypass the formatting pipeline.

Use case:
- When the bot is mentioned in a message, we fetch recent context from GLCache,
  build a prompt, and generate a response via OpenAI.
"""

from __future__ import annotations
import discord
from gregg_limper.config import core, local_llm
from gregg_limper.clients import oai, ollama
from .prompt_builder import build_sys_prompt
from .history_builder import build_history

import logging

logger = logging.getLogger(__name__)

async def handle(message: discord.Message) -> str:
    """
    Generate a reply using a pre-built system prompt.
    """

    cache_msgs = await build_history(message.channel.id, core.CONTEXT_LENGTH)

    # Log history to file for debugging
    with open("debug_history.md", "w", encoding="utf-8") as f:
        for msg in cache_msgs:
            f.write(f"{msg['role']}: {msg['content']}\n\n")
        logger.debug(f"History written to debug_history.md")

    # Generate the system prompt (RAG fetches inside)
    sys_prompt = await build_sys_prompt(message)

    # Log the system prompt for debugging (write to a file due to length)
    with open("debug_sys_prompt.md", "w", encoding="utf-8") as f:
        f.write(sys_prompt)
        logger.debug(f"System prompt written to debug_sys_prompt.md")


    messages = [{"role": "system", "content": sys_prompt}, *cache_msgs]

    if local_llm.USE_LOCAL:
        return await ollama.chat(messages, model=local_llm.LOCAL_MODEL_ID)
    else:
        return await oai.chat(messages, model=core.MSG_MODEL_ID)


__all__ = ["handle"]
