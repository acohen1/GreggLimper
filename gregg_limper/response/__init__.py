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
from .prompt import build_sys_prompt
from .cache_adapter import build_history

async def dispatch(message: discord.Message) -> str:
    """Send to the selected provider."""
    sys_prompt = await build_sys_prompt(message)
    cache_msgs = await build_history(message.channel.id, core.CONTEXT_LENGTH)
    messages = [{"role": "system", "content": sys_prompt}, *cache_msgs]

    if local_llm.USE_LOCAL:
        return await ollama.chat(messages, model=local_llm.LOCAL_MODEL_ID)
    else:
        return await oai.chat(messages, model=core.MSG_MODEL_ID)
