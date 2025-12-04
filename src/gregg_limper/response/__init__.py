"""Entry-point helpers for generating replies."""

from __future__ import annotations

import logging

import discord

from gregg_limper.response.engine import PipelineContext, ResponsePipeline
from gregg_limper.response.sources.payload import build_prompt_payload
from gregg_limper.response.steps.context import ContextGatheringStep
from gregg_limper.response.steps.generation import GenerationStep
from gregg_limper.response.tracer import PipelineTracer
from gregg_limper.tools import ToolContext

logger = logging.getLogger(__name__)


async def handle(message: discord.Message) -> str:
    """Generate a reply using the prompt pipeline."""

    # Initialize Context with just the message and tool context
    # Payload will be built by ContextGatheringStep
    context = PipelineContext(
        discord_message=message,
        tool_context=ToolContext(
            guild_id=getattr(message.guild, "id", None),
            channel_id=getattr(message.channel, "id", None),
            message_id=getattr(message, "id", None),
        )
    )

    # Initialize Steps
    context_step = ContextGatheringStep()
    generation_step = GenerationStep()
    
    # Initialize Tracer
    tracer = PipelineTracer()
    
    # Build Pipeline
    pipeline = ResponsePipeline([
        context_step,
        generation_step
    ], tracer=tracer)

    # Execute
    async with message.channel.typing():
        final_text = await pipeline.run(context)
    
    # Append any artifacts (e.g. URLs) captured from tools
    if context.response_fragments:
        final_text += "\n\n" + "\n".join(context.response_fragments)

    # Debug logging (simplified for now, can be moved to a DebugStep later)
    # _write_debug_payload(payload, context.messages)
    
    return final_text


__all__ = ["handle"]
