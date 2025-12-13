"""
Pipeline step for gathering conversation context and building the prompt payload.
"""
from __future__ import annotations

import logging

from gregg_limper.response.engine import PipelineContext, PipelineStep
from gregg_limper.response.sources.payload import build_prompt_payload

logger = logging.getLogger(__name__)


class ContextGatheringStep(PipelineStep):
    """
    Fetches history, user profiles, and other context to build the PromptPayload.
    """
    
    async def run(self, context: PipelineContext) -> PipelineContext:
        logger.debug("Gathering context for message %s", context.discord_message.id)
        
        # Build the payload using the existing helper
        payload = await build_prompt_payload(context.discord_message)
        
        # Update the context
        context.payload = payload
        
        # Initialize the working messages list from the payload
        # This includes system prompt, tool specs, context messages, and history
        context.messages = list(payload.messages)
        
        return context
