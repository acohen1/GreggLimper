"""
Pipeline step for generating an internal reasoning trace.
"""
from __future__ import annotations

import logging

from gregg_limper.clients import oai
from gregg_limper.config import core
from gregg_limper.response.engine import PipelineContext, PipelineStep

logger = logging.getLogger(__name__)


class ReasoningStep(PipelineStep):
    """
    Analyzes the context and tool results to produce a "Reasoning Trace" (internal plan).
    Injects this trace as a System Message to guide the final generation.
    """

    async def run(self, context: PipelineContext) -> PipelineContext:
        # If no reasoning model is configured, skip
        if not core.REASONING_MODEL_ID:
            logger.debug("No REASONING_MODEL_ID configured, skipping ReasoningStep.")
            return context

        logger.info("Generating reasoning trace...")

        # Construct the reasoning prompt
        # We want the model to see the full context (including tool results)
        # and produce a hidden thought process.
        
        # We'll use a specific system prompt for the reasoning model
        reasoning_system_prompt = (
            "You are the internal monologue of an advanced AI assistant. "
            "Your goal is to analyze the user's request, the conversation history, and any tool outputs. "
            "Decide on the best course of action for the final response. "
            "Do NOT generate the final response yourself. "
            "Instead, output a concise 'Reasoning Trace' that explains your plan. "
            "Focus on: User Intent, Tool Success/Failure, Tone, and Key Information to include."
        )

        # Prepare messages for the reasoning model
        # We use the current context messages but swap the system prompt
        reasoning_messages = [
            {"role": "system", "content": reasoning_system_prompt}
        ]
        
        # Append the rest of the conversation (User, Assistant, Tool outputs)
        # We skip the original system prompt if it exists in context.messages[0]
        start_index = 0
        if context.messages and context.messages[0]["role"] == "system":
            start_index = 1
            
        reasoning_messages.extend(context.messages[start_index:])
        
        # Call the reasoning model
        try:
            trace = await oai.chat(
                reasoning_messages,
                model=core.REASONING_MODEL_ID,
                temperature=0.7 # Slightly higher temp for creative planning
            )
        except Exception as e:
            logger.error("Reasoning generation failed: %s", e)
            # If reasoning fails, we just proceed without it
            return context

        logger.info("Reasoning Trace generated: %s", trace)

        # Inject the trace into the context for the NEXT step (GenerationStep)
        # We add it as a System Message to guide the persona model
        
        # We'll append it to the END of the messages list so it's fresh in context,
        # but mark it as a System instruction.
        # Alternatively, we could prepend it, but appending usually has stronger adherence.
        
        trace_message = {
            "role": "system",
            "content": f"### Internal Reasoning Plan\n{trace}\n\nFollow this plan to generate the response."
        }
        
        context.messages.append(trace_message)
        
        return context
