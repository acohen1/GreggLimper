"""
Pipeline step for iterative response refinement.
"""
from __future__ import annotations

import logging

from gregg_limper.clients import oai
from gregg_limper.config import core
from gregg_limper.response.engine import PipelineContext, PipelineStep
from gregg_limper.response.steps.generation import GenerationStep

logger = logging.getLogger(__name__)


class RefinementStep(PipelineStep):
    """
    Checks if the response is complete and triggers a rewrite if not.
    """
    
    def __init__(self, generation_step: GenerationStep):
        # We need the generation step (or a runner) to execute the rewrite.
        # Since GenerationStep encapsulates the logic to run the model (with tools),
        # we can reuse it.
        self.generation_step = generation_step

    async def run(self, context: PipelineContext) -> PipelineContext:
        # If no check model is configured, skip
        if not core.DETAIL_CHECK_MODEL_ID:
            return context

        loops = 0
        max_loops = core.DETAIL_CHECK_MAX_LOOPS
        
        # We need the history messages for the classifier context
        history_messages = context.payload.history.messages
        
        while loops < max_loops:
            is_complete = await oai.check_completeness(context.response_text, history_messages)
            if is_complete:
                break
                
            loops += 1
            logger.info("Response incomplete, starting refinement loop %d/%d", loops, max_loops)
            
            # Prepare for the next pass
            # We start fresh with the original payload messages each time
            # But we need to construct a specific prompt for the rewrite
            
            # NOTE: We are modifying the context.messages for the *next* run.
            # We start with the original payload messages
            refinement_messages = list(context.payload.messages)
            
            # Append the current (incomplete) response
            refinement_messages.append({"role": "assistant", "content": context.response_text})
            
            # Add the refinement instruction
            refinement_messages.append({"role": "user", "content": oai.get_refinement_instruction()})
            
            # Update context messages so GenerationStep uses them
            context.messages = refinement_messages
            
            # Run the generation step again to get the rewrite
            # GenerationStep will update context.response_text
            context = await self.generation_step.run(context)
            
            if not context.response_text:
                logger.info("Refinement yielded empty response, stopping.")
                break
                
        return context
