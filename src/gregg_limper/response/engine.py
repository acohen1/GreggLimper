"""
Core engine for the Chain-of-Thought response pipeline.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gregg_limper.response.sources.payload import PromptPayload

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Holds the state of the response generation process.
    """
    # The raw Discord message that triggered the pipeline.
    # Used by ContextGatheringStep to build the payload.
    discord_message: Any  # Typed as Any to avoid circular imports with discord.Message
    
    # The prompt payload (history, context, etc.).
    # Populated by ContextGatheringStep.
    payload: PromptPayload | None = None
    
    # Artifacts (e.g. URLs) returned by tools to be appended to the final response.
    response_fragments: list[str] = field(default_factory=list)
    
    # The current working draft of the response.
    # Steps can modify this or replace it entirely.
    response_text: str = ""
    
    # The list of messages used to generate the current response.
    # Steps can append to this (e.g. tool calls, refinement instructions).
    messages: list[dict[str, Any]] = field(default_factory=list)
    
    # Tool context (guild_id, channel_id, etc.)
    tool_context: Any = None  # Typed as Any to avoid circular imports for now, ideally ToolContext


class PipelineStep(ABC):
    """
    Abstract base class for a single step in the pipeline.
    """
    
    @abstractmethod
    async def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the step logic.
        
        Args:
            context: The current pipeline context.
            
        Returns:
            The updated pipeline context.
        """
        pass


class ResponsePipeline:
    """
    Orchestrates the execution of pipeline steps.
    """
    
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps
        
    async def run(self, context: PipelineContext) -> str:
        """
        Run all steps in order and return the final response text.
        """
        current_context = context
        
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            logger.debug("Running pipeline step %d: %s", i + 1, step_name)
            
            try:
                current_context = await step.run(current_context)
            except Exception as e:
                logger.error("Pipeline step %s failed: %s", step_name, e)
                raise
                
        return current_context.response_text
