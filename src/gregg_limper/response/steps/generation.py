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

logger = logging.getLogger(__name__)


class GenerationStep(PipelineStep):
    """
    Generates the initial response from the model, handling tool calls if necessary.
    """
    
    async def run(self, context: PipelineContext) -> PipelineContext:
        # Initialize messages from payload if empty
        if not context.messages:
            context.messages = list(context.payload.messages)
            
        # Simple chat generation
        # We no longer handle tools here; they are executed upstream and injected as context
        result_text = (
            await ollama.chat(context.messages, model=local_llm.LOCAL_MODEL_ID)
            if local_llm.USE_LOCAL
            else await oai.chat(context.messages, model=core.MSG_MODEL_ID)
        )
        context.response_text = result_text
        context.messages.append({"role": "assistant", "content": result_text})
        return context
