"""
Pipeline step for checking response relevancy and regenerating if needed.
"""
from __future__ import annotations

import logging

from gregg_limper.clients import oai, ollama
from gregg_limper.config import core, local_llm
from gregg_limper.response.engine import PipelineContext, PipelineStep

logger = logging.getLogger(__name__)


class RelevancyStep(PipelineStep):
    """
    Checks if the response is relevant/sensical.
    If not, regenerates with increasing temperature to break loops.
    """

    async def run(self, context: PipelineContext) -> PipelineContext:
        # If no check model is configured, skip
        if not core.RELEVANCY_CHECK_MODEL_ID:
            return context

        loops = 0
        max_loops = core.RELEVANCY_CHECK_MAX_LOOPS
        
        # We need the history messages for the classifier context
        history_messages = context.payload.history.messages
        
        # Capture the base messages (User + Tool Results) before we start looping
        # We assume the LAST message in context.messages is the assistant's response from GenerationStep.
        # We slice it off to get the "clean" history.
        base_messages = list(context.messages[:-1])
        
        rejected_attempts = []
        
        while loops < max_loops:
            # Check if the current response is relevant
            is_relevant = await check_relevancy(
                context.response_text, 
                history_messages, 
                model=core.RELEVANCY_CHECK_MODEL_ID
            )
            
            if is_relevant:
                break
                
            # Record the rejection
            rejected_attempts.append({
                "loop": loops + 1,
                "response": context.response_text,
                "temperature": 0.7 if loops == 0 else min(0.7 + (loops * 0.2), 1.5)
            })
                
            loops += 1
            logger.info("Response irrelevant, starting relevancy loop %d/%d", loops, max_loops)
            
            # Calculate temperature scaling
            # Base temp (usually 0.7 or model default) + (loops * 0.2)
            # Cap at 1.5 to prevent total chaos
            base_temp = 0.7
            current_temp = min(base_temp + (loops * 0.2), 1.5)
            logger.info("Regenerating with temperature: %.2f", current_temp)
            
            # Prepare messages for regeneration using Sliding Window
            # We want: Base Context + [Latest Failed Attempt] + [Critique]
            # This ensures the model sees what it did wrong, but doesn't get confused by 
            # a long history of previous failed attempts.
            
            relevancy_messages = list(base_messages)
            relevancy_messages.append({"role": "assistant", "content": context.response_text})
            relevancy_messages.append({"role": "user", "content": get_relevancy_instruction()})
            
            # Update context messages for the API call (and tracing)
            context.messages = relevancy_messages
            
            # Regenerate response
            # We call chat directly to pass the temperature
            result_text = (
                await ollama.chat(context.messages, model=local_llm.LOCAL_MODEL_ID) # Local LLM might not support temp yet, but we focus on OAI
                if local_llm.USE_LOCAL
                else await oai.chat(
                    context.messages, 
                    model=core.MSG_MODEL_ID,
                    temperature=current_temp
                )
            )
            
            context.response_text = result_text
            
            if not context.response_text:
                logger.info("Relevancy loop yielded empty response, stopping.")
                break
        
        
        # Final Cleanup:
        # Ensure context.messages reflects the final state: Base Context + Final Response
        # We discard the intermediate critiques and failed attempts from the message history.
        context.messages = list(base_messages)
        if context.response_text:
            context.messages.append({"role": "assistant", "content": context.response_text})
        
        # Record stats for the tracer
        context.step_metadata = {
            "relevancy_loops": loops,
            "final_temperature": 0.7 if loops == 0 else min(0.7 + (loops * 0.2), 1.5),
            "rejected_attempts": rejected_attempts
        }
                
        return context


async def check_relevancy(response_text: str, history_messages: list[dict], model: str) -> bool:
    """
    Checks if the response is relevant and makes sense in context.
    """
    prompt = (
        "You are a response quality analyzer. Your job is to determine if the ASSISTANT's response "
        "makes sense and is relevant to the CONVERSATION HISTORY.\n\n"
        "CONVERSATION HISTORY:\n"
    )
    
    # Add last few messages for context
    recent_history = history_messages[-5:]
    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        prompt += f"{role.upper()}: {content}\n"
        
    prompt += f"\nASSISTANT CURRENT RESPONSE:\n{response_text}\n\n"
    prompt += (
        "INSTRUCTIONS:\n"
        "- If the response is relevant to the last user message, output 'PASS'.\n"
        "- ALLOW sarcasm, dismissiveness, jokes, and personality-driven responses (e.g. 'idk', 'whatever').\n"
        "- ONLY output 'FAIL' if the response is a complete hallucination or completely unrelated to the conversation.\n"
        "- Output ONLY the word 'PASS' or 'FAIL'."
    )

    try:
        result = await oai.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
        decision = result.strip().upper()
        logger.info("Relevancy check decision: %s", decision)
        return "PASS" in decision
    except Exception as e:
        logger.warning("Relevancy check failed, defaulting to pass: %s", e)
        return True


def get_relevancy_instruction() -> str:
    """
    Returns the instruction prompt for regenerating a non-relevant response.
    """
    return (
        "Your previous response (above) was completely irrelevant or nonsensical. "
        "Try again. Focus on answering the user's last message directly. "
        "Make sense this time."
    )
