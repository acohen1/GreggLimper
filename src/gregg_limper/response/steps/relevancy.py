"""
Pipeline step for checking response relevancy and regenerating if needed.
"""
from __future__ import annotations

import logging
import json
from typing import Any, Dict

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
        
        # Capture the base messages (User + Tool Results) before we start looping
        # We assume the LAST message in context.messages is the assistant's response from GenerationStep.
        # We slice it off to get the "clean" history.
        base_messages = list(context.messages[:-1])

        last_user_message = _get_last_user_message(base_messages)
        tool_notes = _get_tool_notes(base_messages)
        
        rejected_attempts = []
        last_critique: dict[str, Any] | None = None
        
        while loops < max_loops:
            # Check if the current response is relevant
            judge = await check_relevancy(
                response_text=context.response_text,
                user_message=last_user_message,
                tool_notes=tool_notes,
                model=core.RELEVANCY_CHECK_MODEL_ID,
            )
            decision = (judge.get("decision") or "FAIL").upper()
            last_critique = judge
            
            if decision == "PASS":
                break
                
            # Record the rejection
            critique_missing = judge.get("missing") or []
            critique_issues = judge.get("issues") or []
            current_loop = loops + 1  # 1-based loop counter
            current_temp = _compute_temperature(current_loop, max_loops)
            rejected_attempts.append({
                "loop": current_loop,
                "response": context.response_text,
                "temperature": current_temp,
                "decision": decision,
                "missing": critique_missing,
                "issues": critique_issues,
            })
                
            loops += 1
            logger.info("Response irrelevant, starting relevancy loop %d/%d", loops, max_loops)
            
            # Calculate temperature scaling
            # Base temp (0.7) up to a cap (1.0) spread across max_loops
            current_temp = _compute_temperature(
                loops,
                max_loops,
                base=core.RELEVANCY_REGEN_TEMP_MIN,
                cap=core.RELEVANCY_REGEN_TEMP_MAX,
            )
            logger.info("Regenerating with temperature: %.2f", current_temp)
            
            # Prepare messages for regeneration using Sliding Window
            # We want: Base Context + [Latest Failed Attempt] + [Critique]
            # This ensures the model sees what it did wrong, but doesn't get confused by
            # a long history of previous failed attempts.
            
            relevancy_messages = list(base_messages)
            relevancy_messages.append({"role": "assistant", "content": context.response_text})
            relevancy_messages.append({"role": "user", "content": build_recovery_instruction(critique_missing, critique_issues)})
            
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
            "final_temperature": _compute_temperature(
                loops or 0,
                max_loops,
                base=core.RELEVANCY_REGEN_TEMP_MIN,
                cap=core.RELEVANCY_REGEN_TEMP_MAX,
            ),
            "rejected_attempts": rejected_attempts,
            "last_critique": last_critique,
        }
                
        return context


async def check_relevancy(
    *,
    response_text: str,
    user_message: dict[str, str] | None,
    tool_notes: list[str],
    model: str,
) -> Dict[str, Any]:
    """
    Checks if the response is relevant and makes sense in context.
    """
    user_text = user_message.get("content") if user_message else ""
    tools_text = "\n".join(tool_notes) if tool_notes else "None"

    prompt = (
        "You are a strict relevance judge. Compare USER_MESSAGE to ASSISTANT_RESPONSE.\n"
        "- PASS if the response is on-topic and attempts to answer the user, even if it is brief, approximate, uncertain, or only partially complete, as long as it is coherent (not gibberish).\n"
        "- FAIL only if the response is unrelated to the user request, clearly omits the core answer, or is nonsense/word salad.\n"
        "- Ignore tone/persona/politeness entirely; do not ask for apologies or style changes.\n"
        "- Tool notes are optional context; only expect them if the user explicitly asked for that info.\n"
        "- Report only objective content gaps or irrelevance; do not penalize uncertainty if it still addresses the question.\n"
        "Return JSON only: {\"decision\":\"PASS|FAIL\",\"missing\":[...],\"issues\":[...]}"
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_message": user_text,
                    "tool_notes": tools_text,
                    "assistant_response": response_text,
                },
                ensure_ascii=False,
            ),
        },
    ]

    try:
        raw = await oai.chat(messages=messages, model=model)
        decision = raw.strip()
        parsed = json.loads(decision)
        parsed.setdefault("decision", "FAIL")
        parsed.setdefault("missing", [])
        parsed.setdefault("issues", [])
        logger.info("Relevancy check decision=%s missing=%s", parsed.get("decision"), parsed.get("missing"))
        return parsed
    except Exception as e:
        logger.warning("Relevancy check failed, defaulting to PASS: %s", e)
        return {"decision": "PASS", "missing": [], "issues": [str(e)]}


def build_recovery_instruction(missing: list[str], issues: list[str]) -> str:
    """
    Build a targeted regeneration instruction using the critique.
    """
    parts = ["Your previous response was incomplete or off-target. Fix it."]
    if missing:
        parts.append("Include: " + "; ".join(missing))
    if issues:
        parts.append("Issues: " + "; ".join(issues))
    parts.append("Answer the user clearly and fully while keeping the same tone/persona.")
    return " ".join(parts)


def _get_last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last user message from the list, if any."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


def _get_tool_notes(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool/system notes to give the judge more context."""
    notes = []
    for msg in messages:
        if msg.get("role") == "system" and "Tool '" in msg.get("content", ""):
            notes.append(msg.get("content", ""))
    return notes


def _compute_temperature(loop_index: int, max_loops: int, base: float = 0.7, cap: float = 1.0) -> float:
    """
    Calculate temperature for a given loop (1-based), capped at ``cap`` and starting at ``base``.
    """
    if loop_index <= 0:
        return base
    if max_loops <= 0:
        return cap
    fraction = min(loop_index / max_loops, 1.0)
    return min(cap, base + (cap - base) * fraction)
