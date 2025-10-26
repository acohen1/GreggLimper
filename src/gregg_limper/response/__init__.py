"""Entry-point helpers for generating replies."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import discord

from gregg_limper.clients import oai, ollama
from gregg_limper.config import core, local_llm

from .pipeline import build_prompt_payload

logger = logging.getLogger(__name__)


async def handle(message: discord.Message) -> str:
    """Generate a reply using the prompt pipeline."""

    payload = await build_prompt_payload(message)

    _write_debug_file("debug_history.md", payload.history.messages)
    _write_debug_file("debug_context.md", payload.context.semantic_memory)
    _write_debug_file("debug_messages.json", payload.messages, json_dump=True)

    if local_llm.USE_LOCAL:
        return await ollama.chat(payload.messages, model=local_llm.LOCAL_MODEL_ID)
    return await oai.chat(payload.messages, model=core.MSG_MODEL_ID)


def _write_debug_file(
    filename: str, data, json_dump: bool = False
) -> None:  # pragma: no cover - debug helper
    try:
        path = Path(filename)
        if json_dump:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            with path.open("w", encoding="utf-8") as handle:
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            role = item.get("role", "?")
                            content = item.get("content", "")
                            handle.write(f"{role}: {content}\n\n")
                        else:
                            handle.write(f"{item}\n")
                else:
                    handle.write(str(data))
    except Exception as exc:
        logger.debug("Failed to write %s: %s", filename, exc)


__all__ = ["handle"]
