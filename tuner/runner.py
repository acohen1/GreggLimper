from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from discord import Client, Message

from gregg_limper.clients import disc
from gregg_limper.response.context import gather_context
from gregg_limper.response.context_messages import build_context_messages

from .config import DatasetBuildConfig
from .pipeline import TrainingSample
from .pipeline.collector import collect_history, persist_raw_conversations
from .pipeline.formatter import build_prompt_shaped_sample
from .pipeline.relabel import relabel_segment
from .pipeline.segmenter import propose_segments, refine_segments_with_llm
from .pipeline.tool_synth import inject_synthetic_rag_blocks

logger = logging.getLogger(__name__)


async def build_dataset(config: DatasetBuildConfig) -> None:
    """
    High-level orchestration for dataset construction.

    Steps:
        1. Collect filtered Discord history.
        2. Generate deterministic segment candidates.
        3. Run LLM refinement to enforce clean dialogue turns.
        4. Relabel user/assistant roles and inject synthetic tool calls.
        5. Format samples like debug_messages.json and export JSONL.
    """

    logger.info("Starting dataset build for %d channels", len(config.channel_ids))
    client: Client = disc.bot

    raw_conversations = await collect_history(client=client, config=config)
    logger.info("Collected %d channel histories", len(raw_conversations))
    persist_raw_conversations(raw_conversations, destination=config.raw_dump_dir)

    candidates = propose_segments(raw_conversations)
    logger.info("Generated %d segment candidates", len(candidates))

    message_lookup = _index_messages(raw_conversations)
    refined_segments = await refine_segments_with_llm(
        candidates, message_lookup=message_lookup, config=config
    )

    samples: List[TrainingSample] = []
    for segment in refined_segments:
        history_messages = await relabel_segment(
            segment, message_lookup=message_lookup
        )
        augmented_history, synthetic_count = inject_synthetic_rag_blocks(
            history_messages
        )
        last_message = message_lookup.get(segment.message_ids[-1])
        if last_message is None:
            logger.warning(
                "Skipping segment %s due to missing terminal message",
                segment.message_ids,
            )
            continue
        participant_ids = {
            entry.get("author_id")
            for entry in augmented_history
            if isinstance(entry.get("author_id"), int)
        }
        conversation_context = await gather_context(
            last_message, participant_ids=participant_ids
        )
        context_messages = build_context_messages(conversation_context)
        sample = await build_prompt_shaped_sample(
            segment=segment,
            relabeled_history=augmented_history,
            synthetic_tool_uses=synthetic_count,
            context_messages=context_messages,
        )
        samples.append(sample)

    if config.dry_run:
        logger.info(
            "Dry run complete; prepared %d samples but skipped writing.", len(samples)
        )
        return

    _write_dataset(config.output_path, samples)


def _index_messages(conversations) -> Dict[int, Message]:
    lookup: dict[int, Message] = {}
    for convo in conversations:
        for msg in convo.messages:
            lookup[msg.id] = msg
    return lookup


def _write_dataset(path: Path, samples: List[TrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {"messages": sample.messages, "metadata": sample.metadata},
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


__all__ = ["build_dataset"]
