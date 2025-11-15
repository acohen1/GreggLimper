from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from discord import Message

from gregg_limper.response.context import gather_context
from gregg_limper.response.context_messages import build_context_messages

from .config import DatasetBuildConfig
from .discord_client import connect_tuner_client
from .pipeline import TrainingSample
from .pipeline.moderation import moderate_messages
from .pipeline.types import SegmentedConversation
from .util.alias import AliasGenerator, scrub_text
from .pipeline.collector import collect_history, persist_raw_conversations
from .pipeline.formatter import build_prompt_shaped_sample
from .pipeline.relabel import relabel_segment
from .pipeline.segmenter import (
    drop_ineligible_candidates,
    propose_segments,
    refine_segments_with_llm,
)
from .pipeline.tool_synth import (
    build_llm_tool_trigger_decider,
    inject_synthetic_rag_blocks,
)

logger = logging.getLogger(__name__)

TOTAL_PROGRESS_STEPS = 5


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
    if not config.discord_token:
        raise ValueError("Discord API token is required to hydrate history.")
    stats = {
        "channels_requested": len(config.channel_ids),
        "channels_collected": 0,
        "messages_collected": 0,
        "segment_candidates": 0,
        "segments_approved": 0,
        "segments_rejected": 0,
        "synthetic_tool_calls": 0,
        "samples_prepared": 0,
        "sample_cap": config.max_samples,
        "dry_run": config.dry_run,
        "output_path": str(config.output_path),
        "raw_dump_dir": str(config.raw_dump_dir),
    }

    if config.max_samples:
        logger.info("Sample cap enabled: %d max supervised records", config.max_samples)

    async with connect_tuner_client(config.discord_token) as client:
        raw_conversations = await collect_history(
            client=client,
            config=config,
            reuse_raw=config.reuse_raw,
        )

    alias_generator = AliasGenerator(config.pii_salt or "") if config.scrub_pii else None
    alias_map = _scrub_conversations(raw_conversations, alias_generator)

    stats["channels_collected"] = len(raw_conversations)
    stats["messages_collected"] = sum(
        len(convo.messages) for convo in raw_conversations
    )

    if config.reuse_raw:
        _log_stage(
            1,
            f"Reused {stats['messages_collected']} cached messages across {len(raw_conversations)} channels.",
        )
    else:
        persist_raw_conversations(raw_conversations, destination=config.raw_dump_dir)
        _log_stage(
            1,
            f"Hydrated {stats['messages_collected']} messages across {len(raw_conversations)} channels.",
        )

    samples = await _process_conversations(
        config,
        stats,
        raw_conversations,
        alias_map=alias_map,
    )

    if config.dry_run:
        _log_stage(5, "Dry run complete; skipped dataset write.")
        _emit_stats(stats, config=config)
        return

    _write_dataset(config.output_path, samples)
    _log_stage(5, f"Wrote {len(samples)} samples to {config.output_path}")
    _emit_stats(stats, config=config)


async def _process_conversations(
    config: DatasetBuildConfig,
    stats: dict,
    raw_conversations,
    *,
    alias_map: dict[int, str],
) -> List[TrainingSample]:
    candidates = propose_segments(raw_conversations)
    raw_candidate_count = len(candidates)

    _log_stage(
        2,
        f"Chunked conversations into {raw_candidate_count} segment candidates.",
    )

    message_lookup = _index_messages(raw_conversations)
    candidates, pruned = drop_ineligible_candidates(
        candidates,
        message_lookup=message_lookup,
        config=config,
    )
    if pruned:
        logger.info(
            "Segmenter: pruned %d candidates with no eligible assistants (%d remain).",
            pruned,
            len(candidates),
        )

    stats["segment_candidates_raw"] = raw_candidate_count
    stats["segment_candidates"] = len(candidates)

    refined_segments: List[SegmentedConversation] | None = None
    if config.reuse_segments:
        refined_segments = _load_segments(config.segment_dump_dir)
        if refined_segments:
            stats["segments_approved"] = len(refined_segments)
            stats["segments_rejected"] = max(
                stats["segment_candidates"] - stats["segments_approved"], 0
            )
            _log_stage(
                3,
                f"Reused {len(refined_segments)} cached segments.",
            )

    if refined_segments is None:
        refined_segments = await refine_segments_with_llm(
            candidates, message_lookup=message_lookup, config=config
        )
        stats["segments_approved"] = len(refined_segments)
        stats["segments_rejected"] = max(
            stats["segment_candidates"] - stats["segments_approved"], 0
        )
        _persist_segments(config.segment_dump_dir, refined_segments)
    approval_rate = (
        (stats["segments_approved"] / stats["segment_candidates"]) * 100
        if stats["segment_candidates"]
        else 0
    )
    _log_stage(
        3,
        "LLM approved %d/%d segments (%.1f%%)."
        % (
            stats["segments_approved"],
            stats["segment_candidates"],
            approval_rate,
        ),
    )

    tool_decider = (
        build_llm_tool_trigger_decider(config.tool_trigger_model)
        if config.tool_trigger_model
        else None
    )

    samples: List[TrainingSample] = []
    total_segments = len(refined_segments)
    if total_segments == 0:
        _log_stage(4, "No eligible segments after refinement.")
        stats["samples_prepared"] = 0
        return samples

    moderation_model = config.moderation_model
    log_interval = max(1, total_segments // 10)
    for processed, segment in enumerate(refined_segments, start=1):
        history_messages = await relabel_segment(
            segment, message_lookup=message_lookup
        )
        augmented_history, synthetic_count = await inject_synthetic_rag_blocks(
            history_messages, decider=tool_decider
        )
        stats["synthetic_tool_calls"] += synthetic_count
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
        if sample is None:
            continue

        if moderation_model:
            keep = await moderate_messages(sample.messages, model=moderation_model)
            if not keep:
                logger.info(
                    "Moderation flagged segment %s; dropping sample.",
                    segment.message_ids,
                )
                stats.setdefault("samples_blocked_moderation", 0)
                stats["samples_blocked_moderation"] += 1
                continue
        if alias_map:
            sample.metadata["assistant_alias"] = alias_map.get(segment.assigned_assistant_id)

        samples.append(sample)

        if processed % log_interval == 0 or processed == total_segments:
            _log_sample_progress(processed, total_segments, len(samples), config.max_samples)

        if config.max_samples is not None and len(samples) >= config.max_samples:
            logger.info(
                "Reached max sample cap (%d); stopping collection.",
                config.max_samples,
            )
            break

    stats["samples_prepared"] = len(samples)
    cap_note = f" (cap {config.max_samples})" if config.max_samples else ""
    _log_stage(
        4,
        f"Built {stats['samples_prepared']} supervised samples{cap_note} with {stats['synthetic_tool_calls']} synthetic tool calls.",
    )

    return samples


def _index_messages(conversations) -> Dict[int, Message]:
    lookup: dict[int, Message] = {}
    for convo in conversations:
        for msg in convo.messages:
            lookup[msg.id] = msg
    return lookup


def _write_dataset(path: Path, samples: List[TrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_name("records.metadata.jsonl")

    with path.open("w", encoding="utf-8") as handle, metadata_path.open(
        "w", encoding="utf-8"
    ) as meta_handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "messages": sample.messages,
                        "parallel_tool_calls": sample.parallel_tool_calls,
                        "tools": sample.tools,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

            meta_handle.write(
                json.dumps(sample.metadata, ensure_ascii=False)
            )
            meta_handle.write("\n")

    logger.info("Wrote %d samples to %s (metadata at %s)", len(samples), path, metadata_path)


def _scrub_conversations(conversations, alias_generator: AliasGenerator | None) -> dict[int, str]:
    alias_map: dict[int, str] = {}
    if not alias_generator:
        return alias_map

    for convo in conversations:
        for message in convo.messages:
            author = getattr(message, "author", None)
            user_id = getattr(author, "id", None)
            alias = alias_generator.alias(user_id)
            if user_id is not None:
                alias_map[user_id] = alias

            _set_attr_if_possible(author, "display_name", alias)
            _set_attr_if_possible(author, "name", alias)

            for attr in ("clean_content", "content"):
                value = getattr(message, attr, None)
                if isinstance(value, str) and value:
                    _set_attr_if_possible(
                        message,
                        attr,
                        scrub_text(value, alias_generator.alias),
                    )

            mentions = getattr(message, "mentions", None)
            if mentions:
                for mention in mentions:
                    mention_id = getattr(mention, "id", None)
                    mention_alias = alias_generator.alias(mention_id)
                    if mention_id is not None:
                        alias_map.setdefault(mention_id, mention_alias)
                    _set_attr_if_possible(mention, "display_name", mention_alias)
                    _set_attr_if_possible(mention, "name", mention_alias)

    return alias_map


def _set_attr_if_possible(obj, attr: str, value: str) -> None:
    if obj is None:
        return
    try:
        setattr(obj, attr, value)
    except (AttributeError, TypeError):
        pass


def _emit_stats(stats: dict, *, config: DatasetBuildConfig) -> None:
    if not config.print_stats and not config.stats_path:
        return

    payload = json.dumps(stats, indent=2, ensure_ascii=False)
    if config.print_stats:
        logger.info("Dataset build statistics:\n%s", payload)
    if config.stats_path:
        config.stats_path.parent.mkdir(parents=True, exist_ok=True)
        config.stats_path.write_text(payload, encoding="utf-8")
        logger.info("Wrote stats to %s", config.stats_path)


def _log_stage(stage_idx: int, message: str) -> None:
    logger.info(
        "[Progress %d/%d] %s",
        stage_idx,
        TOTAL_PROGRESS_STEPS,
        message,
    )


def _log_sample_progress(processed: int, total: int, built: int, cap: int | None) -> None:
    percent = (processed / total) * 100 if total else 100
    logger.info(
        "Building samples: %d/%d segments processed (%.1f%%), %d samples ready%s.",
        processed,
        total,
        percent,
        built,
        f" / cap {cap}" if cap is not None else "",
    )


def _persist_segments(directory: Path, segments: List[SegmentedConversation]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "segments.json"
    payload = [
        {
            "channel_id": seg.channel_id,
            "message_ids": seg.message_ids,
            "assistant_user_id": seg.assigned_assistant_id,
        }
        for seg in segments
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cached %d refined segments at %s", len(segments), path)


def _load_segments(directory: Path) -> List[SegmentedConversation] | None:
    path = directory / "segments.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            SegmentedConversation(
                channel_id=item["channel_id"],
                message_ids=item["message_ids"],
                assigned_assistant_id=item["assistant_user_id"],
            )
            for item in payload
        ]
        logger.info("Loaded %d cached segments from %s", len(segments), path)
        return segments
    except Exception:
        logger.warning("Failed to load cached segments from %s", path, exc_info=True)
        return None


__all__ = ["build_dataset"]
