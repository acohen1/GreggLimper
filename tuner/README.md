# Tuner Package

Offline tooling for building Gregg Limper finetuning datasets. The tuner CLI collects Discord history within a configurable whitelist/earliest-window, segments conversations, and exports supervised samples ready for OpenAI's finetune API using only user/assistant turns (no system/context/tool scaffolding).

## Config Setup

1. Copy `config/config.sample.toml` to `config.toml` at the repo root (shared with the bot). The tuner reads the `[finetune.*]` sections by default.
2. In `[finetune.dataset]`, set `channels`, `allowed_users`, `earliest`, `max_messages`, `max_samples`, `segment_concurrency`, cache/output paths, and emoji whitelist as needed.
3. In `[finetune.models]`, set the segment/moderation/relevance model IDs.
4. Keep secrets in the environment: set `DISCORD_API_TOKEN` (or your chosen `token_env`) and `OPENAI_API_KEY`. The TOML intentionally excludes inline secrets.

```toml
[finetune.dataset]
channels = [123, 456]
allowed_users = [10, 11]
earliest = "2024-01-01"
max_messages = 20000
max_samples = 50
segment_concurrency = 4
segment_dir = "data/finetune/segments"
output_path = "data/finetune/records.jsonl"
raw_dump_dir = "data/finetune/raw"
stats_path = "data/finetune/stats.json"
assistant_custom_emojis = [":SoloPog:"]
reuse_raw = true
reuse_segments = true
scrub_pii = false
dry_run = false
print_stats = true

[finetune.models]
segment = "gpt-4o-mini"
moderation = "omni-moderation-2024-09-26"

[finetune.discord]
token_env = "DISCORD_API_TOKEN"
```

## Running the CLI

```bash
python -m tuner build-dataset                # uses config.toml by default
python -m tuner build-dataset --config path/to/config.toml
python -m tuner build-dataset --max-samples 25 --dry-run
```

Flags override any TOML value (e.g., `--channels`, `--earliest`, `--segment-model`, etc.), making it easy to iterate without editing the config file. A dry run executes the entire pipeline but skips the JSONL write.

### Caching & Assistant Guards

- `raw_dump_dir` / `--raw-dump-dir`: where raw Discord transcripts (one JSONL per channel) land. Pair with `--reuse-raw` to skip Discord requests when the cache already exists.
- `segment_dir` / `--segment-dir`: persistence for refined segments (`segments.json`). Combine with `--reuse-segments` to bypass the LLM refinement step when iterating on downstream stages.
- `assistant_custom_emojis` / `--assistant-emojis`: whitelist of custom emojis that do **not** disqualify an assistant candidate. The segmenter already rejects speakers who send consecutive replies, emit non-text content (links, attachments, embeds), or use any other custom emoji.
- Need the whitelist fast? Run `python scripts/list_guild_emojis.py` (uses `config.toml` plus `DISCORD_API_TOKEN` by default) or add `--guilds <id ...>` to target specific servers. The script prints each guild’s inventory and automatically rewrites `dataset.assistant_custom_emojis` in your tuner config with every discovered custom emoji.
- `models.moderation` / `--moderation-model`: optional OpenAI moderation model (e.g., `omni-moderation-2024-09-26`). When provided, each formatted sample is screened before it’s added to `records.jsonl`; flagged samples are dropped and counted in the final stats.
- `dataset.scrub_pii` / `--scrub-pii`: enable deterministic aliasing to anonymize user identifiers in the exported dataset. When on, every Discord author/mention is replaced with a friendly pseudonym, and `records.metadata.jsonl` records the alias used for the assistant persona.

By default the CLI writes these artifacts under `data/finetune/`:

```
raw/                     # cached channel history
segments/                # refined segments ready for relabeling
records.jsonl            # final supervised dataset (OpenAI schema)
records.metadata.jsonl   # parallel metadata (channel ids, assistant ids, assistant_alias, etc.)
```

`records.jsonl` is the only required output; `records.metadata.jsonl` is emitted alongside for traceability (ids, aliases, etc.).

## Export Format

Each JSONL line matches OpenAI's chat finetune schema:

```json
{
  "messages": [
    {"role": "user", "content": "name said:\nhello"},
    {"role": "assistant", "content": "final reply"}
  ],
  "metadata": {
    "channel_id": 123,
    "message_ids": [...],
    "assistant_user_id": 42,
    "target_message_id": 987
  }
}
```

## Progress & Limits

Collection logs surface per-channel heartbeats (message counts plus time-window percentage) and stage summaries (segment approvals and sample counts). Set `dataset.max_samples` to stop once a target number of supervised examples are produced.

## Testing

The tuner suite lives under `tests/tuner`. Run `pytest tests/tuner` from an activated virtualenv to validate CLI parsing, collectors, and segmenters.
