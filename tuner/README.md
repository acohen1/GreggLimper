# Tuner Package

Offline tooling for building Gregg Limper finetuning datasets. The tuner CLI mirrors the production prompt stack, collects Discord history within a configurable whitelist/earliest-window, injects synthetic tool calls, and exports supervised samples ready for OpenAI's finetune API.

## Config Setup

1. Copy `tuner/config.sample.toml` to `tuner/config.toml`.
2. Edit the `[dataset]` block with the channel IDs, allowed user IDs, earliest timestamp, max messages per channel, sample cap, segment_concurrency (parallel LLM calls), and output paths you need. The Discord bot must have permission to read message history in every listed channel.
3. Specify the segment/tool trigger model IDs under `[models]`.
4. Keep secrets in the environment: set `DISCORD_API_TOKEN` (or adjust `discord.token_env`) before running the tuner. The TOML intentionally excludes inline secrets.

```toml
[dataset]
channels = [123, 456]
allowed_users = [10, 11]
earliest = "2024-01-01"
max_messages = 20000
max_samples = 50
segment_concurrency = 4
output_path = "data/finetune/records.jsonl"
raw_dump_dir = "data/finetune"
dry_run = false
print_stats = true
stats_path = "data/finetune/stats.json"

[models]
segment = "gpt-4o-mini"
tool_trigger = "gpt-4o-mini"

[discord]
token_env = "DISCORD_API_TOKEN"
```

## Running the CLI

```bash
python -m tuner build-dataset                # uses tuner/config.toml by default
python -m tuner build-dataset --config path/to/config.toml
python -m tuner build-dataset --max-samples 25 --dry-run
```

Flags override any TOML value (e.g., `--channels`, `--earliest`, `--segment-model`, etc.), making it easy to iterate without editing the config file. A dry run executes the entire pipeline but skips the JSONL write.

## Export Format

Each JSONL line matches OpenAI's chat finetune schema:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "content": "### Tools..."},
    ...
    {"role": "user", "content": "name said:\nhello"},
    {"role": "assistant", "content": "final reply"}
  ],
  "parallel_tool_calls": false,
  "tools": [...],
  "metadata": {
    "channel_id": 123,
    "message_ids": [...],
    "assistant_user_id": 42,
    "synthetic_tool_calls": 1,
    "target_message_id": 987
  }
}
```

## Progress & Limits

Collection logs surface per-channel heartbeats (message counts plus time-window percentage) and stage summaries (segment approvals, sample counts, synthetic tool calls). Set `dataset.max_samples` to stop once a target number of supervised examples are produced.

## Testing

The tuner suite lives under `tests/tuner`. Run `pytest tests/tuner` from an activated virtualenv to validate CLI parsing, collectors, segmenters, and synthetic tool logic.
