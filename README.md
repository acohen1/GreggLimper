# Gregg Limper

Discord assistant for knowledge retrieval and response generation. Gregg Limper captures multi-modal Discord conversations, enriches them with retrieval-augmented memory, and produces context-aware replies using OpenAI or a local LLM.

## Table of Contents
- [Overview](#overview)
- [Feature Highlights](#feature-highlights)
- [Architecture](#architecture)
  - [Event Flow](#event-flow)
  - [Message Formatting Pipeline](#message-formatting-pipeline)
  - [Retrieval-Augmented Memory](#retrieval-augmented-memory)
  - [Prompt Orchestration](#prompt-orchestration)
  - [Background Maintenance](#background-maintenance)
- [Directory Layout](#directory-layout)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Run the Bot](#run-the-bot)
  - [Run the Test Suite](#run-the-test-suite)
- [Slash Commands](#slash-commands)
- [Configuration Reference](#configuration-reference)
- [Handler Registries](#handler-registries)
- [Development Notes](#development-notes)
- [Troubleshooting & Debugging](#troubleshooting--debugging)
- [Further Reading](#further-reading)
- [License](#license)

## Overview

Gregg Limper is a Discord assistant that replies like a long-time regular who never forgets an inside joke. Incoming guild traffic is cached, normalized, and selectively ingested into long-term storage so the bot can answer with recent context, previously shared links, and media summaries. Only opted-in messages that pick up pre-approved reaction emoji are promoted into the RAG stores, keeping the corpus focused on what the community highlights. Consent-aware retrieval, configurable storage backends (SQLite plus optional Milvus), and a modular formatter make the bot a natural fit for Discord servers that want reliable recall without retaining unwanted data—while keeping the experience casual and friendly.

## Feature Highlights

- Full Discord slash command integration with automatic cog discovery (`src/gregg_limper/commands`).
- Channel-aware cache that memoizes formatted message fragments and guards ingestion with opt-in consent plus configurable reaction triggers (`memory/cache`).
- Rich media understanding: text normalization, image/GIF captioning, URL summarization, and YouTube metadata/vision blending (`formatter`).
- Retrieval workflow that blends cached history, channel summaries, user profiles, and tool-called semantic search when the model explicitly requests it (`response` + `tools`).
- Operators can highlight high-signal moments by reacting with approved emoji, promoting only those opted-in messages into long-term RAG storage (`event_hooks/reaction_hook.py`).
- Dual model support: OpenAI APIs by default with optional local Ollama fallback (`config/local_llm.py`, `clients/ollama.py`).
- Background maintenance that refreshes stale embeddings, keeps SQLite lean, and reconciles Milvus vector indexes (`memory/rag/scheduler.py`).
- Debug-first ergonomics: cached fragments persist to disk, and every completion request records context and message payloads for inspection (`debug_history.md`, `debug_context.md`, `debug_messages.json`).

## Architecture

### Event Flow

1. `gregg_limper.clients.disc.GLBot` wires Discord events to the bot.
2. `event_hooks.ready_hook.handle` hydrates the cache, validates Milvus (when enabled), and schedules maintenance.
3. `event_hooks.message_hook.handle` filters messages by configured channel IDs, updates the cache, logs memo diagnostics, and triggers the response pipeline when the bot is mentioned.
4. `event_hooks.reaction_hook.handle` ingests opted-in messages when they receive configured trigger reactions.

### Message Formatting Pipeline

The cache relies on the formatter to transform raw Discord messages into stable fragments:

- `formatter.classifier` slices a message into text, images, GIFs, generic links, and YouTube URLs.
- Specialized handlers (`formatter/handlers/`) run concurrently to enrich slices:
  - `ImageHandler` downloads attachments and runs OpenAI vision to describe them.
  - `GIFHandler` scrapes Tenor/Giphy metadata, extracts a representative frame, and captions it with vision.
  - `LinkHandler` invokes the OpenAI web-search model for high-signal summaries.
  - `YouTubeHandler` queries the YouTube Data API and captions thumbnails.
  - `TextHandler` normalizes mentions and drops pure bot pings.
- `formatter.composer` merges fragment lists, assigns deterministic media IDs (`memory/rag/media_id.py`), and returns memo payloads ready for caching, retrieval, or prompt assembly.

### Retrieval-Augmented Memory

- `memory/cache.manager.GLCache` is a singleton that keeps per-channel buffers (`ChannelCacheState`) aligned with memo snapshots on disk (`memory/cache/memo_store.py`).
- `memory/cache/ingestion.py` checks user consent (`memory/rag/consent.py`) and only pushes memoized messages into long-term storage when users opt in.
- `event_hooks.reaction_hook.handle` gates ingestion on configurable emoji reactions so only highlighted, opted-in messages are promoted into the RAG stores.
- Long-term recall lives in `memory/rag`:
  - SQLite stores fragments, metadata, channel summaries, and user profiles (`memory/rag/sql`).
  - Vector embeddings are refreshed and stored via OpenAI (`memory/rag/embeddings.py`), with Milvus providing fast similarity search when `ENABLE_MILVUS` is true (`memory/rag/vector`).
  - `memory/rag/scheduler.start` runs periodic SQL and vector maintenance (embedding refresh, WAL compaction, Milvus sync/compact).
  - Opt-in backfill (`commands/handlers/rag_opt.py`) hydrates historical messages cooperatively using bounded concurrency and rate limits from `config/rag.py`.

### Prompt Orchestration

`response.pipeline.build_prompt_payload` assembles the final chat request:

1. `response.history.build_history` serializes cached messages into user/assistant turns up to `core.CONTEXT_LENGTH`.
2. `response.context.gather_context` pulls channel summaries and opt-in user profiles; deeper history is fetched on demand via tools.
3. Messages are merged with the system prompt from `response.system_prompt.get_system_prompt`, yielding a fully grounded `messages` list.
4. The bot calls OpenAI (`clients/oai.chat_full`) or a configured local Ollama model (`clients/ollama.chat`). When the model issues tool calls, the loop executes the requested tools, appends their outputs, and resubmits the conversation until a final reply is produced.

### Tool Calling

- Tool metadata lives in `src/gregg_limper/tools/__init__.py`; individual handlers reside in `src/gregg_limper/tools/handlers/` and register themselves with the shared decorator.
- The `retrieve_context` tool reuses the RAG pipeline to surface prior fragments only when the assistant asks for them, keeping the base prompt slim.
- Tool execution is logged (`response.__init__`), cached per call signature, and visible in `debug_messages.json` via synthetic `role: "tool"` entries.

### Background Maintenance

- `maintenance.startup` / `maintenance.shutdown` offer reusable periodic task wrappers.
- `memory/rag/sql/sql_tasks` re-embeds stale fragments to enforce `EMB_MODEL_ID`/`EMB_DIM`.
- `memory/rag/sql/admin` prunes fragments older than a retention threshold and vacuums the database.
- `memory/rag/vector/vector_tasks` keeps Milvus in sync, removing orphaned vectors and requesting compaction when necessary.
- `memory/rag/vector/health.validate_connection` provides an early GPU capability check to catch Milvus configuration issues on startup.

## Directory Layout

```text
.
├── src/gregg_limper/
│   ├── clients/           # Discord, OpenAI, and Ollama client facades
│   ├── commands/          # Slash command registration and handlers
│   ├── config/            # Environment-driven configuration dataclasses
│   ├── event_hooks/       # Discord event routers (ready/message/reaction)
│   ├── formatter/         # Message classification and fragment builders
│   ├── memory/            # Cache + RAG storage, ingestion, and scheduling
│   ├── response/          # Prompt assembly and completion orchestration
│   ├── tools/             # Tool registry, execution helpers, and handlers
│   └── maintenance.py     # Shared utilities for background tasks
├── tests/                 # Pytest suites covering cache, formatter, tools, RAG, commands
├── docs/                  # Milvus GPU setup and restart notebooks
├── data/                  # Default SQLite DB and memo snapshots (development)
├── requirements.txt       # Dependency pin list (mirrors extras in pyproject)
├── pyproject.toml         # Build metadata and dependency declarations
└── LICENSE                # MIT license
```

## Quick Start

### Prerequisites

- Python 3.10 or newer and a virtual environment manager (venv, pyenv, uv, etc.).
- Discord application with a bot token and required gateway intents enabled.
- OpenAI API key with access to the specified chat, embedding, image, and web models.
- Google Cloud API key for the YouTube Data API v3.
- Optional: Milvus 2.6+ with GPU support (see notebooks in `docs/`) and an Ollama server if you want to run the bot against a local model.

### Installation

```bash
git clone https://github.com/your-org/gregg-limper.git
cd gregg-limper
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .[test]     # editable install with test extras
# or: pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file and populate the secrets:
   ```bash
   cp .env.example .env
   ```
2. Update all required keys (Discord, OpenAI, Google Cloud) and set `CHANNEL_IDS` to the numeric IDs you want the bot to monitor.
3. If Milvus is not available, set `ENABLE_MILVUS=0` to skip vector indexing.
4. To use a local Ollama model, set `USE_LOCAL=1`, `LOCAL_SERVER_URL`, and `LOCAL_MODEL_ID`.
5. Configure `RAG_REACTION_EMOJIS` with the emoji descriptors (unicode, `<:name:id>`, `name:id`, numeric ID, or `:name:`) that should promote an opted-in message into long-term RAG storage.
6. (Optional) Customize the bot’s persona by pointing `PERSONA_PROMPT_FILE` to a text file (default `data/persona_prompt.txt`, not committed). When the file exists, its contents are appended to the system prompt.

### Run the Bot

```bash
python -m gregg_limper
```

During startup the bot will:
- Load environment variables from `.env` via `python-dotenv`.
- Validate Milvus connectivity when enabled.
- Hydrate the cache from Discord history for `CHANNEL_IDS`, backfilling only opted-in messages that already have a configured reaction trigger.
- Kick off RAG maintenance tasks on a recurring schedule.

### Run the Test Suite

```bash
pytest
```

Pytest targets are organized by subsystem (`tests/cache`, `tests/formatter`, `tests/rag`, etc.), and fixtures in `tests/conftest.py` provide Discord stubs plus temporary storage paths.

## Slash Commands

- `/help` — Lists all registered application commands.
- `/lobotomy` — Demonstration command that returns a playful acknowledgement.
- `/optin enabled:<bool>` — Toggle RAG consent for the caller. When enabling, the bot backfills recent history (bounded concurrency) but only ingests messages that already carry one of the configured reaction triggers, then notifies the user on completion.
- `/rag_status` — Report whether the caller is currently opted in to retrieval.

Additional commands can be added by creating new handlers under `src/gregg_limper/commands/handlers/` and decorating cogs with `@register_cog`.

## Configuration Reference

### Required Credentials

| Variable | Purpose |
| --- | --- |
| `DISCORD_API_TOKEN` | Bot token used by `discord.py` to connect and subscribe to events. |
| `OPENAI_API_KEY` | Grants access to chat, embedding, image, and web models used across the pipeline. |
| `GCLOUD_API_KEY` | Authenticates calls to the YouTube Data API for video metadata. |
| `BOT_USER_ID` | Numeric Discord user ID of the bot; used to filter self-mentions and seed consent. |
| `MSG_MODEL_ID` / `IMG_MODEL_ID` / `WEB_MODEL_ID` | Model identifiers for chat replies, vision captioning, and web summarization respectively. |
| `CHANNEL_IDS` | Comma-separated list of guild channel IDs that the cache should ingest. |

### Core Behaviour

| Variable | Default | Notes |
| --- | --- | --- |
| `CONTEXT_LENGTH` | `10` | Number of recent cached messages included in each prompt. |
| `MAX_IMAGE_MB` | `5` | Rejects oversized image attachments during captioning. |
| `MAX_GIF_MB` | `10` | Guards GIF downloads during metadata extraction. |
| `YT_THUMBNAIL_SIZE` | `medium` | Thumbnail size requested from the YouTube API. |
| `YT_DESC_MAX_LEN` | `200` | Truncation length for video descriptions in fragments. |
| `PERSONA_PROMPT_FILE` | `data/persona_prompt.txt` | Path to a gitignored text file containing persona instructions; if the file exists, its contents are appended to the system prompt. |

### Cache & Retrieval

| Variable | Default | Notes |
| --- | --- | --- |
| `CACHE_LENGTH` | `200` | Rolling window size per channel for the in-memory cache. |
| `MEMO_DIR` | `data/cache` | Directory where memo snapshots (`*.json.gz`) are persisted. |
| `CACHE_INIT_CONCURRENCY` | `20` | Parallelism when formatting history during startup. |
| `CACHE_INGEST_CONCURRENCY` | `20` | Max concurrent ingestion tasks during hydration. |
| `SQL_DB_DIR` | `data/memory.db` | SQLite file used for long-term storage. |
| `EMB_MODEL_ID` / `EMB_DIM` | `text-embedding-3-small` / `1536` | Embedding model and dimension enforced by maintenance. |
| `MAINTENANCE_INTERVAL` | `3600` | Seconds between maintenance cycles. |
| `RAG_OPT_IN_LOOKBACK_DAYS` | `180` | How far back to backfill when a user opts in. |
| `RAG_BACKFILL_CONCURRENCY` | `20` | Concurrency for RAG backfill ingestion tasks. |
| `RAG_REACTION_EMOJIS` | _(empty)_ | Comma-separated list of emoji descriptors that trigger RAG ingestion. Supported forms: unicode literals (`🔥`), full custom emoji (`<:greatprophet:123>`), shorthand `name:id`, bare numeric IDs (`123`), and names with or without surrounding colons (`WOOW` or `:WOOW:`). |
| `RAG_VECTOR_SEARCH_K` | `3` | Maximum RAG fragments returned per query (tool requests are clamped to this). |

### Vector Store & Local Models

| Variable | Default | Notes |
| --- | --- | --- |
| `ENABLE_MILVUS` | `1` | Toggle Milvus integration. Set to `0` to fall back to short-term cache only (RAG retrieval is disabled). |
| `MILVUS_HOST` / `MILVUS_PORT` | `127.0.0.1` / `19530` | Milvus connection parameters. |
| `MILVUS_COLLECTION` | `vectordb` | Collection name for fragment vectors. |
| `MILVUS_NLIST` / `MILVUS_NPROBE` / `MILVUS_DELETE_CHUNK` | `1024` / `32` / `800` | Index tuning knobs mirrored by maintenance utilities. |
| `USE_LOCAL` | `0` | When truthy, `response.handle` calls Ollama instead of OpenAI chat. |
| `LOCAL_MODEL_ID` | `gpt-oss-20b` | Ollama model identifier. |
| `LOCAL_SERVER_URL` | `http://localhost:11434` | Ollama server address. |

## Handler Registries

Formatter handlers, command cogs, and tool handlers all follow the same pattern:
modules live under a dedicated ``handlers`` package, decorate their class with
the package’s registration helper, and are auto-imported at module import time.
The tables below highlight the specifics for each subsystem.

### Formatter Handlers
| Item | Location / Notes |
| --- | --- |
| Directory | `src/gregg_limper/formatter/handlers/` |
| Decorator | `formatter.handlers.register` |
| Runtime API | `formatter.handlers.get(media_type)` |
| When adding | Update `formatter/composer.py`’s `ORDER`, teach `formatter/classifier.py` how to populate the slice, and ensure the corresponding `Fragment` class implements `content_text` for RAG ingestion (see `src/gregg_limper/formatter/handlers/__init__.py`). |

### Command Cogs
| Item | Location / Notes |
| --- | --- |
| Directory | `src/gregg_limper/commands/handlers/` |
| Decorator | `commands.register_cog` |
| Runtime API | `commands.setup(bot)` attaches every registered cog |
| When adding | Provide any Discord slash command definitions within the Cog, note that modules are imported eagerly, and run the bot (or `pytest tests/test_commands_setup.py`) to validate registration (see `src/gregg_limper/commands/__init__.py`). |

### Tool Handlers
| Item | Location / Notes |
| --- | --- |
| Directory | `src/gregg_limper/tools/handlers/` |
| Decorator | `tools.register_tool` |
| Runtime API | `tools.get_registered_tool_specs()` / `tools.get_tool_entry(name)` |
| When adding | Return a `ToolResult` from `run`, add or update tests in `tests/tools/`, consider logging/timeout behaviour, and document new configuration knobs if needed (see `src/gregg_limper/tools/__init__.py`). |

## Development Notes

- **Cache persistence:** Memo snapshots live under `MEMO_DIR` (default `data/cache`). Deleting the files will force the formatter to regenerate fragments on next startup.
- **SQLite storage:** The default database file is `SQL_DB_DIR` (default `data/memory.db`). Use `sqlite3` or `litecli` to inspect fragments, channel summaries, and consent tables. Maintenance jobs vacuum automatically, but you can run them manually via the `memory.rag` façade.
- **Milvus integration:** Ensure a GPU-capable index is available. The notebooks in `docs/` walk through deploying Milvus with GPU acceleration and restarting services. Set `ENABLE_MILVUS=0` to disable long-term RAG search; the bot will answer using only the short-term cache.
- **Local LLMs:** When `USE_LOCAL=1`, completions are routed through Ollama. Embeddings and vision still rely on OpenAI unless you swap out handlers in `clients/oai.py`/`memory/rag/embeddings.py`.

## Troubleshooting & Debugging

- **Missing environment variables:** `config.core.Core` raises a `ValueError` listing missing keys early in startup. Double-check your `.env`.
- **Milvus connection errors:** `ready_hook.handle` runs `vector.health.validate_connection` and will log detailed GPU index failures. Disable Milvus or update credentials if validation raises.
- **Consent issues:** `/rag_status` reflects the persisted consent table. Use `/optin enabled:false` to purge stored fragments for a user (`memory/rag/__init__.py::purge_user`).
- **No reactions, no ingest:** Opted-in messages only reach long-term storage when they have a reaction listed in `RAG_REACTION_EMOJIS`. Verify the config and make sure moderators react to important posts.
- **Prompt audits:** Each completion writes `debug_history.md`, `debug_context.md`, and `debug_messages.json` at the project root, mirroring exactly what was sent to the model.
- **Tool debugging:** Tool executions log their call IDs and arguments at INFO level (`gregg_limper.response`). Tool outputs appear in `debug_messages.json` as `role: "tool"` entries.
- **Cache visibility:** Enable INFO logging to see `Cached msg ...` previews coming from `memory/cache/manager.py`. Use `GLCache().list_formatted_messages` in a REPL to inspect memo payloads.

## Further Reading

- `docs/setup_milvus_gpu.ipynb` and `docs/restart_milvus_gpu.ipynb` for GPU-backed Milvus deployment notes.
- `tests/cache/` for cache hydration and eviction behavior.
- `tests/rag/` for consent flows, backfill parity, and vector synchronization scenarios.
- `tests/formatter/` for fragment classification and serialization expectations.

## License

Released under the [MIT License](LICENSE).
