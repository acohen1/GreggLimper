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
- [Handler Registries](#handler-registries)
- [Tuner Package](#tuner-package)
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
- **Multi-Pass Response Accumulation:** Iteratively prompts the model to flesh out brief responses, using a lightweight classifier to ensure completeness (`response/accumulator.py`).
- Debug-first ergonomics: cached fragments persist to disk, and every completion request records context and message payloads under `data/runtime/` (`debug_history.md`, `debug_context.md`, `debug_messages.json`).

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

### Prompt Orchestration

`response.handle` orchestrates a modular Chain-of-Thought (CoT) pipeline:

1.  **Context Gathering (`steps/context.py`)**: Fetches history, channel summaries, and user profiles to build the initial `PromptPayload`.
2.  **Tool Execution (`steps/tools.py`)**: A dedicated "smart model" (`TOOL_CHECK_MODEL_ID`) analyzes the context to decide if tools are needed. It executes them *before* the main generation, injecting results as system messages and capturing "artifacts" (like GIFs) for direct inclusion.
3.  **Reasoning (`steps/reasoning.py`)**: A "Reasoning Trace" is generated by a smart model (`REASONING_MODEL_ID`). It analyzes the conversation and tool results to produce a hidden internal monologue/plan, which is injected as a system instruction to guide the persona.
4.  **Generation (`steps/generation.py`)**: The main persona model generates the response text, following the reasoning plan and freed from the responsibility of calling tools itself.
5.  **Refinement (`steps/refinement.py`)**: If configured, a "Detail Classifier" checks the response for completeness. If lacking, it triggers an iterative rewrite loop to flesh out the details.

### Tool Calling

- Tool metadata lives in `src/gregg_limper/tools/__init__.py`; individual handlers reside in `src/gregg_limper/tools/handlers/`.
- **Decoupled Execution**: Tools are no longer called by the persona model. Instead, the `ToolExecutionStep` uses a specialized model (e.g., `gpt-5.1-nano`) to handle all functional logic, ensuring reliability and separating "reasoning" from "voice".
- The `retrieve_context` tool reuses the RAG pipeline to surface prior fragments only when the assistant asks for them.
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
│   ├── response/          # CoT Pipeline Engine
│   │   ├── steps/         # Pipeline steps (Context, Tools, Gen, Refine)
│   │   └── sources/       # Data sources (History, Payload, Context)
│   ├── tools/             # Tool registry, execution helpers, and handlers
│   └── maintenance.py     # Shared utilities for background tasks
├── tuner/                 # Standalone finetuner CLI (see tuner/README.md)
├── tests/                 # Root test tree (`tests/gregg_limper/`, `tests/tuner/`)
├── docs/                  # Milvus GPU setup and restart notebooks
├── data/                  # Default SQLite DB and memo snapshots (development)
├── config.toml            # Unified config file (copy from config/config.sample.toml)
├── .env                   # Environment variable overrides (copy from config/.env.example)
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

1. Copy the unified config template and tailor it:
   ```bash
   cp config/config.sample.toml config.toml
   ```
2. Copy `config/.env.example` to `.env` and add secrets only (Discord, OpenAI, Google Cloud).
3. See [config/CONFIG.md](config/CONFIG.md) for the full field breakdown (bot runtime plus `[finetune.*]` for the tuner) and the env var details.

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
pytest                    # run every suite (core + tuner)
pytest tests/gregg_limper # core bot suites only
pytest tests/tuner        # finetuner suites only
```

Pytest targets live under two top-level packages:
- `tests/gregg_limper/` — core bot suites (cache, formatter, rag, tools, etc.) with their own scoped `conftest.py`.
- `tests/tuner/` — finetuner CLI coverage.
`tests/conftest.py` carries any shared fixtures across packages.

## Slash Commands

- `/help` — Lists all registered application commands.
- `/lobotomy` — Demonstration command that returns a playful acknowledgement.
- `/optin enabled:<bool>` — Toggle RAG consent for the caller. When enabling, the bot backfills recent history (bounded concurrency) but only ingests messages that already carry one of the configured reaction triggers, then notifies the user on completion.
- `/rag_status` — Report whether the caller is currently opted in to retrieval.

Additional commands can be added by creating new handlers under `src/gregg_limper/commands/handlers/` and decorating cogs with `@register_cog`.

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
| When adding | Provide any Discord slash command definitions within the Cog, note that modules are imported eagerly, and run the bot (or `pytest tests/gregg_limper/test_commands_setup.py`) to validate registration (see `src/gregg_limper/commands/__init__.py`). |

### Tool Handlers
| Item | Location / Notes |
| --- | --- |
| Directory | `src/gregg_limper/tools/handlers/` |
| Decorator | `tools.register_tool` |
| Runtime API | `tools.get_registered_tool_specs()` / `tools.get_tool_entry(name)` |
| When adding | Return a `ToolResult` from `run`, add or update tests in `tests/gregg_limper/tools/`, consider logging/timeout behaviour, and document new configuration knobs if needed (see `src/gregg_limper/tools/__init__.py`). |

## Tuner Package

Need a supervised dataset that mirrors Gregg Limper's prompt stack? The repo ships with a standalone tuner CLI under [`tuner/`](tuner/README.md). It collects Discord history (respecting whitelisted speakers and earliest cutoffs), injects synthetic `retrieve_context` calls, and exports JSONL records that match OpenAI's chat finetune schema.

- Use the root `config.toml` `[finetune.*]` sections to set channels, allowed users, earliest timestamp, paths, and model IDs (tuner defaults to that file; you can still pass `--config` to point elsewhere).
- Run `python -m tuner build-dataset`; CLI flags override any TOML value when you need to experiment with alternate slices.
- Progress logs report per-channel hydration, LLM segment approvals, and the final supervised sample tally so you can monitor long-running builds.

See [tuner/README.md](tuner/README.md) for full configuration and schema details.

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
- **Prompt audits:** Each completion writes `debug_history.md`, `debug_context.md`, and `debug_messages.json` under `data/runtime/`, mirroring exactly what was sent to the model.
- **Pipeline Tracing:** The `PipelineTracer` writes a comprehensive JSON log to `data/runtime/pipeline_trace.json` after every step (`Context`, `Tools`, `Reasoning`, `Gen`, `Refine`), capturing the exact state evolution of the response.
- **Tool debugging:** Tool executions log their call IDs and arguments at INFO level (`gregg_limper.response`). Tool outputs appear in `debug_messages.json` as `role: "tool"` entries.
- **Cache visibility:** Enable INFO logging to see Cached msg ... previews coming from memory/cache/manager.py. Use GLCache().list_formatted_messages in a REPL to inspect memo payloads.
- **Response Accumulation:** If the bot feels "stuck" in a loop or responses are too long, check DETAIL_CHECK_MAX_LOOPS (default 3) or disable the feature by unsetting DETAIL_CHECK_MODEL_ID.

## Further Reading

- `docs/setup_milvus_gpu.ipynb` for GPU-backed Milvus deployment notes.
- `src/gregg_limper/memory/cache/__init__.py` for cache hydration, eviction, and memo persistence internals.
- `src/gregg_limper/memory/rag/__init__.py` for consent flows, backfill parity, and vector synchronization routines.
- `src/gregg_limper/formatter/handlers/__init__.py` for fragment classification/serialization expectations.

## License

Released under the [MIT License](LICENSE).
