# Configuration Guide

This repo uses a unified `config.toml` at the project root and a minimal `.env` for secrets. Copy `config/config.sample.toml` to `config.toml` and `config/.env.example` to `.env`, then fill in your values.

## Secrets (environment)

Set these in `.env` (the `token_env` keys in `config.toml` reference these names):

- `DISCORD_API_TOKEN`: Discord bot token.
- `OPENAI_API_KEY`: OpenAI API key.
- `GCLOUD_API_KEY`: YouTube Data API key.

## Config Sections (config.toml)

### [gregglimper.discord]
- `channel_ids`: Guild channel IDs to monitor.
- `token_env`: Env var name for the Discord token (default `DISCORD_API_TOKEN`).
- `bot_user_id`: Numeric user ID of the bot.

### [gregglimper.models]
- `message_model`: Chat model for replies.
- `image_model`: Vision model for image/GIF captions.
- `web_model`: Model used for web summaries.

### [gregglimper.limits]
- `context_length`: Recent cached messages included per prompt.
- `max_image_mb`: Reject images above this size.
- `max_gif_mb`: Reject GIFs above this size.
- `yt_thumbnail_size`: YouTube thumbnail size.
- `yt_desc_max_len`: Truncation length for YouTube descriptions.

### [gregglimper.cache]
- `cache_length`: Rolling window size per channel.
- `memo_dir`: Directory for memo snapshots.
- `cache_init_concurrency`: Concurrency when formatting history on startup.
- `cache_ingest_concurrency`: Concurrency for ingestion tasks.

### [gregglimper.retrieval]
- `sql_db_dir`: SQLite path for long-term storage.
- `emb_model_id` / `emb_dim`: Embedding model and dimension.
- `maintenance_interval`: Seconds between maintenance cycles.
- `rag_opt_in_lookback_days`: Backfill window when users opt in.
- `rag_backfill_concurrency`: Concurrency for backfill ingestion.
- `rag_reaction_emojis`: Emoji descriptors that trigger RAG ingestion (unicode, `<:name:id>`, `name:id`, numeric ID, or `:name:`).
- `rag_vector_search_k`: Max fragments returned per retrieval/tool call.

### [gregglimper.milvus]
- `enable_milvus`: Toggle Milvus integration.
- `host`, `port`, `collection`, `nlist`, `nprobe`, `delete_chunk`: Vector index settings.

### [gregglimper.local_llm]
- `use_local`: Route chat through Ollama instead of OpenAI.
- `local_model_id`, `local_server_url`: Ollama model and server settings.

### [finetune.*] (tuner)
- `dataset`: Channels, allowed users, earliest timestamp, caps, paths, emoji whitelist, reuse/scrub flags.
- `models`: Segment/moderation/relevance model IDs.
- `discord`: `token_env` for the tuner Discord client.

