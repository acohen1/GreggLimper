"""
Short-term message cache package.

Modules
=======

``manager``
    Defines :class:`~gregg_limper.memory.cache.manager.GLCache`, the singleton
    cache coordinator that orchestrates channel state, memo persistence, and
    ingestion.
``channel_state``
    Provides :class:`~gregg_limper.memory.cache.channel_state.ChannelCacheState`
    to encapsulate per-channel message buffers and membership indices.
``memo_store``
    Maintains memoized message payloads in memory and synchronizes them with the
    on-disk memo files.
``memo``
    Low-level gzip JSON helpers used by :mod:`memo_store` for reading and
    writing memo files.
``formatting``
    Async helpers that call the shared formatter to build memo payloads,
    including concurrency-limited utilities for bulk formatting.
``serialization``
    Produces caller-facing views of memo payloads, tailoring the serialized
    fragments for LLM or full fidelity consumption.
``ingestion``
    Implements the consent checks and persistence hooks that push memoized
    messages into downstream RAG stores when requested.
``initializer``
    Drives Discord history hydration on startup, seeding channel caches and
    backfilling memo snapshots.
``utils``
    Internal logging helpers used by :mod:`manager` to generate concise fragment
    summaries.
"""

from .manager import GLCache

__all__ = ["GLCache"]
