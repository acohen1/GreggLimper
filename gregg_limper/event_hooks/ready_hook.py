import discord
from gregg_limper.memory.cache import GLCache
from gregg_limper.config import core, milvus, rag
from gregg_limper.memory.rag import scheduler
from gregg_limper.memory.rag.vector.health import validate_connection

import logging

logger = logging.getLogger(__name__)

async def handle(client: discord.Client):
    """Cache initialization on client ready event."""
    logger.info(f"Logged in as {client.user.name} (ID: {client.user.id})")

    # Validate Milvus GPU connection for RAG vector indexing
    try:
        validate_connection()
        logger.info("Milvus GPU validation succeeded")
    except Exception as e:
        if not milvus.MILVUS_OPTIONAL:
            logger.error("Milvus validation failed: %s", e)
        else:
            raise
    
    # Initialize GLCache (hydrate with discord data)
    logger.info(f"Initializing GLCache with configured channel IDs: {core.CHANNEL_IDS}")
    cache = GLCache()
    c_ids = [cid for cid in core.CHANNEL_IDS]
    await cache.initialize(client, c_ids)

    # Kick off RAG embedding maintenance after cache initialization
    await scheduler.start(rag.MAINTENANCE_INTERVAL)
