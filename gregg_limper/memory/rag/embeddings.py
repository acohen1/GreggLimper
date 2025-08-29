"""
Embedding utilities
===================

Centralizes embedding logic so the rest of the codebase does not care
about model details (dimensionality, provider, etc.).
"""

from __future__ import annotations
import numpy as np
import hashlib
from gregg_limper.clients.oai import embed_text
from gregg_limper.config import Config
import logging

logger = logging.getLogger(__name__)

async def embed(text: str) -> np.ndarray:
    try:
        return await embed_text(text)
    except Exception as e:
        logger.error(f"Error embedding text: {e}. Defaulting to zeros vector.")
        return np.zeros((Config.EMB_DIM,), dtype=np.float32)

def to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def blake16(s: str) -> str:
    return hashlib.blake2b((s or "").encode("utf-8"), digest_size=16).hexdigest()

