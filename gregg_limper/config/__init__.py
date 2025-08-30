"""Application configuration"""

import logging
from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .core import Core
from .cache import Cache
from .rag import Rag
from .milvus import Milvus
from .local_llm import LocalLLM

core = Core()
cache = Cache()
rag = Rag()
milvus = Milvus()
local_llm = LocalLLM()


class Config:
    core = core
    cache = cache
    rag = rag
    milvus = milvus
    local_llm = local_llm


__all__ = ["core", "cache", "rag", "milvus", "local_llm", "Config"]
