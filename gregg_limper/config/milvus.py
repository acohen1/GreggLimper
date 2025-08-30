from dataclasses import dataclass
import os


@dataclass
class Milvus:
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "vectordb")
    MILVUS_NLIST: int = int(os.getenv("MILVUS_NLIST", "1024"))
    MILVUS_NPROBE: int = int(os.getenv("MILVUS_NPROBE", "32"))
    MILVUS_DELETE_CHUNK: int = int(os.getenv("MILVUS_DELETE_CHUNK", "800"))
    MILVUS_OPTIONAL: bool = os.getenv("MILVUS_OPTIONAL", "0").lower() in ("1", "true", "yes")
