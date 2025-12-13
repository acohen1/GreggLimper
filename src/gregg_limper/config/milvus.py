import os


class Milvus:
    def __init__(self, config: dict | None = None) -> None:
        milvus_cfg = (config or {}).get("gregglimper", {}).get("milvus", {})
        self.MILVUS_HOST: str = str(milvus_cfg.get("host", os.getenv("MILVUS_HOST", "127.0.0.1")))
        self.MILVUS_PORT: str = str(milvus_cfg.get("port", os.getenv("MILVUS_PORT", "19530")))
        self.MILVUS_COLLECTION: str = str(milvus_cfg.get("collection", os.getenv("MILVUS_COLLECTION", "vectordb")))
        self.MILVUS_NLIST: int = int(milvus_cfg.get("nlist", os.getenv("MILVUS_NLIST", "1024")))
        self.MILVUS_NPROBE: int = int(milvus_cfg.get("nprobe", os.getenv("MILVUS_NPROBE", "32")))
        self.MILVUS_DELETE_CHUNK: int = int(milvus_cfg.get("delete_chunk", os.getenv("MILVUS_DELETE_CHUNK", "800")))
        enable_raw = milvus_cfg.get("enable_milvus", os.getenv("ENABLE_MILVUS", "1"))
        self.ENABLE_MILVUS: bool = str(enable_raw).lower() in ("1", "true", "yes")
