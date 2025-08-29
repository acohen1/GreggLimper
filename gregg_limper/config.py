import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure global logging
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    level=logging.INFO
)

def _split_ids(raw: str) -> list[str]:
    """Split a comma-separated string, trim whitespace, drop empties"""
    return [cid.strip() for cid in raw.split(",") if cid.strip()]

class Config:
    # ------------------------------------------- Required settings ------------------------------------------- #

    # Required secrets
    DISCORD_API_TOKEN = os.getenv("DISCORD_API_TOKEN")
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    BOT_USER_ID = int(os.getenv("BOT_USER_ID"))
    GCLOUD_API_KEY = os.getenv("GCLOUD_API_KEY")

    # Model selection
    COT_MODEL_ID = os.getenv("COT_MODEL_ID")
    MSG_MODEL_ID = os.getenv("MSG_MODEL_ID")
    IMG_MODEL_ID = os.getenv("IMG_MODEL_ID")
    WEB_MODEL_ID = os.getenv("WEB_MODEL_ID")

    # Channel allow-list (must not be empty)
    CHANNEL_IDS = list(map(int, _split_ids(os.getenv("CHANNEL_IDS"))))

    # ------------------------------------------- Optional settings ------------------------------------------- #

    # -- Cache settings --
    CACHE_LENGTH = int(os.getenv("CACHE_LENGTH", "1000"))

    # Response context length (number of messages) --
    CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "50"))

    # max size for image attachment processing
    MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "5"))
    MAX_GIF_MB = int(os.getenv("MAX_GIF_MB", "10"))

    # Youtube processing settings
    YT_THUMBNAIL_SIZE = os.getenv("YT_THUMBNAIL_SIZE", "medium")        # "default", "medium", "high"
    YT_DESC_MAX_LEN = int(os.getenv("YT_DESC_MAX_LEN", "200"))          # max length of video description

    # ------------------------ RAG settings ------------------------ #
    DB_NAME = os.getenv("DB_NAME", "memory.db")
    EMB_MODEL_ID = os.getenv("EMB_MODEL", "text-embedding-3-small")
    EMB_DIM = int(os.getenv("EMB_DIM", 1536))

    # How often to run SQL/Vector DB maintenance tasks (in seconds)
    MAINTENANCE_INTERVAL = int(os.getenv("MAINTENANCE_INTERVAL", "3600"))

    # Milvus settings
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "vectordb")

    # IVF_FLAT index: how many clusters (buckets) to partition the embedding space into.
    # Larger values = finer partitioning (higher recall, more memory/CPU); smaller = coarser (faster, less accurate).
    MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "1024"))

    # Search-time parameter: how many clusters to probe when querying.
    # Larger values = more accurate (higher recall, more latency); smaller = faster (lower recall).
    MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "32"))

    # Maximum number of IDs to include in a single "DELETE rid in (...)" expression.
    # Prevents oversized delete expressions during bulk upserts.
    MILVUS_DELETE_CHUNK = int(os.getenv("MILVUS_DELETE_CHUNK", "800"))

    # If true, startup won't fail when Milvus is unavailable
    MILVUS_OPTIONAL = os.getenv("MILVUS_OPTIONAL", "0").lower() in ("1", "true", "yes")

    # -- Local LLM settings --
    USE_LOCAL = os.getenv("USE_LOCAL", "0").lower() in ("1", "true", "yes")
    LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "gpt-oss-20b")
    LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://localhost:11434")

    # ----- Validation --------------------------------------------------- #

    @classmethod
    def validate(cls) -> None:
        """
        Ensure all required env vars are present and CHANNEL_IDS is non-empty.
        """
        required = [
            ("DISCORD_API_TOKEN", cls.DISCORD_API_TOKEN),
            ("OPENAI_API_KEY",    cls.OPENAI_API_KEY),
            ("GCLOUD_API_KEY",    cls.GCLOUD_API_KEY),
            ("BOT_USER_ID",       cls.BOT_USER_ID),
            ("COT_MODEL_ID",      cls.COT_MODEL_ID),
            ("MSG_MODEL_ID",      cls.MSG_MODEL_ID),
            ("IMG_MODEL_ID",      cls.IMG_MODEL_ID),
            ("WEB_MODEL_ID",      cls.WEB_MODEL_ID),
            ("CHANNEL_IDS",       cls.CHANNEL_IDS),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# Run automatic validation as soon as the module is imported
Config.validate()

