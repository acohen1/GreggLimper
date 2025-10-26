from dataclasses import dataclass
import os


@dataclass
class LocalLLM:
    USE_LOCAL: bool = os.getenv("USE_LOCAL", "0").lower() in ("1", "true", "yes")
    LOCAL_MODEL_ID: str = os.getenv("LOCAL_MODEL_ID", "gpt-oss-20b")
    LOCAL_SERVER_URL: str = os.getenv("LOCAL_SERVER_URL", "http://localhost:11434")
