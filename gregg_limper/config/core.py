from dataclasses import dataclass, field
import os
from typing import List


def _split_ids(raw: str) -> List[int]:
    return [int(cid.strip()) for cid in raw.split(",") if cid.strip()]


@dataclass
class Core:
    DISCORD_API_TOKEN: str = os.getenv("DISCORD_API_TOKEN")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    BOT_USER_ID: int = int(os.getenv("BOT_USER_ID", "0"))
    GCLOUD_API_KEY: str = os.getenv("GCLOUD_API_KEY")
    COT_MODEL_ID: str = os.getenv("COT_MODEL_ID")
    MSG_MODEL_ID: str = os.getenv("MSG_MODEL_ID")
    IMG_MODEL_ID: str = os.getenv("IMG_MODEL_ID")
    WEB_MODEL_ID: str = os.getenv("WEB_MODEL_ID")
    CHANNEL_IDS: List[int] = field(default_factory=lambda: _split_ids(os.getenv("CHANNEL_IDS", "")))
    CONTEXT_LENGTH: int = int(os.getenv("CONTEXT_LENGTH", "10"))
    MAX_IMAGE_MB: int = int(os.getenv("MAX_IMAGE_MB", "5"))
    MAX_GIF_MB: int = int(os.getenv("MAX_GIF_MB", "10"))
    YT_THUMBNAIL_SIZE: str = os.getenv("YT_THUMBNAIL_SIZE", "medium")
    YT_DESC_MAX_LEN: int = int(os.getenv("YT_DESC_MAX_LEN", "200"))

    def __post_init__(self) -> None:
        required = [
            ("DISCORD_API_TOKEN", self.DISCORD_API_TOKEN),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("GCLOUD_API_KEY", self.GCLOUD_API_KEY),
            ("BOT_USER_ID", self.BOT_USER_ID),
            ("COT_MODEL_ID", self.COT_MODEL_ID),
            ("MSG_MODEL_ID", self.MSG_MODEL_ID),
            ("IMG_MODEL_ID", self.IMG_MODEL_ID),
            ("WEB_MODEL_ID", self.WEB_MODEL_ID),
            ("CHANNEL_IDS", self.CHANNEL_IDS),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
