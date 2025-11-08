from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _split_ids(raw: str) -> List[int]:
    return [int(cid.strip()) for cid in raw.split(",") if cid.strip()]


@dataclass
class Core:
    DISCORD_API_TOKEN: str = os.getenv("DISCORD_API_TOKEN")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    BOT_USER_ID: int = int(os.getenv("BOT_USER_ID", "0"))
    GCLOUD_API_KEY: str = os.getenv("GCLOUD_API_KEY")
    MSG_MODEL_ID: str = os.getenv("MSG_MODEL_ID")
    IMG_MODEL_ID: str = os.getenv("IMG_MODEL_ID")
    WEB_MODEL_ID: str = os.getenv("WEB_MODEL_ID")
    CHANNEL_IDS: List[int] = field(default_factory=lambda: _split_ids(os.getenv("CHANNEL_IDS", "")))
    CONTEXT_LENGTH: int = int(os.getenv("CONTEXT_LENGTH", "10"))
    MAX_IMAGE_MB: int = int(os.getenv("MAX_IMAGE_MB", "5"))
    MAX_GIF_MB: int = int(os.getenv("MAX_GIF_MB", "10"))
    YT_THUMBNAIL_SIZE: str = os.getenv("YT_THUMBNAIL_SIZE", "medium")
    YT_DESC_MAX_LEN: int = int(os.getenv("YT_DESC_MAX_LEN", "200"))
    PERSONA_PROMPT_FILE: str = os.getenv("PERSONA_PROMPT_FILE", "data/persona_prompt.txt")
    persona_prompt: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        required = [
            ("DISCORD_API_TOKEN", self.DISCORD_API_TOKEN),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("GCLOUD_API_KEY", self.GCLOUD_API_KEY),
            ("BOT_USER_ID", self.BOT_USER_ID),
            ("MSG_MODEL_ID", self.MSG_MODEL_ID),
            ("IMG_MODEL_ID", self.IMG_MODEL_ID),
            ("WEB_MODEL_ID", self.WEB_MODEL_ID),
            ("CHANNEL_IDS", self.CHANNEL_IDS),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        self.persona_prompt = self._load_persona_prompt()

    def _load_persona_prompt(self) -> str:
        """Load persona text from the configured file."""

        file_path = (self.PERSONA_PROMPT_FILE or "").strip()
        if not file_path:
            return ""

        path = Path(file_path)
        if not path.is_file():
            return ""

        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to read persona prompt file %s: %s", path, exc)
            return ""
