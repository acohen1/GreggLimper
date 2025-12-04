import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _split_ids(raw: str) -> List[int]:
    return [int(cid.strip()) for cid in raw.split(",") if cid.strip()]


class Core:
    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("gregglimper", {})
        discord_cfg = cfg.get("discord", {})
        models_cfg = cfg.get("models", {})
        limits_cfg = cfg.get("limits", {})

        token_env = str(discord_cfg.get("token_env", "DISCORD_API_TOKEN"))
        openai_env = str(discord_cfg.get("openai_key_env", "OPENAI_API_KEY"))
        gcloud_env = str(discord_cfg.get("gcloud_key_env", "GCLOUD_API_KEY"))

        self.DISCORD_API_TOKEN: str | None = os.getenv(token_env)
        self.OPENAI_API_KEY: str | None = os.getenv(openai_env)
        self.GCLOUD_API_KEY: str | None = os.getenv(gcloud_env)

        self.BOT_USER_ID: int = int(discord_cfg.get("bot_user_id") or os.getenv("BOT_USER_ID", "0"))
        channel_ids_cfg = discord_cfg.get("channel_ids")
        if channel_ids_cfg:
            self.CHANNEL_IDS: List[int] = [int(cid) for cid in channel_ids_cfg]
        else:
            self.CHANNEL_IDS = _split_ids(os.getenv("CHANNEL_IDS", ""))

        self.MSG_MODEL_ID: str | None = models_cfg.get("message_model") or os.getenv("MSG_MODEL_ID")
        self.IMG_MODEL_ID: str | None = models_cfg.get("image_model") or os.getenv("IMG_MODEL_ID")
        self.WEB_MODEL_ID: str | None = models_cfg.get("web_model") or os.getenv("WEB_MODEL_ID")
        self.RELEVANCY_CHECK_MODEL_ID: str | None = models_cfg.get("relevancy_check_model") or os.getenv("RELEVANCY_CHECK_MODEL_ID")
        self.TOOL_CHECK_MODEL_ID: str | None = models_cfg.get("tool_check_model") or os.getenv("TOOL_CHECK_MODEL_ID") or self.RELEVANCY_CHECK_MODEL_ID

        self.CONTEXT_LENGTH: int = int(limits_cfg.get("context_length", os.getenv("CONTEXT_LENGTH", "10")))
        self.RELEVANCY_CHECK_MAX_LOOPS: int = int(limits_cfg.get("relevancy_check_max_loops", os.getenv("RELEVANCY_CHECK_MAX_LOOPS", "3")))
        self.RELEVANCY_REGEN_TEMP_MIN: float = float(limits_cfg.get("min_regen_temp", os.getenv("MIN_REGEN_TEMP", "0.7")))
        self.RELEVANCY_REGEN_TEMP_MAX: float = float(limits_cfg.get("max_regen_temp", os.getenv("MAX_REGEN_TEMP", "1.0")))
        self.MAX_IMAGE_MB: int = int(limits_cfg.get("max_image_mb", os.getenv("MAX_IMAGE_MB", "5")))
        self.MAX_GIF_MB: int = int(limits_cfg.get("max_gif_mb", os.getenv("MAX_GIF_MB", "10")))
        self.YT_THUMBNAIL_SIZE: str = str(limits_cfg.get("yt_thumbnail_size", os.getenv("YT_THUMBNAIL_SIZE", "medium")))
        self.YT_DESC_MAX_LEN: int = int(limits_cfg.get("yt_desc_max_len", os.getenv("YT_DESC_MAX_LEN", "200")))
        self.PERSONA_PROMPT_FILE: str = str(
            cfg.get("persona_prompt_file", os.getenv("PERSONA_PROMPT_FILE", "data/persona_prompt.txt"))
        )

        self.persona_prompt: str = self._load_persona_prompt()

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
            logger.info("Persona prompt file %s not found; skipping.", path)
            return ""

        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to read persona prompt file %s: %s", path, exc)
            return ""
