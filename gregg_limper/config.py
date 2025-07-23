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
    CHANNEL_IDS = _split_ids(os.getenv("CHANNEL_IDS"))

    # Cache settings
    CACHE_LENGTH = int(os.getenv("CACHE_LENGTH", "1000"))

    # max size for image attachments
    MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "5"))
    MAX_GIF_MB = int(os.getenv("MAX_GIF_MB", "10"))

    # ----- validation --------------------------------------------------- #

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
