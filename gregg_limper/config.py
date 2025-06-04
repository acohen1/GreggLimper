import os
from dotenv import load_dotenv

load_dotenv()

def _split_ids(raw: str) -> list[str]:
    """Split a comma-separated string, trim whitespace, drop empties"""
    return [cid.strip() for cid in raw.split(",") if cid.strip()]

class Config:
    # Required secrets
    DISCORD_API_TOKEN = os.getenv("DISCORD_API_TOKEN")
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

    BOT_USER_ID = int(os.getenv("BOT_USER_ID", "0"))  # Must be an integer

    # Model selection
    COT_MODEL_ID = os.getenv("COT_MODEL_ID")
    MSG_MODEL_ID = os.getenv("MSG_MODEL_ID")
    IMG_MODEL_ID = os.getenv("IMG_MODEL_ID")
    WEB_MODEL_ID = os.getenv("WEB_MODEL_ID")

    # Cache settings
    CACHE_LENGTH = int(os.getenv("CACHE_LENGTH", "1000"))

    # Channel allow-list (must not be empty)
    CHANNEL_IDS = _split_ids(os.getenv("CHANNEL_IDS", ""))

    # max size for image attachments
    MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "5"))

    # ----- validation --------------------------------------------------- #

    @classmethod
    def validate(cls) -> None:
        """
        Ensure all required env vars are present and CHANNEL_IDS is non-empty.
        """
        required = [
            ("DISCORD_API_TOKEN", cls.DISCORD_API_TOKEN),
            ("OPENAI_API_KEY",    cls.OPENAI_API_KEY),
            ("CHANNEL_IDS",       cls.CHANNEL_IDS),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# Run automatic validation as soon as the module is imported
Config.validate()
