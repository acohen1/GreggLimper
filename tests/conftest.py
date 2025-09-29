import os, sys
import warnings
from pathlib import Path
import types

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure required environment variables for Config.validate()
os.environ.setdefault("DISCORD_API_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("BOT_USER_ID", "1")
os.environ.setdefault("GCLOUD_API_KEY", "test-gcloud")
os.environ.setdefault("COT_MODEL_ID", "model")
os.environ.setdefault("MSG_MODEL_ID", "model")
os.environ.setdefault("IMG_MODEL_ID", "model")
os.environ.setdefault("WEB_MODEL_ID", "model")
os.environ.setdefault("CHANNEL_IDS", "123")

# Silence deprecation warnings surfaced from third-party dependencies during tests
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"^discord\\.player$"
)
warnings.filterwarnings("ignore", "'audioop' is deprecated", DeprecationWarning)


def pytest_configure(config):
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=r"^discord\\.player$"
    )
    warnings.filterwarnings("ignore", "'audioop' is deprecated", DeprecationWarning)

# Stub pymilvus to avoid heavy dependency during tests
pymilvus_stub = types.SimpleNamespace(
    Collection=object,
    CollectionSchema=object,
    DataType=types.SimpleNamespace(INT64=0, FLOAT_VECTOR=1),
    FieldSchema=object,
    connections=types.SimpleNamespace(connect=lambda **k: None),
    utility=types.SimpleNamespace(has_collection=lambda name: False),
)
sys.modules.setdefault("pymilvus", pymilvus_stub)
