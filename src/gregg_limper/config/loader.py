from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config.toml")


def load_raw_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the unified application config (config.toml by default).

    Returns an empty dict when the file is missing so callers can fall back to
    environment variables.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        return {}

    with target.open("rb") as handle:
        return tomllib.load(handle)


__all__ = ["load_raw_config", "DEFAULT_CONFIG_PATH"]
