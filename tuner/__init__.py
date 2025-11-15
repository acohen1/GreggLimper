"""Finetuning helpers for Gregg Limper."""

from __future__ import annotations

import logging

from dotenv import load_dotenv

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

load_dotenv()
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .cli import main

__all__ = ["main"]
