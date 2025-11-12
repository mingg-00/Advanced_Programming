from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


def get_openai_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def get_default_symbol() -> str:
    return os.getenv("DEFAULT_SYMBOL", "005930.KS")


