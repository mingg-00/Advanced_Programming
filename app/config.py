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
    """기본 종목 코드 반환 (삼성전자: 005930)"""
    # 환경변수에서 기본 종목 코드를 가져오고, 없으면 삼성전자(005930) 반환
    # .KS 접미사는 yfinance용이므로 제거하고 6자리 코드만 반환
    default = os.getenv("DEFAULT_SYMBOL", "005930")
    return default.replace(".KS", "") if default else "005930"


