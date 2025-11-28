"""
Wrapper around Kiwoom OpenAPI+ via pykiwoom.

The Kiwoom OpenAPI+ uses COM objects that are only supported on Windows.  The
client below keeps the rest of the application platform agnostic by guarding
imports and providing helpful runtime errors when executed on an unsupported
OS (such as macOS or Linux).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class KiwoomUnavailableError(RuntimeError):
    """Raised when Kiwoom OpenAPI+ is not available on the current platform."""


def _ensure_pykiwoom_available() -> None:
    if os.name != "nt":
        raise KiwoomUnavailableError(
            "Kiwoom OpenAPI+ is only supported on Windows. "
            "Run the Streamlit app from a Windows machine with Kiwoom OpenAPI+ installed."
        )
    try:
        # Import inside the guard to avoid ImportError on unsupported platforms.
        import pykiwoom  # noqa: F401
    except ModuleNotFoundError as exc:
        raise KiwoomUnavailableError(
            "pykiwoom is not installed. Install requirements with `pip install -r requirements.txt`."
        ) from exc


@dataclass
class KiwoomCredentials:
    user_id: str
    password: str
    cert_password: Optional[str] = None


class KiwoomClient:
    """Small façade over pykiwoom's Kiwoom class."""

    def __init__(self) -> None:
        self._kiwoom = None

    def connect(self, credentials: Optional[KiwoomCredentials] = None) -> bool:
        _ensure_pykiwoom_available()

        from pykiwoom.kiwoom import Kiwoom  # type: ignore  # pylint: disable=import-error

        if self._kiwoom is None:
            self._kiwoom = Kiwoom()

        if credentials:
            logger.info("Attempting auto login with provided credentials")
            self._kiwoom.CommConnect()
        else:
            logger.info("Triggering Kiwoom login window")
            self._kiwoom.CommConnect(block=True)

        connected = bool(self._kiwoom.GetConnectState())
        logger.info("Kiwoom connection state: %s", connected)
        return connected

    def is_connected(self) -> bool:
        return bool(self._kiwoom and self._kiwoom.GetConnectState())

    def fetch_daily_ohlcv(
        self,
        code: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        _ensure_pykiwoom_available()

        from pykiwoom.kiwoom import Kiwoom  # type: ignore  # pylint: disable=import-error

        if self._kiwoom is None:
            self._kiwoom = Kiwoom()

        raw = self._kiwoom.block_request(
            "opt10081",
            종목코드=code,
            기준일자=end.strftime("%Y%m%d"),
            수정주가구분=1,
        )

        df = pd.DataFrame(raw)
        df["일자"] = pd.to_datetime(df["일자"])
        df = df[(df["일자"] >= start) & (df["일자"] <= end)]
        df = df.rename(
            columns={
                "일자": "date",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "현재가": "close",
                "거래량": "volume",
            }
        )
        df = df.sort_values("date").set_index("date")
        return df

    @lru_cache(maxsize=128)
    def get_stock_codes(self, market: str = "0") -> pd.DataFrame:
        _ensure_pykiwoom_available()

        from pykiwoom.kiwoom import Kiwoom  # type: ignore  # pylint: disable=import-error

        if self._kiwoom is None:
            self._kiwoom = Kiwoom()

        codes = self._kiwoom.GetCodeListByMarket(market)
        names = [self._kiwoom.GetMasterCodeName(code) for code in codes]
        return pd.DataFrame({"code": codes, "name": names})


