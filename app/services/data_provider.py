"""
Unified entry point for market data.

Uses Kiwoom OpenAPI+ when available, otherwise falls back to Yahoo Finance via
`yfinance` so that developers can iterate on non-Windows environments.  The
fallback should only be used for development because the project requirements
expect Kiwoom data in production.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .kiwoom_client import KiwoomClient, KiwoomUnavailableError

logger = logging.getLogger(__name__)


class MarketDataProvider:
    def __init__(self) -> None:
        self._kiwoom = KiwoomClient()
        self._fallback = None

    def _ensure_fallback(self):
        if self._fallback is None:
            self._fallback = self._init_fallback()
        return self._fallback

    def connect(self) -> None:
        try:
            if self._kiwoom.connect():
                logger.info("Connected to Kiwoom OpenAPI+")
        except KiwoomUnavailableError as exc:
            logger.warning("Kiwoom unavailable: %s", exc)
            self._fallback = self._init_fallback()

    def fetch_ohlcv(
        self, ticker: str, start: datetime, end: datetime, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with interval support.
        - If Kiwoom is connected, only daily is supported; intraday falls back to daily.
        - If yfinance fallback is active, honors interval (e.g., '60m','1d','1wk','1mo').
        """
        try:
            if interval == "1d" and self._kiwoom.is_connected():
                return self._kiwoom.fetch_daily_ohlcv(ticker, start, end)
        except KiwoomUnavailableError:
            pass

        yf = self._ensure_fallback()
        if yf:
            yf_ticker = yf.Ticker(ticker)
            # Map common intervals
            interval_map = {
                "1h": "60m",
                "1d": "1d",
                "1w": "1wk",
                "1mo": "1mo",
            }
            yf_interval = interval_map.get(interval, interval)
            df = yf_ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False)
            if df.empty and yf_interval != "1d":
                # fallback to daily if interval returns empty
                df = yf_ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            return df

        raise RuntimeError("No market data provider available.")

    def fetch_daily_ohlcv(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        try:
            if self._kiwoom.is_connected():
                return self._kiwoom.fetch_daily_ohlcv(ticker, start, end)
        except KiwoomUnavailableError:
            pass

        self._ensure_fallback()

        if self._fallback:
            ticker = self._fallback.Ticker(ticker)
            df = ticker.history(start=start, end=end, auto_adjust=False)
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            return df

        raise RuntimeError("No market data provider available.")

    def list_candidates(self) -> pd.DataFrame:
        try:
            return self._kiwoom.get_stock_codes()
        except KiwoomUnavailableError:
            pass

        if self._fallback is None:
            self._fallback = self._init_fallback()

        if self._fallback:
            info = {
                "code": ["005930.KS", "000660.KS", "035420.KS"],
                "name": ["삼성전자", "SK하이닉스", "NAVER"],
            }
            return pd.DataFrame(info)

        return pd.DataFrame(columns=["code", "name"])

    @staticmethod
    def _init_fallback():
        try:
            import yfinance as yf
        except ModuleNotFoundError:
            logger.error(
                "Install `yfinance` for the development fallback: `pip install yfinance`"
            )
            return None
        logger.info("Using yfinance fallback provider")
        return yf

    def popular_tickers(self) -> pd.DataFrame:
        """
        Return approx. popular KR tickers with latest price if available.
        """
        candidates = ["005930.KS", "000660.KS", "035420.KS", "051910.KS", "207940.KS",
                      "035720.KS", "005380.KS", "006400.KS", "068270.KS", "028260.KS"]
        yf = self._ensure_fallback()
        data = {"code": [], "name": [], "price": []}
        for code in candidates:
            name = code
            price = None
            try:
                if yf:
                    t = yf.Ticker(code)
                    info = t.history(period="1d")
                    if not info.empty:
                        price = float(info["Close"].iloc[-1])
            except Exception:  # pragma: no cover
                pass
            data["code"].append(code)
            data["name"].append(name)
            data["price"].append(price)
        return pd.DataFrame(data)


