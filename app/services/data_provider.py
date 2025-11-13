"""
Unified entry point for market data.

우선 Kiwoom REST API(mockapi)를 사용하고,
안 될 경우 Kiwoom OpenAPI+ 또는 yfinance로 폴백합니다.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from app.services.kiwoom_client import KiwoomClient, KiwoomUnavailableError
from app.services.kiwoom_rest_client import KiwoomRestClient, KiwoomRestError

logger = logging.getLogger(__name__)


class MarketDataProvider:
    def __init__(self) -> None:
        self._kiwoom = KiwoomClient()
        self._kiwoom_rest: Optional[KiwoomRestClient] = None
        self._fallback = None

    def _ensure_fallback(self):
        if self._fallback is None:
            self._fallback = self._init_fallback()
        return self._fallback

    @staticmethod
    def _normalize_ticker_for_fallback(ticker: str) -> str:
        """
        yfinance 등 해외 티커용 프로바이더에서 사용할 수 있도록
        국내 6자리 종목코드를 적절한 형식으로 변환한다.
        - '005930'  -> '005930.KS'
        - 이미 접미사가 있는 경우 그대로 사용
        """
        if "." in ticker:
            return ticker
        if len(ticker) == 6 and ticker.isdigit():
            return f"{ticker}.KS"
        return ticker

    def connect(self) -> None:
        # 1) Kiwoom REST 우선 시도
        try:
            self._kiwoom_rest = KiwoomRestClient()
            logger.info("Connected to Kiwoom REST API (base=%s)", self._kiwoom_rest.base_url)
        except KiwoomRestError as exc:
            logger.warning("Kiwoom REST unavailable: %s", exc)
            self._kiwoom_rest = None

        # 2) REST 실패 시 Kiwoom OpenAPI+ 시도
        if self._kiwoom_rest is None:
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

        우선순위:
        1) Kiwoom REST: 분봉/일봉 차트 API 사용
        2) Kiwoom OpenAPI+: 일봉만 지원
        3) yfinance fallback
        """
        # 1) Kiwoom REST
        if self._kiwoom_rest is not None:
            try:
                if interval in {"1h", "60m"}:
                    df = self._kiwoom_rest.get_minute_chart(ticker, tic_scope="60")
                    return df[(df.index >= start) & (df.index <= end)]
                elif interval in {"1d", "1w", "1mo"}:
                    base_dt = end.strftime("%Y%m%d")
                    daily = self._kiwoom_rest.get_daily_chart(ticker, base_dt=base_dt)
                    daily = daily[(daily.index >= start) & (daily.index <= end)]
                    if interval == "1d":
                        return daily
                    # 주봉/월봉은 일봉을 리샘플링
                    rule = {"1w": "W", "1mo": "M"}[interval]
                    ohlc = daily[["open", "high", "low", "close"]].resample(rule).agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last"}
                    )
                    vol = daily["volume"].resample(rule).sum()
                    out = ohlc.copy()
                    out["volume"] = vol
                    out = out.dropna(how="any")
                    return out
            except KiwoomRestError as exc:
                logger.warning("Kiwoom REST fetch_ohlcv failed: %s", exc)

        # 2) Kiwoom OpenAPI+
        try:
            if interval == "1d" and self._kiwoom.is_connected():
                return self._kiwoom.fetch_daily_ohlcv(ticker, start, end)
        except KiwoomUnavailableError:
            pass

        # 3) yfinance fallback
        yf = self._ensure_fallback()
        if yf:
            yf_ticker = yf.Ticker(self._normalize_ticker_for_fallback(ticker))
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
        # 1) Kiwoom REST
        if self._kiwoom_rest is not None:
            try:
                base_dt = end.strftime("%Y%m%d")
                df = self._kiwoom_rest.get_daily_chart(ticker, base_dt=base_dt)
                return df[(df.index >= start) & (df.index <= end)]
            except KiwoomRestError as exc:
                logger.warning("Kiwoom REST fetch_daily_ohlcv failed: %s", exc)

        # 2) Kiwoom OpenAPI+
        try:
            if self._kiwoom.is_connected():
                return self._kiwoom.fetch_daily_ohlcv(ticker, start, end)
        except KiwoomUnavailableError:
            pass

        # 3) yfinance fallback
        self._ensure_fallback()

        if self._fallback:
            ticker_obj = self._fallback.Ticker(self._normalize_ticker_for_fallback(ticker))
            df = ticker_obj.history(start=start, end=end, auto_adjust=False)
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
        """
        종목 리스트.
        - Kiwoom REST 사용 시: 코스피 대표 50개로 제한 (KRX 코드 형식)
        - 그 외: 기존 OpenAPI+/yfinance 로직
        """
        if self._kiwoom_rest is not None:
            kospi50 = [
                ("005930", "삼성전자"),
                ("000660", "SK하이닉스"),
                ("035420", "NAVER"),
                ("051910", "LG화학"),
                ("207940", "삼성바이오로직스"),
                ("005380", "현대차"),
                ("068270", "셀트리온"),
                ("028260", "삼성물산"),
                ("012330", "현대모비스"),
                ("055550", "신한지주"),
                ("105560", "KB금융"),
                ("004170", "신세계"),
                ("034730", "SK"),
                ("090430", "아모레퍼시픽"),
                ("017670", "SK텔레콤"),
                ("096770", "SK이노베이션"),
                ("003550", "LG"),
                ("000270", "기아"),
                ("034220", "LG디스플레이"),
                ("066570", "LG전자"),
                ("086790", "하나금융지주"),
                ("015760", "한국전력"),
                ("010130", "고려아연"),
                ("010950", "S-Oil"),
                ("009150", "삼성전기"),
                ("047050", "포스코인터내셔널"),
                ("003670", "포스코퓨처엠"),
                ("036570", "엔씨소프트"),
                ("035720", "카카오"),
                ("251270", "넷마블"),
                ("011170", "롯데케미칼"),
                ("010140", "삼성중공업"),
                ("006400", "삼성SDI"),
                ("001040", "CJ"),
                ("011790", "SKC"),
                ("018260", "삼성에스디에스"),
                ("024110", "기업은행"),
                ("081660", "휠라홀딩스"),
                ("011200", "HMM"),
                ("000810", "삼성화재"),
                ("000100", "유한양행"),
                ("000120", "CJ대한통운"),
                ("003410", "쌍용C&E"),
                ("004020", "현대제철"),
                ("005490", "포스코홀딩스"),
                ("010060", "OCI홀딩스"),
                ("010620", "현대미포조선"),
                ("014680", "한솔케미칼"),
                ("018880", "한온시스템"),
                ("032830", "삼성생명"),
            ]
            return pd.DataFrame(kospi50, columns=["code", "name"])

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
        # REST 사용 시: 코스피 50 전체에 현재가를 붙여서 반환
        if self._kiwoom_rest is not None:
            candidates = self.list_candidates()
            rows = {"code": [], "name": [], "price": []}
            for code, name in zip(candidates["code"], candidates["name"]):
                try:
                    price = self._kiwoom_rest.get_current_price(str(code))
                except KiwoomRestError:
                    price = None
                # 가격을 빠짐없이 채우기 위해 실패 시 yfinance로 재시도
                if price is None:
                    try:
                        yf = self._ensure_fallback()
                        if yf:
                            ticker = yf.Ticker(f"{code}.KS")
                            info = ticker.history(period="1d")
                            if not info.empty:
                                price = float(info["Close"].iloc[-1])
                    except Exception:
                        pass
                # 그래도 없으면 0으로 대체
                if price is None:
                    price = 0.0
                # 가격 포맷팅: ,와 '원' 붙이기
                price_str = f"{price:,.0f}원" if price > 0 else "0원"
                rows["code"].append(code)
                rows["name"].append(name)
                rows["price"].append(price_str)
            return pd.DataFrame(rows)

        # fallback: yfinance (대략적인 인기 종목 10개)
        candidates = [
            "005930.KS",
            "000660.KS",
            "035420.KS",
            "051910.KS",
            "207940.KS",
            "035720.KS",
            "005380.KS",
            "006400.KS",
            "068270.KS",
            "028260.KS",
        ]
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
            # 가격 포맷팅: ,와 '원' 붙이기
            price_str = f"{price:,.0f}원" if price and price > 0 else "0원"
            data["code"].append(code)
            data["name"].append(name)
            data["price"].append(price_str)
        return pd.DataFrame(data)



