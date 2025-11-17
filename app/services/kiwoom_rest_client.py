from __future__ import annotations

"""
Kiwoom REST API 클라이언트 (모의투자 도메인 전용).

- 토큰 발급/폐기
- 분봉/일봉 차트 조회
- 현재가 조회

환경 변수:
- KIWOOM_REST_APPKEY
- KIWOOM_REST_SECRET
- KIWOOM_REST_PAPER (모의투자 사용 여부, "1"이면 mockapi 사용)
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests


class KiwoomRestError(RuntimeError):
    """Kiwoom REST 호출 실패 시 사용되는 예외."""


@dataclass
class _TokenInfo:
    token: str
    token_type: str
    expires_at: float  # epoch seconds


class KiwoomRestClient:
    def __init__(self) -> None:
        self.appkey = os.getenv("KIWOOM_REST_APPKEY", "").strip()
        self.secret = os.getenv("KIWOOM_REST_SECRET", "").strip()
        self.use_paper = os.getenv("KIWOOM_REST_PAPER", "1").strip() in {"1", "true", "True"}

        if not self.appkey or not self.secret:
            raise KiwoomRestError("KIWOOM_REST_APPKEY/KIWOOM_REST_SECRET 환경 변수를 설정하세요.")

        self.base_url = "https://mockapi.kiwoom.com" if self.use_paper else "https://api.kiwoom.com"
        self._token: Optional[_TokenInfo] = None

    # ------------------------------------------------------------------
    # 토큰 관리
    # ------------------------------------------------------------------
    def _issue_token(self) -> _TokenInfo:
        url = f"{self.base_url}/oauth2/token"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "api-id": "au10001",
        }
        body = {
            "grant_type": "client_credentials",
            "appkey": self.appkey,
            "secretkey": self.secret,
        }
        resp = requests.post(url, json=body, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise KiwoomRestError(f"토큰 발급 실패: HTTP {resp.status_code} {resp.text}")
        data = resp.json()
        if data.get("return_code", 0) not in (0, "0"):
            raise KiwoomRestError(f"토큰 발급 실패: {data}")

        expires_dt = data.get("expires_dt")
        # 만료일자 형식: YYYYMMDDhhmmss
        expires_at = time.time()
        if expires_dt and len(expires_dt) == 14:
            dt = datetime.strptime(expires_dt, "%Y%m%d%H%M%S")
            expires_at = dt.timestamp()
        # 약간의 여유 시간을 둔다.
        expires_at -= 60

        token_info = _TokenInfo(
            token=data["token"],
            token_type=data.get("token_type", "bearer"),
            expires_at=expires_at,
        )
        self._token = token_info
        return token_info

    def _ensure_token(self) -> _TokenInfo:
        if self._token is None or time.time() >= self._token.expires_at:
            return self._issue_token()
        return self._token

    def revoke_token(self) -> None:
        """명시적으로 토큰 폐기."""
        if not self._token:
            return
        url = f"{self.base_url}/oauth2/revoke"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "api-id": "au10002",
            "authorization": f"{self._token.token_type.capitalize()} {self._token.token}",
        }
        body = {
            "appkey": self.appkey,
            "secretkey": self.secret,
            "token": self._token.token,
        }
        requests.post(url, json=body, headers=headers, timeout=10)
        self._token = None

    # ------------------------------------------------------------------
    # 공통 요청 헬퍼
    # ------------------------------------------------------------------
    def _auth_headers(self, api_id: str) -> Dict[str, str]:
        token_info = self._ensure_token()
        return {
            "Content-Type": "application/json;charset=UTF-8",
            "api-id": api_id,
            "authorization": f"{token_info.token_type.capitalize()} {token_info.token}",
        }

    def _post(self, path: str, api_id: str, body: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers(api_id)
        resp = requests.post(url, json=body, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise KiwoomRestError(f"요청 실패: {path} HTTP {resp.status_code} {resp.text}")
        data = resp.json()
        # 일부 API는 return_code 대신 returnCode를 사용
        if data.get("return_code", 0) not in (0, "0", None) and data.get("returnCode", 0) not in (
            0,
            "0",
            None,
        ):
            raise KiwoomRestError(f"요청 실패: {path} {data}")
        return data

    # ------------------------------------------------------------------
    # 차트 / 시세 API 래퍼
    # ------------------------------------------------------------------
    def get_minute_chart(
        self,
        stk_cd: str,
        tic_scope: str = "1",
        upd_stkpc_tp: str = "1",
    ) -> pd.DataFrame:
        """
        주식분봉차트조회요청(ka10080) → /api/dostk/chart
        """
        body = {
            "stk_cd": stk_cd,
            "tic_scope": tic_scope,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        data = self._post("/api/dostk/chart", api_id="ka10080", body=body)
        rows: List[Dict] = data.get("stk_min_pole_chart_qry", [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        # cntr_tm: YYYYMMDDhhmmss
        df["datetime"] = pd.to_datetime(df["cntr_tm"], format="%Y%m%d%H%M%S")
        df = df.rename(
            columns={
                "open_pric": "open",
                "high_pric": "high",
                "low_pric": "low",
                "cur_prc": "close",
                "trde_qty": "volume",
            }
        )
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("datetime").set_index("datetime")
        # 문자열 가격을 숫자로 변환(+, - 기호 제거)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("[^0-9.]", "", regex=True),
                errors="coerce",
            )
        return df

    def get_daily_chart(
        self,
        stk_cd: str,
        base_dt: Optional[str] = None,
        upd_stkpc_tp: str = "1",
    ) -> pd.DataFrame:
        """
        주식일봉차트조회요청(ka10081) → /api/dostk/chart
        base_dt가 None이면 오늘 날짜 사용.
        """
        if base_dt is None:
            base_dt = datetime.now().strftime("%Y%m%d")
        body = {
            "stk_cd": stk_cd,
            "base_dt": base_dt,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        data = self._post("/api/dostk/chart", api_id="ka10081", body=body)
        rows: List[Dict] = data.get("stk_dt_pole_chart_qry", [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["dt"], format="%Y%m%d")
        df = df.rename(
            columns={
                "open_pric": "open",
                "high_pric": "high",
                "low_pric": "low",
                "cur_prc": "close",
                "trde_qty": "volume",
            }
        )
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("[^0-9.]", "", regex=True),
                errors="coerce",
            )
        return df

    def get_current_price(self, stk_cd: str) -> float:
        """
        간단한 현재가 조회: 시세표성정보요청(ka10007) 사용.
        """
        body = {"stk_cd": stk_cd}
        data = self._post("/api/dostk/mrkcond", api_id="ka10007", body=body)
        price_str = str(data.get("cur_prc", "0"))
        price = pd.to_numeric(price_str.replace("+", "").replace("-", ""), errors="coerce")
        return float(price) if pd.notna(price) else 0.0




