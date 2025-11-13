from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app.components.charts import price_chart
from app.config import get_default_symbol, get_openai_key
from app.services.chatbot import TradingAssistantChatbot
from app.services.data_provider import MarketDataProvider
from app.utils.calculators import (
    ProfitLossInput,
    apply_fx,
    calculate_average_price,
    calculate_profit_loss,
)

logging.basicConfig(level=logging.INFO)


def init_session_state() -> None:
    """세션 상태 초기화: 데이터 제공자, 모델, 포트폴리오 등을 초기화"""
    # 데이터 제공자 초기화 (Kiwoom REST API 또는 fallback)
    if "data_provider" not in st.session_state:
        st.session_state["data_provider"] = MarketDataProvider()
        st.session_state["data_provider"].connect()

    # RL 모델 및 백테스트 결과 초기화
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("backtest", None)
    
    # 기본 종목을 삼성전자(005930)로 설정
    st.session_state.setdefault("selected_symbol", get_default_symbol())
    
    # 챗봇 대화 기록 초기화
    st.session_state.setdefault("chat_history", [])
    
    # 모의투자 포트폴리오 초기화 (초기 자본금 3천만원)
    st.session_state.setdefault("portfolio", {"cash": 30_000_000.0, "positions": {}})  # KRW
    
    # 주문 정보 초기화
    st.session_state.setdefault("orders", {"shares": 0, "amount": 0.0})
    
    # 자동매매 상태 및 로그 초기화
    st.session_state.setdefault("auto_trading_active", False)
    st.session_state.setdefault("auto_trading_logs", [])
    st.session_state.setdefault("trade_logs", [])


def fetch_data(provider: MarketDataProvider, symbol: str, period_days: int = 365) -> pd.DataFrame:
    """지정된 기간의 일봉 데이터를 가져옵니다"""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    return provider.fetch_daily_ohlcv(symbol, start, end)

def fetch_data_with_interval(provider: MarketDataProvider, symbol: str, timeframe: str) -> pd.DataFrame:
    """선택한 기간에 맞는 OHLCV 데이터를 가져옵니다"""
    end = datetime.now()
    if timeframe == "1시간":
        # 정확히 1시간 전부터 지금까지 (단기 분봉/틱 기반)
        start = end - timedelta(hours=1)
        return provider.fetch_ohlcv(symbol, start, end, interval="1h")
    if timeframe == "1일":
        # 1일 전 ~ 지금
        start = end - timedelta(days=1)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    if timeframe == "1주":
        # 1주(7일) 전 ~ 지금
        start = end - timedelta(weeks=1)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    if timeframe == "1달":
        # 1달 ≒ 30일 전 ~ 지금
        start = end - timedelta(days=30)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    if timeframe == "1년":
        # 1년(365일) 전 ~ 지금
        start = end - timedelta(days=365)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    # 전체: 10년 가량
    start = end - timedelta(days=365 * 10)
    return provider.fetch_ohlcv(symbol, start, end, interval="1d")

def _current_price(df: pd.DataFrame) -> float:
    """DataFrame에서 가장 최근 종가를 반환합니다"""
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])

def _code_to_name_map(provider: MarketDataProvider) -> dict[str, str]:
    """종목 코드를 종목명으로 매핑하는 딕셔너리를 반환합니다"""
    df = provider.list_candidates()
    if df.empty:
        return {}
    return {c: n for c, n in zip(df["code"], df["name"])}

def _name_to_code(provider: MarketDataProvider, name: str) -> str | None:
    """종목명을 종목 코드로 변환합니다 (정확 일치 우선, 없으면 부분 일치)"""
    df = provider.list_candidates()
    if df.empty:
        return None
    # 우선 정확히 일치하는 종목명 찾기
    exact = df[df["name"] == name]
    if not exact.empty:
        return str(exact["code"].iloc[0])
    # 정확 일치가 없으면 부분 일치 검색
    partial = df[df["name"].str.contains(name, na=False)]
    if not partial.empty:
        return str(partial["code"].iloc[0])
    return None

def _place_order(symbol: str, price: float, shares: int = 0, amount: float = 0.0) -> None:
    """모의투자 주문을 실행합니다 (매수만 지원)"""
    portfolio = st.session_state["portfolio"]
    cash: float = portfolio["cash"]
    positions: dict = portfolio["positions"]
    
    # 수량이 0이면 금액으로 계산
    executed_shares = shares
    if executed_shares == 0 and amount > 0 and price > 0:
        executed_shares = int(amount // price)
    if executed_shares == 0:
        return
    
    cost = executed_shares * price
    if cost > 0:
        # 매수 주문 처리
        if cash < cost:
            st.warning("예수금이 부족합니다.")
            return
        # 현금 차감 및 포지션 업데이트
        portfolio["cash"] = cash - cost
        pos = positions.get(symbol, {"shares": 0, "avg_price": 0.0})
        new_shares = pos["shares"] + executed_shares
        new_cost = pos["avg_price"] * pos["shares"] + cost
        pos["shares"] = new_shares
        pos["avg_price"] = new_cost / new_shares  # 평균 매입가 계산
        positions[symbol] = pos
        # 매수 로그 기록
        logs: list[str] = st.session_state.get("trade_logs", [])
        logs.append(f"[매수] {symbol} {executed_shares}주 @ {price:,.0f}")
        st.session_state["trade_logs"] = logs[-200:]
    else:
        # 매도는 UI에서 별도로 처리
        pass


def _run_auto_trading_step(provider: MarketDataProvider, symbol: str) -> None:
    """
    학습된 알고리즘(RL 모델) 또는 이동평균 기반 자동매매 알고리즘 한 스텝.
    - RL 모델이 있으면 사용, 없으면 이동평균 기반
    - 실제 포트폴리오(st.session_state['portfolio'])를 사용
    - 수행 결과를 auto_trading_logs에 남김
    """
    df = fetch_data_with_interval(provider, symbol, "1일")
    if df.empty or len(df) < 30:
        return

    price = float(df["close"].iloc[-1])
    portfolio = st.session_state["portfolio"]
    positions: dict = portfolio["positions"]
    pos = positions.get(symbol, {"shares": 0, "avg_price": 0.0})
    holding_shares = pos["shares"]

    logs_auto: list[str] = st.session_state.get("auto_trading_logs", [])
    logs_trade: list[str] = st.session_state.get("trade_logs", [])

    # 학습된 RL 모델이 있으면 사용
    model = st.session_state.get("model")
    if model is not None:
        try:
            from app.services.rl_trading import TradingEnv
            
            # 최근 30일 데이터로 환경 생성
            recent_df = df.tail(30)
            env = TradingEnv(recent_df)
            obs, _ = env.reset()
            
            # 모델로 액션 예측
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # ACTION_BUY = 1, ACTION_SELL = 2
            if action == 1 and holding_shares == 0 and portfolio["cash"] > 0 and price > 0:
                # 매수
                buy_amount = portfolio["cash"] * 0.1
                buy_shares = int(buy_amount // price)
                if buy_shares > 0:
                    _place_order(symbol, price, shares=buy_shares)
                    msg = f"[매수-RL] {symbol} {buy_shares}주 @ {price:,.0f}"
                    logs_auto.append(msg)
                    logs_trade.append(msg)
            elif action == 2 and holding_shares > 0:
                # 매도
                sell_shares = holding_shares
                proceeds = sell_shares * price
                portfolio["cash"] += proceeds
                pos["shares"] = 0
                positions.pop(symbol, None)
                msg = f"[매도-RL] {symbol} {sell_shares}주 @ {price:,.0f}"
                logs_auto.append(msg)
                logs_trade.append(msg)
            # action == 0 (hold)는 아무것도 안 함
            
            st.session_state["auto_trading_logs"] = logs_auto[-100:]
            st.session_state["trade_logs"] = logs_trade[-200:]
            return
        except Exception as exc:
            # RL 모델 실행 실패 시 이동평균으로 폴백
            pass

    # 폴백: 이동평균 기반 알고리즘
    short = df["close"].rolling(window=5).mean().iloc[-1]
    long = df["close"].rolling(window=20).mean().iloc[-1]

    # 매수 조건: 단기 > 장기, 보유 없음
    if short > long and holding_shares == 0 and portfolio["cash"] > 0 and price > 0:
        buy_amount = portfolio["cash"] * 0.1
        buy_shares = int(buy_amount // price)
        if buy_shares > 0:
            _place_order(symbol, price, shares=buy_shares)
            msg = f"[매수-MA] {symbol} {buy_shares}주 @ {price:,.0f} (단기MA {short:,.0f} > 장기MA {long:,.0f})"
            logs_auto.append(msg)
            logs_trade.append(msg)

    # 매도 조건: 단기 < 장기, 보유 있음
    elif short < long and holding_shares > 0:
        sell_shares = holding_shares
        proceeds = sell_shares * price
        portfolio["cash"] += proceeds
        pos["shares"] = 0
        positions.pop(symbol, None)
        msg = f"[매도-MA] {symbol} {sell_shares}주 @ {price:,.0f} (단기MA {short:,.0f} < 장기MA {long:,.0f})"
        logs_auto.append(msg)
        logs_trade.append(msg)

    # 로그 길이 제한 (최근 100개만 유지)
    st.session_state["auto_trading_logs"] = logs_auto[-100:]
    st.session_state["trade_logs"] = logs_trade[-200:]

def dashboard_page() -> None:
    """대시보드 페이지: 주식 차트, 주문, 보유 종목, 거래 로그 표시"""
    st.subheader("모의 주식 대시보드")
    provider: MarketDataProvider = st.session_state["data_provider"]

    # 종목 코드를 종목명으로 변환 (기본값: 삼성전자)
    default_symbol = st.session_state["selected_symbol"]
    code_to_name = _code_to_name_map(provider)
    # 기본값이 없거나 매핑이 안 되면 "삼성전자"로 설정
    selected_name = code_to_name.get(default_symbol, "삼성전자")

    left, right = st.columns([3, 2])
    with left:
        # 종목 검색 (Enter로 검색)
        with st.form(key="dashboard_search_form", clear_on_submit=False):
            col_s1, col_s2 = st.columns([5, 1])
            with col_s1:
                search_name = st.text_input("종목명", value=selected_name, label_visibility="collapsed")
            with col_s2:
                search_submitted = st.form_submit_button("검색", use_container_width=True)
            if search_submitted or (search_name != selected_name and search_name):
                resolved = _name_to_code(provider, search_name)
                if resolved:
                    st.session_state["selected_symbol"] = resolved
                    st.rerun()
                elif search_name:
                    st.warning("해당 종목명을 찾지 못했습니다.")

        timeframe = st.radio(
            "기간", ["1시간", "1일", "1주", "1달", "1년", "전체"], horizontal=True, index=1
        )

        symbol = st.session_state["selected_symbol"]
        data_load_state = st.text("데이터 불러오는 중...")
        data = fetch_data_with_interval(provider, symbol, timeframe)
        data_load_state.text("데이터 로드 완료 ✅")

        # 차트는 전체 너비 사용
        price_chart(data, "")

    with right:
        st.markdown("#### 주문")
        sel_symbol = st.session_state["selected_symbol"]
        current = _current_price(fetch_data_with_interval(provider, sel_symbol, "1일"))
        st.metric("현재가", f"{current:,.0f} KRW")
        # 주문 수량 위로 여백 추가 (총 3줄)
        st.markdown("")
        st.markdown("")
        st.markdown("")
        if "orders" not in st.session_state:
            st.session_state["orders"] = {"shares": 0, "amount": 0.0}
        # 주문 수량을 금액으로 주문 위로 배치
        shares = st.number_input(
            "주문 수량(주)",
            min_value=0,
            step=1,
            value=int(st.session_state["orders"]["shares"]),
        )
        st.session_state["orders"]["shares"] = shares
        amount = st.number_input(
            "금액으로 주문 (KRW)",
            min_value=0.0,
            step=1000.0,
            value=float(st.session_state["orders"]["amount"]),
        )
        st.session_state["orders"]["amount"] = amount
        est_total = shares * current if shares > 0 else amount
        st.write(f"총 주문금액: {est_total:,.0f} KRW")
        b1, b2 = st.columns(2)
        if b1.button("매수"):
            _place_order(st.session_state["selected_symbol"], current, shares=shares, amount=amount)
            st.success("매수 주문이 반영되었습니다.")
        if b2.button("매도"):
            # simple sell: sell all or by shares
            portfolio = st.session_state["portfolio"]
            positions = portfolio["positions"]
            pos = positions.get(st.session_state["selected_symbol"])
            if not pos or pos["shares"] <= 0:
                st.warning("보유 수량이 없습니다.")
            else:
                sell_shares = shares if shares > 0 else pos["shares"]
                sell_shares = min(sell_shares, pos["shares"])
                proceeds = sell_shares * current
                portfolio["cash"] += proceeds
                pos["shares"] -= sell_shares
                if pos["shares"] == 0:
                    positions.pop(st.session_state["selected_symbol"], None)
                else:
                    positions[st.session_state["selected_symbol"]] = pos
                st.success("매도 주문이 반영되었습니다.")
                # 매도 로그 기록
                logs: list[str] = st.session_state.get("trade_logs", [])
                logs.append(f"[매도] {st.session_state['selected_symbol']} {sell_shares}주 @ {current:,.0f}")
                st.session_state["trade_logs"] = logs[-200:]
    st.divider()
    # 좌측: 보유 종목(넓게), 우측: 거래 로그(좁게)
    col_hold, col_log = st.columns([3, 1])
    with col_hold:
        st.markdown("#### 보유 종목")
        rows = []
        for sym, pos in st.session_state["portfolio"]["positions"].items():
            df_sym = fetch_data_with_interval(provider, sym, "1일")
            cur = _current_price(df_sym)
            eval_amt = cur * pos["shares"]
            pnl = (cur - pos["avg_price"]) * pos["shares"]
            pnl_rate = (cur / pos["avg_price"] - 1.0) * 100.0 if pos["avg_price"] > 0 else 0.0
            rows.append(
                {
                    "종목명": code_to_name.get(sym, sym),
                    "매입가": f"{pos['avg_price']:,.0f}",
                    "현재가": f"{cur:,.0f}",
                    "평가손익": f"{pnl:,.0f}",
                    "손익률": f"{pnl_rate:.2f}%",
                    "잔고수량": pos["shares"],
                    "평가금액": f"{eval_amt:,.0f}",
                }
            )

        st.dataframe(pd.DataFrame(rows))
    with col_log:
        st.markdown("#### 거래 로그")
        logs = st.session_state.get("trade_logs", [])
        if logs:
            for line in reversed(logs[-50:]):
                st.write(line)
        else:
            st.info("아직 거래 로그가 없습니다.")



def data_page() -> None:
    """자동매매 페이지: 차트, 자동매매 시스템, 코스피 50 종목 목록 표시"""
    st.subheader("자동매매")
    provider: MarketDataProvider = st.session_state["data_provider"]
    left, right = st.columns([3, 2])
    with left:
        # 종목 코드를 종목명으로 변환 (기본값: 삼성전자)
        code_to_name = _code_to_name_map(provider)
        current_code = st.session_state["selected_symbol"]
        current_name = code_to_name.get(current_code, "삼성전자")

        # 종목명 검색 (Enter로 검색)
        with st.form(key="data_page_search_form", clear_on_submit=False):
            col_name, col_btn = st.columns([5, 1])
            with col_name:
                name_input = st.text_input("종목명", value=current_name, label_visibility="collapsed")
            with col_btn:
                search_submitted = st.form_submit_button("검색", use_container_width=True)
            symbol = current_code
            if search_submitted or (name_input != current_name and name_input):
                resolved = _name_to_code(provider, name_input)
                if resolved:
                    st.session_state["selected_symbol"] = resolved
                    symbol = resolved
                    st.rerun()
                elif name_input:
                    st.warning("해당 종목명을 찾지 못했습니다.")

        timeframe = st.radio("기간", ["1시간", "1일", "1주", "1달", "1년", "전체"], horizontal=True, index=1)
        df = fetch_data_with_interval(provider, symbol, timeframe)
        title_name = code_to_name.get(symbol, symbol)
        price_chart(df, f"{title_name} 차트")
    with right:
        st.markdown("#### 자동매매 시스템")
        model = st.session_state.get("model")
        if model is not None:
            st.write("학습된 RL 알고리즘이 포트폴리오를 자동으로 매수/매도합니다.")
        else:
            st.write("이동평균 기반의 자동매매 알고리즘이 포트폴리오를 자동으로 매수/매도합니다.")

        symbol = st.session_state["selected_symbol"]

        col_start, col_stop = st.columns(2)
        with col_start:
            start_clicked = st.button("시작")
        with col_stop:
            stop_clicked = st.button("정지")

        # 시작/정지 상태 전환
        if start_clicked:
            st.session_state["auto_trading_active"] = True
            st.session_state["auto_trading_logs"].append("[시스템] 자동매매 시작")
        if stop_clicked:
            st.session_state["auto_trading_active"] = False
            st.session_state["auto_trading_logs"].append("[시스템] 자동매매 정지")

        active = st.session_state.get("auto_trading_active", False)
        st.metric("자동매매 상태", "동작 중" if active else "정지")

        # 자동매매가 활성화된 경우 한 스텝 실행
        if active:
            try:
                _run_auto_trading_step(provider, symbol)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"자동매매 실행 중 오류가 발생했습니다: {exc}")

        st.markdown("#### 매매 로그")
        logs = st.session_state.get("auto_trading_logs", [])
        if logs:
            # 최근 로그가 위로 오도록 역순 표시
            for line in reversed(logs[-50:]):
                st.write(line)
        else:
            st.info("아직 매매 로그가 없습니다.")

    # 코스피 50 표는 화면 전체 폭을 사용
    st.markdown("#### 코스피 50")
    kospi50 = provider.popular_tickers()
    st.dataframe(kospi50.reset_index(drop=True), use_container_width=True)


def chatbot_page() -> None:
    """챗봇 페이지: 주식/투자 관련 질문에 답변하는 AI 챗봇"""
    st.subheader("😁무엇이든 물어보세요.")

    # OpenAI API 키 확인
    api_key = get_openai_key()
    if not api_key:
        st.warning("OPENAI_API_KEY 또는 환경설정이 없어 챗봇을 사용할 수 없습니다.")
        return

    # 챗봇 인스턴스 초기화 (세션에 없으면 생성)
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = TradingAssistantChatbot(api_key=api_key)

    # 대화 기록 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    bot: TradingAssistantChatbot = st.session_state["chatbot"]

    # 이전 대화 표시
    for role, text in st.session_state["chat_history"]:
        if role == "user":
            st.markdown(f"**🙋‍♂️ 사용자:** {text}")
        else:
            st.markdown(f"**🤖 챗봇:** {text}")

    # Enter 키로도 전송되도록 form 사용
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input("질문", placeholder="질문을 입력하세요... (Enter로 전송)", label_visibility="collapsed", key="chat_input")
        submitted = st.form_submit_button("보내기", use_container_width=True)

    # form 제출 시 (Enter 또는 버튼 클릭)
    if submitted:
        if prompt and prompt.strip():
            st.session_state["chat_history"].append(("user", prompt))
            try:
                answer = bot.ask(prompt)
                st.session_state["chat_history"].append(("assistant", answer))
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"챗봇 호출 중 오류가 발생했습니다: {exc}")

    col_t1, col_t2, col_t3 = st.columns(3)

    template_prompts = {
        "recommend": "최근 시장 상황과 변동성을 고려해 한국 주식 종목 3~5개를 추천해줘. 각 종목에 대한 간단한 투자 포인트도 한국어로 설명해줘.",
        "market": "오늘 한국 증시(KOSPI/KOSDAQ)의 전반적인 시장 동향을 요약해서 설명해줘. 지수 흐름, 업종별 특징, 투자자별 수급 등 핵심 포인트 위주로 알려줘.",
        "news": "오늘 기준으로 주식 시장에 영향을 줄 만한 주요 뉴스나 이슈를 정리해줘. 가능하면 한국 시장과 관련된 내용을 중심으로 알려줘.",
    }

    if col_t1.button("📈 주식 종목 추천", key="template_recommend", type="secondary"):
        user_msg = "주식 종목 추천을 해줘."
        st.session_state["chat_history"].append(("user", user_msg))
        try:
            answer = bot.ask(template_prompts["recommend"])
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"챗봇 호출 중 오류가 발생했습니다: {exc}")
            return
        st.session_state["chat_history"].append(("assistant", answer))

    if col_t2.button("📊 시장 동향", key="template_market", type="secondary"):
        user_msg = "오늘 시장 동향을 알려줘."
        st.session_state["chat_history"].append(("user", user_msg))
        try:
            answer = bot.ask(template_prompts["market"])
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"챗봇 호출 중 오류가 발생했습니다: {exc}")
            return
        st.session_state["chat_history"].append(("assistant", answer))

    if col_t3.button("📰 뉴스 요약", key="template_news", type="secondary"):
        user_msg = "오늘 시장 관련 주요 뉴스를 요약해줘."
        st.session_state["chat_history"].append(("user", user_msg))
        try:
            answer = bot.ask(template_prompts["news"])
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"챗봇 호출 중 오류가 발생했습니다: {exc}")
            return
        st.session_state["chat_history"].append(("assistant", answer))

    st.markdown("---")
    st.caption(
        "※ 본 챗봇의 답변은 참고용 정보일 뿐이며, 실제 시장 상황과 다르거나 부정확한 정보를 포함할 수 있습니다. "
        "모든 투자 결정과 책임은 사용자 본인에게 있습니다."
    )

def calculator_page() -> None:
    """계산기 페이지: 손익 계산, 평균 단가 계산, 환율 계산"""
    st.subheader("주식 계산기")
    st.markdown("#### 손익 계산")
    col1, col2, col3 = st.columns(3)
    buy_price = col1.number_input("매수가", min_value=0.0, value=100_000.0, step=1_000.0)
    sell_price = col2.number_input("매도가", min_value=0.0, value=105_000.0, step=1_000.0)
    quantity = col3.number_input("수량", min_value=1, value=10, step=1)

    result = calculate_profit_loss(
        ProfitLossInput(
            buy_price=buy_price,
            sell_price=sell_price,
            quantity=quantity,
        )
    )
    st.write(
        f"순이익: {result['net_profit']:,.0f} KRW / 수익률: {result['profit_rate'] * 100:.2f}%"
    )

    st.divider()

    st.markdown("#### 평단가 계산")
    total_cost = st.number_input("총 매수금액", min_value=0.0, value=1_000_000.0, step=10_000.0)
    total_shares = st.number_input("총 수량", min_value=0, value=100, step=1)
    avg_price = calculate_average_price(total_cost, total_shares)
    st.write(f"평단가: {avg_price:,.2f} KRW")

    st.divider()

    st.markdown("#### 환율 계산")
    st.caption("각 국가 통화 간 단순 환전 계산입니다. (yfinance 환율 기준)")

    countries = ["대한민국", "미국", "중국", "독일", "일본", "인도", "영국", "프랑스", "이탈리아", "캐나다", "브라질"]
    code_map = {
        "대한민국": None,
        "미국": "USDKRW=X",
        "중국": "CNYKRW=X",
        "독일": "EURKRW=X",
        "일본": "JPYKRW=X",
        "인도": "INRKRW=X",
        "영국": "GBPKRW=X",
        "프랑스": "EURKRW=X",
        "이탈리아": "EURKRW=X",
        "캐나다": "CADKRW=X",
        "브라질": "BRLKRW=X",
    }
    symbol_map = {
        "대한민국": "KRW",
        "미국": "USD",
        "중국": "CNY",
        "독일": "EUR",
        "일본": "JPY",
        "인도": "INR",
        "영국": "GBP",
        "프랑스": "EUR",
        "이탈리아": "EUR",
        "캐나다": "CAD",
        "브라질": "BRL",
    }

    # 1번째 줄: 왼쪽 국가 선택, 오른쪽 금액 입력 (보유 통화)
    row1_col_country, row1_col_amount = st.columns(2)
    from_country = row1_col_country.selectbox("보유 통화 국가", countries, index=0)
    from_amount_str = row1_col_amount.text_input("보유 금액", value="1000")
    try:
        from_amount = float(from_amount_str.replace(",", "")) if from_amount_str else 0.0
    except ValueError:
        from_amount = 0.0

    # 2번째 줄: 가운데 '=' 표시
    st.markdown("<div style='text-align:center; font-size: 24px;'>=</div>", unsafe_allow_html=True)

    # 3번째 줄: 왼쪽 국가 선택, 오른쪽 금액 결과 (변환 통화)
    row3_col_country, row3_col_amount = st.columns(2)
    to_country = row3_col_country.selectbox("변환 통화 국가", countries, index=1)

    # 각 국가별 환율 (KRW 기준) 조회
    try:
        import yfinance as yf  # type: ignore
        # 대한민국(KRW)일 경우 환율 1.0로 간주
        if from_country == "대한민국":
            rate_from = 1.0
        else:
            t_from = yf.Ticker(code_map[from_country])
            fx_from = t_from.history(period="1d")
            rate_from = float(fx_from["Close"].iloc[-1]) if not fx_from.empty else 1300.0

        if to_country == "대한민국":
            rate_to = 1.0
        else:
            t_to = yf.Ticker(code_map[to_country])
            fx_to = t_to.history(period="1d")
            rate_to = float(fx_to["Close"].iloc[-1]) if not fx_to.empty else 1300.0
    except Exception:
        rate_from = rate_to = 1300.0

    # 환전: 보유통화 -> KRW -> 변환통화
    converted_amount = 0.0
    if rate_from > 0 and rate_to > 0:
        krw_value = from_amount * rate_from
        converted_amount = krw_value / rate_to

    row3_col_amount.text_input(
        "변환 금액",
        value=f"{converted_amount:,.3f} {symbol_map.get(to_country, '')}",
        disabled=True,
    )


def main() -> None:
    """메인 함수: Streamlit 앱 초기화 및 페이지 라우팅"""
    st.set_page_config(page_title="Kiwoom 자동 매매 도우미", layout="wide")
    # 전역 스타일: 배경색 및 포인트 컬러 설정
    st.markdown(
        """
        <style>
        body {
            background-color: #F6F6F6;
        }
        .stApp {
            background-color: #F6F6F6;
        }
        /* 기본 버튼 (본문) */
        .stButton>button {
            background-color: #07C693;
            color: white;
            border-radius: 4px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #06a57e;
            color: white;
        }
        /* 사이드바의 워렌 증권 네비게이션 버튼 */
        section[data-testid="stSidebar"] .stButton>button {
            background-color: #F6F6F6;
            color: #000000;
            border-radius: 4px;
            border: 1px solid #F6F6F6;
            font-weight: 900;
            font-size: 1.1rem;
        }
        section[data-testid="stSidebar"] .stButton>button:hover {
            background-color: #e0e0e0;
            color: #000000;
        }
        /* 사이드바 메뉴 스타일 */
        section[data-testid="stSidebar"] {
            background-color: #F6F6F6;
        }
        section[data-testid="stSidebar"] label {
            font-size: 1.05rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            display: block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            cursor: pointer;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background-color: #e0e0e0;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #07C693;
        }
        /* 챗봇 템플릿 버튼 (흰 배경) */
        .stButton button[kind="secondary"] {
            background-color: white;
            color: #000000;
            border: 1px solid #cccccc;
        }
        .stButton button[kind="secondary"]:hover {
            background-color: #f0f0f0;
            color: #000000;
        }
        /* 챗봇 보내기 버튼 (07C693 색) */
        form[data-testid*="chat_form"] .stButton>button,
        div[data-testid*="chat_form"] .stButton>button {
            background-color: #07C693;
            color: white;
        }
        form[data-testid*="chat_form"] .stButton>button:hover,
        div[data-testid*="chat_form"] .stButton>button:hover {
            background-color: #06a57e;
            color: white;
        }
        /* 검색, 매수, 매도, 시작, 정지, 보내기 버튼 모두 07C693 색으로 강제 적용 */
        /* 사이드바 버튼과 secondary 버튼을 제외한 모든 버튼 */
        button[data-baseweb="button"]:not([kind="secondary"]) {
            background-color: #07C693 !important;
            color: white !important;
            border: none !important;
        }
        button[data-baseweb="button"]:not([kind="secondary"]):hover {
            background-color: #06a57e !important;
            color: white !important;
        }
        /* Streamlit 버튼 컨테이너 내부 버튼 */
        .stButton>button:not([kind="secondary"]) {
            background-color: #07C693 !important;
            color: white !important;
            border: none !important;
        }
        .stButton>button:not([kind="secondary"]):hover {
            background-color: #06a57e !important;
            color: white !important;
        }
        /* form 내부 버튼도 07C693 색 적용 */
        form .stButton>button:not([kind="secondary"]),
        div[data-testid*="form"] .stButton>button:not([kind="secondary"]) {
            background-color: #07C693 !important;
            color: white !important;
        }
        form .stButton>button:not([kind="secondary"]):hover,
        div[data-testid*="form"] .stButton>button:not([kind="secondary"]):hover {
            background-color: #06a57e !important;
            color: white !important;
        }
        /* 검색 버튼 숨기기 (Enter로만 검색) */
        form[data-testid*="dashboard_search_form"] .stButton>button,
        form[data-testid*="data_page_search_form"] .stButton>button {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # 세션 상태 초기화
    init_session_state()

    # 사이드바 제목
    st.sidebar.title("💰워렌 증권")

    # 페이지 상태 관리 (라디오 대신 버튼으로 네비게이션)
    if "page" not in st.session_state:
        st.session_state["page"] = "대시보드"

    # 네비게이션 버튼 생성
    nav_labels = ["대시보드", "자동매매", "챗봇", "계산기"]
    for label in nav_labels:
        if st.sidebar.button(label, use_container_width=True, key=f"nav_{label}"):
            st.session_state["page"] = label

    page = st.session_state["page"]

    # 선택된 페이지에 따라 메인 컨텐츠 렌더링
    if page == "대시보드":
        dashboard_page()
    elif page == "자동매매":
        data_page()
    elif page == "챗봇":
        chatbot_page()
    else:
        calculator_page()

    # 사이드바 하단에 모의투자 계좌 요약 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 모의투자 계좌 요약")
    pf = st.session_state["portfolio"]
    provider: MarketDataProvider = st.session_state["data_provider"]
    
    # 순자산 계산: 현금 + 보유 종목 평가금액
    total_value = pf["cash"]
    unrealized = 0.0  # 평가손익
    for sym, pos in pf["positions"].items():
        df_sym = fetch_data_with_interval(provider, sym, "1일")
        price_sym = _current_price(df_sym)
        total_value += pos["shares"] * price_sym
        unrealized += (price_sym - pos["avg_price"]) * pos["shares"]
    
    # 수익률 계산
    invested = total_value - pf["cash"]
    pnl_rate = (unrealized / invested * 100.0) if invested > 0 else 0.0

    # 계좌 요약 정보 표시
    st.sidebar.markdown(f"<p style='font-size:13px;'>순자산: <b>{total_value:,.0f} 원</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='font-size:13px;'>예수금: <b>{pf['cash']:,.0f} 원</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='font-size:13px;'>평가금: <b>{(total_value - pf['cash']):,.0f} 원</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='font-size:13px;'>손익: <b>{unrealized:,.0f} 원</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='font-size:13px;'>수익률: <b>{pnl_rate:.2f}%</b></p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


