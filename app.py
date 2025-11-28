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
    if "data_provider" not in st.session_state:
        st.session_state["data_provider"] = MarketDataProvider()
        st.session_state["data_provider"].connect()

    st.session_state.setdefault("model", None)
    st.session_state.setdefault("backtest", None)
    st.session_state.setdefault("selected_symbol", get_default_symbol())
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("portfolio", {"cash": 10_000_000.0, "positions": {}})  # KRW
    st.session_state.setdefault("orders", {"shares": 0, "amount": 0.0})


def fetch_data(provider: MarketDataProvider, symbol: str, period_days: int = 365) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=period_days)
    return provider.fetch_daily_ohlcv(symbol, start, end)

def fetch_data_with_interval(provider: MarketDataProvider, symbol: str, timeframe: str) -> pd.DataFrame:
    end = datetime.now()
    if timeframe == "1시간":
        start = end - timedelta(days=14)
        return provider.fetch_ohlcv(symbol, start, end, interval="1h")
    if timeframe == "1일":
        start = end - timedelta(days=365)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    if timeframe == "1주":
        start = end - timedelta(days=365 * 2)
        return provider.fetch_ohlcv(symbol, start, end, interval="1w")
    if timeframe == "1달":
        start = end - timedelta(days=365 * 3)
        return provider.fetch_ohlcv(symbol, start, end, interval="1mo")
    if timeframe == "1년":
        start = end - timedelta(days=365)
        return provider.fetch_ohlcv(symbol, start, end, interval="1d")
    # 전체: 10년 가량
    start = end - timedelta(days=365 * 10)
    return provider.fetch_ohlcv(symbol, start, end, interval="1d")

def _current_price(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])

def _code_to_name_map(provider: MarketDataProvider) -> dict[str, str]:
    df = provider.list_candidates()
    if df.empty:
        return {}
    return {c: n for c, n in zip(df["code"], df["name"])}

def _name_to_code(provider: MarketDataProvider, name: str) -> str | None:
    df = provider.list_candidates()
    if df.empty:
        return None
    # 우선 정확히 일치, 없으면 부분 일치
    exact = df[df["name"] == name]
    if not exact.empty:
        return str(exact["code"].iloc[0])
    partial = df[df["name"].str.contains(name, na=False)]
    if not partial.empty:
        return str(partial["code"].iloc[0])
    return None

def _place_order(symbol: str, price: float, shares: int = 0, amount: float = 0.0) -> None:
    portfolio = st.session_state["portfolio"]
    cash: float = portfolio["cash"]
    positions: dict = portfolio["positions"]
    executed_shares = shares
    if executed_shares == 0 and amount > 0 and price > 0:
        executed_shares = int(amount // price)
    if executed_shares == 0:
        return
    cost = executed_shares * price
    if cost > 0:
        # buy
        if cash < cost:
            st.warning("예수금이 부족합니다.")
            return
        portfolio["cash"] = cash - cost
        pos = positions.get(symbol, {"shares": 0, "avg_price": 0.0})
        new_shares = pos["shares"] + executed_shares
        new_cost = pos["avg_price"] * pos["shares"] + cost
        pos["shares"] = new_shares
        pos["avg_price"] = new_cost / new_shares
        positions[symbol] = pos
    else:
        # sell path not used here; handled in UI separately
        pass

def dashboard_page() -> None:
    st.subheader("자동 매매 대시보드")
    provider: MarketDataProvider = st.session_state["data_provider"]

    candidates = provider.list_candidates()
    default_symbol = st.session_state["selected_symbol"]
    code_to_name = _code_to_name_map(provider)
    selected_name = code_to_name.get(default_symbol, default_symbol)

    left, right = st.columns([3, 2])
    with left:
        col_s1, col_s2 = st.columns([3, 1])
        search_name = col_s1.text_input("종목명", value=selected_name)
        if col_s2.button("검색"):
            resolved = _name_to_code(provider, search_name)
            if resolved:
                st.session_state["selected_symbol"] = resolved
            else:
                st.warning("해당 종목명을 찾지 못했습니다.")
        timeframe = st.radio("기간",["1시간", "1일", "1주", "1달", "1년", "전체"], horizontal=True, index=1)

        symbol = st.session_state["selected_symbol"]
        data_load_state = st.text("데이터 불러오는 중...")
        data = fetch_data_with_interval(provider, symbol, timeframe)
        data_load_state.text("데이터 로드 완료 ✅")

        

        title_name = code_to_name.get(symbol, symbol)
        price_chart(data, f"{title_name} 차트")

    with right:
        st.markdown("#### 주문")
        sel_symbol = st.session_state["selected_symbol"]
        current = _current_price(fetch_data_with_interval(provider, sel_symbol, "1일"))
        st.metric("현재가", f"{current:,.0f} KRW")
        if "orders" not in st.session_state:
            st.session_state["orders"] = {"shares": 0, "amount": 0.0}
        shares = st.number_input("주문 수량(주)", min_value=0, step=1, value=int(st.session_state["orders"]["shares"]))
        st.session_state["orders"]["shares"] = shares
        amount = st.number_input("금액으로 주문 (KRW)", min_value=0.0, step=1000.0, value=float(st.session_state["orders"]["amount"]))
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

    st.divider()
    # 좌측: 보유 자산 요약, 우측: 보유 종목 표
    col_l, col_r = st.columns([1, 3])
    with col_l:
        pf = st.session_state["portfolio"]
        total_value = pf["cash"]
        unrealized = 0.0
        for sym, pos in pf["positions"].items():
            df_sym = fetch_data_with_interval(provider, sym, "1일")
            price_sym = _current_price(df_sym)
            total_value += pos["shares"] * price_sym
            unrealized += (price_sym - pos["avg_price"]) * pos["shares"]
        invested = total_value - pf["cash"]
        pnl_rate = (unrealized / invested * 100.0) if invested > 0 else 0.0
        st.metric("순자산", f"{total_value:,.0f} 원")
        st.metric("손익", f"{unrealized:,.0f} 원")
        st.metric("예수금", f"{pf['cash']:,.0f} 원")
        st.metric("수익률", f"{pnl_rate:.2f}%")
        st.metric("평가금액", f"{(total_value - pf['cash']):,.0f} 원")
    with col_r:
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



def data_page() -> None:
    st.subheader("자동매매")
    provider: MarketDataProvider = st.session_state["data_provider"]
    left, right = st.columns([3, 2])
    with left:
        code_to_name = _code_to_name_map(provider)
        current_code = st.session_state["selected_symbol"]
        current_name = code_to_name.get(current_code, current_code)
        name_input = st.text_input("종목명", value=current_name)
        # 이름을 코드로 변환
        resolved = _name_to_code(provider, name_input) or current_code
        symbol = resolved
        timeframe = st.radio("기간", ["1시간", "1일", "1주", "1달", "1년", "전체"], horizontal=True, index=1)
        df = fetch_data_with_interval(provider, symbol, timeframe)
        title_name = code_to_name.get(symbol, symbol)
        price_chart(df, f"{title_name} 차트")
        popular = provider.popular_tickers()
        st.markdown("#### 사람들이 많이 찾는 종목")
        st.dataframe(popular.head(10))
    with right:
        st.markdown("#### 자동매매 시스템")
        st.write("간단한 자동매매 데모를 실행할 수 있습니다.")
        timesteps = st.number_input("학습 스텝 수", min_value=1_000, max_value=100_000, step=1_000, value=3_000)
        if st.button("학습 시작"):
            with st.spinner("모델 학습 중..."):
                try:
                    from app.services.rl_trading import train_model  # type: ignore
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"RL 모듈 로드 실패: {exc}. requirements 설치를 확인해주세요.")
                    return
                train_df = fetch_data_with_interval(provider, symbol, "1일")
                st.session_state["model"] = train_model(train_df, timesteps)
            st.success("학습 완료")
        if st.button("백테스트"):
            if st.session_state.get("model") is None:
                st.warning("먼저 모델을 학습하세요.")
            else:
                with st.spinner("백테스트 실행 중..."):
                    try:
                        from app.services.rl_trading import backtest  # type: ignore
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"RL 모듈 로드 실패: {exc}. requirements 설치를 확인해주세요.")
                        return
                    result, sharpe_like = backtest(st.session_state["model"], df)
                    st.line_chart(result["cum_returns"], use_container_width=True)
                    st.area_chart(result["value"], use_container_width=True)
                    st.write(f"샤프 유사 지표: {sharpe_like:.4f}")


def chatbot_page() -> None:
    st.subheader("😁무엇이든 물어보세요.")
  
    
    prompt = st.text_input("질문", placeholder="질문을 입력하세요...")
    


def calculator_page() -> None:
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

    st.markdown("#### 평단가 계산")
    total_cost = st.number_input("총 매수금액", min_value=0.0, value=1_000_000.0, step=10_000.0)
    total_shares = st.number_input("총 수량", min_value=0, value=100, step=1)
    avg_price = calculate_average_price(total_cost, total_shares)
    st.write(f"평단가: {avg_price:,.2f} KRW")

    st.markdown("#### 환율 계산")
    amount = st.number_input("금액", min_value=0.0, value=1_000.0, step=10.0)
    country = st.selectbox(
        "국가",
        ["미국", "중국", "독일", "일본", "인도", "영국", "프랑스", "이탈리아", "캐나다", "브라질"],
        index=0,
    )
    code_map = {
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
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(code_map[country])
        fx = t.history(period="1d")
        rate = float(fx["Close"].iloc[-1]) if not fx.empty else 1300.0
    except Exception:
        rate = 1300.0
    col_a, col_b = st.columns(2)
    col_a.metric("원화 → 외화", f"{apply_fx(amount, rate, invert=True):,.2f}")
    col_b.metric("외화 → 원화", f"{apply_fx(amount, rate, invert=False):,.2f}")


def main() -> None:
    st.set_page_config(page_title="Kiwoom 자동 매매 도우미", layout="wide")
    init_session_state()

    st.sidebar.title("💰워렌 증권")
    page = st.sidebar.radio("메뉴", ["대시보드", "자동매매", "챗봇", "계산기"])

    if page == "대시보드":
        dashboard_page()
    elif page == "자동매매":
        data_page()
    elif page == "챗봇":
        chatbot_page()
    else:
        calculator_page()


if __name__ == "__main__":
    main()


