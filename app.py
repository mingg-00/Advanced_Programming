from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app.components.charts import price_chart
from app.config import get_default_symbol, get_openai_key
from app.services.chatbot import TradingAssistantChatbot
from app.services.data_provider import MarketDataProvider
from app.services.rl_trading import backtest, train_model
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


def fetch_data(provider: MarketDataProvider, symbol: str, period_days: int = 365) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=period_days)
    return provider.fetch_daily_ohlcv(symbol, start, end)


def dashboard_page() -> None:
    st.subheader("자동 매매 대시보드")
    provider: MarketDataProvider = st.session_state["data_provider"]

    candidates = provider.list_candidates()
    default_symbol = st.session_state["selected_symbol"]

    symbol = st.selectbox(
        "종목을 선택하세요",
        options=candidates["code"] if not candidates.empty else [default_symbol],
        index=0,
    )
    st.session_state["selected_symbol"] = symbol

    data_load_state = st.text("데이터 불러오는 중...")
    data = fetch_data(provider, symbol)
    data_load_state.text("데이터 로드 완료 ✅")

    st.metric("최근 종가", f"{data['close'].iloc[-1]:,.0f} KRW")
    st.metric("평균 거래량", f"{data['volume'].mean():,.0f}")

    price_chart(data, f"{symbol} 일봉 차트")

    st.divider()
    st.markdown("#### 강화학습 자동 매매 시뮬레이터")
    col1, col2 = st.columns(2)
    timesteps = col1.number_input("학습 스텝 수", min_value=1_000, max_value=100_000, step=1_000, value=5_000)
    period_days = col2.slider("학습 기간 (일)", min_value=90, max_value=720, value=365, step=30)

    if st.button("모델 학습 시작"):
        with st.spinner("모델을 학습 중입니다. 잠시만 기다려주세요..."):
            train_df = fetch_data(provider, symbol, period_days)
            st.session_state["model"] = train_model(train_df, timesteps)
        st.success("학습이 완료되었습니다.")

    if st.session_state["model"] is not None:
        if st.button("백테스트 실행"):
            with st.spinner("백테스트 실행 중..."):
                result, sharpe_like = backtest(st.session_state["model"], data)
                st.session_state["backtest"] = (result, sharpe_like)
            st.success("백테스트가 완료되었습니다.")

    if st.session_state["backtest"] is not None:
        result, sharpe_like = st.session_state["backtest"]
        st.markdown(f"**샤프 유사 지표:** {sharpe_like:.4f}")
        st.line_chart(result["cum_returns"], use_container_width=True)
        st.area_chart(result["value"], use_container_width=True)
        st.dataframe(result.tail(10))


def data_page() -> None:
    st.subheader("종목 데이터 탐색")
    provider: MarketDataProvider = st.session_state["data_provider"]
    candidates = provider.list_candidates()
    st.dataframe(candidates)

    symbol = st.text_input("관심 종목 코드", value=get_default_symbol())
    days = st.slider("조회 기간 (일)", 30, 720, 180, step=30)

    if st.button("데이터 조회"):
        with st.spinner("시세 데이터를 불러오는 중입니다..."):
            df = fetch_data(provider, symbol, days)
            st.dataframe(df.tail(20))
            price_chart(df, f"{symbol} 차트")


def chatbot_page() -> None:
    st.subheader("도우미 챗봇")
    api_key = st.text_input("OpenAI API Key", value=get_openai_key() or "", type="password")

    if "chatbot" not in st.session_state and api_key:
        try:
            st.session_state["chatbot"] = TradingAssistantChatbot(api_key=api_key)
        except RuntimeError as err:
            st.error(str(err))

    prompt = st.text_area("질문을 입력하세요", height=150, placeholder="어떤 종목을 분석해볼까요?")

    if st.button("질문하기", disabled=not api_key):
        if not api_key:
            st.warning("API Key를 입력하세요.")
        else:
            try:
                bot: TradingAssistantChatbot = st.session_state["chatbot"]
                answer = bot.ask(prompt)
                st.session_state["chat_history"].append({"question": prompt, "answer": answer})
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"챗봇 요청 중 오류가 발생했습니다: {exc}")

    for chat in reversed(st.session_state["chat_history"]):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.divider()


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
    rate = st.number_input("환율", min_value=0.0, value=1_300.0, step=10.0)
    col_a, col_b = st.columns(2)
    col_a.metric("원화 → 외화", f"{apply_fx(amount, rate, invert=True):,.2f}")
    col_b.metric("외화 → 원화", f"{apply_fx(amount, rate, invert=False):,.2f}")


def main() -> None:
    st.set_page_config(page_title="Kiwoom 자동 매매 도우미", layout="wide")
    init_session_state()

    st.sidebar.title("슈퍼런 증권")
    page = st.sidebar.radio("메뉴", ["대시보드", "자료실", "챗봇", "계산기"])

    if page == "대시보드":
        dashboard_page()
    elif page == "자료실":
        data_page()
    elif page == "챗봇":
        chatbot_page()
    else:
        calculator_page()


if __name__ == "__main__":
    main()


