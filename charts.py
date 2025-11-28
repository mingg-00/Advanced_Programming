from __future__ import annotations

from io import BytesIO

import mplfinance as mpf
import pandas as pd
import streamlit as st


def price_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("차트를 그릴 데이터가 없습니다.")
        return

    buf = BytesIO()
    mpf.plot(
        df,
        type="candle",
        mav=(5, 20, 60),
        volume=True,
        style="yahoo",
        title=title,
        savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
    )
    st.image(buf)


