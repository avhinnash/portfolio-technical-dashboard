from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Portfolio Technical Dashboard",
    page_icon="📈",
    layout="wide",
)

PORTFOLIO: Dict[str, float] = {
    "IXG": 0.20,
    "IFRA": 0.14,
    "IGV": 0.10,
    "XAR": 0.10,
    "XLP": 0.10,
    "LLY": 0.05,
    "INDA": 0.05,
    "EMXC": 0.03,
    "EWZ": 0.03,
    "STRK": 0.10,
    "STRF": 0.10,
}

DISPLAY_NAMES: Dict[str, str] = {
    "IXG": "Global Financials",
    "IFRA": "U.S. Infrastructure",
    "IGV": "Expanded Tech-Software",
    "XAR": "Aerospace & Defense",
    "XLP": "Consumer Staples",
    "LLY": "Eli Lilly",
    "INDA": "India",
    "EMXC": "Emerging Markets ex-China",
    "EWZ": "Brazil",
    "STRK": "Strategy Series A Preferred",
    "STRF": "Strategy Series A Perpetual Preferred",
}

MA_WINDOWS = [20, 50, 100, 200]
EMA_WINDOWS = [8, 21]


@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers: Tuple[str, ...], start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
    """Download enough history to calculate the 200 DMA before the one-year display window."""
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    output: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required):
                continue

            df = df[required].dropna(subset=["Open", "High", "Low", "Close"])
            if df.empty:
                continue

            for window in MA_WINDOWS:
                df[f"DMA {window}"] = df["Close"].rolling(window).mean()

            for window in EMA_WINDOWS:
                df[f"EMA {window}"] = df["Close"].ewm(span=window, adjust=False).mean()

            # One-year anchored VWAP: cumulative typical-price x volume divided by cumulative volume
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            dollar_volume = typical_price * df["Volume"].fillna(0)
            cum_volume = df["Volume"].fillna(0).cumsum().replace(0, np.nan)
            df["1Y AVWAP"] = dollar_volume.cumsum() / cum_volume

            output[ticker] = df
        except Exception:
            continue

    return output


def build_chart(df: pd.DataFrame, ticker: str, show_volume: bool = True) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        )
    )

    line_styles = {
        "DMA 20": ("#fbc02d", 1.3),
        "DMA 50": ("#42a5f5", 1.4),
        "DMA 100": ("#7e57c2", 1.4),
        "DMA 200": ("#ef6c00", 1.7),
        "EMA 8": ("#66bb6a", 1.0),
        "EMA 21": ("#ec407a", 1.0),
        "1Y AVWAP": ("#fafafa", 2.0),
    }

    for column, (color, width) in line_styles.items():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                name=column,
                line=dict(color=color, width=width),
                hovertemplate=f"{column}: $%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{ticker} · {DISPLAY_NAMES.get(ticker, ticker)}",
            x=0.01,
            xanchor="left",
        ),
        template="plotly_dark",
        height=680 if show_volume else 610,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        yaxis=dict(title="Price (USD)", side="right"),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
        ]
    )

    return fig


def technical_snapshot(df: pd.DataFrame) -> Dict[str, object]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    price = float(last["Close"])
    change = price - float(prev["Close"])
    change_pct = change / float(prev["Close"]) if float(prev["Close"]) else np.nan

    return {
        "Price": price,
        "Daily change": change,
        "Daily change %": change_pct,
        "20 DMA": float(last["DMA 20"]) if pd.notna(last["DMA 20"]) else np.nan,
        "50 DMA": float(last["DMA 50"]) if pd.notna(last["DMA 50"]) else np.nan,
        "100 DMA": float(last["DMA 100"]) if pd.notna(last["DMA 100"]) else np.nan,
        "200 DMA": float(last["DMA 200"]) if pd.notna(last["DMA 200"]) else np.nan,
        "8 EMA": float(last["EMA 8"]) if pd.notna(last["EMA 8"]) else np.nan,
        "21 EMA": float(last["EMA 21"]) if pd.notna(last["EMA 21"]) else np.nan,
        "1Y AVWAP": float(last["1Y AVWAP"]) if pd.notna(last["1Y AVWAP"]) else np.nan,
        "Above 20 DMA": price > float(last["DMA 20"]) if pd.notna(last["DMA 20"]) else False,
        "Above 50 DMA": price > float(last["DMA 50"]) if pd.notna(last["DMA 50"]) else False,
        "Above 200 DMA": price > float(last["DMA 200"]) if pd.notna(last["DMA 200"]) else False,
        "EMA trend": "Bullish" if pd.notna(last["EMA 8"]) and pd.notna(last["EMA 21"]) and last["EMA 8"] > last["EMA 21"] else "Bearish",
    }


st.title("Portfolio Technical Dashboard")
st.caption(
    "Daily candles with 20, 50, 100 and 200-day moving averages, 8 and 21-day EMAs, "
    "and a one-year anchored VWAP. Data refreshes every 15 minutes while the app is running."
)

with st.sidebar:
    st.header("Portfolio")
    selected = st.multiselect(
        "Tickers",
        options=list(PORTFOLIO.keys()),
        default=list(PORTFOLIO.keys()),
    )
    chart_mode = st.radio("View", ["Single chart", "Chart grid"], index=0)
    selected_ticker = st.selectbox(
        "Active ticker",
        options=selected if selected else list(PORTFOLIO.keys()),
        index=0,
        disabled=(chart_mode == "Chart grid"),
    )
    columns_per_row = st.slider(
        "Charts per row",
        min_value=1,
        max_value=3,
        value=2,
        disabled=(chart_mode == "Single chart"),
    )
    st.divider()
    st.markdown("**Indicator definition**")
    st.caption(
        "1Y AVWAP is anchored to the first trading session in the displayed one-year window "
        "and uses typical price: (high + low + close) ÷ 3."
    )

today = dt.date.today()
display_start = today - dt.timedelta(days=365)
download_start = display_start - dt.timedelta(days=320)
download_end = today + dt.timedelta(days=1)

tickers_to_download = tuple(selected or PORTFOLIO.keys())

with st.spinner("Loading market data..."):
    histories = download_history(tickers_to_download, download_start, download_end)

# Re-anchor VWAP to the exact one-year display window and trim the chart.
display_data: Dict[str, pd.DataFrame] = {}
for ticker, full_df in histories.items():
    trimmed = full_df.loc[full_df.index.date >= display_start].copy()
    if trimmed.empty:
        continue
    typical_price = (trimmed["High"] + trimmed["Low"] + trimmed["Close"]) / 3.0
    volume = trimmed["Volume"].fillna(0)
    trimmed["1Y AVWAP"] = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    display_data[ticker] = trimmed

missing = [t for t in tickers_to_download if t not in display_data]
if missing:
    st.warning(
        "No usable Yahoo Finance history was returned for: "
        + ", ".join(missing)
        + ". Recently issued securities may have less than a full year of history."
    )

if not display_data:
    st.error("No market data was returned. Check your internet connection and try again.")
    st.stop()

# Portfolio snapshot
rows: List[Dict[str, object]] = []
for ticker, df in display_data.items():
    snap = technical_snapshot(df)
    rows.append(
        {
            "Ticker": ticker,
            "Weight": PORTFOLIO.get(ticker, np.nan),
            "Price": snap["Price"],
            "Day": snap["Daily change %"],
            "vs 20 DMA": snap["Price"] / snap["20 DMA"] - 1 if snap["20 DMA"] else np.nan,
            "vs 50 DMA": snap["Price"] / snap["50 DMA"] - 1 if snap["50 DMA"] else np.nan,
            "vs 200 DMA": snap["Price"] / snap["200 DMA"] - 1 if snap["200 DMA"] else np.nan,
            "vs AVWAP": snap["Price"] / snap["1Y AVWAP"] - 1 if snap["1Y AVWAP"] else np.nan,
            "EMA trend": snap["EMA trend"],
        }
    )

snapshot_df = pd.DataFrame(rows).set_index("Ticker").sort_index()

st.subheader("Portfolio snapshot")
st.dataframe(
    snapshot_df.style.format(
        {
            "Weight": "{:.0%}",
            "Price": "${:,.2f}",
            "Day": "{:+.2%}",
            "vs 20 DMA": "{:+.2%}",
            "vs 50 DMA": "{:+.2%}",
            "vs 200 DMA": "{:+.2%}",
            "vs AVWAP": "{:+.2%}",
        },
        na_rep="—",
    ),
    use_container_width=True,
    height=min(470, 38 + 35 * len(snapshot_df)),
)

st.divider()

if chart_mode == "Single chart":
    if selected_ticker not in display_data:
        st.error(f"No chart data is available for {selected_ticker}.")
    else:
        df = display_data[selected_ticker]
        snap = technical_snapshot(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last price", f"${snap['Price']:,.2f}", f"{snap['Daily change %']:+.2%}")
        c2.metric("vs. 20 DMA", f"{snap['Price'] / snap['20 DMA'] - 1:+.2%}" if snap["20 DMA"] else "—")
        c3.metric("vs. 50 DMA", f"{snap['Price'] / snap['50 DMA'] - 1:+.2%}" if snap["50 DMA"] else "—")
        c4.metric("vs. 200 DMA", f"{snap['Price'] / snap['200 DMA'] - 1:+.2%}" if snap["200 DMA"] else "—")
        c5.metric("vs. 1Y AVWAP", f"{snap['Price'] / snap['1Y AVWAP'] - 1:+.2%}" if snap["1Y AVWAP"] else "—")

        st.plotly_chart(build_chart(df, selected_ticker), use_container_width=True)
else:
    ordered = [t for t in selected if t in display_data]
    for start_idx in range(0, len(ordered), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, ticker in zip(cols, ordered[start_idx:start_idx + columns_per_row]):
            with col:
                st.plotly_chart(
                    build_chart(display_data[ticker], ticker, show_volume=False),
                    use_container_width=True,
                    key=f"chart-{ticker}",
                )

st.caption(
    "Market data source: Yahoo Finance through yfinance. Prices may be delayed and are intended "
    "for analytical use, not order execution."
)
