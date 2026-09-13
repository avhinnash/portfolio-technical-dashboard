from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Portfolio Dashboard",
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
    "IGV": "Software",
    "XAR": "Aerospace & Defense",
    "XLP": "Consumer Staples",
    "LLY": "Eli Lilly",
    "INDA": "India",
    "EMXC": "Emerging Markets ex-China",
    "EWZ": "Brazil",
    "STRK": "Strategy Preferred",
    "STRF": "Strategy Perpetual Preferred",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
}

MA_WINDOWS = [20, 50, 100, 200]
EMA_WINDOWS = [8, 21]


@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers: Tuple[str, ...], start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
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

    out: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out

    for ticker in tickers:
        try:
            # Recent yfinance versions may return MultiIndex columns even for
            # a single ticker, so handle both MultiIndex and flat-column cases.
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0)
                level1 = raw.columns.get_level_values(1)

                if ticker in level0:
                    df = raw[ticker].copy()
                elif ticker in level1:
                    df = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
            else:
                df = raw.copy()

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if not all(c in df.columns for c in required):
                continue

            df = df[required].dropna(subset=["Close"])
            if not df.empty:
                out[ticker] = df
        except Exception as exc:
            st.warning(f"Could not process {ticker}: {exc}")

    return out


def add_indicators(df: pd.DataFrame, display_start: dt.date) -> pd.DataFrame:
    x = df.copy()
    for w in MA_WINDOWS:
        x[f"DMA {w}"] = x["Close"].rolling(w).mean()
    for w in EMA_WINDOWS:
        x[f"EMA {w}"] = x["Close"].ewm(span=w, adjust=False).mean()
    x = x.loc[x.index.date >= display_start].copy()
    if x.empty:
        return x
    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    vol = x["Volume"].fillna(0)
    x["1Y AVWAP"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    return x


def technical_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ))
    styles = {
        "DMA 20": ("#fbc02d", 1.3), "DMA 50": ("#42a5f5", 1.4),
        "DMA 100": ("#7e57c2", 1.4), "DMA 200": ("#ef6c00", 1.8),
        "EMA 8": ("#66bb6a", 1.0), "EMA 21": ("#ec407a", 1.0), "1Y AVWAP": ("#ffffff", 2.0),
    }
    for col, (color, width) in styles.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=color, width=width)))
    fig.update_layout(
        title=f"{ticker} · {DISPLAY_NAMES.get(ticker, ticker)}", template="plotly_dark", height=680,
        margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11)),
        yaxis=dict(title="Price (USD)", side="right"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def adjusted_close(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: df["Adj Close"] for t, df in data.items()}).sort_index()


def annualized_vol(returns: pd.DataFrame) -> pd.Series:
    return returns.std(skipna=True) * np.sqrt(252)


def beta_to(returns: pd.DataFrame, benchmark: str) -> pd.Series:
    out = {}
    b = returns[benchmark]
    for c in returns.columns:
        pair = pd.concat([returns[c], b], axis=1).dropna()
        if len(pair) < 30 or pair.iloc[:, 1].var() == 0:
            out[c] = np.nan
        else:
            out[c] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
    return pd.Series(out)


def portfolio_stats(returns: pd.DataFrame, weights: pd.Series):
    cols = [c for c in weights.index if c in returns.columns]
    clean = returns[cols].dropna()
    if len(clean) < 20:
        return np.nan, pd.Series(dtype=float), pd.DataFrame(), len(clean)
    w = weights.loc[cols].astype(float)
    w = w / w.sum()
    cov_ann = clean.cov() * 252
    variance = float(w.values @ cov_ann.values @ w.values)
    vol = np.sqrt(max(variance, 0))
    marginal = cov_ann.values @ w.values
    contrib = w.values * marginal
    pct_contrib = contrib / variance if variance > 0 else np.repeat(np.nan, len(cols))
    return vol, pd.Series(pct_contrib, index=cols), cov_ann, len(clean)


def stress_corr(returns: pd.DataFrame, benchmark: str, q: float):
    b = returns[benchmark].dropna()
    threshold = b.quantile(q)
    dates = b[b <= threshold].index
    stressed = returns.loc[returns.index.intersection(dates)]
    return stressed.corr(min_periods=10), len(stressed), threshold


def heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
    fig.update_layout(
        title=title, template="plotly_dark", height=max(520, 48 * len(matrix.columns)),
        margin=dict(l=10, r=10, t=60, b=10), coloraxis_colorbar=dict(title="ρ"),
    )
    return fig


st.title("Portfolio Technical & Risk Dashboard")
st.caption(
    "Live historical data via Yahoo Finance. Includes technical indicators, correlation, volatility, beta, "
    "covariance, risk contribution, and downside-regime analysis."
)

tab1, tab2 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk"])
today = dt.date.today()
end_date = today + dt.timedelta(days=1)

with tab1:
    ticker = st.selectbox("Ticker", list(PORTFOLIO.keys()), index=0)
    display_start = today - dt.timedelta(days=365)
    start_date = display_start - dt.timedelta(days=330)
    with st.spinner("Loading market data..."):
        hist = download_history((ticker,), start_date, end_date)
    if ticker not in hist:
        st.error(f"No history returned for {ticker}.")
    else:
        df = add_indicators(hist[ticker], display_start)
        if df.empty:
            st.error(f"No display-period data available for {ticker}.")
        else:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            price = float(last["Close"])
            day_pct = price / float(prev["Close"]) - 1 if float(prev["Close"]) else np.nan
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Last price", f"${price:,.2f}", f"{day_pct:+.2%}")
            c2.metric("vs 20 DMA", f"{price / last['DMA 20'] - 1:+.2%}" if pd.notna(last["DMA 20"]) else "—")
            c3.metric("vs 50 DMA", f"{price / last['DMA 50'] - 1:+.2%}" if pd.notna(last["DMA 50"]) else "—")
            c4.metric("vs 200 DMA", f"{price / last['DMA 200'] - 1:+.2%}" if pd.notna(last["DMA 200"]) else "—")
            c5.metric("vs 1Y AVWAP", f"{price / last['1Y AVWAP'] - 1:+.2%}" if pd.notna(last["1Y AVWAP"]) else "—")
            st.plotly_chart(technical_chart(df, ticker), use_container_width=True)
            st.info("1Y AVWAP is anchored to the first trading session in the displayed one-year window and uses typical price = (High + Low + Close) / 3.")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        lookback = st.selectbox("Lookback", ["1Y", "3Y", "5Y"], index=2)
    with c2:
        stress_pct = st.selectbox("Stress sample", [10, 20, 25], index=1, format_func=lambda x: f"Worst {x}% of SPY days")
    with c3:
        selected = st.multiselect("Holdings", options=list(PORTFOLIO.keys()), default=list(PORTFOLIO.keys()))

    if selected:
        years = {"1Y": 1, "3Y": 3, "5Y": 5}[lookback]
        start = today - dt.timedelta(days=int(365.25 * years) + 30)
        tickers = tuple(dict.fromkeys(selected + ["SPY", "QQQ"]))
        with st.spinner("Loading risk-history data..."):
            raw = download_history(tickers, start, end_date)
        prices = adjusted_close(raw)
        available = [c for c in selected if c in prices.columns]
        missing = [c for c in selected if c not in prices.columns]
        if missing:
            st.warning("No usable data returned for: " + ", ".join(missing))
        if not available:
            st.error("No usable holdings data returned.")
        else:
            returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            corr = returns[available].corr(min_periods=30)
            vols = annualized_vol(returns[available])
            beta_spy = beta_to(returns[available + ["SPY"]].dropna(how="all"), "SPY").reindex(available)
            beta_qqq = beta_to(returns[available + ["QQQ"]].dropna(how="all"), "QQQ").reindex(available)
            weights = pd.Series(PORTFOLIO).reindex(available).dropna()
            pvol, rc, cov_ann, common_obs = portfolio_stats(returns, weights)
            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Portfolio annualized vol", f"{pvol:.1%}" if pd.notna(pvol) else "—")
            top2.metric("Common observations", f"{common_obs:,}")
            top3.metric("Selected target weight", f"{weights.sum():.0%}")
            top4.metric("Lookback", lookback)

            summary = pd.DataFrame(index=available)
            summary["Target weight"] = pd.Series(PORTFOLIO).reindex(available)
            summary["Annualized vol"] = vols.reindex(available)
            summary["Beta vs SPY"] = beta_spy
            summary["Beta vs QQQ"] = beta_qqq
            summary["Risk contribution"] = rc.reindex(available)
            summary["Available obs"] = [int(returns[c].notna().sum()) for c in available]
            st.markdown("### Risk snapshot")
            st.dataframe(summary.style.format({
                "Target weight": "{:.0%}", "Annualized vol": "{:.1%}", "Beta vs SPY": "{:.2f}",
                "Beta vs QQQ": "{:.2f}", "Risk contribution": "{:.1%}", "Available obs": "{:,.0f}",
            }, na_rep="—"), use_container_width=True)

            st.markdown("### Correlation matrix")
            st.plotly_chart(heatmap(corr, f"{lookback} pairwise daily-return correlation"), use_container_width=True)

            if not rc.empty:
                rc_df = rc.sort_values().rename("Risk contribution").reset_index()
                rc_df.columns = ["Ticker", "Risk contribution"]
                fig_rc = px.bar(rc_df, x="Risk contribution", y="Ticker", orientation="h", text="Risk contribution", title="Contribution to portfolio variance")
                fig_rc.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig_rc.update_xaxes(tickformat=".0%")
                fig_rc.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_rc, use_container_width=True)

            if "SPY" in returns.columns:
                stress_input_cols = available + ["SPY"]
                stress_matrix, stress_obs, threshold = stress_corr(returns[stress_input_cols], benchmark="SPY", q=stress_pct / 100.0)
                st.markdown(f"### Stress correlation · worst {stress_pct}% of SPY days")
                st.plotly_chart(heatmap(stress_matrix.loc[available, available], f"Stress correlation on SPY days ≤ {threshold:.2%} ({stress_obs} sessions)"), use_container_width=True)

            with st.expander("Annualized covariance matrix"):
                if cov_ann.empty:
                    st.info("Not enough overlapping observations.")
                else:
                    st.dataframe(cov_ann.style.format("{:.4f}"), use_container_width=True)

            st.caption(
                "Methodology: daily adjusted-close percentage returns; annualized volatility uses √252; "
                "portfolio volatility uses wᵀΣw; risk contribution uses each holding's component contribution to total portfolio variance. "
                "STRK and STRF have shorter histories, which shortens common-history calculations."
            )
