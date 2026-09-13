from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Allocation Mix", page_icon="🥧", layout="wide")

CORE_TARGET = {
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

CRYPTO_QTY = {
    "BTC": 1.559331,
    "ETH": 4.2596421,
    "SOL": 24.905188,
    "AVAX": 27.027027,
    "LINK": 227.37717,
}

CRYPTO_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
}


@st.cache_data(ttl=900, show_spinner=False)
def latest_prices(tickers: tuple[str, ...]) -> pd.Series:
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=14)
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
    out = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    df = raw[ticker]
                elif ticker in raw.columns.get_level_values(1):
                    df = raw.xs(ticker, axis=1, level=1)
                else:
                    continue
            else:
                df = raw
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[col].dropna()
            if not series.empty:
                out[ticker] = float(series.iloc[-1])
        except Exception:
            continue
    return pd.Series(out, dtype=float)


def pie(values: dict[str, float], title: str, hole: float = 0.38):
    df = pd.DataFrame({"Allocation": list(values.keys()), "Value": list(values.values())})
    df = df[df["Value"] > 0]
    fig = px.pie(df, names="Allocation", values="Value", title=title, hole=hole)
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>")
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=60, b=10), legend_title_text="")
    return fig


st.title("Current and Future Portfolio Allocation")
st.caption("MSTR is grouped with your digital-asset exposure in the headline Crypto slice. The smaller pies break that slice into BTC, ETH, SOL, AVAX, LINK, and MSTR.")

tickers = tuple(list(CRYPTO_TICKERS.values()) + ["MSTR"])
prices = latest_prices(tickers)
missing = [t for t in tickers if t not in prices.index]

if missing:
    st.error("Unable to load current prices for: " + ", ".join(missing))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    total_now = st.number_input(
        "Current total portfolio value ($)",
        min_value=1.0,
        value=float(st.session_state.get("trans_total", 250000.0)),
        step=10000.0,
        format="%.0f",
        key="allocation_total",
    )
    st.caption("Total investable portfolio used to infer the dollars currently held in the diversified core after subtracting fixed crypto and MSTR.")
with c2:
    mstr_shares = st.number_input(
        "Fixed MSTR shares",
        min_value=0,
        value=int(st.session_state.get("trans_mstr", 200)),
        step=100,
        key="allocation_mstr_shares",
    )
    st.caption("MSTR is treated as part of the Crypto slice here because it is economically tied to your broader digital-asset exposure.")
with c3:
    horizon = st.number_input("Future allocation year", min_value=1, max_value=30, value=10, step=1)
    st.caption("The future pies show the projected mix at this horizon without selling the fixed crypto quantities or MSTR shares.")

t1, t2, t3, t4 = st.columns(4)
with t1:
    annual_contribution = st.number_input("Annual new money to core ($)", min_value=0.0, value=30000.0, step=5000.0, format="%.0f")
    st.caption("New money is added only to the diversified core, which is the primary mechanism for diluting crypto exposure without selling it.")
with t2:
    core_return = st.number_input("Core expected return", min_value=-0.20, max_value=0.30, value=0.075, step=0.005, format="%.3f")
    st.caption("Annual return assumption for the diversified core/index portfolio.")
with t3:
    crypto_return = st.number_input("Crypto expected return", min_value=-0.50, max_value=1.00, value=0.10, step=0.01, format="%.2f")
    st.caption("Applied to BTC, ETH, SOL, AVAX, and LINK as a group. Higher crypto returns slow dilution from new core contributions.")
with t4:
    mstr_return = st.number_input("MSTR covered-call expected return", min_value=-0.50, max_value=1.00, value=0.12, step=0.01, format="%.2f")
    st.caption("Expected annual return on the MSTR sleeve after incorporating your covered-call strategy.")

crypto_values_now = {
    coin: CRYPTO_QTY[coin] * float(prices[ticker])
    for coin, ticker in CRYPTO_TICKERS.items()
}
mstr_now = float(mstr_shares) * float(prices["MSTR"])
crypto_plus_mstr_now = float(sum(crypto_values_now.values()) + mstr_now)
core_now = max(float(total_now) - crypto_plus_mstr_now, 0.0)

if crypto_plus_mstr_now > total_now:
    st.warning("The fixed crypto + MSTR market value exceeds the total portfolio value entered. Increase the portfolio value so the allocation pies are economically meaningful.")

core_weights = pd.Series(CORE_TARGET, dtype=float)
core_weights = core_weights / core_weights.sum()

headline_now = {asset: core_now * float(weight) for asset, weight in core_weights.items()}
headline_now["Crypto"] = crypto_plus_mstr_now

crypto_breakout_now = dict(crypto_values_now)
crypto_breakout_now["MSTR"] = mstr_now

crypto_future = {coin: value * ((1 + float(crypto_return)) ** int(horizon)) for coin, value in crypto_values_now.items()}
mstr_future = mstr_now * ((1 + float(mstr_return)) ** int(horizon))
core_future = core_now
for _ in range(int(horizon)):
    core_future = core_future * (1 + float(core_return)) + float(annual_contribution)

crypto_plus_mstr_future = float(sum(crypto_future.values()) + mstr_future)
headline_future = {asset: core_future * float(weight) for asset, weight in core_weights.items()}
headline_future["Crypto"] = crypto_plus_mstr_future

crypto_breakout_future = dict(crypto_future)
crypto_breakout_future["MSTR"] = mstr_future

current_total_modeled = sum(headline_now.values())
future_total_modeled = sum(headline_future.values())
current_crypto_weight = crypto_plus_mstr_now / current_total_modeled if current_total_modeled > 0 else 0.0
future_crypto_weight = crypto_plus_mstr_future / future_total_modeled if future_total_modeled > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Crypto + MSTR value now", f"${crypto_plus_mstr_now:,.0f}")
m2.metric("Crypto + MSTR weight now", f"{current_crypto_weight:.1%}")
m3.metric(f"Crypto + MSTR weight in {int(horizon)}Y", f"{future_crypto_weight:.1%}", f"{future_crypto_weight-current_crypto_weight:+.1%}")
m4.metric(f"Projected portfolio value in {int(horizon)}Y", f"${future_total_modeled:,.0f}")

st.markdown("### Headline allocation")
p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(pie(headline_now, "Current allocation"), use_container_width=True)
with p2:
    st.plotly_chart(pie(headline_future, f"Projected allocation in {int(horizon)} years"), use_container_width=True)

st.markdown("### Crypto slice breakout")
b1, b2 = st.columns(2)
with b1:
    st.plotly_chart(pie(crypto_breakout_now, "Current Crypto composition", hole=0.45), use_container_width=True)
with b2:
    st.plotly_chart(pie(crypto_breakout_future, f"Projected Crypto composition in {int(horizon)} years", hole=0.45), use_container_width=True)

st.caption("The headline Crypto slice is BTC + ETH + SOL + AVAX + LINK + MSTR. The future chart assumes no sales of these positions; their market values change only through the return assumptions, while all new contributions are directed to the core portfolio in your existing target proportions.")