from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Debt Tracker", page_icon="💳", layout="wide")

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
STOCKS_OPTIONS_DEFAULT = 53000.0
TODAY = dt.date.today()
MODEL_START = dt.date(TODAY.year, TODAY.month, 1)


@st.cache_data(ttl=900, show_spinner=False)
def latest_crypto_prices() -> pd.Series:
    tickers = tuple(CRYPTO_TICKERS.values())
    end = TODAY + dt.timedelta(days=1)
    start = end - dt.timedelta(days=14)
    raw = yf.download(
        list(tickers), start=start, end=end, interval="1d", auto_adjust=False,
        actions=False, group_by="ticker", threads=True, progress=False,
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
            s = df[col].dropna()
            if not s.empty:
                out[ticker] = float(s.iloc[-1])
        except Exception:
            continue
    return pd.Series(out, dtype=float)


def monthly_payment(balance: float, annual_rate: float, months: int) -> float:
    if months <= 0:
        return balance
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return balance / months
    return balance * r / (1 - (1 + r) ** (-months))


def loan_schedule(
    name: str,
    principal: float,
    origination: pd.Timestamp,
    annual_rate: float,
    interim_payment: float,
    repayment_start: pd.Timestamp,
    repayment_years: int,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.date_range(origination, end_date, freq="MS")
    balance = float(principal)
    rows = []
    full_payment = None
    for date in dates:
        beginning = balance
        interest = beginning * annual_rate / 12.0
        balance += interest
        if date < repayment_start:
            payment = min(balance, interim_payment)
            phase = "School / interim"
        else:
            if full_payment is None:
                full_payment = monthly_payment(balance, annual_rate, repayment_years * 12)
            payment = min(balance, full_payment)
            phase = "Amortizing repayment"
        principal_paid = max(payment - interest, 0.0)
        balance = max(balance - payment, 0.0)
        rows.append({
            "Date": date,
            "Debt": name,
            "Beginning balance": beginning,
            "Interest": interest,
            "Payment": payment,
            "Principal paid": principal_paid,
            "Ending balance": balance,
            "Phase": phase,
        })
        if balance <= 0.01:
            break
    return pd.DataFrame(rows)


def consumer_schedule(
    starting_balance: float,
    payment_pct: float,
    annual_rate: float,
    start: pd.Timestamp,
    balloon_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.date_range(start, balloon_date, freq="MS")
    balance = float(starting_balance)
    rows = []
    for date in dates:
        beginning = balance
        interest = beginning * annual_rate / 12.0
        balance += interest
        if date >= balloon_date:
            payment = balance
            phase = "Balloon"
        else:
            payment = min(beginning * payment_pct, balance)
            phase = "Monthly payment"
        principal_paid = max(payment - interest, 0.0)
        balance = max(balance - payment, 0.0)
        rows.append({
            "Date": date,
            "Debt": "Consumer debt",
            "Beginning balance": beginning,
            "Interest": interest,
            "Payment": payment,
            "Principal paid": principal_paid,
            "Ending balance": balance,
            "Phase": phase,
        })
    return pd.DataFrame(rows)


def balance_at(df: pd.DataFrame, date: pd.Timestamp, original: float, origination: pd.Timestamp) -> float:
    if date < origination:
        return 0.0
    eligible = df[df["Date"] <= date]
    if eligible.empty:
        return float(original)
    return float(eligible.iloc[-1]["Ending balance"])


st.title("Debt & Liability Tracker")
st.caption("Tracks consumer debt, current and planned education borrowing, monthly debt service, and liabilities relative to your investable assets.")

st.markdown("### Consumer debt")
c1, c2, c3, c4 = st.columns(4)
with c1:
    consumer_balance = st.number_input("Consumer debt balance ($)", min_value=0.0, value=4742.0, step=100.0, format="%.0f")
    st.caption("Default current balance: $4,742.")
with c2:
    consumer_payment_pct = st.number_input("Monthly payment (% of balance)", min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.3f")
    st.caption("Default is 1% of the beginning balance each month until the balloon.")
with c3:
    consumer_rate = st.number_input("Consumer APR", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
    st.caption("No APR was provided, so this defaults to 0%. Enter the actual rate if interest accrues.")
with c4:
    consumer_balloon = st.date_input("Balloon date", value=dt.date(2027, 6, 1))
    st.caption("Any remaining balance is paid in full on June 1, 2027.")

st.markdown("### Education loans")
st.write("Defaults assume $100,000 borrowed in August 2026 and another $100,000 in August 2027, both at 7.29% fixed, with $25 monthly payments until full repayment begins in June 2028.")

e1, e2, e3, e4 = st.columns(4)
with e1:
    first_loan = st.number_input("Aug 2026 borrowing ($)", min_value=0.0, value=100000.0, step=5000.0, format="%.0f")
    st.caption("Total federal + private borrowing for the first year.")
with e2:
    second_loan = st.number_input("Aug 2027 borrowing ($)", min_value=0.0, value=100000.0, step=5000.0, format="%.0f")
    st.caption("Planned second-year borrowing at similar terms.")
with e3:
    education_rate = st.number_input("Fixed education-loan APR", min_value=0.0, max_value=0.30, value=0.0729, step=0.001, format="%.4f")
    st.caption("7.29% fixed is applied to both annual draws until you enter separate terms.")
with e4:
    interim_payment = st.number_input("Monthly payment before Jun 2028 ($)", min_value=0.0, value=25.0, step=5.0, format="%.0f")
    st.caption("Applied separately to each outstanding annual draw before full repayment begins.")

e5, e6, e7 = st.columns(3)
with e5:
    repayment_start = st.date_input("Full repayment starts", value=dt.date(2028, 6, 1))
    st.caption("June 2028 is modeled as the first regular amortizing-payment month.")
with e6:
    repayment_years = st.number_input("Repayment term after graduation (years)", min_value=1, max_value=30, value=10, step=1)
    st.caption("Used to estimate the monthly payment after graduation. Change this when your actual repayment plan is known.")
with e7:
    federal_share = st.number_input("Federal share of education borrowing", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.2f")
    st.caption("Tracking split only. The remaining share is labeled Private. Both currently use the same 7.29% assumption.")

private_share = 1.0 - float(federal_share)
repayment_start_ts = pd.Timestamp(repayment_start)
model_end = repayment_start_ts + pd.DateOffset(years=int(repayment_years) + 1)

consumer = consumer_schedule(
    float(consumer_balance), float(consumer_payment_pct), float(consumer_rate),
    pd.Timestamp(MODEL_START), pd.Timestamp(consumer_balloon),
)
loan_2026 = loan_schedule(
    "Education loan 2026", float(first_loan), pd.Timestamp(2026, 8, 1),
    float(education_rate), float(interim_payment), repayment_start_ts,
    int(repayment_years), model_end,
)
loan_2027 = loan_schedule(
    "Education loan 2027", float(second_loan), pd.Timestamp(2027, 8, 1),
    float(education_rate), float(interim_payment), repayment_start_ts,
    int(repayment_years), model_end,
)
schedule = pd.concat([consumer, loan_2026, loan_2027], ignore_index=True)

asof = pd.Timestamp(MODEL_START)
consumer_now = balance_at(consumer, asof, float(consumer_balance), pd.Timestamp(MODEL_START))
loan_2026_now = balance_at(loan_2026, asof, float(first_loan), pd.Timestamp(2026, 8, 1))
loan_2027_now = balance_at(loan_2027, asof, float(second_loan), pd.Timestamp(2027, 8, 1))
education_now = loan_2026_now + loan_2027_now
total_debt_now = consumer_now + education_now

# Live asset baseline: $53k stocks/options + current crypto market value.
crypto_prices = latest_crypto_prices()
crypto_value = 0.0
for coin, ticker in CRYPTO_TICKERS.items():
    if ticker in crypto_prices.index:
        crypto_value += CRYPTO_QTY[coin] * float(crypto_prices[ticker])
default_assets = STOCKS_OPTIONS_DEFAULT + crypto_value

m1, m2, m3, m4 = st.columns(4)
m1.metric("Modeled liabilities now", f"${total_debt_now:,.0f}")
m2.metric("Consumer debt", f"${consumer_now:,.0f}")
m3.metric("Education debt outstanding", f"${education_now:,.0f}")
m4.metric("Planned Aug 2027 borrowing", f"${second_loan:,.0f}" if asof < pd.Timestamp(2027, 8, 1) else "$0")

st.markdown("### Assets vs liabilities")
a1, a2, a3, a4 = st.columns(4)
with a1:
    investable_assets = st.number_input("Investable assets ($)", min_value=0.0, value=float(default_assets), step=5000.0, format="%.0f")
    st.caption(f"Default = $53,000 stocks/options + live crypto market value (${crypto_value:,.0f}).")
with a2:
    net_position = float(investable_assets) - total_debt_now
    st.metric("Assets minus liabilities", f"${net_position:,.0f}")
with a3:
    debt_to_assets = total_debt_now / float(investable_assets) if investable_assets > 0 else np.nan
    st.metric("Debt / investable assets", f"{debt_to_assets:.1%}" if np.isfinite(debt_to_assets) else "—")
with a4:
    st.metric("Live crypto value", f"${crypto_value:,.0f}")

st.markdown("### Current liability breakout")
breakout = pd.DataFrame({
    "Liability": ["Consumer debt", "Federal education", "Private education"],
    "Balance": [consumer_now, education_now * float(federal_share), education_now * private_share],
})
fig_pie = px.pie(breakout, names="Liability", values="Balance", hole=0.4, title="Current modeled liabilities")
fig_pie.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>")
fig_pie.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Liability path")
monthly = schedule.groupby("Date", as_index=False).agg({"Ending balance": "sum", "Payment": "sum", "Interest": "sum"})
fig_balance = px.line(monthly, x="Date", y="Ending balance", title="Modeled total debt balance over time")
fig_balance.update_yaxes(tickprefix="$", tickformat=",.0f")
fig_balance.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig_balance, use_container_width=True)

fig_payment = px.bar(monthly, x="Date", y="Payment", title="Modeled monthly debt service")
fig_payment.update_yaxes(tickprefix="$", tickformat=",.0f")
fig_payment.update_layout(template="plotly_dark", height=460)
st.plotly_chart(fig_payment, use_container_width=True)

# Estimated post-graduation payment from the first repayment rows.
def first_regular_payment(df: pd.DataFrame) -> float:
    x = df[df["Date"] >= repayment_start_ts]
    return float(x.iloc[0]["Payment"]) if not x.empty else 0.0

pmt_2026 = first_regular_payment(loan_2026)
pmt_2027 = first_regular_payment(loan_2027)
consumer_balloon_rows = consumer[consumer["Date"] == pd.Timestamp(consumer_balloon)]
consumer_balloon_amt = float(consumer_balloon_rows.iloc[0]["Payment"]) if not consumer_balloon_rows.empty else 0.0

st.markdown("### Key obligations")
summary = pd.DataFrame([
    {
        "Liability": "Consumer debt",
        "Principal": float(consumer_balance),
        "APR": float(consumer_rate),
        "Current/interim payment": f"{float(consumer_payment_pct):.1%} of balance",
        "Major date": consumer_balloon,
        "Major obligation": f"Balloon ≈ ${consumer_balloon_amt:,.0f}",
    },
    {
        "Liability": "Education loan 2026",
        "Principal": float(first_loan),
        "APR": float(education_rate),
        "Current/interim payment": f"${float(interim_payment):,.0f}/mo",
        "Major date": repayment_start,
        "Major obligation": f"Est. payment ≈ ${pmt_2026:,.0f}/mo",
    },
    {
        "Liability": "Education loan 2027",
        "Principal": float(second_loan),
        "APR": float(education_rate),
        "Current/interim payment": f"${float(interim_payment):,.0f}/mo",
        "Major date": repayment_start,
        "Major obligation": f"Est. payment ≈ ${pmt_2027:,.0f}/mo",
    },
])
st.dataframe(summary.style.format({"Principal": "${:,.0f}", "APR": "{:.2%}"}), use_container_width=True, hide_index=True)

with st.expander("Monthly debt schedule"):
    st.dataframe(
        schedule.style.format({
            "Beginning balance": "${:,.0f}", "Interest": "${:,.2f}",
            "Payment": "${:,.2f}", "Principal paid": "${:,.2f}", "Ending balance": "${:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.info("The education-loan schedule is a planning model. It assumes monthly compounding at 7.29%, $25 monthly payments per annual draw until June 2028, then level amortization over the selected term. The federal/private split is for tracking only until separate rates, fees, and repayment rules are entered.")