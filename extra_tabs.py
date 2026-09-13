from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

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


@st.cache_data(ttl=900, show_spinner=False)
def _latest_prices(tickers: tuple[str, ...]) -> pd.Series:
    end = dt.date.today() + dt.timedelta(days=1)
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
            series = df[col].dropna()
            if not series.empty:
                out[ticker] = float(series.iloc[-1])
        except Exception:
            continue
    return pd.Series(out, dtype=float)


def _pie(values: dict[str, float], title: str, hole: float = 0.38):
    df = pd.DataFrame({"Allocation": list(values.keys()), "Value": list(values.values())})
    df = df[df["Value"] > 0]
    fig = px.pie(df, names="Allocation", values="Value", title=title, hole=hole)
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>")
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=60, b=10), legend_title_text="")
    return fig


def render_allocation_tab() -> None:
    st.markdown("## Current and Future Portfolio Allocation")
    st.caption("MSTR is grouped with BTC, ETH, SOL, AVAX and LINK in the headline Crypto slice. The smaller pies show the crypto composition separately.")

    tickers = tuple(list(CRYPTO_TICKERS.values()) + ["MSTR"])
    prices = _latest_prices(tickers)
    missing = [t for t in tickers if t not in prices.index]
    if missing:
        st.error("Unable to load current prices for: " + ", ".join(missing))
        return

    crypto_values_now = {coin: CRYPTO_QTY[coin] * float(prices[t]) for coin, t in CRYPTO_TICKERS.items()}
    crypto_assets_now = float(sum(crypto_values_now.values()))
    default_total = 53000.0 + crypto_assets_now

    c1, c2, c3 = st.columns(3)
    with c1:
        total_now = st.number_input(
            "Current total portfolio value ($)", min_value=1.0, value=float(default_total),
            step=5000.0, format="%.0f", key="allocation_total_main",
        )
        st.caption("Default = $53,000 of stocks/options plus the live market value of your fixed crypto inventory.")
    with c2:
        mstr_shares = st.number_input("Fixed MSTR shares", min_value=0, value=200, step=100, key="allocation_mstr_main")
        st.caption("MSTR is included inside the Crypto slice for concentration analysis, but its market value is part of the $53k stocks/options baseline rather than added on top.")
    with c3:
        horizon = st.number_input("Future allocation year", min_value=1, max_value=30, value=10, step=1, key="allocation_horizon_main")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        annual_contribution = st.number_input("Annual new money to core ($)", min_value=0.0, value=30000.0, step=5000.0, format="%.0f", key="allocation_contrib_main")
    with t2:
        core_return = st.number_input("Core expected return", min_value=-0.20, max_value=0.30, value=0.075, step=0.005, format="%.3f", key="allocation_core_ret_main")
    with t3:
        crypto_return = st.number_input("Crypto expected return", min_value=-0.50, max_value=1.00, value=0.10, step=0.01, format="%.2f", key="allocation_crypto_ret_main")
    with t4:
        mstr_return = st.number_input("MSTR covered-call expected return", min_value=-0.50, max_value=1.00, value=0.12, step=0.01, format="%.2f", key="allocation_mstr_ret_main")

    mstr_now = float(mstr_shares) * float(prices["MSTR"])
    crypto_plus_mstr_now = crypto_assets_now + mstr_now
    core_now = max(float(total_now) - crypto_plus_mstr_now, 0.0)
    if crypto_plus_mstr_now > total_now:
        st.warning("Crypto + MSTR market value exceeds the total portfolio value entered. Increase total portfolio value so the allocation is economically meaningful.")

    core_weights = pd.Series(CORE_TARGET, dtype=float)
    core_weights /= core_weights.sum()
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
        st.plotly_chart(_pie(headline_now, "Current allocation"), use_container_width=True)
    with p2:
        st.plotly_chart(_pie(headline_future, f"Projected allocation in {int(horizon)} years"), use_container_width=True)

    st.markdown("### Crypto slice breakout")
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(_pie(crypto_breakout_now, "Current Crypto composition", hole=0.45), use_container_width=True)
    with b2:
        st.plotly_chart(_pie(crypto_breakout_future, f"Projected Crypto composition in {int(horizon)} years", hole=0.45), use_container_width=True)


def _add_months(d: dt.date, months: int) -> dt.date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return dt.date(year, month, 1)


def _monthly_payment(balance: float, annual_rate: float, months: int) -> float:
    if months <= 0:
        return balance
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return balance / months
    return balance * r / (1.0 - (1.0 + r) ** (-months))


def _consumer_schedule(starting_balance: float, payment_pct: float, annual_rate: float, balloon_date: dt.date, start_date: dt.date) -> pd.DataFrame:
    rows = []
    balance = float(starting_balance)
    date = start_date
    while balance > 0.01 and date <= balloon_date:
        interest = balance * annual_rate / 12.0
        beginning = balance
        balance += interest
        if date >= balloon_date:
            payment = balance
            balloon = balance
        else:
            payment = min(balance, beginning * payment_pct)
            balloon = 0.0
        balance -= payment
        rows.append({"Date": pd.Timestamp(date), "Debt": "Consumer debt", "Beginning balance": beginning, "Interest": interest, "Payment": payment, "Balloon": balloon, "Ending balance": max(balance, 0.0)})
        date = _add_months(date, 1)
    return pd.DataFrame(rows)


def _education_schedule(name: str, principal: float, annual_rate: float, origination_date: dt.date, interim_payment: float, repayment_start: dt.date, repayment_years: int, projection_end: dt.date, model_start: dt.date) -> pd.DataFrame:
    rows = []
    balance = 0.0
    date = min(model_start, origination_date)
    full_payment = None
    while date <= projection_end:
        if date < origination_date:
            date = _add_months(date, 1)
            continue
        if date == origination_date:
            balance += principal
        beginning = balance
        interest = beginning * annual_rate / 12.0
        balance += interest
        if date < repayment_start:
            payment = min(balance, interim_payment)
        else:
            if full_payment is None:
                full_payment = _monthly_payment(balance, annual_rate, repayment_years * 12)
            payment = min(balance, full_payment)
        balance -= payment
        rows.append({"Date": pd.Timestamp(date), "Debt": name, "Beginning balance": beginning, "Interest": interest, "Payment": payment, "Balloon": 0.0, "Ending balance": max(balance, 0.0), "Scheduled full payment": 0.0 if full_payment is None else full_payment})
        if balance <= 0.01 and date >= repayment_start:
            break
        date = _add_months(date, 1)
    return pd.DataFrame(rows)


def render_debt_tab() -> None:
    today = dt.date.today()
    model_start = dt.date(today.year, today.month, 1)
    st.markdown("## Debt & Liability Tracker")
    st.caption("Tracks consumer debt, education borrowing, accrued interest, balloon obligations and projected post-graduation amortization.")

    st.markdown("### Consumer debt")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        consumer_balance = st.number_input("Current consumer balance ($)", min_value=0.0, value=4742.0, step=100.0, format="%.0f", key="debt_consumer_balance")
    with c2:
        consumer_payment_pct = st.number_input("Monthly payment (% of balance)", min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.3f", key="debt_consumer_payment")
    with c3:
        consumer_rate = st.number_input("Consumer APR", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f", key="debt_consumer_rate")
        st.caption("Defaults to 0% because you have not specified the APR.")
    with c4:
        consumer_balloon = st.date_input("Balloon date", value=dt.date(2027, 6, 1), key="debt_consumer_balloon")

    st.markdown("### Education loans")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        first_principal = st.number_input("August 2026 draw ($)", min_value=0.0, value=100000.0, step=5000.0, format="%.0f", key="debt_loan1_principal")
    with e2:
        second_principal = st.number_input("August 2027 draw ($)", min_value=0.0, value=100000.0, step=5000.0, format="%.0f", key="debt_loan2_principal")
    with e3:
        loan_rate = st.number_input("Education loan APR", min_value=0.0, max_value=0.30, value=0.0729, step=0.001, format="%.4f", key="debt_loan_rate")
    with e4:
        interim_payment = st.number_input("Monthly payment before Jun 2028 ($)", min_value=0.0, value=25.0, step=5.0, format="%.0f", key="debt_interim_payment")

    e5, e6, e7, e8 = st.columns(4)
    with e5:
        first_draw = st.date_input("First draw date", value=dt.date(2026, 8, 1), key="debt_first_draw")
    with e6:
        second_draw = st.date_input("Second draw date", value=dt.date(2027, 8, 1), key="debt_second_draw")
    with e7:
        repayment_start = st.date_input("Full repayment starts", value=dt.date(2028, 6, 1), key="debt_repay_start")
    with e8:
        repayment_years = st.number_input("Repayment term (years)", min_value=1, max_value=30, value=10, step=1, key="debt_repay_years")

    split1, split2 = st.columns(2)
    with split1:
        federal_share = st.number_input("Federal share of each draw", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.2f", key="debt_federal_share")
        st.caption("Editable placeholder until you enter the actual federal/private split.")
    with split2:
        private_share = 1.0 - float(federal_share)
        st.metric("Private share of each draw", f"{private_share:.0%}")

    projection_years = st.number_input("Debt projection horizon (years)", min_value=2, max_value=30, value=15, step=1, key="debt_projection_years")
    projection_end = _add_months(model_start, int(projection_years) * 12)

    consumer = _consumer_schedule(float(consumer_balance), float(consumer_payment_pct), float(consumer_rate), consumer_balloon, model_start)
    loan_2026 = _education_schedule("Education loan 2026", float(first_principal), float(loan_rate), first_draw, float(interim_payment), repayment_start, int(repayment_years), projection_end, model_start)
    loan_2027 = _education_schedule("Education loan 2027", float(second_principal), float(loan_rate), second_draw, float(interim_payment), repayment_start, int(repayment_years), projection_end, model_start)
    schedule = pd.concat([consumer, loan_2026, loan_2027], ignore_index=True, sort=False)

    current_by_debt = {}
    for debt, grp in schedule.groupby("Debt"):
        eligible = grp[grp["Date"] <= pd.Timestamp(model_start)]
        current_by_debt[debt] = 0.0 if eligible.empty else float(eligible.iloc[-1]["Ending balance"])
    current_total = float(sum(current_by_debt.values()))

    def balance_before_repayment(df: pd.DataFrame) -> float:
        rows = df[df["Date"] < pd.Timestamp(repayment_start)]
        return float(rows.iloc[-1]["Ending balance"]) if not rows.empty else 0.0

    bal_2026 = balance_before_repayment(loan_2026)
    bal_2027 = balance_before_repayment(loan_2027)
    pmt_2026 = _monthly_payment(bal_2026 * (1 + float(loan_rate) / 12.0), float(loan_rate), int(repayment_years) * 12) if bal_2026 > 0 else 0.0
    pmt_2027 = _monthly_payment(bal_2027 * (1 + float(loan_rate) / 12.0), float(loan_rate), int(repayment_years) * 12) if bal_2027 > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modeled liabilities now", f"${current_total:,.0f}")
    m2.metric("Consumer debt", f"${current_by_debt.get('Consumer debt', 0.0):,.0f}")
    m3.metric("Education debt now", f"${current_by_debt.get('Education loan 2026', 0.0) + current_by_debt.get('Education loan 2027', 0.0):,.0f}")
    m4.metric("Projected monthly payment from Jun 2028", f"${pmt_2026 + pmt_2027:,.0f}")

    st.markdown("### Liability balances over time")
    balance_pivot = schedule.pivot_table(index="Date", columns="Debt", values="Ending balance", aggfunc="last").sort_index().ffill().fillna(0.0)
    balance_pivot["Total liabilities"] = balance_pivot.sum(axis=1)
    plot_df = balance_pivot.reset_index().melt(id_vars="Date", var_name="Liability", value_name="Balance")
    fig = px.line(plot_df, x="Date", y="Balance", color="Liability", title="Projected debt balances")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Education debt composition")
    total_draws = float(first_principal) + float(second_principal)
    split_df = pd.DataFrame({"Type": ["Federal", "Private"], "Principal": [total_draws * float(federal_share), total_draws * private_share]})
    st.dataframe(split_df.style.format({"Principal": "${:,.0f}"}), use_container_width=True, hide_index=True)

    with st.expander("Monthly debt schedule"):
        st.dataframe(schedule.style.format({
            "Beginning balance": "${:,.0f}", "Interest": "${:,.0f}", "Payment": "${:,.0f}",
            "Balloon": "${:,.0f}", "Ending balance": "${:,.0f}", "Scheduled full payment": "${:,.0f}",
        }), use_container_width=True, hide_index=True)

    st.info("Education-loan projections are planning estimates. Actual federal/private capitalization, grace periods, fees and repayment rules can differ from this simplified model.")
