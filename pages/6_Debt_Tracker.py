from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Debt Tracker", page_icon="💳", layout="wide")

TODAY = dt.date.today()
MODEL_START = dt.date(TODAY.year, TODAY.month, 1)
DEFAULT_REPAYMENT_START = dt.date(2028, 6, 1)


def add_months(d: dt.date, months: int) -> dt.date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return dt.date(year, month, 1)


def monthly_payment(balance: float, annual_rate: float, months: int) -> float:
    if months <= 0:
        return balance
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return balance / months
    return balance * r / (1.0 - (1.0 + r) ** (-months))


def consumer_schedule(
    starting_balance: float,
    payment_pct: float,
    annual_rate: float,
    balloon_date: dt.date,
    start_date: dt.date,
) -> pd.DataFrame:
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
        rows.append({
            "Date": pd.Timestamp(date),
            "Debt": "Consumer debt",
            "Beginning balance": beginning,
            "Interest": interest,
            "Payment": payment,
            "Balloon": balloon,
            "Ending balance": max(balance, 0.0),
        })
        date = add_months(date, 1)
    return pd.DataFrame(rows)


def education_schedule(
    name: str,
    principal: float,
    annual_rate: float,
    origination_date: dt.date,
    interim_payment: float,
    repayment_start: dt.date,
    repayment_years: int,
    projection_end: dt.date,
) -> pd.DataFrame:
    rows = []
    balance = 0.0
    date = min(MODEL_START, origination_date)
    full_payment = None

    while date <= projection_end:
        if date < origination_date:
            date = add_months(date, 1)
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
                full_payment = monthly_payment(balance, annual_rate, repayment_years * 12)
            payment = min(balance, full_payment)

        balance -= payment
        rows.append({
            "Date": pd.Timestamp(date),
            "Debt": name,
            "Beginning balance": beginning,
            "Interest": interest,
            "Payment": payment,
            "Balloon": 0.0,
            "Ending balance": max(balance, 0.0),
            "Scheduled full payment": 0.0 if full_payment is None else full_payment,
        })

        if balance <= 0.01 and date >= repayment_start:
            break
        date = add_months(date, 1)

    return pd.DataFrame(rows)


st.title("Debt & Liability Tracker")
st.caption("Tracks your consumer debt and education borrowing, including pre-graduation payments, accrued interest, balloon obligations, and projected post-graduation amortization.")

st.markdown("### Consumer debt")
c1, c2, c3, c4 = st.columns(4)
with c1:
    consumer_balance = st.number_input("Current consumer balance ($)", min_value=0.0, value=4742.0, step=100.0, format="%.0f")
    st.caption("Current outstanding consumer balance.")
with c2:
    consumer_payment_pct = st.number_input("Monthly payment (% of balance)", min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.3f")
    st.caption("Default is 1% of the beginning balance each month until the balloon date.")
with c3:
    consumer_rate = st.number_input("Consumer APR", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
    st.caption("Set to 0% because no consumer-debt interest rate was specified. Update this if interest accrues.")
with c4:
    consumer_balloon = st.date_input("Balloon date", value=dt.date(2027, 6, 1))
    st.caption("Any remaining consumer balance is paid in full on this date.")

st.markdown("### Education loans")
st.write("The defaults model two $100,000 education draws at 7.29% fixed: one in August 2026 and another in August 2027. Each receives a $25 monthly payment until full repayment starts in June 2028.")

e1, e2, e3, e4 = st.columns(4)
with e1:
    loan_principal = st.number_input("Principal per annual draw ($)", min_value=0.0, value=100000.0, step=5000.0, format="%.0f")
    st.caption("Applied separately to the August 2026 and August 2027 draws.")
with e2:
    loan_rate = st.number_input("Education loan APR", min_value=0.0, max_value=0.30, value=0.0729, step=0.001, format="%.4f")
    st.caption("Fixed annual rate used for both modeled education loans.")
with e3:
    interim_payment = st.number_input("Monthly payment before repayment ($)", min_value=0.0, value=25.0, step=5.0, format="%.0f")
    st.caption("Paid monthly before June 2028. Because this is below monthly interest, the balance grows before graduation.")
with e4:
    repayment_start = st.date_input("Full repayment starts", value=DEFAULT_REPAYMENT_START)
    st.caption("The model begins amortizing the then-outstanding balance starting this month.")

e5, e6, e7 = st.columns(3)
with e5:
    first_draw = st.date_input("First $100k draw", value=dt.date(2026, 8, 1))
    st.caption("First-year education borrowing.")
with e6:
    second_draw = st.date_input("Second $100k draw", value=dt.date(2027, 8, 1))
    st.caption("Second-year borrowing at the same default terms.")
with e7:
    repayment_years = st.number_input("Post-graduation amortization term (years)", min_value=1, max_value=30, value=10, step=1)
    st.caption("Used to estimate the regular monthly payment once full repayment starts. Ten years is the modeling default, not a claim about your servicer's final plan.")

projection_years = st.number_input("Debt projection horizon (years)", min_value=2, max_value=30, value=15, step=1)
projection_end = add_months(MODEL_START, int(projection_years) * 12)

consumer = consumer_schedule(
    float(consumer_balance),
    float(consumer_payment_pct),
    float(consumer_rate),
    consumer_balloon,
    MODEL_START,
)
loan_2026 = education_schedule(
    "Education loan 2026",
    float(loan_principal),
    float(loan_rate),
    first_draw,
    float(interim_payment),
    repayment_start,
    int(repayment_years),
    projection_end,
)
loan_2027 = education_schedule(
    "Education loan 2027",
    float(loan_principal),
    float(loan_rate),
    second_draw,
    float(interim_payment),
    repayment_start,
    int(repayment_years),
    projection_end,
)

schedule = pd.concat([consumer, loan_2026, loan_2027], ignore_index=True, sort=False)

# Current liability is the latest modeled balance on or before the current month for each debt.
current_rows = []
for debt, grp in schedule.groupby("Debt"):
    eligible = grp[grp["Date"] <= pd.Timestamp(MODEL_START)]
    if eligible.empty:
        current_balance = 0.0
    else:
        current_balance = float(eligible.iloc[-1]["Ending balance"])
    current_rows.append((debt, current_balance))
current_by_debt = dict(current_rows)
current_total = float(sum(current_by_debt.values()))

# Balances immediately before full repayment begins.
def balance_before_repayment(df: pd.DataFrame) -> float:
    rows = df[df["Date"] < pd.Timestamp(repayment_start)]
    return float(rows.iloc[-1]["Ending balance"]) if not rows.empty else 0.0

bal_2026_repay = balance_before_repayment(loan_2026)
bal_2027_repay = balance_before_repayment(loan_2027)
pmt_2026 = monthly_payment(bal_2026_repay * (1 + float(loan_rate) / 12.0), float(loan_rate), int(repayment_years) * 12) if bal_2026_repay > 0 else 0.0
pmt_2027 = monthly_payment(bal_2027_repay * (1 + float(loan_rate) / 12.0), float(loan_rate), int(repayment_years) * 12) if bal_2027_repay > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Modeled liabilities now", f"${current_total:,.0f}")
m2.metric("Consumer debt", f"${current_by_debt.get('Consumer debt', 0.0):,.0f}")
m3.metric("Education debt now", f"${current_by_debt.get('Education loan 2026', 0.0) + current_by_debt.get('Education loan 2027', 0.0):,.0f}")
m4.metric("Projected monthly payment from Jun 2028", f"${pmt_2026 + pmt_2027:,.0f}")

st.caption("Education balances accrue interest monthly while the $25 payment is active. The June 2028 payment estimate assumes both balances then amortize over the term selected above.")

st.markdown("### Liability balances over time")
balance_pivot = schedule.pivot_table(index="Date", columns="Debt", values="Ending balance", aggfunc="last").sort_index().ffill().fillna(0.0)
balance_pivot["Total liabilities"] = balance_pivot.sum(axis=1)
plot_df = balance_pivot.reset_index().melt(id_vars="Date", var_name="Liability", value_name="Balance")
fig = px.line(plot_df, x="Date", y="Balance", color="Liability", title="Projected debt balances")
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
fig.update_layout(template="plotly_dark", height=520)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Key obligations")
consumer_balloon_amt = float(consumer.loc[consumer["Date"] == pd.Timestamp(consumer_balloon), "Payment"].iloc[0]) if not consumer.empty and any(consumer["Date"] == pd.Timestamp(consumer_balloon)) else 0.0
summary = pd.DataFrame([
    {
        "Liability": "Consumer debt",
        "Original/current principal": float(consumer_balance),
        "APR": float(consumer_rate),
        "Interim payment": f"{float(consumer_payment_pct):.1%} of balance",
        "Major date": consumer_balloon,
        "Major obligation": f"Balloon ≈ ${consumer_balloon_amt:,.0f}",
    },
    {
        "Liability": "Education loan 2026",
        "Original/current principal": float(loan_principal),
        "APR": float(loan_rate),
        "Interim payment": f"${float(interim_payment):,.0f}/mo",
        "Major date": repayment_start,
        "Major obligation": f"Est. full payment ≈ ${pmt_2026:,.0f}/mo",
    },
    {
        "Liability": "Education loan 2027",
        "Original/current principal": float(loan_principal),
        "APR": float(loan_rate),
        "Interim payment": f"${float(interim_payment):,.0f}/mo",
        "Major date": repayment_start,
        "Major obligation": f"Est. full payment ≈ ${pmt_2027:,.0f}/mo",
    },
])
st.dataframe(summary.style.format({"Original/current principal": "${:,.0f}", "APR": "{:.2%}"}), use_container_width=True, hide_index=True)

with st.expander("Monthly debt schedule"):
    display = schedule.copy()
    st.dataframe(
        display.style.format({
            "Beginning balance": "${:,.0f}",
            "Interest": "${:,.0f}",
            "Payment": "${:,.0f}",
            "Balloon": "${:,.0f}",
            "Ending balance": "${:,.0f}",
            "Scheduled full payment": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.info("The education-loan schedule is a planning model. It assumes monthly compounding at the entered fixed APR, $25 monthly payments until June 2028, and then level monthly amortization over the selected term. Actual federal/private servicing rules, capitalization timing, fees, grace periods, and repayment plans can differ.")