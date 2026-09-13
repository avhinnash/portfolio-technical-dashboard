from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📈", layout="wide")

CORE_TARGET: Dict[str, float] = {
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
MSTR_CC = "MSTR CC"

DISPLAY_NAMES = {
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
    "MSTR": "Strategy common stock",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "SOXX": "Semiconductors",
    MSTR_CC: "MSTR covered-call sleeve",
    **{v: k for k, v in CRYPTO_TICKERS.items()},
}

DEFAULT_CUSTOM_RETURNS = {
    "IXG": 0.075,
    "IFRA": 0.075,
    "IGV": 0.085,
    "XAR": 0.080,
    "XLP": 0.060,
    "LLY": 0.080,
    "INDA": 0.090,
    "EMXC": 0.085,
    "EWZ": 0.080,
    "STRK": 0.085,
    "STRF": 0.085,
    MSTR_CC: 0.12,
    "BTC-USD": 0.12,
    "ETH-USD": 0.13,
    "SOL-USD": 0.15,
    "AVAX-USD": 0.15,
    "LINK-USD": 0.14,
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
            if isinstance(raw.columns, pd.MultiIndex):
                l0, l1 = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
                if ticker in l0:
                    df = raw[ticker].copy()
                elif ticker in l1:
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
        except Exception:
            continue
    return out


def adjusted_close(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: df["Adj Close"] for t, df in data.items()}).sort_index()


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
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
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
    return fig


def heatmap(matrix: pd.DataFrame, title: str, fmt: str = ".2f", center_zero: bool = False, colorbar: str = "Value") -> go.Figure:
    zmax = float(np.nanmax(np.abs(matrix.values))) if center_zero and matrix.size else None
    fig = px.imshow(
        matrix,
        text_auto=fmt,
        aspect="auto",
        color_continuous_scale="RdBu_r" if center_zero else "Viridis",
        zmin=-zmax if center_zero and zmax else None,
        zmax=zmax if center_zero and zmax else None,
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=max(520, 44 * len(matrix.columns)),
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar=dict(title=colorbar),
    )
    return fig


def annualized_vol(returns: pd.DataFrame) -> pd.Series:
    return returns.std(skipna=True) * np.sqrt(252)


def beta_to(returns: pd.DataFrame, benchmark: str) -> pd.Series:
    out = {}
    b = returns[benchmark]
    for c in returns.columns:
        pair = pd.concat([returns[c], b], axis=1).dropna()
        out[c] = np.nan if len(pair) < 30 or pair.iloc[:, 1].var() == 0 else pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
    return pd.Series(out)


def geometric_annual_return(r: pd.Series) -> float:
    x = r.dropna()
    if len(x) < 30:
        return np.nan
    gross = (1 + x).clip(lower=1e-8).prod()
    years = len(x) / 252.0
    return gross ** (1 / years) - 1 if years > 0 else np.nan


def covered_call_terms(delta: float, dte: int, iv: float, rf: float) -> tuple[float, float, float]:
    T = max(dte, 1) / 365.0
    sigma = max(iv, 1e-6)
    d1 = norm.ppf(np.clip(delta, 0.01, 0.99))
    strike_ratio = np.exp((rf + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    premium_ratio = norm.cdf(d1) - strike_ratio * np.exp(-rf * T) * norm.cdf(d2)
    annualized_premium = premium_ratio * 365.0 / max(dte, 1)
    return float(strike_ratio), float(max(premium_ratio, 0.0)), float(max(annualized_premium, 0.0))


def synthetic_mstr_cc_returns(mstr_returns: pd.Series, call_delta: float, dte: int, iv: float, rf: float, income_capture: float) -> pd.Series:
    _, _, annualized_premium = covered_call_terms(call_delta, dte, iv, rf)
    r = mstr_returns.fillna(0.0)
    downside = np.minimum(r, 0.0)
    upside = np.maximum(r, 0.0)
    daily_income = annualized_premium * np.clip(income_capture, 0.0, 1.0) / 252.0
    return (downside + (1.0 - call_delta) * upside + daily_income).rename(MSTR_CC)


def expected_return_vector(method: str, returns: pd.DataFrame, assets: list[str], custom: pd.Series, rf: float, market_return: float) -> pd.Series:
    if method == "Custom assumptions":
        return custom.reindex(assets).astype(float)
    if method == "Historical geometric":
        return pd.Series({a: geometric_annual_return(returns[a]) for a in assets})
    if "SPY" not in returns.columns:
        return custom.reindex(assets).astype(float)
    betas = beta_to(returns[assets + ["SPY"]].dropna(how="all"), "SPY").reindex(assets)
    return rf + betas * (market_return - rf)


def port_metrics(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    vol = float(np.sqrt(max(variance, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def optimize_weights(mu: np.ndarray, cov: np.ndarray, rf: float, bounds: list[tuple[float, float]], x0: np.ndarray):
    def objective(w):
        ret, vol, _ = port_metrics(w, mu, cov, rf)
        return -((ret - rf) / vol) if vol > 0 else 1e6

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1500})


def exact_shift_heatmap(weights: pd.Series, cov: pd.DataFrame, shift: float = 0.01) -> pd.DataFrame:
    assets = list(weights.index)
    w0 = weights.values.astype(float)
    covv = cov.loc[assets, assets].values.astype(float)
    base_vol = float(np.sqrt(max(w0 @ covv @ w0, 0.0)))
    out = pd.DataFrame(np.nan, index=assets, columns=assets)
    for i, src in enumerate(assets):
        for j, dst in enumerate(assets):
            if i == j:
                out.iloc[i, j] = 0.0
                continue
            if w0[i] + 1e-12 < shift:
                continue
            w = w0.copy()
            w[i] -= shift
            w[j] += shift
            new_vol = float(np.sqrt(max(w @ covv @ w, 0.0)))
            out.iloc[i, j] = (new_vol - base_vol) * 10000.0
    return out


def hedge_scenario_table(spot: float, delta: float, gamma: float, theta_day: float, vega: float, days: int, vol_shock_per_10pct_down: float) -> pd.DataFrame:
    rows = []
    for move in [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        ds = spot * move
        iv_points = (-move / 0.10) * vol_shock_per_10pct_down if move < 0 else (-move / 0.10) * vol_shock_per_10pct_down * 0.35
        delta_pnl = delta * ds
        gamma_pnl = 0.5 * gamma * ds**2
        vega_pnl = vega * iv_points
        theta_pnl = theta_day * days
        total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        rows.append({
            "SOXX move": move,
            "SOXX price": spot * (1 + move),
            "IV shock (pts)": iv_points,
            "Delta P&L": delta_pnl,
            "Gamma P&L": gamma_pnl,
            "Vega P&L": vega_pnl,
            "Theta P&L": theta_pnl,
            "Estimated put P&L": total,
        })
    return pd.DataFrame(rows)


def crypto_transition_projection(crypto_start: float, core_start: float, mstr_start: float, annual_contribution: float, years: int, core_return: float, crypto_return: float, mstr_return: float) -> pd.DataFrame:
    rows = []
    crypto, core, mstr = crypto_start, core_start, mstr_start
    for y in range(years + 1):
        total = crypto + core + mstr
        rows.append({
            "Year": y,
            "Core indexes": core,
            "Crypto": crypto,
            "MSTR CC": mstr,
            "Total": total,
            "Crypto weight": crypto / total if total > 0 else np.nan,
            "Core weight": core / total if total > 0 else np.nan,
            "MSTR weight": mstr / total if total > 0 else np.nan,
        })
        if y < years:
            crypto *= (1 + crypto_return)
            mstr *= (1 + mstr_return)
            core = core * (1 + core_return) + annual_contribution
    return pd.DataFrame(rows)


st.title("Portfolio Technical, Risk & Transition Dashboard")
st.caption("Technical monitoring, covariance and risk attribution, Sharpe optimization, option overlays, and a no-sale crypto dilution plan.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan"])
today = dt.date.today()
end_date = today + dt.timedelta(days=1)

with tab1:
    tech_options = list(CORE_TARGET.keys()) + ["MSTR", "SOXX"] + list(CRYPTO_TICKERS.values())
    ticker = st.selectbox("Ticker", tech_options, index=0)
    display_start = today - dt.timedelta(days=365)
    start_date = display_start - dt.timedelta(days=330)
    hist = download_history((ticker,), start_date, end_date)
    if ticker not in hist:
        st.error(f"No history returned for {ticker}.")
    else:
        df = add_indicators(hist[ticker], display_start)
        if df.empty:
            st.error(f"No display-period data available for {ticker}.")
        else:
            st.plotly_chart(technical_chart(df, ticker), use_container_width=True)

with tab2:
    r1, r2 = st.columns(2)
    with r1:
        lookback = st.selectbox("Risk lookback", ["1Y", "3Y", "5Y"], index=2)
    with r2:
        include_crypto = st.checkbox("Include fixed crypto positions", value=True)
    years = {"1Y": 1, "3Y": 3, "5Y": 5}[lookback]
    start = today - dt.timedelta(days=int(365.25 * years) + 30)
    risk_assets = list(CORE_TARGET.keys()) + ["MSTR"]
    if include_crypto:
        risk_assets += list(CRYPTO_TICKERS.values())
    tickers = tuple(dict.fromkeys(risk_assets + ["SPY", "QQQ"]))
    raw = download_history(tickers, start, end_date)
    prices = adjusted_close(raw)
    available = [a for a in risk_assets if a in prices.columns]
    if len(available) < 2:
        st.error("Not enough usable history for risk analysis.")
    else:
        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        corr = returns[available].corr(min_periods=30)
        common = returns[available].dropna()
        cov_ann = common.cov() * 252
        st.markdown("### Correlation heatmap")
        st.plotly_chart(heatmap(corr, f"{lookback} daily-return correlation", ".2f", True, "Correlation"), use_container_width=True)
        st.caption("Correlation shows direction co-movement on a -1 to +1 scale. It ignores each asset's absolute volatility.")
        st.markdown("### Annualized covariance heatmap")
        st.plotly_chart(heatmap(cov_ann, f"{lookback} annualized covariance", ".3f", True, "Covariance"), use_container_width=True)
        st.caption("Covariance combines co-movement and volatility. Diagonal cells are each asset's variance. Large positive off-diagonal cells mean the pair amplifies portfolio risk when both weights are increased.")

with tab3:
    st.markdown("### Portfolio construction assumptions")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        opt_lookback = st.selectbox("Covariance lookback", ["1Y", "3Y", "5Y"], index=2, key="opt_lookback")
        st.caption("History window used to estimate the covariance matrix. Longer windows are steadier, shorter windows react faster to regime changes.")
    with a2:
        return_model = st.selectbox("Expected-return model", ["Custom assumptions", "Historical geometric", "CAPM / market-implied"], index=0)
        st.caption("The optimizer can only maximize Sharpe relative to the return assumptions you give it. Custom assumptions are usually the most controllable input.")
    with a3:
        rf = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
        st.caption("Subtracted from expected portfolio return in the Sharpe numerator. A higher risk-free rate makes risky assets work harder to justify their volatility.")
    with a4:
        market_return = st.number_input("Market expected return", min_value=-0.10, max_value=0.30, value=0.085, step=0.005, format="%.3f")
        st.caption("Used only by the CAPM / market-implied return model.")

    st.markdown("#### Fixed positions and MSTR covered calls")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        total_portfolio_value = st.number_input("Total portfolio value ($)", min_value=1.0, value=250000.0, step=10000.0, format="%.0f")
        st.caption("Used to translate your fixed crypto quantities and MSTR shares into current portfolio weights.")
    with b2:
        mstr_shares = st.number_input("MSTR shares", min_value=0, value=200, step=100)
        st.caption("Current MSTR shares. The model treats this as a fixed position when optimizing.")
    with b3:
        call_delta = st.number_input("Short-call delta", min_value=0.05, max_value=0.90, value=0.35, step=0.05, format="%.2f")
        st.caption("Higher call delta collects more premium but gives away more upside. Around 0.35 is your normal setting; 0.50 approximates ATM.")
    with b4:
        call_dte = st.number_input("Call DTE", min_value=7, max_value=180, value=47, step=1)
        st.caption("Days to expiration at entry. Your normal 45–50 DTE range balances premium decay and rolling frequency.")
    b5, b6 = st.columns(2)
    with b5:
        mstr_iv = st.number_input("Assumed MSTR IV", min_value=0.10, max_value=3.00, value=0.80, step=0.05, format="%.2f")
        st.caption("Used for the covered-call premium approximation. Higher IV means higher theoretical call premium.")
    with b6:
        income_capture = st.number_input("Premium income capture", min_value=0.0, max_value=1.0, value=0.70, step=0.05, format="%.2f")
        st.caption("Fraction of theoretical premium you expect to keep after rolls, buybacks, slippage and imperfect execution.")

    st.markdown("#### SOXX put overlay using your actual Greeks")
    g1, g2, g3 = st.columns(3)
    with g1:
        put_delta = st.number_input("Position delta (share equivalents)", value=0.0, step=10.0, format="%.1f")
        st.caption("Enter the broker's position-level delta. Long puts should normally be negative. Example: -80 means the position behaves initially like short 80 SOXX shares.")
    with g2:
        put_gamma = st.number_input("Position gamma (delta shares per $1)", value=0.0, step=0.1, format="%.3f")
        st.caption("How much position delta changes for a $1 move in SOXX. Positive gamma makes protection accelerate as SOXX falls.")
    with g3:
        put_market_value = st.number_input("Put market value ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
        st.caption("Current market value of the SOXX put position. This is capital at risk, separate from delta-equivalent exposure.")
    g4, g5, g6 = st.columns(3)
    with g4:
        put_theta = st.number_input("Position theta ($ / day)", value=0.0, step=10.0, format="%.2f")
        st.caption("Expected one-day time decay with other inputs unchanged. Long puts normally have negative theta.")
    with g5:
        put_vega = st.number_input("Position vega ($ / IV point)", value=0.0, step=10.0, format="%.2f")
        st.caption("Dollar P&L for a one-point change in implied volatility. Long puts normally have positive vega.")
    with g6:
        scenario_days = st.number_input("Scenario horizon (days)", min_value=0, max_value=365, value=10, step=1)
        st.caption("Number of days of theta decay included in the hedge stress table.")
    vol_shock = st.number_input("IV-point rise per 10% SOXX decline", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    st.caption("Stress assumption for vega. Example: 10 means a 10% SOXX decline is paired with a +10-point IV shock; upside moves use a smaller reverse shock.")

    years = {"1Y": 1, "3Y": 3, "5Y": 5}[opt_lookback]
    start = today - dt.timedelta(days=int(365.25 * years) + 30)
    needed = tuple(dict.fromkeys(list(CORE_TARGET.keys()) + ["SPY", "MSTR", "SOXX"] + list(CRYPTO_TICKERS.values())))
    opt_raw = download_history(needed, start, end_date)
    opt_prices = adjusted_close(opt_raw)

    required = ["MSTR", "SOXX"] + list(CRYPTO_TICKERS.values())
    missing = [x for x in required if x not in opt_prices.columns]
    if missing:
        st.error("Missing market data for: " + ", ".join(missing))
    else:
        latest = opt_prices.ffill().iloc[-1]
        crypto_values = {coin: CRYPTO_QTY[coin] * float(latest[ticker]) for coin, ticker in CRYPTO_TICKERS.items()}
        crypto_total = float(sum(crypto_values.values()))
        mstr_value = float(mstr_shares) * float(latest["MSTR"])
        fixed_capital = crypto_total + mstr_value
        if fixed_capital > total_portfolio_value:
            st.warning("Fixed MSTR + crypto market value exceeds the portfolio value you entered. Increase total portfolio value so weights are economically meaningful.")

        st.markdown("##### Fixed crypto inventory")
        crypto_table = pd.DataFrame({
            "Asset": list(CRYPTO_QTY.keys()),
            "Quantity": [CRYPTO_QTY[c] for c in CRYPTO_QTY],
            "Price": [float(latest[CRYPTO_TICKERS[c]]) for c in CRYPTO_QTY],
            "Market value": [crypto_values[c] for c in CRYPTO_QTY],
            "Portfolio weight": [crypto_values[c] / total_portfolio_value for c in CRYPTO_QTY],
        })
        st.dataframe(crypto_table.style.format({"Quantity": "{:.6f}", "Price": "${:,.2f}", "Market value": "${:,.0f}", "Portfolio weight": "{:.1%}"}), use_container_width=True)

        base_returns = opt_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        base_returns[MSTR_CC] = synthetic_mstr_cc_returns(base_returns["MSTR"], float(call_delta), int(call_dte), float(mstr_iv), rf, float(income_capture))
        assets = list(CORE_TARGET.keys()) + [MSTR_CC] + list(CRYPTO_TICKERS.values())
        available_assets = [a for a in assets if a in base_returns.columns]
        common = base_returns[available_assets].dropna()

        if len(common) < 30:
            st.error("Not enough overlapping observations for covariance estimation. STRK/STRF history may be limiting the sample.")
        else:
            cov_ann = common.cov() * 252
            fixed_weights = {CRYPTO_TICKERS[c]: crypto_values[c] / total_portfolio_value for c in CRYPTO_QTY}
            fixed_weights[MSTR_CC] = mstr_value / total_portfolio_value
            fixed_total = sum(fixed_weights.values())
            core_budget = max(1.0 - fixed_total, 0.0)
            core_sum = sum(CORE_TARGET.values())
            current_weights = {a: CORE_TARGET[a] / core_sum * core_budget for a in CORE_TARGET}
            current_weights.update(fixed_weights)
            w_current = pd.Series(current_weights).reindex(available_assets).fillna(0.0)
            if w_current.sum() > 0:
                w_current = w_current / w_current.sum()

            editor = pd.DataFrame({
                "Asset": available_assets,
                "Current weight %": [w_current.get(a, 0.0) * 100 for a in available_assets],
                "Expected return %": [DEFAULT_CUSTOM_RETURNS.get(a, 0.08) * 100 for a in available_assets],
                "Max weight %": [w_current.get(a, 0.0) * 100 if a in fixed_weights else 35.0 for a in available_assets],
                "Fixed": [a in fixed_weights for a in available_assets],
            })
            edited = st.data_editor(
                editor,
                hide_index=True,
                use_container_width=True,
                disabled=["Asset", "Current weight %", "Fixed"],
                column_config={
                    "Expected return %": st.column_config.NumberColumn(min_value=-100.0, max_value=200.0, step=0.5),
                    "Max weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                },
                key="optimizer_editor_v2",
            )
            custom_mu = pd.Series(edited["Expected return %"].to_numpy() / 100.0, index=edited["Asset"])
            mu_series = expected_return_vector(return_model, base_returns, available_assets, custom_mu, rf, market_return).fillna(custom_mu)
            mu = mu_series.values.astype(float)
            cov = cov_ann.loc[available_assets, available_assets].values.astype(float)
            w0 = w_current.reindex(available_assets).fillna(0.0).values

            bounds = []
            for _, row in edited.iterrows():
                asset = row["Asset"]
                if asset in fixed_weights:
                    fixed = float(w_current[asset])
                    bounds.append((fixed, fixed))
                else:
                    bounds.append((0.0, float(row["Max weight %"]) / 100.0))

            current_ret, current_vol, current_sharpe = port_metrics(w0, mu, cov, rf)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected return", f"{current_ret:.1%}")
            c2.metric("Volatility", f"{current_vol:.1%}")
            c3.metric("Excess return", f"{current_ret - rf:.1%}")
            c4.metric("Sharpe", f"{current_sharpe:.2f}")

            with st.expander("How the Sharpe optimizer works", expanded=True):
                st.markdown(
                    f"""
**Current Sharpe** = (expected return − risk-free rate) ÷ portfolio volatility

= ({current_ret:.2%} − {rf:.2%}) ÷ {current_vol:.2%} = **{current_sharpe:.2f}**

The optimizer chooses weights that maximize **expected excess return per unit of volatility** while forcing all weights to sum to 100% and respecting your fixed crypto/MSTR positions and max-weight limits. It is not discovering a true future portfolio. It is finding the mathematically best portfolio **conditional on your expected-return assumptions and the historical covariance matrix**.

A higher Sharpe can come from more expected return, less covariance-driven volatility, or both. Because expected returns are uncertain, the most useful interpretation is often the *direction* of the recommended shifts rather than treating the exact optimized weight as precise.
"""
                )

            st.markdown("### Covariance and 1% risk-shift heatmaps")
            st.plotly_chart(heatmap(cov_ann.loc[available_assets, available_assets], "Annualized covariance matrix", ".3f", True, "Covariance"), use_container_width=True)
            shift = exact_shift_heatmap(w_current.reindex(available_assets), cov_ann.loc[available_assets, available_assets], 0.01)
            st.plotly_chart(heatmap(shift, "Change in annualized portfolio volatility from moving 1% of capital", ".0f", True, "Vol change (bp)"), use_container_width=True)
            st.caption("Read each cell as: move 1% FROM the row asset TO the column asset. Negative values reduce modeled annualized volatility; positive values increase it. Blank cells mean the source weight is below 1%.")

            result = optimize_weights(mu, cov, rf, bounds, w0)
            st.markdown("### Maximum-Sharpe portfolio under your fixed-position constraints")
            if result.success:
                w_opt = result.x
                opt_ret, opt_vol, opt_sharpe = port_metrics(w_opt, mu, cov, rf)
                compare = pd.DataFrame({"Asset": available_assets, "Current": w0, "Max-Sharpe": w_opt})
                compare["Change"] = compare["Max-Sharpe"] - compare["Current"]
                st.dataframe(compare.style.format({"Current": "{:.1%}", "Max-Sharpe": "{:.1%}", "Change": "{:+.1%}"}), use_container_width=True)
                d1, d2, d3 = st.columns(3)
                d1.metric("Optimized return", f"{opt_ret:.1%}", f"{opt_ret-current_ret:+.1%}")
                d2.metric("Optimized volatility", f"{opt_vol:.1%}", f"{opt_vol-current_vol:+.1%}")
                d3.metric("Optimized Sharpe", f"{opt_sharpe:.2f}", f"{opt_sharpe-current_sharpe:+.2f}")
            else:
                st.warning("No feasible maximum-Sharpe solution was found. The fixed positions and max weights may leave too little room for the remaining assets.")

        st.markdown("### SOXX put stress table from your Greeks")
        soxx_spot = float(latest["SOXX"])
        scenario = hedge_scenario_table(soxx_spot, float(put_delta), float(put_gamma), float(put_theta), float(put_vega), int(scenario_days), float(vol_shock))
        st.dataframe(
            scenario.style.format({
                "SOXX move": "{:+.0%}", "SOXX price": "${:,.2f}", "IV shock (pts)": "{:+.1f}",
                "Delta P&L": "${:+,.0f}", "Gamma P&L": "${:+,.0f}", "Vega P&L": "${:+,.0f}",
                "Theta P&L": "${:+,.0f}", "Estimated put P&L": "${:+,.0f}",
            }),
            use_container_width=True,
        )
        st.caption("This is a local Greek approximation. It becomes less accurate for very large moves because delta, gamma, vega and implied volatility themselves change as the market moves.")

with tab4:
    st.markdown("### Derisk crypto by dilution rather than forced selling")
    st.write("Your crypto quantities are fixed in the model. This planner assumes you direct new savings to the core index/equity portfolio while allowing crypto to remain untouched.")

    start = today - dt.timedelta(days=30)
    tks = tuple(list(CRYPTO_TICKERS.values()) + ["MSTR"])
    recent = adjusted_close(download_history(tks, start, end_date))
    if any(t not in recent.columns for t in tks):
        st.error("Unable to load all current crypto/MSTR prices for the transition plan.")
    else:
        latest = recent.ffill().iloc[-1]
        crypto_values = {c: CRYPTO_QTY[c] * float(latest[t]) for c, t in CRYPTO_TICKERS.items()}
        crypto_start = float(sum(crypto_values.values()))
        mstr_shares = st.number_input("Fixed MSTR shares for transition plan", min_value=0, value=200, step=100, key="trans_mstr")
        mstr_start = float(mstr_shares) * float(latest["MSTR"])
        total_now = st.number_input("Current total portfolio value ($)", min_value=1.0, value=250000.0, step=10000.0, format="%.0f", key="trans_total")
        core_start = max(float(total_now) - crypto_start - mstr_start, 0.0)

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            annual_contribution = st.number_input("Annual new money to core ($)", min_value=0.0, value=30000.0, step=5000.0, format="%.0f")
            st.caption("All new contributions are directed to the core portfolio. This is what mechanically dilutes crypto weight without selling it.")
        with t2:
            horizon = st.number_input("Years", min_value=1, max_value=30, value=10, step=1)
            st.caption("Projection horizon. The model compounds each sleeve once per year for simplicity.")
        with t3:
            core_return = st.number_input("Core expected return", min_value=-0.20, max_value=0.30, value=0.075, step=0.005, format="%.3f")
            st.caption("Annual return assumption for the diversified core/index portfolio.")
        with t4:
            crypto_return = st.number_input("Crypto expected return", min_value=-0.50, max_value=1.00, value=0.10, step=0.01, format="%.2f")
            st.caption("Annual return assumption for the existing fixed crypto inventory. Higher crypto returns slow the dilution process.")
        mstr_return = st.number_input("MSTR covered-call expected return", min_value=-0.50, max_value=1.00, value=0.12, step=0.01, format="%.2f")
        st.caption("Expected annual return on the fixed MSTR covered-call sleeve, including your option-income strategy.")

        projection = crypto_transition_projection(crypto_start, core_start, mstr_start, float(annual_contribution), int(horizon), float(core_return), float(crypto_return), float(mstr_return))
        p1, p2, p3 = st.columns(3)
        p1.metric("Crypto value now", f"${crypto_start:,.0f}")
        p2.metric("Crypto weight now", f"{projection.iloc[0]['Crypto weight']:.1%}")
        p3.metric(f"Crypto weight in {int(horizon)}Y", f"{projection.iloc[-1]['Crypto weight']:.1%}")

        weights_long = projection.melt(id_vars="Year", value_vars=["Core weight", "Crypto weight", "MSTR weight"], var_name="Sleeve", value_name="Weight")
        fig = px.line(weights_long, x="Year", y="Weight", color="Sleeve", markers=True, title="Portfolio mix over time without selling crypto")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            projection[["Year", "Core indexes", "Crypto", "MSTR CC", "Total", "Core weight", "Crypto weight", "MSTR weight"]].style.format({
                "Core indexes": "${:,.0f}", "Crypto": "${:,.0f}", "MSTR CC": "${:,.0f}", "Total": "${:,.0f}",
                "Core weight": "{:.1%}", "Crypto weight": "{:.1%}", "MSTR weight": "{:.1%}",
            }),
            use_container_width=True,
        )

        st.markdown("#### Where new money goes")
        core_weights = pd.Series(CORE_TARGET) / sum(CORE_TARGET.values())
        contribution_table = pd.DataFrame({
            "Asset": core_weights.index,
            "Target share of new money": core_weights.values,
            "Annual dollars": core_weights.values * float(annual_contribution),
        })
        st.dataframe(contribution_table.style.format({"Target share of new money": "{:.1%}", "Annual dollars": "${:,.0f}"}), use_container_width=True)
        st.caption("This uses your existing core target proportions as the destination for new money. Crypto quantities remain unchanged; its portfolio weight falls only if the rest of the portfolio grows faster through contributions and/or returns.")