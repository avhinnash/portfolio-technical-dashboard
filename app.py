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

PORTFOLIO: Dict[str, float] = {
    "IXG": 0.20, "IFRA": 0.14, "IGV": 0.10, "XAR": 0.10, "XLP": 0.10,
    "LLY": 0.05, "INDA": 0.05, "EMXC": 0.03, "EWZ": 0.03, "STRK": 0.10, "STRF": 0.10,
}
HEDGE = "SOXX PUT"
MSTR_CC = "MSTR CC"
OPTIMIZER_ASSETS = list(PORTFOLIO.keys()) + [MSTR_CC, HEDGE]

DISPLAY_NAMES = {
    "IXG": "Global Financials", "IFRA": "U.S. Infrastructure", "IGV": "Software",
    "XAR": "Aerospace & Defense", "XLP": "Consumer Staples", "LLY": "Eli Lilly",
    "INDA": "India", "EMXC": "Emerging Markets ex-China", "EWZ": "Brazil",
    "STRK": "Strategy Preferred", "STRF": "Strategy Perpetual Preferred",
    "MSTR": "Strategy common stock", "SPY": "S&P 500", "QQQ": "Nasdaq-100",
    "SOXX": "Semiconductors", HEDGE: "Long SOXX Put Hedge",
    MSTR_CC: "MSTR covered-call sleeve",
}

DEFAULT_CUSTOM_RETURNS = {
    "IXG": 0.075, "IFRA": 0.075, "IGV": 0.085, "XAR": 0.080, "XLP": 0.060,
    "LLY": 0.080, "INDA": 0.090, "EMXC": 0.085, "EWZ": 0.080,
    "STRK": 0.085, "STRF": 0.085, MSTR_CC: 0.12, HEDGE: -0.20,
}
MA_WINDOWS = [20, 50, 100, 200]
EMA_WINDOWS = [8, 21]


@st.cache_data(ttl=900, show_spinner=False)
def download_history(tickers: Tuple[str, ...], start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        list(tickers), start=start, end=end, interval="1d", auto_adjust=False,
        actions=False, group_by="ticker", threads=True, progress=False,
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
        out[c] = np.nan if len(pair) < 30 or pair.iloc[:, 1].var() == 0 else pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
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
    pct_contrib = (w.values * marginal / variance) if variance > 0 else np.repeat(np.nan, len(cols))
    return vol, pd.Series(pct_contrib, index=cols), cov_ann, len(clean)


def stress_corr(returns: pd.DataFrame, benchmark: str, q: float):
    b = returns[benchmark].dropna()
    threshold = b.quantile(q)
    dates = b[b <= threshold].index
    stressed = returns.loc[returns.index.intersection(dates)]
    return stressed.corr(min_periods=10), len(stressed), threshold


def heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
    fig.update_layout(title=title, template="plotly_dark", height=max(520, 48 * len(matrix.columns)), margin=dict(l=10, r=10, t=60, b=10), coloraxis_colorbar=dict(title="ρ"))
    return fig


def synthetic_soxx_put_returns(soxx_returns: pd.Series, annual_carry: float, convexity: float) -> pd.Series:
    """One-unit short-delta SOXX hedge proxy. Position size is set separately from entered short delta."""
    r = soxx_returns.fillna(0.0)
    downside = np.minimum(r, 0.0)
    proxy = -r + convexity * downside.pow(2) - annual_carry / 252.0
    return proxy.rename(HEDGE)


def covered_call_terms(delta: float, dte: int, iv: float, rf: float) -> tuple[float, float, float]:
    T = max(dte, 1) / 365.0
    sigma = max(iv, 1e-6)
    d1 = norm.ppf(np.clip(delta, 0.01, 0.99))
    strike_ratio = np.exp((rf + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    premium_ratio = norm.cdf(d1) - strike_ratio * np.exp(-rf * T) * norm.cdf(d2)
    annualized_premium = premium_ratio * 365.0 / max(dte, 1)
    return float(strike_ratio), float(max(premium_ratio, 0.0)), float(max(annualized_premium, 0.0))


def synthetic_mstr_cc_returns(mstr_returns: pd.Series, call_delta: float, dte: int, iv: float, rf: float, income_capture: float) -> tuple[pd.Series, float, float]:
    _, premium_ratio, annualized_premium = covered_call_terms(call_delta, dte, iv, rf)
    r = mstr_returns.fillna(0.0)
    downside = np.minimum(r, 0.0)
    upside = np.maximum(r, 0.0)
    daily_income = annualized_premium * np.clip(income_capture, 0.0, 1.0) / 252.0
    proxy = downside + (1.0 - call_delta) * upside + daily_income
    return proxy.rename(MSTR_CC), premium_ratio, annualized_premium


def geometric_annual_return(r: pd.Series) -> float:
    x = r.dropna()
    if len(x) < 30:
        return np.nan
    gross = (1 + x).clip(lower=1e-8).prod()
    years = len(x) / 252.0
    return gross ** (1 / years) - 1 if years > 0 else np.nan


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
    var = float(weights @ cov @ weights)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def optimize_weights(objective: str, mu: np.ndarray, cov: np.ndarray, rf: float, bounds: list[tuple[float, float]], x0: np.ndarray, target_return: float | None = None, vol_cap: float | None = None):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if objective == "Minimum volatility":
        fun = lambda w: float(w @ cov @ w)
    elif objective == "Maximum Sharpe":
        def fun(w):
            ret, vol, _ = port_metrics(w, mu, cov, rf)
            return -((ret - rf) / vol) if vol > 0 else 1e6
    elif objective == "Minimum volatility for target return":
        target = float(target_return or 0.0)
        constraints.append({"type": "ineq", "fun": lambda w: float(w @ mu) - target})
        fun = lambda w: float(w @ cov @ w)
    else:
        cap = float(vol_cap or 0.20)
        constraints.append({"type": "ineq", "fun": lambda w: cap**2 - float(w @ cov @ w)})
        fun = lambda w: -float(w @ mu)
    return minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, bounds: list[tuple[float, float]], x0: np.ndarray) -> pd.DataFrame:
    low, high = max(float(np.min(mu)), 0.0), float(np.max(mu))
    rows = []
    for target in np.linspace(low, high, 30):
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        res = minimize(lambda w: float(w @ cov @ w), x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            ret, vol, _ = port_metrics(res.x, mu, cov, 0.0)
            rows.append({"Return": ret, "Volatility": vol})
    return pd.DataFrame(rows).drop_duplicates().sort_values("Volatility") if rows else pd.DataFrame()


def monte_carlo_portfolio(mu: float, vol: float, horizons: list[int], n_sims: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for years in horizons:
        z = rng.standard_normal(n_sims)
        terminal = np.exp((mu - 0.5 * vol**2) * years + vol * np.sqrt(years) * z) - 1.0
        qs = np.quantile(terminal, [0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append({"Horizon": f"{years}Y", "5th": qs[0], "25th": qs[1], "Median": qs[2], "75th": qs[3], "95th": qs[4], "P(Loss)": float(np.mean(terminal < 0))})
    return pd.DataFrame(rows).set_index("Horizon")


st.title("Portfolio Technical, Risk & Optimization Dashboard")
st.caption("Live historical data via Yahoo Finance. Includes technical indicators, covariance, volatility, beta, risk contribution, downside-regime analysis, Monte Carlo ranges, covered-call income modeling, and portfolio optimization.")

tab1, tab2, tab3 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer"])
today = dt.date.today()
end_date = today + dt.timedelta(days=1)

with tab1:
    ticker = st.selectbox("Ticker", list(PORTFOLIO.keys()) + ["MSTR"], index=0)
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

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        lookback = st.selectbox("Lookback", ["1Y", "3Y", "5Y"], index=2, key="risk_lookback")
    with c2:
        stress_pct = st.selectbox("Stress sample", [10, 20, 25], index=1, format_func=lambda x: f"Worst {x}% of SPY days")
    with c3:
        selected = st.multiselect("Holdings", options=list(PORTFOLIO.keys()) + ["MSTR"], default=list(PORTFOLIO.keys()))
    if selected:
        years = {"1Y": 1, "3Y": 3, "5Y": 5}[lookback]
        start = today - dt.timedelta(days=int(365.25 * years) + 30)
        tickers = tuple(dict.fromkeys(selected + ["SPY", "QQQ"]))
        raw = download_history(tickers, start, end_date)
        prices = adjusted_close(raw)
        available = [c for c in selected if c in prices.columns]
        if available:
            returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            corr = returns[available].corr(min_periods=30)
            vols = annualized_vol(returns[available])
            beta_spy = beta_to(returns[available + ["SPY"]].dropna(how="all"), "SPY").reindex(available)
            beta_qqq = beta_to(returns[available + ["QQQ"]].dropna(how="all"), "QQQ").reindex(available)
            weights = pd.Series(PORTFOLIO).reindex(available).dropna()
            if not weights.empty:
                pvol, rc, cov_ann, common_obs = portfolio_stats(returns, weights)
            else:
                pvol, rc, cov_ann, common_obs = np.nan, pd.Series(dtype=float), pd.DataFrame(), 0
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Core portfolio annualized vol", f"{pvol:.1%}" if pd.notna(pvol) else "—")
            a2.metric("Common observations", f"{common_obs:,}")
            a3.metric("Selected core target weight", f"{weights.sum():.0%}" if not weights.empty else "—")
            a4.metric("Lookback", lookback)
            summary = pd.DataFrame(index=available)
            summary["Target weight"] = pd.Series(PORTFOLIO).reindex(available)
            summary["Annualized vol"] = vols.reindex(available)
            summary["Beta vs SPY"] = beta_spy
            summary["Beta vs QQQ"] = beta_qqq
            summary["Risk contribution"] = rc.reindex(available)
            summary["Available obs"] = [int(returns[c].notna().sum()) for c in available]
            st.dataframe(summary.style.format({"Target weight": "{:.0%}", "Annualized vol": "{:.1%}", "Beta vs SPY": "{:.2f}", "Beta vs QQQ": "{:.2f}", "Risk contribution": "{:.1%}", "Available obs": "{:,.0f}"}, na_rep="—"), use_container_width=True)
            st.plotly_chart(heatmap(corr, f"{lookback} pairwise daily-return correlation"), use_container_width=True)
            if "SPY" in returns.columns:
                stress_matrix, stress_obs, threshold = stress_corr(returns[available + ["SPY"]], "SPY", stress_pct / 100.0)
                st.plotly_chart(heatmap(stress_matrix.loc[available, available], f"Stress correlation on SPY days ≤ {threshold:.2%} ({stress_obs} sessions)"), use_container_width=True)
        else:
            st.error("No usable holdings data returned.")

with tab3:
    st.markdown("### Portfolio construction assumptions")
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        opt_lookback = st.selectbox("Covariance lookback", ["1Y", "3Y", "5Y"], index=2)
        st.caption("History window used to estimate volatility, correlation and covariance. Longer windows are steadier but can underweight recent regime changes.")
    with o2:
        return_model = st.selectbox("Expected-return model", ["Custom assumptions", "Historical geometric", "CAPM / market-implied"], index=0)
        st.caption("Controls the return forecast used by the optimizer. Custom lets you supply your own forward view instead of extrapolating history.")
    with o3:
        rf = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
        st.caption("Used in Sharpe ratios, CAPM expected returns and the covered-call option-pricing approximation.")
    with o4:
        market_return = st.number_input("Market expected return", min_value=-0.10, max_value=0.30, value=0.085, step=0.005, format="%.3f")
        st.caption("Forward expected return for SPY when the CAPM / market-implied model is selected.")

    st.markdown("#### MSTR covered-call sleeve")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        mstr_shares = st.number_input("MSTR shares", min_value=0, max_value=10000, value=200, step=100)
        st.caption("Number of MSTR shares you own. Every 100 shares supports one fully covered call contract.")
    with cc2:
        call_style = st.selectbox("Call target", ["35 delta", "ATM (~50 delta)"], index=0)
        st.caption("Higher short-call delta produces more premium and more upside truncation. ATM is more aggressive than 35 delta.")
    with cc3:
        call_dte = st.number_input("Call DTE", min_value=7, max_value=180, value=47, step=1)
        st.caption("Days to expiration at entry. Shorter cycles usually decay faster but require more frequent rolling and execution.")
    with cc4:
        mstr_iv = st.number_input("Assumed MSTR IV", min_value=0.10, max_value=3.00, value=0.80, step=0.05, format="%.2f")
        st.caption("Implied volatility assumption used to estimate call premium. Higher IV increases modeled premium and option value.")
    cc5, cc6, cc7, cc8 = st.columns(4)
    with cc5:
        income_capture = st.number_input("Premium income capture", min_value=0.0, max_value=1.0, value=0.70, step=0.05, format="%.2f")
        st.caption("Fraction of theoretical premium you expect to retain after rolls, buybacks, slippage and imperfect execution.")
    with cc6:
        total_portfolio_value = st.number_input("Total portfolio value ($)", min_value=1.0, value=250000.0, step=10000.0, format="%.0f")
        st.caption("Used to convert fixed share positions and SOXX delta-equivalent exposure into portfolio weights.")
    with cc7:
        max_mstr_cc_weight = st.number_input("Max MSTR CC weight", min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f")
        st.caption("Maximum allocation the optimizer may assign to the MSTR covered-call sleeve unless the current weight is locked.")
    with cc8:
        lock_mstr_weight = st.checkbox("Lock current MSTR weight", value=False)
        st.caption("When checked, optimization cannot change the portfolio weight implied by your entered MSTR share count.")

    call_delta = 0.35 if call_style.startswith("35") else 0.50
    contracts = int(mstr_shares // 100)

    st.markdown("#### SOXX long-put hedge")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        put_short_delta_shares = st.number_input("Net short delta (SOXX shares)", min_value=0.0, max_value=10000.0, value=0.0, step=10.0, format="%.0f")
        st.caption("Your total current put delta expressed as SOXX-share equivalents. Example: two puts at -0.40 delta = 2 × 100 × 0.40 = 80 short-delta shares. Enter 80, not -80.")
    with h2:
        put_carry = st.number_input("Annual premium/carry drag", min_value=0.0, max_value=1.0, value=0.25, step=0.025, format="%.3f")
        st.caption("Estimated annual cost of maintaining the long-put hedge as a percentage of its delta-equivalent notional. Higher values reduce expected return.")
    with h3:
        put_convexity = st.number_input("Convexity factor", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        st.caption("Adds gamma-like acceleration when SOXX falls sharply. Higher values make the hedge respond more strongly in large downside moves.")
    with h4:
        max_put_weight = st.number_input("Max put weight", min_value=0.0, max_value=0.50, value=0.15, step=0.01, format="%.2f")
        st.caption("Maximum SOXX hedge allocation the optimizer is allowed to use. This is a risk limit, not the current hedge size.")

    years = {"1Y": 1, "3Y": 3, "5Y": 5}[opt_lookback]
    start = today - dt.timedelta(days=int(365.25 * years) + 30)
    needed = tuple(dict.fromkeys(list(PORTFOLIO.keys()) + ["SPY", "SOXX", "MSTR"]))
    with st.spinner("Loading optimizer history..."):
        opt_raw = download_history(needed, start, end_date)
    opt_prices = adjusted_close(opt_raw)

    if "SOXX" not in opt_prices.columns or "MSTR" not in opt_prices.columns:
        st.error("SOXX and MSTR history are required to build the option-strategy proxies.")
    else:
        base_returns = opt_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        base_returns[HEDGE] = synthetic_soxx_put_returns(base_returns["SOXX"], put_carry, put_convexity)
        cc_returns, cc_premium_yield, cc_annualized_premium = synthetic_mstr_cc_returns(base_returns["MSTR"], call_delta, int(call_dte), float(mstr_iv), rf, income_capture)
        base_returns[MSTR_CC] = cc_returns

        latest_mstr = float(opt_prices["MSTR"].dropna().iloc[-1])
        latest_soxx = float(opt_prices["SOXX"].dropna().iloc[-1])
        mstr_market_value = float(mstr_shares) * latest_mstr
        current_mstr_weight = min(mstr_market_value / float(total_portfolio_value), 1.0) if total_portfolio_value > 0 else 0.0
        put_delta_notional = float(put_short_delta_shares) * latest_soxx
        current_put_weight = min(put_delta_notional / float(total_portfolio_value), 1.0) if total_portfolio_value > 0 else 0.0
        strike_ratio, _, _ = covered_call_terms(call_delta, int(call_dte), float(mstr_iv), rf)
        implied_strike = latest_mstr * strike_ratio

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("MSTR market value", f"${mstr_market_value:,.0f}")
        p2.metric("Covered calls", f"{contracts} contracts")
        p3.metric("Modeled call strike", f"${implied_strike:,.0f}")
        p4.metric("SOXX delta notional", f"${put_delta_notional:,.0f}")
        p5.metric("SOXX delta hedge", f"{current_put_weight:.1%} of portfolio")
        st.caption(f"The SOXX hedge starts at {current_put_weight:.1%} because {put_short_delta_shares:,.0f} short-delta shares × ${latest_soxx:,.2f} SOXX = ${put_delta_notional:,.0f} of delta-equivalent notional. This is separate from option premium paid and will change as put delta changes.")

        available_assets = [a for a in OPTIMIZER_ASSETS if a in base_returns.columns]
        if len(available_assets) < 2:
            st.error("Not enough asset history to run the optimizer.")
        else:
            st.markdown("#### Editable portfolio")
            remaining = max(1.0 - current_mstr_weight - current_put_weight, 0.0)
            initial_weights = {a: w * remaining for a, w in PORTFOLIO.items()}
            initial_weights[MSTR_CC] = current_mstr_weight
            initial_weights[HEDGE] = current_put_weight
            editor = pd.DataFrame({
                "Asset": available_assets,
                "Weight %": [initial_weights.get(a, 0.0) * 100 for a in available_assets],
                "Expected return %": [DEFAULT_CUSTOM_RETURNS.get(a, 0.07) * 100 for a in available_assets],
                "Max weight %": [max_put_weight * 100 if a == HEDGE else max_mstr_cc_weight * 100 if a == MSTR_CC else 35.0 for a in available_assets],
            })
            edited = st.data_editor(
                editor, hide_index=True, use_container_width=True, disabled=["Asset"],
                column_config={
                    "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, help="Current portfolio allocation. The model normalizes entered weights to 100%."),
                    "Expected return %": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=0.5, help="Forward annual return assumption used when Custom assumptions is selected."),
                    "Max weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, help="Upper bound the optimizer may assign to each sleeve."),
                }, key="optimizer_editor",
            )
            st.caption("Weight % is your current allocation. Expected return % is the forward assumption used by the optimizer. Max weight % is a hard allocation ceiling during optimization.")

            w_user = pd.Series(edited["Weight %"].to_numpy() / 100.0, index=edited["Asset"])
            custom_mu = pd.Series(edited["Expected return %"].to_numpy() / 100.0, index=edited["Asset"])
            max_w = pd.Series(edited["Max weight %"].to_numpy() / 100.0, index=edited["Asset"])

            if w_user.sum() <= 0:
                st.error("Enter at least one positive portfolio weight.")
            else:
                w_current = w_user / w_user.sum()
                st.caption(f"Entered weights sum to {w_user.sum():.1%}; calculations normalize them to 100%.")
                common = base_returns[available_assets].dropna()
                if len(common) < 30:
                    st.error("Not enough overlapping observations for covariance estimation. STRK/STRF short histories may be the limiting factor.")
                else:
                    cov_ann = common.cov() * 252
                    mu_series = expected_return_vector(return_model, base_returns, available_assets, custom_mu, rf, market_return).fillna(custom_mu)
                    mu = mu_series.values.astype(float)
                    cov = cov_ann.loc[available_assets, available_assets].values.astype(float)
                    w0 = w_current.reindex(available_assets).fillna(0.0).values
                    bounds = []
                    for a in available_assets:
                        if a == MSTR_CC and lock_mstr_weight:
                            fixed = float(w_current.get(MSTR_CC, current_mstr_weight))
                            bounds.append((fixed, fixed))
                        else:
                            bounds.append((0.0, float(max_w.get(a, 0.35))))

                    current_ret, current_vol, current_sharpe = port_metrics(w0, mu, cov, rf)
                    p_marginal = cov @ w0
                    p_var = float(w0 @ cov @ w0)
                    risk_contrib = (w0 * p_marginal / p_var) if p_var > 0 else np.repeat(np.nan, len(w0))

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Expected annual return", f"{current_ret:.1%}")
                    m2.metric("Annualized volatility", f"{current_vol:.1%}")
                    m3.metric("Sharpe ratio", f"{current_sharpe:.2f}" if np.isfinite(current_sharpe) else "—")
                    m4.metric("1Y ~95% range", f"{current_ret - 1.96 * current_vol:.1%} to {current_ret + 1.96 * current_vol:.1%}")

                    alloc_risk = pd.DataFrame({"Asset": available_assets, "Portfolio weight": w0, "Risk contribution": risk_contrib})
                    long = alloc_risk.melt(id_vars="Asset", value_vars=["Portfolio weight", "Risk contribution"], var_name="Measure", value_name="Value")
                    fig_alloc = px.bar(long, x="Asset", y="Value", color="Measure", barmode="group", title="Capital weight vs contribution to portfolio risk")
                    fig_alloc.update_yaxes(tickformat=".0%")
                    fig_alloc.update_layout(template="plotly_dark", height=480)
                    st.plotly_chart(fig_alloc, use_container_width=True)

                    st.markdown("### Probabilistic return range")
                    sims = monte_carlo_portfolio(current_ret, current_vol, [1, 3, 5, 10], n_sims=10000)
                    st.dataframe(sims.style.format("{:.1%}"), use_container_width=True)

                    st.markdown("### Optimize weights")
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        objective = st.selectbox("Objective", ["Maximum Sharpe", "Minimum volatility", "Minimum volatility for target return", "Maximum return under volatility cap"])
                        st.caption("Selects what the optimizer is trying to maximize or minimize while respecting your weight limits.")
                    with q2:
                        target_return = st.number_input("Target annual return", min_value=-0.20, max_value=0.50, value=0.08, step=0.005, format="%.3f")
                        st.caption("Used only for the target-return objective. The optimizer seeks the lowest volatility portfolio that meets this return.")
                    with q3:
                        vol_cap = st.number_input("Volatility cap", min_value=0.01, max_value=1.00, value=0.15, step=0.005, format="%.3f")
                        st.caption("Used only for the volatility-cap objective. The optimizer seeks the highest return while staying below this annualized volatility.")

                    result = optimize_weights(objective, mu, cov, rf, bounds, w0, target_return=target_return, vol_cap=vol_cap)
                    if not result.success:
                        st.warning("Optimizer could not find a feasible solution with the current assumptions and max-weight constraints. Try loosening the constraints.")
                    else:
                        w_opt = result.x
                        opt_ret, opt_vol, opt_sharpe = port_metrics(w_opt, mu, cov, rf)
                        compare = pd.DataFrame({"Asset": available_assets, "Current": w0, "Optimized": w_opt})
                        compare["Change"] = compare["Optimized"] - compare["Current"]
                        st.dataframe(compare.style.format({"Current": "{:.1%}", "Optimized": "{:.1%}", "Change": "{:+.1%}"}), use_container_width=True)
                        z1, z2, z3 = st.columns(3)
                        z1.metric("Optimized return", f"{opt_ret:.1%}", f"{opt_ret - current_ret:+.1%}")
                        z2.metric("Optimized volatility", f"{opt_vol:.1%}", f"{opt_vol - current_vol:+.1%}")
                        z3.metric("Optimized Sharpe", f"{opt_sharpe:.2f}" if np.isfinite(opt_sharpe) else "—", f"{opt_sharpe - current_sharpe:+.2f}" if np.isfinite(opt_sharpe) and np.isfinite(current_sharpe) else None)
                        frontier = efficient_frontier(mu, cov, bounds, w0)
                        if not frontier.empty:
                            fig_frontier = px.line(frontier, x="Volatility", y="Return", title="Efficient frontier")
                            fig_frontier.add_trace(go.Scatter(x=[current_vol], y=[current_ret], mode="markers+text", name="Current", text=["Current"], textposition="top center", marker=dict(size=12)))
                            fig_frontier.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode="markers+text", name="Optimized", text=["Optimized"], textposition="top center", marker=dict(size=12)))
                            fig_frontier.update_xaxes(tickformat=".0%")
                            fig_frontier.update_yaxes(tickformat=".0%")
                            fig_frontier.update_layout(template="plotly_dark", height=520)
                            st.plotly_chart(fig_frontier, use_container_width=True)
                        st.markdown("#### Optimized probabilistic range")
                        opt_sims = monte_carlo_portfolio(opt_ret, opt_vol, [1, 3, 5, 10], n_sims=10000)
                        st.dataframe(opt_sims.style.format("{:.1%}"), use_container_width=True)

                    with st.expander("Optimizer covariance matrix"):
                        st.dataframe(cov_ann.loc[available_assets, available_assets].style.format("{:.4f}"), use_container_width=True)

                    st.info("MSTR CC and SOXX PUT are strategy proxies, not live option valuations. The SOXX short-delta input represents current delta-equivalent exposure; actual put delta changes with SOXX price, time, implied volatility and gamma. The MSTR sleeve models full downside, reduced upside based on short-call delta, and Black-Scholes premium carry with an execution haircut.")
