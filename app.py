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

st.set_page_config(page_title="Portfolio Dashboard", page_icon="📈", layout="wide")

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

HEDGE = "SOXX PUT"
OPTIMIZER_ASSETS = list(PORTFOLIO.keys()) + [HEDGE]

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
    "SOXX": "Semiconductors",
    HEDGE: "Long SOXX Put Hedge",
}

DEFAULT_CUSTOM_RETURNS: Dict[str, float] = {
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
    HEDGE: -0.200,
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
    styles = {
        "DMA 20": ("#fbc02d", 1.3),
        "DMA 50": ("#42a5f5", 1.4),
        "DMA 100": ("#7e57c2", 1.4),
        "DMA 200": ("#ef6c00", 1.8),
        "EMA 8": ("#66bb6a", 1.0),
        "EMA 21": ("#ec407a", 1.0),
        "1Y AVWAP": ("#ffffff", 2.0),
    }
    for col, (color, width) in styles.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=color, width=width)))
    fig.update_layout(
        title=f"{ticker} · {DISPLAY_NAMES.get(ticker, ticker)}",
        template="plotly_dark",
        height=680,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
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
        title=title,
        template="plotly_dark",
        height=max(520, 48 * len(matrix.columns)),
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar=dict(title="ρ"),
    )
    return fig


def synthetic_soxx_put_returns(soxx_returns: pd.Series, downside_beta: float, annual_carry: float, convexity: float) -> pd.Series:
    """Simplified daily proxy for a rolling long-put sleeve."""
    r = soxx_returns.fillna(0.0)
    downside = np.minimum(r, 0.0)
    proxy = -downside_beta * r + convexity * downside.pow(2) - annual_carry / 252.0
    return proxy.rename(HEDGE)


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
    low = max(float(np.min(mu)), 0.0)
    high = float(np.max(mu))
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
st.caption("Live historical data via Yahoo Finance. Includes technical indicators, covariance, volatility, beta, risk contribution, downside-regime analysis, Monte Carlo ranges, and portfolio optimization.")

tab1, tab2, tab3 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer"])
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
        lookback = st.selectbox("Lookback", ["1Y", "3Y", "5Y"], index=2, key="risk_lookback")
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
            st.dataframe(summary.style.format({"Target weight": "{:.0%}", "Annualized vol": "{:.1%}", "Beta vs SPY": "{:.2f}", "Beta vs QQQ": "{:.2f}", "Risk contribution": "{:.1%}", "Available obs": "{:,.0f}"}, na_rep="—"), use_container_width=True)

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

            st.caption("Methodology: daily adjusted-close percentage returns; annualized volatility uses √252; portfolio volatility uses wᵀΣw; risk contribution uses each holding's component contribution to total portfolio variance. STRK and STRF have shorter histories, which shortens common-history calculations.")

with tab3:
    st.markdown("### Portfolio construction assumptions")
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        opt_lookback = st.selectbox("Covariance lookback", ["1Y", "3Y", "5Y"], index=2)
    with o2:
        return_model = st.selectbox("Expected-return model", ["Custom assumptions", "Historical geometric", "CAPM / market-implied"], index=0)
    with o3:
        rf = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    with o4:
        market_return = st.number_input("Market expected return", min_value=-0.10, max_value=0.30, value=0.085, step=0.005, format="%.3f")

    st.markdown("#### SOXX long-put proxy")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        put_downside_beta = st.number_input("Downside sensitivity", min_value=0.5, max_value=10.0, value=3.0, step=0.25)
    with h2:
        put_carry = st.number_input("Annual premium/carry drag", min_value=0.0, max_value=1.0, value=0.25, step=0.025, format="%.3f")
    with h3:
        put_convexity = st.number_input("Convexity factor", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with h4:
        max_put_weight = st.number_input("Max put weight", min_value=0.0, max_value=0.50, value=0.15, step=0.01, format="%.2f")

    years = {"1Y": 1, "3Y": 3, "5Y": 5}[opt_lookback]
    start = today - dt.timedelta(days=int(365.25 * years) + 30)
    needed = tuple(dict.fromkeys(list(PORTFOLIO.keys()) + ["SPY", "SOXX"]))
    with st.spinner("Loading optimizer history..."):
        opt_raw = download_history(needed, start, end_date)
    opt_prices = adjusted_close(opt_raw)

    if "SOXX" not in opt_prices.columns:
        st.error("SOXX history is required to build the put-hedge proxy.")
    else:
        base_returns = opt_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        base_returns[HEDGE] = synthetic_soxx_put_returns(base_returns["SOXX"], put_downside_beta, put_carry, put_convexity)
        available_assets = [a for a in OPTIMIZER_ASSETS if a in base_returns.columns]
        missing_assets = [a for a in OPTIMIZER_ASSETS if a not in available_assets]
        if missing_assets:
            st.warning("Optimizer is missing usable history for: " + ", ".join(missing_assets))

        if len(available_assets) < 2:
            st.error("Not enough asset history to run the optimizer.")
        else:
            st.markdown("#### Editable portfolio")
            initial_weights = {**PORTFOLIO, HEDGE: 0.0}
            editor = pd.DataFrame({"Asset": available_assets, "Weight %": [initial_weights.get(a, 0.0) * 100 for a in available_assets], "Expected return %": [DEFAULT_CUSTOM_RETURNS.get(a, 0.07) * 100 for a in available_assets], "Max weight %": [max_put_weight * 100 if a == HEDGE else 35.0 for a in available_assets]})
            edited = st.data_editor(
                editor,
                hide_index=True,
                use_container_width=True,
                disabled=["Asset"],
                column_config={
                    "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                    "Expected return %": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=0.5),
                    "Max weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                },
                key="optimizer_editor",
            )

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
                    bounds = [(0.0, float(max_w.get(a, 0.35))) for a in available_assets]

                    current_ret, current_vol, current_sharpe = port_metrics(w0, mu, cov, rf)
                    p_marginal = cov @ w0
                    p_var = float(w0 @ cov @ w0)
                    risk_contrib = (w0 * p_marginal / p_var) if p_var > 0 else np.repeat(np.nan, len(w0))

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Expected annual return", f"{current_ret:.1%}")
                    m2.metric("Annualized volatility", f"{current_vol:.1%}")
                    m3.metric("Sharpe ratio", f"{current_sharpe:.2f}" if np.isfinite(current_sharpe) else "—")
                    m4.metric("1Y ~95% range", f"{current_ret - 1.96 * current_vol:.1%} to {current_ret + 1.96 * current_vol:.1%}")

                    alloc_risk = pd.DataFrame({"Asset": available_assets, "Portfolio weight": w0, "Risk contribution": risk_contrib, "Expected return": mu_series.reindex(available_assets).values, "Annualized vol": np.sqrt(np.diag(cov))})
                    long = alloc_risk.melt(id_vars="Asset", value_vars=["Portfolio weight", "Risk contribution"], var_name="Measure", value_name="Value")
                    fig_alloc = px.bar(long, x="Asset", y="Value", color="Measure", barmode="group", title="Capital weight vs contribution to portfolio risk")
                    fig_alloc.update_yaxes(tickformat=".0%")
                    fig_alloc.update_layout(template="plotly_dark", height=480)
                    st.plotly_chart(fig_alloc, use_container_width=True)

                    st.markdown("### Probabilistic return range")
                    sims = monte_carlo_portfolio(current_ret, current_vol, [1, 3, 5, 10], n_sims=10000)
                    st.dataframe(sims.style.format("{:.1%}"), use_container_width=True)
                    st.caption("Monte Carlo ranges use a constant expected return and volatility with lognormal compounding. They are scenario ranges, not forecasts or guarantees.")

                    st.markdown("### Optimize weights")
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        objective = st.selectbox("Objective", ["Maximum Sharpe", "Minimum volatility", "Minimum volatility for target return", "Maximum return under volatility cap"])
                    with q2:
                        target_return = st.number_input("Target annual return", min_value=-0.20, max_value=0.50, value=0.08, step=0.005, format="%.3f")
                    with q3:
                        vol_cap = st.number_input("Volatility cap", min_value=0.01, max_value=1.00, value=0.15, step=0.005, format="%.3f")

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

                    st.info("SOXX PUT is a simplified rolling-hedge proxy, not a live option valuation. Its return stream is driven by user-set downside sensitivity, annual carry drag, and convexity. Actual put returns depend on strike, tenor, implied volatility, skew, path, and roll timing.")
