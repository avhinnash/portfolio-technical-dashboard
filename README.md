# Portfolio Technical & Risk Dashboard

A Streamlit dashboard for monitoring the technical setup and portfolio-level risk characteristics of the selected investment portfolio.

## Target portfolio

| Ticker | Exposure | Weight |
| --- | --- | ---: |
| IXG | Global Financials | 20% |
| IFRA | U.S. Infrastructure | 14% |
| IGV | Software | 10% |
| XAR | Aerospace & Defense | 10% |
| XLP | Consumer Staples | 10% |
| LLY | Eli Lilly | 5% |
| INDA | India | 5% |
| EMXC | Emerging Markets ex-China | 3% |
| EWZ | Brazil | 3% |
| STRK | Strategy Preferred | 10% |
| STRF | Strategy Perpetual Preferred | 10% |

Total target weight: 100%.

## Technical Charts

The Technical Charts tab provides a one-year view of daily price action for each portfolio holding, including:

- Daily candlesticks
- 20-day simple moving average
- 50-day simple moving average
- 100-day simple moving average
- 200-day simple moving average
- 8-day exponential moving average
- 21-day exponential moving average
- One-year anchored VWAP
- Current price and daily percentage change
- Distance from the 20 DMA, 50 DMA, 200 DMA, and anchored VWAP

### Anchored VWAP definition

The dashboard anchors VWAP to the first trading session in the displayed one-year window.

Typical Price = (High + Low + Close) / 3

AVWAP = cumulative(Typical Price × Volume) / cumulative(Volume)

This is different from an intraday VWAP or a rolling 252-session VWAP.

## Portfolio Risk

The Portfolio Risk tab treats each ETF or security as its own portfolio exposure rather than decomposing every ETF into its underlying stocks.

The analysis supports selectable 1-year, 3-year, and 5-year lookback periods.

### Risk snapshot

For every selected holding, the dashboard calculates:

- Target portfolio weight
- Annualized historical volatility
- Beta versus SPY
- Beta versus QQQ
- Contribution to total portfolio variance
- Number of available return observations

### Correlation matrix

The dashboard calculates pairwise Pearson correlations using daily adjusted-close returns.

Pairwise calculation allows securities with shorter histories to remain in the matrix without unnecessarily shortening every other correlation pair.

### Portfolio volatility

Portfolio volatility is calculated from the annualized covariance matrix and target portfolio weights:

Portfolio Variance = wᵀΣw

Portfolio Volatility = √(wᵀΣw)

When only a subset of holdings is selected, the selected weights are normalized to 100% for the portfolio volatility and risk-contribution calculations.

### Risk contribution

The dashboard estimates each holding's component contribution to total portfolio variance. This is intended to show which positions actually drive portfolio risk rather than simply which positions have the largest nominal weights.

Risk contributions sum to approximately 100% of portfolio variance.

### Stress correlation

A separate downside-regime matrix recalculates correlations using only weak S&P 500 trading days.

The user can analyze correlations during the:

- Worst 10% of SPY trading days
- Worst 20% of SPY trading days
- Worst 25% of SPY trading days

This is designed to test whether apparent diversification persists during broad equity-market selloffs, when correlations often increase.

### Covariance matrix

The annualized covariance matrix is available in an expandable section for deeper portfolio analysis.

## STRK and STRF history

STRK and STRF have materially shorter trading histories than most of the other portfolio holdings. Their inclusion therefore shortens the common-history sample used for full-portfolio covariance, volatility, and risk-contribution calculations.

Pairwise correlation calculations are less affected because each pair uses its own overlapping observations.

The dashboard intentionally uses actual STRK and STRF trading history rather than presenting a synthetic proxy as historical performance.

## Methodology

- Market data: Yahoo Finance through `yfinance`
- Return series: daily adjusted-close percentage returns
- Volatility: daily standard deviation × √252
- Correlation: Pearson correlation of daily returns
- Covariance: daily return covariance × 252
- Beta: covariance with benchmark / benchmark variance
- Portfolio variance: wᵀΣw
- Data cache: 15 minutes while the application is running

SPY and QQQ are downloaded as benchmarks for risk analysis but are not portfolio holdings.

## Run locally

Python 3.10 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

The application should then open in your browser, normally at `http://localhost:8501`.

## Streamlit Community Cloud

The repository can be deployed directly through Streamlit Community Cloud using:

- Repository: `avhinnash/portfolio-technical-dashboard`
- Branch: `main`
- Main file: `app.py`

Changes committed to the connected GitHub repository should trigger a new Streamlit deployment.

## Important data note

Yahoo Finance data may be delayed and is intended here for portfolio analysis rather than order execution. Historical statistics are backward-looking and can change materially across market regimes.
