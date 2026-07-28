# Portfolio Technical Dashboard

A Streamlit dashboard for the selected portfolio:

- IXG 20%
- IFRA 14%
- IGV 10%
- XAR 10%
- XLP 10%
- LLY 5%
- INDA 5%
- EMXC 3%
- EWZ 3%
- STRK 10%
- STRF 10%

## Included indicators

- One year of daily candlesticks
- 20, 50, 100 and 200-day simple moving averages
- 8 and 21-day exponential moving averages
- One-year anchored VWAP
- Portfolio-wide technical snapshot
- Single-chart and multi-chart grid views

## Run it

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment if desired.
4. Install dependencies:

   pip install -r requirements.txt

5. Start the dashboard:

   streamlit run app.py

The browser should open automatically. Market data is retrieved from Yahoo Finance and cached for 15 minutes.

## Annual VWAP definition

The dashboard uses a one-year anchored VWAP, beginning with the first trading session in the displayed one-year window:

Typical Price = (High + Low + Close) / 3

AVWAP = cumulative(Typical Price × Volume) / cumulative(Volume)

This is different from a daily intraday VWAP and from a rolling 252-session VWAP.
