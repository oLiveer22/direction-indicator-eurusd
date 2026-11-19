# EURUSD · Level 1 macro panel

## Deploy (free, ~10 min)
1. Create a free FRED API key: fred.stlouisfed.org → My Account → API Keys.
2. Create a GitHub repo and upload: app.py, macro_core.py, requirements.txt, forecasts.csv.
3. share.streamlit.io → New app → pick the repo, main file `app.py`.
4. App → Settings → Secrets: `FRED_API_KEY = "your_key"`.

## Daily use
- Sidebar: set Fed / ECB tone and duration by hand after each meeting or speech (bookmark the URL to keep it).
- "Data releases": charts of the last 12/24 months + Actual / Previous / Forecast tables.
- Forecast column: type it in the table, or add permanently to forecasts.csv (indicator,period,forecast).

## Known limits
- Consensus forecasts are not available from free APIs.
- FRED values are the current vintage (revised), not the first print.
- If a series fails, the "Data problems" expander at the bottom lists it.
