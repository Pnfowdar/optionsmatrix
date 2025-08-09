# Options Yield Matrix v7

A Streamlit web application designed to visualize and analyze options data, focusing on yield potential and probability of profit (POP) for option selling strategies (Puts and Calls). It fetches data from `marketdata.app` and `yfinance`, providing interactive filtering and display capabilities.

This version defaults to using the cost-effective **cached data feed** from `marketdata.app` for options chains, with an option to fetch **live data** on demand for individual tickers during market hours.

## Key Features

**Data Display & Visualization:**

*   **Options Matrix:** Displays key options data (Yield, Delta, POP, IV, Premium, etc.) in a pivot table format (Expiry Date vs. Strike Price).
*   **Configurable Display Value:** Choose which primary metric to display within the main matrix cells (e.g., Monthly Yield %, Actual Yield %, Delta, POP %, Premium $).
*   **Multi-Index Header:** Matrix columns show Strike Price and the percentage distance from the current spot price.
*   **Scoring & Color Coding:** Calculates a composite score based on POP and Monthly Yield (configurable bounds) and applies a Red-Yellow-Green color gradient to the matrix cells based on this score.
*   **Inspector Table:** Provides a detailed, sortable table view of filtered contracts, allowing inspection by specific Expiry Date or Strike Price.
*   **Market Status Indicator:** Shows whether the US market is currently open or closed.
*   **Data Refresh Timestamps:** Displays the last refresh time (converted to a local timezone).
*   **Feed Type Indicator:** Shows whether the displayed options data is from the "Cached" feed or "Live" feed (after manual refresh).

**Data Fetching & Management:**

*   **Dual Feed System:**
    *   Defaults to `marketdata.app` **Cached Feed** for options chain data (cost-effective).
    *   Provides a per-ticker "🔄 Live" button to fetch **Live Feed** (`sip`) data on demand during market hours (uses API credits).
*   **Spot Price:** Fetches current underlying stock price from `yfinance` with fallbacks.
*   **Ticker Validation:** Downloads and caches a list of valid Nasdaq tickers for the "Add Ticker" suggestion list. Updates periodically.
*   **Async Operations:** Utilizes `asyncio` and `httpx` for efficient, non-blocking API calls.
*   **Local Caching:** Implements local caching (`aiocache`) for API responses (expirations, chains, IV) to reduce redundant calls and save API credits. TTLs adjust based on market open/close status.
*   **Rate Limiting:** Uses `aiolimiter` to respect `marketdata.app` API rate limits.
*   **Session State Management:** Stores the primary options DataFrame for each ticker in Streamlit's session state, allowing updates (e.g., switching from cached to live) without refetching everything.

**User Interaction & Configuration:**

*   **Watchlist:**
    *   Persistent watchlist stored in `watchlist.json`.
    *   Add tickers via a searchable dropdown (populated from Nasdaq list) or manual text input.
    *   Remove tickers individually.
    *   Select tickers via checkboxes to display their matrices.
*   **Interactive Filtering:** Sidebar controls to filter options data based on:
    *   Days to Expiration (DTE) range.
    *   Minimum Probability of Profit (POP %).
    *   Absolute Delta range.
    *   Minimum Monthly Yield %.
    *   Strategy (PUT / CALL).
*   **Refresh Controls:**
    *   Separate button to refresh underlying spot prices (`yfinance`).
    *   Separate button to refresh base options data (clears `marketdata.app` local cache and session state DataFrames, refetches using **Cached** feed).
    *   Per-ticker button to refresh with **Live** options data (active only when market is open).
*   **Configuration File (`config.toml`):** Define default tickers, filter ranges, risk-free rate, and scoring bounds.
*   **API Token Security:** Uses Streamlit Secrets (`secrets.toml`) to manage the `marketdata.app` API token.

## Requirements

Uses Python 3. Key dependencies are listed in `requirements.txt`, including:

*   streamlit
*   pandas
*   httpx
*   yfinance
*   toml
*   aiolimiter
*   aiocache
*   nest_asyncio
*   scipy

## Configuration

1.  **`config.toml`:** Create this file in the same directory as `app.py`. Define parameters like:
    ```toml
    # Default tickers for watchlist if watchlist.json doesn't exist
    TICKERS = ["SPY", "QQQ", "IWM"]

    # Default Filter Values
    MIN_DTE = 2
    MAX_DTE = 60
    MIN_POP_FILTER = 65.0
    DELTA_RANGE_FILTER = [0.10, 0.35] # Min/Max absolute delta
    MIN_YIELD_FILTER = 3.0 # Minimum Monthly Yield %

    # Calculation Parameters
    TARGET_DELTA = 0.3 # Used potentially in future logic, not critical now
    RISK_FREE_RATE = 0.045 # Annual risk-free rate for POP calculation

    # Scoring Bounds (Used for color gradient)
    # Values outside these bounds will be clipped to 0 or 1 after normalization
    POP_SCORE_BOUNDS = [70.0, 85.0] # Min/Max POP % for scoring range
    YIELD_SCORE_BOUNDS = [3.0, 10.0] # Min/Max Monthly Yield % for scoring range
    ```

2.  **`secrets.toml`:** Create a file named `.streamlit/secrets.toml` in your project directory (create the `.streamlit` folder if it doesn't exist). Add your `marketdata.app` API token:
    ```toml
    [api]
    marketdata_token = "YOUR_MARKETDATA_APP_API_TOKEN"
    ```

## How to Run

1.  **Clone the repository (if applicable).**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create and populate `config.toml` and `.streamlit/secrets.toml`** as described above.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
6.  Open the provided URL in your web browser.

## Notes & Caveats

*   **API Costs:** Fetching **Live** options data using the per-ticker "🔄 Live" button consumes `marketdata.app` API credits based on the number of contracts returned per request. Use this feature judiciously. The default **Cached** feed is significantly cheaper for chain data.
*   **Market Hours:** The "🔄 Live" button is only enabled when the US market is detected as open (9:30 AM - 4:00 PM ET, Mon-Fri).
*   **Data Accuracy:** Data accuracy depends on the providers (`marketdata.app`, `yfinance`). Cached data may have some delay. POP calculation is a theoretical estimate based on the Black-Scholes model.
*   **Error Handling:** Basic error handling is included, but API issues or unexpected data formats might still cause errors. Check console logs for details.
*   **Watchlist File:** Ensure the application has write permissions in its directory to save the `watchlist.json` and `valid_tickers.json` files.

---

## Supabase: Persistent Watchlist Storage

This app can persist the sidebar watchlist in Supabase. If Supabase is configured, the app will load/save the watchlist there and also mirror it to `watchlist.json` as a local fallback.

### 1) Create the table (SQL)

Run this SQL in your Supabase project (SQL editor):

```sql
create table if not exists public.watchlist (
  id text primary key,
  tickers jsonb not null default '[]'::jsonb
);

-- Option A (Easiest): keep RLS disabled for this table.
-- If RLS is currently enabled, you can disable it with:
-- alter table public.watchlist disable row level security;
```

We store a single row with `id = 'default'` and the `tickers` array as JSON.

### 2) Allow writes from the Streamlit app (Option A)

Option A — Easiest (but less strict): leave Row Level Security (RLS) disabled for this table. Use the anon public key in your Streamlit secrets. This allows the app to read and write via the REST API with the anon key.

No policies are required when RLS is disabled.

### 3) Add Supabase credentials to Streamlit secrets

Add these to `.streamlit/secrets.toml` (locally) or Streamlit Cloud’s Secrets UI:

```toml
[supabase]
url = "https://dngilzirbpubhkoaohmk.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRuZ2lsemlyYnB1Ymhrb2FvaG1rIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ3MjkyNTMsImV4cCI6MjA3MDMwNTI1M30.fwyPEqcX84QYK5jLfdMbH62lMbWbkPh4FSfWzgrWNEU"

[api]
marketdata_token = "YOUR_MARKETDATA_APP_API_TOKEN"
```

The app will automatically detect these credentials. If absent, it will fall back to using the local `watchlist.json` file.

### 4) Dependencies

Ensure you have installed dependencies:

```bash
pip install -r requirements.txt
```

This includes the Supabase Python client (`supabase`).

### 5) Behavior

* On startup, the app tries to load the watchlist from Supabase. If the table is empty, it initializes it with defaults from `config.toml`.
* On add/remove in the sidebar, the app upserts the list to Supabase and mirrors to `watchlist.json`.
* If Supabase is unreachable, local file persistence continues to work.