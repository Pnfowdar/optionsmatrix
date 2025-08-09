# app.py
import streamlit as st
import toml
import asyncio
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta, time
import time as pytime
import nest_asyncio
from zoneinfo import ZoneInfo
import httpx
import csv
from io import StringIO
from marketdata_client import MarketDataClient
from supabase_utils import supabase_available, load_watchlist_supabase, save_watchlist_supabase

# ==============================================================================
#  1. Set Page Config & Apply Nest Asyncio FIRST
# ==============================================================================
st.set_page_config(page_title="Δ Options Matrix", layout="wide")
try: nest_asyncio.apply()
except RuntimeError: pass # Already applied
except Exception as e: print(f"ERROR Applying nest_asyncio: {e}")

# ==============================================================================
#  2. Load Configuration
# ==============================================================================
CONFIG_FILE = "config.toml"
try: cfg = toml.load(CONFIG_FILE)
except Exception as e:
    st.error(f"Error loading {CONFIG_FILE}: {e}. Using defaults."); cfg = {"TICKERS": ["SPY", "QQQ"], "MIN_DTE": 2, "MAX_DTE": 60, "MIN_POP_FILTER": 60.0, "DELTA_RANGE_FILTER": [0.10, 0.35], "MIN_YIELD_FILTER": 3.0, "TARGET_DELTA": 0.3, "RISK_FREE_RATE": 0.045, "POP_SCORE_BOUNDS": [70.0, 80.0], "YIELD_SCORE_BOUNDS": [3.0, 8.0]}
WATCHFILE = Path(__file__).parent / "watchlist.json"

# ==============================================================================
#  3. Import Local Modules
# ==============================================================================
try:
    from marketdata_client import MarketDataClient, clear_marketdata_cache
    from matrix import build_matrix, compute_score_matrix, _is_market_open
except ImportError as e: st.error(f"FATAL: Import error: {e}. App cannot run."); st.stop()

# ==============================================================================
#  4. Define Constants & Functions
# ==============================================================================
NASDAQ_TICKER_LIST_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&download=true"
VALID_TICKERS_FILE = Path(__file__).parent / "valid_tickers.json"
TICKER_FILE_UPDATE_INTERVAL_DAYS = 7

# --- Persistence (Supabase + local fallback) ---
def load_watchlist() -> list[str]:
    """Load watchlist from Supabase if available; otherwise use local JSON.

    If Supabase is configured but empty, initialize it with defaults.
    """
    defaults = cfg.get("TICKERS", ["SPY", "QQQ"])[:]
    # Try Supabase first
    if supabase_available():
        try:
            remote = load_watchlist_supabase()
            if remote:
                print(f"DEBUG [load_watchlist] Loaded {len(remote)} tickers from Supabase")
                return remote
            # Initialize remote with defaults if empty
            saved = save_watchlist_supabase(defaults)
            print(f"INFO [load_watchlist] Supabase empty. Initialized with defaults: saved={saved}")
            return defaults
        except Exception as e:
            print(f"WARN [load_watchlist] Supabase error, falling back to file: {e}")

    # Local file fallback
    print(f"DEBUG [load_watchlist] Attempting to load from: {WATCHFILE}")
    if WATCHFILE.exists():
        try:
            data = json.loads(WATCHFILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(t, str) for t in data):
                print(f"DEBUG [load_watchlist] SUCCESS: Loaded {len(data)} tickers from file.")
                return data
            else:
                st.warning(f"{WATCHFILE} has invalid format. Reverting to defaults.")
                print(f"WARN [load_watchlist] Invalid format in file. Saving defaults.")
                save_watchlist(defaults)
                return defaults
        except json.JSONDecodeError as e:
            st.error(f"Error decoding watchlist file: {e}. Using defaults.")
            print(f"ERROR [load_watchlist] JSON decode error: {e}")
            return defaults
        except Exception as e:
            st.error(f"Error loading watchlist from {WATCHFILE}: {e}. Using defaults.")
            print(f"ERROR [load_watchlist] General load error: {e}")
            return defaults
    else:
        print(f"DEBUG [load_watchlist] {WATCHFILE} not found. Creating with defaults: {defaults}")
        save_watchlist(defaults)
        return defaults


def save_watchlist(ticker_list: list[str]):
    """Save watchlist to Supabase if available and mirror to local JSON.

    Shows user-friendly errors if both writes fail.
    """
    # Normalize list
    unique_tickers = sorted(list(set(filter(None, map(str.strip, ticker_list)))))

    # Try Supabase write first (non-fatal if it fails)
    supabase_ok = False
    if supabase_available():
        try:
            supabase_ok = save_watchlist_supabase(unique_tickers)
            if supabase_ok:
                print(f"DEBUG [save_watchlist] Upserted {len(unique_tickers)} tickers to Supabase")
        except Exception as e:
            print(f"WARN [save_watchlist] Supabase save failed: {e}")

    # Always mirror to local file for safety
    file_ok = False
    try:
        WATCHFILE.write_text(json.dumps(unique_tickers, indent=2), encoding="utf-8")
        file_ok = True
        print(f"DEBUG [save_watchlist] Saved {len(unique_tickers)} tickers to {WATCHFILE}")
    except (IOError, OSError) as e:
        st.error(f"CRITICAL: Could not write watchlist file '{WATCHFILE}'. Error: {e}")
        print(f"ERROR [save_watchlist] File write failed (IO/OS): {e}")
    except Exception as e:
        st.error(f"Error saving watchlist locally: {e}")
        print(f"ERROR [save_watchlist] Unexpected save error: {e}")

    if not (supabase_ok or file_ok):
        st.error("Failed to save watchlist to both Supabase and local file. Please check logs.")

# --- Ticker Validation ---
@st.cache_resource(ttl=3600)
def load_valid_tickers() -> set:
    """Loads valid ticker set from JSON (cached)."""
    if VALID_TICKERS_FILE.exists():
        try: data = json.loads(VALID_TICKERS_FILE.read_text(encoding="utf-8")); return set(data) if isinstance(data, list) else set()
        except Exception: return set()
    return set()

async def fetch_and_save_tickers():
    """Downloads and saves Nasdaq ticker list."""
    st.toast("Updating Nasdaq ticker list...")
    headers = { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json, */*', 'Origin': 'https://www.nasdaq.com', 'Referer': 'https://www.nasdaq.com/' }
    try:
        async with httpx.AsyncClient(timeout=45.0, headers=headers, follow_redirects=True) as client:
            r = await client.get(NASDAQ_TICKER_LIST_URL); r.raise_for_status()
        content_type = r.headers.get('content-type', '').lower(); valid_symbols = set()
        if 'json' in content_type: # JSON Parsing
            data = r.json(); rows = data.get('data', {}).get('rows', [])
            if rows and isinstance(rows, list):
                for row in rows: s = row.get('symbol');
                if s and isinstance(s, str) and '/' not in s and '$' not in s and '.' not in s: valid_symbols.add(s.strip().upper())
        elif 'csv' in content_type or 'text/plain' in content_type: # Basic CSV handling
             reader = csv.DictReader(StringIO(r.text)); sym_col = 'Symbol'
             if sym_col in (reader.fieldnames or []):
                 for row in reader: s = row.get(sym_col);
                 if s and isinstance(s, str) and '/' not in s and '$' not in s and '.' not in s: valid_symbols.add(s.strip().upper())
        if not valid_symbols: st.error("Failed to extract symbols."); return False
        VALID_TICKERS_FILE.write_text(json.dumps(sorted(list(valid_symbols)), indent=2), encoding="utf-8")
        st.toast(f"Ticker list updated: {len(valid_symbols)} symbols."); load_valid_tickers.clear(); return True
    except Exception as e: st.error(f"Ticker list update failed: {e}"); return False

def check_and_update_ticker_list():
    """Checks ticker list age and triggers update."""
    needs_update = False
    if not VALID_TICKERS_FILE.exists(): needs_update = True
    else:
        try: fmtime = datetime.fromtimestamp(VALID_TICKERS_FILE.stat().st_mtime, tz=ZoneInfo("UTC")); needs_update = (datetime.now(ZoneInfo("UTC")) - fmtime) > timedelta(days=TICKER_FILE_UPDATE_INTERVAL_DAYS)
        except Exception: needs_update = True
    if needs_update:
        print("DEBUG Ticker list update required, scheduling fetch task...")
        try: loop = asyncio.get_event_loop(); loop.create_task(fetch_and_save_tickers()) if loop.is_running() else asyncio.run(fetch_and_save_tickers())
        except Exception as e: print(f"ERROR running ticker update task: {e}")

# --- Data Fetching ---
async def _fetch_matrix_data(symbol, strategy, token, min_dte, max_dte, feed_type='cached'):
    """Fetches matrix data using build_matrix with specified feed type."""
    print(f"DEBUG [_fetch_matrix_data] Fetching for {symbol} | Feed: {feed_type}")
    client = MarketDataClient(token)
    try: pivot, rows = await build_matrix( symbol, client, strategy, cfg.get("TARGET_DELTA", 0.3), min_dte, max_dte, feed_type=feed_type, cache_ttl=60 )
    except Exception as e: print(f"ERROR [_fetch_matrix_data] {feed_type} fetch for {symbol}: {e}"); return pd.DataFrame(), []
    finally: await client.close() # Ensure client is closed
    return pivot, rows

@st.cache_data(ttl=60)
def _fetch_price(symbol: str, price_refresh_trigger: float) -> float:
    """
    Fetches the latest spot price via marketdata.app (Streamlit cached).
    """
    try:
        token = st.secrets["api"]["marketdata_token"]

        async def get_price():
            client = MarketDataClient(token)
            # pull the 'last' quote from the cached feed
            price = await client.quote(symbol, feed="cached")
            await client.close()
            return price

        return asyncio.run(get_price())
    except Exception as e:
        print(f"ERROR [_fetch_price] marketdata.app failed for {symbol}: {e}")
        return 0.0

# --- Data Processing ---
def _process_and_validate_df(rows: list, sym: str) -> pd.DataFrame:
    """Converts rows to DataFrame, cleans, validates, returns df or empty df."""
    if not rows: return pd.DataFrame()
    try:
        df = pd.DataFrame(rows);
        if df.empty: return pd.DataFrame()
        num_cols = ['strike','delta','POP','monthly_yield','actual_yield','premium_dollars','iv','bid','mid','ask','last','Price Used','dte']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'expiry' in df.columns: df['expiry_date_obj'] = pd.to_datetime(df['expiry'], errors='coerce')
        ess_cols = ['strike','dte','delta','POP','monthly_yield','Price Used','expiry_date_obj']
        df = df.dropna(subset=ess_cols); df = df[df['dte'] > 0]; df = df[df['strike'] > 0]
        df.fillna({'last': 0.0}, inplace=True)
        return df
    except Exception as e: print(f"ERROR [_process_df] for {sym}: {e}"); st.warning(f"Processing error for {sym}."); return pd.DataFrame()

# --- Helper Function _update_df_with_live_quotes REMOVED ---
# (No longer needed as we replace the whole DataFrame)

# --- Renderer Definition ---
async def render(sym: str, box: st.container, price_trigger: float, options_trigger: float, api_token: str):
    """Renders the options matrix and inspector for a single symbol."""
    df_state_key = f"df_{sym}"; fetch_ts_key = f"options_fetch_ts_{sym}"; feed_type_key = f"feed_type_{sym}"

    # --- Get or Fetch Base Data (Cached) ---
    if df_state_key not in st.session_state or options_trigger > st.session_state.get(fetch_ts_key, 0):
        print(f"DEBUG [render] State MISS/EXPIRED for {sym}. Fetching BASE (cached)...")
        with st.spinner(f"Fetching cached data for {sym}..."):
            _, base_rows = await _fetch_matrix_data( sym, st.session_state.strategy_value, api_token, st.session_state.min_dte_value, st.session_state.max_dte_value, feed_type='cached')
        df_processed = _process_and_validate_df(base_rows, sym)
        st.session_state[df_state_key] = df_processed; st.session_state[fetch_ts_key] = options_trigger; st.session_state[feed_type_key] = "Cached"
        print(f"DEBUG [render] Stored BASE df for {sym}, shape: {df_processed.shape}")
    else:
        df_processed = st.session_state[df_state_key] # Use existing df
    current_feed_type = st.session_state.get(feed_type_key, "Cached")

    # --- Spot Price ---
    try: spot_price = _fetch_price(sym, price_trigger)
    except Exception as e: st.error(f"Spot price error: {e}"); spot_price = 0.0

    # --- Header Display ---
    market_is_open = _is_market_open(); market_status = "🟢 Open" if market_is_open else "🔴 Closed"; price_display = f"${spot_price:.2f}" if spot_price > 0 else "$N/A"
    try: local_tz = ZoneInfo("Australia/Brisbane"); updated_ts = datetime.now(ZoneInfo("UTC")).astimezone(local_tz).strftime("%H:%M:%S %Z")
    except Exception: updated_ts = datetime.now().strftime("%H:%M:%S")
    box.subheader(f"{sym} ({price_display})")
    cap_col, btn_col = box.columns([4, 1])
    cap_col.caption(f"Status: {market_status} | Feed: {current_feed_type} | Refreshed: {updated_ts}")

    # --- Live Quote Refresh Button ---
    if btn_col.button("🔄 Live", key=f"live_btn_{sym}", help="Fetch LIVE options data (Uses credits!)", disabled=(not market_is_open)): # Changed Button Text & Disabled when market closed
        print(f"DEBUG [render] Live button clicked for {sym}.")
        with st.spinner(f"Fetching live data for {sym}..."):
            _, live_rows = await _fetch_matrix_data( sym, st.session_state.strategy_value, api_token, st.session_state.min_dte_value, st.session_state.max_dte_value, feed_type='sip') # Fetch LIVE
        if live_rows:
            # Process the live rows into a DataFrame
            df_live_processed = _process_and_validate_df(live_rows, sym)
            if not df_live_processed.empty:
                # Replace the DataFrame in session state with the new live one
                st.session_state[df_state_key] = df_live_processed
                st.session_state[feed_type_key] = "Live" # Update feed type indicator
                print(f"DEBUG [render] Replaced df for {sym} with LIVE data, shape {df_live_processed.shape}. Rerunning.")
                st.rerun() # Refresh UI immediately
            else:
                 box.warning(f"Live data for {sym} was empty after processing.", icon="⚠️")
        else: box.warning(f"Failed to fetch live data for {sym}. Displaying previous data.", icon="⚠️")

    # --- Process and Display (using df_processed from session state) ---
    if df_processed.empty: box.info(f"No options data available for {sym} (Feed: {current_feed_type}).", icon="ℹ️"); return

    # --- Filtering ---
    try: # Filter the current df_processed
        pop_min=st.session_state.pop_slider_value; delta_min, delta_max=st.session_state.delta_slider_value; yield_min=st.session_state.yield_slider_value; dte_min=st.session_state.min_dte_value; dte_max=st.session_state.max_dte_value
        df_filtered = df_processed[ (df_processed["POP"] >= pop_min) & (df_processed["delta"].between(delta_min, delta_max)) & (df_processed["monthly_yield"] >= yield_min) & (df_processed["dte"].between(dte_min, dte_max)) ].copy()
    except Exception as e: box.error(f"Filter error: {e}"); return
    if df_filtered.empty: box.warning("No contracts match filters."); return

    # --- Pivot Tables and Score Matrix ---
    try: # Pivot and score logic remains the same
        col_map={"Monthly Yield %":"monthly_yield","Actual Yield %":"actual_yield","Δ":"delta","POP":"POP","Premium ($)":"premium_dollars"}
        value_col=col_map.get(st.session_state.display_mode_value,"monthly_yield");
        if value_col not in df_filtered.columns: box.error(f"Display col '{value_col}' missing."); return
        disp_pivot=pd.pivot_table(df_filtered,index="expiry",columns="strike",values=value_col,aggfunc='mean')
        score_matrix=pd.DataFrame()
        if "monthly_yield" in df_filtered.columns and "POP" in df_filtered.columns:
            mp=pd.pivot_table(df_filtered,index="expiry",columns="strike",values="monthly_yield",aggfunc='mean')
            pp=pd.pivot_table(df_filtered,index="expiry",columns="strike",values="POP",aggfunc='mean')
            if not mp.empty and not pp.empty: ci=mp.index.intersection(pp.index); cc=mp.columns.intersection(pp.columns);
            if not ci.empty and not cc.empty: score_matrix=compute_score_matrix(mp.loc[ci,cc],pp.loc[ci,cc])
    except Exception as e: box.error(f"Pivot/Score error: {e}"); return
    if disp_pivot.empty: box.warning("Filtered data resulted in empty pivot."); return

    # --- MultiIndex Header ---
    try: # MultiIndex logic remains the same
        strikes_num=pd.to_numeric(disp_pivot.columns,errors='coerce'); valid_mask=strikes_num.notna(); valid_strikes=strikes_num[valid_mask].tolist(); orig_valid_cols=disp_pivot.columns[valid_mask].tolist(); mi=None
        if valid_strikes:
            valid_spot=spot_price > 0; dist_vals=[round(((k-spot_price)/spot_price)*100,1) if valid_spot else 0.0 for k in valid_strikes]
            mi_tuples=[(f"{k:.2f}",f"{d:+.1f}%") for k,d in zip(valid_strikes,dist_vals)]; mi=pd.MultiIndex.from_tuples(mi_tuples,names=["Strike","Δ% Spot"])
            disp_pivot=disp_pivot[orig_valid_cols]; disp_pivot.columns=mi
            if not score_matrix.empty: # Align score matrix
                 sc=pd.to_numeric(score_matrix.columns,errors='coerce'); sv=sc.notna(); soc=score_matrix.columns[sv].tolist(); ck=[c for c in soc if c in orig_valid_cols]
                 if ck: sf=score_matrix[ck]; sf.columns=mi; score_matrix=sf.reindex(index=disp_pivot.index,columns=disp_pivot.columns).fillna(0.0)
                 else: score_matrix=pd.DataFrame(0.0,index=disp_pivot.index,columns=disp_pivot.columns)
            else: score_matrix=pd.DataFrame(0.0,index=disp_pivot.index,columns=disp_pivot.columns)
    except Exception as e: box.error(f"Multi-index error: {e}"); mi=None

    # --- Style and Display Matrix ---
    try: # Styling logic remains the same
        fmt_map={"monthly_yield":"{:.2f}%","actual_yield":"{:.2f}%","delta":"{:.2f}","POP":"{:.1f}%","premium_dollars":"${:.2f}"}
        fmt_filter=fmt_map.get(value_col,"{:.2f}")
        score_aligned=score_matrix.reindex(index=disp_pivot.index, columns=disp_pivot.columns).fillna(0.0) if not score_matrix.empty else pd.DataFrame(0.0,index=disp_pivot.index,columns=disp_pivot.columns)
        styled=disp_pivot.style.format(fmt_filter,na_rep="-",precision=2)
        if not score_aligned.empty and not (score_aligned==0).all().all():
            try: styled=styled.background_gradient(cmap="RdYlGn",gmap=score_aligned.astype(float),axis=None)
            except Exception as bg_err: box.warning(f"Gradient failed: {bg_err}")
        styled=styled.highlight_null(color="#AAA")
        box.dataframe(styled,use_container_width=True)
    except Exception as e: box.error(f"Style/Display error: {e}"); box.dataframe(disp_pivot.fillna("-"))


    # --- Inspector ---
    try: # Inspector logic remains the same
        box.markdown("---"); box.markdown("#### Inspector")
        if df_filtered.empty: box.info("No filtered data for inspection."); return
        sub_inspect=df_filtered.copy(); mode=box.radio("Inspect by",["Expiry","Strike"],horizontal=True,index=0,key=f"inspect_mode_{sym}")
        if mode=="Expiry": opts=sorted(sub_inspect["expiry"].unique()); sel=box.selectbox("Expiry",opts,index=0,key=f"insp_sel_exp_{sym}"); sub_inspect=sub_inspect[sub_inspect["expiry"]==sel]
        else: opts=sorted(sub_inspect["strike"].unique()); sel=box.selectbox("Strike",opts,format_func=lambda x:f"{x:.2f}",index=0,key=f"insp_sel_str_{sym}"); sub_inspect=sub_inspect[sub_inspect["strike"]==sel]
        if sub_inspect.empty: box.info("No data matches selection."); return
        rename_map={"expiry":"Expiry","strike":"Strike","dte":"DTE","delta":"Δ","POP":"POP %","monthly_yield":"Mnth Yield %","actual_yield":"Act Yield %","premium_dollars":"Premium $","iv":"IV %","bid":"Bid","mid":"Mid","ask":"Ask","last":"Option Last Close","Price Used":"Price Used"}
        sub_display=sub_inspect.rename(columns={k:v for k,v in rename_map.items() if k in sub_inspect.columns})
        desired_cols=["Expiry","Strike","DTE","Δ","POP %","Mnth Yield %","Act Yield %","Premium $","IV %","Bid","Mid","Ask","Option Last Close","Price Used"]
        display_cols=[col for col in desired_cols if col in sub_display.columns]
        fmt_ins={"Strike":"{:.2f}","Δ":"{:.2f}","POP %":"{:.1f}%","Mnth Yield %":"{:.2f}%","Act Yield %":"{:.2f}%","Premium $":"${:.2f}","IV %":"{:.1f}%","Bid":"{:.2f}","Mid":"{:.2f}","Ask":"{:.2f}","Option Last Close":"{:.2f}","Price Used":"{:.2f}"}
        final_fmt_ins={k:v for k,v in fmt_ins.items() if k in display_cols}
        sort_col="Strike" if mode=="Expiry" else "Expiry";
        if sort_col in sub_display.columns: sub_display=sub_display.sort_values(by=sort_col)
        box.dataframe(sub_display[display_cols].style.format(final_fmt_ins,na_rep="-",precision=2),use_container_width=True,hide_index=True)
    except Exception as e: box.error(f"Inspector error: {e}")


# ==============================================================================
#  5. Main Script Execution Flow
# ==============================================================================
# --- Initial Setup ---
check_and_update_ticker_list()
valid_ticker_list = sorted(list(load_valid_tickers()))

# --- App Title & CSS ---
st.title("Δ Options Yield Matrix")
# Updated CSS for smaller sidebar elements
st.markdown("""
<style>
    /* General Sidebar spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.1rem !important; } /* Reduce gap */
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.1rem !important; } /* Reduce margin */
    /* Watchlist item style */
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stTextInput { line-height: 1.1 !important; padding: 0.05rem 0 !important; margin-bottom: 0 !important;}
    [data-testid="stSidebar"] .stButton>button { padding: 0px 4px !important; font-size: 0.65rem !important; margin-left: 3px; line-height: 1 !important; height: 1.5em !important;} /* Smaller button */
    [data-testid="stSidebar"] .stCheckbox p { margin-bottom: 0 !important; font-size: 0.85rem !important; line-height: 1.2 !important;} /* Smaller checkbox label */
    /* Main Area Styling */
    .stDataFrame div[data-testid="stDataFrameData"] > div > div > div.data { background-color: inherit !important; }
    td[class^="col"] { background-color: inherit !important; color: inherit !important; }
    .stDataFrame th[data-testid="stTick"] { font-size: 0.8rem; padding: 4px !important; }
    div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def refresh_prices():
    # only bump the price-trigger and clear the price cache
    st.session_state.price_refresh_trigger = pytime.time()
    _fetch_price.clear()

def refresh_options():
    # only bump the options-trigger and clear your matrix caches
    st.session_state.options_refresh_trigger = pytime.time()
    for key in list(st.session_state):
        if key.startswith("df_") or key.startswith("options_fetch_ts_") or key.startswith("feed_type_"):
            del st.session_state[key]
    # clear the shared MarketDataClient cache
    asyncio.run(clear_marketdata_cache())

# --- Session State Initialization ---
default_state = {"price_refresh_trigger": pytime.time(), "options_refresh_trigger": pytime.time(), "manual_ticker_input_value": "", "display_mode_value": cfg.get("DISPLAY_MODE", "Monthly Yield %"), "strategy_value": cfg.get("DEFAULT_STRATEGY", "PUT"), "min_dte_value": cfg.get("MIN_DTE", 2), "max_dte_value": cfg.get("MAX_DTE", 60), "pop_slider_value": cfg.get("MIN_POP_FILTER", 60.0), "delta_slider_value": tuple(cfg.get("DELTA_RANGE_FILTER", [0.10, 0.35])), "yield_slider_value": cfg.get("MIN_YIELD_FILTER", 3.0)}
for key, default_value in default_state.items():
    st.session_state.setdefault(key, default_value)
# Load watchlist into state only if not already present or empty
if "tickers" not in st.session_state or not st.session_state.tickers:
    print("DEBUG [app.py] Initializing 'tickers' state from load_watchlist()")
    st.session_state.tickers = load_watchlist()

# --- Refresh Buttons ---
col1, col2, _ = st.columns([1.3, 1.3, 5])
col1.button(
    "🔄 Prices",
    key="refresh_price_btn",
    help="Reload stock prices (keeping filters)",
    on_click=refresh_prices
)
col2.button(
    "🔄 Options",
    key="refresh_options_btn",
    help="Reload options data (keeping filters)",
    on_click=refresh_options
)

# --- Sidebar Widgets ---
st.sidebar.header("Watchlist")
selected_tickers_sidebar = [] # Initialize list for selected tickers
# Ensure tickers in state are clean *before* iterating for display
st.session_state.tickers = sorted(list(set(filter(None, map(str.strip, st.session_state.get("tickers", [])))))) # Use .get for safety
for sym in st.session_state.tickers: # Watchlist display loop
    cols=st.sidebar.columns([0.8, 0.2]); is_selected=cols[0].checkbox(sym,value=False,key=f"chk_{sym}") # Unique key
    if is_selected: selected_tickers_sidebar.append(sym) # <<< Populate the list
    if cols[1].button("✕",key=f"del_{sym}",help=f"Remove {sym}"): # Unique key
        st.session_state.tickers.remove(sym); save_watchlist(st.session_state.tickers)
        if f"chk_{sym}" in st.session_state: del st.session_state[f"chk_{sym}"] # Clean up checkbox state
        st.rerun()

st.sidebar.markdown("---"); st.sidebar.markdown("**Add Ticker**") # Add Ticker Section
SELECTBOX_PLACEHOLDER = "Type or select..."
options_with_placeholder = [SELECTBOX_PLACEHOLDER] + valid_ticker_list
ticker_to_add_select=st.sidebar.selectbox("Search/Select",options=options_with_placeholder, index=0, key="add_ticker_selectbox")
manual_ticker_input=st.sidebar.text_input("Or enter manually", key="manual_widget", value=st.session_state.manual_ticker_input_value, placeholder="e.g., MSFT").strip().upper()
st.session_state.manual_ticker_input_value = manual_ticker_input
if st.sidebar.button("➕ Add", key="add_ticker_btn"):
    ticker_to_add = manual_ticker_input if manual_ticker_input else (ticker_to_add_select if ticker_to_add_select != SELECTBOX_PLACEHOLDER else "")
    if ticker_to_add:
        if ticker_to_add not in st.session_state.tickers:
            st.session_state.tickers.append(ticker_to_add)
            save_watchlist(st.session_state.tickers) # Save the updated list
            st.session_state.manual_ticker_input_value = "" # Clear manual input
            st.rerun()
        else: st.sidebar.warning(f"'{ticker_to_add}' already in watchlist.")
    else: st.sidebar.warning("Please select or enter a ticker.")

st.sidebar.header("Filters") # Filters Section
# Display Mode Radio — bind directly into session_state.display_mode_value
st.sidebar.radio(
    "Show in matrix:",
    ["Monthly Yield %","Actual Yield %","Δ","POP","Premium ($)"],
    key="display_mode_value"
)
# Strategy Radio
st.sidebar.radio(
    "Strategy:", ["PUT","CALL"],
    key="strategy_value"
)
# DTE Inputs
st.sidebar.number_input("Min DTE:", min_value=0, max_value=365, key="min_dte_value")
st.sidebar.number_input("Max DTE:", min_value=st.session_state.min_dte_value+1, max_value=730, key="max_dte_value")
# Sliders
st.sidebar.slider("Min POP (%):", 0.0, 100.0, key="pop_slider_value", format="%.1f%%")
st.sidebar.slider("|Δ| range:", 0.0, 1.0, key="delta_slider_value", format="%.2f")
st.sidebar.slider("Min Mnth Yield (%):", 0.0, 25.0, key="yield_slider_value", format="%.1f%%")

# --- API Token Check ---
api_token = st.secrets.get("api", {}).get("marketdata_token")
if not api_token: st.error("MarketData API token missing in secrets.toml."); st.stop()

async def main_async_render_loop():
    """The main async loop that renders selected tickers."""
    selected_tickers = selected_tickers_sidebar
    if not selected_tickers:
        st.info("Select tickers in sidebar watchlist."); st.stop()
    price_trigger = st.session_state.price_refresh_trigger
    options_trigger = st.session_state.options_refresh_trigger

    # No outer spinner—just kick off all renders immediately
    render_tasks = [
        render(sym, st.container(), price_trigger, options_trigger, api_token)
        for sym in selected_tickers
    ]
    await asyncio.gather(*render_tasks)


# --- Script Entry Point ---
if __name__ == "__main__":
    print(f"DEBUG [app.py] Script run start: {datetime.now()}")
    try: # Manage event loop
        loop = asyncio.get_event_loop()
        if loop.is_running(): loop.create_task(main_async_render_loop())
        else: asyncio.run(main_async_render_loop())
    except Exception as e: st.error(f"Main execution error: {e}"); print(f"ERROR [app.py] Main execution: {e}")