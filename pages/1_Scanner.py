import streamlit as st
import pandas as pd
import asyncio
import json
from pathlib import Path
import toml
import math
from datetime import date
from scipy.stats import norm

# Make nested asyncio calls safe in Streamlit
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

from marketdata_client import MarketDataClient, clear_marketdata_cache
from matrix import build_matrix, R
from supabase_utils import (
    supabase_available,
    load_scanner_rules_supabase,
    save_scanner_rules_supabase,
    load_scanner_prefs_supabase,
    save_scanner_prefs_supabase,
)

# -----------------------------------------------------------------------------
# Config & Constants
# -----------------------------------------------------------------------------
CONFIG_FILE = "config.toml"
try:
    cfg = toml.load(CONFIG_FILE)
except Exception:
    cfg = {"MIN_DTE": 2, "MAX_DTE": 60, "TARGET_DELTA": 0.3}

APP_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RULES_FILE = APP_ROOT / "scanner_rules.json"
LOCAL_WATCHLIST_FILE = APP_ROOT / "watchlist.json"
LOCAL_PREFS_FILE = APP_ROOT / "scanner_prefs.json"
DEFAULT_STRATEGY = "PUT"

# -----------------------------------------------------------------------------
# Persistence Helpers (local + Supabase)
# -----------------------------------------------------------------------------

# Ensure full-width layout even if global page_config isn't wide
st.markdown(
    """
    <style>
    .block-container { max-width: 96% !important; padding-left: 2rem; padding-right: 2rem; }
    .st-expander { width: 100%; }
    .stTabs { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _load_rules_local() -> list[dict]:
    try:
        if LOCAL_RULES_FILE.exists():
            data = json.loads(LOCAL_RULES_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"WARN [Scanner._load_rules_local] {e}")
    return []


def _save_rules_local(rules: list[dict]) -> bool:
    try:
        LOCAL_RULES_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"ERROR [Scanner._save_rules_local] {e}")
        return False


def load_rules() -> list[dict]:
    # Try Supabase first
    if supabase_available():
        try:
            remote = load_scanner_rules_supabase()
            if remote:
                # Mirror to local for offline use
                _save_rules_local(remote)
                return remote
        except Exception as e:
            print(f"WARN [Scanner.load_rules] Supabase error, falling back to file: {e}")
    # Local fallback
    return _load_rules_local()


def save_rules(rules: list[dict]) -> bool:
    ok_remote = True
    if supabase_available():
        try:
            ok_remote = save_scanner_rules_supabase(rules)
        except Exception as e:
            print(f"WARN [Scanner.save_rules] Supabase save failed: {e}")
            ok_remote = False
    ok_local = _save_rules_local(rules)
    return ok_remote and ok_local


# --- Preferences (Min/Max DTE) persistence ---
def _load_prefs_local() -> dict:
    try:
        if LOCAL_PREFS_FILE.exists():
            data = json.loads(LOCAL_PREFS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"WARN [Scanner._load_prefs_local] {e}")
    return {}


def _save_prefs_local(prefs: dict) -> bool:
    try:
        LOCAL_PREFS_FILE.write_text(json.dumps(prefs or {}, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"WARN [Scanner._save_prefs_local] {e}")
        return False


def load_prefs() -> dict:
    # Try Supabase first
    if supabase_available():
        try:
            remote = load_scanner_prefs_supabase()
            if isinstance(remote, dict) and remote:
                _save_prefs_local(remote)  # mirror to local
                return remote
        except Exception as e:
            print(f"WARN [Scanner.load_prefs] Supabase error, falling back to file: {e}")
    # Local fallback
    return _load_prefs_local()


def save_prefs(prefs: dict) -> bool:
    ok_remote = True
    if supabase_available():
        try:
            ok_remote = save_scanner_prefs_supabase(prefs)
        except Exception as e:
            print(f"WARN [Scanner.save_prefs] Supabase save failed: {e}")
            ok_remote = False
    ok_local = _save_prefs_local(prefs)
    return ok_remote and ok_local


def import_watchlist_as_rules() -> list[dict]:
    try:
        if LOCAL_WATCHLIST_FILE.exists():
            tickers = json.loads(LOCAL_WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(tickers, list):
                # Create a default PUT rule with placeholders per ticker
                return [
                    {
                        "symbol": str(t).strip().upper(),
                        "desired_strike": 0.0,
                        "min_monthly_yield": 3.0,
                        "min_pop": 0.0,
                        "strategy": DEFAULT_STRATEGY,
                    }
                    for t in tickers
                    if isinstance(t, str) and t.strip()
                ]
    except Exception as e:
        print(f"WARN [Scanner.import_watchlist_as_rules] {e}")
    return []

# -----------------------------------------------------------------------------
# Async Scan Engine (fetch only required strategies per symbol)
# -----------------------------------------------------------------------------
async def _scan_once_async(rules: list[dict], min_dte: int, max_dte: int) -> pd.DataFrame:
    token = st.secrets.get("api", {}).get("marketdata_token")
    if not token:
        st.error("MarketData API token missing in secrets.toml.")
        return pd.DataFrame()

    # Group rules by symbol and collect required strategies per symbol
    by_symbol: dict[str, list[dict]] = {}
    strategies_by_symbol: dict[str, set[str]] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol", "")).strip().upper()
        if not sym:
            continue
        strat = str(r.get("strategy", DEFAULT_STRATEGY)).strip().upper()
        if strat not in ("PUT", "CALL"):
            strat = DEFAULT_STRATEGY
        by_symbol.setdefault(sym, []).append(r)
        strategies_by_symbol.setdefault(sym, set()).add(strat)

    client = MarketDataClient(token)

    async def fetch_symbol(sym: str):
        needed = strategies_by_symbol.get(sym, {DEFAULT_STRATEGY})
        tasks = []
        for strat in needed:
            tasks.append(
                build_matrix(
                    symbol=sym,
                    client=client,
                    strategy=strat,
                    target_delta=cfg.get("TARGET_DELTA", 0.3),
                    min_dte=min_dte,
                    max_dte=max_dte,
                    feed_type="cached",
                    cache_ttl=60,
                )
            )
        rows_by_strategy: dict[str, list[dict]] = {"PUT": [], "CALL": []}
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for strat, res in zip(needed, results):
                if isinstance(res, Exception):
                    print(f"ERROR [Scanner.fetch_symbol] {sym} {strat}: {res}")
                    continue
                _pivot, rows = res
                for o in rows or []:
                    o["_strategy"] = strat
                rows_by_strategy[strat] = rows or []
        except Exception as e:
            print(f"ERROR [Scanner.fetch_symbol] {sym}: {e}")
        return sym, rows_by_strategy

    # Fetch all symbols concurrently
    if not by_symbol:
        return pd.DataFrame()
    results = await asyncio.gather(*(fetch_symbol(sym) for sym in by_symbol.keys()))

    # Build final hit list
    hits: list[dict] = []
    for sym, rows_map in results:
        if not rows_map:
            continue
        # apply each rule for this symbol
        for rule in by_symbol.get(sym, []):
            desired_strike = float(rule.get("desired_strike", 0) or 0)
            min_yield = float(rule.get("min_monthly_yield", 0) or 0)
            min_pop = float(rule.get("min_pop", 0) or 0)
            strategy = str(rule.get("strategy", DEFAULT_STRATEGY)).upper()
            if strategy not in ("PUT", "CALL"):
                strategy = DEFAULT_STRATEGY

            # For PUTs: strike <= desired; For CALLs: strike >= desired
            def strike_ok(k: float) -> bool:
                if desired_strike <= 0:
                    return True
                return (k <= desired_strike) if strategy == "PUT" else (k >= desired_strike)

            relevant_rows = rows_map.get(strategy, [])
            for o in relevant_rows:
                try:
                    k = float(o.get("strike", 0) or 0)
                    y = float(o.get("monthly_yield", 0) or 0)
                    p = float(o.get("POP", 0) or 0)
                    if strike_ok(k) and y >= min_yield and p >= min_pop:
                        rec = {
                            "symbol": sym,
                            "strategy": strategy,
                            "expiry": o.get("expiry"),
                            "dte": int(o.get("dte", 0) or 0),
                            "strike": k,
                            "monthly_yield": round(y, 2),
                            "POP": p,
                            "delta": float(o.get("delta", 0) or 0),
                            "premium($)": float(o.get("premium_dollars", 0) or 0),
                            "price_used": float(o.get("Price Used", 0) or 0),
                            "method": o.get("Price Method", ""),
                            "iv(%)": float(o.get("iv", 0) or 0),
                            "breakeven": float(o.get("breakeven", 0) or 0),
                            "breakeven_%": float(o.get("breakeven_pct", 0) or 0),
                        }
                        hits.append(rec)
                except Exception:
                    continue

    try:
        await client.close()
    except Exception:
        pass

    if not hits:
        return pd.DataFrame()
    df = pd.DataFrame(hits)
    df.sort_values(by=["monthly_yield", "POP", "dte"], ascending=[False, False, True], inplace=True)
    return df

# Refresh only the current matches using live feed (sip), bypassing cache.
async def _refresh_matches_live(matches: pd.DataFrame, min_dte: int, max_dte: int) -> pd.DataFrame:
    token = st.secrets.get("api", {}).get("marketdata_token")
    if not token or matches is None or matches.empty:
        return pd.DataFrame()

    client = MarketDataClient(token)

    # Group by symbol/strategy/expiry and collect required strikes for each group
    need: dict[tuple[str, str, str], set[float]] = {}
    for _, row in matches.iterrows():
        try:
            sym = str(row.get("symbol", "")).strip().upper()
            strat = str(row.get("strategy", "PUT")).strip().upper()
            exp = str(row.get("expiry", ""))
            k = float(row.get("strike", 0) or 0)
            if not (sym and exp and strat in ("PUT", "CALL") and k > 0):
                continue
            need.setdefault((sym, strat, exp), set()).add(k)
        except Exception:
            continue

    today = date.today()
    results: list[dict] = []

    async def fetch_group(sym: str, strat: str, exp: str, strikes: set[float]):
        side = 'call' if strat == 'CALL' else 'put'
        try:
            # Live spot and chain, no cache
            spot = await client.quote(sym, feed='sip', ttl=0)
            chain = await client.chain(sym, exp, side, feed='sip', ttl=0)
            exp_date = date.fromisoformat(exp)
            dte = (exp_date - today).days
            if dte <= 0 or not isinstance(chain, list):
                return []

            out: list[dict] = []
            for o in chain:
                try:
                    K = float(o.get('strike', 0) or 0)
                    if K not in strikes:
                        continue
                    bid = float(o.get('bid', 0) or 0)
                    ask = float(o.get('ask', bid) or bid)
                    last = float(o.get('last', 0) or 0)
                    delta_raw, iv_raw = o.get('delta'), o.get('iv')
                    if not (K>0 and bid>=0 and isinstance(delta_raw,(int,float)) and isinstance(iv_raw,(int,float)) and iv_raw>=0):
                        continue
                    delta_val, iv_val = float(delta_raw), float(iv_raw)

                    mid = (bid + ask) / 2.0
                    threshold = 0.5 * ask
                    # If bid is zero, use zero (matches matrix.py)
                    if bid == 0.0:
                        price_used = 0.0; method = "bid_zero"
                    elif bid >= threshold:
                        price_used = mid; method = "midpoint"
                    elif (threshold <= last <= ask) and (last >= bid):
                        price_used = last; method = "last_close"
                    else:
                        # 20% of bid, clamped within [bid, ask]
                        price_used = min(max(bid * 1.2, bid), ask)
                        method = "bid_plus_20%"

                    actual_yield = (price_used / K) * 100.0 if K>0 else 0.0
                    monthly_yield = actual_yield * (30.0 / dte) if dte>0 else 0.0
                    premium = price_used * 100.0
                    collateral = K * 100.0
                    max_loss = max(collateral - premium, 0.0)
                    breakeven = K - price_used if strat=='PUT' else K + price_used
                    breakeven_pct = 0.0
                    try:
                        if spot and spot > 0:
                            diff = (spot - breakeven) if strat=='PUT' else (breakeven - spot)
                            breakeven_pct = (diff / spot) * 100.0
                    except Exception:
                        breakeven_pct = 0.0

                    # POP calculation (same as matrix)
                    pop = 0.0
                    T = dte / 365.0
                    if spot and spot>0 and iv_val>0 and K>0 and T>0:
                        BE = (K + price_used) if strat=='CALL' else max(K - price_used, 1e-9)
                        try:
                            d2 = (math.log(spot/BE) + (R - 0.5*iv_val**2)*T) / (iv_val*math.sqrt(T)) if iv_val>0 and T>0 else 0.0
                            pop = (1 - norm.cdf(d2)) if strat=='CALL' else norm.cdf(d2)
                            pop = max(0.0, min(pop, 1.0))
                        except Exception:
                            pop = 0.0

                    out.append({
                        "symbol": sym,
                        "strategy": strat,
                        "expiry": exp,
                        "dte": dte,
                        "strike": K,
                        "monthly_yield": round(monthly_yield, 2),
                        "POP": round(pop*100.0, 1),
                        "delta": float(delta_val),
                        "premium($)": round(premium, 2),
                        "price_used": float(price_used),
                        "method": method,
                        "iv(%)": round(iv_val*100.0, 1),
                        "breakeven": round(breakeven, 2),
                        "breakeven_%": round(breakeven_pct, 2),
                    })
                except Exception:
                    continue
            return out
        except Exception as e:
            print(f"ERROR [Scanner._refresh_matches_live] {sym} {exp} {strat}: {e}")
            return []

    # Schedule all groups concurrently
    tasks = [fetch_group(sym, strat, exp, strikes) for (sym, strat, exp), strikes in need.items()]
    try:
        groups = await asyncio.gather(*tasks, return_exceptions=True)
        for g in groups:
            if isinstance(g, Exception) or not g:
                continue
            results.extend(g)
    except Exception as e:
        print(f"ERROR [Scanner._refresh_matches_live] gather: {e}")

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df.sort_values(by=["monthly_yield", "POP", "dte"], ascending=[False, False, True], inplace=True)
    return df

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🔎 Scanner")

# Initialize session state for rules
if "scanner_rules" not in st.session_state:
    st.session_state.scanner_rules = load_rules() or [
        {"symbol": "SPY", "desired_strike": 0.0, "min_monthly_yield": 3.0, "min_pop": 0.0, "strategy": DEFAULT_STRATEGY}
    ]

# Controls
# Provide optional scanner-specific overrides with session persistence
# Prefs from Supabase/local take precedence, then SCANNER_MIN_DTE, then MIN_DTE.
prefs = {}
try:
    prefs = load_prefs() or {}
except Exception:
    prefs = {}
default_min_dte = int(prefs.get("min_dte", cfg.get("SCANNER_MIN_DTE", cfg.get("MIN_DTE", 2))))
default_max_dte = int(prefs.get("max_dte", cfg.get("MAX_DTE", 60)))
if "min_dte" not in st.session_state:
    st.session_state.min_dte = default_min_dte
if "max_dte" not in st.session_state:
    st.session_state.max_dte = default_max_dte
col_a, col_b = st.columns(2)
with col_a:
    min_dte = st.number_input(
        "Min DTE",
        min_value=0,
        max_value=365,
        key="min_dte",
        help="Default comes from SCANNER_MIN_DTE if set, else MIN_DTE."
    )
with col_b:
    max_dte = st.number_input(
        "Max DTE",
        min_value=0,
        max_value=730,
        key="max_dte"
    )

# Save current DTE values as preset (Supabase with local fallback)
sp_col1, sp_col2 = st.columns([1, 3])
with sp_col1:
    if st.button("💾 Save Preset (DTE)"):
        ok = save_prefs({"min_dte": int(st.session_state.min_dte), "max_dte": int(st.session_state.max_dte)})
        if ok:
            st.success("Scanner DTE preset saved.")
        else:
            st.warning("Failed to save preset (see logs). Using local fallback if available.")

st.markdown("---")

st.subheader("Scanner Rules")
with st.form("rules_form"):
    # Normalize existing rules so 'min_pop' is always present in the editor
    normalized_rules: list[dict] = []
    for r in (st.session_state.scanner_rules or []):
        if isinstance(r, dict):
            strat_val = str(r.get("strategy", DEFAULT_STRATEGY)).strip().upper()
            strat_val = strat_val if strat_val in ("PUT", "CALL") else DEFAULT_STRATEGY
            normalized_rules.append(
                {
                    "symbol": str(r.get("symbol", "")).strip().upper(),
                    "desired_strike": float(r.get("desired_strike", 0) or 0),
                    "min_monthly_yield": float(r.get("min_monthly_yield", 0) or 0),
                    "min_pop": float(r.get("min_pop", 0) or 0),
                    "strategy": strat_val,
                }
            )
    rules_df = st.data_editor(
        pd.DataFrame(normalized_rules),
        key="rules_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="Ticker symbol (e.g., SPY)"),
            "desired_strike": st.column_config.NumberColumn("Desired Strike", help="0 to ignore strike filter", step=0.5),
            "min_monthly_yield": st.column_config.NumberColumn("Min Monthly Yield %", help="0 to ignore yield filter", step=0.1, min_value=0.0, max_value=100.0),
            "min_pop": st.column_config.NumberColumn("Min POP %", help="0 to ignore POP filter", step=1.0, min_value=0.0, max_value=100.0),
            "strategy": st.column_config.SelectboxColumn("Strategy", options=["PUT", "CALL"]),
        },
    )
    fcol1, fcol2 = st.columns([1, 1])
    save_clicked = fcol1.form_submit_button("💾 Save Rules", type="primary")
    import_clicked = fcol2.form_submit_button("📥 Import Watchlist")

    if save_clicked:
        # Normalize and store back into session (preserve duplicates and order)
        new_rules: list[dict] = []
        for _, row in rules_df.fillna(0).iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            strat = str(row.get("strategy", DEFAULT_STRATEGY)).strip().upper()
            if strat not in ("PUT", "CALL"):
                strat = DEFAULT_STRATEGY
            new_rules.append(
                {
                    "symbol": sym,
                    "desired_strike": float(row.get("desired_strike", 0) or 0),
                    "min_monthly_yield": float(row.get("min_monthly_yield", 0) or 0),
                    "min_pop": float(row.get("min_pop", 0) or 0),
                    "strategy": strat,
                }
            )
        st.session_state.scanner_rules = new_rules
        if save_rules(st.session_state.scanner_rules):
            st.success("Rules saved.")
        else:
            st.warning("Failed to save rules (see logs).")

    if import_clicked:
        imported = import_watchlist_as_rules()
        if imported:
            # Merge: add only new symbols; keep existing entries intact
            existing_syms = {str(r.get("symbol", "")).strip().upper() for r in st.session_state.scanner_rules if isinstance(r, dict)}
            to_add = [r for r in imported if isinstance(r, dict) and str(r.get("symbol", "")).strip().upper() not in existing_syms]
            if to_add:
                st.session_state.scanner_rules = (st.session_state.scanner_rules or []) + to_add
                st.success(f"Added {len(to_add)} new ticker(s) from watchlist.")
                st.rerun()
            else:
                st.info("No new tickers to add from watchlist.")
        else:
            st.info("No watchlist found or import failed.")

col3, col4 = st.columns([1.8, 3])

with col3:
    if st.button("🧹 Refresh (clear API cache)"):
        try:
            loop = asyncio.get_event_loop()
            # nest_asyncio allows this in Streamlit
            loop.run_until_complete(clear_marketdata_cache())
            st.success("MarketData cache cleared.")
        except Exception as e:
            st.warning(f"Cache clear failed: {e}")

with col4:
    run_scan = st.button("🔄 Run Scan", use_container_width=True)

st.markdown("---")

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

results_df = pd.DataFrame()
if run_scan:
    # Validate DTE window
    _min = int(st.session_state.get("min_dte", 0))
    _max = int(st.session_state.get("max_dte", 0))
    if _min >= _max:
        st.error("Min DTE must be less than Max DTE.")
        results_df = pd.DataFrame()
    else:
        with st.spinner("Scanning..."):
            try:
                loop = asyncio.get_event_loop()
                results_df = loop.run_until_complete(_scan_once_async(st.session_state.scanner_rules, _min, _max))
            except RuntimeError:
                # Fallback if loop state is odd
                results_df = asyncio.run(_scan_once_async(st.session_state.scanner_rules, _min, _max))
            except Exception as e:
                st.error(f"Scan error: {e}")
                results_df = pd.DataFrame()
    # Persist results for UI reruns
    st.session_state.results_df = results_df.copy()
else:
    # Use last results so toggles (which cause reruns) don't clear the table
    results_df = st.session_state.get("results_df", pd.DataFrame())

if not results_df.empty:
    # Offer live refresh for current matches only
    lcol1, lcol2 = st.columns([3, 1])
    with lcol2:
        refresh_live = st.button("⚡ Refresh Matches (Live)")

    live_df = None
    if refresh_live:
        with st.spinner("Refreshing matches with live data..."):
            try:
                loop = asyncio.get_event_loop()
                live_df = loop.run_until_complete(_refresh_matches_live(results_df, min_dte, max_dte))
            except RuntimeError:
                live_df = asyncio.run(_refresh_matches_live(results_df, min_dte, max_dte))
            except Exception as e:
                st.error(f"Live refresh error: {e}")
                live_df = pd.DataFrame()

    # Ensure live_df is defined across reruns
    live_df = st.session_state.get("live_df", None)
    display_df = live_df if (live_df is not None and not live_df.empty) else results_df
    st.subheader("Matches (Live)" if live_df is not None else "Matches")

    # Toggle grouped view for readability
    gcol1, gcol2 = st.columns([1.5, 1])
    with gcol1:
        grouped_view = st.checkbox("Group by ticker (expand/collapse)", value=True, key="matches_grouped_view")
    with gcol2:
        expand_all = st.checkbox("Expand all", value=(display_df["symbol"].nunique() == 1), disabled=not grouped_view, key="matches_expand_all")

    # Column configuration for readability
    colcfg = {
        "monthly_yield": st.column_config.NumberColumn("Monthly Yield %", format="%.2f"),
        "POP": st.column_config.NumberColumn("POP %", format="%.1f"),
        "iv(%)": st.column_config.NumberColumn("IV %", format="%.1f"),
        "premium($)": st.column_config.NumberColumn("Premium $", format="%.2f"),
        "breakeven_%": st.column_config.NumberColumn("Breakeven %", format="%.2f"),
        "breakeven": st.column_config.NumberColumn("Breakeven $", format="%.2f"),
        "price_used": st.column_config.NumberColumn("Price Used $", format="%.2f"),
        "dte": st.column_config.NumberColumn("DTE", format="%d"),
        "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
    }

    # Reorder columns for clarity
    desired_cols = [
        "strategy", "expiry", "dte", "strike",
        "monthly_yield", "POP", "premium($)", "iv(%)",
        "breakeven", "breakeven_%", "price_used", "method",
    ]
    existing_cols = [c for c in desired_cols if c in display_df.columns]

    if not grouped_view:
        df_flat = display_df.copy()
        df_flat = df_flat.sort_values(by=["symbol", "monthly_yield", "POP", "dte"], ascending=[True, False, False, True])
        st.dataframe(df_flat, use_container_width=True, hide_index=True, column_config=colcfg)
    else:
        for sym in sorted(display_df["symbol"].unique()):
            sub = display_df[display_df["symbol"] == sym].copy()
            sub = sub.sort_values(by=["monthly_yield", "POP", "dte"], ascending=[False, False, True])
            expander = st.expander(f"{sym} • {len(sub)} matches", expanded=expand_all)
            with expander:
                # Tabs per strategy present
                strategies = [s for s in ["PUT", "CALL"] if s in set(sub.get("strategy", "PUT"))] or ["PUT"]
                if len(strategies) > 1:
                    tabs = st.tabs(strategies)
                    for t, strat in zip(tabs, strategies):
                        with t:
                            view = sub[sub.get("strategy", "PUT") == strat]
                            if existing_cols:
                                view = view[["symbol"] + existing_cols] if "symbol" in view.columns else view[existing_cols]
                            st.dataframe(view, use_container_width=True, hide_index=True, column_config=colcfg)
                else:
                    view = sub
                    if existing_cols:
                        view = view[["symbol"] + existing_cols] if "symbol" in view.columns else view[existing_cols]
                    st.dataframe(view, use_container_width=True, hide_index=True, column_config=colcfg)

    # Full CSV download of current display set
    csv = display_df.sort_values(by=["symbol", "monthly_yield", "POP", "dte"], ascending=[True, False, False, True]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=("scanner_results_live.csv" if live_df is not None else "scanner_results.csv"),
        mime="text/csv",
    )
else:
    st.info("No matches yet. Edit rules and click Run Scan.")
