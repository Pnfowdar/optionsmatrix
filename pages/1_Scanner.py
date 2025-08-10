import streamlit as st
import pandas as pd
import asyncio
import json
from pathlib import Path
import toml

# Make nested asyncio calls safe in Streamlit
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

from marketdata_client import MarketDataClient, clear_marketdata_cache
from matrix import build_matrix
from supabase_utils import (
    supabase_available,
    load_scanner_rules_supabase,
    save_scanner_rules_supabase,
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
DEFAULT_STRATEGY = "PUT"

# -----------------------------------------------------------------------------
# Persistence Helpers (local + Supabase)
# -----------------------------------------------------------------------------

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
                    if strike_ok(k) and y >= min_yield:
                        rec = {
                            "symbol": sym,
                            "strategy": strategy,
                            "expiry": o.get("expiry"),
                            "dte": int(o.get("dte", 0) or 0),
                            "strike": k,
                            "monthly_yield": round(y, 2),
                            "POP": float(o.get("POP", 0) or 0),
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

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🔎 Scanner")

# Initialize session state for rules
if "scanner_rules" not in st.session_state:
    st.session_state.scanner_rules = load_rules() or [
        {"symbol": "SPY", "desired_strike": 0.0, "min_monthly_yield": 3.0, "strategy": DEFAULT_STRATEGY}
    ]

# Controls
col_a, col_b = st.columns(2)
with col_a:
    min_dte = st.number_input("Min DTE", min_value=0, max_value=365, value=int(cfg.get("MIN_DTE", 14)))
with col_b:
    max_dte = st.number_input("Max DTE", min_value=min_dte + 1, max_value=730, value=int(cfg.get("MAX_DTE", 60)))

st.markdown("---")

st.subheader("Scanner Rules")
with st.form("rules_form"):
    rules_df = st.data_editor(
        pd.DataFrame(st.session_state.scanner_rules),
        key="rules_editor",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="Ticker symbol (e.g., SPY)"),
            "desired_strike": st.column_config.NumberColumn("Desired Strike", help="0 to ignore strike filter", step=0.5),
            "min_monthly_yield": st.column_config.NumberColumn("Min Monthly Yield %", step=0.1),
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

results_df = pd.DataFrame()
if run_scan:
    with st.spinner("Scanning..."):
        try:
            loop = asyncio.get_event_loop()
            results_df = loop.run_until_complete(_scan_once_async(st.session_state.scanner_rules, min_dte, max_dte))
        except RuntimeError:
            # Fallback if loop state is odd
            results_df = asyncio.run(_scan_once_async(st.session_state.scanner_rules, min_dte, max_dte))
        except Exception as e:
            st.error(f"Scan error: {e}")
            results_df = pd.DataFrame()

if not results_df.empty:
    st.subheader("Matches")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="scanner_results.csv", mime="text/csv")
else:
    st.info("No matches yet. Edit rules and click Run Scan.")
