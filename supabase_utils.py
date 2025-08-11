# supabase_utils.py

from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import streamlit as st

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

TABLE_NAME = "watchlist"
ROW_ID = "default"  # single-row storage for this app instance

# --- Scanner Rules Persistence (Supabase) ---
SCANNER_TABLE_NAME = "scanner_rules"
SCANNER_ROW_ID = ROW_ID  # reuse same default id

# --- Scanner Preferences (Supabase) ---
PREFS_TABLE_NAME = "scanner_prefs"
PREFS_ROW_ID = ROW_ID


def _get_supabase_client() -> Optional["Client"]:
    """Return a Supabase client if credentials exist and library is installed."""
    if create_client is None:
        return None
    supa = st.secrets.get("supabase", {})
    url = supa.get("url")
    key = supa.get("key")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def supabase_available() -> bool:
    return _get_supabase_client() is not None


def load_watchlist_supabase() -> List[str]:
    """Load the watchlist from Supabase. Returns [] if none found or any error."""
    client = _get_supabase_client()
    if client is None:
        return []
    try:
        # select the single row by id
        res = client.table(TABLE_NAME).select("id,tickers").eq("id", ROW_ID).execute()
        data = getattr(res, "data", None) or []
        if not data:
            return []
        row = data[0]
        tickers = row.get("tickers")
        # tickers may come as list or JSON string depending on table type
        if isinstance(tickers, str):
            try:
                tickers = json.loads(tickers)
            except Exception:
                return []
        if isinstance(tickers, list) and all(isinstance(t, str) for t in tickers):
            return tickers
        return []
    except Exception as e:
        print(f"WARN [supabase_utils.load] {e}")
        return []


def save_watchlist_supabase(tickers: List[str]) -> bool:
    """Upsert the watchlist into Supabase. Returns True on success."""
    client = _get_supabase_client()
    if client is None:
        return False
    try:
        # ensure normalized list
        unique_sorted = sorted(list({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()}))
        payload = {"id": ROW_ID, "tickers": unique_sorted}
        # upsert on primary key id
        res = client.table(TABLE_NAME).upsert(payload, on_conflict="id").execute()
        # success if no error raised and data present
        return True
    except Exception as e:
        print(f"ERROR [supabase_utils.save] {e}")
        return False


def load_scanner_rules_supabase() -> list[dict]:
    """Load scanner rules from Supabase. Returns [] if none found or any error.

    Expected row shape: { id: TEXT, rules: JSONB(list[dict]) }
    """
    client = _get_supabase_client()
    if client is None:
        return []
    try:
        res = client.table(SCANNER_TABLE_NAME).select("id,rules").eq("id", SCANNER_ROW_ID).execute()
        data = getattr(res, "data", None) or []
        if not data:
            return []
        row = data[0]
        rules = row.get("rules")
        # Handle JSON coming back as string
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                return []
        # Validate list-of-dicts shape (keep duplicates as-is)
        if isinstance(rules, list) and all(isinstance(r, dict) for r in rules):
            return rules
        return []
    except Exception as e:
        print(f"WARN [supabase_utils.load_scanner_rules] {e}")
        return []


def save_scanner_rules_supabase(rules: list[dict]) -> bool:
    """Upsert scanner rules into Supabase. Returns True on success.

    Does minimal validation and preserves ordering/duplicates.
    """
    client = _get_supabase_client()
    if client is None:
        return False
    try:
        # Ensure serializable list-of-dicts
        safe_rules: list[Dict[str, Any]] = []
        for r in rules or []:
            if isinstance(r, dict):
                # Copy shallow to avoid mutating caller
                entry = dict(r)
                # Normalize basic fields if present
                sym = entry.get("symbol")
                if isinstance(sym, str):
                    entry["symbol"] = sym.strip().upper()
                # Strategy normalization
                strat = entry.get("strategy")
                if isinstance(strat, str):
                    s = strat.strip().upper()
                    entry["strategy"] = "CALL" if s == "CALL" else "PUT"
                safe_rules.append(entry)
        payload = {"id": SCANNER_ROW_ID, "rules": safe_rules}
        client.table(SCANNER_TABLE_NAME).upsert(payload, on_conflict="id").execute()
        return True
    except Exception as e:
        print(f"ERROR [supabase_utils.save_scanner_rules] {e}")
        return False


def load_scanner_prefs_supabase() -> Dict[str, Any]:
    """Load scanner preferences (e.g., min_dte, max_dte) from Supabase.

    Expected row shape: { id: TEXT, prefs: JSONB(object) }
    Returns an empty dict on error or if none found.
    """
    client = _get_supabase_client()
    if client is None:
        return {}
    try:
        res = client.table(PREFS_TABLE_NAME).select("id,prefs").eq("id", PREFS_ROW_ID).execute()
        data = getattr(res, "data", None) or []
        if not data:
            return {}
        row = data[0]
        prefs = row.get("prefs")
        if isinstance(prefs, str):
            try:
                prefs = json.loads(prefs)
            except Exception:
                return {}
        return prefs if isinstance(prefs, dict) else {}
    except Exception as e:
        print(f"WARN [supabase_utils.load_scanner_prefs] {e}")
        return {}


def save_scanner_prefs_supabase(prefs: Dict[str, Any]) -> bool:
    """Upsert scanner preferences into Supabase. Returns True on success."""
    client = _get_supabase_client()
    if client is None:
        return False
    try:
        payload = {"id": PREFS_ROW_ID, "prefs": dict(prefs or {})}
        client.table(PREFS_TABLE_NAME).upsert(payload, on_conflict="id").execute()
        return True
    except Exception as e:
        print(f"ERROR [supabase_utils.save_scanner_prefs] {e}")
        return False
