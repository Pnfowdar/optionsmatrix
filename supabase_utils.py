# supabase_utils.py

from __future__ import annotations
from typing import List, Optional
import json
import streamlit as st

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

TABLE_NAME = "watchlist"
ROW_ID = "default"  # single-row storage for this app instance


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
