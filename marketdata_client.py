# marketdata_client.py

import httpx
import asyncio
import json
import pandas as pd
from aiolimiter import AsyncLimiter
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import toml
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

BASE = "https://api.marketdata.app/v1"
LIMIT = AsyncLimiter(120, 60) # Rate limit (adjust per your plan)
CACHE = Cache(Cache.MEMORY, serializer=JsonSerializer()) # In-memory cache

class MarketDataClient:
    """Async wrapper for marketdata.app endpoints with caching and feed support."""

    def __init__(self, token: str):
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0, # Adjust timeout if needed
            follow_redirects=True # Allow redirects
        )
        self.token = token
        print("MarketDataClient initialized.")

    async def _get(self, path: str, params: Dict[str, Any] | None = None, ttl: int = 300, feed: Optional[str] = None):
        """Internal GET handler with caching, rate limiting, feed support, and DEBUG prints."""
        url = f"{BASE}{path}"
        request_params = params.copy() if params else {}
        if feed:
            request_params['feed'] = feed

        key_params = tuple(sorted(request_params.items()))
        cache_key = (url, key_params)
        # print(f"DEBUG [_get] Using Cache Key: {cache_key}") # Less verbose cache key log

        # 1. Check cache
        if ttl > 0:
            try:
                cached_data = await CACHE.get(cache_key)
                if cached_data is not None:
                    # print(f"DEBUG [_get] Cache HIT for {cache_key}") # Indicate cache hit
                    return cached_data
                # else:
                    # print(f"DEBUG [_get] Cache MISS for {cache_key}") # Indicate cache miss
            except Exception as e:
                print(f"WARN: Cache GET error for key {cache_key}: {e}")

        # 2. Fetch from API
        print(f"DEBUG [_get] Fetching from API: {url} | Params: {request_params}")
        async with LIMIT:
            response_obj = None
            try:
                r = await self.http.get(url, params=request_params)
                response_obj = r
                print(f"DEBUG [_get] API Response Status: {r.status_code} for {url} (Feed: {feed})") # Log status code including feed

                # --- HANDLE CACHED FEED STATUS CODES ---
                # 204 NO CONTENT means cache miss on API side, return empty dict
                if r.status_code == 204:
                    print(f"INFO [_get] Received 204 NO CONTENT for {url} (Feed: {feed}). Cache empty on server.")
                    # No need to cache this empty response, just return it
                    return {}
                # 203 NON-AUTHORITATIVE INFO means cache hit on API side, proceed normally
                elif r.status_code == 203:
                     print(f"INFO [_get] Received 203 NON-AUTHORITATIVE INFO for {url} (Feed: {feed}). Served from API cache.")
                     # Proceed to parse JSON, status is considered success by raise_for_status

                # Raise errors for actual client/server issues (4xx/5xx)
                r.raise_for_status()

            except httpx.HTTPStatusError as e:
                print(f"ERROR: HTTP Status {e.response.status_code} fetching {e.request.url}. Response: {e.response.text[:500]}")
                return {}
            except httpx.RequestError as e:
                print(f"ERROR: Network Request error for {e.request.url}: {e}")
                return {}
            except Exception as e:
                print(f"ERROR: Unexpected error during HTTP request for {url}: {e}")
                if response_obj: print(f"ERROR Context: Status={response_obj.status_code}, Text={response_obj.text[:300]}")
                return {}

        # 3. Parse JSON (only if status code was not 204)
        data = {}
        try:
            # If we reach here, status was likely 200 or 203, expecting JSON body
            data = r.json()
            if not isinstance(data, (dict, list)):
                 print(f"WARN: Expected dict or list after JSON parse for {url}, got {type(data)}. Response: {str(data)[:200]}")
                 return {}
            # print(f"DEBUG [_get] Parsed JSON type: {type(data)} for {url}") # Less verbose
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON received from {url} (Status: {r.status_code}). Cannot decode. Response: {r.text[:500]}")
            return {}
        except Exception as e:
            print(f"ERROR: Unexpected error parsing JSON for {url}: {e}")
            return {}

        # 4. Cache the result (only if data is not empty and TTL > 0)
        if ttl > 0 and data:
            try:
                await CACHE.set(cache_key, data, ttl=ttl)
                # print(f"DEBUG [_get] Data cached for {cache_key} with TTL={ttl}s") # Less verbose
            except Exception as e:
                 print(f"WARN: Cache SET error for key {cache_key}: {e}")

        return data

    async def expirations(self, symbol: str) -> List[str]:
        """Fetches expiration dates (cached). Uses default live feed."""
        # print(f"DEBUG [expirations] Fetching expirations for {symbol}") # Less verbose
        # Expirations typically don't need the 'cached' feed override
        data = await self._get(f"/options/expirations/{symbol}/", ttl=3600) # Cache for 1 hour
        exps = data.get("expirations", []) if isinstance(data, dict) else []
        # print(f"DEBUG [expirations] Got {len(exps)} expirations for {symbol}") # Less verbose
        return exps

    async def quote(
        self,
        symbol: str,
        feed: str = "cached",
        ttl: int = 60
    ) -> float:
        """
        Fetch the latest spot price for `symbol` from marketdata.app.
        Calls GET /v1/stocks/quotes/{symbol}/?feed={feed}.
        Returns the first element of the "last" array, or 0.0 on any error.
        """
        # 1) Hit the stocks quotes endpoint
        data = await self._get(f"/stocks/quotes/{symbol}/", feed=feed, ttl=ttl)
        # 2) Ensure we got an OK response
        if not isinstance(data, dict) or data.get("s") != "ok":
            return 0.0
        # 3) Pull the first element of the "last" list
        last_arr = data.get("last", [])
        if isinstance(last_arr, list) and last_arr and isinstance(last_arr[0], (int, float)):
            return float(last_arr[0])
        return 0.0


    async def chain(self, symbol: str, expiry: str, side: str, feed: str = 'cached', greeks: bool = True, ttl: int = 60) -> List[Dict[str, Any]]:
        """Fetches option chain (cached). Default feed is now 'cached'."""
        # print(f"DEBUG [chain] Fetching chain for {symbol} {expiry} {side} (Feed: {feed}, TTL: {ttl})") # Less verbose
        params = {"expiration": expiry, "side": side}
        if greeks: params["greeks"] = "true"

        # Call _get, passing the feed parameter specified (which defaults to 'cached' here)
        data = await self._get(f"/options/chain/{symbol}/", params=params, feed=feed, ttl=ttl)

        # --- Processing logic remains the same ---
        if not isinstance(data, dict):
            # print(f"WARN [chain] Expected dict response for chain {symbol} {expiry}, got {type(data)}. Returning empty list.")
            return []
        if data.get('s') == 'no_data':
            # print(f"INFO [chain] API reported 'no_data' for {symbol} {expiry} {side}.")
            return []
        essential_keys = ['strike', 'bid', 'ask', 'delta', 'optionSymbol', 'iv']
        if not all(k in data for k in essential_keys):
            # print(f"WARN [chain] Missing essential keys in chain data dict for {symbol} {expiry} {side}. Keys present: {list(data.keys())}. Returning empty list.")
            return []
        try:
            df = pd.DataFrame(data)
            if df.empty:
                # print(f"DEBUG [chain] DataFrame created but is empty for {symbol} {expiry} {side}.")
                return []
            records = df.to_dict(orient="records")
            # print(f"DEBUG [chain] Successfully processed {len(records)} contracts for {symbol} {expiry} {side}.") # Less verbose
            return records
        except ValueError as e:
             print(f"ERROR [chain] Could not convert chain data to DataFrame for {symbol} {expiry} {side}. Error: {e}. Data snippet: {str(data)[:300]}")
             return []
        except Exception as e:
            print(f"ERROR [chain] Unexpected error processing chain data for {symbol} {expiry} {side}: {e}")
            return []

    # --- iv_rank and get_current_iv remain unchanged, using default live feed ---
    async def iv_rank(self, symbol: str) -> float:
        today = datetime.utcnow().date()
        start = today - timedelta(days=365)
        try:
            data = await self._get(f"/impliedvol/{symbol}", params={"date": start.isoformat()}, ttl=14400) # 4 hour TTL
        except Exception as e:
            print(f"ERROR: Failed to fetch IV history for {symbol}: {e}")
            return 0.0
        if not isinstance(data, dict): return 0.0
        history = data.get("impliedVolatility", [])
        ivs = [pt.get("iv") for pt in history if isinstance(pt, dict) and isinstance(pt.get("iv"), (float, int))]
        if len(ivs) < 2: return 0.0
        iv_now = ivs[-1]
        iv_min, iv_max = min(ivs), max(ivs)
        return round((iv_now - iv_min) / (iv_max - iv_min) * 100, 1) if iv_max != iv_min else 0.0

    async def get_current_iv(self, symbol: str) -> float:
        try:
            data = await self._get(f"/impliedvol/{symbol}", ttl=900) # 15 min TTL
            if not isinstance(data, dict): return 0.0
            history = data.get("impliedVolatility", [])
            if history and isinstance(history, list):
                last_point = history[-1]
                if isinstance(last_point, dict):
                    iv = last_point.get("iv")
                    if isinstance(iv, (int, float)): return round(iv * 100, 1)
        except Exception as e: print(f"Error fetching current IV for {symbol}: {e}")
        return 0.0

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.http.aclose()
        print("MarketDataClient HTTP client closed.")

async def clear_marketdata_cache():
    """Clears the aiocache used by MarketDataClient."""
    try:
        await CACHE.clear()
        print("MarketDataClient cache cleared successfully via clear_marketdata_cache().")
    except Exception as e:
        print(f"Error clearing MarketDataClient cache: {e}")