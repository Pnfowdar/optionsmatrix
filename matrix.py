from datetime import datetime, date, time
from zoneinfo import ZoneInfo
import math, toml
import pandas as pd
from typing import List, Tuple, Dict, Any
import asyncio  # Async support
from marketdata_client import MarketDataClient

# Lightweight normal CDF (avoid SciPy dependency)
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# Load config with error handling
try:
    cfg = toml.load("config.toml")
    R = cfg.get("RISK_FREE_RATE", 0.045)
    POP_SCORE_BOUNDS = cfg.get("POP_SCORE_BOUNDS", [70.0, 80.0])
    YIELD_SCORE_BOUNDS = cfg.get("YIELD_SCORE_BOUNDS", [3.0, 8.0])
    MIN_POP = POP_SCORE_BOUNDS[0] / 100.0
    MAX_POP = POP_SCORE_BOUNDS[1] / 100.0
    MIN_YLD = YIELD_SCORE_BOUNDS[0] / 100.0
    MAX_YLD = YIELD_SCORE_BOUNDS[1] / 100.0
    print("DEBUG [matrix.py] Config loaded successfully.")
except FileNotFoundError:
    print("WARN [matrix.py] config.toml not found, using default values.")
    R = 0.045; MIN_POP, MAX_POP = 0.70, 0.80; MIN_YLD, MAX_YLD = 0.03, 0.08
except Exception as e:
    print(f"ERROR [matrix.py] loading config.toml: {e}. Using defaults.")
    R = 0.045; MIN_POP, MAX_POP = 0.70, 0.80; MIN_YLD, MAX_YLD = 0.03, 0.08


def _is_market_open() -> bool:
    """Checks if the US stock market is currently open."""
    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        return time(9, 30) <= now_et.time() <= time(16, 0)
    except Exception as e:
        print(f"ERROR [matrix.py] checking market open status: {e}")
        return False


def _filter_expiries(expiries: List[str], min_dte: int, max_dte: int) -> List[str]:
    """Filters expiry strings to valid Fridays within DTE range."""
    today = datetime.utcnow().date()
    valid_fridays: List[str] = []
    for e_str in expiries:
        try:
            expiry_date = date.fromisoformat(e_str)
            dte = (expiry_date - today).days
            if min_dte <= dte <= max_dte and expiry_date.weekday() == 4:
               valid_fridays.append(e_str)
        except Exception:
            continue
    return valid_fridays


def compute_score_matrix(monthly: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Computes a score based on normalized POP and Monthly Yield."""
    if monthly.empty or pop.empty:
        return pd.DataFrame()
    pop_f = pop / 100.0; yld_f = monthly / 100.0
    pop_range = MAX_POP - MIN_POP; yld_range = MAX_YLD - MIN_YLD
    # Ensure DataFrame shape even when range collapses
    if pop_range > 0:
        pop_n = ((pop_f - MIN_POP) / pop_range).clip(0, 1)
    else:
        pop_n = pd.DataFrame(0.5, index=pop_f.index, columns=pop_f.columns)
    if yld_range > 0:
        yld_n = ((yld_f - MIN_YLD) / yld_range).clip(0, 1)
    else:
        yld_n = pd.DataFrame(0.5, index=yld_f.index, columns=yld_f.columns)
    score = (pop_n + yld_n) / 2.0
    return score.astype(float).fillna(0.0)


async def build_matrix(
    symbol: str,
    client: MarketDataClient,
    strategy: str,
    target_delta: float,
    min_dte: int,
    max_dte: int,
    feed_type: str = 'sip',
    cache_ttl: int = 60
) -> Tuple[pd.DataFrame, list]:
    """Builds the options matrix data."""
    today = datetime.utcnow().date()
    is_open = _is_market_open()
    ttl = cache_ttl if is_open else max(cache_ttl * 10, 3600)

    expiries = await client.expirations(symbol)
    # pull the spot price directly from marketdata.app
    spot_price = await client.quote(symbol, feed=feed_type)
    filtered = _filter_expiries(expiries, min_dte, max_dte)
    if not filtered or spot_price <= 0:
        return pd.DataFrame(), []

    side = 'call' if strategy.upper() == 'CALL' else 'put'
    rows: List[Dict[str, Any]] = []

    for exp in filtered:
        try:
            chain = await client.chain(symbol, exp, side, feed=feed_type, ttl=ttl)
            exp_date = date.fromisoformat(exp)
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            T = dte / 365.0
        except Exception:
            continue

        for o in chain or []:
            try:
                K = float(o['strike']); bid = float(o['bid'])
                ask = float(o.get('ask', bid)); last = float(o.get('last', 0.0))
                delta_raw, iv_raw = o.get('delta'), o.get('iv')
                if not (K>0 and bid>=0 and isinstance(delta_raw,(int,float)) and isinstance(iv_raw,(int,float)) and iv_raw>=0):
                    continue
                delta, iv_val = float(delta_raw), float(iv_raw)

                mid = (bid + ask) / 2.0
                threshold = 0.5 * ask
                spread = max(ask - bid, 0.0)
                # If bid is zero, use zero to avoid confusing inflated estimates
                if bid == 0.0:
                    price_used = 0.0; method = "bid_zero"
                elif bid >= threshold:
                    price_used = mid; method = "midpoint"
                elif (threshold <= last <= ask) and (last >= bid):
                    price_used = last; method = "last_close"
                else:
                    # Fallback: use 20% on the bid price (not spread)
                    # Example: bid=0.01 -> price_used=0.012 (rounding for display may show 0.01)
                    price_used = bid * 1.2
                    # Clamp to [bid, ask] to avoid anomalies
                    price_used = min(max(price_used, bid), ask)
                    method = "bid_plus_20%"

                # Metrics based solely on price_used
                actual_yield = (price_used / K) * 100.0 if K>0 else 0.0
                monthly_yield = actual_yield * (30.0 / dte) if dte>0 else 0.0
                premium = price_used * 100.0
                # Additional metrics
                collateral = K * 100.0
                max_loss = max(collateral - premium, 0.0)
                breakeven = K - price_used if side=='put' else K + price_used
                breakeven_pct = 0.0
                try:
                    if spot_price > 0:
                        # Cushion from spot to breakeven for puts is (spot - BE)/spot
                        # For calls it's (BE - spot)/spot
                        diff = (spot_price - breakeven) if side=='put' else (breakeven - spot_price)
                        breakeven_pct = (diff / spot_price) * 100.0
                except Exception:
                    breakeven_pct = 0.0

                # POP calculation
                pop = 0.0
                if spot_price>0 and iv_val>0 and K>0 and T>0:
                    BE = (K + price_used) if side=='call' else max(K - price_used, 1e-9)
                    try:
                        d2 = (math.log(spot_price/BE) + (R - 0.5*iv_val**2)*T) / (iv_val*math.sqrt(T)) if iv_val>0 and T>0 else 0.0
                        pop = (1 - _norm_cdf(d2)) if side=='call' else _norm_cdf(d2)
                        pop = max(0.0, min(pop, 1.0))
                    except Exception:
                        pop = 0.0

                rows.append({
                    'expiry': exp,
                    'strike': K,
                    'delta': abs(delta),
                    'POP': round(pop*100,1),
                    'monthly_yield': round(monthly_yield,2),
                    'actual_yield': round(actual_yield,2),
                    'premium_dollars': round(premium,2),
                    'collateral': round(collateral,2),
                    'max_loss': round(max_loss,2),
                    'breakeven': round(breakeven,2),
                    'breakeven_pct': round(breakeven_pct,2),
                    'iv': round(iv_val*100,1),
                    'bid': round(bid,2),
                    'mid': round(mid,2),
                    'ask': round(ask,2),
                    'last': round(last,2),
                    'Price Used': round(price_used,2),
                    'Price Method': method,
                    'dte': dte
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(), []
    df = pd.DataFrame(rows)
    pivot = pd.pivot_table(df, index='expiry', columns='strike', values='monthly_yield', aggfunc='mean')
    return pivot, rows
