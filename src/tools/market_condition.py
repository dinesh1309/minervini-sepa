"""
Market Condition Detection for SEPA Trading Workflow

Detects the 4 market conditions per Minervini/MarketSmith methodology:
1. Confirmed Uptrend — FTD confirmed, index above key MAs
2. Uptrend Under Pressure — too many distribution days
3. Rally Attempt — bouncing from low, waiting for FTD
4. Downtrend — below 200 SMA or rally attempt failed

Also detects Follow Through Days (FTD) and Distribution Days.

NOTE: This is an APPROXIMATION of MarketSmith's proprietary model.
MarketSmith uses additional proprietary signals. This detector will be
directionally correct but may not match MarketSmith exactly.
Always cross-reference with MarketSmith for final confirmation.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    """Result of market condition analysis."""
    condition: str  # "CONFIRMED_UPTREND", "UNDER_PRESSURE", "RALLY_ATTEMPT", "DOWNTREND"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    index_name: str
    index_close: float
    as_of: str  # Date of the data
    sma_50: float
    sma_200: float
    above_50: bool
    above_200: bool
    distribution_days_25: int  # Distribution days in last 25 sessions
    rally_attempt_day: int  # 0 if not in rally attempt, else day count
    ftd_detected: bool
    ftd_date: Optional[str]
    details: str


def count_distribution_days(df: pd.DataFrame, lookback: int = 25) -> int:
    """
    Count distribution days in the last N sessions.

    A distribution day = index closes DOWN on HIGHER volume than previous day.
    Minervini/IBD: 4-5 distribution days in 25 sessions = uptrend under pressure.

    Args:
        df: DataFrame with Close and Volume columns
        lookback: Number of sessions to check

    Returns:
        Number of distribution days
    """
    recent = df.tail(lookback + 1)  # +1 to compare with previous day
    if len(recent) < 2:
        return 0

    dist_days = 0
    for i in range(1, len(recent)):
        price_down = recent['Close'].iloc[i] < recent['Close'].iloc[i-1]
        volume_up = recent['Volume'].iloc[i] > recent['Volume'].iloc[i-1]
        # Meaningful decline (at least 0.2%)
        pct_change = (recent['Close'].iloc[i] - recent['Close'].iloc[i-1]) / recent['Close'].iloc[i-1]
        meaningful = pct_change < -0.002

        if price_down and volume_up and meaningful:
            dist_days += 1

    return dist_days


def detect_rally_attempt(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    Detect if market is in a Rally Attempt and count days.

    Rally Attempt starts when:
    - Index makes a new low within recent history
    - Then has an up day (Day 1 of Rally Attempt)
    - Count continues as long as index holds above Day 1 low
    - If index undercuts Day 1 low → reset

    Args:
        df: DataFrame with OHLCV data
        lookback: Days to look back for the low

    Returns:
        Dict with rally_attempt status, day count, key levels
    """
    recent = df.tail(lookback)
    if len(recent) < 5:
        return {"in_rally_attempt": False, "day_count": 0}

    # Find the lowest low in recent history
    low_idx = recent['Low'].idxmin()
    low_val = recent['Low'].min()
    low_pos = recent.index.get_loc(low_idx)

    # Look for first up day after the low
    rally_start = None
    rally_start_low = None
    for i in range(low_pos + 1, len(recent)):
        if recent['Close'].iloc[i] > recent['Close'].iloc[i-1]:
            rally_start = i
            rally_start_low = recent['Low'].iloc[i]
            break

    if rally_start is None:
        return {"in_rally_attempt": False, "day_count": 0, "low": float(low_val)}

    # Count days holding above the rally attempt low
    day_count = 0
    for i in range(rally_start, len(recent)):
        if recent['Low'].iloc[i] < low_val:
            # Undercut — rally attempt killed
            return {
                "in_rally_attempt": False,
                "day_count": 0,
                "killed": True,
                "low": float(low_val),
                "killed_date": str(recent.index[i])[:10]
            }
        day_count += 1

    return {
        "in_rally_attempt": True,
        "day_count": day_count,
        "start_date": str(recent.index[rally_start])[:10],
        "low": float(low_val),
        "key_level": float(low_val),  # Must hold above this
    }


def detect_ftd(df: pd.DataFrame, rally_info: Dict) -> Dict[str, Any]:
    """
    Detect Follow Through Day within a Rally Attempt.

    FTD criteria (Minervini/MarketSmith):
    - Must be Day 4 or later of Rally Attempt
    - Index up 1.5%+ from previous close
    - Volume higher than previous day

    Args:
        df: DataFrame with OHLCV data
        rally_info: Output from detect_rally_attempt()

    Returns:
        Dict with FTD detection result
    """
    if not rally_info.get("in_rally_attempt") or rally_info.get("day_count", 0) < 4:
        return {"ftd_detected": False, "reason": "Not in rally attempt or < Day 4"}

    recent = df.tail(20)
    start_date = rally_info.get("start_date")

    if not start_date:
        return {"ftd_detected": False, "reason": "No start date"}

    # Find the rally start in the dataframe
    for i in range(1, len(recent)):
        date_str = str(recent.index[i])[:10]
        prev_close = recent['Close'].iloc[i-1]
        curr_close = recent['Close'].iloc[i]
        prev_vol = recent['Volume'].iloc[i-1]
        curr_vol = recent['Volume'].iloc[i]

        pct_change = (curr_close - prev_close) / prev_close * 100

        if pct_change >= 1.5 and curr_vol > prev_vol:
            # Check if this is Day 4+ of rally
            # Simple: if we're in a rally attempt with 4+ days, any qualifying day is FTD
            if rally_info["day_count"] >= 4:
                return {
                    "ftd_detected": True,
                    "date": date_str,
                    "pct_change": round(pct_change, 2),
                    "volume_increase": True,
                }

    return {"ftd_detected": False, "reason": "No qualifying day found (need +1.5% on higher volume, Day 4+)"}


def detect_market_condition(
    df: pd.DataFrame,
    index_name: str = "Index",
    sma_periods: List[int] = [50, 200]
) -> MarketCondition:
    """
    Detect current market condition from index OHLCV data.

    Logic:
    1. Calculate SMAs
    2. Check price position vs SMAs
    3. Count distribution days
    4. Check for rally attempt
    5. Check for FTD
    6. Determine condition

    Args:
        df: DataFrame with OHLCV data (at least 200 rows)
        index_name: Name for display (e.g., "Nifty 50", "S&P 500")
        sma_periods: SMA periods to calculate

    Returns:
        MarketCondition dataclass
    """
    if df is None or len(df) < 200:
        return MarketCondition(
            condition="UNKNOWN",
            confidence="LOW",
            index_name=index_name,
            index_close=0,
            as_of="",
            sma_50=0, sma_200=0,
            above_50=False, above_200=False,
            distribution_days_25=0,
            rally_attempt_day=0,
            ftd_detected=False, ftd_date=None,
            details="Insufficient data (need 200+ rows)"
        )

    # Calculate SMAs
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    last = df.iloc[-1]
    close = float(last['Close'])
    sma50 = float(last['SMA_50'])
    sma200 = float(last['SMA_200'])
    as_of = str(df.index[-1])[:10]

    above_50 = close > sma50
    above_200 = close > sma200
    sma50_above_200 = sma50 > sma200

    # Count distribution days
    dist_days = count_distribution_days(df, lookback=25)

    # Detect rally attempt
    rally_info = detect_rally_attempt(df, lookback=20)
    rally_day = rally_info.get("day_count", 0) if rally_info.get("in_rally_attempt") else 0

    # Detect FTD
    ftd_info = detect_ftd(df, rally_info)
    ftd_detected = ftd_info.get("ftd_detected", False)
    ftd_date = ftd_info.get("date")

    # DETERMINE CONDITION
    #
    # Decision tree:
    #
    #   Is price above 200 SMA?
    #   ├── YES: Is price above 50 SMA AND 50 > 200?
    #   │   ├── YES: Distribution days < 5?
    #   │   │   ├── YES → CONFIRMED UPTREND
    #   │   │   └── NO → UNDER PRESSURE
    #   │   └── NO (above 200 but below 50):
    #   │       └── UNDER PRESSURE
    #   └── NO (below 200 SMA):
    #       ├── In Rally Attempt?
    #       │   ├── YES: FTD detected?
    #       │   │   ├── YES → CONFIRMED UPTREND (fresh)
    #       │   │   └── NO → RALLY ATTEMPT (Day X)
    #       │   └── NO → DOWNTREND
    #       └── DOWNTREND

    if ftd_detected:
        condition = "CONFIRMED_UPTREND"
        confidence = "MEDIUM"  # Fresh FTD, needs confirmation
        details = f"FTD detected on {ftd_date}. Fresh uptrend — progressive exposure 25-50%."
    elif above_200 and above_50 and sma50_above_200:
        if dist_days < 5:
            condition = "CONFIRMED_UPTREND"
            confidence = "HIGH"
            details = f"Price above all key MAs. {dist_days} distribution days (safe < 5). Full deployment allowed."
        else:
            condition = "UNDER_PRESSURE"
            confidence = "HIGH"
            details = f"Above MAs but {dist_days} distribution days in 25 sessions. Reduce exposure, tighten stops."
    elif above_200 and not above_50:
        condition = "UNDER_PRESSURE"
        confidence = "MEDIUM"
        details = f"Above 200 SMA but below 50 SMA. Weakening. {dist_days} distribution days."
    elif rally_info.get("in_rally_attempt"):
        condition = "RALLY_ATTEMPT"
        confidence = "MEDIUM"
        key_level = rally_info.get("key_level", 0)
        details = f"Rally Attempt Day {rally_day}. Must hold above {key_level:.2f}. Waiting for FTD (Day 4+, +1.5% on higher volume)."
    else:
        condition = "DOWNTREND"
        confidence = "HIGH"
        if rally_info.get("killed"):
            details = f"Rally attempt killed on {rally_info.get('killed_date', '?')}. Below key MAs. 0% new positions."
        else:
            details = f"Below 200 SMA ({sma200:.2f}). {dist_days} distribution days. 0% new positions — capital preservation."

    return MarketCondition(
        condition=condition,
        confidence=confidence,
        index_name=index_name,
        index_close=round(close, 2),
        as_of=as_of,
        sma_50=round(sma50, 2),
        sma_200=round(sma200, 2),
        above_50=above_50,
        above_200=above_200,
        distribution_days_25=dist_days,
        rally_attempt_day=rally_day,
        ftd_detected=ftd_detected,
        ftd_date=ftd_date,
        details=details
    )


def format_condition_emoji(condition: str) -> str:
    """Get emoji for market condition."""
    return {
        "CONFIRMED_UPTREND": "UPTREND",
        "UNDER_PRESSURE": "CAUTION",
        "RALLY_ATTEMPT": "WATCHING",
        "DOWNTREND": "DOWNTREND",
        "UNKNOWN": "UNKNOWN",
    }.get(condition, "?")


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
    from src.tools.market_data import get_stock_data

    for symbol, name in [("^NSEI", "Nifty 50"), ("^GSPC", "S&P 500")]:
        print(f"\n=== {name} ===")
        df = get_stock_data(symbol, period='1y', auto_suffix=False)
        if df is not None:
            mc = detect_market_condition(df, index_name=name)
            print(f"Condition: {mc.condition} ({mc.confidence})")
            print(f"Close: {mc.index_close} | SMA50: {mc.sma_50} | SMA200: {mc.sma_200}")
            print(f"Above 50: {mc.above_50} | Above 200: {mc.above_200}")
            print(f"Distribution days (25): {mc.distribution_days_25}")
            print(f"Rally Attempt Day: {mc.rally_attempt_day}")
            print(f"FTD: {mc.ftd_detected} ({mc.ftd_date})")
            print(f"Details: {mc.details}")
