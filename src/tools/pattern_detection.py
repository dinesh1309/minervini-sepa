"""
VCP Pattern Detection Tools for SEPA Trading Workflow

This module provides functions to detect Volatility Contraction Patterns (VCP),
a key component of the Minervini SEPA methodology.

A VCP is characterized by:
- Successive price contractions (T1 > T2 > T3...)
- Each contraction is smaller than the previous
- Volume dries up during consolidation
- Base length typically 3-6 weeks
"""

import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Contraction:
    """Represents a single contraction in a VCP pattern."""
    start_idx: int
    end_idx: int
    high: float
    low: float
    depth_pct: float
    avg_volume: float


@dataclass
class VCPPattern:
    """Represents a detected VCP pattern."""
    symbol: str
    is_valid: bool
    contractions: List[Contraction]
    base_length_days: int
    pivot_price: float
    volume_dry_up: bool
    score: float  # Quality score 0-100
    details: str


def find_swing_highs_lows(
    df: pd.DataFrame,
    window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Find local swing highs and lows in price data.
    
    Args:
        df: DataFrame with 'High' and 'Low' columns
        window: Number of bars on each side to compare
    
    Returns:
        Tuple of (swing_highs, swing_lows) as boolean Series
    """
    highs = df['High']
    lows = df['Low']
    
    swing_highs = pd.Series(False, index=df.index)
    swing_lows = pd.Series(False, index=df.index)
    
    for i in range(window, len(df) - window):
        # Check if this is a swing high
        if all(highs.iloc[i] >= highs.iloc[i-window:i]) and \
           all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
            swing_highs.iloc[i] = True
        
        # Check if this is a swing low
        if all(lows.iloc[i] <= lows.iloc[i-window:i]) and \
           all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
            swing_lows.iloc[i] = True
    
    return swing_highs, swing_lows


def identify_contractions(
    df: pd.DataFrame,
    lookback_days: int = 65
) -> List[Contraction]:
    """
    Identify price contractions in the recent price history.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_days: Number of days to analyze
    
    Returns:
        List of Contraction objects
    """
    contractions = []
    
    # Use recent data
    recent = df.tail(lookback_days).copy()
    
    if len(recent) < 20:
        return contractions
    
    # Find swing points
    swing_highs, swing_lows = find_swing_highs_lows(recent, window=3)
    
    high_indices = recent.index[swing_highs].tolist()
    low_indices = recent.index[swing_lows].tolist()
    
    if len(high_indices) < 2 or len(low_indices) < 1:
        return contractions
    
    # Identify contractions between swing highs
    for i in range(len(high_indices) - 1):
        start_idx = high_indices[i]
        end_idx = high_indices[i + 1]
        
        # Get data between highs
        mask = (recent.index >= start_idx) & (recent.index <= end_idx)
        segment = recent[mask]
        
        if len(segment) < 3:
            continue
        
        high = segment['High'].max()
        low = segment['Low'].min()
        depth_pct = ((high - low) / high) * 100
        avg_volume = segment['Volume'].mean()
        
        # Find integer index for start/end
        start_int = recent.index.get_loc(start_idx)
        end_int = recent.index.get_loc(end_idx)
        
        contractions.append(Contraction(
            start_idx=start_int,
            end_idx=end_int,
            high=high,
            low=low,
            depth_pct=round(depth_pct, 2),
            avg_volume=avg_volume
        ))
    
    return contractions


def check_volume_dry_up(
    df: pd.DataFrame,
    threshold: float = 0.5,
    lookback: int = 20
) -> Tuple[bool, float]:
    """
    Check if volume has dried up in recent consolidation.
    
    Args:
        df: DataFrame with Volume column
        threshold: Volume should be below this ratio of 50-day average
        lookback: Days to check for dry-up
    
    Returns:
        Tuple of (is_dry, volume_ratio)
    """
    if len(df) < 50:
        return False, 1.0
    
    avg_volume_50 = df['Volume'].tail(50).mean()
    recent_volume = df['Volume'].tail(lookback).mean()
    
    ratio = recent_volume / avg_volume_50 if avg_volume_50 > 0 else 1.0
    is_dry = ratio < threshold
    
    return is_dry, round(ratio, 2)


def detect_vcp(
    df: pd.DataFrame,
    symbol: str = "",
    config: Optional[Dict] = None
) -> VCPPattern:
    """
    Detect Volatility Contraction Pattern in price data.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol for reference
        config: VCP criteria configuration
    
    Returns:
        VCPPattern object with detection results
    """
    if config is None:
        config = {
            'min_contractions': 2,
            'max_first_contraction_pct': 35,
            'contraction_ratio': 0.5,
            'volume_dry_up_threshold': 0.5,
            'base_length_days_min': 20,
            'base_length_days_max': 65
        }
    
    # Default invalid result
    invalid_result = VCPPattern(
        symbol=symbol,
        is_valid=False,
        contractions=[],
        base_length_days=0,
        pivot_price=0.0,
        volume_dry_up=False,
        score=0.0,
        details="Insufficient data"
    )
    
    if df is None or len(df) < config['base_length_days_min']:
        return invalid_result
    
    # Identify contractions
    lookback = config.get('base_length_days_max', 65)
    contractions = identify_contractions(df, lookback_days=lookback)
    
    if len(contractions) < config['min_contractions']:
        invalid_result.details = f"Only {len(contractions)} contractions found (need {config['min_contractions']})"
        return invalid_result
    
    # Check first contraction depth
    if contractions[0].depth_pct > config['max_first_contraction_pct']:
        invalid_result.details = f"First contraction too deep: {contractions[0].depth_pct}%"
        invalid_result.contractions = contractions
        return invalid_result
    
    # Check contraction ratio (each subsequent should be smaller)
    valid_ratio = True
    for i in range(1, len(contractions)):
        expected_max = contractions[i-1].depth_pct * config['contraction_ratio'] * 1.5  # Some tolerance
        if contractions[i].depth_pct > expected_max:
            valid_ratio = False
            break
    
    # MINERVINI RULE: Check for progressively smaller contractions
    # Each contraction MUST be smaller than the previous one.
    # Allow some tolerance (up to 1.2x previous — not strict 50% but must trend down)
    decreasing = True
    decreasing_count = 0
    increasing_count = 0
    for i in range(1, len(contractions)):
        if contractions[i].depth_pct < contractions[i-1].depth_pct:
            decreasing_count += 1
        else:
            increasing_count += 1
            decreasing = False

    # Check if OVERALL trend is contracting (last < first, even if not every step)
    overall_contracting = False
    if len(contractions) >= 2:
        overall_contracting = contractions[-1].depth_pct < contractions[0].depth_pct

    # MINERVINI RULE: Volume should decrease with contractions
    volume_decreasing = True
    if len(contractions) >= 2:
        for i in range(1, len(contractions)):
            if contractions[i].avg_volume > contractions[i-1].avg_volume * 1.2:
                volume_decreasing = False
                break

    # Check overall volume dry-up
    volume_dry, volume_ratio = check_volume_dry_up(
        df,
        threshold=config['volume_dry_up_threshold']
    )

    # Calculate base length
    if contractions:
        base_start = contractions[0].start_idx
        base_end = contractions[-1].end_idx
        base_length = base_end - base_start
    else:
        base_length = 0

    # Determine pivot price (resistance at the tightest contraction)
    pivot_price = 0.0
    if contractions:
        # Pivot = high of the tightest (smallest depth) contraction
        tightest = min(contractions, key=lambda c: c.depth_pct)
        pivot_price = tightest.high

    # Calculate quality score (Minervini-strict)
    #
    # VCP SCORING:
    #   GATE (must pass all to be valid):
    #     - 2+ contractions
    #     - First contraction <= 35%
    #     - Overall contracting (last depth < first depth)
    #   QUALITY (determines score):
    #     30 pts: Every contraction smaller than previous (strict decreasing)
    #     20 pts: Overall contracting (last < first, even if not every step)
    #     20 pts: Volume decreasing across contractions
    #     15 pts: Volume dried up (recent < 50% of 50-day avg)
    #     15 pts: Base length in valid range (20-65 days)
    #
    score = 0.0
    has_min_contractions = len(contractions) >= config['min_contractions']
    first_depth_ok = contractions[0].depth_pct <= config['max_first_contraction_pct'] if contractions else False

    if has_min_contractions and first_depth_ok:
        if decreasing:
            score += 30  # Strict: every contraction smaller
        elif overall_contracting:
            score += 20  # Looser: last < first overall

        if volume_decreasing:
            score += 20  # Volume tightening with price

        if volume_dry:
            score += 15  # Recent volume dried up

        if config['base_length_days_min'] <= base_length <= config['base_length_days_max']:
            score += 15  # Good base length

    # VALIDITY: Must have gate conditions + minimum score
    is_valid = (
        has_min_contractions and
        first_depth_ok and
        overall_contracting and  # GATE: must be contracting overall
        score >= 40
    )

    details = f"{len(contractions)} contractions, "
    details += f"depths: {[round(float(c.depth_pct), 1) for c in contractions]}, "
    if decreasing:
        details += "strictly decreasing, "
    elif overall_contracting:
        details += "overall contracting (not strict), "
    else:
        details += "NOT contracting, "
    details += f"vol {'decreasing' if volume_decreasing else 'NOT decreasing'}, "
    details += f"vol ratio: {volume_ratio}, "
    details += f"base: {base_length} days"
    
    return VCPPattern(
        symbol=symbol,
        is_valid=is_valid,
        contractions=contractions,
        base_length_days=base_length,
        pivot_price=round(pivot_price, 2),
        volume_dry_up=volume_dry,
        score=round(score, 1),
        details=details
    )


def identify_pivot(
    df: pd.DataFrame,
    vcp: Optional[VCPPattern] = None
) -> Dict[str, Any]:
    """
    Identify the breakout pivot price from consolidation.
    
    Args:
        df: DataFrame with OHLCV data
        vcp: Optional VCP pattern (uses its pivot if available)
    
    Returns:
        Dictionary with pivot information
    """
    if vcp and vcp.pivot_price > 0:
        pivot = vcp.pivot_price
    else:
        # Find resistance from recent highs
        recent = df.tail(20)
        pivot = recent['High'].max()
    
    current_price = df['Close'].iloc[-1]
    distance_pct = ((pivot - current_price) / current_price) * 100
    
    return {
        'pivot_price': round(pivot, 2),
        'current_price': round(current_price, 2),
        'distance_to_pivot_pct': round(distance_pct, 2),
        'near_pivot': distance_pct <= 5  # Within 5% of pivot
    }


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Test on a stock
    symbol = "TITAN.NS"
    print(f"Detecting VCP pattern for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")
    
    vcp = detect_vcp(df, symbol=symbol)
    print(f"VCP Valid: {vcp.is_valid}")
    print(f"Score: {vcp.score}")
    print(f"Details: {vcp.details}")
    print(f"Pivot: {vcp.pivot_price}")
    
    pivot_info = identify_pivot(df, vcp)
    print(f"Pivot Info: {pivot_info}")
