"""
Technical Analysis Tools for SEPA Trading Workflow

This module provides functions for calculating moving averages,
relative strength, and other technical indicators used in the
Minervini trend template analysis.
"""

import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_moving_averages(
    df: pd.DataFrame,
    periods: List[int] = [50, 150, 200]
) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) for given periods.
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of periods for SMA calculation
    
    Returns:
        DataFrame with added SMA columns (e.g., SMA_50, SMA_150, SMA_200)
    
    Example:
        >>> df = calculate_moving_averages(price_df)
        >>> print(df[['Close', 'SMA_50', 'SMA_150', 'SMA_200']].tail())
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    for period in periods:
        col_name = f"SMA_{period}"
        df[col_name] = df['Close'].rolling(window=period).mean()
    
    return df


def get_sma_trend_direction(
    df: pd.DataFrame,
    sma_period: int = 200,
    lookback_days: int = 21
) -> str:
    """
    Determine if an SMA is trending up or down.
    
    Args:
        df: DataFrame with SMA column
        sma_period: The SMA period to check (e.g., 200)
        lookback_days: Number of trading days to look back (21 ≈ 1 month)
    
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    col_name = f"SMA_{sma_period}"
    
    if col_name not in df.columns:
        df = calculate_moving_averages(df, periods=[sma_period])
    
    if len(df) < lookback_days + 1:
        return "UNKNOWN"
    
    sma_values = df[col_name].dropna()
    
    if len(sma_values) < lookback_days + 1:
        return "UNKNOWN"
    
    current_sma = sma_values.iloc[-1]
    past_sma = sma_values.iloc[-lookback_days]
    
    # Calculate percentage change
    pct_change = ((current_sma - past_sma) / past_sma) * 100
    
    # Threshold for determining trend (0.5% change is meaningful)
    if pct_change > 0.5:
        return "UP"
    elif pct_change < -0.5:
        return "DOWN"
    else:
        return "FLAT"


def calculate_52_week_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate 52-week high, low, and current price position.
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
    
    Returns:
        Dictionary with:
        - high_52w: 52-week high
        - low_52w: 52-week low
        - current_price: Latest close
        - pct_from_high: % distance from 52-week high
        - pct_above_low: % above 52-week low
    """
    if df is None or df.empty:
        return {}
    
    # Get last 252 trading days (approximately 1 year)
    lookback = min(252, len(df))
    recent_df = df.tail(lookback)
    
    high_52w = recent_df['High'].max()
    low_52w = recent_df['Low'].min()
    current_price = df['Close'].iloc[-1]
    
    # Calculate percentages
    pct_from_high = ((high_52w - current_price) / high_52w) * 100
    pct_above_low = ((current_price - low_52w) / low_52w) * 100
    
    return {
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'current_price': round(current_price, 2),
        'pct_from_high': round(pct_from_high, 2),
        'pct_above_low': round(pct_above_low, 2)
    }


def get_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    period_weeks: int = 52
) -> float:
    """
    Calculate Relative Strength (RS) ranking vs benchmark (0-100 scale).
    
    Args:
        stock_df: Stock price DataFrame
        benchmark_df: Benchmark (e.g., NIFTY 500) price DataFrame
        period_weeks: Lookback period in weeks
    
    Returns:
        RS ranking from 0 to 100 (higher = outperforming)
    
    Note:
        This is a simplified RS calculation. For true RS ranking,
        you'd compare against the entire universe of stocks.
    """
    if stock_df is None or benchmark_df is None:
        return 0.0
    
    if stock_df.empty or benchmark_df.empty:
        return 0.0
    
    # Trading days in period (5 days/week * weeks)
    lookback = min(period_weeks * 5, len(stock_df), len(benchmark_df))
    
    if lookback < 5:
        return 0.0
    
    # Get returns over the period
    stock_start = stock_df['Close'].iloc[-lookback]
    stock_end = stock_df['Close'].iloc[-1]
    stock_return = ((stock_end - stock_start) / stock_start) * 100
    
    benchmark_start = benchmark_df['Close'].iloc[-lookback]
    benchmark_end = benchmark_df['Close'].iloc[-1]
    benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
    
    # Calculate relative performance
    relative_perf = stock_return - benchmark_return
    
    # Convert to 0-100 scale
    # Assuming typical relative performance range of -50% to +50%
    rs_rating = 50 + relative_perf
    rs_rating = max(0, min(100, rs_rating))  # Clamp to 0-100
    
    return round(rs_rating, 1)


def calculate_rs_ranking_universe(
    stock_returns: Dict[str, float]
) -> Dict[str, int]:
    """
    Calculate RS ranking for a universe of stocks.
    
    Args:
        stock_returns: Dictionary of symbol -> return percentage
    
    Returns:
        Dictionary of symbol -> percentile ranking (0-100)
    """
    if not stock_returns:
        return {}
    
    # Sort by returns
    sorted_stocks = sorted(stock_returns.items(), key=lambda x: x[1])
    total = len(sorted_stocks)
    
    rankings = {}
    for rank, (symbol, _) in enumerate(sorted_stocks, 1):
        percentile = int((rank / total) * 100)
        rankings[symbol] = percentile
    
    return rankings


def check_rs_line_trend(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    min_uptrend_weeks: int = 6
) -> Dict[str, any]:
    """
    Check RS line trend direction per Minervini's workbook notes:
    - RS line should NOT be in a strong downtrend
    - Prefer RS line in an uptrend for at least 6 weeks, preferably 13+ weeks

    The RS line is computed as stock_price / benchmark_price.

    Args:
        stock_df: Stock price DataFrame
        benchmark_df: Benchmark price DataFrame
        min_uptrend_weeks: Minimum weeks the RS line should be trending up (default 6)

    Returns:
        Dictionary with:
        - rs_not_downtrending: bool (True if RS line is NOT in a strong downtrend)
        - rs_uptrend_weeks: int (number of weeks RS line has been in an uptrend)
        - rs_uptrend_sufficient: bool (True if uptrend >= min_uptrend_weeks)
    """
    result = {
        'rs_not_downtrending': True,
        'rs_uptrend_weeks': 0,
        'rs_uptrend_sufficient': False
    }

    if stock_df is None or benchmark_df is None:
        return result
    if stock_df.empty or benchmark_df.empty:
        return result

    # Align dates between stock and benchmark
    common_dates = stock_df.index.intersection(benchmark_df.index)
    if len(common_dates) < 30:  # Need at least ~6 weeks of data
        return result

    stock_close = stock_df.loc[common_dates, 'Close']
    bench_close = benchmark_df.loc[common_dates, 'Close']

    # Compute RS line (stock / benchmark ratio)
    rs_line = stock_close / bench_close

    # Smooth the RS line with a 5-day MA to reduce noise
    rs_smoothed = rs_line.rolling(window=5).mean().dropna()

    if len(rs_smoothed) < 30:
        return result

    # Check 1: RS line should NOT be in a strong downtrend
    # Look at the last 13 weeks (~65 trading days) for overall direction
    lookback_13w = min(65, len(rs_smoothed))
    rs_start_13w = rs_smoothed.iloc[-lookback_13w]
    rs_end = rs_smoothed.iloc[-1]
    pct_change_13w = ((rs_end - rs_start_13w) / rs_start_13w) * 100

    # Strong downtrend = RS line declined more than 10% over 13 weeks
    rs_not_downtrending = pct_change_13w > -10.0

    # Check 2: Count consecutive weeks of RS line uptrend
    # Resample to weekly and check consecutive positive changes
    weekly_rs = rs_smoothed.resample('W').last().dropna()
    weekly_changes = weekly_rs.diff().dropna()

    uptrend_weeks = 0
    for change in reversed(weekly_changes.values):
        if change > 0:
            uptrend_weeks += 1
        else:
            break

    result['rs_not_downtrending'] = rs_not_downtrending
    result['rs_uptrend_weeks'] = uptrend_weeks
    result['rs_uptrend_sufficient'] = uptrend_weeks >= min_uptrend_weeks

    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_volume_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume-based metrics.
    
    Args:
        df: DataFrame with Volume column
    
    Returns:
        Dictionary with volume metrics
    """
    if df is None or df.empty or 'Volume' not in df.columns:
        return {}
    
    vol = df['Volume']
    
    avg_volume_50 = vol.tail(50).mean()
    avg_volume_20 = vol.tail(20).mean()
    latest_volume = vol.iloc[-1]
    
    return {
        'avg_volume_50d': int(avg_volume_50),
        'avg_volume_20d': int(avg_volume_20),
        'latest_volume': int(latest_volume),
        'volume_ratio_50d': round(latest_volume / avg_volume_50, 2) if avg_volume_50 > 0 else 0,
        'volume_ratio_20d': round(latest_volume / avg_volume_20, 2) if avg_volume_20 > 0 else 0
    }


def check_trend_template(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
    rs_percentile: Optional[float] = None,
    low_cheat: bool = False
) -> Dict[str, any]:
    """
    Check all 8 points of Minervini's Trend Template.

    Args:
        df: Stock price DataFrame
        benchmark_df: Benchmark for RS calculation (optional, used only if rs_percentile not provided)
        config: Trend template criteria from config file
        rs_percentile: Pre-computed RS percentile ranking (0-100) from universe-wide ranking.
                       If provided, this is used instead of the simplified benchmark comparison.
        low_cheat: If True, exempts criterion 8 (price above 50-DMA) for "Low Cheat" setups
                   where the stock is pulling back to a rising 50-day MA.

    Returns:
        Dictionary with results for each criterion and overall pass/fail
    """
    # Default config values
    if config is None:
        config = {
            'price_above_150_sma': True,
            'price_above_200_sma': True,
            'sma_150_above_200': True,
            'sma_200_trending_up_months': 1,
            'sma_50_above_150_and_200': True,
            'price_above_50_sma': True,
            'price_above_52w_low_pct': 25,
            'price_within_52w_high_pct': 25,
            'rs_ranking_minimum': 70
        }
    
    results = {
        'checks': {},
        'all_passed': False
    }
    
    if df is None or df.empty or len(df) < 200:
        results['error'] = "Insufficient data (need 200+ days)"
        return results
    
    # Calculate indicators
    df = calculate_moving_averages(df, periods=[50, 150, 200])
    metrics_52w = calculate_52_week_metrics(df)
    
    current_price = df['Close'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_150 = df['SMA_150'].iloc[-1]
    sma_200 = df['SMA_200'].iloc[-1]
    
    # Check 1: Price above 150-day SMA
    check_1 = current_price > sma_150
    results['checks']['price_above_150_sma'] = {
        'passed': check_1,
        'value': f"Price: {current_price:.2f}, SMA150: {sma_150:.2f}"
    }
    
    # Check 2: Price above 200-day SMA
    check_2 = current_price > sma_200
    results['checks']['price_above_200_sma'] = {
        'passed': check_2,
        'value': f"Price: {current_price:.2f}, SMA200: {sma_200:.2f}"
    }
    
    # Check 3: 150-day SMA above 200-day SMA
    check_3 = sma_150 > sma_200
    results['checks']['sma_150_above_200'] = {
        'passed': check_3,
        'value': f"SMA150: {sma_150:.2f}, SMA200: {sma_200:.2f}"
    }
    
    # Check 4: 200-day SMA trending up
    # Resolve template variant to lookback months
    variant_months = {
        '1_month': 1,
        '4_month': 4,
        '5_month': 5,
    }
    template_variant = config.get('template_variant')
    if template_variant and template_variant in variant_months:
        trending_months = variant_months[template_variant]
    else:
        trending_months = config.get('sma_200_trending_up_months', 1)

    lookback = trending_months * 21  # ~21 trading days per month
    trend = get_sma_trend_direction(df, sma_period=200, lookback_days=lookback)
    check_4 = trend == "UP"
    variant_label = f" [{template_variant}]" if template_variant else ""
    results['checks']['sma_200_trending_up'] = {
        'passed': check_4,
        'value': f"Trend: {trend} (lookback: {trending_months} months{variant_label})"
    }
    
    # Check 5: 50-day SMA above 150 and 200
    check_5 = sma_50 > sma_150 and sma_50 > sma_200
    results['checks']['sma_50_above_150_and_200'] = {
        'passed': check_5,
        'value': f"SMA50: {sma_50:.2f}"
    }
    
    # Check 6: Price above 50-day SMA (exception: "Low Cheat" setups)
    price_above_50 = current_price > sma_50
    if low_cheat and not price_above_50:
        # Low Cheat exception: price can be at or slightly below 50-DMA
        check_6 = True
        results['checks']['price_above_50_sma'] = {
            'passed': True,
            'value': f"Price: {current_price:.2f}, SMA50: {sma_50:.2f} (Low Cheat exception applied)"
        }
    else:
        check_6 = price_above_50
        results['checks']['price_above_50_sma'] = {
            'passed': check_6,
            'value': f"Price: {current_price:.2f}, SMA50: {sma_50:.2f}"
        }
    
    # Check 7: Price at least X% above 52-week low
    min_above_low = config.get('price_above_52w_low_pct', 25)
    check_7 = metrics_52w.get('pct_above_low', 0) >= min_above_low
    results['checks']['price_above_52w_low'] = {
        'passed': check_7,
        'value': f"{metrics_52w.get('pct_above_low', 0):.1f}% above low (need {min_above_low}%)"
    }
    
    # Check 8: Price within X% of 52-week high
    max_from_high = config.get('price_within_52w_high_pct', 25)
    check_8 = metrics_52w.get('pct_from_high', 100) <= max_from_high
    results['checks']['price_within_52w_high'] = {
        'passed': check_8,
        'value': f"{metrics_52w.get('pct_from_high', 100):.1f}% from high (max {max_from_high}%)"
    }
    
    # Check 9 (Optional): RS Ranking
    min_rs = config.get('rs_ranking_minimum', 70)
    if rs_percentile is not None:
        # Use pre-computed universe-wide percentile ranking (preferred method)
        check_9 = rs_percentile >= min_rs
        results['checks']['rs_ranking'] = {
            'passed': check_9,
            'value': f"RS Percentile: {rs_percentile:.1f} (need {min_rs})"
        }
    elif benchmark_df is not None:
        # Fallback: simplified benchmark comparison
        rs = get_relative_strength(df, benchmark_df)
        check_9 = rs >= min_rs
        results['checks']['rs_ranking'] = {
            'passed': check_9,
            'value': f"RS (vs benchmark): {rs:.1f} (need {min_rs})"
        }
    else:
        check_9 = True  # Skip if no benchmark and no percentile
        results['checks']['rs_ranking'] = {
            'passed': True,
            'value': "Skipped (no benchmark or universe data)"
        }
    
    # Check 10 (Additional): RS line NOT in a strong downtrend
    # Check 11 (Additional): RS line uptrend duration (preferably 6+ weeks, ideally 13+)
    if benchmark_df is not None:
        min_rs_uptrend_weeks = config.get('rs_line_min_uptrend_weeks', 6)
        rs_trend = check_rs_line_trend(df, benchmark_df, min_uptrend_weeks=min_rs_uptrend_weeks)

        check_10 = rs_trend['rs_not_downtrending']
        results['checks']['rs_line_not_downtrending'] = {
            'passed': check_10,
            'value': f"RS line not in strong downtrend: {check_10}"
        }

        check_11 = rs_trend['rs_uptrend_sufficient']
        results['checks']['rs_line_uptrend'] = {
            'passed': check_11,
            'value': f"RS uptrend: {rs_trend['rs_uptrend_weeks']} weeks (need {min_rs_uptrend_weeks})"
        }
    else:
        check_10 = True  # Skip if no benchmark
        check_11 = True
        results['checks']['rs_line_not_downtrending'] = {
            'passed': True,
            'value': "Skipped (no benchmark)"
        }
        results['checks']['rs_line_uptrend'] = {
            'passed': True,
            'value': "Skipped (no benchmark)"
        }

    # Overall result
    # Core 8 criteria (checks 1-9 map to workbook criteria 1-8, with RS ranking)
    # + RS line trend checks (10-11) from additional workbook notes
    all_checks = [
        check_1, check_2, check_3, check_4, check_5, check_6,
        check_7, check_8, check_9, check_10, check_11
    ]
    results['all_passed'] = all(all_checks)
    results['passed_count'] = sum(all_checks)
    results['total_checks'] = len(all_checks)

    return results


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import yfinance as yf
    
    # Fetch sample data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period="2y")
    
    # Calculate SMAs
    df = calculate_moving_averages(df)
    print("Moving Averages calculated:")
    print(df[['Close', 'SMA_50', 'SMA_150', 'SMA_200']].tail())
    
    # 52-week metrics
    metrics = calculate_52_week_metrics(df)
    print(f"\n52-week metrics: {metrics}")
    
    # SMA trend
    trend = get_sma_trend_direction(df, sma_period=200)
    print(f"\n200-day SMA trend: {trend}")
    
    # Full trend template check
    results = check_trend_template(df)
    print(f"\nTrend Template Results:")
    print(f"Overall: {'PASS' if results['all_passed'] else 'FAIL'}")
    print(f"Passed {results['passed_count']}/{results['total_checks']} checks")
