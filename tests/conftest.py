"""
Shared fixtures for SEPA screener tests.

Provides:
- Known watchlist stocks (5 India, 5 US) for eval fixtures
- Mock OHLCV DataFrames matching canonical schema
- Mock fundamentals data
- Mock providers for aggregator testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── Watchlist fixtures (10 known stocks for eval) ──────────────────────

WATCHLIST_INDIA = [
    {"symbol": "SAILIFE", "name": "Sai Life Sciences", "exchange": "NSE", "yfinance_symbol": "SAILIFE.NS"},
    {"symbol": "LUPIN", "name": "Lupin", "exchange": "NSE", "yfinance_symbol": "LUPIN.NS"},
    {"symbol": "ANANDRATHI", "name": "Anand Rathi Wealth", "exchange": "NSE", "yfinance_symbol": "ANANDRATHI.NS"},
    {"symbol": "DATAPATTNS", "name": "Data Patterns", "exchange": "NSE", "yfinance_symbol": "DATAPATTNS.NS"},
    {"symbol": "TDPOWERSYS", "name": "TD Power Systems", "exchange": "NSE", "yfinance_symbol": "TDPOWERSYS.NS"},
]

WATCHLIST_US = [
    {"symbol": "TSM", "name": "Taiwan Semiconductor", "exchange": "NYSE", "yfinance_symbol": "TSM"},
    {"symbol": "NVT", "name": "nVent Electric", "exchange": "NYSE", "yfinance_symbol": "NVT"},
    {"symbol": "ADI", "name": "Analog Devices", "exchange": "NASDAQ", "yfinance_symbol": "ADI"},
    {"symbol": "MU", "name": "Micron Technology", "exchange": "NASDAQ", "yfinance_symbol": "MU"},
    {"symbol": "TMDX", "name": "TransMedics Group", "exchange": "NASDAQ", "yfinance_symbol": "TMDX"},
]


@pytest.fixture
def watchlist_india():
    return WATCHLIST_INDIA.copy()


@pytest.fixture
def watchlist_us():
    return WATCHLIST_US.copy()


@pytest.fixture
def watchlist_all():
    return WATCHLIST_INDIA + WATCHLIST_US


# ── Canonical DataFrame fixtures ───────────────────────────────────────

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]


def make_ohlcv(
    days: int = 252,
    start_price: float = 100.0,
    trend: str = "up",
    symbol: str = "TEST",
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame in canonical schema.

    Args:
        days: Number of trading days
        start_price: Starting close price
        trend: "up" (Stage 2 uptrend), "down" (Stage 4), or "flat"
        symbol: For labeling only (not in DataFrame)

    Returns:
        DataFrame with canonical columns, DatetimeIndex (UTC), float64
    """
    dates = pd.date_range(
        end=datetime.utcnow().date(),
        periods=days,
        freq="B",  # business days
        tz="UTC",
    )

    if trend == "up":
        # Steady uptrend — should pass Trend Template
        daily_return = 1.001  # ~0.1% per day ≈ +28% per year
        noise_scale = 0.01
    elif trend == "down":
        daily_return = 0.999
        noise_scale = 0.015
    else:  # flat
        daily_return = 1.0
        noise_scale = 0.008

    np.random.seed(hash(symbol) % 2**31)
    noise = np.random.normal(0, noise_scale, days)
    closes = start_price * np.cumprod(np.full(days, daily_return) + noise)
    closes = np.maximum(closes, 0.01)  # no negative prices

    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, days)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, days)))
    opens = lows + (highs - lows) * np.random.uniform(0.3, 0.7, days)
    volume = np.random.randint(50_000, 500_000, days).astype(float)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "adj_close": closes,  # simplified: adj_close = close
        },
        index=dates,
    )
    df.index.name = "date"
    return df.astype("float64")


@pytest.fixture
def canonical_ohlcv_uptrend():
    """252-day uptrend OHLCV in canonical schema. Should pass TT."""
    return make_ohlcv(days=252, start_price=50.0, trend="up", symbol="UPTREND")


@pytest.fixture
def canonical_ohlcv_downtrend():
    """252-day downtrend OHLCV in canonical schema. Should fail TT."""
    return make_ohlcv(days=252, start_price=200.0, trend="down", symbol="DOWNTREND")


@pytest.fixture
def canonical_ohlcv_short():
    """Only 100 days — not enough for 200 SMA. Should be flagged as insufficient."""
    return make_ohlcv(days=100, start_price=100.0, trend="up", symbol="SHORT")


# ── Mock fundamentals ──────────────────────────────────────────────────

@pytest.fixture
def fundamentals_strong():
    """Strong SEPA fundamentals — should PASS."""
    return {
        "market_cap": 5_000_000_000,
        "pe_ratio": 35.0,
        "eps_ttm": 4.50,
        "roe": 22.5,
        "debt_equity": 15.0,
        "profit_margin": 18.5,
        "revenue_growth": 28.0,
        "earnings_growth": 35.0,
        "eps_growth_yoy_calc": 42.0,
        "institutional_pct": 25.0,
        "sector": "Technology",
        "industry": "Semiconductors",
        "quarterly_eps": [
            {"quarter": "2026-01-01", "eps": 1.25},
            {"quarter": "2025-10-01", "eps": 1.10},
            {"quarter": "2025-07-01", "eps": 0.95},
            {"quarter": "2025-04-01", "eps": 0.85},
            {"quarter": "2025-01-01", "eps": 0.80},
            {"quarter": "2024-10-01", "eps": 0.70},
        ],
        "eps_yoy_changes": [
            {"quarter": "2026-01-01", "current": 1.25, "year_ago": 0.80, "yoy_pct": 56.3},
            {"quarter": "2025-10-01", "current": 1.10, "year_ago": 0.70, "yoy_pct": 57.1},
        ],
        "eps_accelerating": True,
        "eps_decel_count": 0,
        "latest_eps_negative": False,
        "min_growth_pass": True,
        "eps_verdict": "PASS (accelerating)",
    }


@pytest.fixture
def fundamentals_weak():
    """Weak fundamentals — should FAIL (negative EPS, decelerating)."""
    return {
        "market_cap": 200_000_000,
        "pe_ratio": None,
        "eps_ttm": -0.50,
        "roe": -5.0,
        "debt_equity": 85.0,
        "profit_margin": -3.0,
        "revenue_growth": 5.0,
        "earnings_growth": -20.0,
        "eps_growth_yoy_calc": -25.0,
        "institutional_pct": 8.0,
        "sector": "Consumer Cyclical",
        "industry": "Apparel",
        "quarterly_eps": [
            {"quarter": "2026-01-01", "eps": -0.15},
            {"quarter": "2025-10-01", "eps": -0.10},
            {"quarter": "2025-07-01", "eps": 0.05},
            {"quarter": "2025-04-01", "eps": 0.10},
            {"quarter": "2025-01-01", "eps": 0.20},
            {"quarter": "2024-10-01", "eps": 0.25},
        ],
        "eps_yoy_changes": [],
        "eps_accelerating": False,
        "eps_decel_count": 3,
        "latest_eps_negative": True,
        "min_growth_pass": False,
        "eps_verdict": "REJECT (negative EPS)",
    }
