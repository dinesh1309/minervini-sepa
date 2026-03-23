"""
DataProvider — Abstract base class for all data sources.

Every provider normalizes its output to the canonical schema defined here.
This ensures the rest of the pipeline (TT, fundamentals, VCP) never cares
which source the data came from.

Canonical OHLCV Schema:
    Columns: ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    Index:   DatetimeIndex, UTC timezone, name='date'
    Dtypes:  All float64
    Sorted:  Ascending by date (oldest first)

    ┌──────────┬────────┬────────┬────────┬────────┬──────────┬───────────┐
    │ date     │ open   │ high   │ low    │ close  │ volume   │ adj_close │
    │ (index)  │ f64    │ f64    │ f64    │ f64    │ f64      │ f64       │
    ├──────────┼────────┼────────┼────────┼────────┼──────────┼───────────┤
    │ 2025-03  │ 150.2  │ 152.8  │ 149.1  │ 151.5  │ 1200000  │ 151.5     │
    │ ...      │ ...    │ ...    │ ...    │ ...    │ ...      │ ...       │
    └──────────┴────────┴────────┴────────┴────────┴──────────┴───────────┘

Canonical Fundamentals Schema:
    Dict with standardized keys — see FundamentalsData below.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Canonical column names ─────────────────────────────────────────────

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]  # adj_close can be missing


# ── Validation helpers ─────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> List[str]:
    """
    Validate a DataFrame against the canonical OHLCV schema.
    Returns list of issues found (empty = valid).
    """
    issues = []

    if df is None or df.empty:
        issues.append(f"{symbol}: DataFrame is None or empty")
        return issues

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"{symbol}: missing columns {missing}")

    # Check index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append(f"{symbol}: index is {type(df.index).__name__}, expected DatetimeIndex")

    # Check for NaN in close prices
    if "close" in df.columns:
        nan_count = df["close"].isna().sum()
        if nan_count > 0:
            issues.append(f"{symbol}: {nan_count} NaN values in close prices")

    # Check for non-positive close prices
    if "close" in df.columns:
        bad_prices = (df["close"] <= 0).sum()
        if bad_prices > 0:
            issues.append(f"{symbol}: {bad_prices} non-positive close prices")

    # Check data is sorted ascending
    if len(df) > 1 and df.index[0] > df.index[-1]:
        issues.append(f"{symbol}: data not sorted ascending by date")

    return issues


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame to canonical schema.
    Handles common column name variations from different providers.
    """
    if df is None or df.empty:
        return df

    # Column name mapping (provider variants → canonical)
    column_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "adj_close": "adj_close",
        "Adjusted_close": "adj_close",
    }

    df = df.rename(columns=column_map)

    # Ensure all canonical columns exist
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Keep only canonical columns (drop extras)
    keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df = df[keep]

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    df.index.name = "date"

    # Sort ascending
    df = df.sort_index()

    # Remove rows with NaN close
    df = df.dropna(subset=["close"])

    # Remove non-positive prices
    df = df[df["close"] > 0]

    # Cast to float64
    for col in df.columns:
        df[col] = df[col].astype("float64")

    return df


# ── Data quality result ────────────────────────────────────────────────

@dataclass
class DataResult:
    """
    Result from a data fetch attempt.

    Wraps the data with metadata about which provider returned it,
    whether it's complete, and any issues found.
    """
    symbol: str
    data: Any  # DataFrame for OHLCV, dict for fundamentals
    provider: str  # which provider returned this
    is_complete: bool  # True if no NaN gaps, sufficient history, etc.
    issues: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Data exists and has no critical issues."""
        return self.data is not None and len(self.issues) == 0


# ── DataProvider ABC ───────────────────────────────────────────────────

class DataProvider(ABC):
    """
    Abstract base class for all data sources.

    Every provider must:
    1. Implement all abstract methods
    2. Return OHLCV data in canonical schema (use normalize_ohlcv)
    3. Return fundamentals in standardized dict format
    4. Handle its own rate limiting and error recovery
    5. Never raise exceptions to the caller — return DataResult with issues

    Provider lifecycle:
        ┌──────────┐     ┌──────────────┐     ┌──────────────┐
        │ init()   │ ──▶ │ health_check │ ──▶ │ get_ohlcv()  │
        │ (config) │     │ (am I alive?)│     │ get_batch()  │
        └──────────┘     └──────────────┘     │ get_fundies()│
                                               └──────────────┘
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name (e.g., 'yfinance', 'openbb')."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if this provider is responsive.
        Returns True if the provider can serve requests right now.
        Should be fast (<5 seconds).
        """
        ...

    @abstractmethod
    def get_ohlcv(self, symbol: str, period: str = "1y") -> DataResult:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock symbol (provider-native format, e.g., 'RELIANCE.NS')
            period: Lookback period ('1y', '2y', '6mo', etc.)

        Returns:
            DataResult with canonical DataFrame or issues
        """
        ...

    @abstractmethod
    def get_batch_ohlcv(self, symbols: List[str], period: str = "1y") -> Dict[str, DataResult]:
        """
        Fetch OHLCV data for multiple symbols in one call.

        Args:
            symbols: List of stock symbols
            period: Lookback period

        Returns:
            Dict mapping symbol → DataResult
        """
        ...

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> DataResult:
        """
        Fetch fundamental data for a single symbol.

        Returns DataResult with dict containing standardized keys:
            market_cap, pe_ratio, eps_ttm, roe, debt_equity,
            profit_margin, revenue_growth, earnings_growth,
            institutional_pct, sector, industry,
            quarterly_eps (list of dicts), eps_yoy_changes (list of dicts),
            eps_accelerating (bool), eps_decel_count (int),
            latest_eps_negative (bool), min_growth_pass (bool),
            eps_verdict (str)
        """
        ...

    @abstractmethod
    def get_stock_list(self, market: str) -> DataResult:
        """
        Fetch list of tradeable stocks for a market.

        Args:
            market: 'india' or 'us'

        Returns:
            DataResult with list of dicts:
                [{"symbol": "RELIANCE", "name": "Reliance Industries",
                  "exchange": "NSE", "yfinance_symbol": "RELIANCE.NS"}, ...]
        """
        ...

    def format_symbol(self, symbol: str, market: str) -> str:
        """
        Convert a generic symbol to this provider's native format.
        Override in subclasses if needed (e.g., adding .NS suffix for yfinance).
        """
        return symbol
