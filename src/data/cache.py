"""
CacheLayer — Parquet-based disk cache for market data.

Sits between the DataAggregator and providers. On cache hit,
returns stored data without calling any provider. On miss,
the aggregator fetches from providers and stores the result.

Cache structure:
    data/cache/
    ├── ohlcv/
    │   ├── RELIANCE.NS_2026-03-24.parquet
    │   ├── TSM_2026-03-24.parquet
    │   └── ...
    └── fundamentals/
        ├── RELIANCE.NS_2026-03-24.json
        ├── TSM_2026-03-24.json
        └── ...

Expiry:
    - OHLCV: 1 day (refetch after market close)
    - Fundamentals: 7 days (quarterly data doesn't change daily)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class CacheLayer:
    """
    Parquet-based disk cache for OHLCV and fundamentals data.

    Usage:
        cache = CacheLayer(cache_dir="data/cache")
        df = cache.get_ohlcv("RELIANCE.NS")  # None if miss
        cache.put_ohlcv("RELIANCE.NS", df, provider="yfinance")
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        ohlcv_expiry_days: int = 1,
        fundamentals_expiry_days: int = 7,
    ):
        self.cache_dir = Path(cache_dir)
        self.ohlcv_dir = self.cache_dir / "ohlcv"
        self.fundamentals_dir = self.cache_dir / "fundamentals"
        self.ohlcv_expiry = timedelta(days=ohlcv_expiry_days)
        self.fundamentals_expiry = timedelta(days=fundamentals_expiry_days)

        # Create dirs
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)

    # ── OHLCV Cache ────────────────────────────────────────────────────

    def _ohlcv_path(self, symbol: str) -> Path:
        """Path for a symbol's cached OHLCV data."""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return self.ohlcv_dir / f"{safe_symbol}.parquet"

    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data if it exists and is fresh.
        Returns None on cache miss or expired data.
        """
        path = self._ohlcv_path(symbol)

        if not path.exists():
            return None

        # Check expiry
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > self.ohlcv_expiry:
            logger.debug(f"Cache expired for {symbol} OHLCV (age: {datetime.now() - mtime})")
            return None

        try:
            df = pd.read_parquet(path)
            # Restore DatetimeIndex
            if "date" in df.columns:
                df = df.set_index("date")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            logger.debug(f"Cache hit for {symbol} OHLCV ({len(df)} rows)")
            return df

        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")
            return None

    def put_ohlcv(self, symbol: str, df: pd.DataFrame, provider: str = ""):
        """Store OHLCV data in cache."""
        if df is None or df.empty:
            return

        path = self._ohlcv_path(symbol)

        try:
            # Reset index for Parquet storage (DatetimeIndex as column)
            store_df = df.copy()
            if isinstance(store_df.index, pd.DatetimeIndex):
                store_df = store_df.reset_index()

            store_df.to_parquet(path, engine="pyarrow", index=False)
            logger.debug(f"Cached {symbol} OHLCV ({len(df)} rows) from {provider}")

        except Exception as e:
            logger.warning(f"Cache write error for {symbol}: {e}")

    # ── Fundamentals Cache ─────────────────────────────────────────────

    def _fundamentals_path(self, symbol: str) -> Path:
        """Path for a symbol's cached fundamentals data."""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return self.fundamentals_dir / f"{safe_symbol}.json"

    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get cached fundamentals if fresh."""
        path = self._fundamentals_path(symbol)

        if not path.exists():
            return None

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > self.fundamentals_expiry:
            logger.debug(f"Cache expired for {symbol} fundamentals")
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.debug(f"Cache hit for {symbol} fundamentals")
            return data

        except Exception as e:
            logger.warning(f"Cache read error for {symbol} fundamentals: {e}")
            return None

    def put_fundamentals(self, symbol: str, data: Dict, provider: str = ""):
        """Store fundamentals in cache."""
        if data is None:
            return

        path = self._fundamentals_path(symbol)

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Cached {symbol} fundamentals from {provider}")

        except Exception as e:
            logger.warning(f"Cache write error for {symbol} fundamentals: {e}")

    # ── Cache Management ───────────────────────────────────────────────

    def clear_all(self):
        """Clear entire cache."""
        import shutil
        for d in [self.ohlcv_dir, self.fundamentals_dir]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")

    def clear_expired(self):
        """Remove only expired cache entries."""
        removed = 0
        now = datetime.now()

        for path in self.ohlcv_dir.glob("*.parquet"):
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if now - mtime > self.ohlcv_expiry:
                path.unlink()
                removed += 1

        for path in self.fundamentals_dir.glob("*.json"):
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if now - mtime > self.fundamentals_expiry:
                path.unlink()
                removed += 1

        if removed:
            logger.info(f"Cleared {removed} expired cache entries")

    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        ohlcv_count = len(list(self.ohlcv_dir.glob("*.parquet")))
        fund_count = len(list(self.fundamentals_dir.glob("*.json")))

        ohlcv_size = sum(f.stat().st_size for f in self.ohlcv_dir.glob("*.parquet"))
        fund_size = sum(f.stat().st_size for f in self.fundamentals_dir.glob("*.json"))

        return {
            "ohlcv_entries": ohlcv_count,
            "fundamentals_entries": fund_count,
            "ohlcv_size_mb": round(ohlcv_size / (1024 * 1024), 2),
            "fundamentals_size_mb": round(fund_size / (1024 * 1024), 2),
            "total_size_mb": round((ohlcv_size + fund_size) / (1024 * 1024), 2),
        }
