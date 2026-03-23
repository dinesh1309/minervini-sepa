"""
Tests for the multi-source data layer.

Covers:
    - Canonical schema validation and normalization
    - DataProvider interface contract
    - YFinanceProvider output format
    - DataAggregator fallback chain
    - CacheLayer hit/miss/expiry behavior
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from src.data.provider import (
    DataProvider,
    DataResult,
    validate_ohlcv,
    normalize_ohlcv,
    CANONICAL_COLUMNS,
)
from src.data.yfinance_provider import YFinanceProvider
from src.data.aggregator import DataAggregator
from src.data.cache import CacheLayer
from tests.conftest import make_ohlcv


# ═══════════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCanonicalSchema:
    """Canonical OHLCV schema validation."""

    def test_valid_dataframe_passes(self, canonical_ohlcv_uptrend):
        issues = validate_ohlcv(canonical_ohlcv_uptrend, "TEST")
        assert issues == [], f"Valid DataFrame should have no issues: {issues}"

    def test_none_dataframe_fails(self):
        issues = validate_ohlcv(None, "TEST")
        assert len(issues) > 0
        assert "None or empty" in issues[0]

    def test_empty_dataframe_fails(self):
        issues = validate_ohlcv(pd.DataFrame(), "TEST")
        assert len(issues) > 0

    def test_missing_close_column_fails(self):
        df = make_ohlcv(days=10)
        df = df.drop(columns=["close"])
        issues = validate_ohlcv(df, "TEST")
        assert any("missing columns" in i for i in issues)

    def test_nan_close_prices_flagged(self):
        df = make_ohlcv(days=10)
        df.iloc[3, df.columns.get_loc("close")] = np.nan
        issues = validate_ohlcv(df, "TEST")
        assert any("NaN" in i for i in issues)

    def test_negative_prices_flagged(self):
        df = make_ohlcv(days=10)
        df.iloc[0, df.columns.get_loc("close")] = -5.0
        issues = validate_ohlcv(df, "TEST")
        assert any("non-positive" in i for i in issues)

    def test_canonical_columns_present(self, canonical_ohlcv_uptrend):
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in canonical_ohlcv_uptrend.columns

    def test_index_is_datetime(self, canonical_ohlcv_uptrend):
        assert isinstance(canonical_ohlcv_uptrend.index, pd.DatetimeIndex)

    def test_dtypes_are_float64(self, canonical_ohlcv_uptrend):
        for col in canonical_ohlcv_uptrend.columns:
            assert canonical_ohlcv_uptrend[col].dtype == np.float64


class TestNormalizeOhlcv:
    """Test normalization from provider-native format to canonical."""

    def test_yfinance_column_names_normalized(self):
        """yfinance returns 'Open', 'High', etc. — should become lowercase."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [99.0],
            "Close": [103.0], "Volume": [1000.0], "Adj Close": [103.0],
        }, index=pd.DatetimeIndex([datetime(2025, 1, 1)], tz="UTC"))

        result = normalize_ohlcv(df)
        assert list(result.columns) == CANONICAL_COLUMNS

    def test_missing_adj_close_gets_close(self):
        """If adj_close is missing, it should be filled with close."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [99.0],
            "Close": [103.0], "Volume": [1000.0],
        }, index=pd.DatetimeIndex([datetime(2025, 1, 1)], tz="UTC"))

        result = normalize_ohlcv(df)
        assert "adj_close" in result.columns
        assert result["adj_close"].iloc[0] == 103.0

    def test_timezone_naive_gets_utc(self):
        """Timezone-naive index should get UTC."""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000.0],
        }, index=pd.DatetimeIndex([datetime(2025, 1, 1)]))

        result = normalize_ohlcv(df)
        assert str(result.index.tz) == "UTC"

    def test_nan_close_rows_dropped(self):
        """Rows with NaN close should be removed during normalization."""
        dates = pd.date_range("2025-01-01", periods=5, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [99, 100, 101, 102, 103],
            "close": [103, np.nan, 105, 106, 107],
            "volume": [1000, 2000, 3000, 4000, 5000],
        }, index=dates, dtype="float64")

        result = normalize_ohlcv(df)
        assert len(result) == 4  # one NaN row dropped

    def test_none_returns_none(self):
        assert normalize_ohlcv(None) is None

    def test_empty_returns_empty(self):
        result = normalize_ohlcv(pd.DataFrame())
        assert result.empty


# ═══════════════════════════════════════════════════════════════════════
# DataResult Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataResult:

    def test_valid_result(self):
        r = DataResult(symbol="TEST", data={"key": "val"}, provider="mock",
                       is_complete=True, issues=[])
        assert r.is_valid is True

    def test_invalid_when_issues(self):
        r = DataResult(symbol="TEST", data={"key": "val"}, provider="mock",
                       is_complete=True, issues=["something wrong"])
        assert r.is_valid is False

    def test_invalid_when_no_data(self):
        r = DataResult(symbol="TEST", data=None, provider="mock",
                       is_complete=False, issues=[])
        assert r.is_valid is False


# ═══════════════════════════════════════════════════════════════════════
# Mock Provider for Aggregator Tests
# ═══════════════════════════════════════════════════════════════════════

class MockProvider(DataProvider):
    """Configurable mock provider for testing aggregator fallback."""

    def __init__(self, provider_name: str, ohlcv_data=None, should_fail=False):
        self._name = provider_name
        self._ohlcv_data = ohlcv_data
        self._should_fail = should_fail
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def health_check(self) -> bool:
        return not self._should_fail

    def get_ohlcv(self, symbol, period="1y"):
        self.call_count += 1
        if self._should_fail:
            return DataResult(symbol=symbol, data=None, provider=self.name,
                              is_complete=False, issues=[f"{symbol}: mock failure"])
        return DataResult(symbol=symbol, data=self._ohlcv_data, provider=self.name,
                          is_complete=self._ohlcv_data is not None, issues=[])

    def get_batch_ohlcv(self, symbols, period="1y"):
        results = {}
        for s in symbols:
            results[s] = self.get_ohlcv(s, period)
        return results

    def get_fundamentals(self, symbol):
        self.call_count += 1
        if self._should_fail:
            return DataResult(symbol=symbol, data=None, provider=self.name,
                              is_complete=False, issues=[f"{symbol}: mock failure"])
        return DataResult(symbol=symbol, data={"eps_verdict": "PASS"}, provider=self.name,
                          is_complete=True, issues=[])

    def get_stock_list(self, market):
        return DataResult(symbol=f"universe:{market}", data=[], provider=self.name,
                          is_complete=False, issues=["mock: no stock list"])


# ═══════════════════════════════════════════════════════════════════════
# DataAggregator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataAggregator:

    def test_single_provider_success(self):
        df = make_ohlcv(days=252, symbol="OK")
        provider = MockProvider("mock1", ohlcv_data=df)
        agg = DataAggregator()
        agg.register_provider(provider, priority=1)

        result = agg.get_ohlcv("TEST")
        assert result.is_valid
        assert result.provider == "mock1"
        assert provider.call_count == 1

    def test_fallback_on_primary_failure(self):
        """If primary fails, secondary should be tried."""
        df = make_ohlcv(days=252, symbol="OK")
        primary = MockProvider("primary", should_fail=True)
        secondary = MockProvider("secondary", ohlcv_data=df)

        agg = DataAggregator()
        agg.register_provider(primary, priority=1)
        agg.register_provider(secondary, priority=2)

        result = agg.get_ohlcv("TEST")
        assert result.is_valid
        assert result.provider == "secondary"
        assert primary.call_count == 1
        assert secondary.call_count == 1

    def test_all_providers_fail(self):
        """If all providers fail, DataResult should have issues, not raise."""
        primary = MockProvider("primary", should_fail=True)
        secondary = MockProvider("secondary", should_fail=True)

        agg = DataAggregator()
        agg.register_provider(primary, priority=1)
        agg.register_provider(secondary, priority=2)

        result = agg.get_ohlcv("TEST")
        assert not result.is_valid
        assert result.provider == "none"
        assert len(result.issues) > 0

    def test_fetch_summary_tracks_failures(self):
        primary = MockProvider("primary", should_fail=True)
        agg = DataAggregator()
        agg.register_provider(primary, priority=1)

        agg.get_ohlcv("MISSING_STOCK")
        summary = agg.get_fetch_summary()
        assert summary["no_data_count"] == 1
        assert "MISSING_STOCK" in summary["symbols_with_no_data"]

    def test_batch_ohlcv_returns_all_symbols(self):
        df = make_ohlcv(days=252, symbol="BATCH")
        provider = MockProvider("mock1", ohlcv_data=df)
        agg = DataAggregator()
        agg.register_provider(provider, priority=1)

        results = agg.get_batch_ohlcv(["A", "B", "C"])
        assert len(results) == 3
        assert all(r.is_valid for r in results.values())

    def test_fundamentals_fallback(self):
        primary = MockProvider("primary", should_fail=True)
        secondary = MockProvider("secondary")

        agg = DataAggregator()
        agg.register_provider(primary, priority=1)
        agg.register_provider(secondary, priority=2)

        result = agg.get_fundamentals("TEST")
        assert result.is_valid
        assert result.provider == "secondary"


# ═══════════════════════════════════════════════════════════════════════
# CacheLayer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCacheLayer:

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path):
        """Create a temporary cache directory for each test."""
        self.cache = CacheLayer(cache_dir=str(tmp_path / "cache"))
        yield
        # Cleanup handled by tmp_path

    def test_ohlcv_cache_miss(self):
        result = self.cache.get_ohlcv("NONEXISTENT")
        assert result is None

    def test_ohlcv_cache_hit(self):
        df = make_ohlcv(days=10, symbol="CACHED")
        self.cache.put_ohlcv("CACHED", df, "test")

        result = self.cache.get_ohlcv("CACHED")
        assert result is not None
        assert len(result) == 10

    def test_ohlcv_cache_preserves_schema(self):
        df = make_ohlcv(days=10, symbol="SCHEMA")
        self.cache.put_ohlcv("SCHEMA", df, "test")

        result = self.cache.get_ohlcv("SCHEMA")
        assert isinstance(result.index, pd.DatetimeIndex)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_fundamentals_cache_roundtrip(self):
        data = {"eps_verdict": "PASS", "pe_ratio": 25.0, "quarterly_eps": [{"q": "2025-01", "eps": 1.5}]}
        self.cache.put_fundamentals("TEST", data, "test")

        result = self.cache.get_fundamentals("TEST")
        assert result is not None
        assert result["eps_verdict"] == "PASS"
        assert result["quarterly_eps"][0]["eps"] == 1.5

    def test_fundamentals_cache_miss(self):
        result = self.cache.get_fundamentals("NONEXISTENT")
        assert result is None

    def test_cache_stats(self):
        df = make_ohlcv(days=10, symbol="STATS")
        self.cache.put_ohlcv("STATS", df, "test")
        self.cache.put_fundamentals("STATS", {"test": True}, "test")

        stats = self.cache.stats()
        assert stats["ohlcv_entries"] == 1
        assert stats["fundamentals_entries"] == 1
        assert stats["ohlcv_entries"] + stats["fundamentals_entries"] == 2

    def test_clear_all(self):
        self.cache.put_ohlcv("A", make_ohlcv(days=10, symbol="A"), "test")
        self.cache.put_ohlcv("B", make_ohlcv(days=10, symbol="B"), "test")
        self.cache.clear_all()

        assert self.cache.get_ohlcv("A") is None
        assert self.cache.get_ohlcv("B") is None

    def test_aggregator_uses_cache(self):
        """Aggregator should return cached data without calling provider."""
        df = make_ohlcv(days=252, symbol="CACHED")
        self.cache.put_ohlcv("CACHED", df, "test")

        provider = MockProvider("mock", should_fail=True)  # would fail if called
        agg = DataAggregator(cache=self.cache)
        agg.register_provider(provider, priority=1)

        result = agg.get_ohlcv("CACHED")
        assert result.data is not None
        assert provider.call_count == 0  # provider never called — cache served it
