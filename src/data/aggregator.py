"""
DataAggregator — Multi-source data fetching with fallback chain.

Orchestrates multiple DataProviders:
    1. Try primary provider
    2. Detect bad data (NaN, stale, missing)
    3. Fallback to secondary provider
    4. Log everything (audit trail)
    5. Never silently drop a stock

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │  Request   │ ──▶ │   Cache    │ ──▶ │  Provider  │
    │ (symbol,   │     │  (Parquet) │     │   Chain    │
    │  type)     │     │  hit/miss  │     │  1→2→...→N │
    └────────────┘     └────────────┘     └────────────┘
                              │                   │
                              ▼                   ▼
                       ┌────────────┐     ┌────────────┐
                       │ Return     │     │  Validate  │
                       │ cached     │     │  + cache   │
                       │ data       │     │  result    │
                       └────────────┘     └────────────┘
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from .provider import DataProvider, DataResult, validate_ohlcv

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Multi-source data aggregator with fallback chain.

    Usage:
        agg = DataAggregator()
        agg.register_provider(YFinanceProvider(), priority=1)
        agg.register_provider(OpenBBProvider(), priority=2)

        result = agg.get_ohlcv("RELIANCE.NS", period="1y")
        # Tries provider 1, falls back to 2 if bad data
    """

    def __init__(self, cache=None):
        """
        Args:
            cache: Optional CacheLayer instance. If None, no caching.
        """
        self._providers: List[DataProvider] = []
        self._cache = cache
        self._fetch_log: List[Dict] = []  # audit trail

    def register_provider(self, provider: DataProvider, priority: int = 100):
        """
        Register a data provider.
        Lower priority number = tried first.
        """
        self._providers.append((priority, provider))
        self._providers.sort(key=lambda x: x[0])
        logger.info(f"Registered provider '{provider.name}' with priority {priority}")

    @property
    def providers(self) -> List[DataProvider]:
        """Registered providers in priority order."""
        return [p for _, p in self._providers]

    # ── OHLCV ──────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, period: str = "1y") -> DataResult:
        """
        Fetch OHLCV with fallback chain.
        Tries each provider in priority order until valid data is found.
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get_ohlcv(symbol)
            if cached is not None:
                self._log_fetch(symbol, "ohlcv", "cache", "hit", [])
                return DataResult(
                    symbol=symbol,
                    data=cached,
                    provider="cache",
                    is_complete=len(validate_ohlcv(cached, symbol)) == 0,
                    issues=[],
                )

        # Try providers in order
        all_issues = []
        for _, provider in self._providers:
            result = provider.get_ohlcv(symbol, period=period)
            self._log_fetch(symbol, "ohlcv", provider.name,
                            "success" if result.is_valid else "failed",
                            result.issues)

            if result.is_valid and result.data is not None:
                # Cache the good result
                if self._cache:
                    self._cache.put_ohlcv(symbol, result.data, provider.name)
                return result

            all_issues.extend(result.issues)

        # All providers failed
        return DataResult(
            symbol=symbol,
            data=None,
            provider="none",
            is_complete=False,
            issues=all_issues or [f"{symbol}: no provider returned valid OHLCV data"],
        )

    def get_batch_ohlcv(self, symbols: List[str], period: str = "1y") -> Dict[str, DataResult]:
        """
        Batch-fetch OHLCV for multiple symbols.

        Strategy:
            1. Check cache for each symbol
            2. Batch-fetch uncached symbols from primary provider
            3. Retry missing from secondary providers individually
            4. Log everything
        """
        results: Dict[str, DataResult] = {}
        uncached_symbols = []

        # Step 1: Check cache
        if self._cache:
            for symbol in symbols:
                cached = self._cache.get_ohlcv(symbol)
                if cached is not None:
                    results[symbol] = DataResult(
                        symbol=symbol,
                        data=cached,
                        provider="cache",
                        is_complete=len(validate_ohlcv(cached, symbol)) == 0,
                        issues=[],
                    )
                else:
                    uncached_symbols.append(symbol)
        else:
            uncached_symbols = list(symbols)

        if not uncached_symbols:
            return results

        # Step 2: Batch-fetch from primary provider
        for _, provider in self._providers:
            if not uncached_symbols:
                break

            batch_results = provider.get_batch_ohlcv(uncached_symbols, period=period)

            still_missing = []
            for symbol in uncached_symbols:
                if symbol in batch_results and batch_results[symbol].is_valid:
                    result = batch_results[symbol]
                    results[symbol] = result
                    if self._cache and result.data is not None:
                        self._cache.put_ohlcv(symbol, result.data, provider.name)
                    self._log_fetch(symbol, "ohlcv", provider.name, "success", [])
                else:
                    issues = batch_results[symbol].issues if symbol in batch_results else [f"{symbol}: not returned in batch"]
                    self._log_fetch(symbol, "ohlcv", provider.name, "failed", issues)
                    still_missing.append(symbol)

            uncached_symbols = still_missing

        # Step 3: Any still missing get a DataResult with issues
        for symbol in uncached_symbols:
            results[symbol] = DataResult(
                symbol=symbol,
                data=None,
                provider="none",
                is_complete=False,
                issues=[f"{symbol}: no provider returned valid OHLCV data"],
            )

        logger.info(
            f"Batch OHLCV: {len([r for r in results.values() if r.is_valid])}/{len(symbols)} valid, "
            f"{len(uncached_symbols)} missing"
        )
        return results

    # ── Fundamentals ───────────────────────────────────────────────────

    def get_fundamentals(self, symbol: str) -> DataResult:
        """Fetch fundamentals with fallback chain."""
        # Check cache first
        if self._cache:
            cached = self._cache.get_fundamentals(symbol)
            if cached is not None:
                self._log_fetch(symbol, "fundamentals", "cache", "hit", [])
                return DataResult(
                    symbol=symbol,
                    data=cached,
                    provider="cache",
                    is_complete=True,
                    issues=[],
                )

        all_issues = []
        for _, provider in self._providers:
            result = provider.get_fundamentals(symbol)
            self._log_fetch(symbol, "fundamentals", provider.name,
                            "success" if result.is_valid else "failed",
                            result.issues)

            if result.is_valid and result.data is not None:
                if self._cache:
                    self._cache.put_fundamentals(symbol, result.data, provider.name)
                return result

            all_issues.extend(result.issues)

        return DataResult(
            symbol=symbol,
            data=None,
            provider="none",
            is_complete=False,
            issues=all_issues or [f"{symbol}: no provider returned valid fundamentals"],
        )

    # ── Stock List ─────────────────────────────────────────────────────

    def get_stock_list(self, market: str) -> DataResult:
        """Fetch stock list with fallback chain."""
        all_issues = []
        for _, provider in self._providers:
            result = provider.get_stock_list(market)
            self._log_fetch(f"universe:{market}", "stock_list", provider.name,
                            "success" if result.is_valid else "failed",
                            result.issues)

            if result.is_valid and result.data:
                return result

            all_issues.extend(result.issues)

        return DataResult(
            symbol=f"universe:{market}",
            data=[],
            provider="none",
            is_complete=False,
            issues=all_issues or [f"No provider returned stock list for {market}"],
        )

    # ── Logging / Audit Trail ──────────────────────────────────────────

    def _log_fetch(self, symbol: str, data_type: str, provider: str,
                   status: str, issues: List[str]):
        """Log every data fetch attempt for audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "data_type": data_type,
            "provider": provider,
            "status": status,
            "issues": issues,
        }
        self._fetch_log.append(entry)
        if status == "failed" and issues:
            logger.debug(f"Fetch {data_type} {symbol} via {provider}: {status} — {issues}")

    def get_fetch_log(self) -> List[Dict]:
        """Return the full audit trail of fetch attempts."""
        return self._fetch_log.copy()

    def get_fetch_summary(self) -> Dict:
        """Summarize fetch results for reporting."""
        total = len(self._fetch_log)
        successes = len([e for e in self._fetch_log if e["status"] == "success"])
        failures = len([e for e in self._fetch_log if e["status"] == "failed"])
        cache_hits = len([e for e in self._fetch_log if e["status"] == "hit"])

        # Collect all symbols that had NO successful fetch
        failed_symbols = set()
        succeeded_symbols = set()
        for entry in self._fetch_log:
            if entry["status"] == "success" or entry["status"] == "hit":
                succeeded_symbols.add(entry["symbol"])
            elif entry["status"] == "failed":
                failed_symbols.add(entry["symbol"])

        truly_failed = failed_symbols - succeeded_symbols

        return {
            "total_fetches": total,
            "successes": successes,
            "failures": failures,
            "cache_hits": cache_hits,
            "symbols_with_no_data": sorted(truly_failed),
            "no_data_count": len(truly_failed),
        }

    def clear_log(self):
        """Clear the fetch log (call between scan runs)."""
        self._fetch_log.clear()
