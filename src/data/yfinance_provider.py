"""
YFinanceProvider — wraps yfinance as a DataProvider.

Reuses existing rate limiting and data validation from market_data.py.
Normalizes all output to canonical OHLCV schema.

Known limitations:
    - NaN gaps in Indian quarterly EPS data (~2 of 6 quarters for some stocks)
    - Rate limited (0.5s between requests)
    - Batch download can silently drop symbols (we detect + retry)
"""

import time
import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from .provider import DataProvider, DataResult, normalize_ohlcv, validate_ohlcv

logger = logging.getLogger(__name__)

# Rate limiting — shared across all YFinanceProvider instances
MIN_REQUEST_INTERVAL = 0.5
_last_request_time = 0


def _rate_limit():
    """Ensure minimum interval between yfinance API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


class YFinanceProvider(DataProvider):
    """
    Data provider using yfinance (Yahoo Finance).

    Handles:
        - India stocks: auto-adds .NS suffix for NSE
        - US stocks: uses symbol as-is
        - Batch downloads with threading
        - Rate limiting (0.5s between requests)
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def health_check(self) -> bool:
        """Quick check — fetch 1 day of SPY data."""
        try:
            _rate_limit()
            df = yf.Ticker("SPY").history(period="1d")
            return df is not None and not df.empty
        except Exception:
            return False

    def format_symbol(self, symbol: str, market: str) -> str:
        """Add .NS suffix for Indian stocks."""
        if market == "india":
            clean = symbol.replace(".NS", "").replace(".BO", "").strip()
            return f"{clean}.NS"
        return symbol

    def get_ohlcv(self, symbol: str, period: str = "1y") -> DataResult:
        """Fetch OHLCV for a single symbol, normalized to canonical schema."""
        _rate_limit()

        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=period)

            if raw is None or raw.empty:
                return DataResult(
                    symbol=symbol,
                    data=None,
                    provider=self.name,
                    is_complete=False,
                    issues=[f"{symbol}: no data returned from yfinance"],
                )

            # Normalize to canonical schema
            df = normalize_ohlcv(raw)

            # Validate
            issues = validate_ohlcv(df, symbol)

            # Check completeness — 200+ trading days for TT
            is_complete = len(df) >= 200 and len(issues) == 0

            if len(df) < 200:
                issues.append(f"{symbol}: only {len(df)} trading days (need 200+)")

            return DataResult(
                symbol=symbol,
                data=df,
                provider=self.name,
                is_complete=is_complete,
                issues=issues,
            )

        except Exception as e:
            return DataResult(
                symbol=symbol,
                data=None,
                provider=self.name,
                is_complete=False,
                issues=[f"{symbol}: yfinance error — {e}"],
            )

    def get_batch_ohlcv(self, symbols: List[str], period: str = "1y") -> Dict[str, DataResult]:
        """
        Batch-fetch OHLCV for multiple symbols.

        After the batch call, verifies all requested symbols were returned.
        Missing symbols are retried individually.
        """
        results: Dict[str, DataResult] = {}

        if not symbols:
            return results

        _rate_limit()

        try:
            raw = yf.download(
                symbols,
                period=period,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        df = raw
                    else:
                        df = raw[symbol].copy() if symbol in raw.columns.get_level_values(0) else pd.DataFrame()

                    if df is not None and not df.empty:
                        # Drop any multi-index level artifacts
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(0)

                        df = normalize_ohlcv(df)
                        issues = validate_ohlcv(df, symbol)
                        is_complete = len(df) >= 200 and len(issues) == 0

                        if len(df) < 200:
                            issues.append(f"{symbol}: only {len(df)} trading days (need 200+)")

                        results[symbol] = DataResult(
                            symbol=symbol,
                            data=df,
                            provider=self.name,
                            is_complete=is_complete,
                            issues=issues,
                        )
                except Exception as e:
                    logger.debug(f"Batch parse error for {symbol}: {e}")

        except Exception as e:
            logger.warning(f"Batch download failed: {e}. Falling back to individual fetches.")

        # Retry missing symbols individually
        missing = [s for s in symbols if s not in results]
        if missing:
            logger.info(f"Batch missed {len(missing)} symbols. Retrying individually...")
            for symbol in missing:
                results[symbol] = self.get_ohlcv(symbol, period=period)

        return results

    def get_fundamentals(self, symbol: str) -> DataResult:
        """
        Fetch SEPA-relevant fundamental data.

        Uses quarterly_income_stmt for EPS (Minervini-faithful),
        plus info dict for supplementary metrics.
        """
        _rate_limit()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # ── Quarterly EPS from income statement ──
            quarterly_eps = []
            try:
                qi = ticker.quarterly_income_stmt
                if qi is not None and not qi.empty:
                    for col_name in ["Diluted EPS", "Basic EPS"]:
                        if col_name in qi.index:
                            eps_row = qi.loc[col_name].dropna().head(6)
                            for date, val in eps_row.items():
                                quarterly_eps.append({
                                    "quarter": str(date)[:10],
                                    "eps": round(float(val), 2),
                                })
                            break
            except Exception:
                pass

            # ── EPS YoY changes (Minervini: primary metric) ──
            eps_yoy_changes = []
            if len(quarterly_eps) >= 5:
                for i in range(min(4, len(quarterly_eps) - 4)):
                    current_q = quarterly_eps[i]
                    yago_idx = i + 4
                    if yago_idx < len(quarterly_eps):
                        yago_eps = quarterly_eps[yago_idx]["eps"]
                        curr_eps = current_q["eps"]

                        if yago_eps and yago_eps <= 0 and curr_eps > 0:
                            eps_yoy_changes.append({
                                "quarter": current_q["quarter"],
                                "current": curr_eps,
                                "year_ago": yago_eps,
                                "yoy_pct": None,
                                "note": "EASY COMP (year-ago negative)",
                            })
                        elif yago_eps and 0 < yago_eps < 1.0:
                            pct = round(((curr_eps - yago_eps) / yago_eps) * 100, 1)
                            eps_yoy_changes.append({
                                "quarter": current_q["quarter"],
                                "current": curr_eps,
                                "year_ago": yago_eps,
                                "yoy_pct": pct,
                                "note": "EASY COMP (year-ago very low)",
                            })
                        elif yago_eps and yago_eps > 0:
                            pct = round(((curr_eps - yago_eps) / yago_eps) * 100, 1)
                            eps_yoy_changes.append({
                                "quarter": current_q["quarter"],
                                "current": curr_eps,
                                "year_ago": yago_eps,
                                "yoy_pct": pct,
                            })

            # ── EPS acceleration (Minervini: each qtr's YoY >= previous) ──
            eps_accelerating = False
            eps_decel_count = 0
            if len(eps_yoy_changes) >= 2:
                for i in range(len(eps_yoy_changes) - 1):
                    curr_pct = eps_yoy_changes[i].get("yoy_pct")
                    prev_pct = eps_yoy_changes[i + 1].get("yoy_pct")
                    if curr_pct is not None and prev_pct is not None:
                        if curr_pct < prev_pct:
                            eps_decel_count += 1
                if (eps_yoy_changes[0].get("yoy_pct") is not None
                        and eps_yoy_changes[-1].get("yoy_pct") is not None):
                    eps_accelerating = eps_yoy_changes[0]["yoy_pct"] > eps_yoy_changes[-1]["yoy_pct"]

            # ── Quarterly Revenue YoY (NEW — Minervini: sales must confirm EPS) ──
            quarterly_revenue = []
            revenue_yoy_changes = []
            try:
                qi = ticker.quarterly_income_stmt
                if qi is not None and not qi.empty:
                    for col_name in ["Total Revenue", "Revenue"]:
                        if col_name in qi.index:
                            rev_row = qi.loc[col_name].dropna().head(6)
                            for date, val in rev_row.items():
                                quarterly_revenue.append({
                                    "quarter": str(date)[:10],
                                    "revenue": round(float(val), 0),
                                })
                            break

                if len(quarterly_revenue) >= 5:
                    for i in range(min(4, len(quarterly_revenue) - 4)):
                        curr = quarterly_revenue[i]
                        yago_idx = i + 4
                        if yago_idx < len(quarterly_revenue):
                            yago_rev = quarterly_revenue[yago_idx]["revenue"]
                            curr_rev = curr["revenue"]
                            if yago_rev and yago_rev > 0:
                                pct = round(((curr_rev - yago_rev) / yago_rev) * 100, 1)
                                revenue_yoy_changes.append({
                                    "quarter": curr["quarter"],
                                    "current": curr_rev,
                                    "year_ago": yago_rev,
                                    "yoy_pct": pct,
                                })
            except Exception:
                pass

            # ── Derived metrics ──
            eps_growth_yoy = (
                eps_yoy_changes[0]["yoy_pct"]
                if eps_yoy_changes and eps_yoy_changes[0].get("yoy_pct") is not None
                else None
            )
            latest_eps_negative = quarterly_eps[0]["eps"] < 0 if quarterly_eps else False
            min_growth_pass = eps_growth_yoy is not None and eps_growth_yoy >= 20

            rev_growth = info.get("revenueGrowth")
            rev_growth_pct = round(rev_growth * 100, 1) if rev_growth else None
            earn_growth = info.get("earningsGrowth")
            earn_growth_pct = round(earn_growth * 100, 1) if earn_growth else None

            # ── EPS verdict (Minervini SEPA) ──
            if latest_eps_negative:
                eps_verdict = "REJECT (negative EPS)"
            elif eps_decel_count >= 2:
                eps_verdict = "REJECT (2+ decel)"
            elif eps_accelerating and min_growth_pass:
                eps_verdict = "PASS (accelerating)"
            elif min_growth_pass:
                eps_verdict = "PASS (>20% growth)"
            elif eps_growth_yoy is not None and eps_growth_yoy > 0:
                eps_verdict = "CAUTION (weak growth)"
            elif not eps_yoy_changes:
                eps_verdict = "INSUFFICIENT DATA"
            else:
                eps_verdict = "FAIL"

            fundamentals = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": round(info.get("trailingPE", 0), 1) if info.get("trailingPE") else None,
                "eps_ttm": info.get("trailingEps"),
                "roe": round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
                "debt_equity": round(info.get("debtToEquity", 0), 1) if info.get("debtToEquity") else None,
                "profit_margin": round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else None,
                "operating_margin": round(info.get("operatingMargins", 0) * 100, 1) if info.get("operatingMargins") else None,
                "revenue_growth_annual": rev_growth_pct,
                "earnings_growth_annual": earn_growth_pct,
                "eps_growth_yoy_calc": eps_growth_yoy,
                "institutional_pct": round(info.get("heldPercentInstitutions", 0) * 100, 1) if info.get("heldPercentInstitutions") else None,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                # Quarterly EPS (Minervini primary)
                "quarterly_eps": quarterly_eps[:6],
                "eps_yoy_changes": eps_yoy_changes,
                "eps_accelerating": eps_accelerating,
                "eps_decel_count": eps_decel_count,
                "latest_eps_negative": latest_eps_negative,
                "min_growth_pass": min_growth_pass,
                "eps_verdict": eps_verdict,
                # Quarterly Revenue (NEW)
                "quarterly_revenue": quarterly_revenue[:6],
                "revenue_yoy_changes": revenue_yoy_changes,
            }

            # Determine completeness
            issues = []
            if not quarterly_eps:
                issues.append(f"{symbol}: no quarterly EPS data")
            if eps_verdict == "INSUFFICIENT DATA":
                issues.append(f"{symbol}: insufficient EPS history for YoY calculation")

            return DataResult(
                symbol=symbol,
                data=fundamentals,
                provider=self.name,
                is_complete=len(issues) == 0,
                issues=issues,
            )

        except Exception as e:
            return DataResult(
                symbol=symbol,
                data=None,
                provider=self.name,
                is_complete=False,
                issues=[f"{symbol}: yfinance fundamentals error — {e}"],
            )

    def get_stock_list(self, market: str) -> DataResult:
        """
        Fetch stock list. Delegates to stock_universe.py for India.
        US list is not yet implemented in yfinance (handled by UniverseAgent).
        """
        issues = []

        if market == "india":
            try:
                from ..tools.stock_universe import get_all_indian_stocks
                stocks = get_all_indian_stocks()
                return DataResult(
                    symbol=f"universe:{market}",
                    data=stocks,
                    provider=self.name,
                    is_complete=len(stocks) > 0,
                    issues=[] if stocks else [f"No stocks returned for {market}"],
                )
            except Exception as e:
                return DataResult(
                    symbol=f"universe:{market}",
                    data=[],
                    provider=self.name,
                    is_complete=False,
                    issues=[f"Failed to fetch {market} stock list: {e}"],
                )
        else:
            return DataResult(
                symbol=f"universe:{market}",
                data=[],
                provider=self.name,
                is_complete=False,
                issues=[f"yfinance does not provide {market} stock lists — use UniverseAgent"],
            )
