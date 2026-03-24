"""
Fundamental Analysis for SEPA Trading Workflow (Consolidated)

Single source of truth for Minervini-faithful fundamental analysis:
- Quarterly EPS YoY from income statement (not info dict summaries)
- Quarterly Revenue/Sales YoY from income statement
- Easy comp detection (negative or very low year-ago values)
- Acceleration tracking (quarter-over-quarter improvement)
- SEPA EPS verdict: PASS / REJECT / CAUTION / FAIL / INSUFFICIENT DATA
- SEPA Revenue verdict: same pattern
- Summary metrics: ROE, D/E, margins, institutional %, market cap
"""

import logging
from typing import Dict, Optional, Any, List

import yfinance as yf

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Core: Quarterly YoY Analysis (EPS + Revenue)
# ═══════════════════════════════════════════════════════════════════════


def _extract_quarterly_series(
    ticker: yf.Ticker,
    row_names: List[str],
    n_quarters: int = 6,
) -> List[Dict]:
    """
    Extract a quarterly time series from quarterly_income_stmt.

    Args:
        ticker: yfinance Ticker object
        row_names: Row label candidates in priority order
                   (e.g. ['Diluted EPS', 'Basic EPS'] or ['Total Revenue'])
        n_quarters: Max quarters to return

    Returns:
        List of {"quarter": str, "value": float} dicts, most recent first.
    """
    try:
        qi = ticker.quarterly_income_stmt
        if qi is None or qi.empty:
            return []

        for row_name in row_names:
            if row_name in qi.index:
                series = qi.loc[row_name].dropna().head(n_quarters)
                return [
                    {"quarter": str(date)[:10], "value": round(float(val), 2)}
                    for date, val in series.items()
                ]
    except Exception as e:
        logger.debug(f"Failed to extract {row_names}: {e}")

    return []


def _compute_yoy_changes(
    quarterly: List[Dict],
    label: str = "EPS",
) -> List[Dict]:
    """
    Compute YoY % change for each quarter vs same quarter year-ago.

    Expects quarterly list ordered most-recent-first, with at least 5 entries
    to compute YoY for the most recent quarter (index 0 vs index 4).

    Returns list of YoY change dicts with easy comp detection.
    """
    yoy_changes = []

    if len(quarterly) < 5:
        return yoy_changes

    for i in range(min(4, len(quarterly) - 4)):
        current = quarterly[i]
        yago_idx = i + 4
        if yago_idx >= len(quarterly):
            break

        year_ago = quarterly[yago_idx]
        curr_val = current["value"]
        yago_val = year_ago["value"]

        entry = {
            "quarter": current["quarter"],
            "current": curr_val,
            "year_ago": yago_val,
            "yoy_pct": None,
        }

        if yago_val is not None and yago_val <= 0 and curr_val > 0:
            entry["note"] = f"EASY COMP (year-ago negative {label})"
            yoy_changes.append(entry)
        elif yago_val is not None and 0 < yago_val < 1.0 and label == "EPS":
            pct = round(((curr_val - yago_val) / yago_val) * 100, 1)
            entry["yoy_pct"] = pct
            entry["note"] = f"EASY COMP (year-ago very low {label})"
            yoy_changes.append(entry)
        elif yago_val is not None and yago_val > 0:
            pct = round(((curr_val - yago_val) / yago_val) * 100, 1)
            entry["yoy_pct"] = pct
            yoy_changes.append(entry)

    return yoy_changes


def _check_acceleration(yoy_changes: List[Dict]) -> tuple:
    """
    Check for acceleration: each quarter's YoY should be >= previous quarter's YoY.

    Returns:
        (is_accelerating: bool, decel_count: int)
    """
    decel_count = 0

    if len(yoy_changes) < 2:
        return False, 0

    for i in range(len(yoy_changes) - 1):
        curr_pct = yoy_changes[i].get("yoy_pct")
        prev_pct = yoy_changes[i + 1].get("yoy_pct")
        if curr_pct is not None and prev_pct is not None:
            if curr_pct < prev_pct:
                decel_count += 1

    # Accelerating = most recent YoY > oldest YoY
    first_pct = yoy_changes[0].get("yoy_pct")
    last_pct = yoy_changes[-1].get("yoy_pct")
    is_accelerating = (
        first_pct is not None
        and last_pct is not None
        and first_pct > last_pct
    )

    return is_accelerating, decel_count


def _make_verdict(
    latest_value_negative: bool,
    decel_count: int,
    is_accelerating: bool,
    min_growth_pass: bool,
    growth_yoy: Optional[float],
    yoy_changes: List[Dict],
) -> str:
    """Generate SEPA verdict string."""
    if latest_value_negative:
        return "REJECT (negative EPS)"
    if decel_count >= 2:
        return "REJECT (2+ decel)"
    if is_accelerating and min_growth_pass:
        return "PASS (accelerating)"
    if min_growth_pass:
        return "PASS (>20% growth)"
    if growth_yoy is not None and growth_yoy > 0:
        return "CAUTION (weak growth)"
    if not yoy_changes:
        return "INSUFFICIENT DATA"
    return "FAIL"


def _make_revenue_verdict(
    growth_yoy: Optional[float],
    is_accelerating: bool,
    decel_count: int,
    yoy_changes: List[Dict],
    min_growth_pct: float = 15.0,
) -> str:
    """Generate revenue verdict string."""
    min_growth_pass = growth_yoy is not None and growth_yoy >= min_growth_pct

    if is_accelerating and min_growth_pass:
        return "PASS (accelerating)"
    if min_growth_pass:
        return f"PASS (>{min_growth_pct}% growth)"
    if growth_yoy is not None and growth_yoy > 0:
        return "CAUTION (weak growth)"
    if growth_yoy is not None and growth_yoy <= 0:
        return "FAIL (declining)"
    if not yoy_changes:
        return "INSUFFICIENT DATA"
    return "FAIL"


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def get_fundamentals(symbol: str, market: str = "us") -> Dict[str, Any]:
    """
    Fetch SEPA-relevant fundamental data for a single stock.

    This is the Minervini-faithful version that pulls quarterly EPS and
    revenue from income statements (not summary info dict) and computes
    YoY changes, acceleration, and verdicts.

    Args:
        symbol: Stock symbol (e.g. 'TSM', 'LUPIN')
        market: 'india' or 'us' — determines .NS suffix for yfinance

    Returns:
        Dict with all fundamental metrics, quarterly breakdowns, and verdicts.
        On error, returns {"error": str}.
    """
    try:
        suffix = ".NS" if market == "india" else ""
        full = f"{symbol}{suffix}"
        ticker = yf.Ticker(full)

        info = ticker.info or {}

        # ── Quarterly EPS ────────────────────────────────────────────
        quarterly_eps_raw = _extract_quarterly_series(
            ticker, ["Diluted EPS", "Basic EPS"], n_quarters=6
        )
        quarterly_eps = [
            {"quarter": q["quarter"], "eps": q["value"]}
            for q in quarterly_eps_raw
        ]

        eps_yoy_changes = _compute_yoy_changes(quarterly_eps_raw, label="EPS")
        eps_accelerating, eps_decel_count = _check_acceleration(eps_yoy_changes)

        eps_growth_yoy = (
            eps_yoy_changes[0]["yoy_pct"]
            if eps_yoy_changes and eps_yoy_changes[0].get("yoy_pct") is not None
            else None
        )

        latest_eps_negative = quarterly_eps[0]["eps"] < 0 if quarterly_eps else False
        min_eps_growth_pass = eps_growth_yoy is not None and eps_growth_yoy >= 20

        eps_verdict = _make_verdict(
            latest_eps_negative, eps_decel_count, eps_accelerating,
            min_eps_growth_pass, eps_growth_yoy, eps_yoy_changes,
        )

        # ── Quarterly Revenue ────────────────────────────────────────
        quarterly_rev_raw = _extract_quarterly_series(
            ticker, ["Total Revenue", "Revenue"], n_quarters=6
        )
        quarterly_revenue = [
            {"quarter": q["quarter"], "revenue": q["value"]}
            for q in quarterly_rev_raw
        ]

        rev_yoy_changes = _compute_yoy_changes(quarterly_rev_raw, label="Revenue")
        rev_accelerating, rev_decel_count = _check_acceleration(rev_yoy_changes)

        rev_growth_yoy = (
            rev_yoy_changes[0]["yoy_pct"]
            if rev_yoy_changes and rev_yoy_changes[0].get("yoy_pct") is not None
            else None
        )

        revenue_verdict = _make_revenue_verdict(
            rev_growth_yoy, rev_accelerating, rev_decel_count, rev_yoy_changes,
        )

        # ── Summary info metrics ─────────────────────────────────────
        rev_growth = info.get("revenueGrowth")
        rev_growth_pct = round(rev_growth * 100, 1) if rev_growth else None

        earn_growth = info.get("earningsGrowth")
        earn_growth_pct = round(earn_growth * 100, 1) if earn_growth else None

        return {
            # Summary metrics
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE", 0), 1) if info.get("trailingPE") else None,
            "eps_ttm": info.get("trailingEps"),
            "roe": round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
            "debt_equity": round(info.get("debtToEquity", 0), 1) if info.get("debtToEquity") else None,
            "profit_margin": round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else None,
            "operating_margin": round(info.get("operatingMargins", 0) * 100, 1) if info.get("operatingMargins") else None,
            "gross_margin": round(info.get("grossMargins", 0) * 100, 1) if info.get("grossMargins") else None,
            "revenue_growth": rev_growth_pct,
            "earnings_growth": earn_growth_pct,
            "institutional_pct": round(info.get("heldPercentInstitutions", 0) * 100, 1) if info.get("heldPercentInstitutions") else None,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            # Quarterly EPS analysis
            "quarterly_eps": quarterly_eps[:6],
            "eps_yoy_changes": eps_yoy_changes,
            "eps_growth_yoy_calc": eps_growth_yoy,
            "eps_accelerating": eps_accelerating,
            "eps_decel_count": eps_decel_count,
            "latest_eps_negative": latest_eps_negative,
            "min_growth_pass": min_eps_growth_pass,
            "eps_verdict": eps_verdict,
            # Quarterly Revenue analysis (NEW)
            "quarterly_revenue": quarterly_revenue[:6],
            "rev_yoy_changes": rev_yoy_changes,
            "rev_growth_yoy_calc": rev_growth_yoy,
            "rev_accelerating": rev_accelerating,
            "rev_decel_count": rev_decel_count,
            "revenue_verdict": revenue_verdict,
        }
    except Exception as e:
        return {"error": str(e)}


def check_fundamentals(
    symbol: str,
    config: Optional[Dict] = None,
    market: str = "us",
) -> Dict[str, Any]:
    """
    Perform comprehensive SEPA fundamental check.

    Backward-compatible wrapper used by FundamentalAgent. Returns a dict
    with 'all_passed' bool and detailed sub-checks.

    Args:
        symbol: Stock symbol
        config: Fundamental criteria config (from fundamental_criteria.yaml)
        market: 'india' or 'us'

    Returns:
        Dict with all_passed, earnings_check, revenue_check, margins_check, raw data.
    """
    if config is None:
        config = {
            "min_eps_growth_yoy_pct": 20,
            "min_revenue_growth_pct": 15,
            "margin_expansion_required": False,
            "min_institutional_holding_pct": 10,
        }

    data = get_fundamentals(symbol, market=market)

    if "error" in data:
        return {
            "symbol": symbol,
            "all_passed": False,
            "error": data["error"],
            "earnings_check": {"passed_min_growth": False, "details": data["error"]},
            "revenue_check": {"passed_min_growth": False},
            "margins_check": {"passed": False},
        }

    # Earnings check
    min_eps = config.get("min_eps_growth_yoy_pct", 20)
    eps_growth = data.get("eps_growth_yoy_calc")
    passed_earnings = eps_growth is not None and eps_growth >= min_eps
    eps_accel_required = config.get("eps_acceleration_required", False)
    if eps_accel_required and not data.get("eps_accelerating", False):
        passed_earnings = False

    earnings_check = {
        "passed_min_growth": passed_earnings,
        "eps_acceleration": data.get("eps_accelerating", False),
        "latest_growth_pct": eps_growth or 0.0,
        "verdict": data.get("eps_verdict", "INSUFFICIENT DATA"),
        "details": data.get("eps_verdict", "INSUFFICIENT DATA"),
    }

    # Revenue check
    min_rev = config.get("min_revenue_growth_pct", 15)
    rev_growth = data.get("rev_growth_yoy_calc")
    passed_revenue = rev_growth is not None and rev_growth >= min_rev

    revenue_check = {
        "passed_min_growth": passed_revenue,
        "latest_growth_pct": rev_growth or 0.0,
        "verdict": data.get("revenue_verdict", "INSUFFICIENT DATA"),
    }

    # Margins check
    margins_ok = True
    if config.get("margin_expansion_required", False):
        margins_ok = (data.get("profit_margin") or 0) > 10.0

    margins_check = {
        "passed": margins_ok,
        "profit_margin": data.get("profit_margin"),
        "operating_margin": data.get("operating_margin"),
        "gross_margin": data.get("gross_margin"),
    }

    # Institutional check
    min_inst = config.get("min_institutional_holding_pct", 0)
    inst_pct = data.get("institutional_pct") or 0
    passed_institutional = inst_pct >= min_inst if min_inst > 0 else True

    all_passed = passed_earnings and passed_revenue and margins_ok and passed_institutional

    return {
        "symbol": symbol,
        "all_passed": all_passed,
        "earnings_check": earnings_check,
        "revenue_check": revenue_check,
        "margins_check": margins_check,
        "institutional_pct": inst_pct,
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Legacy wrappers (kept for backward compatibility with __init__.py)
# ═══════════════════════════════════════════════════════════════════════


def analyze_earnings_growth(
    data: Dict[str, Any],
    min_growth_pct: float = 20.0,
) -> Dict[str, Any]:
    """Analyze EPS growth from fundamentals data dict."""
    eps_growth = data.get("eps_growth_yoy_calc") or data.get("earnings_growth") or 0
    return {
        "passed_min_growth": eps_growth >= min_growth_pct,
        "eps_acceleration": data.get("eps_accelerating", False),
        "latest_growth_pct": eps_growth,
        "details": data.get("eps_verdict", "INSUFFICIENT DATA"),
    }


def analyze_revenue_growth(
    data: Dict[str, Any],
    min_growth_pct: float = 15.0,
) -> Dict[str, Any]:
    """Analyze revenue growth from fundamentals data dict."""
    rev_growth = data.get("rev_growth_yoy_calc") or data.get("revenue_growth") or 0
    return {
        "passed_min_growth": rev_growth >= min_growth_pct,
        "latest_growth_pct": rev_growth,
        "verdict": data.get("revenue_verdict", "INSUFFICIENT DATA"),
    }


def analyze_margins(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze profit margins from fundamentals data dict."""
    return {
        "profit_margin": data.get("profit_margin") or 0,
        "operating_margin": data.get("operating_margin") or 0,
        "gross_margin": data.get("gross_margin") or 0,
        "healthy_margins": (data.get("profit_margin") or 0) > 5.0,
    }


if __name__ == "__main__":
    # Quick test
    symbol = "TSM"
    print(f"Analyzing fundamentals for {symbol}...")
    result = get_fundamentals(symbol, market="us")
    print(f"EPS Verdict: {result.get('eps_verdict')}")
    print(f"Revenue Verdict: {result.get('revenue_verdict')}")
    print(f"Quarterly EPS: {result.get('quarterly_eps')}")
    print(f"Quarterly Revenue: {result.get('quarterly_revenue')}")
    print()
    check = check_fundamentals(symbol)
    print(f"Overall Pass: {check['all_passed']}")
