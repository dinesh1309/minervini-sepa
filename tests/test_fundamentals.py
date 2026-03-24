"""
Tests for the consolidated fundamental analysis module.

Covers:
    - Quarterly EPS YoY computation (from raw series)
    - Quarterly Revenue YoY computation
    - Easy comp detection
    - Acceleration/deceleration checks
    - EPS verdict logic
    - Revenue verdict logic
    - check_fundamentals backward compatibility
    - Legacy wrapper functions
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.tools.fundamental_analysis import (
    _extract_quarterly_series,
    _compute_yoy_changes,
    _check_acceleration,
    _make_verdict,
    _make_revenue_verdict,
    get_fundamentals,
    check_fundamentals,
    analyze_earnings_growth,
    analyze_revenue_growth,
    analyze_margins,
)


# ═══════════════════════════════════════════════════════════════════════
# YoY Computation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeYoyChanges:
    """Test _compute_yoy_changes with synthetic data."""

    def test_normal_growth(self):
        """6 quarters with steady growth should produce 2 YoY entries."""
        quarterly = [
            {"quarter": "2026-01-01", "value": 1.50},
            {"quarter": "2025-10-01", "value": 1.40},
            {"quarter": "2025-07-01", "value": 1.30},
            {"quarter": "2025-04-01", "value": 1.20},
            {"quarter": "2025-01-01", "value": 1.00},
            {"quarter": "2024-10-01", "value": 0.90},
        ]
        result = _compute_yoy_changes(quarterly, label="EPS")
        assert len(result) == 2
        # 1.50 vs 1.00 = 50% growth
        assert result[0]["yoy_pct"] == 50.0
        # 1.40 vs 0.90 = 55.6% growth
        assert result[1]["yoy_pct"] == 55.6

    def test_easy_comp_negative_year_ago(self):
        """Negative year-ago EPS should be flagged as easy comp."""
        quarterly = [
            {"quarter": "2026-01-01", "value": 1.00},
            {"quarter": "2025-10-01", "value": 0.80},
            {"quarter": "2025-07-01", "value": 0.50},
            {"quarter": "2025-04-01", "value": 0.30},
            {"quarter": "2025-01-01", "value": -0.50},  # negative year-ago
            {"quarter": "2024-10-01", "value": 0.60},
        ]
        result = _compute_yoy_changes(quarterly, label="EPS")
        assert len(result) == 2
        assert result[0]["yoy_pct"] is None  # can't compute % off negative
        assert "EASY COMP" in result[0].get("note", "")

    def test_easy_comp_very_low_year_ago(self):
        """Very low year-ago EPS (< 1.0) should be flagged for EPS."""
        quarterly = [
            {"quarter": "2026-01-01", "value": 5.00},
            {"quarter": "2025-10-01", "value": 4.50},
            {"quarter": "2025-07-01", "value": 4.00},
            {"quarter": "2025-04-01", "value": 3.50},
            {"quarter": "2025-01-01", "value": 0.50},  # very low
            {"quarter": "2024-10-01", "value": 3.00},
        ]
        result = _compute_yoy_changes(quarterly, label="EPS")
        assert any("EASY COMP" in e.get("note", "") for e in result)

    def test_insufficient_data(self):
        """Less than 5 quarters should return empty list."""
        quarterly = [
            {"quarter": "2026-01-01", "value": 1.50},
            {"quarter": "2025-10-01", "value": 1.40},
            {"quarter": "2025-07-01", "value": 1.30},
        ]
        result = _compute_yoy_changes(quarterly, label="EPS")
        assert result == []

    def test_revenue_no_easy_comp_for_low(self):
        """Revenue YoY should NOT flag very low year-ago as easy comp (only EPS does)."""
        quarterly = [
            {"quarter": "2026-01-01", "value": 5.00},
            {"quarter": "2025-10-01", "value": 4.50},
            {"quarter": "2025-07-01", "value": 4.00},
            {"quarter": "2025-04-01", "value": 3.50},
            {"quarter": "2025-01-01", "value": 0.50},
            {"quarter": "2024-10-01", "value": 3.00},
        ]
        result = _compute_yoy_changes(quarterly, label="Revenue")
        # For Revenue, low year-ago (0.50) with positive value computes normal %
        assert result[0]["yoy_pct"] is not None
        assert "EASY COMP" not in result[0].get("note", "")


# ═══════════════════════════════════════════════════════════════════════
# Acceleration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCheckAcceleration:

    def test_accelerating(self):
        """Most recent YoY > oldest YoY = accelerating."""
        changes = [
            {"yoy_pct": 50.0},
            {"yoy_pct": 30.0},
        ]
        is_accel, decel = _check_acceleration(changes)
        assert is_accel is True
        assert decel == 0

    def test_decelerating(self):
        """Most recent YoY < oldest YoY = decelerating."""
        changes = [
            {"yoy_pct": 20.0},
            {"yoy_pct": 40.0},
        ]
        is_accel, decel = _check_acceleration(changes)
        assert is_accel is False
        assert decel == 1

    def test_two_plus_decel(self):
        """Multiple consecutive decelerations."""
        changes = [
            {"yoy_pct": 10.0},
            {"yoy_pct": 20.0},
            {"yoy_pct": 30.0},
        ]
        is_accel, decel = _check_acceleration(changes)
        assert decel == 2

    def test_insufficient_data(self):
        changes = [{"yoy_pct": 50.0}]
        is_accel, decel = _check_acceleration(changes)
        assert is_accel is False
        assert decel == 0


# ═══════════════════════════════════════════════════════════════════════
# Verdict Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEpsVerdict:

    def test_reject_negative_eps(self):
        v = _make_verdict(True, 0, False, False, None, [])
        assert v == "REJECT (negative EPS)"

    def test_reject_two_plus_decel(self):
        v = _make_verdict(False, 2, False, True, 25.0, [{"yoy_pct": 25.0}])
        assert v == "REJECT (2+ decel)"

    def test_pass_accelerating(self):
        v = _make_verdict(False, 0, True, True, 30.0, [{"yoy_pct": 30.0}])
        assert v == "PASS (accelerating)"

    def test_pass_strong_growth(self):
        v = _make_verdict(False, 0, False, True, 25.0, [{"yoy_pct": 25.0}])
        assert v == "PASS (>20% growth)"

    def test_caution_weak_growth(self):
        v = _make_verdict(False, 0, False, False, 10.0, [{"yoy_pct": 10.0}])
        assert v == "CAUTION (weak growth)"

    def test_insufficient_data(self):
        v = _make_verdict(False, 0, False, False, None, [])
        assert v == "INSUFFICIENT DATA"

    def test_fail(self):
        v = _make_verdict(False, 0, False, False, -5.0, [{"yoy_pct": -5.0}])
        assert v == "FAIL"


class TestRevenueVerdict:

    def test_pass_accelerating(self):
        v = _make_revenue_verdict(25.0, True, 0, [{"yoy_pct": 25.0}])
        assert v == "PASS (accelerating)"

    def test_pass_strong(self):
        v = _make_revenue_verdict(20.0, False, 0, [{"yoy_pct": 20.0}])
        assert "PASS" in v

    def test_caution_weak(self):
        v = _make_revenue_verdict(5.0, False, 0, [{"yoy_pct": 5.0}])
        assert v == "CAUTION (weak growth)"

    def test_fail_declining(self):
        v = _make_revenue_verdict(-10.0, False, 0, [{"yoy_pct": -10.0}])
        assert v == "FAIL (declining)"

    def test_insufficient_data(self):
        v = _make_revenue_verdict(None, False, 0, [])
        assert v == "INSUFFICIENT DATA"


# ═══════════════════════════════════════════════════════════════════════
# get_fundamentals Integration Tests (mocked yfinance)
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_ticker(
    eps_data=None,
    revenue_data=None,
    info=None,
):
    """Create a mock yfinance Ticker with configurable income statement."""
    mock = MagicMock()

    # Build quarterly_income_stmt DataFrame
    dates = pd.date_range("2024-10-01", periods=6, freq="QS")[::-1]
    rows = {}

    if eps_data is not None:
        rows["Diluted EPS"] = eps_data
    if revenue_data is not None:
        rows["Total Revenue"] = revenue_data

    if rows:
        mock.quarterly_income_stmt = pd.DataFrame(rows, index=dates).T
    else:
        mock.quarterly_income_stmt = pd.DataFrame()

    mock.info = info or {}
    return mock


class TestGetFundamentals:

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_strong_growth_stock(self, mock_ticker_cls):
        """Stock with 6 quarters of growing EPS should get PASS verdict."""
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[1.50, 1.40, 1.30, 1.20, 1.00, 0.90],
            revenue_data=[500, 480, 450, 420, 380, 350],
            info={
                "marketCap": 5_000_000_000,
                "trailingPE": 30.0,
                "trailingEps": 5.40,
                "returnOnEquity": 0.22,
                "debtToEquity": 15.0,
                "profitMargins": 0.18,
                "operatingMargins": 0.25,
                "grossMargins": 0.45,
                "revenueGrowth": 0.28,
                "earningsGrowth": 0.35,
                "heldPercentInstitutions": 0.65,
                "sector": "Technology",
                "industry": "Semiconductors",
            },
        )

        result = get_fundamentals("TSM", market="us")

        assert "error" not in result
        assert result["eps_verdict"].startswith("PASS")
        assert result["quarterly_eps"][0]["eps"] == 1.50
        assert len(result["quarterly_eps"]) == 6
        # Revenue should also be present
        assert len(result["quarterly_revenue"]) == 6
        assert result["revenue_verdict"] != "INSUFFICIENT DATA"

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_negative_eps_stock(self, mock_ticker_cls):
        """Stock with negative latest EPS should get REJECT."""
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[-0.20, -0.10, 0.05, 0.10, 0.20, 0.25],
            info={"marketCap": 200_000_000},
        )

        result = get_fundamentals("BADSTOCK", market="us")

        assert result["eps_verdict"] == "REJECT (negative EPS)"
        assert result["latest_eps_negative"] is True

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_india_market_uses_ns_suffix(self, mock_ticker_cls):
        """India market should append .NS to symbol."""
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            info={},
        )

        get_fundamentals("LUPIN", market="india")
        mock_ticker_cls.assert_called_once_with("LUPIN.NS")

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_error_handling(self, mock_ticker_cls):
        """Exception should return error dict, not raise."""
        mock_ticker_cls.side_effect = Exception("API down")

        result = get_fundamentals("BROKEN", market="us")
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
# check_fundamentals Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestCheckFundamentals:

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_returns_all_passed_true_for_strong_stock(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[2.00, 1.80, 1.60, 1.40, 1.00, 0.80],
            revenue_data=[1000, 950, 900, 850, 700, 650],
            info={
                "profitMargins": 0.20,
                "operatingMargins": 0.30,
                "grossMargins": 0.50,
                "heldPercentInstitutions": 0.40,
            },
        )

        result = check_fundamentals("STRONG", market="us")

        assert result["all_passed"] is True
        assert result["earnings_check"]["passed_min_growth"] is True
        assert result["revenue_check"]["passed_min_growth"] is True

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_returns_all_passed_false_for_weak_stock(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[-0.50, -0.30, -0.10, 0.10, 0.20, 0.30],
            revenue_data=[100, 105, 110, 115, 120, 125],
            info={"profitMargins": -0.05},
        )

        result = check_fundamentals("WEAK", market="us")

        assert result["all_passed"] is False

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_error_returns_dict_not_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("timeout")

        result = check_fundamentals("FAIL", market="us")
        assert result["all_passed"] is False
        assert "error" in result

    @patch("src.tools.fundamental_analysis.yf.Ticker")
    def test_config_override(self, mock_ticker_cls):
        """Custom config thresholds should be respected."""
        mock_ticker_cls.return_value = _make_mock_ticker(
            eps_data=[1.50, 1.40, 1.30, 1.20, 1.00, 0.90],
            revenue_data=[500, 480, 450, 420, 380, 350],
            info={"profitMargins": 0.20, "heldPercentInstitutions": 0.05},
        )

        # Very high bar — should fail institutional check
        config = {
            "min_eps_growth_yoy_pct": 20,
            "min_revenue_growth_pct": 15,
            "min_institutional_holding_pct": 50,  # 50% — most stocks won't pass
        }
        result = check_fundamentals("TEST", config=config, market="us")
        assert result["all_passed"] is False


# ═══════════════════════════════════════════════════════════════════════
# Legacy Wrapper Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLegacyWrappers:

    def test_analyze_earnings_growth(self, fundamentals_strong):
        result = analyze_earnings_growth(fundamentals_strong)
        assert result["passed_min_growth"] is True
        assert result["eps_acceleration"] is True

    def test_analyze_revenue_growth(self):
        data = {"rev_growth_yoy_calc": 25.0, "revenue_verdict": "PASS (>15% growth)"}
        result = analyze_revenue_growth(data)
        assert result["passed_min_growth"] is True

    def test_analyze_margins(self):
        data = {"profit_margin": 18.0, "operating_margin": 25.0, "gross_margin": 45.0}
        result = analyze_margins(data)
        assert result["healthy_margins"] is True
        assert result["profit_margin"] == 18.0
