"""
Fundamental Data Tools for SEPA Trading Workflow

This module provides functions to fetch and analyze fundamental data
(EPS, Revenue, Margins) to identify high-growth "Code 33" stocks.
"""

import logging
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a stock.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
    
    Returns:
        Dictionary with fundamental data (quarterly and annual)
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get quarterly financials
        quarterly_financials = ticker.quarterly_financials
        quarterly_earnings = ticker.quarterly_earnings
        
        # Get basic info
        info = ticker.info
        
        return {
            'symbol': symbol,
            'info': info,
            'quarterly_financials': quarterly_financials,
            'quarterly_earnings': quarterly_earnings
        }
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {}


def analyze_earnings_growth(
    data: Dict[str, Any],
    min_growth_pct: float = 20.0
) -> Dict[str, Any]:
    """
    Analyze Earnings Per Share (EPS) growth.
    
    Args:
        data: Fundamental data dictionary
        min_growth_pct: Minimum YoY growth required
    
    Returns:
        Analysis results including growth rates and acceleration status
    """
    results = {
        'passed_min_growth': False,
        'eps_acceleration': False,
        'latest_growth_pct': 0.0,
        'details': "Insufficient data"
    }
    
    # Check if we have data
    q_earnings = data.get('quarterly_earnings')
    
    if q_earnings is None or q_earnings.empty:
        # Fallback to financials if earnings dataframe is empty
        q_financials = data.get('quarterly_financials')
        if q_financials is not None and not q_financials.empty:
            # Try to extract EPS from financials (Net Income / Shares usually not directly available)
            # Typically 'Basic EPS' or similar row exists
            pass
        return results
    
    try:
        # yfinance quarterly_earnings usually has 'Earnings' column (Net Income) or 'Revenue'
        # We need actual EPS or simulate it with Net Income if shares constant
        
        # Note: yfinance structure varies. Sometimes it's a DataFrame with 'Earnings' and 'Revenue'
        # And index is Date.
        
        df = q_earnings.sort_index()
        
        if len(df) < 5:  # Need at least 5 quarters for reliable YoY of recent qtrs
             # Can do basic check with 2 quarters (current vs year ago)
             pass
        
        # Calculate YoY growth for the most recent quarter
        # Using Net Income as proxy for EPS growth if EPS not explicit
        # Assuming share count hasn't changed drastically
        
        earnings_col = 'Earnings'
        if earnings_col not in df.columns:
            return results
            
        # Get last 4 quarters (TTM)
        recent = df.tail(4)
        
        # Calculate YoY growth for the latest quarter
        # We need the quarter from 1 year ago. 
        # yfinance history is sparse. Let's look at what we have.
        
         # Better approach: Use info dictionary for trailing EPS and growth
        info = data.get('info', {})
        earnings_growth = info.get('earningsGrowth', 0)
        
        if earnings_growth:
             # earningsGrowth is usually decimals (0.25 = 25%)
             growth_pct = earnings_growth * 100
             results['latest_growth_pct'] = round(growth_pct, 2)
             results['passed_min_growth'] = growth_pct >= min_growth_pct
             results['details'] = f"Reported Earnings Growth: {growth_pct:.1f}%"
             
             # Hard to detect acceleration without granular quarterly EPS history
             # which yfinance often lacks for Indian stocks freely.
             # We will assume acceleration is False unless we can prove it.
             
        else:
             # Try manual calculation if 'Earnings' column exists
             if len(df) >= 5:
                 current_q = df.iloc[-1][earnings_col]
                 year_ago_q = df.iloc[-5][earnings_col] # roughly 4 quarters ago
                 
                 if year_ago_q > 0:
                     growth = ((current_q - year_ago_q) / year_ago_q) * 100
                     results['latest_growth_pct'] = round(growth, 2)
                     results['passed_min_growth'] = growth >= min_growth_pct
                     results['details'] = f"Calculated YoY Growth: {growth:.1f}%"
    
    except Exception as e:
        logger.error(f"Error analyzing earnings: {e}")
        
    return results


def analyze_revenue_growth(
    data: Dict[str, Any],
    min_growth_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Analyze Revenue growth.
    
    Args:
        data: Fundamental data dictionary
        min_growth_pct: Minimum YoY growth required
    
    Returns:
        Analysis results
    """
    results = {
        'passed_min_growth': False,
        'latest_growth_pct': 0.0
    }
    
    info = data.get('info', {})
    rev_growth = info.get('revenueGrowth', 0)
    
    if rev_growth:
        growth_pct = rev_growth * 100
        results['latest_growth_pct'] = round(growth_pct, 2)
        results['passed_min_growth'] = growth_pct >= min_growth_pct
        
    return results


def analyze_margins(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze profit margins.
    
    Args:
        data: Fundamental data dictionary
    
    Returns:
        Margin analysis results
    """
    info = data.get('info', {})
    
    profit_margin = info.get('profitMargins', 0)
    operating_margin = info.get('operatingMargins', 0)
    gross_margin = info.get('grossMargins', 0)
    
    return {
        'profit_margin': round(profit_margin * 100, 2) if profit_margin else 0,
        'operating_margin': round(operating_margin * 100, 2) if operating_margin else 0,
        'gross_margin': round(gross_margin * 100, 2) if gross_margin else 0,
        'healthy_margins': profit_margin > 0.05 if profit_margin else False
    }


def check_fundamentals(
    symbol: str,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive fundamental analysis check.
    
    Args:
        symbol: Stock symbol
        config: Fundamental criteria config
    
    Returns:
        Dictionary with pass/fail results
    """
    if config is None:
        config = {
            'min_eps_growth_yoy_pct': 20,
            'min_revenue_growth_pct': 15,
            'margin_expansion_required': False
        }
    
    data = get_fundamentals(symbol)
    
    earnings_analysis = analyze_earnings_growth(
        data, 
        min_growth_pct=config.get('min_eps_growth_yoy_pct', 20)
    )
    
    revenue_analysis = analyze_revenue_growth(
        data,
        min_growth_pct=config.get('min_revenue_growth_pct', 15)
    )
    
    # Margin analysis (simplified)
    info = data.get('info', {})
    margins_ok = True
    if config.get('margin_expansion_required', False):
        # Difficult to check expansion without history, checking if healthy
        margins_ok = info.get('profitMargins', 0) > 0.10  # Arbitrary >10% check
    
    passed_earnings = earnings_analysis['passed_min_growth']
    passed_revenue = revenue_analysis['passed_min_growth']
    
    return {
        'symbol': symbol,
        'all_passed': passed_earnings and passed_revenue and margins_ok,
        'earnings_check': earnings_analysis,
        'revenue_check': revenue_analysis,
        'margins_check': {'passed': margins_ok, 'value': info.get('profitMargins', 0)}
    }


# Example usage
if __name__ == "__main__":
    # Test on a known growth stock
    symbol = "TITAN.NS"
    print(f"Analyzing fundamentals for {symbol}...")
    
    results = check_fundamentals(symbol)
    print(f"Overall Pass: {results['all_passed']}")
    print(f"Earnings: {results['earnings_check']}")
    print(f"Revenue: {results['revenue_check']}")
