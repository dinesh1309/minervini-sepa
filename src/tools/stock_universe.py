"""
Stock Universe Tools for SEPA Trading Workflow

This module provides functions to fetch and manage the complete list of
Indian equities from NSE and BSE exchanges.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
STOCK_LIST_CACHE_FILE = CACHE_DIR / "indian_stocks.json"
CACHE_EXPIRY_DAYS = 1


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_cache_valid() -> bool:
    """Check if cached stock list is still valid."""
    if not STOCK_LIST_CACHE_FILE.exists():
        return False
    
    modified_time = datetime.fromtimestamp(STOCK_LIST_CACHE_FILE.stat().st_mtime)
    return datetime.now() - modified_time < timedelta(days=CACHE_EXPIRY_DAYS)


def _fetch_nse_stocks() -> List[Dict]:
    """
    Fetch list of stocks from NSE.
    Uses NSE's official API endpoint.
    """
    stocks = []
    
    try:
        # NSE equity list endpoint
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        # Skip header
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 2:
                symbol = parts[0].strip().strip('"')
                name = parts[1].strip().strip('"') if len(parts) > 1 else symbol
                
                # Filter out invalid entries
                if symbol and not symbol.startswith(' '):
                    stocks.append({
                        "symbol": symbol,
                        "name": name,
                        "exchange": "NSE",
                        "yfinance_symbol": f"{symbol}.NS"
                    })
        
        logger.info(f"Fetched {len(stocks)} stocks from NSE")
        
    except Exception as e:
        logger.error(f"Error fetching NSE stocks: {e}")
        # Fallback: Try nsetools if available
        try:
            from nsetools import Nse
            nse = Nse()
            stock_codes = nse.get_stock_codes()
            for symbol, name in stock_codes.items():
                if symbol != 'SYMBOL':  # Skip header
                    stocks.append({
                        "symbol": symbol,
                        "name": name,
                        "exchange": "NSE",
                        "yfinance_symbol": f"{symbol}.NS"
                    })
            logger.info(f"Fetched {len(stocks)} stocks from NSE via nsetools")
        except ImportError:
            logger.warning("nsetools not available for fallback")
    
    return stocks


def _fetch_bse_stocks() -> List[Dict]:
    """
    Fetch list of stocks from BSE.
    Uses BSE's official API endpoint.
    """
    stocks = []
    
    try:
        # BSE equity list endpoint
        url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Atea=&status=Active"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.bseindia.com/"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        for item in data:
            symbol = item.get('scrip_cd', '')
            name = item.get('scrip_name', '')
            
            if symbol:
                stocks.append({
                    "symbol": str(symbol),
                    "name": name,
                    "exchange": "BSE",
                    "yfinance_symbol": f"{symbol}.BO"
                })
        
        logger.info(f"Fetched {len(stocks)} stocks from BSE")
        
    except Exception as e:
        logger.error(f"Error fetching BSE stocks: {e}")
        # Fallback: Try bsedata if available
        try:
            from bsedata.bse import BSE
            bse = BSE()
            # Note: bsedata has limited stock list functionality
            logger.warning("BSE fallback via bsedata has limited functionality")
        except ImportError:
            logger.warning("bsedata not available for fallback")
    
    return stocks


def get_all_indian_stocks(
    force_refresh: bool = False,
    include_bse: bool = False
) -> List[Dict]:
    """
    Fetch complete list of Indian stocks from NSE (and optionally BSE).
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data.
        include_bse: If True, include BSE-only stocks (slower).
    
    Returns:
        List of stock dictionaries with keys:
        - symbol: Trading symbol
        - name: Company name
        - exchange: NSE or BSE
        - yfinance_symbol: Symbol for yfinance API (.NS or .BO suffix)
    
    Example:
        >>> stocks = get_all_indian_stocks()
        >>> print(f"Found {len(stocks)} stocks")
        >>> print(stocks[0])
        {'symbol': 'RELIANCE', 'name': 'Reliance Industries', 'exchange': 'NSE', 'yfinance_symbol': 'RELIANCE.NS'}
    """
    _ensure_cache_dir()
    
    # Check cache
    if not force_refresh and _is_cache_valid():
        try:
            with open(STOCK_LIST_CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
                logger.info(f"Loaded {len(cached_data)} stocks from cache")
                return cached_data
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    
    # Fetch fresh data
    all_stocks = []
    
    # Always fetch NSE stocks (primary market)
    nse_stocks = _fetch_nse_stocks()
    all_stocks.extend(nse_stocks)
    
    # Optionally fetch BSE stocks
    if include_bse:
        bse_stocks = _fetch_bse_stocks()
        
        # Deduplicate: Keep NSE version if stock is listed on both
        nse_symbols = {s['symbol'] for s in nse_stocks}
        unique_bse = [s for s in bse_stocks if s['symbol'] not in nse_symbols]
        all_stocks.extend(unique_bse)
    
    # Filter out suspended/delisted stocks (basic filtering)
    filtered_stocks = filter_active_stocks(all_stocks)
    
    # Save to cache
    try:
        with open(STOCK_LIST_CACHE_FILE, 'w') as f:
            json.dump(filtered_stocks, f, indent=2)
        logger.info(f"Cached {len(filtered_stocks)} stocks")
    except Exception as e:
        logger.warning(f"Error saving cache: {e}")
    
    return filtered_stocks


def filter_active_stocks(stocks: List[Dict]) -> List[Dict]:
    """
    Filter out delisted, suspended, or invalid stocks.
    
    Args:
        stocks: List of stock dictionaries
    
    Returns:
        Filtered list of active stocks
    """
    filtered = []
    
    # Patterns to exclude (delisted, suspended, etc.)
    exclude_patterns = [
        '-BE',  # Trade-to-trade segment
        '-BL',  # Block deal
        '-IL',  # Illiquid
        '-SM',  # SME platform (small caps with different rules)
    ]
    
    for stock in stocks:
        symbol = stock.get('symbol', '')
        
        # Skip if symbol matches exclusion patterns
        if any(pattern in symbol for pattern in exclude_patterns):
            continue
        
        # Skip if symbol is too short or suspicious
        if len(symbol) < 2:
            continue
        
        filtered.append(stock)
    
    logger.info(f"Filtered to {len(filtered)} active stocks from {len(stocks)} total")
    return filtered


def get_nifty50_stocks() -> List[str]:
    """
    Get list of NIFTY 50 constituent symbols.
    Useful for testing and benchmarking.
    
    Returns:
        List of NIFTY 50 stock symbols (without .NS suffix)
    """
    # NIFTY 50 constituents (as of early 2024, update periodically)
    return [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
        "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
        "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
        "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
        "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "LTIM"
    ]


def get_nifty500_stocks() -> List[str]:
    """
    Get list of NIFTY 500 constituent symbols.
    The broader index covers most investable stocks.
    
    Returns:
        List of NIFTY 500 stock symbols
    """
    # For a complete list, we'd need to fetch from NSE
    # This is a placeholder that returns NIFTY 50 + common large/mid caps
    # In production, fetch from: https://www.niftyindices.com/
    nifty50 = get_nifty50_stocks()
    
    # Add some additional large/mid caps (simplified list)
    additional = [
        "ABBOTINDIA", "ACC", "ADANIGREEN", "ADANITRANS", "ALKEM",
        "AMBUJACEM", "AUROPHARMA", "BANDHANBNK", "BANKBARODA", "BEL",
        "BERGEPAINT", "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN",
        "COLPAL", "CONCOR", "CROMPTON", "DABUR", "DLF",
        "ESCORTS", "FEDERALBNK", "GAIL", "GODREJCP", "GODREJPROP",
        "HAVELLS", "HDFC", "IDFCFIRSTB", "IGL", "INDUSTOWER",
        "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD", "LICHSGFIN",
        "LUPIN", "MARICO", "MCDOWELL-N", "MFSL", "MOTHERSON",
        "MUTHOOTFIN", "NAUKRI", "NMDC", "OBEROIRLTY", "OFSS",
        "PAGEIND", "PEL", "PETRONET", "PFC", "PIDILITIND",
        "PIIND", "PNB", "POLYCAB", "RECLTD", "SAIL",
        "SHREECEM", "SIEMENS", "SRF", "SRTRANSFIN", "TATAELXSI",
        "TATAPOWER", "TORNTPHARM", "TRENT", "TVSMOTOR", "VEDL",
        "VOLTAS", "ZEEL"
    ]
    
    return nifty50 + additional


# Example usage
if __name__ == "__main__":
    # Test the function
    stocks = get_all_indian_stocks()
    print(f"Total stocks: {len(stocks)}")
    
    if stocks:
        print(f"Sample stock: {stocks[0]}")
    
    # Test NIFTY 50
    nifty50 = get_nifty50_stocks()
    print(f"NIFTY 50 count: {len(nifty50)}")
