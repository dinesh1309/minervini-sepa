"""
Market Data Tools for SEPA Trading Workflow

This module provides functions to fetch OHLCV price data for Indian stocks
using yfinance with proper suffix handling and rate limiting.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from functools import wraps

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
MIN_REQUEST_INTERVAL = 0.5  # seconds between requests
_last_request_time = 0


def _rate_limit():
    """Ensure minimum interval between API requests."""
    global _last_request_time
    
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    
    _last_request_time = time.time()


def _add_exchange_suffix(symbol: str, exchange: str = "NSE") -> str:
    """
    Add appropriate exchange suffix for yfinance.
    
    Args:
        symbol: Stock symbol without suffix
        exchange: Exchange name (NSE or BSE)
    
    Returns:
        Symbol with appropriate suffix (.NS or .BO)
    """
    # Remove any existing suffix
    symbol = symbol.replace('.NS', '').replace('.BO', '').strip()
    
    if exchange.upper() == "BSE":
        return f"{symbol}.BO"
    return f"{symbol}.NS"  # Default to NSE


def get_stock_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    exchange: str = "NSE",
    auto_suffix: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV historical data for a stock.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        exchange: Exchange (NSE or BSE) - used if auto_suffix is True
        auto_suffix: Whether to automatically add .NS/.BO suffix
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails.
        Columns: Open, High, Low, Close, Volume, Adj Close
    
    Example:
        >>> df = get_stock_data("RELIANCE", period="1y")
        >>> print(df.tail())
    """
    _rate_limit()
    
    # Add suffix if needed
    if auto_suffix and not (symbol.endswith('.NS') or symbol.endswith('.BO')):
        symbol = _add_exchange_suffix(symbol, exchange)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Validate and clean data
        df = _validate_and_clean_data(df, symbol)
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def get_stock_data_range(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    exchange: str = "NSE"
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a specific date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
        exchange: Exchange (NSE or BSE)
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails.
    """
    _rate_limit()
    
    if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
        symbol = _add_exchange_suffix(symbol, exchange)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
            return None
        
        df = _validate_and_clean_data(df, symbol)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def _validate_and_clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and clean OHLCV data.
    
    Args:
        df: Raw DataFrame from yfinance
        symbol: Stock symbol (for logging)
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    original_len = len(df)
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Remove rows where Close is NaN (essential column)
    df = df.dropna(subset=['Close'])
    
    # Remove rows with zero or negative prices
    df = df[df['Close'] > 0]
    
    # Remove rows with zero volume (except for legitimate cases)
    # Some days might have zero volume due to holidays
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    if len(df) < original_len:
        logger.debug(f"Cleaned {original_len - len(df)} invalid rows for {symbol}")
    
    return df


def get_current_price(symbol: str, exchange: str = "NSE") -> Optional[float]:
    """
    Get the current/latest price for a stock.
    
    Args:
        symbol: Stock symbol
        exchange: Exchange (NSE or BSE)
    
    Returns:
        Current price or None if unavailable
    """
    df = get_stock_data(symbol, period="5d", exchange=exchange)
    
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    
    return None


def get_batch_stock_data(
    symbols: list,
    period: str = "1y",
    exchange: str = "NSE"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stocks efficiently.
    
    Args:
        symbols: List of stock symbols
        period: Data period
        exchange: Exchange for all symbols
    
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    results = {}
    
    # Add suffixes
    formatted_symbols = []
    for symbol in symbols:
        if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
            formatted_symbols.append(_add_exchange_suffix(symbol, exchange))
        else:
            formatted_symbols.append(symbol)
    
    try:
        # yfinance supports batch downloads
        _rate_limit()
        data = yf.download(
            formatted_symbols,
            period=period,
            group_by='ticker',
            threads=True,
            progress=False
        )
        
        for symbol, formatted in zip(symbols, formatted_symbols):
            try:
                if len(formatted_symbols) == 1:
                    df = data
                else:
                    df = data[formatted].copy()
                
                if not df.empty:
                    df = _validate_and_clean_data(df, symbol)
                    results[symbol] = df
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Batch download failed: {e}")
        # Fallback to individual downloads
        for symbol in symbols:
            df = get_stock_data(symbol, period=period, exchange=exchange)
            if df is not None:
                results[symbol] = df
    
    logger.info(f"Fetched data for {len(results)}/{len(symbols)} stocks")
    return results


# Example usage
if __name__ == "__main__":
    # Test single stock
    df = get_stock_data("RELIANCE", period="6mo")
    if df is not None:
        print(f"RELIANCE data shape: {df.shape}")
        print(df.tail(3))
    
    # Test current price
    price = get_current_price("TCS")
    print(f"TCS current price: {price}")
    
    # Test batch download
    stocks = ["INFY", "HDFCBANK", "ICICIBANK"]
    batch_data = get_batch_stock_data(stocks, period="1mo")
    print(f"Batch download got {len(batch_data)} stocks")
