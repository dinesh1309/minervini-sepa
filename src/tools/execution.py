"""
Execution Tools for SEPA Trading Workflow

This module provides functions for:
- Position sizing based on risk parameters
- Paper trading execution and logging
- Portfolio metrics calculation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAPER_TRADES_FILE = DATA_DIR / "paper_trades.json"


def _ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TradeOrder:
    """Represents a trade order."""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: Optional[float] = None
    timestamp: str = ""
    status: str = "PENDING"  # PENDING, EXECUTED, CANCELLED
    notes: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    stop_loss: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    portfolio_value: float,
    max_risk_pct: float = 2.0,
    max_position_pct: float = 25.0
) -> Dict[str, Any]:
    """
    Calculate position size based on risk parameters.
    
    Uses the formula: Shares = (Portfolio * Risk%) / (Entry - Stop)
    
    Args:
        entry_price: Planned entry price
        stop_loss: Stop loss price
        portfolio_value: Total portfolio value
        max_risk_pct: Maximum risk per trade as % of portfolio
        max_position_pct: Maximum position size as % of portfolio
    
    Returns:
        Dictionary with position sizing details
    """
    if entry_price <= 0 or stop_loss <= 0 or portfolio_value <= 0:
        return {
            'error': 'Invalid input values',
            'shares': 0,
            'position_value': 0
        }
    
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share <= 0:
        return {
            'error': 'Stop loss must be different from entry',
            'shares': 0,
            'position_value': 0
        }
    
    # Calculate dollar risk
    dollar_risk = portfolio_value * (max_risk_pct / 100)
    
    # Calculate shares based on risk
    shares_from_risk = int(dollar_risk / risk_per_share)
    
    # Calculate max shares based on position size limit
    max_position_value = portfolio_value * (max_position_pct / 100)
    shares_from_position = int(max_position_value / entry_price)
    
    # Use the smaller of the two
    shares = min(shares_from_risk, shares_from_position)
    
    # Calculate actual values
    position_value = shares * entry_price
    actual_risk = shares * risk_per_share
    actual_risk_pct = (actual_risk / portfolio_value) * 100
    position_pct = (position_value / portfolio_value) * 100
    
    # Calculate stop loss percentage
    stop_loss_pct = (risk_per_share / entry_price) * 100
    
    return {
        'shares': shares,
        'position_value': round(position_value, 2),
        'position_pct': round(position_pct, 2),
        'risk_amount': round(actual_risk, 2),
        'risk_pct': round(actual_risk_pct, 2),
        'stop_loss_pct': round(stop_loss_pct, 2),
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'risk_reward_1_target': round(entry_price + risk_per_share, 2),
        'risk_reward_2_target': round(entry_price + (2 * risk_per_share), 2),
        'risk_reward_3_target': round(entry_price + (3 * risk_per_share), 2)
    }


def execute_paper_trade(order: TradeOrder) -> Dict[str, Any]:
    """
    Log a paper trade to the trading journal.
    
    Args:
        order: TradeOrder object with trade details
    
    Returns:
        Execution result with trade ID
    """
    _ensure_data_dir()
    
    # Load existing trades
    trades = []
    if PAPER_TRADES_FILE.exists():
        try:
            with open(PAPER_TRADES_FILE, 'r') as f:
                trades = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading trades file: {e}")
            trades = []
    
    # Create trade record
    trade_id = len(trades) + 1
    trade_record = {
        'id': trade_id,
        **asdict(order),
        'status': 'EXECUTED',
        'executed_at': datetime.now().isoformat()
    }
    
    trades.append(trade_record)
    
    # Save trades
    try:
        with open(PAPER_TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        logger.info(f"Paper trade executed: {order.action} {order.quantity} {order.symbol} @ {order.entry_price}")
    except Exception as e:
        logger.error(f"Error saving trade: {e}")
        return {'success': False, 'error': str(e)}
    
    return {
        'success': True,
        'trade_id': trade_id,
        'trade': trade_record
    }


def get_open_positions() -> List[Dict]:
    """
    Get list of open positions from paper trading journal.
    
    Returns:
        List of open position dictionaries
    """
    if not PAPER_TRADES_FILE.exists():
        return []
    
    try:
        with open(PAPER_TRADES_FILE, 'r') as f:
            trades = json.load(f)
    except Exception:
        return []
    
    # Calculate net positions per symbol
    positions = {}
    
    for trade in trades:
        symbol = trade['symbol']
        quantity = trade['quantity']
        price = trade['entry_price']
        action = trade['action']
        
        if symbol not in positions:
            positions[symbol] = {
                'symbol': symbol,
                'net_quantity': 0,
                'avg_entry': 0,
                'total_cost': 0,
                'stop_loss': trade.get('stop_loss', 0),
                'entry_date': trade.get('timestamp', '')
            }
        
        if action == 'BUY':
            # Update average entry
            new_cost = positions[symbol]['total_cost'] + (quantity * price)
            new_qty = positions[symbol]['net_quantity'] + quantity
            positions[symbol]['avg_entry'] = new_cost / new_qty if new_qty > 0 else 0
            positions[symbol]['total_cost'] = new_cost
            positions[symbol]['net_quantity'] = new_qty
            positions[symbol]['stop_loss'] = trade.get('stop_loss', positions[symbol]['stop_loss'])
        elif action == 'SELL':
            positions[symbol]['net_quantity'] -= quantity
            positions[symbol]['total_cost'] -= quantity * positions[symbol]['avg_entry']
    
    # Filter to only open positions (net_quantity > 0)
    open_positions = [p for p in positions.values() if p['net_quantity'] > 0]
    
    return open_positions


def get_portfolio_metrics() -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics.
    
    Returns:
        Dictionary with win rate, average gain/loss, etc.
    """
    if not PAPER_TRADES_FILE.exists():
        return {
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0.0,
            'avg_gain_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0
        }
    
    try:
        with open(PAPER_TRADES_FILE, 'r') as f:
            trades = json.load(f)
    except Exception:
        return {'error': 'Could not read trades'}
    
    # Find completed round-trip trades
    completed_trades = []
    
    # Group by symbol and find buy-sell pairs
    symbols = set(t['symbol'] for t in trades)
    
    for symbol in symbols:
        symbol_trades = [t for t in trades if t['symbol'] == symbol]
        buys = [t for t in symbol_trades if t['action'] == 'BUY']
        sells = [t for t in symbol_trades if t['action'] == 'SELL']
        
        # Match buys with sells (FIFO)
        for buy, sell in zip(buys, sells):
            pnl_pct = ((sell['entry_price'] - buy['entry_price']) / buy['entry_price']) * 100
            completed_trades.append({
                'symbol': symbol,
                'buy_price': buy['entry_price'],
                'sell_price': sell['entry_price'],
                'pnl_pct': pnl_pct,
                'is_winner': pnl_pct > 0
            })
    
    if not completed_trades:
        return {
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0.0,
            'avg_gain_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0
        }
    
    winners = [t for t in completed_trades if t['is_winner']]
    losers = [t for t in completed_trades if not t['is_winner']]
    
    total = len(completed_trades)
    win_count = len(winners)
    loss_count = len(losers)
    
    avg_gain = sum(t['pnl_pct'] for t in winners) / win_count if winners else 0
    avg_loss = sum(t['pnl_pct'] for t in losers) / loss_count if losers else 0
    
    total_gains = sum(t['pnl_pct'] for t in winners)
    total_losses = abs(sum(t['pnl_pct'] for t in losers))
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
    
    return {
        'total_trades': total,
        'winners': win_count,
        'losers': loss_count,
        'win_rate': round((win_count / total) * 100, 1) if total > 0 else 0.0,
        'avg_gain_pct': round(avg_gain, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'expectancy': round(
            (avg_gain * (win_count/total)) + (avg_loss * (loss_count/total)), 2
        ) if total > 0 else 0
    }


def get_trade_history(symbol: Optional[str] = None) -> List[Dict]:
    """
    Get trade history, optionally filtered by symbol.
    
    Args:
        symbol: Optional symbol to filter by
    
    Returns:
        List of trade records
    """
    if not PAPER_TRADES_FILE.exists():
        return []
    
    try:
        with open(PAPER_TRADES_FILE, 'r') as f:
            trades = json.load(f)
    except Exception:
        return []
    
    if symbol:
        trades = [t for t in trades if t['symbol'] == symbol]
    
    return trades


# Example usage
if __name__ == "__main__":
    # Test position sizing
    sizing = calculate_position_size(
        entry_price=1500,
        stop_loss=1425,  # 5% stop
        portfolio_value=1000000,
        max_risk_pct=2.0
    )
    print("Position Sizing:")
    print(f"  Shares: {sizing['shares']}")
    print(f"  Position Value: ₹{sizing['position_value']:,.2f}")
    print(f"  Risk Amount: ₹{sizing['risk_amount']:,.2f}")
    print(f"  Risk %: {sizing['risk_pct']}%")
    
    # Test paper trade
    order = TradeOrder(
        symbol="TITAN.NS",
        action="BUY",
        quantity=sizing['shares'],
        entry_price=1500,
        stop_loss=1425
    )
    
    result = execute_paper_trade(order)
    print(f"\nPaper Trade Result: {result}")
    
    # Test metrics
    metrics = get_portfolio_metrics()
    print(f"\nPortfolio Metrics: {metrics}")
