"""
Position Management Agent for SEPA Trading Workflow

This agent manages open positions including trailing stops,
partial profit taking, and exit signals.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import get_stock_data, get_current_price, get_open_positions, calculate_moving_averages
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionAgent(BaseAgent):
    """
    Agent 6: Position Management
    
    Manages open positions:
    - Move stop to breakeven at 2x risk (free roll)
    - Partial profit taking at 20% gain
    - Trailing stop using 20-day MA
    - Exit on heavy volume breakdown
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("position_criteria.yaml")
            config = config.get("position_management", {})
        
        super().__init__(
            name="PositionAgent",
            description="Manages open positions and generates exit signals",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="get_current_price",
                description="Get current price for a stock",
                parameters={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"]
                },
                function=get_current_price
            ),
            ToolDefinition(
                name="get_open_positions",
                description="Get list of open positions",
                parameters={"type": "object", "properties": {}},
                function=get_open_positions
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a Position Management Agent for SEPA trading.

Your responsibilities:
- Monitor open positions
- Move stops to breakeven when profit exceeds 2x initial risk
- Take partial profits at 20% gain
- Trail stops using 20-day moving average
- Generate sell signals on breakdown

Return position updates and any sell signals."""
    
    def review_positions(self, positions: List[Dict] = None) -> AgentResult:
        """Review open positions and generate updates."""
        if positions is None:
            positions = get_open_positions()
        
        updates = []
        sell_signals = []
        
        free_roll_trigger = self.config.get("free_roll_trigger_multiple", 2)
        partial_profit_pct = self.config.get("partial_profit_trigger_pct", 20)
        trailing_ma_days = self.config.get("trailing_stop_ma_days", 20)
        
        for pos in positions:
            symbol = pos.get("symbol")
            entry_price = pos.get("avg_entry", 0)
            stop_loss = pos.get("stop_loss", 0)
            quantity = pos.get("net_quantity", 0)
            
            if entry_price <= 0 or quantity <= 0:
                continue
            
            try:
                current_price = get_current_price(symbol)
                
                if current_price is None:
                    continue
                
                # Calculate P&L
                pnl = (current_price - entry_price) / entry_price * 100
                initial_risk = (entry_price - stop_loss) / entry_price * 100 if stop_loss > 0 else 7
                risk_multiple = pnl / initial_risk if initial_risk > 0 else 0
                
                update = {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl_pct": round(pnl, 2),
                    "risk_multiple": round(risk_multiple, 2),
                    "current_stop": stop_loss,
                    "actions": []
                }
                
                # Check for free roll (move stop to breakeven)
                if risk_multiple >= free_roll_trigger and stop_loss < entry_price:
                    update["new_stop"] = entry_price
                    update["actions"].append(f"Move stop to breakeven ({entry_price:.2f})")
                
                # Check for partial profit
                if pnl >= partial_profit_pct:
                    update["actions"].append(f"Consider taking partial profits ({pnl:.1f}% gain)")
                
                # Check 20-day MA for trailing stop
                df = get_stock_data(symbol, period="3mo")
                if df is not None and len(df) >= trailing_ma_days:
                    df = calculate_moving_averages(df, periods=[trailing_ma_days])
                    ma_value = df[f"SMA_{trailing_ma_days}"].iloc[-1]
                    
                    # If price closes below 20-day MA on heavy volume, exit
                    if current_price < ma_value:
                        update["actions"].append(f"Price below {trailing_ma_days}-day MA - watch for exit")
                        
                        # Check volume
                        avg_vol = df["Volume"].tail(50).mean()
                        recent_vol = df["Volume"].iloc[-1]
                        if recent_vol > avg_vol * 1.5:
                            sell_signals.append({
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": f"Broke {trailing_ma_days}MA on heavy volume",
                                "current_price": current_price,
                                "quantity": quantity
                            })
                
                # Check if stop hit
                if current_price <= stop_loss:
                    sell_signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Stop loss triggered",
                        "current_price": current_price,
                        "quantity": quantity
                    })
                
                updates.append(update)
                
            except Exception as e:
                logger.error(f"Error reviewing position {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data={
                "position_updates": updates,
                "sell_signals": sell_signals,
                "total_positions": len(positions)
            },
            message=f"Reviewed {len(positions)} positions. {len(sell_signals)} sell signals generated.",
            reasoning=f"Position review complete. {len([u for u in updates if u.get('actions')])} positions need attention."
        )
    
    def run(self, input_data: Any = None, max_iterations: int = 5) -> AgentResult:
        positions = None
        if isinstance(input_data, dict):
            positions = input_data.get("positions")
        elif isinstance(input_data, list):
            positions = input_data
        
        logger.info(f"[{self.name}] Reviewing positions")
        return self.review_positions(positions)
