"""
Entry Point Agent for SEPA Trading Workflow

This agent identifies optimal entry points (pivot breakouts)
for stocks with valid VCP patterns.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import get_stock_data, identify_pivot, calculate_volume_metrics
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntryPointAgent(BaseAgent):
    """
    Agent 4: Entry Point Detection
    
    Identifies optimal pivot/breakout points:
    - Tight area breakouts
    - Volume expansion on breakout (≥1.5x)
    - Proximity to 52-week high
    - Prioritizes first-base patterns
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("entry_criteria.yaml")
            config = config.get("entry_point", {})
        
        super().__init__(
            name="EntryPointAgent",
            description="Identifies optimal pivot breakout points",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="identify_pivot",
                description="Identify breakout pivot price",
                parameters={
                    "type": "object",
                    "properties": {
                        "df": {"type": "object"},
                        "vcp": {"type": "object", "description": "VCP pattern data"}
                    },
                    "required": ["df"]
                },
                function=identify_pivot
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are an Entry Point Agent for SEPA trading.

Your task is to identify optimal entry points:
- Find the pivot price (resistance at tightest consolidation)
- Check for volume expansion on breakout
- Verify proximity to 52-week high
- Generate actionable buy signals

Return:
{
    "buy_signals": [...],
    "not_ready": [...],
    "summary": "..."
}"""
    
    def analyze_stocks(self, stocks: List[Dict]) -> AgentResult:
        """Analyze VCP stocks for entry points."""
        buy_signals = []
        not_ready = []
        
        volume_multiplier = self.config.get("volume_expansion_multiplier", 1.5)
        max_from_high = self.config.get("proximity_to_52w_high_pct", 15)
        
        for stock in stocks:
            symbol = stock.get("symbol") if isinstance(stock, dict) else stock
            pivot_price = stock.get("pivot_price", 0) if isinstance(stock, dict) else 0
            
            try:
                df = get_stock_data(symbol, period="3mo")
                
                if df is None or df.empty:
                    not_ready.append({"symbol": symbol, "reason": "No data"})
                    continue
                
                # Get pivot info
                pivot_info = identify_pivot(df)
                
                if pivot_price == 0:
                    pivot_price = pivot_info.get("pivot_price", 0)
                
                current_price = pivot_info.get("current_price", 0)
                distance_to_pivot = pivot_info.get("distance_to_pivot_pct", 100)
                
                # Check volume
                vol_metrics = calculate_volume_metrics(df)
                volume_ratio = vol_metrics.get("volume_ratio_50d", 0)
                
                # Determine if ready for entry
                is_near_pivot = distance_to_pivot <= 5  # Within 5% of pivot
                has_volume = volume_ratio >= volume_multiplier
                
                if is_near_pivot:
                    buy_signals.append({
                        "symbol": symbol,
                        "pivot_price": pivot_price,
                        "current_price": current_price,
                        "distance_to_pivot_pct": distance_to_pivot,
                        "volume_ratio": volume_ratio,
                        "volume_confirmation": has_volume,
                        "action": "BUY" if has_volume else "WATCH",
                        "notes": "Near pivot" + (" with volume" if has_volume else " - wait for volume")
                    })
                else:
                    not_ready.append({
                        "symbol": symbol,
                        "pivot_price": pivot_price,
                        "current_price": current_price,
                        "distance_to_pivot_pct": distance_to_pivot,
                        "reason": f"Price {distance_to_pivot:.1f}% from pivot - not ready"
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing entry for {symbol}: {e}")
                not_ready.append({"symbol": symbol, "reason": str(e)})
        
        # Sort by closest to pivot
        buy_signals.sort(key=lambda x: x.get("distance_to_pivot_pct", 100))
        
        return AgentResult(
            success=True,
            data={
                "buy_signals": buy_signals,
                "not_ready": not_ready,
                "actionable_count": len([s for s in buy_signals if s.get("action") == "BUY"])
            },
            message=f"Generated {len(buy_signals)} potential entry signals.",
            reasoning=f"{len([s for s in buy_signals if s.get('action')=='BUY'])} stocks ready for immediate entry."
        )
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        if isinstance(input_data, dict):
            stocks = input_data.get("vcp_stocks", [])
        elif isinstance(input_data, list):
            stocks = input_data
        else:
            stocks = []
        
        if not stocks:
            return AgentResult(success=False, data=None, message="No stocks provided")
        
        logger.info(f"[{self.name}] Analyzing {len(stocks)} stocks for entry points")
        return self.analyze_stocks(stocks)
