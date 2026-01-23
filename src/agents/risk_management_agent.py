"""
Risk Management Agent for SEPA Trading Workflow

This agent calculates position sizes and stop losses
to ensure proper risk management.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import calculate_position_size, TradeOrder
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManagementAgent(BaseAgent):
    """
    Agent 5: Risk Management
    
    Calculates:
    - Stop loss placement (6-7% target, 10% max)
    - Position sizing based on risk
    - Portfolio risk per trade (max 2%)
    - Enforces no averaging down rule
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None, portfolio_value: float = 1000000):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("risk_criteria.yaml")
            config = config.get("risk_management", {})
        
        self.portfolio_value = portfolio_value
        
        super().__init__(
            name="RiskManagementAgent",
            description="Calculates position sizes and manages risk",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="calculate_position_size",
                description="Calculate position size based on risk",
                parameters={
                    "type": "object",
                    "properties": {
                        "entry_price": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "portfolio_value": {"type": "number"},
                        "max_risk_pct": {"type": "number"}
                    },
                    "required": ["entry_price", "stop_loss", "portfolio_value"]
                },
                function=calculate_position_size
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a Risk Management Agent for SEPA trading.

Your rules:
- Maximum stop loss: 10%
- Target stop loss: 6-7%
- Maximum risk per trade: 2% of portfolio
- Never average down on losing positions

Calculate proper position sizes and stop losses for each trade."""
    
    def process_signals(self, signals: List[Dict]) -> AgentResult:
        """Process buy signals and add risk parameters."""
        trade_orders = []
        rejected = []
        
        max_stop_pct = self.config.get("max_stop_loss_pct", 10)
        target_stop_pct = self.config.get("target_stop_loss_pct", 7)
        max_risk_pct = self.config.get("max_portfolio_risk_per_trade", 2.0)
        
        for signal in signals:
            symbol = signal.get("symbol")
            entry_price = signal.get("current_price") or signal.get("pivot_price", 0)
            
            if entry_price <= 0:
                rejected.append({"symbol": symbol, "reason": "Invalid entry price"})
                continue
            
            # Calculate stop loss (use target %, but accept up to max %)
            stop_loss = entry_price * (1 - target_stop_pct / 100)
            actual_stop_pct = target_stop_pct
            
            # Check if VCP provides a better stop level
            vcp_low = signal.get("vcp_low")
            if vcp_low and vcp_low > 0:
                potential_stop = vcp_low * 0.99  # Just below VCP low
                potential_stop_pct = ((entry_price - potential_stop) / entry_price) * 100
                
                if potential_stop_pct <= max_stop_pct:
                    stop_loss = potential_stop
                    actual_stop_pct = potential_stop_pct
            
            # Calculate position size
            sizing = calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                portfolio_value=self.portfolio_value,
                max_risk_pct=max_risk_pct
            )
            
            if sizing.get("error"):
                rejected.append({"symbol": symbol, "reason": sizing["error"]})
                continue
            
            # Create trade order
            order = {
                "symbol": symbol,
                "action": "BUY",
                "entry_price": entry_price,
                "stop_loss": round(stop_loss, 2),
                "stop_loss_pct": round(actual_stop_pct, 2),
                "quantity": sizing["shares"],
                "position_value": sizing["position_value"],
                "risk_amount": sizing["risk_amount"],
                "risk_pct": sizing["risk_pct"],
                "target_1r": sizing["risk_reward_1_target"],
                "target_2r": sizing["risk_reward_2_target"],
                "target_3r": sizing["risk_reward_3_target"],
                "status": "PENDING_APPROVAL"
            }
            
            trade_orders.append(order)
        
        return AgentResult(
            success=True,
            data={
                "trade_orders": trade_orders,
                "rejected": rejected,
                "total_risk": sum(o["risk_amount"] for o in trade_orders),
                "total_exposure": sum(o["position_value"] for o in trade_orders)
            },
            message=f"Generated {len(trade_orders)} trade orders with risk parameters.",
            reasoning=f"Total portfolio risk: ${sum(o['risk_amount'] for o in trade_orders):,.0f}"
        )
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        if isinstance(input_data, dict):
            signals = input_data.get("buy_signals", [])
        elif isinstance(input_data, list):
            signals = input_data
        else:
            signals = []
        
        # Filter to only actionable signals
        actionable = [s for s in signals if s.get("action") == "BUY"]
        
        if not actionable:
            return AgentResult(
                success=True,
                data={"trade_orders": [], "rejected": []},
                message="No actionable signals to process"
            )
        
        logger.info(f"[{self.name}] Processing {len(actionable)} buy signals")
        return self.process_signals(actionable)
