"""
Portfolio Review Agent for SEPA Trading Workflow

This agent provides feedback on overall portfolio performance
and adjusts position sizing recommendations.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import get_portfolio_metrics, get_trade_history
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioReviewAgent(BaseAgent):
    """
    Agent 7: Portfolio Review
    
    Provides feedback loop:
    - Tracks batting average (win rate)
    - Tightens stops below 40% win rate
    - Allows pyramiding only on winners
    - Enforces max 25% position size
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("portfolio_criteria.yaml")
            config = config.get("portfolio_review", {})
        
        super().__init__(
            name="PortfolioReviewAgent",
            description="Reviews portfolio performance and adjusts strategies",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="get_portfolio_metrics",
                description="Get portfolio performance metrics",
                parameters={"type": "object", "properties": {}},
                function=get_portfolio_metrics
            ),
            ToolDefinition(
                name="get_trade_history",
                description="Get trade history",
                parameters={"type": "object", "properties": {}},
                function=get_trade_history
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a Portfolio Review Agent for SEPA trading.

Your role is to:
- Analyze overall trading performance
- Adjust risk parameters based on recent results
- Provide recommendations for position sizing
- Identify patterns in winning/losing trades

Return performance analysis and recommendations."""
    
    def review_performance(self) -> AgentResult:
        """Review portfolio performance and generate recommendations."""
        metrics = get_portfolio_metrics()
        history = get_trade_history()
        
        min_win_rate = self.config.get("min_batting_average", 0.50)
        tighten_threshold = self.config.get("tighten_stops_below_win_rate", 0.40)
        max_position_pct = self.config.get("max_position_size_pct", 25)
        
        recommendations = []
        risk_adjustment = 1.0  # Multiplier for position size
        
        win_rate = metrics.get("win_rate", 0) / 100  # Convert to decimal
        
        # Check if trading poorly
        if win_rate < tighten_threshold:
            risk_adjustment = 0.5  # Cut position size in half
            recommendations.append({
                "type": "REDUCE_RISK",
                "reason": f"Win rate ({win_rate*100:.1f}%) below {tighten_threshold*100}%",
                "action": "Reduce position sizes by 50%"
            })
        elif win_rate < min_win_rate:
            risk_adjustment = 0.75
            recommendations.append({
                "type": "CAUTION",
                "reason": f"Win rate ({win_rate*100:.1f}%) below target {min_win_rate*100}%",
                "action": "Reduce position sizes by 25%"
            })
        elif win_rate >= 0.6:
            risk_adjustment = 1.25
            recommendations.append({
                "type": "INCREASE_RISK",
                "reason": f"Win rate ({win_rate*100:.1f}%) is strong",
                "action": "Can increase position sizes by 25%"
            })
        
        # Analyze profit factor
        profit_factor = metrics.get("profit_factor", 0)
        if profit_factor < 1.5 and profit_factor > 0:
            recommendations.append({
                "type": "IMPROVE_RR",
                "reason": f"Profit factor ({profit_factor:.2f}) is low",
                "action": "Focus on letting winners run longer"
            })
        
        # Check expectancy
        expectancy = metrics.get("expectancy", 0)
        if expectancy < 0:
            recommendations.append({
                "type": "WARNING",
                "reason": "Negative expectancy",
                "action": "Review and refine entry/exit criteria"
            })
        
        return AgentResult(
            success=True,
            data={
                "metrics": metrics,
                "recommendations": recommendations,
                "risk_adjustment": risk_adjustment,
                "trade_count": len(history),
                "status": "HEALTHY" if win_rate >= min_win_rate else "NEEDS_ATTENTION"
            },
            message=f"Portfolio review complete. Win rate: {win_rate*100:.1f}%",
            reasoning=f"Generated {len(recommendations)} recommendations based on performance analysis."
        )
    
    def run(self, input_data: Any = None, max_iterations: int = 5) -> AgentResult:
        logger.info(f"[{self.name}] Reviewing portfolio performance")
        return self.review_performance()


if __name__ == "__main__":
    agent = PortfolioReviewAgent()
    result = agent.run()
    print(f"Metrics: {result.data.get('metrics')}")
    print(f"Recommendations: {result.data.get('recommendations')}")
