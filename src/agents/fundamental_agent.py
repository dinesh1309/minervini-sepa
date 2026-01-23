"""
Fundamental Overlay Agent for SEPA Trading Workflow

This agent filters stocks based on fundamental criteria including
earnings growth, revenue growth, and margin expansion ("Code 33").
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import check_fundamentals
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    """
    Agent 2: Fundamental Overlay
    
    Filters for high-quality stocks with:
    - Strong EPS growth (≥20% YoY)
    - EPS acceleration (sequential improvement)
    - Revenue growth (≥15%)
    - Expanding margins
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("fundamental_criteria.yaml")
            config = config.get("fundamentals", {})
        
        super().__init__(
            name="FundamentalAgent",
            description="Filters stocks based on earnings and revenue growth",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="check_fundamentals",
                description="Check fundamental criteria for a stock",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["symbol"]
                },
                function=check_fundamentals
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a Fundamental Analysis Agent for the SEPA trading methodology.

Your task is to identify stocks with strong fundamental characteristics:
- Earnings Per Share (EPS) growth of at least 20% year-over-year
- EPS acceleration (each quarter better than the last)
- Revenue growth of at least 15%
- Expanding profit margins
- Positive earnings surprises

Return a JSON response with:
{
    "qualified_stocks": [...],
    "disqualified_stocks": [...],
    "summary": "Analysis summary"
}"""
    
    def analyze_stocks(self, stocks: List[Dict]) -> AgentResult:
        """
        Analyze stocks from trend template results for fundamentals.
        
        Args:
            stocks: List of stock dicts with 'symbol' key
        
        Returns:
            AgentResult with qualified and disqualified stocks
        """
        qualified = []
        disqualified = []
        
        for stock in stocks:
            symbol = stock.get("symbol") if isinstance(stock, dict) else stock
            
            try:
                results = check_fundamentals(symbol, config=self.config)
                
                if results.get("all_passed", False):
                    qualified.append({
                        "symbol": symbol,
                        "earnings": results.get("earnings_check", {}),
                        "revenue": results.get("revenue_check", {}),
                        "margins": results.get("margins_check", {})
                    })
                else:
                    disqualified.append({
                        "symbol": symbol,
                        "reason": "Failed fundamental criteria",
                        "details": results
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
                disqualified.append({
                    "symbol": symbol,
                    "reason": f"Error: {str(e)}"
                })
        
        return AgentResult(
            success=True,
            data={
                "qualified_stocks": qualified,
                "disqualified_stocks": disqualified,
                "total_analyzed": len(stocks),
                "qualified_count": len(qualified)
            },
            message=f"Analyzed {len(stocks)} stocks. {len(qualified)} passed fundamental criteria.",
            reasoning=f"{len(qualified)} stocks show strong earnings and revenue growth."
        )
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        """Run the fundamental agent on stocks passed from trend template."""
        if isinstance(input_data, dict):
            stocks = input_data.get("passed_stocks", [])
        elif isinstance(input_data, list):
            stocks = input_data
        else:
            stocks = []
        
        if not stocks:
            return AgentResult(
                success=False,
                data=None,
                message="No stocks provided"
            )
        
        logger.info(f"[{self.name}] Analyzing {len(stocks)} stocks")
        return self.analyze_stocks(stocks)


if __name__ == "__main__":
    agent = FundamentalAgent()
    test_stocks = [{"symbol": "TITAN"}, {"symbol": "RELIANCE"}]
    result = agent.run(test_stocks)
    print(f"Qualified: {len(result.data.get('qualified_stocks', []))}")
