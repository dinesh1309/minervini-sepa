"""
Trend Template Agent for SEPA Trading Workflow

This agent filters stocks based on Minervini's 8-point Trend Template
to identify stocks in confirmed Stage 2 uptrends.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult, get_ai_client
from ..tools import (
    get_stock_data,
    check_trend_template,
    get_nifty50_stocks,
    get_batch_stock_data
)
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendTemplateAgent(BaseAgent):
    """
    Agent 1: Trend Template Filter
    
    Filters stocks based on the 8-point Trend Template:
    1. Price above 150-day SMA
    2. Price above 200-day SMA
    3. 150-day SMA above 200-day SMA
    4. 200-day SMA trending up (1+ months)
    5. 50-day SMA above 150 and 200
    6. Price above 50-day SMA
    7. Price 30%+ above 52-week low
    8. Price within 25% of 52-week high
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        # Load config from file if not provided
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("trend_template_criteria.yaml")
            config = config.get("trend_template", {})
        
        super().__init__(
            name="TrendTemplateAgent",
            description="Filters stocks based on Minervini's 8-point Trend Template",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        """Set up tools for trend template analysis."""
        self.tools = [
            ToolDefinition(
                name="get_stock_data",
                description="Fetch OHLCV price data for a stock symbol",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')"
                        },
                        "period": {
                            "type": "string",
                            "description": "Data period: 1y, 2y, etc.",
                            "default": "2y"
                        }
                    },
                    "required": ["symbol"]
                },
                function=get_stock_data
            ),
            ToolDefinition(
                name="check_trend_template",
                description="Check if a stock passes all 8 points of the Trend Template",
                parameters={
                    "type": "object",
                    "properties": {
                        "df": {
                            "type": "object",
                            "description": "DataFrame with OHLCV data"
                        },
                        "config": {
                            "type": "object",
                            "description": "Optional criteria configuration"
                        }
                    },
                    "required": ["df"]
                },
                function=check_trend_template
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a Trend Template Analysis Agent specializing in Mark Minervini's SEPA methodology.

Your task is to analyze stocks against the 8-point Trend Template criteria:
1. Price above 150-day SMA
2. Price above 200-day SMA
3. 150-day SMA above 200-day SMA
4. 200-day SMA trending up for at least 1 month
5. 50-day SMA above both 150 and 200-day SMAs
6. Current price above 50-day SMA
7. Price at least 30% above 52-week low
8. Price within 25% of 52-week high

For each stock, use the available tools to:
1. Get the stock's price data
2. Run the trend template check

Return a structured JSON response with:
{
    "passed_stocks": ["SYMBOL1", "SYMBOL2", ...],
    "failed_stocks": [{"symbol": "SYMBOL", "failed_criteria": [...]}],
    "summary": "Brief summary of results"
}"""
    
    def analyze_stocks(self, symbols: List[str]) -> AgentResult:
        """
        Analyze a list of stocks against the Trend Template.
        
        Args:
            symbols: List of stock symbols to analyze
        
        Returns:
            AgentResult with passed and failed stocks
        """
        passed = []
        failed = []
        
        for symbol in symbols:
            try:
                # Get stock data
                df = get_stock_data(symbol, period="2y")
                
                if df is None or len(df) < 200:
                    failed.append({
                        "symbol": symbol,
                        "failed_criteria": ["insufficient_data"]
                    })
                    continue
                
                # Check trend template
                results = check_trend_template(df, config=self.config)
                
                if results.get("all_passed", False):
                    passed.append({
                        "symbol": symbol,
                        "passed_count": results.get("passed_count", 0),
                        "checks": results.get("checks", {})
                    })
                else:
                    failed_checks = [
                        k for k, v in results.get("checks", {}).items()
                        if not v.get("passed", False)
                    ]
                    failed.append({
                        "symbol": symbol,
                        "failed_criteria": failed_checks,
                        "passed_count": results.get("passed_count", 0)
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                failed.append({
                    "symbol": symbol,
                    "failed_criteria": [f"error: {str(e)}"]
                })
        
        return AgentResult(
            success=True,
            data={
                "passed_stocks": passed,
                "failed_stocks": failed,
                "total_analyzed": len(symbols),
                "passed_count": len(passed),
                "pass_rate": f"{(len(passed)/len(symbols)*100):.1f}%" if symbols else "0%"
            },
            message=f"Analyzed {len(symbols)} stocks. {len(passed)} passed trend template.",
            reasoning=f"Out of {len(symbols)} stocks, {len(passed)} meet all Stage 2 uptrend criteria."
        )
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        """
        Run the trend template agent.
        
        Args:
            input_data: List of stock symbols or dict with 'symbols' key
        
        Returns:
            AgentResult with analysis results
        """
        # Extract symbols from input
        if isinstance(input_data, list):
            symbols = input_data
        elif isinstance(input_data, dict):
            symbols = input_data.get("symbols", [])
        else:
            symbols = []
        
        if not symbols:
            return AgentResult(
                success=False,
                data=None,
                message="No symbols provided for analysis"
            )
        
        logger.info(f"[{self.name}] Analyzing {len(symbols)} stocks")
        return self.analyze_stocks(symbols)


# Example usage
if __name__ == "__main__":
    # Test with NIFTY 50 stocks
    agent = TrendTemplateAgent()
    
    # Test with a few stocks
    test_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "TITAN"]
    
    result = agent.run(test_symbols)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Passed: {len(result.data.get('passed_stocks', []))}")
    print(f"Failed: {len(result.data.get('failed_stocks', []))}")
