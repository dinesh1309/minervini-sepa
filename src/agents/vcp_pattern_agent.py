"""
VCP Pattern Agent for SEPA Trading Workflow

This agent detects Volatility Contraction Patterns (VCP) in stocks
that have passed the trend template and fundamental filters.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_agent import BaseAgent, ToolDefinition, AgentResult
from ..tools import get_stock_data, detect_vcp
from ..utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VCPPatternAgent(BaseAgent):
    """
    Agent 3: VCP Pattern Detection
    
    Identifies Volatility Contraction Patterns:
    - Multiple contractions (T1 > T2 > T3)
    - Each pullback smaller than previous
    - Volume dries up during consolidation
    - Base length 3-6 weeks typically
    """
    
    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_config("vcp_criteria.yaml")
            config = config.get("vcp_pattern", {})
        
        super().__init__(
            name="VCPPatternAgent",
            description="Detects Volatility Contraction Patterns",
            ai_client=ai_client,
            config=config
        )
    
    def _setup_tools(self):
        self.tools = [
            ToolDefinition(
                name="detect_vcp",
                description="Detect VCP pattern in price data",
                parameters={
                    "type": "object",
                    "properties": {
                        "df": {"type": "object", "description": "OHLCV DataFrame"},
                        "symbol": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["df"]
                },
                function=detect_vcp
            )
        ]
    
    def _get_system_prompt(self) -> str:
        return """You are a VCP Pattern Detection Agent for the SEPA methodology.

Your task is to identify Volatility Contraction Patterns (VCP):
- Look for progressively tighter price contractions
- Each pullback should be smaller than the previous
- Volume should dry up during consolidation
- The pattern indicates supply absorption

Return a JSON response with:
{
    "vcp_stocks": [...],
    "no_pattern_stocks": [...],
    "summary": "Pattern detection summary"
}"""
    
    def analyze_stocks(self, stocks: List[Dict]) -> AgentResult:
        """Analyze stocks for VCP patterns."""
        vcp_stocks = []
        no_pattern = []
        
        for stock in stocks:
            symbol = stock.get("symbol") if isinstance(stock, dict) else stock
            
            try:
                df = get_stock_data(symbol, period="6mo")
                
                if df is None or len(df) < 20:
                    no_pattern.append({"symbol": symbol, "reason": "Insufficient data"})
                    continue
                
                vcp = detect_vcp(df, symbol=symbol, config=self.config)
                
                if vcp.is_valid:
                    vcp_stocks.append({
                        "symbol": symbol,
                        "score": vcp.score,
                        "pivot_price": vcp.pivot_price,
                        "base_length": vcp.base_length_days,
                        "volume_dry_up": vcp.volume_dry_up,
                        "contractions": len(vcp.contractions),
                        "details": vcp.details
                    })
                else:
                    no_pattern.append({
                        "symbol": symbol,
                        "reason": vcp.details,
                        "score": vcp.score
                    })
                    
            except Exception as e:
                logger.error(f"Error detecting VCP for {symbol}: {e}")
                no_pattern.append({"symbol": symbol, "reason": str(e)})
        
        # Sort by VCP quality score
        vcp_stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return AgentResult(
            success=True,
            data={
                "vcp_stocks": vcp_stocks,
                "no_pattern_stocks": no_pattern,
                "total_analyzed": len(stocks),
                "vcp_count": len(vcp_stocks)
            },
            message=f"Found {len(vcp_stocks)} stocks with valid VCP patterns.",
            reasoning=f"Analyzed {len(stocks)} stocks. {len(vcp_stocks)} show valid VCP setups."
        )
    
    def run(self, input_data: Any, max_iterations: int = 5) -> AgentResult:
        if isinstance(input_data, dict):
            stocks = input_data.get("qualified_stocks", [])
        elif isinstance(input_data, list):
            stocks = input_data
        else:
            stocks = []
        
        if not stocks:
            return AgentResult(success=False, data=None, message="No stocks provided")
        
        logger.info(f"[{self.name}] Analyzing {len(stocks)} stocks for VCP patterns")
        return self.analyze_stocks(stocks)


if __name__ == "__main__":
    agent = VCPPatternAgent()
    result = agent.run([{"symbol": "TITAN"}, {"symbol": "TCS"}])
    print(f"VCP Stocks: {len(result.data.get('vcp_stocks', []))}")
