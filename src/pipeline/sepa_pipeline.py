"""
SEPA Pipeline Orchestration

This module wires all agents together in sequence to create
the complete SEPA trading workflow:

Flow: Stock Universe → Trend Template → Fundamentals → VCP → Entry → Risk → Execute

Also includes scheduling and result storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

from ..agents import (
    TrendTemplateAgent,
    FundamentalAgent,
    VCPPatternAgent,
    EntryPointAgent,
    RiskManagementAgent,
    PositionAgent,
    PortfolioReviewAgent,
    get_ai_client,
    AgentResult
)
from ..tools import get_nifty50_stocks, get_nifty500_stocks, get_all_indian_stocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Results storage
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scan_results"


@dataclass
class PipelineResult:
    """Result from a complete pipeline run."""
    timestamp: str
    status: str
    stocks_scanned: int
    trend_passed: int
    fundamental_passed: int
    vcp_patterns: int
    buy_signals: int
    trade_orders: int
    stages: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class SEPAPipeline:
    """
    Main pipeline that orchestrates all SEPA agents.
    
    The pipeline runs agents in sequence, passing results
    from one stage to the next.
    """
    
    def __init__(
        self,
        portfolio_value: float = 1000000,
        model: str = "openai:gpt-4o",
        use_ai_reasoning: bool = False
    ):
        """
        Initialize the SEPA pipeline.
        
        Args:
            portfolio_value: Total portfolio value for position sizing
            model: AI model for agent reasoning (if enabled)
            use_ai_reasoning: Whether to use LLM for analysis (slower but more context)
        """
        self.portfolio_value = portfolio_value
        self.use_ai_reasoning = use_ai_reasoning
        
        # Initialize shared AI client if using AI reasoning
        self.ai_client = get_ai_client(model) if use_ai_reasoning else None
        
        # Initialize agents
        self._init_agents()
        
        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _init_agents(self):
        """Initialize all pipeline agents."""
        self.trend_agent = TrendTemplateAgent(ai_client=self.ai_client)
        self.fundamental_agent = FundamentalAgent(ai_client=self.ai_client)
        self.vcp_agent = VCPPatternAgent(ai_client=self.ai_client)
        self.entry_agent = EntryPointAgent(ai_client=self.ai_client)
        self.risk_agent = RiskManagementAgent(
            ai_client=self.ai_client,
            portfolio_value=self.portfolio_value
        )
        self.position_agent = PositionAgent(ai_client=self.ai_client)
        self.portfolio_agent = PortfolioReviewAgent(ai_client=self.ai_client)
    
    def run_full_scan(
        self,
        symbols: Optional[List[str]] = None,
        universe: str = "nifty50"
    ) -> PipelineResult:
        """
        Run the complete SEPA scan pipeline.
        
        Args:
            symbols: Optional list of symbols to scan (overrides universe)
            universe: Stock universe if symbols not provided:
                      'nifty50', 'nifty500', or 'all'
        
        Returns:
            PipelineResult with complete scan results
        """
        timestamp = datetime.now().isoformat()
        errors = []
        stages = {}
        
        # Get stock universe
        if symbols:
            stock_list = symbols
        elif universe == "nifty50":
            stock_list = get_nifty50_stocks()
        elif universe == "nifty500":
            stock_list = get_nifty500_stocks()
        else:
            all_stocks = get_all_indian_stocks()
            stock_list = [s["symbol"] for s in all_stocks[:100]]  # Limit for performance
        
        logger.info(f"Starting SEPA scan on {len(stock_list)} stocks")
        
        # Stage 1: Trend Template Filter
        logger.info("Stage 1: Running Trend Template Agent...")
        try:
            trend_result = self.trend_agent.run(stock_list)
            passed_stocks = trend_result.data.get("passed_stocks", [])
            failed_stocks = trend_result.data.get("failed_stocks", [])
            
            stages["trend_template"] = {
                "success": trend_result.success,
                "passed": len(passed_stocks),
                "failed": len(failed_stocks),
                "message": trend_result.message,
                # Detailed results for UI
                "passed_stocks_detail": passed_stocks,
                "failed_stocks_detail": failed_stocks
            }
            trend_passed = passed_stocks
        except Exception as e:
            logger.error(f"Trend Template Agent error: {e}")
            errors.append(f"Stage 1 error: {str(e)}")
            trend_passed = []
            stages["trend_template"] = {"error": str(e)}
        
        # Stage 2: Fundamental Filter
        logger.info("Stage 2: Running Fundamental Agent...")
        try:
            if trend_passed:
                fund_result = self.fundamental_agent.run({"passed_stocks": trend_passed})
                qualified_stocks = fund_result.data.get("qualified_stocks", [])
                disqualified = fund_result.data.get("disqualified_stocks", [])
                
                stages["fundamentals"] = {
                    "success": fund_result.success,
                    "qualified": len(qualified_stocks),
                    "disqualified": len(disqualified),
                    "message": fund_result.message,
                    "qualified_stocks_detail": qualified_stocks,
                    "disqualified_stocks_detail": disqualified
                }
                fund_passed = qualified_stocks
            else:
                fund_passed = []
                stages["fundamentals"] = {"skipped": "No stocks from Stage 1"}
        except Exception as e:
            logger.error(f"Fundamental Agent error: {e}")
            errors.append(f"Stage 2 error: {str(e)}")
            fund_passed = []
            stages["fundamentals"] = {"error": str(e)}
        
        # Stage 3: VCP Pattern Detection
        logger.info("Stage 3: Running VCP Pattern Agent...")
        try:
            if fund_passed:
                vcp_result = self.vcp_agent.run({"qualified_stocks": fund_passed})
                vcp_found = vcp_result.data.get("vcp_stocks", [])
                no_pattern = vcp_result.data.get("no_pattern_stocks", [])
                
                stages["vcp_patterns"] = {
                    "success": vcp_result.success,
                    "patterns_found": len(vcp_found),
                    "no_pattern": len(no_pattern),
                    "message": vcp_result.message,
                    "vcp_stocks_detail": vcp_found,
                    "no_pattern_detail": no_pattern
                }
                vcp_stocks = vcp_found
            else:
                vcp_stocks = []
                stages["vcp_patterns"] = {"skipped": "No stocks from Stage 2"}
        except Exception as e:
            logger.error(f"VCP Agent error: {e}")
            errors.append(f"Stage 3 error: {str(e)}")
            vcp_stocks = []
            stages["vcp_patterns"] = {"error": str(e)}
        
        # Stage 4: Entry Point Identification
        logger.info("Stage 4: Running Entry Point Agent...")
        try:
            if vcp_stocks:
                entry_result = self.entry_agent.run({"vcp_stocks": vcp_stocks})
                stages["entry_points"] = {
                    "success": entry_result.success,
                    "buy_signals": len(entry_result.data.get("buy_signals", [])),
                    "actionable": entry_result.data.get("actionable_count", 0),
                    "message": entry_result.message
                }
                buy_signals = entry_result.data.get("buy_signals", [])
            else:
                buy_signals = []
                stages["entry_points"] = {"skipped": "No VCP patterns found"}
        except Exception as e:
            logger.error(f"Entry Point Agent error: {e}")
            errors.append(f"Stage 4 error: {str(e)}")
            buy_signals = []
            stages["entry_points"] = {"error": str(e)}
        
        # Stage 5: Risk Management
        logger.info("Stage 5: Running Risk Management Agent...")
        try:
            if buy_signals:
                risk_result = self.risk_agent.run({"buy_signals": buy_signals})
                stages["risk_management"] = {
                    "success": risk_result.success,
                    "trade_orders": len(risk_result.data.get("trade_orders", [])),
                    "total_risk": risk_result.data.get("total_risk", 0),
                    "message": risk_result.message
                }
                trade_orders = risk_result.data.get("trade_orders", [])
            else:
                trade_orders = []
                stages["risk_management"] = {"skipped": "No buy signals"}
        except Exception as e:
            logger.error(f"Risk Agent error: {e}")
            errors.append(f"Stage 5 error: {str(e)}")
            trade_orders = []
            stages["risk_management"] = {"error": str(e)}
        
        # Build final result
        result = PipelineResult(
            timestamp=timestamp,
            status="COMPLETE" if not errors else "COMPLETE_WITH_ERRORS",
            stocks_scanned=len(stock_list),
            trend_passed=len(trend_passed),
            fundamental_passed=len(fund_passed),
            vcp_patterns=len(vcp_stocks),
            buy_signals=len(buy_signals),
            trade_orders=len(trade_orders),
            stages=stages,
            errors=errors
        )
        
        # Save results
        self._save_results(result, trade_orders)
        
        logger.info(f"Pipeline complete: {result.trade_orders} trade orders generated")
        return result
    
    def run_position_review(self) -> AgentResult:
        """
        Run position management review on open positions.
        
        Returns:
            AgentResult from position management agent
        """
        logger.info("Running position review...")
        return self.position_agent.run()
    
    def run_portfolio_review(self) -> AgentResult:
        """
        Run portfolio performance review.
        
        Returns:
            AgentResult from portfolio review agent
        """
        logger.info("Running portfolio review...")
        return self.portfolio_agent.run()
    
    def _save_results(self, result: PipelineResult, trade_orders: List[Dict]):
        """Save scan results to file."""
        try:
            # Save summary
            timestamp = result.timestamp.replace(":", "-").split(".")[0]
            summary_file = RESULTS_DIR / f"scan_{timestamp}.json"
            
            with open(summary_file, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # Save trade orders if any
            if trade_orders:
                orders_file = RESULTS_DIR / f"orders_{timestamp}.json"
                with open(orders_file, "w") as f:
                    json.dump(trade_orders, f, indent=2, default=str)
            
            logger.info(f"Results saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_latest_results(self) -> Optional[Dict]:
        """Get the most recent scan results."""
        try:
            files = sorted(RESULTS_DIR.glob("scan_*.json"), reverse=True)
            if files:
                with open(files[0], "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
        return None


def run_daily_scan(
    universe: str = "nifty500",
    portfolio_value: float = 1000000
) -> PipelineResult:
    """
    Convenience function to run the daily SEPA scan.
    
    Args:
        universe: Stock universe to scan
        portfolio_value: Portfolio value for sizing
    
    Returns:
        PipelineResult from the scan
    """
    pipeline = SEPAPipeline(portfolio_value=portfolio_value)
    return pipeline.run_full_scan(universe=universe)


# Example usage
if __name__ == "__main__":
    # Run a test scan on NIFTY 50
    print("Starting SEPA Pipeline Test...")
    
    pipeline = SEPAPipeline(portfolio_value=1000000)
    
    # Test with a few stocks
    result = pipeline.run_full_scan(
        symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "TITAN"]
    )
    
    print(f"\n=== Pipeline Results ===")
    print(f"Status: {result.status}")
    print(f"Stocks Scanned: {result.stocks_scanned}")
    print(f"Trend Passed: {result.trend_passed}")
    print(f"Fundamental Passed: {result.fundamental_passed}")
    print(f"VCP Patterns: {result.vcp_patterns}")
    print(f"Buy Signals: {result.buy_signals}")
    print(f"Trade Orders: {result.trade_orders}")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    
    print("\n=== Stage Details ===")
    for stage, details in result.stages.items():
        print(f"{stage}: {details}")
