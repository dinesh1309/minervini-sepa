"""
SEPA Trading Workflow - Tools Package

This package provides all the data tools required for the
Minervini SEPA agentic trading workflow.
"""

from .stock_universe import (
    get_all_indian_stocks,
    get_nifty50_stocks,
    get_nifty500_stocks,
    filter_active_stocks
)

from .market_data import (
    get_stock_data,
    get_stock_data_range,
    get_current_price,
    get_batch_stock_data
)

from .technical_analysis import (
    calculate_moving_averages,
    get_sma_trend_direction,
    calculate_52_week_metrics,
    get_relative_strength,
    calculate_rs_ranking_universe,
    check_rs_line_trend,
    calculate_atr,
    calculate_volume_metrics,
    check_trend_template
)

from .fundamental_analysis import (
    get_fundamentals,
    analyze_earnings_growth,
    analyze_revenue_growth,
    analyze_margins,
    check_fundamentals
)

from .pattern_detection import (
    detect_vcp,
    identify_pivot,
    VCPPattern,
    Contraction
)

from .execution import (
    calculate_position_size,
    execute_paper_trade,
    get_open_positions,
    get_portfolio_metrics,
    get_trade_history,
    TradeOrder,
    Position
)

__all__ = [
    # Stock Universe
    'get_all_indian_stocks',
    'get_nifty50_stocks',
    'get_nifty500_stocks',
    'filter_active_stocks',
    
    # Market Data
    'get_stock_data',
    'get_stock_data_range',
    'get_current_price',
    'get_batch_stock_data',
    
    # Technical Analysis
    'calculate_moving_averages',
    'get_sma_trend_direction',
    'calculate_52_week_metrics',
    'get_relative_strength',
    'calculate_rs_ranking_universe',
    'check_rs_line_trend',
    'calculate_atr',
    'calculate_volume_metrics',
    'check_trend_template',
    
    # Fundamental Analysis
    'get_fundamentals',
    'analyze_earnings_growth',
    'analyze_revenue_growth',
    'analyze_margins',
    'check_fundamentals',
    
    # Pattern Detection
    'detect_vcp',
    'identify_pivot',
    'VCPPattern',
    'Contraction',
    
    # Execution
    'calculate_position_size',
    'execute_paper_trade',
    'get_open_positions',
    'get_portfolio_metrics',
    'get_trade_history',
    'TradeOrder',
    'Position'
]
