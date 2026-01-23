"""
SEPA Trading Workflow - Streamlit Dashboard

Main entry point for the Streamlit UI providing:
- Stock screening dashboard
- VCP pattern visualization
- Trade signal management
- Portfolio performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Page config
st.set_page_config(
    page_title="SEPA Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .status-pass {
        color: #00d26a;
        font-weight: 600;
    }
    .status-fail {
        color: #ff4757;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


def load_scan_results():
    """Load the latest scan results."""
    try:
        from src.pipeline import SEPAPipeline
        pipeline = SEPAPipeline()
        return pipeline.get_latest_results()
    except Exception as e:
        st.warning(f"Could not load results: {e}")
        return None


def load_portfolio_metrics():
    """Load portfolio performance metrics."""
    try:
        from src.tools import get_portfolio_metrics, get_open_positions
        return {
            "metrics": get_portfolio_metrics(),
            "positions": get_open_positions()
        }
    except Exception as e:
        return {"metrics": {}, "positions": []}


def render_header():
    """Render the main header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📈 SEPA Trading Dashboard</p>', unsafe_allow_html=True)
        st.caption("Minervini Trend Template • VCP Pattern Detection • Risk Management")
    with col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def run_realtime_scan_with_trace(stage1_container):
    """Run scan with real-time stock-by-stock updates and detailed trace in Stage 1 expander."""
    from src.tools import get_stock_data, check_trend_template, get_nifty50_stocks, get_nifty500_stocks

    universe = st.session_state.get("universe", "nifty50")

    if universe == "nifty500":
        symbols = get_nifty500_stocks()
    else:
        symbols = get_nifty50_stocks()

    # Progress bar at the top
    st.info(f"🔍 Scanning {len(symbols)} stocks from {universe.upper()}")
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_results = []
    passed_stocks = []
    failed_stocks = []

    # Create placeholder for the live updating table inside Stage 1 expander
    with stage1_container:
        st.markdown("##### 📊 Real-Time 8-Point Criteria Analysis")
        current_stock_display = st.empty()
        live_table_placeholder = st.empty()

    for i, symbol in enumerate(symbols):
        # Update progress
        progress = (i + 1) / len(symbols)
        progress_bar.progress(progress)
        status_text.text(f"Stage 1: Trend Template — Analyzing {symbol} ({i+1}/{len(symbols)})")

        try:
            df = get_stock_data(symbol, period="2y")

            if df is None or len(df) < 200:
                row = {
                    "symbol": symbol,
                    "price": "N/A",
                    "c1_icon": "⚠️", "c1_val": "No data", "c1_pass": False,
                    "c2_icon": "⚠️", "c2_val": "", "c2_pass": False,
                    "c3_icon": "⚠️", "c3_val": "", "c3_pass": False,
                    "c4_icon": "⚠️", "c4_val": "", "c4_pass": False,
                    "c5_icon": "⚠️", "c5_val": "", "c5_pass": False,
                    "c6_icon": "⚠️", "c6_val": "", "c6_pass": False,
                    "c7_icon": "⚠️", "c7_val": "", "c7_pass": False,
                    "c8_icon": "⚠️", "c8_val": "", "c8_pass": False,
                    "all_passed": False,
                    "passed_count": 0,
                    "error": "Insufficient data"
                }
                failed_stocks.append(row)
                all_results.append(row)
                current_stock_display.warning(f"⏳ {symbol}: Insufficient data")
                continue

            # Run trend template check
            tt_result = check_trend_template(df)
            checks = tt_result.get("checks", {})
            current_price = df['Close'].iloc[-1]

            # Extract each check with values
            def format_check(check_data):
                passed = check_data.get("passed", False)
                value = check_data.get("value", "")
                icon = "✅" if passed else "❌"
                return icon, passed, value

            c1 = checks.get("price_above_150_sma", {})
            c2 = checks.get("price_above_200_sma", {})
            c3 = checks.get("sma_150_above_200", {})
            c4 = checks.get("sma_200_trending_up", {})
            c5 = checks.get("sma_50_above_150_and_200", {})
            c6 = checks.get("price_above_50_sma", {})
            c7 = checks.get("price_above_52w_low", {})
            c8 = checks.get("price_within_52w_high", {})

            c1_icon, c1_pass, c1_val = format_check(c1)
            c2_icon, c2_pass, c2_val = format_check(c2)
            c3_icon, c3_pass, c3_val = format_check(c3)
            c4_icon, c4_pass, c4_val = format_check(c4)
            c5_icon, c5_pass, c5_val = format_check(c5)
            c6_icon, c6_pass, c6_val = format_check(c6)
            c7_icon, c7_pass, c7_val = format_check(c7)
            c8_icon, c8_pass, c8_val = format_check(c8)

            all_passed = tt_result.get("all_passed", False)
            passed_count = tt_result.get("passed_count", 0)

            row = {
                "symbol": symbol,
                "price": f"₹{current_price:,.2f}",
                "c1_icon": c1_icon, "c1_val": c1_val, "c1_pass": c1_pass,
                "c2_icon": c2_icon, "c2_val": c2_val, "c2_pass": c2_pass,
                "c3_icon": c3_icon, "c3_val": c3_val, "c3_pass": c3_pass,
                "c4_icon": c4_icon, "c4_val": c4_val, "c4_pass": c4_pass,
                "c5_icon": c5_icon, "c5_val": c5_val, "c5_pass": c5_pass,
                "c6_icon": c6_icon, "c6_val": c6_val, "c6_pass": c6_pass,
                "c7_icon": c7_icon, "c7_val": c7_val, "c7_pass": c7_pass,
                "c8_icon": c8_icon, "c8_val": c8_val, "c8_pass": c8_pass,
                "all_passed": all_passed,
                "passed_count": passed_count,
                "checks": checks
            }

            if all_passed:
                passed_stocks.append(row)
                icon = "✅"
            else:
                failed_stocks.append(row)
                icon = "❌"

            all_results.append(row)

            # Show current stock being processed
            current_stock_display.markdown(
                f"{icon} **{symbol}** — {row['price']} — {passed_count}/8 criteria | "
                f"{c1_icon}{c2_icon}{c3_icon}{c4_icon}{c5_icon}{c6_icon}{c7_icon}{c8_icon}"
            )

            # Update the live table with all results so far
            _update_live_trace_table(live_table_placeholder, all_results)

        except Exception as e:
            row = {
                "symbol": symbol,
                "price": "Error",
                "c1_icon": "❌", "c2_icon": "❌", "c3_icon": "❌", "c4_icon": "❌",
                "c5_icon": "❌", "c6_icon": "❌", "c7_icon": "❌", "c8_icon": "❌",
                "all_passed": False,
                "passed_count": 0,
                "error": str(e)
            }
            failed_stocks.append(row)
            all_results.append(row)
            current_stock_display.error(f"❌ {symbol}: Error - {str(e)[:50]}")

    # Clear progress display
    progress_bar.empty()
    status_text.empty()
    current_stock_display.empty()

    # Store results in session state
    st.session_state["stage1_trace_results"] = {
        "all": all_results,
        "passed": passed_stocks,
        "failed": failed_stocks,
        "total": len(symbols)
    }

    return passed_stocks, failed_stocks, all_results


def _update_live_trace_table(placeholder, results):
    """Update the live trace table with current results."""
    import pandas as pd

    df_data = []
    for r in results:
        df_data.append({
            "Symbol": r.get("symbol", ""),
            "Price": r.get("price", "N/A"),
            "①": r.get("c1_icon", "?"),
            "②": r.get("c2_icon", "?"),
            "③": r.get("c3_icon", "?"),
            "④": r.get("c4_icon", "?"),
            "⑤": r.get("c5_icon", "?"),
            "⑥": r.get("c6_icon", "?"),
            "⑦": r.get("c7_icon", "?"),
            "⑧": r.get("c8_icon", "?"),
            "Score": f"{r.get('passed_count', 0)}/8",
            "Result": "✅" if r.get("all_passed") else "❌"
        })

    df = pd.DataFrame(df_data)
    placeholder.dataframe(df, use_container_width=True, hide_index=True, height=400)


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("### 📈 SEPA Scanner")
        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        portfolio_value = st.number_input(
            "Portfolio Value (₹)",
            min_value=100000,
            max_value=100000000,
            value=st.session_state.get("portfolio_value", 1000000),
            step=100000,
            format="%d"
        )
        st.session_state["portfolio_value"] = portfolio_value

        universe = st.selectbox(
            "Stock Universe",
            ["NIFTY 50", "NIFTY 500"],
            index=0
        )
        st.session_state["universe"] = universe.lower().replace(" ", "")

        st.markdown("---")

        # Real-time scan button with Stage 1 trace
        if st.button("🔍 Run Scan with Trace", use_container_width=True, type="primary"):
            st.session_state["run_realtime_scan"] = True
            st.rerun()

        st.markdown("---")
        st.caption("Minervini SEPA Trading System")


def render_stage1_trace_results():
    """Display Stage 1 trace results with detailed criteria breakdown."""
    results = st.session_state.get("stage1_trace_results", {})
    all_results = results.get("all", [])
    passed = results.get("passed", [])
    failed = results.get("failed", [])

    if not all_results:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scanned", len(all_results))
    with col2:
        st.metric("Passed All 8", len(passed), delta="Stage 2 Qualified")
    with col3:
        st.metric("Failed", len(failed))

    # Tabs for passed/failed
    tab1, tab2 = st.tabs(["✅ Passed Stocks", "❌ Failed Stocks"])

    with tab1:
        if passed:
            for stock in passed:
                with st.expander(f"**{stock['symbol']}** — {stock.get('price', 'N/A')} — ✅ PASSED ({stock.get('passed_count', 8)}/8)", expanded=False):
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"**① Price > 150 SMA:** {stock.get('c1_icon', '?')}")
                        st.caption(f"{stock.get('c1_val', '')}")
                        st.markdown(f"**② Price > 200 SMA:** {stock.get('c2_icon', '?')}")
                        st.caption(f"{stock.get('c2_val', '')}")
                    with cols[1]:
                        st.markdown(f"**③ 150 > 200 SMA:** {stock.get('c3_icon', '?')}")
                        st.caption(f"{stock.get('c3_val', '')}")
                        st.markdown(f"**④ 200 SMA Trend:** {stock.get('c4_icon', '?')}")
                        st.caption(f"{stock.get('c4_val', '')}")
                    with cols[2]:
                        st.markdown(f"**⑤ 50 > 150 & 200:** {stock.get('c5_icon', '?')}")
                        st.caption(f"{stock.get('c5_val', '')}")
                        st.markdown(f"**⑥ Price > 50 SMA:** {stock.get('c6_icon', '?')}")
                        st.caption(f"{stock.get('c6_val', '')}")
                    with cols[3]:
                        st.markdown(f"**⑦ >30% above Low:** {stock.get('c7_icon', '?')}")
                        st.caption(f"{stock.get('c7_val', '')}")
                        st.markdown(f"**⑧ <25% from High:** {stock.get('c8_icon', '?')}")
                        st.caption(f"{stock.get('c8_val', '')}")
        else:
            st.info("No stocks passed all 8 criteria")

    with tab2:
        if failed:
            for stock in failed[:20]:
                failed_criteria = []
                if not stock.get('c1_pass', True): failed_criteria.append("①")
                if not stock.get('c2_pass', True): failed_criteria.append("②")
                if not stock.get('c3_pass', True): failed_criteria.append("③")
                if not stock.get('c4_pass', True): failed_criteria.append("④")
                if not stock.get('c5_pass', True): failed_criteria.append("⑤")
                if not stock.get('c6_pass', True): failed_criteria.append("⑥")
                if not stock.get('c7_pass', True): failed_criteria.append("⑦")
                if not stock.get('c8_pass', True): failed_criteria.append("⑧")

                passed_count = stock.get('passed_count', 0)
                with st.expander(f"**{stock['symbol']}** — {stock.get('price', 'N/A')} — ❌ FAILED ({passed_count}/8) — Failed: {', '.join(failed_criteria)}", expanded=False):
                    if "error" in stock and stock.get("error"):
                        st.error(stock["error"])
                    else:
                        cols = st.columns(4)
                        with cols[0]:
                            st.markdown(f"**① Price > 150 SMA:** {stock.get('c1_icon', '?')}")
                            st.caption(f"{stock.get('c1_val', '')}")
                            st.markdown(f"**② Price > 200 SMA:** {stock.get('c2_icon', '?')}")
                            st.caption(f"{stock.get('c2_val', '')}")
                        with cols[1]:
                            st.markdown(f"**③ 150 > 200 SMA:** {stock.get('c3_icon', '?')}")
                            st.caption(f"{stock.get('c3_val', '')}")
                            st.markdown(f"**④ 200 SMA Trend:** {stock.get('c4_icon', '?')}")
                            st.caption(f"{stock.get('c4_val', '')}")
                        with cols[2]:
                            st.markdown(f"**⑤ 50 > 150 & 200:** {stock.get('c5_icon', '?')}")
                            st.caption(f"{stock.get('c5_val', '')}")
                            st.markdown(f"**⑥ Price > 50 SMA:** {stock.get('c6_icon', '?')}")
                            st.caption(f"{stock.get('c6_val', '')}")
                        with cols[3]:
                            st.markdown(f"**⑦ >30% above Low:** {stock.get('c7_icon', '?')}")
                            st.caption(f"{stock.get('c7_val', '')}")
                            st.markdown(f"**⑧ <25% from High:** {stock.get('c8_icon', '?')}")
                            st.caption(f"{stock.get('c8_val', '')}")

            if len(failed) > 20:
                st.caption(f"... and {len(failed) - 20} more")
        else:
            st.info("All stocks passed!")

    # Legend
    with st.expander("📖 Criteria Legend", expanded=False):
        st.markdown("""
| # | Criteria | Description |
|---|----------|-------------|
| ① | Price > 150 SMA | Current price above 150-day moving average |
| ② | Price > 200 SMA | Current price above 200-day moving average |
| ③ | 150 > 200 SMA | 150-day SMA is above 200-day SMA |
| ④ | 200 SMA ↑ | 200-day SMA is trending upward |
| ⑤ | 50 > 150 & 200 | 50-day SMA is above both 150 and 200 |
| ⑥ | Price > 50 SMA | Current price above 50-day moving average |
| ⑦ | >30% above Low | Price is at least 30% above 52-week low |
| ⑧ | <25% from High | Price is within 25% of 52-week high |
        """)


def render_overview_tab():
    """Render the main overview/dashboard tab."""
    st.subheader("📊 Market Overview")

    # Check if real-time scan was triggered
    if st.session_state.get("run_realtime_scan"):
        st.session_state["run_realtime_scan"] = False

        st.markdown("### 🔍 Real-Time SEPA Pipeline Scan")

        # Create Stage 1 expander FIRST, then run scan inside it
        stage1_expander = st.expander("📈 **Stage 1: Trend Template** — Scanning...", expanded=True)

        # Run the scan with real-time trace inside the expander
        passed, failed, all_results = run_realtime_scan_with_trace(stage1_expander)

        # Show summary
        st.success(f"✅ Stage 1 Complete! {len(passed)} passed / {len(failed)} failed")

        # Display final detailed results in Stage 1 expander
        with stage1_expander:
            st.markdown("---")
            render_stage1_trace_results()

        return

    # Check for existing Stage 1 trace results to display
    stage1_results = st.session_state.get("stage1_trace_results")
    if stage1_results:
        st.markdown("### 📊 Last Scan Results")

        # Show Stage 1 results in expander
        passed_count = len(stage1_results.get("passed", []))
        failed_count = len(stage1_results.get("failed", []))
        with st.expander(f"📈 **Stage 1: Trend Template** — {passed_count} passed / {failed_count} failed", expanded=True):
            render_stage1_trace_results()

        st.markdown("---")

    # Standard metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    scan_result = st.session_state.get("scan_result")
    stage1 = st.session_state.get("stage1_trace_results", {})

    with col1:
        total = stage1.get("total", scan_result.stocks_scanned if scan_result else 0)
        st.metric(label="Stocks Scanned", value=total if total else "—")

    with col2:
        passed_val = len(stage1.get("passed", [])) if stage1 else (scan_result.trend_passed if scan_result else 0)
        st.metric(label="Trend Passed", value=passed_val if passed_val else "—")

    with col3:
        st.metric(
            label="VCP Patterns",
            value=scan_result.vcp_patterns if scan_result else "—"
        )

    with col4:
        st.metric(
            label="Buy Signals",
            value=scan_result.buy_signals if scan_result else "—"
        )

    with col5:
        st.metric(
            label="Trade Orders",
            value=scan_result.trade_orders if scan_result else "—",
            delta="Ready" if scan_result and scan_result.trade_orders > 0 else None
        )

    st.markdown("---")

    # Pipeline stages with detailed reports (from full pipeline scan)
    if scan_result and scan_result.stages:
        st.subheader("🔄 Pipeline Stages - Detailed Report")

        stages = scan_result.stages

        # Stage 1: Trend Template (from full pipeline)
        if "trend_template" in stages and not stage1_results:
            stage_data = stages["trend_template"]
            with st.expander(f"📈 **Stage 1: Trend Template** — {stage_data.get('passed', 0)} passed / {stage_data.get('failed', 0)} failed", expanded=True):
                if "error" in stage_data:
                    st.error(f"Error: {stage_data['error']}")
                elif "skipped" in stage_data:
                    st.info(f"Skipped: {stage_data['skipped']}")
                else:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### ✅ Passed Stocks")
                        passed_detail = stage_data.get("passed_stocks_detail", [])
                        if passed_detail:
                            for stock in passed_detail:
                                symbol = stock.get("symbol", stock) if isinstance(stock, dict) else stock
                                with st.container():
                                    st.success(f"**{symbol}** — Passed all 8 criteria")
                                    if isinstance(stock, dict) and "checks" in stock:
                                        checks = stock.get("checks", {})
                                        for check_name, check_data in checks.items():
                                            passed = check_data.get("passed", False)
                                            icon = "✓" if passed else "✗"
                                            st.caption(f"  {icon} {check_name}")
                        else:
                            st.info("No stocks passed")

                    with col2:
                        st.markdown("##### ❌ Failed Stocks")
                        failed_detail = stage_data.get("failed_stocks_detail", [])
                        if failed_detail:
                            for stock in failed_detail[:10]:  # Limit to first 10
                                symbol = stock.get("symbol", stock) if isinstance(stock, dict) else stock
                                failed_criteria = stock.get("failed_criteria", []) if isinstance(stock, dict) else []
                                st.error(f"**{symbol}** — Failed: {', '.join(failed_criteria[:3])}")
                            if len(failed_detail) > 10:
                                st.caption(f"... and {len(failed_detail) - 10} more")
                        else:
                            st.info("No failures")

        # Stage 2: Fundamentals
        if "fundamentals" in stages:
            stage_data = stages["fundamentals"]
            qualified = stage_data.get('qualified', 0)
            disqualified = stage_data.get('disqualified', 0)
            with st.expander(f"💰 **Stage 2: Fundamentals** — {qualified} qualified / {disqualified} disqualified"):
                if "error" in stage_data:
                    st.error(f"Error: {stage_data['error']}")
                elif "skipped" in stage_data:
                    st.info(f"Skipped: {stage_data['skipped']}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### ✅ Qualified (Strong Growth)")
                        for stock in stage_data.get("qualified_stocks_detail", []):
                            symbol = stock.get("symbol", "") if isinstance(stock, dict) else stock
                            earnings = stock.get("earnings", {}) if isinstance(stock, dict) else {}
                            revenue = stock.get("revenue", {}) if isinstance(stock, dict) else {}
                            st.success(f"**{symbol}**")
                            st.caption(f"  EPS: {earnings.get('latest_growth_pct', 0)}% | Revenue: {revenue.get('latest_growth_pct', 0)}%")
                    
                    with col2:
                        st.markdown("##### ❌ Disqualified")
                        for stock in stage_data.get("disqualified_stocks_detail", [])[:10]:
                            symbol = stock.get("symbol", "") if isinstance(stock, dict) else stock
                            reason = stock.get("reason", "Criteria not met") if isinstance(stock, dict) else "Criteria not met"
                            st.error(f"**{symbol}** — {reason}")
        
        # Stage 3: VCP Patterns
        if "vcp_patterns" in stages:
            stage_data = stages["vcp_patterns"]
            patterns = stage_data.get('patterns_found', 0)
            no_pattern = stage_data.get('no_pattern', 0)
            with st.expander(f"📊 **Stage 3: VCP Patterns** — {patterns} patterns found"):
                if "error" in stage_data:
                    st.error(f"Error: {stage_data['error']}")
                elif "skipped" in stage_data:
                    st.info(f"Skipped: {stage_data['skipped']}")
                else:
                    vcp_stocks = stage_data.get("vcp_stocks_detail", [])
                    if vcp_stocks:
                        for stock in vcp_stocks:
                            symbol = stock.get("symbol", "")
                            score = stock.get("score", 0)
                            pivot = stock.get("pivot_price", 0)
                            contractions = stock.get("contractions", 0)
                            st.success(f"**{symbol}** — Score: {score}/100 | Pivot: ₹{pivot:,.2f} | Contractions: {contractions}")
                    else:
                        st.info("No VCP patterns detected")
                    
                    no_pattern_stocks = stage_data.get("no_pattern_detail", [])
                    if no_pattern_stocks:
                        with st.container():
                            st.markdown("**No pattern detected:**")
                            for stock in no_pattern_stocks[:5]:
                                symbol = stock.get("symbol", "") if isinstance(stock, dict) else stock
                                reason = stock.get("reason", "") if isinstance(stock, dict) else ""
                                st.caption(f"  {symbol}: {reason}")
        
        # Stage 4 & 5: Entry Points and Risk
        if "entry_points" in stages:
            stage_data = stages["entry_points"]
            signals = stage_data.get('buy_signals', 0)
            actionable = stage_data.get('actionable', 0)
            with st.expander(f"🎯 **Stage 4: Entry Points** — {actionable} actionable signals"):
                if "error" in stage_data or "skipped" in stage_data:
                    st.info(stage_data.get("error") or stage_data.get("skipped"))
                else:
                    st.write(f"Total signals: {signals}, Actionable: {actionable}")
        
        if "risk_management" in stages:
            stage_data = stages["risk_management"]
            orders = stage_data.get('trade_orders', 0)
            risk = stage_data.get('total_risk', 0)
            with st.expander(f"⚖️ **Stage 5: Risk Management** — {orders} trade orders"):
                if "error" in stage_data or "skipped" in stage_data:
                    st.info(stage_data.get("error") or stage_data.get("skipped"))
                else:
                    st.write(f"Trade orders: {orders}")
                    st.write(f"Total risk: ₹{risk:,.0f}")
    else:
        st.info("👆 Click 'Run Full Scan' in the sidebar to start screening stocks.")


def render_signals_tab():
    """Render the trade signals tab."""
    st.subheader("🎯 Trade Signals")
    
    scan_result = st.session_state.get("scan_result")
    
    if not scan_result:
        st.info("No scan results available. Run a scan first.")
        return
    
    # Load trade orders from results
    stages = scan_result.stages
    risk_stage = stages.get("risk_management", {})
    
    if "trade_orders" not in risk_stage:
        st.info("No trade signals generated in the last scan.")
        return
    
    # This would need to be loaded from storage in real implementation
    st.warning("Trade orders are saved to `data/scan_results/`. Load from there for full details.")
    
    # Demo table
    demo_data = {
        "Symbol": ["TITAN", "TCS"],
        "Action": ["BUY", "BUY"],
        "Entry Price": ["₹3,450", "₹4,120"],
        "Stop Loss": ["₹3,208", "₹3,832"],
        "Quantity": [29, 24],
        "Risk %": ["7.0%", "7.0%"],
        "Status": ["PENDING", "PENDING"]
    }
    
    df = pd.DataFrame(demo_data)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                options=["PENDING", "APPROVED", "REJECTED", "EXECUTED"]
            )
        }
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Approve All", type="primary"):
            st.success("All orders approved!")
    with col2:
        if st.button("❌ Reject All"):
            st.warning("All orders rejected.")


def render_positions_tab():
    """Render the open positions tab."""
    st.subheader("📂 Open Positions")
    
    portfolio = load_portfolio_metrics()
    positions = portfolio.get("positions", [])
    
    if not positions:
        st.info("No open positions. Start trading to see positions here.")
        
        # Demo data
        with st.expander("📋 Demo Data"):
            demo_positions = {
                "Symbol": ["RELIANCE.NS", "INFY.NS"],
                "Qty": [100, 50],
                "Entry": ["₹2,850", "₹1,720"],
                "Current": ["₹2,920", "₹1,680"],
                "P&L": ["+2.5%", "-2.3%"],
                "Stop Loss": ["₹2,650", "₹1,600"]
            }
            st.dataframe(pd.DataFrame(demo_positions), use_container_width=True, hide_index=True)
        return
    
    # Real positions table
    df = pd.DataFrame(positions)
    st.dataframe(df, use_container_width=True)


def render_performance_tab():
    """Render the portfolio performance tab."""
    st.subheader("📈 Portfolio Performance")
    
    portfolio = load_portfolio_metrics()
    metrics = portfolio.get("metrics", {})
    
    if not metrics or metrics.get("total_trades", 0) == 0:
        st.info("No trading history yet. Execute some trades to see performance metrics.")
        return
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics.get("total_trades", 0))
    with col2:
        win_rate = metrics.get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate}%", delta="Good" if win_rate >= 50 else "Review")
    with col3:
        st.metric("Avg Gain", f"{metrics.get('avg_gain_pct', 0)}%")
    with col4:
        st.metric("Profit Factor", metrics.get("profit_factor", 0))
    
    st.markdown("---")
    
    # Equity curve placeholder
    st.subheader("📉 Equity Curve")
    st.info("Equity curve will be displayed here once you have trade history.")


def render_config_tab():
    """Render the configuration editor tab."""
    st.subheader("⚙️ Agent Configuration")
    
    st.markdown("""
    Edit the criteria for each agent below. Changes are saved to the YAML config files.
    """)
    
    config_tabs = st.tabs([
        "Trend Template",
        "Fundamentals",
        "VCP Pattern",
        "Entry Point",
        "Risk Management"
    ])
    
    with config_tabs[0]:
        st.markdown("#### Trend Template Criteria (8-Point Check)")
        st.markdown("""
        The Trend Template identifies stocks in confirmed Stage 2 uptrends.
        All 8 criteria must be met.
        """)
        
        st.markdown("##### Moving Average Conditions")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("1. Price > 150-day SMA", value=True, disabled=True)
            st.checkbox("2. Price > 200-day SMA", value=True, disabled=True)
            st.checkbox("3. 150-day SMA > 200-day SMA", value=True, disabled=True)
        with col2:
            st.checkbox("4. 50-day SMA > 150 & 200", value=True, disabled=True)
            st.checkbox("5. Price > 50-day SMA", value=True, disabled=True)
        
        st.markdown("##### Trend & Strength Conditions")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("4. 200-SMA trending up (months)", value=1, min_value=1, max_value=6, 
                          help="200-day SMA should be trending up for at least 1 month (preferably 4-5)")
            st.number_input("6. Price above 52w low (%)", value=30, min_value=10, max_value=50,
                          help="Current price should be at least 30% above its 52-week low")
        with col2:
            st.number_input("7. Price within 52w high (%)", value=25, min_value=10, max_value=50,
                          help="Current price should be within 25% of 52-week high")
            st.number_input("8. RS Ranking minimum", value=70, min_value=50, max_value=99,
                          help="Relative Strength ranking should be 70+ (preferably 80-90s)")
    
    with config_tabs[1]:
        st.markdown("#### Fundamental Criteria")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Min EPS Growth YoY (%)", value=20, min_value=5, max_value=100)
        with col2:
            st.number_input("Min Revenue Growth (%)", value=15, min_value=5, max_value=100)
    
    with config_tabs[2]:
        st.markdown("#### VCP Pattern Criteria")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Min Contractions", value=2, min_value=1, max_value=5)
            st.number_input("Max First Contraction (%)", value=35, min_value=20, max_value=50)
        with col2:
            st.number_input("Volume Dry-up Threshold", value=0.5, min_value=0.3, max_value=0.8)
            st.number_input("Base Length Max (days)", value=65, min_value=30, max_value=120)
    
    with config_tabs[3]:
        st.markdown("#### Entry Point Criteria")
        st.number_input("Volume Expansion Multiplier", value=1.5, min_value=1.0, max_value=3.0)
        st.number_input("Proximity to 52w High (%)", value=15, min_value=5, max_value=30)
    
    with config_tabs[4]:
        st.markdown("#### Risk Management Criteria")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Target Stop Loss (%)", value=7, min_value=3, max_value=10)
            st.number_input("Max Stop Loss (%)", value=10, min_value=5, max_value=15)
        with col2:
            st.number_input("Max Risk per Trade (%)", value=2.0, min_value=0.5, max_value=5.0)
            st.number_input("Max Positions", value=10, min_value=3, max_value=20)
    
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved! (Demo - not actually saved)")


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    # Main content - Overview only for now
    render_overview_tab()

    # TODO: Uncomment to enable other tabs
    # tabs = st.tabs([
    #     "📊 Overview",
    #     "🎯 Trade Signals",
    #     "📂 Positions",
    #     "📈 Performance",
    #     "⚙️ Configuration"
    # ])
    #
    # with tabs[0]:
    #     render_overview_tab()
    # with tabs[1]:
    #     render_signals_tab()
    # with tabs[2]:
    #     render_positions_tab()
    # with tabs[3]:
    #     render_performance_tab()
    # with tabs[4]:
    #     render_config_tab()


if __name__ == "__main__":
    main()
