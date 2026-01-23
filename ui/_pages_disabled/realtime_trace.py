"""
Real-Time Stage Trace Page

Shows live progress as each stock is processed through the pipeline,
with actual values for each criteria.
"""

import streamlit as st
import pandas as pd
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="Real-Time Trace", page_icon="🔍", layout="wide")

st.title("🔍 Real-Time Stage Trace")
st.markdown("Watch stocks being analyzed in real-time with detailed criteria values")


def run_realtime_trend_template_scan():
    """Run Trend Template scan with real-time updates."""
    from src.tools import get_stock_data, check_trend_template, get_nifty50_stocks, get_nifty500_stocks
    
    # Get settings
    universe = st.session_state.get("universe", "nifty50")
    
    if universe == "nifty500":
        symbols = get_nifty500_stocks()
    else:
        symbols = get_nifty50_stocks()
    
    st.info(f"Scanning {len(symbols)} stocks from {universe.upper()}")
    
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results table
    st.subheader("📊 Trend Template Analysis - All 8 Criteria")
    
    # Column headers
    col_headers = [
        "Symbol", "Price", 
        "①P>150", "②P>200", "③150>200", "④200↑", 
        "⑤50>All", "⑥P>50", "⑦>30%Low", "⑧<25%Hi", 
        "Pass/Fail"
    ]
    
    # Create placeholders for real-time updates
    results_container = st.container()
    
    passed_stocks = []
    failed_stocks = []
    all_results = []
    
    for i, symbol in enumerate(symbols):
        # Update progress
        progress = (i + 1) / len(symbols)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
        
        try:
            # Get stock data
            df = get_stock_data(symbol, period="2y")
            
            if df is None or len(df) < 200:
                row = {
                    "symbol": symbol,
                    "price": "N/A",
                    "c1_val": "❌", "c1_pass": False,
                    "c2_val": "❌", "c2_pass": False,
                    "c3_val": "❌", "c3_pass": False,
                    "c4_val": "❌", "c4_pass": False,
                    "c5_val": "❌", "c5_pass": False,
                    "c6_val": "❌", "c6_pass": False,
                    "c7_val": "❌", "c7_pass": False,
                    "c8_val": "❌", "c8_pass": False,
                    "all_passed": False,
                    "error": "Insufficient data"
                }
                failed_stocks.append(row)
                all_results.append(row)
                continue
            
            # Run trend template check
            results = check_trend_template(df)
            checks = results.get("checks", {})
            
            current_price = df['Close'].iloc[-1]
            
            # Extract each check
            c1 = checks.get("price_above_150_sma", {})
            c2 = checks.get("price_above_200_sma", {})
            c3 = checks.get("sma_150_above_200", {})
            c4 = checks.get("sma_200_trending_up", {})
            c5 = checks.get("sma_50_above_150_and_200", {})
            c6 = checks.get("price_above_50_sma", {})
            c7 = checks.get("price_above_52w_low", {})
            c8 = checks.get("price_within_52w_high", {})
            
            def format_check(check_data):
                passed = check_data.get("passed", False)
                value = check_data.get("value", "")
                icon = "✅" if passed else "❌"
                return icon, passed, value
            
            c1_icon, c1_pass, c1_val = format_check(c1)
            c2_icon, c2_pass, c2_val = format_check(c2)
            c3_icon, c3_pass, c3_val = format_check(c3)
            c4_icon, c4_pass, c4_val = format_check(c4)
            c5_icon, c5_pass, c5_val = format_check(c5)
            c6_icon, c6_pass, c6_val = format_check(c6)
            c7_icon, c7_pass, c7_val = format_check(c7)
            c8_icon, c8_pass, c8_val = format_check(c8)
            
            all_passed = results.get("all_passed", False)
            
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
                "passed_count": results.get("passed_count", 0)
            }
            
            if all_passed:
                passed_stocks.append(row)
            else:
                failed_stocks.append(row)
            
            all_results.append(row)
            
        except Exception as e:
            row = {"symbol": symbol, "error": str(e), "all_passed": False}
            failed_stocks.append(row)
            all_results.append(row)
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"Scan complete! {len(passed_stocks)} passed, {len(failed_stocks)} failed")
    
    # Store results in session state
    st.session_state["trend_trace_results"] = all_results
    st.session_state["trend_passed"] = passed_stocks
    st.session_state["trend_failed"] = failed_stocks
    
    return passed_stocks, failed_stocks, all_results


def display_results():
    """Display the results in a detailed table."""
    all_results = st.session_state.get("trend_trace_results", [])
    passed = st.session_state.get("trend_passed", [])
    failed = st.session_state.get("trend_failed", [])
    
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
    
    st.markdown("---")
    
    # Tabs for passed/failed
    tab1, tab2, tab3 = st.tabs(["✅ Passed Stocks", "❌ Failed Stocks", "📋 All Results"])
    
    with tab1:
        if passed:
            st.markdown("### Stocks Passing All 8 Trend Template Criteria")
            for stock in passed:
                with st.expander(f"**{stock['symbol']}** - {stock.get('price', 'N/A')} - ✅ PASSED ({stock.get('passed_count', 8)}/8)", expanded=False):
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"**① Price > 150 SMA:** {stock.get('c1_icon', '?')}")
                        st.caption(stock.get('c1_val', ''))
                        st.markdown(f"**② Price > 200 SMA:** {stock.get('c2_icon', '?')}")
                        st.caption(stock.get('c2_val', ''))
                    with cols[1]:
                        st.markdown(f"**③ 150 > 200 SMA:** {stock.get('c3_icon', '?')}")
                        st.caption(stock.get('c3_val', ''))
                        st.markdown(f"**④ 200 SMA Trend:** {stock.get('c4_icon', '?')}")
                        st.caption(stock.get('c4_val', ''))
                    with cols[2]:
                        st.markdown(f"**⑤ 50 > 150 & 200:** {stock.get('c5_icon', '?')}")
                        st.caption(stock.get('c5_val', ''))
                        st.markdown(f"**⑥ Price > 50 SMA:** {stock.get('c6_icon', '?')}")
                        st.caption(stock.get('c6_val', ''))
                    with cols[3]:
                        st.markdown(f"**⑦ >30% above Low:** {stock.get('c7_icon', '?')}")
                        st.caption(stock.get('c7_val', ''))
                        st.markdown(f"**⑧ <25% from High:** {stock.get('c8_icon', '?')}")
                        st.caption(stock.get('c8_val', ''))
        else:
            st.info("No stocks passed all 8 criteria")
    
    with tab2:
        if failed:
            st.markdown("### Stocks That Failed One or More Criteria")
            # Show which criteria most commonly failed
            for stock in failed[:20]:  # Limit to first 20
                failed_criteria = []
                if not stock.get('c1_pass', True): failed_criteria.append("①")
                if not stock.get('c2_pass', True): failed_criteria.append("②")
                if not stock.get('c3_pass', True): failed_criteria.append("③")
                if not stock.get('c4_pass', True): failed_criteria.append("④")
                if not stock.get('c5_pass', True): failed_criteria.append("⑤")
                if not stock.get('c6_pass', True): failed_criteria.append("⑥")
                if not stock.get('c7_pass', True): failed_criteria.append("⑦")
                if not stock.get('c8_pass', True): failed_criteria.append("⑧")
                
                with st.expander(f"**{stock['symbol']}** - Failed: {', '.join(failed_criteria)}", expanded=False):
                    if "error" in stock:
                        st.error(stock["error"])
                    else:
                        cols = st.columns(4)
                        with cols[0]:
                            st.markdown(f"**①** {stock.get('c1_icon', '?')} | **②** {stock.get('c2_icon', '?')}")
                        with cols[1]:
                            st.markdown(f"**③** {stock.get('c3_icon', '?')} | **④** {stock.get('c4_icon', '?')}")
                        with cols[2]:
                            st.markdown(f"**⑤** {stock.get('c5_icon', '?')} | **⑥** {stock.get('c6_icon', '?')}")
                        with cols[3]:
                            st.markdown(f"**⑦** {stock.get('c7_icon', '?')} | **⑧** {stock.get('c8_icon', '?')}")
            
            if len(failed) > 20:
                st.caption(f"... and {len(failed) - 20} more")
        else:
            st.info("All stocks passed!")
    
    with tab3:
        # Create summary DataFrame
        df_data = []
        for r in all_results:
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
                "Result": "✅ PASS" if r.get("all_passed") else "❌ FAIL"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


# Main UI
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    universe = st.selectbox(
        "Stock Universe",
        ["NIFTY 50", "NIFTY 500"],
        index=0,
        key="trace_universe"
    )
    st.session_state["universe"] = universe.lower().replace(" ", "")

with col2:
    st.write("")
    st.write("")
    if st.button("🚀 Start Real-Time Scan", type="primary", use_container_width=True):
        run_realtime_trend_template_scan()

# Display results if available
display_results()


# Legend
st.markdown("---")
st.markdown("""
### 📖 Trend Template Criteria Legend

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
