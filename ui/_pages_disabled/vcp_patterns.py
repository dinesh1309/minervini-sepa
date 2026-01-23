"""
VCP Patterns Analysis Page

Dedicated page for viewing and analyzing VCP patterns in stocks.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="VCP Patterns", page_icon="📊", layout="wide")

st.title("📊 VCP Pattern Analysis")
st.markdown("Analyze Volatility Contraction Patterns in individual stocks")

# Symbol input
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter Stock Symbol", value="TITAN", placeholder="e.g., RELIANCE, TCS")
with col2:
    period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y"], index=1)

if st.button("🔍 Analyze Pattern", type="primary"):
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            from src.tools import get_stock_data, detect_vcp, calculate_moving_averages
            from ui.components.charts import create_vcp_chart
            
            # Get data
            df = get_stock_data(symbol, period=period)
            
            if df is None or df.empty:
                st.error(f"Could not fetch data for {symbol}")
            else:
                # Add SMAs
                df = calculate_moving_averages(df)
                
                # Detect VCP
                vcp = detect_vcp(df, symbol=symbol)
                
                # Display results
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("VCP Valid", "✅ Yes" if vcp.is_valid else "❌ No")
                with col2:
                    st.metric("Quality Score", f"{vcp.score}/100")
                with col3:
                    st.metric("Pivot Price", f"₹{vcp.pivot_price:,.2f}" if vcp.pivot_price else "—")
                with col4:
                    st.metric("Base Length", f"{vcp.base_length_days} days")
                
                # Chart
                contractions = [
                    {"end_idx": c.end_idx, "high": c.high, "depth_pct": c.depth_pct}
                    for c in vcp.contractions
                ] if vcp.contractions else None
                
                fig = create_vcp_chart(
                    df.tail(90),  # Last 90 days
                    symbol,
                    contractions=contractions,
                    pivot_price=vcp.pivot_price
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Details
                with st.expander("📋 Pattern Details"):
                    st.write(f"**Details:** {vcp.details}")
                    st.write(f"**Volume Dry-up:** {'Yes' if vcp.volume_dry_up else 'No'}")
                    st.write(f"**Contractions:** {len(vcp.contractions)}")
                    
                    if vcp.contractions:
                        for i, c in enumerate(vcp.contractions):
                            st.write(f"  - T{i+1}: {c.depth_pct:.1f}% depth")
                
        except ImportError as e:
            st.error(f"Import error: {e}. Please install dependencies: pip install -r requirements.txt")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Enter a stock symbol and click 'Analyze Pattern' to view VCP analysis.")
