"""
Chart Components for SEPA Dashboard

Provides reusable chart components for:
- Stock price charts with moving averages
- VCP pattern visualization
- Volume analysis
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, List


def create_price_chart(
    df: pd.DataFrame,
    symbol: str,
    show_smas: bool = True,
    show_volume: bool = True,
    height: int = 500
) -> go.Figure:
    """
    Create a candlestick chart with moving averages and volume.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol for title
        show_smas: Whether to show SMA lines
        show_volume: Whether to show volume subplot
        height: Chart height in pixels
    
    Returns:
        Plotly Figure object
    """
    # Create subplot with volume
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00d26a',
            decreasing_line_color='#ff4757'
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None
    )
    
    # Add SMAs if present
    if show_smas:
        sma_colors = {
            'SMA_50': '#ffd700',
            'SMA_150': '#00bcd4',
            'SMA_200': '#e91e63'
        }
        
        for col, color in sma_colors.items():
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=col.replace('_', ' '),
                        line=dict(color=color, width=1.5)
                    ),
                    row=1 if show_volume else None,
                    col=1 if show_volume else None
                )
    
    # Volume bars
    if show_volume:
        colors = ['#00d26a' if c >= o else '#ff4757' 
                  for c, o in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Layout
    fig.update_layout(
        title=f"{symbol} - Price Chart",
        height=height,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_vcp_chart(
    df: pd.DataFrame,
    symbol: str,
    contractions: Optional[List[Dict]] = None,
    pivot_price: Optional[float] = None,
    height: int = 450
) -> go.Figure:
    """
    Create a chart highlighting the VCP pattern.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        contractions: List of contraction data
        pivot_price: Pivot/breakout price
        height: Chart height
    
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=2)
        ),
        row=1, col=1
    )
    
    # Add high/low area
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['High'],
            mode='lines',
            name='High',
            line=dict(color='rgba(102, 126, 234, 0.3)', width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Low'],
            mode='lines',
            name='Low',
            line=dict(color='rgba(102, 126, 234, 0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add pivot line
    if pivot_price:
        fig.add_hline(
            y=pivot_price,
            line_dash="dash",
            line_color="#ffd700",
            annotation_text=f"Pivot: ₹{pivot_price:,.2f}",
            row=1, col=1
        )
    
    # Mark contractions
    if contractions:
        for i, c in enumerate(contractions):
            # Add annotation for each contraction
            fig.add_annotation(
                x=df.index[min(c.get('end_idx', 0), len(df)-1)],
                y=c.get('high', df['High'].max()),
                text=f"T{i+1}: {c.get('depth_pct', 0):.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ffd700",
                font=dict(color="#ffd700", size=10),
                row=1, col=1
            )
    
    # Volume with fade effect for dry-up
    avg_vol = df['Volume'].tail(50).mean()
    colors = [
        '#ffd700' if v < avg_vol * 0.5 else '#667eea'
        for v in df['Volume']
    ]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} - VCP Pattern Analysis",
        height=height,
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_metrics_gauge(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    threshold_good: float = 50,
    height: int = 200
) -> go.Figure:
    """
    Create a gauge chart for metrics like win rate.
    
    Args:
        value: Current value
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        threshold_good: Value above which is considered good
        height: Chart height
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [min_val, threshold_good], 'color': "rgba(255, 71, 87, 0.3)"},
                {'range': [threshold_good, max_val], 'color': "rgba(0, 210, 106, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#ffd700", 'width': 2},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def create_equity_curve(
    dates: List[str],
    equity: List[float],
    height: int = 400
) -> go.Figure:
    """
    Create an equity curve chart.
    
    Args:
        dates: List of date strings
        equity: List of equity values
        height: Chart height
    
    Returns:
        Plotly Figure object
    """
    df = pd.DataFrame({'Date': dates, 'Equity': equity})
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate drawdown
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00d26a', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 210, 106, 0.1)'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4757', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 71, 87, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        height=height,
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_yaxes(title_text="Equity (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig
