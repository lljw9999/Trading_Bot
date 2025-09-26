#!/usr/bin/env python3
"""
Test PnL Curve Multi-Timeframe Chart
"""

import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime


def fetch_pnl_data(timeframe="24h"):
    """Fetch PnL curve data from API"""
    try:
        response = requests.get(
            f"http://localhost:8002/api/pnl-curve?timeframe={timeframe}"
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching PnL data: {e}")
        return None


def create_pnl_curve_chart(data, timeframe="24h"):
    """Create PnL curve chart with performance metrics"""
    if not data or not data.get("data"):
        print("No PnL data available")
        return None

    df = pd.DataFrame(data["data"])
    metrics = data.get("metrics", {})

    # Convert timestamps to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Create subplots: PnL curve and drawdown
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(f"Portfolio Equity Curve ({timeframe})", "Drawdown"),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # Add total equity curve
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["equity"],
            mode="lines",
            name="Total Equity",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Add BTC PnL component
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["btc_pnl"],
            mode="lines",
            name="BTC PnL",
            line=dict(color="orange", width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Add ETH PnL component
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["eth_pnl"],
            mode="lines",
            name="ETH PnL",
            line=dict(color="purple", width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Calculate and add drawdown
    equity_values = df["equity"].values
    peak = pd.Series(equity_values).expanding().max()
    drawdown = (df["equity"] - peak) / peak * 100

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=drawdown,
            mode="lines",
            name="Drawdown %",
            line=dict(color="red", width=1),
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.1)",
        ),
        row=2,
        col=1,
    )

    # Add zero line for drawdown
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f"Portfolio Performance Analysis - {timeframe.upper()}",
        height=700,
        template="plotly_white",
        showlegend=True,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"""Performance Metrics:
• Total Return: {metrics.get('total_return', 0):.2%}
• Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
• Win Rate: {metrics.get('win_rate', 0):.2%}
• Volatility: {metrics.get('volatility', 0):.2%}""",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),
            )
        ],
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_multi_timeframe_dashboard():
    """Create dashboard with multiple timeframes"""
    timeframes = ["1h", "24h", "7d", "30d"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"PnL Curve - {tf.upper()}" for tf in timeframes],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    colors = ["blue", "green", "orange", "red"]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for i, (timeframe, color, pos) in enumerate(zip(timeframes, colors, positions)):
        data = fetch_pnl_data(timeframe)
        if data and data.get("data"):
            df = pd.DataFrame(data["data"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["equity"],
                    mode="lines",
                    name=f"{timeframe.upper()}",
                    line=dict(color=color, width=2),
                ),
                row=pos[0],
                col=pos[1],
            )

            # Add metrics to trace name instead
            metrics = data.get("metrics", {})
            fig.update_traces(
                name=f'{timeframe.upper()} (R:{metrics.get("total_return", 0):.1%})',
                row=pos[0],
                col=pos[1],
            )

    fig.update_layout(
        title="Multi-Timeframe PnL Analysis",
        height=600,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def main():
    print("🚀 Testing PnL Curve Charts...")

    # Test individual timeframes
    timeframes = ["24h", "7d", "30d"]

    for tf in timeframes:
        print(f"📊 Fetching {tf} PnL data...")
        data = fetch_pnl_data(tf)

        if data:
            print(f"✅ {tf} data fetched - {data.get('count', 0)} points")
            metrics = data.get("metrics", {})
            print(
                f"   Metrics: Return {metrics.get('total_return', 0):.2%}, "
                f"MaxDD {metrics.get('max_drawdown', 0):.2%}, "
                f"Sharpe {metrics.get('sharpe_ratio', 0):.2f}"
            )

            # Create chart
            fig = create_pnl_curve_chart(data, tf)
            if fig:
                filename = f"pnl_curve_{tf}.html"
                fig.write_html(filename)
                print(f"✅ Chart saved as {filename}")
        else:
            print(f"❌ No data for {tf}")

    # Create multi-timeframe dashboard
    print("📊 Creating multi-timeframe dashboard...")
    multi_fig = create_multi_timeframe_dashboard()
    if multi_fig:
        multi_fig.write_html("pnl_multi_timeframe.html")
        print("✅ Multi-timeframe dashboard saved as pnl_multi_timeframe.html")


if __name__ == "__main__":
    main()
