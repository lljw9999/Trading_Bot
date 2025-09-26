#!/usr/bin/env python3
"""
Test Prediction vs Market Chart with Plotly
"""

import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import json


def fetch_model_data(symbol="BTC"):
    """Fetch model vs market data from API"""
    try:
        response = requests.get(
            f"http://localhost:8002/api/model-price-series?symbol={symbol}"
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def create_prediction_chart(data, symbol="BTC"):
    """Create prediction vs market chart with confidence bands"""
    if not data or not data.get("data"):
        print("No data available")
        return None

    df = pd.DataFrame(data["data"])

    # Convert timestamps to datetime
    df["datetime"] = pd.to_datetime(df["ts"], unit="s")

    # Create the plot
    fig = go.Figure()

    # Add confidence band (fill area)
    fig.add_trace(
        go.Scatter(
            x=list(df["datetime"]) + list(df["datetime"][::-1]),
            y=list(df["ci_high"]) + list(df["ci_low"][::-1]),
            fill="toself",
            fillcolor="rgba(0,100,250,0.2)",
            line=dict(width=0),
            name="Confidence Interval",
            showlegend=True,
        )
    )

    # Add market price (solid line)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["market"],
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name=f"{symbol} Market Price",
        )
    )

    # Add model prediction (dotted line)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["model"],
            mode="lines",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            name="Model Prediction",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Prediction vs Market Price",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        margin=dict(l=30, r=10, t=40, b=30),
        legend=dict(orientation="h", y=-0.2),
        height=500,
        template="plotly_white",
    )

    return fig


def create_dashboard_html(btc_data, eth_data):
    """Create a complete dashboard HTML with both charts"""

    btc_fig = create_prediction_chart(btc_data, "BTC")
    eth_fig = create_prediction_chart(eth_data, "ETH")

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Bitcoin Prediction vs Market",
            "Ethereum Prediction vs Market",
        ),
        vertical_spacing=0.08,
    )

    if btc_data and btc_data.get("data"):
        btc_df = pd.DataFrame(btc_data["data"])
        btc_df["datetime"] = pd.to_datetime(btc_df["ts"], unit="s")

        # Add BTC traces
        fig.add_trace(
            go.Scatter(
                x=btc_df["datetime"],
                y=btc_df["market"],
                mode="lines",
                name="BTC Market",
                line=dict(color="#f7931a"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=btc_df["datetime"],
                y=btc_df["model"],
                mode="lines",
                name="BTC Model",
                line=dict(dash="dot", color="#f7931a"),
            ),
            row=1,
            col=1,
        )

    if eth_data and eth_data.get("data"):
        eth_df = pd.DataFrame(eth_data["data"])
        eth_df["datetime"] = pd.to_datetime(eth_df["ts"], unit="s")

        # Add ETH traces
        fig.add_trace(
            go.Scatter(
                x=eth_df["datetime"],
                y=eth_df["market"],
                mode="lines",
                name="ETH Market",
                line=dict(color="#627eea"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=eth_df["datetime"],
                y=eth_df["model"],
                mode="lines",
                name="ETH Model",
                line=dict(dash="dot", color="#627eea"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=800,
        title_text="Crypto Prediction Dashboard",
        showlegend=True,
        template="plotly_white",
    )

    # Save as HTML
    fig.write_html("prediction_dashboard.html")
    print("✅ Dashboard saved as prediction_dashboard.html")

    return fig


def main():
    print("🚀 Testing Prediction Chart...")

    # Start the dashboard server first if not running
    print("Make sure the dashboard is running at http://localhost:8002")

    # Fetch data for both symbols
    print("📊 Fetching BTC data...")
    btc_data = fetch_model_data("BTC")

    print("📊 Fetching ETH data...")
    eth_data = fetch_model_data("ETH")

    if btc_data or eth_data:
        print("✅ Data fetched successfully")

        # Create individual charts
        if btc_data:
            btc_fig = create_prediction_chart(btc_data, "BTC")
            if btc_fig:
                btc_fig.write_html("btc_prediction.html")
                print("✅ BTC chart saved as btc_prediction.html")

        if eth_data:
            eth_fig = create_prediction_chart(eth_data, "ETH")
            if eth_fig:
                eth_fig.write_html("eth_prediction.html")
                print("✅ ETH chart saved as eth_prediction.html")

        # Create combined dashboard
        create_dashboard_html(btc_data, eth_data)

    else:
        print("❌ No data available. Make sure dashboard is running.")


if __name__ == "__main__":
    main()
