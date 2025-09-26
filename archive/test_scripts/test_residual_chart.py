#!/usr/bin/env python3
"""
Test Residual Histogram/Distribution Chart
"""

import requests
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime


def fetch_residual_data(symbol="BTC", period="24h"):
    """Fetch residual data from API"""
    try:
        response = requests.get(
            f"http://localhost:8002/api/residuals?symbol={symbol}&period={period}"
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching residuals: {e}")
        return None


def create_residual_histogram(data, symbol="BTC"):
    """Create residual histogram chart"""
    if not data or not data.get("histogram"):
        print("No histogram data available")
        return None

    # Extract histogram data
    histogram = data["histogram"]
    stats = data.get("stats", {})

    bin_centers = [item["bin_center"] for item in histogram]
    counts = [item["count"] for item in histogram]

    # Create histogram
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            name=f"{symbol} Residuals",
            marker_color="rgba(55, 128, 191, 0.7)",
            marker_line=dict(color="rgba(55, 128, 191, 1.0)", width=1),
        )
    )

    # Add vertical line for mean
    if "mean" in stats:
        fig.add_vline(
            x=stats["mean"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {stats['mean']:.2f}",
        )

    # Add vertical line for median
    if "median" in stats:
        fig.add_vline(
            x=stats["median"],
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {stats['median']:.2f}",
        )

    # Update layout
    fig.update_layout(
        title=f"{symbol} Residual Distribution (Market - Model)",
        xaxis_title="Residual Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Stats: μ={stats.get('mean', 0):.2f}, σ={stats.get('std', 0):.2f}, n={stats.get('count', 0)}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
    )

    return fig


def create_residual_distribution(data, symbol="BTC"):
    """Create residual distribution with normal overlay"""
    if not data or not data.get("residuals"):
        print("No residual data available")
        return None

    residuals = np.array(data["residuals"])
    stats = data.get("stats", {})

    # Create distribution plot
    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name=f"{symbol} Residuals",
            opacity=0.7,
            marker_color="lightblue",
        )
    )

    # Overlay normal distribution
    if len(residuals) > 0:
        mean = np.mean(residuals)
        std = np.std(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = (
            len(residuals)
            * (residuals.max() - residuals.min())
            / 30
            * (1 / (std * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        )

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode="lines",
                name="Normal Distribution",
                line=dict(color="red", width=2),
            )
        )

    fig.update_layout(
        title=f"{symbol} Residual Distribution vs Normal",
        xaxis_title="Residual Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
    )

    return fig


def create_residual_qq_plot(data, symbol="BTC"):
    """Create Q-Q plot to check normality"""
    if not data or not data.get("residuals"):
        return None

    residuals = np.array(data["residuals"])

    # Create Q-Q plot using plotly
    from scipy import stats

    # Calculate theoretical quantiles (normal distribution)
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = go.Figure()

    # Add Q-Q scatter plot
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            name="Q-Q Plot",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    # Add reference line (y=x)
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Normal",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title=f"{symbol} Q-Q Plot (Normality Check)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        height=500,
    )

    return fig


def main():
    print("🚀 Testing Residual Charts...")

    # Fetch data
    print("📊 Fetching residual data...")
    btc_data = fetch_residual_data("BTC", "24h")

    if btc_data:
        print("✅ Residual data fetched successfully")
        print(f"Stats: {btc_data.get('stats', {})}")

        # Create histogram
        hist_fig = create_residual_histogram(btc_data, "BTC")
        if hist_fig:
            hist_fig.write_html("residual_histogram.html")
            print("✅ Residual histogram saved as residual_histogram.html")

        # Create distribution plot
        dist_fig = create_residual_distribution(btc_data, "BTC")
        if dist_fig:
            dist_fig.write_html("residual_distribution.html")
            print("✅ Residual distribution saved as residual_distribution.html")

        # Create Q-Q plot (requires scipy)
        try:
            qq_fig = create_residual_qq_plot(btc_data, "BTC")
            if qq_fig:
                qq_fig.write_html("residual_qq_plot.html")
                print("✅ Q-Q plot saved as residual_qq_plot.html")
        except ImportError:
            print("⚠️  scipy not available, skipping Q-Q plot")

    else:
        print("❌ No residual data available")


if __name__ == "__main__":
    main()
