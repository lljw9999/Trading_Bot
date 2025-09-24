#!/usr/bin/env python3
"""
Test Entropy & Q-Spread Time-Series Mini-Charts
"""

import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime


def fetch_entropy_qspread_data():
    """Fetch entropy and Q-spread data from API"""
    try:
        response = requests.get("http://localhost:8002/api/entropy-qspread")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching entropy/Q-spread data: {e}")
        return None


def create_entropy_sparkline(entropy_data):
    """Create compact entropy sparkline chart"""
    if not entropy_data:
        return None

    df = pd.DataFrame(entropy_data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    fig = go.Figure()

    # Add entropy line
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["value"],
            mode="lines",
            name="Policy Entropy",
            line=dict(color="blue", width=2),
            fill="tonexty",
            fillcolor="rgba(0,100,250,0.1)",
        )
    )

    # Add warning zone (entropy < 0.3 indicates policy collapse risk)
    fig.add_hline(
        y=0.3, line_dash="dash", line_color="red", annotation_text="Collapse Risk"
    )
    fig.add_hline(
        y=0.1, line_dash="dot", line_color="darkred", annotation_text="Critical"
    )

    fig.update_layout(
        title="Policy Entropy (Last Hour)",
        height=200,
        margin=dict(l=30, r=10, t=30, b=30),
        showlegend=False,
        template="plotly_white",
        yaxis_title="Entropy",
    )

    return fig


def create_qspread_sparkline(qspread_data):
    """Create compact Q-spread sparkline chart"""
    if not qspread_data:
        return None

    df = pd.DataFrame(qspread_data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    fig = go.Figure()

    # Add Q-spread line
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["value"],
            mode="lines",
            name="Q-Value Spread",
            line=dict(color="green", width=2),
            fill="tonexty",
            fillcolor="rgba(0,200,100,0.1)",
        )
    )

    # Add mean line
    mean_val = df["value"].mean()
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Mean: {mean_val:.1f}",
    )

    fig.update_layout(
        title="Q-Value Spread (Last Hour)",
        height=200,
        margin=dict(l=30, r=10, t=30, b=30),
        showlegend=False,
        template="plotly_white",
        yaxis_title="Q-Spread",
    )

    return fig


def create_combined_policy_monitor(data):
    """Create combined policy monitoring dashboard"""
    if not data:
        return None

    entropy_data = data.get("entropy_series", [])
    qspread_data = data.get("qspread_series", [])
    stats = data.get("stats", {})

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Policy Entropy (Collapse Detection)", "Q-Value Spread"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Entropy plot
    if entropy_data:
        entropy_df = pd.DataFrame(entropy_data)
        entropy_df["datetime"] = pd.to_datetime(entropy_df["timestamp"], unit="s")

        fig.add_trace(
            go.Scatter(
                x=entropy_df["datetime"],
                y=entropy_df["value"],
                mode="lines+markers",
                name="Entropy",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=1,
        )

        # Add warning zones
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0.1, line_dash="dot", line_color="darkred", row=1, col=1)

    # Q-spread plot
    if qspread_data:
        qspread_df = pd.DataFrame(qspread_data)
        qspread_df["datetime"] = pd.to_datetime(qspread_df["timestamp"], unit="s")

        fig.add_trace(
            go.Scatter(
                x=qspread_df["datetime"],
                y=qspread_df["value"],
                mode="lines+markers",
                name="Q-Spread",
                line=dict(color="green", width=2),
                marker=dict(size=4),
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title="RL Policy Monitoring Dashboard",
        height=500,
        template="plotly_white",
        showlegend=False,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"""Current Status:
• Entropy: {stats.get('entropy', {}).get('current', 0):.3f}
• Collapse Risk: {stats.get('entropy', {}).get('policy_collapse_risk', 'UNKNOWN')}
• Q-Spread: {stats.get('qspread', {}).get('current', 0):.1f}
• Data Points: {data.get('count', 0)}""",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),
            )
        ],
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Entropy", row=1, col=1)
    fig.update_yaxes(title_text="Q-Spread", row=2, col=1)

    return fig


def create_policy_status_gauge(stats):
    """Create gauge chart for policy health status"""
    if not stats or "entropy" not in stats:
        return None

    entropy_current = stats["entropy"].get("current", 0)
    collapse_risk = stats["entropy"].get("policy_collapse_risk", "UNKNOWN")

    # Map entropy to health score (0-100)
    health_score = min(100, max(0, entropy_current * 50))  # Scale 0-2 entropy to 0-100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Policy Health<br>Risk: {collapse_risk}"},
            delta={"reference": 75},  # Target health score
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 25], "color": "red"},
                    {"range": [25, 50], "color": "orange"},
                    {"range": [50, 75], "color": "yellow"},
                    {"range": [75, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        height=300, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def main():
    print("🚀 Testing Entropy & Q-Spread Charts...")

    # Fetch data
    print("📊 Fetching entropy and Q-spread data...")
    data = fetch_entropy_qspread_data()

    if data:
        print("✅ Policy monitoring data fetched successfully")
        stats = data.get("stats", {})
        print(f"Current Entropy: {stats.get('entropy', {}).get('current', 0):.3f}")
        print(
            f"Collapse Risk: {stats.get('entropy', {}).get('policy_collapse_risk', 'UNKNOWN')}"
        )
        print(f"Current Q-Spread: {stats.get('qspread', {}).get('current', 0):.1f}")

        # Create individual sparklines
        entropy_fig = create_entropy_sparkline(data.get("entropy_series", []))
        if entropy_fig:
            entropy_fig.write_html("entropy_sparkline.html")
            print("✅ Entropy sparkline saved as entropy_sparkline.html")

        qspread_fig = create_qspread_sparkline(data.get("qspread_series", []))
        if qspread_fig:
            qspread_fig.write_html("qspread_sparkline.html")
            print("✅ Q-spread sparkline saved as qspread_sparkline.html")

        # Create combined monitoring dashboard
        combined_fig = create_combined_policy_monitor(data)
        if combined_fig:
            combined_fig.write_html("policy_monitor_dashboard.html")
            print(
                "✅ Policy monitoring dashboard saved as policy_monitor_dashboard.html"
            )

        # Create policy health gauge
        gauge_fig = create_policy_status_gauge(stats)
        if gauge_fig:
            gauge_fig.write_html("policy_health_gauge.html")
            print("✅ Policy health gauge saved as policy_health_gauge.html")

    else:
        print("❌ No policy monitoring data available")


if __name__ == "__main__":
    main()
