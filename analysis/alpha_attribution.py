#!/usr/bin/env python3
"""
Alpha Attribution & Decay Tracker
Compute per-alpha contribution to P&L, hit rate, turnover, decay half-life, and marginal Sharpe.
Tag alphas as boost/fade/pause based on performance.
"""
import os
import sys
import json
import datetime
import pathlib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.stats import linregress


@dataclass
class AlphaMetrics:
    """Alpha performance metrics."""

    name: str
    pnl_contribution: float
    hit_rate: float
    turnover: float
    decay_half_life_days: float
    marginal_sharpe: float
    signal_mass_pct: float
    action: str  # boost/fade/pause
    reason: str


class AlphaAttributor:
    def __init__(self, window_days: int = 14):
        self.window_days = window_days
        self.cutoff_date = datetime.datetime.now() - datetime.timedelta(
            days=window_days
        )

    def load_signal_data(self) -> Dict[str, pd.DataFrame]:
        """Load signal data for all alpha models."""
        # Mock signal data - in production would load from Redis/DB
        signal_data = {}

        alpha_names = [
            "ma_momentum",
            "mean_rev",
            "momo_fast",
            "news_sent_alpha",
            "ob_pressure",
            "big_bet_flag",
        ]

        # Generate synthetic signal history
        dates = pd.date_range(
            end=datetime.datetime.now(), periods=self.window_days * 24, freq="H"
        )

        for alpha in alpha_names:
            # Different alpha characteristics
            if alpha == "ma_momentum":
                # Trending alpha - good recent performance
                signals = np.random.normal(0.1, 0.3, len(dates))
                decay_factor = 0.99  # Slow decay
            elif alpha == "mean_rev":
                # Mean reversion - deteriorating
                signals = np.random.normal(0.05, 0.4, len(dates)) * np.exp(
                    -np.arange(len(dates)) / 100
                )
                decay_factor = 0.95  # Fast decay
            elif alpha == "news_sent_alpha":
                # Sentiment - volatile but profitable
                signals = np.random.normal(0.08, 0.5, len(dates))
                decay_factor = 0.97
            elif alpha == "ob_pressure":
                # Order book - consistent but small edge
                signals = np.random.normal(0.03, 0.2, len(dates))
                decay_factor = 0.98
            elif alpha == "big_bet_flag":
                # Big bet - infrequent but high impact
                signals = np.random.choice([0, 0, 0, 0.5, -0.3], len(dates))
                decay_factor = 0.96
            else:  # momo_fast
                # Fast momentum - noisy, losing edge
                signals = np.random.normal(-0.02, 0.35, len(dates))
                decay_factor = 0.93

            # Apply decay over time
            decay_multiplier = np.power(decay_factor, np.arange(len(dates)))
            signals = signals * decay_multiplier

            signal_data[alpha] = pd.DataFrame(
                {
                    "timestamp": dates,
                    "signal": signals,
                    "signal_abs": np.abs(signals),
                    "realized_pnl": signals
                    * np.random.normal(0.8, 0.2, len(dates)),  # Imperfect realization
                }
            )

        return signal_data

    def compute_pnl_contribution(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute each alpha's contribution to total P&L."""
        contributions = {}
        total_pnl = 0

        for alpha, df in signal_data.items():
            alpha_pnl = df["realized_pnl"].sum()
            contributions[alpha] = alpha_pnl
            total_pnl += alpha_pnl

        # Normalize to percentages
        if total_pnl != 0:
            contributions = {k: (v / total_pnl) * 100 for k, v in contributions.items()}

        return contributions

    def compute_hit_rate(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute hit rate (% of signals that are profitable)."""
        hit_rates = {}

        for alpha, df in signal_data.items():
            # Hit = signal and realized PnL have same sign
            hits = ((df["signal"] > 0) & (df["realized_pnl"] > 0)) | (
                (df["signal"] < 0) & (df["realized_pnl"] < 0)
            )
            hit_rates[alpha] = hits.mean() if len(df) > 0 else 0

        return hit_rates

    def compute_turnover(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute signal turnover (average absolute signal change)."""
        turnovers = {}

        for alpha, df in signal_data.items():
            if len(df) > 1:
                signal_changes = df["signal"].diff().abs()
                turnovers[alpha] = signal_changes.mean()
            else:
                turnovers[alpha] = 0

        return turnovers

    def compute_decay_half_life(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Estimate decay half-life by fitting exponential decay to signal correlation."""
        half_lives = {}

        for alpha, df in signal_data.items():
            if len(df) < 48:  # Need at least 48 hours
                half_lives[alpha] = 999  # Very high = no decay detected
                continue

            try:
                # Compute rolling correlation of signal with 24h-lagged signal
                signals = df["signal"].values
                correlations = []

                for lag in range(1, min(len(signals) // 2, 168)):  # Up to 1 week lag
                    if lag < len(signals):
                        corr = np.corrcoef(signals[:-lag], signals[lag:])[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)

                if not correlations:
                    half_lives[alpha] = 999
                    continue

                # Fit exponential decay: corr(t) = exp(-t/τ)
                lags = np.arange(1, len(correlations) + 1)
                log_corrs = np.log(np.maximum(correlations, 1e-6))  # Avoid log(0)

                if len(log_corrs) > 3:
                    slope, _, _, _, _ = linregress(lags, log_corrs)
                    tau = -1 / slope if slope < 0 else 999
                    half_life_hours = tau * np.log(2)
                    half_lives[alpha] = max(
                        0.1, half_life_hours / 24
                    )  # Convert to days
                else:
                    half_lives[alpha] = 999

            except Exception:
                half_lives[alpha] = 999

        return half_lives

    def compute_marginal_sharpe(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute marginal Sharpe ratio (incremental Sharpe when alpha is added)."""
        marginal_sharpes = {}

        # Combine all alphas for portfolio Sharpe
        all_pnl = pd.DataFrame()
        for alpha, df in signal_data.items():
            all_pnl[alpha] = df["realized_pnl"]

        if all_pnl.empty:
            return {alpha: 0 for alpha in signal_data.keys()}

        # Portfolio without each alpha
        portfolio_pnl = all_pnl.sum(axis=1)
        portfolio_sharpe = (
            portfolio_pnl.mean() / (portfolio_pnl.std() + 1e-6) * np.sqrt(24 * 365)
        )  # Annualized

        for alpha in signal_data.keys():
            # Portfolio excluding this alpha
            exclude_pnl = all_pnl.drop(columns=[alpha]).sum(axis=1)
            exclude_sharpe = (
                exclude_pnl.mean() / (exclude_pnl.std() + 1e-6) * np.sqrt(24 * 365)
            )

            # Marginal contribution
            marginal_sharpes[alpha] = portfolio_sharpe - exclude_sharpe

        return marginal_sharpes

    def compute_signal_mass(
        self, signal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Compute each alpha's share of total signal mass."""
        signal_masses = {}
        total_mass = 0

        for alpha, df in signal_data.items():
            alpha_mass = df["signal_abs"].sum()
            signal_masses[alpha] = alpha_mass
            total_mass += alpha_mass

        # Convert to percentages
        if total_mass > 0:
            signal_masses = {
                k: (v / total_mass) * 100 for k, v in signal_masses.items()
            }

        return signal_masses

    def determine_alpha_actions(
        self, metrics: List[AlphaMetrics]
    ) -> List[AlphaMetrics]:
        """Determine boost/fade/pause actions for each alpha."""
        # Thresholds (configurable)
        BOOST_SHARPE_THRESHOLD = 0.1
        FADE_SHARPE_THRESHOLD = -0.05
        MIN_DECAY_DAYS = 3
        MIN_HIT_RATE = 0.48

        for metric in metrics:
            reasons = []

            # Decision logic
            if (
                metric.marginal_sharpe >= BOOST_SHARPE_THRESHOLD
                and metric.decay_half_life_days >= MIN_DECAY_DAYS
                and metric.hit_rate >= MIN_HIT_RATE
            ):
                action = "boost"
                reasons.append(f"High Sharpe ({metric.marginal_sharpe:.3f})")

            elif (
                metric.marginal_sharpe <= FADE_SHARPE_THRESHOLD
                or metric.decay_half_life_days < MIN_DECAY_DAYS
                or metric.hit_rate < MIN_HIT_RATE - 0.05
            ):

                if metric.decay_half_life_days < 1:  # Very fast decay
                    action = "pause"
                    reasons.append(f"Rapid decay ({metric.decay_half_life_days:.1f}d)")
                else:
                    action = "fade"
                    reasons.append(f"Low Sharpe ({metric.marginal_sharpe:.3f})")

            else:
                action = "hold"
                reasons.append("Marginal performance")

            # Additional checks
            if metric.hit_rate < MIN_HIT_RATE:
                reasons.append(f"Low hit rate ({metric.hit_rate:.1%})")
            if metric.signal_mass_pct > 50:
                reasons.append("High signal mass - diversify")

            metric.action = action
            metric.reason = "; ".join(reasons)

        return metrics

    def generate_param_server_hints(
        self, metrics: List[AlphaMetrics]
    ) -> Dict[str, float]:
        """Generate parameter server weight hints."""
        hints = {}

        for metric in metrics:
            base_weight = 1.0

            if metric.action == "boost":
                weight_multiplier = 1.5
            elif metric.action == "fade":
                weight_multiplier = 0.5
            elif metric.action == "pause":
                weight_multiplier = 0.1
            else:  # hold
                weight_multiplier = 1.0

            # Adjust by marginal Sharpe
            sharpe_adjustment = 1 + np.clip(metric.marginal_sharpe, -0.5, 0.5)

            hints[metric.name] = base_weight * weight_multiplier * sharpe_adjustment

        return hints

    def run_attribution(self) -> Tuple[List[AlphaMetrics], Dict[str, Any]]:
        """Run full alpha attribution analysis."""
        print(f"📊 Running alpha attribution analysis ({self.window_days}d window)...")

        # Load data
        signal_data = self.load_signal_data()
        print(f"   Loaded data for {len(signal_data)} alphas")

        # Compute all metrics
        pnl_contribs = self.compute_pnl_contribution(signal_data)
        hit_rates = self.compute_hit_rate(signal_data)
        turnovers = self.compute_turnover(signal_data)
        half_lives = self.compute_decay_half_life(signal_data)
        marginal_sharpes = self.compute_marginal_sharpe(signal_data)
        signal_masses = self.compute_signal_mass(signal_data)

        # Create metrics objects
        metrics = []
        for alpha in signal_data.keys():
            metric = AlphaMetrics(
                name=alpha,
                pnl_contribution=pnl_contribs.get(alpha, 0),
                hit_rate=hit_rates.get(alpha, 0),
                turnover=turnovers.get(alpha, 0),
                decay_half_life_days=half_lives.get(alpha, 999),
                marginal_sharpe=marginal_sharpes.get(alpha, 0),
                signal_mass_pct=signal_masses.get(alpha, 0),
                action="hold",  # Will be set by determine_alpha_actions
                reason="",
            )
            metrics.append(metric)

        # Determine actions
        metrics = self.determine_alpha_actions(metrics)

        # Generate parameter hints
        param_hints = self.generate_param_server_hints(metrics)

        # Summary stats
        summary = {
            "analysis_window_days": self.window_days,
            "total_alphas": len(metrics),
            "actions": {
                "boost": len([m for m in metrics if m.action == "boost"]),
                "fade": len([m for m in metrics if m.action == "fade"]),
                "pause": len([m for m in metrics if m.action == "pause"]),
                "hold": len([m for m in metrics if m.action == "hold"]),
            },
            "avg_marginal_sharpe": np.mean([m.marginal_sharpe for m in metrics]),
            "avg_hit_rate": np.mean([m.hit_rate for m in metrics]),
            "avg_decay_days": np.mean(
                [
                    m.decay_half_life_days
                    for m in metrics
                    if m.decay_half_life_days < 999
                ]
            ),
        }

        return metrics, {"summary": summary, "param_hints": param_hints}


def generate_report_json(
    metrics: List[AlphaMetrics], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate JSON report."""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "analysis_metadata": metadata,
        "alpha_metrics": [
            {
                "name": m.name,
                "pnl_contribution_pct": round(m.pnl_contribution, 2),
                "hit_rate": round(m.hit_rate, 4),
                "turnover": round(m.turnover, 6),
                "decay_half_life_days": round(m.decay_half_life_days, 2),
                "marginal_sharpe": round(m.marginal_sharpe, 4),
                "signal_mass_pct": round(m.signal_mass_pct, 2),
                "action": m.action,
                "reason": m.reason,
            }
            for m in metrics
        ],
    }


def generate_report_markdown(
    metrics: List[AlphaMetrics], metadata: Dict[str, Any]
) -> str:
    """Generate markdown report."""
    summary = metadata["summary"]

    # Sort by marginal Sharpe descending
    sorted_metrics = sorted(metrics, key=lambda x: x.marginal_sharpe, reverse=True)

    md = f"""# Alpha Attribution Analysis

**Analysis Date:** {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Window:** {summary['analysis_window_days']} days  
**Total Alphas:** {summary['total_alphas']}

## Executive Summary

**Portfolio Health:**
- **Average Marginal Sharpe:** {summary['avg_marginal_sharpe']:.3f}
- **Average Hit Rate:** {summary['avg_hit_rate']:.1%}
- **Average Decay Half-Life:** {summary['avg_decay_days']:.1f} days

**Action Summary:**
- **🚀 Boost:** {summary['actions']['boost']} alphas
- **📉 Fade:** {summary['actions']['fade']} alphas  
- **⏸️ Pause:** {summary['actions']['pause']} alphas
- **➖ Hold:** {summary['actions']['hold']} alphas

## Alpha Performance Ranking

| Rank | Alpha | Sharpe | Hit Rate | Decay (days) | Signal Mass | Action | Reason |
|------|-------|--------|----------|--------------|-------------|--------|---------|
"""

    for i, m in enumerate(sorted_metrics, 1):
        action_emoji = {"boost": "🚀", "fade": "📉", "pause": "⏸️", "hold": "➖"}[
            m.action
        ]
        md += f"| {i} | {m.name} | {m.marginal_sharpe:.3f} | {m.hit_rate:.1%} | {m.decay_half_life_days:.1f} | {m.signal_mass_pct:.1f}% | {action_emoji} {m.action.upper()} | {m.reason} |\n"

    md += f"""

## Detailed Metrics

### Performance Breakdown
"""

    for m in sorted_metrics:
        status = (
            "✅"
            if m.action in ["boost", "hold"]
            else "⚠️" if m.action == "fade" else "❌"
        )

        md += f"""
### {status} {m.name}

- **P&L Contribution:** {m.pnl_contribution:+.2f}%
- **Hit Rate:** {m.hit_rate:.1%}
- **Signal Turnover:** {m.turnover:.4f}
- **Decay Half-Life:** {m.decay_half_life_days:.2f} days
- **Marginal Sharpe:** {m.marginal_sharpe:+.4f}
- **Signal Mass:** {m.signal_mass_pct:.2f}%
- **Action:** **{m.action.upper()}**
- **Reason:** {m.reason}
"""

    md += """

## Parameter Server Hints

The following weight adjustments are recommended:

```json
"""
    md += json.dumps(metadata["param_hints"], indent=2)
    md += """
```

## Next Steps

1. **Implement boost recommendations** for high-Sharpe alphas
2. **Fade or pause laggards** to reduce noise
3. **Monitor decay patterns** for early warning of alpha degradation
4. **Re-run analysis** in 7 days to track improvements

---
*Generated by Alpha Attribution & Decay Tracker - M11 Alpha Uplift Program*
"""

    return md


def main():
    """Main alpha attribution function."""
    parser = argparse.ArgumentParser(description="Alpha Attribution & Decay Tracker")
    parser.add_argument(
        "--window", default="14d", help="Analysis window (e.g., 14d, 30d)"
    )
    parser.add_argument(
        "--out", default="artifacts/alpha_attr", help="Output directory"
    )
    args = parser.parse_args()

    # Parse window
    window_str = args.window.lower()
    if window_str.endswith("d"):
        window_days = int(window_str[:-1])
    else:
        window_days = int(window_str)

    try:
        # Create output directory
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        output_dir = Path(args.out) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run attribution analysis
        attributor = AlphaAttributor(window_days=window_days)
        metrics, metadata = attributor.run_attribution()

        # Generate reports
        json_report = generate_report_json(metrics, metadata)
        md_report = generate_report_markdown(metrics, metadata)

        # Save files
        json_path = output_dir / "report.json"
        md_path = output_dir / "report.md"

        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)

        with open(md_path, "w") as f:
            f.write(md_report)

        # Create latest symlinks
        latest_json = Path(args.out) / "report_latest.json"
        latest_md = Path(args.out) / "report_latest.md"

        for latest, target in [(latest_json, json_path), (latest_md, md_path)]:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(target)

        # Display summary
        summary = metadata["summary"]
        print(f"\n📊 Alpha Attribution Complete:")
        print(
            f"  Boost: {summary['actions']['boost']} | Fade: {summary['actions']['fade']} | Pause: {summary['actions']['pause']}"
        )
        print(f"  Avg Marginal Sharpe: {summary['avg_marginal_sharpe']:.3f}")
        print(f"  Avg Hit Rate: {summary['avg_hit_rate']:.1%}")

        boost_alphas = [m.name for m in metrics if m.action == "boost"]
        fade_alphas = [m.name for m in metrics if m.action in ["fade", "pause"]]

        if boost_alphas:
            print(f"  🚀 Boost: {', '.join(boost_alphas)}")
        if fade_alphas:
            print(f"  📉 Fade/Pause: {', '.join(fade_alphas)}")

        print(f"\n📄 Reports:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        return 0

    except Exception as e:
        print(f"❌ Alpha attribution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
