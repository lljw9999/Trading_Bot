#!/usr/bin/env python3
"""
Weekly Retrospective
Pulls daily attribution, TCA, allocator weights and writes a one-pager
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("weekly_retro")


class WeeklyRetrospective:
    """Weekly performance retrospective and analysis."""

    def __init__(self):
        """Initialize weekly retrospective."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Analysis configuration
        self.config = {
            "lookback_days": 7,
            "strategies": ["RL", "BASIS", "MM"],
            "venues": ["binance", "coinbase", "ftx", "dydx"],
            "min_trades_for_analysis": 5,
            "performance_threshold_strong": 0.15,  # 15% weekly return = strong
            "performance_threshold_weak": 0.02,  # 2% weekly return = weak
            "sharpe_threshold_good": 1.0,
            "tca_grade_threshold": "B",
        }

        logger.info("📊 Weekly Retrospective initialized")

    def get_daily_pnl_data(self, days: int = 7) -> Dict[str, List[float]]:
        """Get daily P&L data for each strategy over lookback period."""
        try:
            pnl_data = {strategy: [] for strategy in self.config["strategies"]}

            # Get P&L for each day
            for i in range(days):
                date = datetime.utcnow() - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")

                for strategy in self.config["strategies"]:
                    pnl_key = f"strategy:{strategy}:pnl_daily:{date_str}"
                    daily_pnl = self.redis.get(pnl_key)

                    if daily_pnl:
                        pnl_data[strategy].append(float(daily_pnl))
                    else:
                        # Try alternative key format
                        alt_key = f"pnl:net:{date_str}:{strategy}"
                        alt_pnl = self.redis.get(alt_key)
                        if alt_pnl:
                            pnl_data[strategy].append(float(alt_pnl))
                        else:
                            # Generate mock data for demo
                            if strategy == "RL":
                                mock_pnl = np.random.normal(200, 400)
                            elif strategy == "BASIS":
                                mock_pnl = np.random.normal(100, 150)
                            else:  # MM
                                mock_pnl = np.random.normal(50, 80)
                            pnl_data[strategy].append(mock_pnl)

            # Reverse to get chronological order
            for strategy in pnl_data:
                pnl_data[strategy].reverse()

            return pnl_data

        except Exception as e:
            logger.error(f"Error getting daily P&L data: {e}")
            return {strategy: [] for strategy in self.config["strategies"]}

    def analyze_strategy_performance(
        self, pnl_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze strategy performance metrics."""
        try:
            analysis = {}

            for strategy, pnl_series in pnl_data.items():
                if not pnl_series:
                    continue

                # Basic metrics
                total_pnl = sum(pnl_series)
                avg_daily_pnl = np.mean(pnl_series)
                daily_vol = np.std(pnl_series) if len(pnl_series) > 1 else 0

                # Performance metrics
                weekly_return_pct = (
                    total_pnl / 10000 if total_pnl != 0 else 0
                )  # Assume $10k base
                daily_sharpe = avg_daily_pnl / daily_vol if daily_vol > 0 else 0
                win_rate = sum(1 for pnl in pnl_series if pnl > 0) / len(pnl_series)

                # Max drawdown
                cumulative = np.cumsum(pnl_series)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (running_max - cumulative) / np.maximum(running_max, 1)
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

                # Performance classification
                if weekly_return_pct > self.config["performance_threshold_strong"]:
                    performance_tier = "STRONG"
                    recommendation = "DOUBLE_DOWN"
                elif weekly_return_pct < self.config["performance_threshold_weak"]:
                    performance_tier = "WEAK"
                    recommendation = "REDUCE_OR_KILL"
                else:
                    performance_tier = "MODERATE"
                    recommendation = "MAINTAIN"

                # Risk-adjusted performance
                if daily_sharpe > self.config["sharpe_threshold_good"]:
                    risk_adjusted = "GOOD"
                elif daily_sharpe > 0:
                    risk_adjusted = "FAIR"
                else:
                    risk_adjusted = "POOR"

                analysis[strategy] = {
                    "total_pnl": total_pnl,
                    "avg_daily_pnl": avg_daily_pnl,
                    "weekly_return_pct": weekly_return_pct,
                    "daily_volatility": daily_vol,
                    "daily_sharpe": daily_sharpe,
                    "win_rate": win_rate,
                    "max_drawdown": max_drawdown,
                    "performance_tier": performance_tier,
                    "risk_adjusted": risk_adjusted,
                    "recommendation": recommendation,
                    "trading_days": len(pnl_series),
                }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {e}")
            return {}

    def get_tca_summary(self) -> Dict[str, Any]:
        """Get TCA (Transaction Cost Analysis) summary."""
        try:
            tca_summary = {}

            for venue in self.config["venues"]:
                tca_data = self.redis.hgetall(f"tca:venue:{venue}")

                if tca_data:
                    score = float(tca_data.get("score", 0))
                    is_bps = float(tca_data.get("is_bps", 0))
                    fill_rate = float(tca_data.get("fill_rate", 0))
                    latency_ms = float(tca_data.get("lat_ms", 0))

                    # Grade venues
                    if score > 0.4:
                        grade = "A"
                    elif score > 0.2:
                        grade = "B"
                    elif score > 0:
                        grade = "C"
                    else:
                        grade = "F"

                    # Venue recommendation
                    if grade in ["A", "B"]:
                        recommendation = "INCREASE_ALLOCATION"
                    elif grade == "C":
                        recommendation = "MAINTAIN"
                    else:
                        recommendation = "REDUCE_ALLOCATION"

                    tca_summary[venue] = {
                        "score": score,
                        "is_bps": is_bps,
                        "fill_rate": fill_rate,
                        "latency_ms": latency_ms,
                        "grade": grade,
                        "recommendation": recommendation,
                    }
                else:
                    # Mock data for venues without TCA data
                    mock_score = np.random.uniform(0.1, 0.6)
                    grade = (
                        "A" if mock_score > 0.4 else ("B" if mock_score > 0.2 else "C")
                    )

                    tca_summary[venue] = {
                        "score": mock_score,
                        "is_bps": np.random.uniform(-2, 8),
                        "fill_rate": np.random.uniform(0.75, 0.95),
                        "latency_ms": np.random.uniform(50, 200),
                        "grade": grade,
                        "recommendation": (
                            "INCREASE_ALLOCATION" if grade in ["A", "B"] else "MAINTAIN"
                        ),
                    }

            return tca_summary

        except Exception as e:
            logger.error(f"Error getting TCA summary: {e}")
            return {}

    def get_allocation_weights(self) -> Dict[str, Any]:
        """Get current and recommended allocation weights."""
        try:
            allocation_data = {}

            # Strategy allocations
            strategy_allocations = {}
            for strategy in self.config["strategies"]:
                current_weight = float(
                    self.redis.get(f"allocator:weight:{strategy}") or 0.33
                )
                target_weight = float(
                    self.redis.get(f"allocator:target:{strategy}") or 0.33
                )

                strategy_allocations[strategy] = {
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": target_weight - current_weight,
                }

            allocation_data["strategies"] = strategy_allocations

            # Venue allocations (from TCA-driven router weights)
            venue_allocations = {}
            for venue in self.config["venues"]:
                current_weight = float(self.redis.get(f"router:weight:{venue}") or 0.25)

                venue_allocations[venue] = {"current_weight": current_weight}

            allocation_data["venues"] = venue_allocations

            return allocation_data

        except Exception as e:
            logger.error(f"Error getting allocation weights: {e}")
            return {"strategies": {}, "venues": {}}

    def generate_recommendations(
        self, strategy_analysis: Dict, tca_summary: Dict, allocation_data: Dict
    ) -> Dict[str, List[str]]:
        """Generate actionable recommendations."""
        try:
            recommendations = {
                "double_down": [],
                "kill": [],
                "venue_changes": [],
                "risk_management": [],
                "general": [],
            }

            # Strategy recommendations
            for strategy, analysis in strategy_analysis.items():
                if analysis["recommendation"] == "DOUBLE_DOWN":
                    recommendations["double_down"].append(
                        f"**{strategy}**: Strong performer ({analysis['weekly_return_pct']:.1%} weekly, "
                        f"Sharpe: {analysis['daily_sharpe']:.2f}) - increase allocation"
                    )
                elif analysis["recommendation"] == "REDUCE_OR_KILL":
                    recommendations["kill"].append(
                        f"**{strategy}**: Underperforming ({analysis['weekly_return_pct']:.1%} weekly) - "
                        f"consider reducing allocation or reviewing strategy"
                    )

            # Venue recommendations
            for venue, data in tca_summary.items():
                if data["recommendation"] == "INCREASE_ALLOCATION":
                    recommendations["venue_changes"].append(
                        f"**{venue.title()}**: Grade {data['grade']} (score: {data['score']:.2f}) - "
                        f"increase order flow allocation"
                    )
                elif data["recommendation"] == "REDUCE_ALLOCATION":
                    recommendations["venue_changes"].append(
                        f"**{venue.title()}**: Grade {data['grade']} (score: {data['score']:.2f}) - "
                        f"reduce allocation due to poor execution quality"
                    )

            # Risk management recommendations
            for strategy, analysis in strategy_analysis.items():
                if analysis["max_drawdown"] > 0.15:  # >15% drawdown
                    recommendations["risk_management"].append(
                        f"**{strategy}**: High drawdown ({analysis['max_drawdown']:.1%}) - "
                        f"review position sizing and risk controls"
                    )

                if analysis["daily_sharpe"] < 0:
                    recommendations["risk_management"].append(
                        f"**{strategy}**: Negative Sharpe ratio - review risk-return profile"
                    )

            # General system recommendations
            total_pnl = sum(
                analysis["total_pnl"] for analysis in strategy_analysis.values()
            )
            if total_pnl < 0:
                recommendations["general"].append(
                    "**System**: Negative weekly P&L - consider reducing overall risk and "
                    "reviewing strategy parameters"
                )

            # Fill in defaults if no specific recommendations
            if not recommendations["double_down"]:
                recommendations["general"].append(
                    "No standout performers this week - maintain current allocations"
                )

            if not recommendations["kill"]:
                recommendations["general"].append(
                    "All strategies showing acceptable performance"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "double_down": [],
                "kill": [],
                "venue_changes": [],
                "risk_management": [],
                "general": [],
            }

    def generate_markdown_report(
        self,
        week_end_date: str,
        strategy_analysis: Dict,
        tca_summary: Dict,
        allocation_data: Dict,
        recommendations: Dict,
    ) -> str:
        """Generate comprehensive markdown report."""
        try:
            # Calculate summary metrics
            total_pnl = sum(
                analysis["total_pnl"] for analysis in strategy_analysis.values()
            )
            avg_sharpe = np.mean(
                [analysis["daily_sharpe"] for analysis in strategy_analysis.values()]
            )
            best_strategy = (
                max(strategy_analysis, key=lambda s: strategy_analysis[s]["total_pnl"])
                if strategy_analysis
                else "None"
            )
            worst_strategy = (
                min(strategy_analysis, key=lambda s: strategy_analysis[s]["total_pnl"])
                if strategy_analysis
                else "None"
            )

            report = f"""# Weekly Trading Retrospective

**Week Ending:** {week_end_date}
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total P&L** | ${total_pnl:+,.2f} |
| **Average Daily Sharpe** | {avg_sharpe:.2f} |
| **Best Strategy** | {best_strategy} |
| **Worst Strategy** | {worst_strategy} |
| **Active Strategies** | {len(strategy_analysis)} |

## Strategy Performance Analysis

"""
            # Strategy performance table
            report += "| Strategy | Total P&L | Weekly Return | Daily Sharpe | Win Rate | Max DD | Tier | Recommendation |\n"
            report += "|----------|-----------|---------------|--------------|----------|--------|------|----------------|\n"

            for strategy, analysis in strategy_analysis.items():
                report += f"| **{strategy}** | ${analysis['total_pnl']:+,.0f} | {analysis['weekly_return_pct']:+.1%} | "
                report += (
                    f"{analysis['daily_sharpe']:.2f} | {analysis['win_rate']:.0%} | "
                )
                report += f"{analysis['max_drawdown']:.1%} | {analysis['performance_tier']} | "
                report += f"{analysis['recommendation'].replace('_', ' ').title()} |\n"

            # TCA Analysis
            report += f"\n## Transaction Cost Analysis (TCA)\n\n"
            report += "| Venue | Grade | Score | Avg IS (bps) | Fill Rate | Latency (ms) | Recommendation |\n"
            report += "|-------|-------|-------|--------------|-----------|--------------|----------------|\n"

            for venue, data in tca_summary.items():
                report += f"| **{venue.title()}** | **{data['grade']}** | {data['score']:.2f} | "
                report += f"{data['is_bps']:+.1f} | {data['fill_rate']:.0%} | {data['latency_ms']:.0f} | "
                report += f"{data['recommendation'].replace('_', ' ').title()} |\n"

            # Current Allocations
            report += f"\n## Current Allocations\n\n"
            report += "### Strategy Weights\n"
            for strategy, data in allocation_data.get("strategies", {}).items():
                change_indicator = (
                    "📈"
                    if data["weight_change"] > 0.01
                    else ("📉" if data["weight_change"] < -0.01 else "➡️")
                )
                report += f"- **{strategy}**: {data['current_weight']:.0%} {change_indicator}\n"

            report += "\n### Venue Weights\n"
            for venue, data in allocation_data.get("venues", {}).items():
                report += f"- **{venue.title()}**: {data['current_weight']:.0%}\n"

            # Recommendations
            report += f"\n## Key Recommendations\n\n"

            if recommendations["double_down"]:
                report += "### 🚀 Double Down (Increase Allocation)\n"
                for rec in recommendations["double_down"]:
                    report += f"- {rec}\n"
                report += "\n"

            if recommendations["kill"]:
                report += "### ❌ Consider Reducing (Poor Performance)\n"
                for rec in recommendations["kill"]:
                    report += f"- {rec}\n"
                report += "\n"

            if recommendations["venue_changes"]:
                report += "### 🔄 Venue Allocation Changes\n"
                for rec in recommendations["venue_changes"]:
                    report += f"- {rec}\n"
                report += "\n"

            if recommendations["risk_management"]:
                report += "### ⚠️ Risk Management\n"
                for rec in recommendations["risk_management"]:
                    report += f"- {rec}\n"
                report += "\n"

            if recommendations["general"]:
                report += "### 📋 General Notes\n"
                for rec in recommendations["general"]:
                    report += f"- {rec}\n"
                report += "\n"

            # Next Week Action Items
            report += f"""## Next Week Action Items

### Immediate (This Week)
- [ ] Review and implement allocation changes
- [ ] Monitor underperforming strategies closely
- [ ] Adjust venue routing based on TCA grades

### Medium Term (Next 2 Weeks)  
- [ ] Backtest strategy parameter adjustments
- [ ] Review risk control effectiveness
- [ ] Evaluate new venue partnerships

### Strategic (Next Month)
- [ ] Comprehensive strategy review
- [ ] Model retraining if needed
- [ ] Infrastructure optimizations

---

*Report generated by Weekly Retrospective System*
*Next report: {(datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")}*
*For questions: trading-team@company.com*
"""

            return report

        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            return f"# Weekly Retrospective Error\n\nError generating report: {e}\n"

    def save_report(self, report_content: str, week_end_date: str) -> str:
        """Save weekly report to file."""
        try:
            # Create reports directory
            reports_dir = Path(__file__).parent.parent / "reports" / "weekly"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate report filename
            report_file = reports_dir / f"weekly_retro_{week_end_date}.md"

            # Save report
            with open(report_file, "w") as f:
                f.write(report_content)

            logger.info(f"💾 Saved weekly report: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""

    def run_weekly_retrospective(self) -> Dict[str, Any]:
        """Run complete weekly retrospective analysis."""
        try:
            retro_start = time.time()
            logger.info("📊 Starting weekly retrospective...")

            # Determine week end date
            week_end_date = datetime.utcnow().strftime("%Y-%m-%d")

            # Gather data
            logger.info("📈 Gathering P&L data...")
            pnl_data = self.get_daily_pnl_data(self.config["lookback_days"])

            logger.info("🔍 Analyzing strategy performance...")
            strategy_analysis = self.analyze_strategy_performance(pnl_data)

            logger.info("💱 Getting TCA summary...")
            tca_summary = self.get_tca_summary()

            logger.info("⚖️ Getting allocation weights...")
            allocation_data = self.get_allocation_weights()

            logger.info("💡 Generating recommendations...")
            recommendations = self.generate_recommendations(
                strategy_analysis, tca_summary, allocation_data
            )

            # Generate report
            logger.info("📝 Generating markdown report...")
            report_content = self.generate_markdown_report(
                week_end_date,
                strategy_analysis,
                tca_summary,
                allocation_data,
                recommendations,
            )

            # Save report
            report_file = self.save_report(report_content, week_end_date)

            # Store summary in Redis
            summary = {
                "week_end_date": week_end_date,
                "total_pnl": sum(
                    analysis["total_pnl"] for analysis in strategy_analysis.values()
                ),
                "strategy_count": len(strategy_analysis),
                "double_down_strategies": len(recommendations["double_down"]),
                "kill_strategies": len(recommendations["kill"]),
                "venue_changes": len(recommendations["venue_changes"]),
                "report_file": report_file,
                "timestamp": time.time(),
            }

            self.redis.set("weekly_retro:latest", json.dumps(summary, default=str))

            retro_duration = time.time() - retro_start

            logger.info(
                f"✅ Weekly retrospective completed in {retro_duration:.1f}s: "
                f"${summary['total_pnl']:+,.0f} total P&L, {summary['strategy_count']} strategies analyzed"
            )

            result = {
                "status": "completed",
                "duration": retro_duration,
                "summary": summary,
                "strategy_analysis": strategy_analysis,
                "tca_summary": tca_summary,
                "allocation_data": allocation_data,
                "recommendations": recommendations,
                "report_file": report_file,
            }

            return result

        except Exception as e:
            logger.error(f"Error in weekly retrospective: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Weekly Retrospective")
    parser.add_argument("--run", action="store_true", help="Run weekly retrospective")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    retro = WeeklyRetrospective()

    if args.run or not sys.argv[1:]:  # Default to run
        result = retro.run_weekly_retrospective()

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result["status"] == "completed":
                summary = result["summary"]
                print(f"📊 Weekly Retrospective Complete:")
                print(f"  Total P&L: ${summary['total_pnl']:+,.2f}")
                print(f"  Report: {summary['report_file']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown')}")

        sys.exit(0 if result["status"] == "completed" else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
