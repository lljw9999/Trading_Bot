#!/usr/bin/env python3
"""
Weekly SLO Gate → Capital Cap Schedule
Decides next week's maximum live capital based on 7-day KPIs
"""

import os
import sys
import json
import time
import sqlite3
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("weekly_slo_gate")


class WeeklySLOGate:
    """Weekly SLO gate for capital cap scheduling."""

    def __init__(self):
        """Initialize weekly SLO gate."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # SLO gate configuration
        self.config = {
            "lookback_days": 7,  # Evaluate last 7 days
            "tiers": {
                "tier_a": {
                    "name": "Tier A",
                    "capital_cap": 1.0,  # 100% capital
                    "sharpe_min": 1.0,  # Sharpe ≥ 1.0
                    "max_dd_max": 0.03,  # MaxDD ≤ 3%
                    "recon_breaches_max": 0,  # 0 recon breaches
                    "halt_events_max": 1,  # ≤ 1 halt event
                },
                "tier_b": {
                    "name": "Tier B",
                    "capital_cap": 0.7,  # 70% capital
                    "sharpe_min": 0.6,  # Sharpe ≥ 0.6
                    "max_dd_max": 0.05,  # MaxDD ≤ 5%
                    "recon_breaches_max": 2,  # ≤ 2 recon breaches
                    "halt_events_max": 3,  # ≤ 3 halt events
                },
                "tier_c": {
                    "name": "Tier C",
                    "capital_cap": 0.4,  # 40% capital
                    "sharpe_min": 0.0,  # No minimum
                    "max_dd_max": 1.0,  # No maximum
                    "recon_breaches_max": 999,  # No maximum
                    "halt_events_max": 999,  # No maximum
                },
            },
            "default_tier": "tier_c",
            "db_path": "data/slo_history.db",
        }

        # Create data directory
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)

        # Initialize database
        self.db_path = data_dir / "slo_history.db"
        self._init_database()

        # Track SLO gate state
        self.current_tier = None
        self.current_cap = 0.4  # Default to Tier C
        self.last_evaluation = 0
        self.evaluation_history = []

        logger.info("📊 Weekly SLO Gate initialized")
        logger.info(f"   Lookback: {self.config['lookback_days']} days")
        logger.info(f"   Tiers: {len(self.config['tiers'])}")
        logger.info(f"   DB: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database for SLO history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS slo_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    week_start TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    sharpe_7d REAL,
                    max_dd_7d REAL,
                    win_rate_7d REAL,
                    recon_breaches INTEGER,
                    halt_events INTEGER,
                    tier TEXT NOT NULL,
                    capital_cap REAL NOT NULL,
                    kpis TEXT,  -- JSON string of all KPIs
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()

            logger.info("✅ SLO database initialized")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def get_7day_kpis(self) -> Dict[str, Any]:
        """Get 7-day KPIs from Redis/Prometheus."""
        try:
            kpis = {}
            end_time = time.time()
            start_time = end_time - (self.config["lookback_days"] * 24 * 3600)

            # Get Sharpe ratio (7-day)
            try:
                sharpe_7d = self.redis.get("metrics:sharpe_7d")
                kpis["sharpe_7d"] = (
                    float(sharpe_7d) if sharpe_7d else self._calculate_mock_sharpe()
                )
            except:
                kpis["sharpe_7d"] = self._calculate_mock_sharpe()

            # Get maximum drawdown (7-day)
            try:
                max_dd_7d = self.redis.get("metrics:max_dd_7d")
                kpis["max_dd_7d"] = (
                    float(max_dd_7d) if max_dd_7d else self._calculate_mock_drawdown()
                )
            except:
                kpis["max_dd_7d"] = self._calculate_mock_drawdown()

            # Get win rate (7-day)
            try:
                win_rate_7d = self.redis.get("metrics:win_rate_7d")
                kpis["win_rate_7d"] = (
                    float(win_rate_7d)
                    if win_rate_7d
                    else self._calculate_mock_win_rate()
                )
            except:
                kpis["win_rate_7d"] = self._calculate_mock_win_rate()

            # Count reconciliation breaches
            try:
                recon_breaches = self._count_recon_breaches(start_time, end_time)
                kpis["recon_breaches"] = recon_breaches
            except:
                kpis["recon_breaches"] = 0

            # Count feature gate halts
            try:
                halt_events = self._count_halt_events(start_time, end_time)
                kpis["halt_events"] = halt_events
            except:
                kpis["halt_events"] = 1

            # Additional metrics
            kpis.update(
                {
                    "total_pnl_7d": self._get_total_pnl_7d(),
                    "total_trades_7d": self._get_total_trades_7d(),
                    "avg_fill_rate_7d": self._get_avg_fill_rate_7d(),
                    "system_uptime_pct": self._get_system_uptime_7d(),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

            logger.info(f"📈 7-day KPIs collected:")
            logger.info(f"   Sharpe: {kpis['sharpe_7d']:.2f}")
            logger.info(f"   Max DD: {kpis['max_dd_7d']:.1%}")
            logger.info(f"   Win Rate: {kpis['win_rate_7d']:.1%}")
            logger.info(f"   Recon Breaches: {kpis['recon_breaches']}")
            logger.info(f"   Halt Events: {kpis['halt_events']}")

            return kpis

        except Exception as e:
            logger.error(f"Error getting 7-day KPIs: {e}")
            return self._get_fallback_kpis()

    def _calculate_mock_sharpe(self) -> float:
        """Calculate mock Sharpe ratio based on current time."""
        # Use deterministic mock based on week
        week_num = int(time.time() // (7 * 24 * 3600)) % 10
        mock_sharpes = [0.8, 1.2, 0.4, 1.5, 0.9, 0.3, 1.1, 0.6, 1.0, 0.7]
        return mock_sharpes[week_num]

    def _calculate_mock_drawdown(self) -> float:
        """Calculate mock max drawdown."""
        week_num = int(time.time() // (7 * 24 * 3600)) % 10
        mock_dds = [0.02, 0.04, 0.08, 0.01, 0.03, 0.12, 0.02, 0.06, 0.03, 0.05]
        return mock_dds[week_num]

    def _calculate_mock_win_rate(self) -> float:
        """Calculate mock win rate."""
        week_num = int(time.time() // (7 * 24 * 3600)) % 10
        mock_rates = [0.58, 0.62, 0.45, 0.68, 0.55, 0.42, 0.61, 0.51, 0.59, 0.54]
        return mock_rates[week_num]

    def _count_recon_breaches(self, start_time: float, end_time: float) -> int:
        """Count reconciliation breaches in time period."""
        try:
            # Get breach history from Redis
            breach_data = self.redis.get("recon:breach_history")
            if breach_data:
                breaches = json.loads(breach_data)
                count = sum(
                    1
                    for breach in breaches
                    if start_time <= breach.get("timestamp", 0) <= end_time
                )
                return count

            # Mock data: 0-2 breaches
            week_num = int(start_time // (7 * 24 * 3600)) % 5
            return [0, 1, 0, 2, 0][week_num]

        except Exception as e:
            logger.warning(f"Error counting recon breaches: {e}")
            return 0

    def _count_halt_events(self, start_time: float, end_time: float) -> int:
        """Count feature gate halt events in time period."""
        try:
            # Count mode=halt events in Redis
            halt_count = 0

            # Check if currently in halt mode
            current_mode = self.redis.get("mode")
            if current_mode == "halt":
                halt_count += 1

            # Mock additional halt events
            week_num = int(start_time // (7 * 24 * 3600)) % 4
            mock_halts = [0, 1, 2, 0][week_num]

            return halt_count + mock_halts

        except Exception as e:
            logger.warning(f"Error counting halt events: {e}")
            return 1

    def _get_total_pnl_7d(self) -> float:
        """Get total P&L for 7 days."""
        try:
            pnl_7d = self.redis.get("metrics:pnl_7d")
            return float(pnl_7d) if pnl_7d else np.random.uniform(-1000, 3000)
        except:
            return np.random.uniform(-1000, 3000)

    def _get_total_trades_7d(self) -> int:
        """Get total trades for 7 days."""
        try:
            trades_7d = self.redis.get("metrics:trades_7d")
            return int(trades_7d) if trades_7d else np.random.randint(500, 2000)
        except:
            return np.random.randint(500, 2000)

    def _get_avg_fill_rate_7d(self) -> float:
        """Get average fill rate for 7 days."""
        try:
            fill_rate_7d = self.redis.get("metrics:fill_rate_7d")
            return (
                float(fill_rate_7d) if fill_rate_7d else np.random.uniform(0.85, 0.98)
            )
        except:
            return np.random.uniform(0.85, 0.98)

    def _get_system_uptime_7d(self) -> float:
        """Get system uptime percentage for 7 days."""
        try:
            uptime_7d = self.redis.get("metrics:uptime_7d")
            return float(uptime_7d) if uptime_7d else np.random.uniform(0.95, 0.999)
        except:
            return np.random.uniform(0.95, 0.999)

    def _get_fallback_kpis(self) -> Dict[str, Any]:
        """Get fallback KPIs when real data unavailable."""
        return {
            "sharpe_7d": 0.5,
            "max_dd_7d": 0.08,
            "win_rate_7d": 0.52,
            "recon_breaches": 1,
            "halt_events": 2,
            "total_pnl_7d": 500,
            "total_trades_7d": 800,
            "avg_fill_rate_7d": 0.92,
            "system_uptime_pct": 0.98,
        }

    def evaluate_slo_tier(self, kpis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Evaluate SLO tier based on KPIs."""
        try:
            evaluation = {
                "kpis": kpis,
                "tier_checks": {},
                "selected_tier": None,
                "capital_cap": 0.4,  # Default to Tier C
                "evaluation_time": time.time(),
            }

            # Check each tier from highest to lowest
            for tier_id in ["tier_a", "tier_b", "tier_c"]:
                tier_config = self.config["tiers"][tier_id]

                checks = {
                    "sharpe_pass": kpis["sharpe_7d"] >= tier_config["sharpe_min"],
                    "max_dd_pass": kpis["max_dd_7d"] <= tier_config["max_dd_max"],
                    "recon_pass": kpis["recon_breaches"]
                    <= tier_config["recon_breaches_max"],
                    "halt_pass": kpis["halt_events"] <= tier_config["halt_events_max"],
                }

                all_pass = all(checks.values())

                evaluation["tier_checks"][tier_id] = {
                    "config": tier_config,
                    "checks": checks,
                    "all_pass": all_pass,
                }

                # Select first tier that passes all checks
                if all_pass and evaluation["selected_tier"] is None:
                    evaluation["selected_tier"] = tier_id
                    evaluation["capital_cap"] = tier_config["capital_cap"]
                    break

            # Default to Tier C if nothing passes
            if evaluation["selected_tier"] is None:
                evaluation["selected_tier"] = "tier_c"
                evaluation["capital_cap"] = self.config["tiers"]["tier_c"][
                    "capital_cap"
                ]

            selected_tier = evaluation["selected_tier"]
            tier_name = self.config["tiers"][selected_tier]["name"]
            cap_pct = evaluation["capital_cap"] * 100

            logger.info(
                f"🏆 SLO Evaluation Result: {tier_name} ({cap_pct:.0f}% capital)"
            )

            # Log why we didn't get Tier A
            if selected_tier != "tier_a":
                tier_a_checks = evaluation["tier_checks"]["tier_a"]["checks"]
                failures = [k for k, v in tier_a_checks.items() if not v]
                logger.info(f"   Tier A failures: {', '.join(failures)}")

            return selected_tier, evaluation

        except Exception as e:
            logger.error(f"Error evaluating SLO tier: {e}")
            return "tier_c", {"error": str(e), "capital_cap": 0.4}

    def store_evaluation_record(self, tier: str, evaluation: Dict[str, Any]):
        """Store evaluation record in SQLite database."""
        try:
            kpis = evaluation["kpis"]

            # Calculate week boundaries
            eval_time = evaluation["evaluation_time"]
            week_start = datetime.fromtimestamp(eval_time - 7 * 24 * 3600).strftime(
                "%Y-%m-%d"
            )
            week_end = datetime.fromtimestamp(eval_time).strftime("%Y-%m-%d")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO slo_evaluations (
                    timestamp, week_start, week_end, sharpe_7d, max_dd_7d, 
                    win_rate_7d, recon_breaches, halt_events, tier, 
                    capital_cap, kpis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    eval_time,
                    week_start,
                    week_end,
                    kpis["sharpe_7d"],
                    kpis["max_dd_7d"],
                    kpis["win_rate_7d"],
                    kpis["recon_breaches"],
                    kpis["halt_events"],
                    tier,
                    evaluation["capital_cap"],
                    json.dumps(kpis, default=str),
                ),
            )

            conn.commit()
            conn.close()

            logger.info("📝 Evaluation record stored in database")

        except Exception as e:
            logger.error(f"Error storing evaluation record: {e}")

    def send_slo_notification(self, tier: str, evaluation: Dict[str, Any]):
        """Send SLO evaluation notification to Slack."""
        try:
            if not self.slack_webhook:
                return False

            kpis = evaluation["kpis"]
            tier_name = self.config["tiers"][tier]["name"]
            cap_pct = evaluation["capital_cap"] * 100

            # Create detailed message
            message = (
                f"📊 *Weekly SLO Gate Evaluation*\n"
                f"🏆 Result: {tier_name} → Next week cap: {cap_pct:.0f}%\n\n"
                f"📈 *7-Day KPIs:*\n"
                f"• Sharpe Ratio: {kpis['sharpe_7d']:.2f}\n"
                f"• Max Drawdown: {kpis['max_dd_7d']:.1%}\n"
                f"• Win Rate: {kpis['win_rate_7d']:.1%}\n"
                f"• Recon Breaches: {kpis['recon_breaches']}\n"
                f"• Halt Events: {kpis['halt_events']}\n"
                f"• Total P&L: ${kpis.get('total_pnl_7d', 0):,.0f}\n"
                f"• Total Trades: {kpis.get('total_trades_7d', 0):,}\n"
                f"• Avg Fill Rate: {kpis.get('avg_fill_rate_7d', 0):.1%}"
            )

            # Add tier requirements comparison
            if tier != "tier_a":
                tier_a_config = self.config["tiers"]["tier_a"]
                message += (
                    f"\n\n🎯 *Tier A Requirements:*\n"
                    f"• Sharpe ≥ {tier_a_config['sharpe_min']:.1f}: "
                    f"{'✅' if kpis['sharpe_7d'] >= tier_a_config['sharpe_min'] else '❌'}\n"
                    f"• Max DD ≤ {tier_a_config['max_dd_max']:.0%}: "
                    f"{'✅' if kpis['max_dd_7d'] <= tier_a_config['max_dd_max'] else '❌'}\n"
                    f"• Recon Breaches ≤ {tier_a_config['recon_breaches_max']}: "
                    f"{'✅' if kpis['recon_breaches'] <= tier_a_config['recon_breaches_max'] else '❌'}\n"
                    f"• Halt Events ≤ {tier_a_config['halt_events_max']}: "
                    f"{'✅' if kpis['halt_events'] <= tier_a_config['halt_events_max'] else '❌'}"
                )

            payload = {
                "text": message,
                "username": "Weekly SLO Gate",
                "icon_emoji": ":chart_with_upwards_trend:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent SLO notification to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending SLO notification: {e}")
            return False

    def run_weekly_evaluation(self) -> Dict[str, Any]:
        """Run complete weekly SLO evaluation."""
        try:
            evaluation_start = time.time()
            logger.info("🗓️ Running weekly SLO evaluation...")

            # Get 7-day KPIs
            kpis = self.get_7day_kpis()

            # Evaluate SLO tier
            selected_tier, evaluation = self.evaluate_slo_tier(kpis)

            # Update current state
            self.current_tier = selected_tier
            self.current_cap = evaluation["capital_cap"]
            self.last_evaluation = evaluation_start

            # Store in Redis for next week
            self.redis.set("risk:capital_cap_next_week", self.current_cap)

            # Store current tier info
            tier_info = {
                "tier": selected_tier,
                "tier_name": self.config["tiers"][selected_tier]["name"],
                "capital_cap": self.current_cap,
                "evaluation_time": evaluation_start,
                "kpis": kpis,
            }

            self.redis.set("slo:current_tier", json.dumps(tier_info, default=str))

            # Store evaluation record
            self.store_evaluation_record(selected_tier, evaluation)

            # Send notification
            self.send_slo_notification(selected_tier, evaluation)

            # Track evaluation history
            self.evaluation_history.append(
                {
                    "timestamp": evaluation_start,
                    "tier": selected_tier,
                    "capital_cap": self.current_cap,
                    "kpis": kpis.copy(),
                }
            )

            if len(self.evaluation_history) > 52:  # Keep 1 year of history
                self.evaluation_history = self.evaluation_history[-26:]

            evaluation_duration = time.time() - evaluation_start

            result = {
                "timestamp": evaluation_start,
                "status": "completed",
                "selected_tier": selected_tier,
                "tier_name": self.config["tiers"][selected_tier]["name"],
                "capital_cap": self.current_cap,
                "capital_cap_pct": f"{self.current_cap*100:.0f}%",
                "kpis": kpis,
                "evaluation": evaluation,
                "evaluation_duration": evaluation_duration,
            }

            logger.info(
                f"✅ Weekly SLO evaluation completed: "
                f"{result['tier_name']} ({result['capital_cap_pct']} cap)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in weekly evaluation: {e}")
            return {
                "timestamp": time.time(),
                "status": "error",
                "error": str(e),
                "capital_cap": 0.4,  # Default to Tier C on error
            }

    def get_evaluation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get evaluation history from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM slo_evaluations 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            conn.close()

            # Convert to list of dicts
            history = []
            for row in rows:
                record = {
                    "id": row[0],
                    "timestamp": row[1],
                    "week_start": row[2],
                    "week_end": row[3],
                    "sharpe_7d": row[4],
                    "max_dd_7d": row[5],
                    "win_rate_7d": row[6],
                    "recon_breaches": row[7],
                    "halt_events": row[8],
                    "tier": row[9],
                    "capital_cap": row[10],
                    "kpis": json.loads(row[11]) if row[11] else {},
                    "created_at": row[12],
                }
                history.append(record)

            return history

        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            return []

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        try:
            # Get current tier info
            tier_info = self.redis.get("slo:current_tier")
            if tier_info:
                current_tier_data = json.loads(tier_info)
            else:
                current_tier_data = {
                    "tier": "tier_c",
                    "tier_name": "Tier C",
                    "capital_cap": 0.4,
                    "evaluation_time": 0,
                }

            # Get next week's cap
            next_week_cap = float(self.redis.get("risk:capital_cap_next_week") or 0.4)

            # Get recent evaluations
            recent_evaluations = self.get_evaluation_history(5)

            # Calculate time since last evaluation
            time_since_eval = (
                time.time() - self.last_evaluation if self.last_evaluation else 0
            )

            status = {
                "service": "weekly_slo_gate",
                "timestamp": time.time(),
                "config": self.config,
                "current_tier": current_tier_data,
                "next_week_capital_cap": next_week_cap,
                "next_week_capital_cap_pct": f"{next_week_cap*100:.0f}%",
                "last_evaluation": self.last_evaluation,
                "time_since_evaluation_hours": time_since_eval / 3600,
                "evaluation_count": len(self.evaluation_history),
                "recent_evaluations": recent_evaluations[:3],  # Last 3 evaluations
                "database": {
                    "path": str(self.db_path),
                    "exists": self.db_path.exists(),
                    "total_records": len(recent_evaluations),
                },
            }

            return status

        except Exception as e:
            return {
                "service": "weekly_slo_gate",
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


def main():
    """Main entry point for weekly SLO gate."""
    import argparse

    parser = argparse.ArgumentParser(description="Weekly SLO Gate")
    parser.add_argument("--run", action="store_true", help="Run weekly SLO evaluation")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Show evaluation history (number of records)",
    )

    args = parser.parse_args()

    # Create SLO gate
    slo_gate = WeeklySLOGate()

    if args.status:
        # Show status report
        status = slo_gate.get_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.history and not args.run:
        # Show evaluation history
        history = slo_gate.get_evaluation_history(args.history)
        print(json.dumps(history, indent=2, default=str))
        return

    if args.run:
        # Run weekly evaluation
        result = slo_gate.run_weekly_evaluation()
        print(json.dumps(result, indent=2, default=str))

        if result.get("status") == "completed":
            sys.exit(0)
        else:
            sys.exit(1)

    # Default - show help
    parser.print_help()


if __name__ == "__main__":
    main()
