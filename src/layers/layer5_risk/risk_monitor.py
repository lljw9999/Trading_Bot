"""
Risk Monitoring Service

Real-time risk monitoring and alerting system that runs independently
of the main trading system to provide oversight and early warning.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum

from .advanced_risk_manager import AdvancedRiskManager, RiskLevel, RiskMetrics
from .hedge_executor_deribit import DeribitHedgeExecutor


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert structure."""

    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    metric_value: float
    threshold: float
    symbol: Optional[str] = None
    recommended_action: Optional[str] = None


class RiskMonitor:
    """
    Independent risk monitoring service.

    Provides real-time risk oversight, alerting, and emergency controls.
    """

    def __init__(
        self,
        risk_manager: AdvancedRiskManager,
        monitoring_interval: int = 5,  # seconds
        alert_cooldown: int = 60,
    ):  # seconds
        """
        Initialize risk monitor.

        Args:
            risk_manager: Advanced risk manager instance
            monitoring_interval: Monitoring frequency in seconds
            alert_cooldown: Minimum time between similar alerts
        """
        self.risk_manager = risk_manager
        self.monitoring_interval = monitoring_interval
        self.alert_cooldown = alert_cooldown

        # Initialize hedge executor
        self.hedge_executor = DeribitHedgeExecutor()

        # Alert tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.last_alert_time: Dict[str, datetime] = {}

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.last_risk_check = datetime.now()

        # Alert thresholds
        self.alert_thresholds = {
            "var_warning": 0.05,  # 5% VaR warning
            "var_critical": 0.08,  # 8% VaR critical
            "drawdown_warning": 0.10,  # 10% drawdown warning
            "drawdown_critical": 0.15,  # 15% drawdown critical
            "leverage_warning": 3.0,  # 3x leverage warning
            "leverage_critical": 4.0,  # 4x leverage critical
            "concentration_warning": 0.5,  # 50% concentration warning
            "concentration_critical": 0.7,  # 70% concentration critical
            "correlation_warning": 0.8,  # 80% correlation warning
            "correlation_critical": 0.9,  # 90% correlation critical
        }

        # Performance tracking
        self.monitoring_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "emergency_stops": 0,
            "uptime_start": datetime.now(),
        }

        self.logger = logging.getLogger("risk_monitor")
        self.logger.info("Risk Monitor initialized")

    async def start_monitoring(self):
        """Start continuous risk monitoring."""
        if self.is_monitoring:
            self.logger.warning("Risk monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Risk monitoring started")

    async def stop_monitoring(self):
        """Stop risk monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Risk monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                await self._perform_risk_check()
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            # Restart monitoring after error
            await asyncio.sleep(10)
            if self.is_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _perform_risk_check(self):
        """Perform comprehensive risk check."""
        try:
            self.monitoring_stats["total_checks"] += 1
            self.last_risk_check = datetime.now()

            # Get current portfolio data (would come from portfolio manager)
            current_positions = self._get_current_positions()
            portfolio_value = self._get_portfolio_value()

            if not current_positions or portfolio_value <= 0:
                return

            # Get comprehensive risk metrics
            risk_metrics = self.risk_manager.get_comprehensive_risk_metrics(
                current_positions, portfolio_value
            )

            # Check all risk categories
            await self._check_var_risk(risk_metrics)
            await self._check_drawdown_risk(risk_metrics)
            await self._check_leverage_risk(risk_metrics)
            await self._check_concentration_risk(risk_metrics)
            await self._check_correlation_risk(risk_metrics)
            await self._check_stress_test_risk(risk_metrics)

            # Check for emergency conditions
            await self._check_emergency_conditions(risk_metrics)

            # Check hedge status and rebalancing needs
            await self._check_hedge_status()

            # Clean up old alerts
            self._cleanup_old_alerts()

            # Capture latency data for sim-to-real training
            self._capture_latency_histogram()

        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            await self._create_alert(
                AlertSeverity.CRITICAL, "system", f"Risk check failed: {e}", 0.0, 0.0
            )

    def _capture_latency_histogram(self):
        """Capture latency histogram for sim-to-real training."""
        import redis
        import json

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        try:
            lat = r.hgetall("lat_hops")
            if lat:
                r.rpush("lat_hist", json.dumps(lat))
                r.ltrim("lat_hist", -5000, -1)
        except Exception as e:
            self.logger.error(f"Error capturing latency histogram: {e}")

    async def _check_var_risk(self, risk_metrics: RiskMetrics):
        """Check VaR risk levels."""
        var_1d = risk_metrics.var_result.var_1d

        if var_1d > self.alert_thresholds["var_critical"]:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "var",
                f"VaR exceeds critical threshold: {var_1d:.2%}",
                var_1d,
                self.alert_thresholds["var_critical"],
                recommended_action="Reduce position sizes immediately",
            )
        elif var_1d > self.alert_thresholds["var_warning"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                "var",
                f"VaR exceeds warning threshold: {var_1d:.2%}",
                var_1d,
                self.alert_thresholds["var_warning"],
                recommended_action="Monitor positions closely",
            )

    async def _check_drawdown_risk(self, risk_metrics: RiskMetrics):
        """Check drawdown risk levels."""
        max_drawdown = risk_metrics.max_drawdown

        if max_drawdown > self.alert_thresholds["drawdown_critical"]:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "drawdown",
                f"Drawdown exceeds critical threshold: {max_drawdown:.2%}",
                max_drawdown,
                self.alert_thresholds["drawdown_critical"],
                recommended_action="Consider emergency stop",
            )
        elif max_drawdown > self.alert_thresholds["drawdown_warning"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                "drawdown",
                f"Drawdown exceeds warning threshold: {max_drawdown:.2%}",
                max_drawdown,
                self.alert_thresholds["drawdown_warning"],
                recommended_action="Review risk management",
            )

    async def _check_leverage_risk(self, risk_metrics: RiskMetrics):
        """Check leverage risk levels."""
        leverage = risk_metrics.leverage_ratio

        if leverage > self.alert_thresholds["leverage_critical"]:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "leverage",
                f"Leverage exceeds critical threshold: {leverage:.2f}x",
                leverage,
                self.alert_thresholds["leverage_critical"],
                recommended_action="Reduce leverage immediately",
            )
        elif leverage > self.alert_thresholds["leverage_warning"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                "leverage",
                f"Leverage exceeds warning threshold: {leverage:.2f}x",
                leverage,
                self.alert_thresholds["leverage_warning"],
                recommended_action="Monitor leverage closely",
            )

    async def _check_concentration_risk(self, risk_metrics: RiskMetrics):
        """Check concentration risk levels."""
        concentration = risk_metrics.concentration_risk

        if concentration > self.alert_thresholds["concentration_critical"]:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "concentration",
                f"Portfolio concentration exceeds critical threshold: {concentration:.2%}",
                concentration,
                self.alert_thresholds["concentration_critical"],
                recommended_action="Diversify positions",
            )
        elif concentration > self.alert_thresholds["concentration_warning"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                "concentration",
                f"Portfolio concentration exceeds warning threshold: {concentration:.2%}",
                concentration,
                self.alert_thresholds["concentration_warning"],
                recommended_action="Consider diversification",
            )

    async def _check_correlation_risk(self, risk_metrics: RiskMetrics):
        """Check correlation risk levels."""
        if risk_metrics.correlation_matrix is None:
            return

        try:
            # Find maximum correlation
            import numpy as np

            corr_matrix = risk_metrics.correlation_matrix

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(corr_matrix, k=1)
            max_correlation = np.max(np.abs(upper_triangle))

            if max_correlation > self.alert_thresholds["correlation_critical"]:
                await self._create_alert(
                    AlertSeverity.CRITICAL,
                    "correlation",
                    f"Maximum correlation exceeds critical threshold: {max_correlation:.2%}",
                    max_correlation,
                    self.alert_thresholds["correlation_critical"],
                    recommended_action="Reduce correlated positions",
                )
            elif max_correlation > self.alert_thresholds["correlation_warning"]:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "correlation",
                    f"Maximum correlation exceeds warning threshold: {max_correlation:.2%}",
                    max_correlation,
                    self.alert_thresholds["correlation_warning"],
                    recommended_action="Monitor correlations",
                )
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {e}")

    async def _check_stress_test_risk(self, risk_metrics: RiskMetrics):
        """Check stress test results."""
        if not risk_metrics.stress_results:
            return

        # Find worst-case scenario
        worst_scenario = min(risk_metrics.stress_results, key=lambda x: x.estimated_pnl)

        # Alert if survival probability is low
        if worst_scenario.survival_probability < 0.5:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "stress_test",
                f"Low survival probability in {worst_scenario.scenario_name}: {worst_scenario.survival_probability:.2%}",
                worst_scenario.survival_probability,
                0.5,
                recommended_action="Review stress test scenarios",
            )

    async def _check_emergency_conditions(self, risk_metrics: RiskMetrics):
        """Check for emergency conditions requiring immediate action."""
        emergency_conditions = []

        # Multiple critical alerts
        critical_alerts = [
            alert
            for alert in self.active_alerts
            if alert.severity == AlertSeverity.CRITICAL
        ]

        if len(critical_alerts) >= 3:
            emergency_conditions.append("Multiple critical alerts active")

        # Extreme risk level
        if risk_metrics.risk_level == RiskLevel.EXTREME:
            emergency_conditions.append("Extreme risk level detected")

        # Very high VaR
        if risk_metrics.var_result.var_1d > 0.15:  # 15% VaR
            emergency_conditions.append(
                f"Extreme VaR: {risk_metrics.var_result.var_1d:.2%}"
            )

        # Activate emergency procedures if needed
        if emergency_conditions:
            await self._activate_emergency_procedures(emergency_conditions)

    async def _activate_emergency_procedures(self, conditions: List[str]):
        """Activate emergency procedures."""
        self.monitoring_stats["emergency_stops"] += 1

        # Activate kill switch
        self.risk_manager.kill_switch_active = True

        # Create emergency alert
        await self._create_alert(
            AlertSeverity.EMERGENCY,
            "emergency",
            f"Emergency procedures activated: {', '.join(conditions)}",
            len(conditions),
            3.0,
            recommended_action="IMMEDIATE MANUAL INTERVENTION REQUIRED",
        )

        self.logger.critical(f"EMERGENCY: {', '.join(conditions)}")

    async def _check_hedge_status(self):
        """Check hedge status and manage hedge positions."""
        try:
            # Get current gross exposure
            current_positions = self._get_current_positions()
            gross_exposure = sum(
                abs(position) for position in current_positions.values()
            )

            # Check if we need tail risk hedging
            if gross_exposure > 50000:  # If gross > $50k, consider hedging
                active_hedge = self.hedge_executor.get_active_hedge()

                if not active_hedge:
                    # No active hedge, check if we should open one
                    await self._check_hedge_entry_signal(gross_exposure)
                else:
                    # Monitor existing hedge
                    hedge_status = self.hedge_executor.monitor_hedge()

                    if hedge_status.get("needs_rebalance"):
                        await self._create_alert(
                            AlertSeverity.WARNING,
                            "hedge",
                            f"Hedge needs rebalancing: {hedge_status.get('notional_drift_pct', 0):.1f}% drift",
                            hedge_status.get("notional_drift_pct", 0),
                            self.hedge_executor.hedge_config["rebalance_threshold_pct"]
                            * 100,
                            recommended_action="Consider hedge rebalancing",
                        )
            else:
                # Low exposure, close hedge if active
                active_hedge = self.hedge_executor.get_active_hedge()
                if active_hedge:
                    await self._check_hedge_exit_signal(gross_exposure)

        except Exception as e:
            self.logger.error(f"Error checking hedge status: {e}")

    async def _check_hedge_entry_signal(self, gross_exposure: float):
        """Check if we should enter a hedge position."""
        try:
            # Calculate suggested hedge notional (25% of gross exposure)
            hedge_notional = min(gross_exposure * 0.25, 10000)  # Cap at $10k

            # Check risk conditions that warrant hedging
            hedge_triggers = []

            # High VaR trigger
            import redis

            r = redis.Redis(host="localhost", port=6379, decode_responses=True)

            try:
                var_1d = float(r.get("risk:var_1d") or 0)
                if var_1d > 0.06:  # 6% VaR
                    hedge_triggers.append(f"High VaR: {var_1d:.2%}")
            except:
                pass

            # Market stress trigger
            try:
                market_stress = float(r.get("market:stress_indicator") or 0)
                if market_stress > 0.7:  # 70% stress
                    hedge_triggers.append(f"Market stress: {market_stress:.1%}")
            except:
                pass

            # High volatility trigger
            try:
                btc_vol = float(r.get("market:btc_vol_30d") or 0)
                if btc_vol > 0.8:  # 80% annualized vol
                    hedge_triggers.append(f"High volatility: {btc_vol:.1%}")
            except:
                pass

            # If we have hedge triggers, open hedge
            if hedge_triggers and hedge_notional >= 100:
                self.logger.info(f"🛡️ Hedge entry signal: {', '.join(hedge_triggers)}")

                # Open hedge in background
                asyncio.create_task(
                    self._execute_hedge_entry(hedge_notional, hedge_triggers)
                )

        except Exception as e:
            self.logger.error(f"Error checking hedge entry signal: {e}")

    async def _check_hedge_exit_signal(self, gross_exposure: float):
        """Check if we should exit hedge position."""
        try:
            # Exit hedge if exposure dropped significantly
            if gross_exposure < 25000:  # Less than $25k gross
                self.logger.info("🛡️ Hedge exit signal: Low gross exposure")

                # Close hedge in background
                asyncio.create_task(self._execute_hedge_exit("Low exposure"))

        except Exception as e:
            self.logger.error(f"Error checking hedge exit signal: {e}")

    async def _execute_hedge_entry(self, notional: float, triggers: List[str]):
        """Execute hedge entry."""
        try:
            result = self.hedge_executor.open_spread("BTC", notional)

            if result.get("success"):
                await self._create_alert(
                    AlertSeverity.INFO,
                    "hedge",
                    f"Hedge opened: ${notional:,.0f} | Triggers: {', '.join(triggers)}",
                    notional,
                    0.0,
                    recommended_action="Monitor hedge performance",
                )
                self.logger.info(f"✅ Hedge opened successfully: ${notional:,.0f}")
            else:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "hedge",
                    f"Hedge opening failed: {result.get('error', 'Unknown error')}",
                    notional,
                    0.0,
                    recommended_action="Check hedge executor status",
                )

        except Exception as e:
            self.logger.error(f"Error executing hedge entry: {e}")

    async def _execute_hedge_exit(self, reason: str):
        """Execute hedge exit."""
        try:
            result = self.hedge_executor.close_spread()

            if result.get("success"):
                pnl = result.get("realized_pnl", 0)
                await self._create_alert(
                    AlertSeverity.INFO,
                    "hedge",
                    f"Hedge closed: ${pnl:+,.0f} P&L | Reason: {reason}",
                    pnl,
                    0.0,
                    recommended_action="Review hedge performance",
                )
                self.logger.info(f"✅ Hedge closed successfully: ${pnl:+,.0f} P&L")
            else:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "hedge",
                    f"Hedge closing failed: {result.get('error', 'Unknown error')}",
                    0.0,
                    0.0,
                    recommended_action="Manual hedge intervention needed",
                )

        except Exception as e:
            self.logger.error(f"Error executing hedge exit: {e}")

    async def _create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        metric_value: float,
        threshold: float,
        symbol: Optional[str] = None,
        recommended_action: Optional[str] = None,
    ):
        """Create and process risk alert."""
        # Check cooldown
        alert_key = f"{category}_{severity.value}"
        current_time = datetime.now()

        if alert_key in self.last_alert_time:
            time_since_last = (
                current_time - self.last_alert_time[alert_key]
            ).total_seconds()
            if time_since_last < self.alert_cooldown:
                return

        # Create alert
        alert = RiskAlert(
            timestamp=current_time,
            severity=severity,
            category=category,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            symbol=symbol,
            recommended_action=recommended_action,
        )

        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = current_time

        # Update stats
        self.monitoring_stats["alerts_generated"] += 1

        # Log alert
        log_message = f"RISK ALERT [{severity.value.upper()}] {category}: {message}"
        if recommended_action:
            log_message += f" | Action: {recommended_action}"

        if severity == AlertSeverity.EMERGENCY:
            self.logger.critical(log_message)
        elif severity == AlertSeverity.CRITICAL:
            self.logger.error(log_message)
        elif severity == AlertSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # Send alert (could integrate with notification system)
        await self._send_alert_notification(alert)

    async def _send_alert_notification(self, alert: RiskAlert):
        """Send alert notification (placeholder for notification system)."""
        # This would integrate with email, SMS, Slack, etc.
        try:
            # For now, just log the alert
            self.logger.info(f"Alert notification sent: {alert.message}")

            # Could integrate with external services here
            # await send_slack_notification(alert)
            # await send_email_notification(alert)

        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")

    def _cleanup_old_alerts(self):
        """Clean up old alerts."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)

        # Remove old alerts from active list
        self.active_alerts = [
            alert for alert in self.active_alerts if alert.timestamp > cutoff_time
        ]

        # Limit alert history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def _get_current_positions(self) -> Dict[str, float]:
        """Get current positions (placeholder - would integrate with portfolio manager)."""
        # This would get real position data from the portfolio manager
        # For now, return dummy data
        return {"BTCUSDT": 25000.0, "ETHUSDT": 15000.0, "SOLUSDT": 5000.0}

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value (placeholder)."""
        # This would get real portfolio value
        return 100000.0

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts."""
        alert_counts = {"emergency": 0, "critical": 0, "warning": 0, "info": 0}

        for alert in self.active_alerts:
            alert_counts[alert.severity.value] += 1

        return {
            "active_alerts": len(self.active_alerts),
            "alert_counts": alert_counts,
            "recent_alerts": [asdict(alert) for alert in self.active_alerts[-5:]],
            "kill_switch_active": self.risk_manager.kill_switch_active,
            "last_check": self.last_risk_check.isoformat(),
            "monitoring_active": self.is_monitoring,
        }

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        uptime = datetime.now() - self.monitoring_stats["uptime_start"]

        return {
            **self.monitoring_stats,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "avg_checks_per_hour": self.monitoring_stats["total_checks"]
            / max(1, uptime.total_seconds() / 3600),
            "alert_rate": self.monitoring_stats["alerts_generated"]
            / max(1, self.monitoring_stats["total_checks"]),
            "is_monitoring": self.is_monitoring,
            "last_check": self.last_risk_check.isoformat(),
        }

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an active alert."""
        try:
            if 0 <= alert_index < len(self.active_alerts):
                alert = self.active_alerts.pop(alert_index)
                self.logger.info(f"Alert acknowledged: {alert.message}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False

    def acknowledge_all_alerts(self) -> int:
        """Acknowledge all active alerts."""
        count = len(self.active_alerts)
        self.active_alerts.clear()
        self.logger.info(f"All {count} alerts acknowledged")
        return count

    def update_alert_thresholds(self, new_thresholds: Dict[str, float]):
        """Update alert thresholds."""
        self.alert_thresholds.update(new_thresholds)
        self.logger.info(f"Alert thresholds updated: {new_thresholds}")

    async def manual_risk_check(self) -> Dict[str, Any]:
        """Perform manual risk check and return results."""
        await self._perform_risk_check()
        return {
            "check_time": datetime.now().isoformat(),
            "active_alerts": len(self.active_alerts),
            "kill_switch_active": self.risk_manager.kill_switch_active,
            "recent_alerts": [asdict(alert) for alert in self.active_alerts[-10:]],
        }


# Factory function
def create_risk_monitor(risk_manager: AdvancedRiskManager, **kwargs) -> RiskMonitor:
    """Create risk monitor instance."""
    return RiskMonitor(risk_manager, **kwargs)
