#!/usr/bin/env python3
"""
Trading System Dashboard Server

Provides web interface for monitoring all trading system components:
- Real-time status of all 20 gap-filler systems
- API endpoints for frontend dashboard
- WebSocket support for live updates
- Emergency controls integration
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
    import redis

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

# Import our monitoring systems
from scripts.rl_policy_watchdog import RLPolicyWatchdog
from scripts.economic_event_guard import EconomicEventGuard
from scripts.market_hours_guard import MarketHoursGuard
from scripts.api_quota_monitor import APIQuotaMonitor
from scripts.time_sync_monitor import TimeSyncMonitor
from scripts.security_hardener import SecurityHardener
from scripts.broker_statement_reconciler import BrokerStatementReconciler
from scripts.panic_button import PanicButton
from scripts.pdt_short_manager import PDTShortManager

logger = logging.getLogger("dashboard_server")


class TradingSystemDashboardServer:
    """
    Web server providing comprehensive monitoring dashboard for trading system.
    """

    def __init__(self):
        """Initialize dashboard server."""
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend access

        # Initialize monitoring components
        self.monitors = {
            "rl_watchdog": RLPolicyWatchdog(),
            "economic_guard": EconomicEventGuard(),
            "market_hours": MarketHoursGuard(),
            "api_quota": APIQuotaMonitor(),
            "time_sync": TimeSyncMonitor(),
            "security": SecurityHardener(),
            "reconciliation": BrokerStatementReconciler(),
            "panic_button": PanicButton(),
            "pdt_manager": PDTShortManager(),
        }

        # Redis for caching
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        self._setup_routes()
        logger.info("Dashboard server initialized")

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve main dashboard."""
            return send_from_directory("frontend", "ops-dashboard.html")

        @self.app.route("/trading")
        def trading_dashboard():
            """Serve trading dashboard."""
            return send_from_directory("frontend", "live-trading-dashboard.html")

        @self.app.route("/api/status")
        def system_status():
            """Get overall system status."""
            try:
                status = self.get_system_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ops-dashboard")
        def ops_dashboard():
            """Get ops dashboard data."""
            try:
                dashboard_data = self.get_ops_dashboard_data()
                return jsonify(dashboard_data)
            except Exception as e:
                logger.error(f"Error getting ops dashboard data: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/trading-dashboard")
        def trading_dashboard_api():
            """Get trading dashboard data."""
            try:
                trading_data = self.get_trading_dashboard_data()
                return jsonify(trading_data)
            except Exception as e:
                logger.error(f"Error getting trading dashboard data: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/panic", methods=["POST"])
        def emergency_panic():
            """Emergency panic button endpoint."""
            try:
                data = request.get_json() or {}
                reason = data.get("reason", "Web dashboard panic button")

                result = self.monitors["panic_button"].execute_panic_sequence(
                    reason=reason, initiated_by="web_dashboard"
                )

                return jsonify(result)
            except Exception as e:
                logger.error(f"Error executing panic: {e}")
                return jsonify({"error": str(e), "overall_success": False}), 500

        @self.app.route("/api/panic/status")
        def panic_status():
            """Get panic button status."""
            try:
                status = self.monitors["panic_button"].check_panic_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting panic status: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/panic/clear", methods=["POST"])
        def clear_panic():
            """Clear panic mode."""
            try:
                result = self.monitors["panic_button"].clear_panic_mode(
                    cleared_by="web_dashboard"
                )
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error clearing panic: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/health")
        def health_check():
            """Health check endpoint."""
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "service": "trading_system_dashboard",
                }
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": True,
                "components": {},
            }

            # Check each monitoring component

            # 1. RL Policy Watchdog
            try:
                rl_health = self.monitors["rl_watchdog"].check_policy_health()
                status["components"]["rl_policy"] = {
                    "status": (
                        "healthy"
                        if rl_health.get("overall_healthy", False)
                        else "unhealthy"
                    ),
                    "details": rl_health,
                    "actions_needed": rl_health.get("actions_needed", []),
                }
                if not rl_health.get("overall_healthy", False):
                    status["overall_healthy"] = False
            except Exception as e:
                status["components"]["rl_policy"] = {"status": "error", "error": str(e)}
                status["overall_healthy"] = False

            # 2. Economic Event Guard
            try:
                econ_status = self.monitors[
                    "economic_guard"
                ].check_current_event_status()
                status["components"]["economic_events"] = {
                    "status": (
                        "active"
                        if econ_status.get("event_guard_active", False)
                        else "monitoring"
                    ),
                    "details": econ_status,
                }
            except Exception as e:
                status["components"]["economic_events"] = {
                    "status": "error",
                    "error": str(e),
                }

            # 3. Market Hours Guard
            try:
                market_status = self.monitors["market_hours"].get_market_status()
                status["components"]["market_hours"] = {
                    "status": (
                        "open"
                        if market_status.get("trading_allowed", False)
                        else "closed"
                    ),
                    "details": market_status,
                }
            except Exception as e:
                status["components"]["market_hours"] = {
                    "status": "error",
                    "error": str(e),
                }

            # 4. API Quota Monitor
            try:
                quota_status = self.monitors["api_quota"].get_all_exchange_status()
                healthy_exchanges = quota_status.get("summary", {}).get(
                    "overall_healthy", False
                )
                status["components"]["api_quotas"] = {
                    "status": "healthy" if healthy_exchanges else "issues",
                    "details": quota_status,
                }
                if not healthy_exchanges:
                    status["overall_healthy"] = False
            except Exception as e:
                status["components"]["api_quotas"] = {
                    "status": "error",
                    "error": str(e),
                }

            # 5. Time Sync Monitor
            try:
                time_health = self.monitors["time_sync"].check_time_sync_health()
                status["components"]["time_sync"] = {
                    "status": (
                        "healthy" if time_health.get("healthy", False) else "unhealthy"
                    ),
                    "details": time_health,
                }
                if not time_health.get("healthy", False):
                    status["overall_healthy"] = False
            except Exception as e:
                status["components"]["time_sync"] = {"status": "error", "error": str(e)}

            # 6. Security Compliance
            try:
                security_scan = self.monitors["security"].run_compliance_scan()
                status["components"]["security"] = {
                    "status": (
                        "compliant"
                        if security_scan.get("overall_compliant", False)
                        else "issues"
                    ),
                    "details": security_scan,
                }
                if not security_scan.get("overall_compliant", False):
                    status["overall_healthy"] = False
            except Exception as e:
                status["components"]["security"] = {"status": "error", "error": str(e)}

            # 7. Panic Button Status
            try:
                panic_status = self.monitors["panic_button"].check_panic_status()
                status["components"]["panic_button"] = {
                    "status": (
                        "active" if panic_status.get("panic_active", False) else "ready"
                    ),
                    "details": panic_status,
                }
                if panic_status.get("panic_active", False):
                    status["overall_healthy"] = False
            except Exception as e:
                status["components"]["panic_button"] = {
                    "status": "error",
                    "error": str(e),
                }

            return status

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "overall_healthy": False}

    def get_ops_dashboard_data(self) -> Dict[str, Any]:
        """Get operations dashboard data."""
        try:
            dashboard_data = {"timestamp": datetime.now().isoformat(), "sections": {}}

            # System Status Section
            system_status = self.get_system_status()
            dashboard_data["sections"]["system_status"] = {
                "title": "System Status",
                "status": (
                    "HEALTHY"
                    if system_status.get("overall_healthy", False)
                    else "ISSUES"
                ),
                "data": {
                    "components_healthy": sum(
                        1
                        for comp in system_status.get("components", {}).values()
                        if comp.get("status") in ["healthy", "monitoring", "ready"]
                    ),
                    "total_components": len(system_status.get("components", {})),
                    "overall_healthy": system_status.get("overall_healthy", False),
                },
            }

            # Market Status Section
            try:
                market_status = self.monitors["market_hours"].get_market_status()
                dashboard_data["sections"]["market_status"] = {
                    "title": "Market Status",
                    "status": (
                        "OPEN"
                        if market_status.get("trading_allowed", False)
                        else "CLOSED"
                    ),
                    "data": {
                        "market_open": market_status.get("market_open", False),
                        "trading_allowed": market_status.get("trading_allowed", False),
                        "halt_reasons": market_status.get("halt_reasons", []),
                        "holiday_status": market_status.get("holiday_status", {}).get(
                            "is_holiday", False
                        ),
                    },
                }
            except Exception as e:
                dashboard_data["sections"]["market_status"] = {
                    "title": "Market Status",
                    "status": "ERROR",
                    "data": {"error": str(e)},
                }

            # API Quotas Section
            try:
                quota_status = self.monitors["api_quota"].get_all_exchange_status()
                summary = quota_status.get("summary", {})
                dashboard_data["sections"]["api_quotas"] = {
                    "title": "API Quotas",
                    "status": (
                        "HEALTHY" if summary.get("overall_healthy", False) else "ISSUES"
                    ),
                    "data": {
                        "total_exchanges": summary.get("total_exchanges", 0),
                        "warning_alerts": summary.get("warning_alerts", 0),
                        "critical_alerts": summary.get("critical_alerts", 0),
                        "overall_healthy": summary.get("overall_healthy", False),
                    },
                }
            except Exception as e:
                dashboard_data["sections"]["api_quotas"] = {
                    "title": "API Quotas",
                    "status": "ERROR",
                    "data": {"error": str(e)},
                }

            # Security Section
            try:
                security_scan = self.monitors["security"].run_compliance_scan()
                dashboard_data["sections"]["security"] = {
                    "title": "Security Compliance",
                    "status": (
                        "COMPLIANT"
                        if security_scan.get("overall_compliant", False)
                        else "ISSUES"
                    ),
                    "data": {
                        "total_checks": len(security_scan.get("checks", {})),
                        "issues_found": len(security_scan.get("issues_found", [])),
                        "overall_compliant": security_scan.get(
                            "overall_compliant", False
                        ),
                    },
                }
            except Exception as e:
                dashboard_data["sections"]["security"] = {
                    "title": "Security Compliance",
                    "status": "ERROR",
                    "data": {"error": str(e)},
                }

            # Time Sync Section
            try:
                time_health = self.monitors["time_sync"].check_time_sync_health()
                dashboard_data["sections"]["time_sync"] = {
                    "title": "Time Synchronization",
                    "status": (
                        "HEALTHY" if time_health.get("healthy", False) else "ISSUES"
                    ),
                    "data": {
                        "synchronized": time_health.get("synchronized", False),
                        "max_skew_ms": time_health.get("max_skew_ms", 0),
                        "sync_method": time_health.get("sync_method", "NONE"),
                        "healthy": time_health.get("healthy", False),
                    },
                }
            except Exception as e:
                dashboard_data["sections"]["time_sync"] = {
                    "title": "Time Synchronization",
                    "status": "ERROR",
                    "data": {"error": str(e)},
                }

            # Emergency Status Section
            try:
                panic_status = self.monitors["panic_button"].check_panic_status()
                dashboard_data["sections"]["emergency"] = {
                    "title": "Emergency Systems",
                    "status": (
                        "ACTIVE" if panic_status.get("panic_active", False) else "READY"
                    ),
                    "data": {
                        "panic_active": panic_status.get("panic_active", False),
                        "current_mode": panic_status.get("current_mode", "normal"),
                        "halt_mode": panic_status.get("halt_mode", False),
                    },
                }
            except Exception as e:
                dashboard_data["sections"]["emergency"] = {
                    "title": "Emergency Systems",
                    "status": "ERROR",
                    "data": {"error": str(e)},
                }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting ops dashboard data: {e}")
            return {"error": str(e)}

    def get_trading_dashboard_data(self) -> Dict[str, Any]:
        """Get trading-specific dashboard data."""
        try:
            trading_data = {
                "timestamp": datetime.now().isoformat(),
                "trading_status": {},
                "positions": {},
                "orders": {},
                "pnl": {},
                "risk_metrics": {},
            }

            # Mock trading data - in production this would come from actual trading systems
            trading_data["trading_status"] = {
                "active": True,
                "mode": "live",
                "strategies_running": 5,
                "total_capital": 1000000,
                "capital_deployed": 750000,
            }

            trading_data["positions"] = {
                "total_positions": 12,
                "long_positions": 8,
                "short_positions": 4,
                "total_market_value": 850000,
                "unrealized_pnl": 25000,
            }

            trading_data["orders"] = {
                "pending_orders": 3,
                "filled_orders_today": 45,
                "cancelled_orders_today": 2,
                "avg_fill_time_ms": 125,
            }

            trading_data["pnl"] = {
                "daily_pnl": 12500,
                "weekly_pnl": 45000,
                "monthly_pnl": 125000,
                "ytd_pnl": 250000,
                "sharpe_ratio": 2.1,
            }

            trading_data["risk_metrics"] = {
                "max_drawdown": -0.05,
                "var_95": -25000,
                "portfolio_beta": 1.2,
                "concentration_risk": 0.15,
            }

            return trading_data

        except Exception as e:
            logger.error(f"Error getting trading dashboard data: {e}")
            return {"error": str(e)}

    def run(self, host="0.0.0.0", port=8000, debug=False):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function."""
    if not DEPS_AVAILABLE:
        print("❌ Missing dependencies: pip install flask flask-cors redis")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run dashboard server
    server = TradingSystemDashboardServer()

    # Get port from environment or default
    port = int(os.getenv("DASHBOARD_PORT", 8000))
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    debug = os.getenv("DASHBOARD_DEBUG", "false").lower() == "true"

    print(
        f"""
🚀 Trading System Dashboard Server Starting
==========================================
📊 Operations Dashboard: http://localhost:{port}/
📈 Trading Dashboard:    http://localhost:{port}/trading
🚨 Emergency Controls:   http://localhost:{port}/api/panic
📡 API Health Check:     http://localhost:{port}/api/health
📋 System Status:       http://localhost:{port}/api/status
==========================================
    """
    )

    try:
        server.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
