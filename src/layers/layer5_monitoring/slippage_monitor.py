#!/usr/bin/env python3
"""
Real-Time Slippage Monitoring and Adaptive Execution Rules
Monitors execution quality and adjusts trading parameters dynamically
"""

import numpy as np
import pandas as pd
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SlippageAlert:
    """Slippage alert structure"""

    alert_type: str  # 'HIGH_SLIPPAGE', 'VENUE_ISSUE', 'MARKET_IMPACT'
    symbol: str
    venue: str
    slippage_bps: float
    threshold_bps: float
    timestamp: datetime
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "symbol": self.symbol,
            "venue": self.venue,
            "slippage_bps": self.slippage_bps,
            "threshold_bps": self.threshold_bps,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
        }


@dataclass
class MarketCondition:
    """Current market condition assessment"""

    volatility_regime: str  # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
    liquidity_score: float  # 0.0 to 1.0
    spread_environment: str  # 'TIGHT', 'NORMAL', 'WIDE'
    market_stress_level: float  # 0.0 to 1.0
    timestamp: datetime


class SlippageMonitor:
    """Real-time slippage monitoring and analysis"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

        # Sliding window for real-time analysis
        self.execution_window = deque(maxlen=1000)  # Last 1000 executions
        self.slippage_by_venue = {}
        self.slippage_by_symbol = {}
        self.alerts = deque(maxlen=100)  # Last 100 alerts

        # Adaptive thresholds
        self.dynamic_thresholds = {}

        # Performance tracking
        self.venue_performance = {}
        self.market_conditions = None

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for slippage monitoring")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for slippage monitoring"""
        return {
            "slippage_thresholds": {
                "crypto": {"low": 5.0, "medium": 10.0, "high": 20.0, "critical": 50.0},
                "stocks": {"low": 3.0, "medium": 7.0, "high": 15.0, "critical": 30.0},
            },
            "monitoring_windows": {
                "real_time": 60,  # 1 minute
                "short_term": 300,  # 5 minutes
                "medium_term": 900,  # 15 minutes
                "long_term": 3600,  # 1 hour
            },
            "alert_cooldown_seconds": 300,  # 5 minutes between similar alerts
            "market_impact_threshold": 0.1,  # 10 bps market impact threshold
            "venue_comparison_enabled": True,
        }

    def record_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record execution and calculate slippage metrics"""

        # Calculate slippage
        intended_price = execution_data.get("intended_price", 0)
        executed_price = execution_data.get("executed_price", 0)
        side = execution_data.get("side", "buy").lower()

        if intended_price > 0:
            if side == "buy":
                slippage_bps = (
                    (executed_price - intended_price) / intended_price
                ) * 10000
            else:  # sell
                slippage_bps = (
                    (intended_price - executed_price) / intended_price
                ) * 10000
        else:
            slippage_bps = 0.0

        # Enrich execution data
        enriched_execution = {
            **execution_data,
            "slippage_bps": slippage_bps,
            "timestamp": datetime.now(),
            "processing_timestamp": datetime.now().isoformat(),
        }

        # Add to sliding window
        self.execution_window.append(enriched_execution)

        # Update venue and symbol tracking
        symbol = execution_data.get("symbol", "UNKNOWN")
        venue = execution_data.get("venue", "UNKNOWN")

        self._update_venue_tracking(venue, enriched_execution)
        self._update_symbol_tracking(symbol, enriched_execution)

        # Check for alerts
        alerts = self._check_slippage_alerts(enriched_execution)

        # Store in Redis
        self._store_execution_data(enriched_execution)

        logger.info(
            f"📊 Execution recorded: {symbol} on {venue}, "
            f"Slippage: {slippage_bps:.1f}bps"
        )

        return {
            "execution": enriched_execution,
            "alerts": alerts,
            "analysis": self._analyze_recent_performance(),
        }

    def _update_venue_tracking(self, venue: str, execution: Dict[str, Any]):
        """Update venue-specific performance tracking"""
        if venue not in self.venue_performance:
            self.venue_performance[venue] = {
                "executions": deque(maxlen=200),
                "avg_slippage_bps": 0.0,
                "success_rate": 1.0,
                "last_updated": datetime.now(),
            }

        venue_data = self.venue_performance[venue]
        venue_data["executions"].append(execution)

        # Calculate rolling statistics
        recent_slippages = [ex["slippage_bps"] for ex in venue_data["executions"]]
        venue_data["avg_slippage_bps"] = np.mean(recent_slippages)
        venue_data["last_updated"] = datetime.now()

    def _update_symbol_tracking(self, symbol: str, execution: Dict[str, Any]):
        """Update symbol-specific performance tracking"""
        if symbol not in self.slippage_by_symbol:
            self.slippage_by_symbol[symbol] = deque(maxlen=100)

        self.slippage_by_symbol[symbol].append(execution)

    def _check_slippage_alerts(self, execution: Dict[str, Any]) -> List[SlippageAlert]:
        """Check for slippage alerts and anomalies"""
        alerts = []

        symbol = execution.get("symbol", "UNKNOWN")
        venue = execution.get("venue", "UNKNOWN")
        slippage_bps = execution.get("slippage_bps", 0)

        # Determine asset type for thresholds
        asset_type = (
            "crypto"
            if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA"])
            else "stocks"
        )
        thresholds = self.config["slippage_thresholds"][asset_type]

        # Check absolute slippage thresholds
        if abs(slippage_bps) >= thresholds["critical"]:
            severity = "CRITICAL"
        elif abs(slippage_bps) >= thresholds["high"]:
            severity = "HIGH"
        elif abs(slippage_bps) >= thresholds["medium"]:
            severity = "MEDIUM"
        elif abs(slippage_bps) >= thresholds["low"]:
            severity = "LOW"
        else:
            severity = None

        if severity:
            alert = SlippageAlert(
                alert_type="HIGH_SLIPPAGE",
                symbol=symbol,
                venue=venue,
                slippage_bps=slippage_bps,
                threshold_bps=thresholds[severity.lower()],
                timestamp=datetime.now(),
                severity=severity,
            )
            alerts.append(alert)
            self.alerts.append(alert)

            # Send alert to monitoring system
            self._send_alert(alert)

        # Check for venue-specific issues
        venue_alerts = self._check_venue_performance(venue, execution)
        alerts.extend(venue_alerts)

        return alerts

    def _check_venue_performance(
        self, venue: str, execution: Dict[str, Any]
    ) -> List[SlippageAlert]:
        """Check for venue-specific performance issues"""
        alerts = []

        if venue not in self.venue_performance:
            return alerts

        venue_data = self.venue_performance[venue]
        recent_executions = list(venue_data["executions"])

        if len(recent_executions) < 5:
            return alerts

        # Check for deteriorating performance
        recent_slippages = [ex["slippage_bps"] for ex in recent_executions[-5:]]
        avg_recent_slippage = np.mean([abs(s) for s in recent_slippages])

        # Compare with venue's historical performance
        all_slippages = [ex["slippage_bps"] for ex in recent_executions]
        historical_avg = (
            np.mean([abs(s) for s in all_slippages[:-5]])
            if len(all_slippages) > 5
            else 0
        )

        # Alert if recent performance is significantly worse
        if historical_avg > 0 and avg_recent_slippage > historical_avg * 2.0:
            alert = SlippageAlert(
                alert_type="VENUE_ISSUE",
                symbol=execution.get("symbol", "UNKNOWN"),
                venue=venue,
                slippage_bps=avg_recent_slippage,
                threshold_bps=historical_avg * 2.0,
                timestamp=datetime.now(),
                severity="MEDIUM",
            )
            alerts.append(alert)
            self.alerts.append(alert)

        return alerts

    def _send_alert(self, alert: SlippageAlert):
        """Send alert to monitoring systems"""
        try:
            if self.redis_client:
                # Store in Redis for dashboard
                alert_key = f"slippage_alert:{int(datetime.now().timestamp())}"
                self.redis_client.setex(alert_key, 3600, json.dumps(alert.to_dict()))

                # Publish to alert channel
                self.redis_client.publish(
                    "slippage_alerts", json.dumps(alert.to_dict())
                )

            # Log alert
            logger.warning(
                f"🚨 {alert.severity} slippage alert: {alert.symbol} on {alert.venue}, "
                f"Slippage: {alert.slippage_bps:.1f}bps (threshold: {alert.threshold_bps:.1f}bps)"
            )

        except Exception as e:
            logger.error(f"Error sending slippage alert: {e}")

    def assess_market_conditions(self) -> MarketCondition:
        """Assess current market conditions for adaptive execution"""

        if not self.execution_window:
            return MarketCondition("NORMAL", 0.5, "NORMAL", 0.0, datetime.now())

        # Analyze recent executions
        recent_executions = list(self.execution_window)[-50:]  # Last 50 executions

        # Volatility assessment based on slippage variance
        slippages = [abs(ex["slippage_bps"]) for ex in recent_executions]
        slippage_std = np.std(slippages) if slippages else 0

        if slippage_std < 2.0:
            volatility_regime = "LOW"
        elif slippage_std < 5.0:
            volatility_regime = "NORMAL"
        elif slippage_std < 10.0:
            volatility_regime = "HIGH"
        else:
            volatility_regime = "EXTREME"

        # Liquidity assessment based on execution success and slippage
        avg_slippage = np.mean(slippages) if slippages else 0
        liquidity_score = max(
            0.0, min(1.0, 1.0 - (avg_slippage / 20.0))
        )  # Normalize to 0-1

        # Spread environment (simplified)
        spreads = [
            ex.get("spread_bps", 5.0) for ex in recent_executions if "spread_bps" in ex
        ]
        avg_spread = np.mean(spreads) if spreads else 5.0

        if avg_spread < 3.0:
            spread_environment = "TIGHT"
        elif avg_spread < 8.0:
            spread_environment = "NORMAL"
        else:
            spread_environment = "WIDE"

        # Market stress level
        high_slippage_rate = (
            sum(1 for s in slippages if s > 10.0) / len(slippages) if slippages else 0
        )
        market_stress_level = min(1.0, high_slippage_rate * 2.0)

        self.market_conditions = MarketCondition(
            volatility_regime=volatility_regime,
            liquidity_score=liquidity_score,
            spread_environment=spread_environment,
            market_stress_level=market_stress_level,
            timestamp=datetime.now(),
        )

        return self.market_conditions

    def get_adaptive_execution_params(self, symbol: str) -> Dict[str, Any]:
        """Get adaptive execution parameters based on current conditions"""

        conditions = self.assess_market_conditions()

        # Base parameters
        params = {
            "order_type": "marketable_limit",
            "aggressiveness": 0.5,
            "max_slippage_bps": 10.0,
            "use_twap": False,
            "delay_trade": False,
        }

        # Adjust based on volatility
        if conditions.volatility_regime == "EXTREME":
            params["order_type"] = "limit"
            params["aggressiveness"] = 0.2
            params["max_slippage_bps"] = 15.0
            params["use_twap"] = True
        elif conditions.volatility_regime == "HIGH":
            params["aggressiveness"] = 0.3
            params["max_slippage_bps"] = 8.0
        elif conditions.volatility_regime == "LOW":
            params["order_type"] = "market"
            params["aggressiveness"] = 0.8
            params["max_slippage_bps"] = 5.0

        # Adjust based on liquidity
        if conditions.liquidity_score < 0.3:
            params["use_twap"] = True
            params["aggressiveness"] = min(params["aggressiveness"], 0.3)

        # Adjust based on spread environment
        if conditions.spread_environment == "WIDE":
            params["delay_trade"] = True
            params["max_slippage_bps"] *= 1.5

        # Adjust based on market stress
        if conditions.market_stress_level > 0.7:
            params["delay_trade"] = True
            params["order_type"] = "limit"

        logger.info(f"📋 Adaptive params for {symbol}: {params}")

        return params

    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent execution performance"""

        if not self.execution_window:
            return {"status": "no_data"}

        recent_executions = list(self.execution_window)[-20:]  # Last 20 executions

        # Performance metrics
        slippages = [ex["slippage_bps"] for ex in recent_executions]

        analysis = {
            "total_executions": len(recent_executions),
            "avg_slippage_bps": np.mean([abs(s) for s in slippages]),
            "median_slippage_bps": np.median([abs(s) for s in slippages]),
            "max_slippage_bps": np.max([abs(s) for s in slippages]),
            "std_slippage_bps": np.std([abs(s) for s in slippages]),
            "positive_slippage_rate": sum(1 for s in slippages if s > 0)
            / len(slippages),
            "high_slippage_rate": sum(1 for s in slippages if abs(s) > 10)
            / len(slippages),
            "timestamp": datetime.now().isoformat(),
        }

        # Venue breakdown
        venue_breakdown = {}
        for ex in recent_executions:
            venue = ex.get("venue", "UNKNOWN")
            if venue not in venue_breakdown:
                venue_breakdown[venue] = {"count": 0, "avg_slippage": 0.0}
            venue_breakdown[venue]["count"] += 1
            venue_breakdown[venue]["avg_slippage"] += abs(ex["slippage_bps"])

        for venue in venue_breakdown:
            if venue_breakdown[venue]["count"] > 0:
                venue_breakdown[venue]["avg_slippage"] /= venue_breakdown[venue][
                    "count"
                ]

        analysis["venue_breakdown"] = venue_breakdown

        return analysis

    def _store_execution_data(self, execution: Dict[str, Any]):
        """Store execution data in Redis"""
        if not self.redis_client:
            return

        try:
            # Store individual execution
            key = f"execution_monitor:{execution.get('symbol', 'UNKNOWN')}:{int(datetime.now().timestamp())}"
            self.redis_client.setex(
                key, 7200, json.dumps(execution, default=str)
            )  # 2 hour expiry

            # Update real-time metrics
            metrics_key = "slippage_metrics"
            current_metrics = self.redis_client.get(metrics_key)

            if current_metrics:
                metrics = json.loads(current_metrics)
            else:
                metrics = {
                    "total_executions": 0,
                    "total_slippage_bps": 0.0,
                    "avg_slippage_bps": 0.0,
                }

            # Update metrics
            metrics["total_executions"] += 1
            metrics["total_slippage_bps"] += abs(execution["slippage_bps"])
            metrics["avg_slippage_bps"] = (
                metrics["total_slippage_bps"] / metrics["total_executions"]
            )
            metrics["last_execution"] = execution
            metrics["last_updated"] = datetime.now().isoformat()

            self.redis_client.setex(metrics_key, 3600, json.dumps(metrics, default=str))

        except Exception as e:
            logger.error(f"Error storing execution data: {e}")

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""

        conditions = self.assess_market_conditions()
        analysis = self._analyze_recent_performance()

        # Recent alerts
        recent_alerts = [alert.to_dict() for alert in list(self.alerts)[-10:]]

        # Venue performance summary
        venue_summary = {}
        for venue, data in self.venue_performance.items():
            venue_summary[venue] = {
                "avg_slippage_bps": data["avg_slippage_bps"],
                "execution_count": len(data["executions"]),
                "last_updated": data["last_updated"].isoformat(),
            }

        dashboard_data = {
            "market_conditions": {
                "volatility_regime": conditions.volatility_regime,
                "liquidity_score": conditions.liquidity_score,
                "spread_environment": conditions.spread_environment,
                "market_stress_level": conditions.market_stress_level,
                "timestamp": conditions.timestamp.isoformat(),
            },
            "performance_analysis": analysis,
            "recent_alerts": recent_alerts,
            "venue_performance": venue_summary,
            "total_executions": len(self.execution_window),
            "alert_count": len(self.alerts),
            "last_updated": datetime.now().isoformat(),
        }

        return dashboard_data


def generate_demo_executions():
    """Generate demo execution data for testing"""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "NVDA"]
    venues = ["binance", "coinbase", "alpaca"]

    executions = []
    base_time = datetime.now() - timedelta(minutes=30)

    for i in range(50):
        symbol = np.random.choice(symbols)
        venue = np.random.choice(venues)

        # Simulate varying slippage conditions
        if i > 40:  # Recent executions with higher slippage
            slippage_base = 8.0
        elif i > 30:
            slippage_base = 5.0
        else:
            slippage_base = 3.0

        execution = {
            "symbol": symbol,
            "venue": venue,
            "side": np.random.choice(["buy", "sell"]),
            "intended_price": 100.0,
            "executed_price": 100.0 + np.random.normal(0, slippage_base / 100),
            "quantity": np.random.uniform(0.1, 5.0),
            "spread_bps": np.random.uniform(2, 8),
            "timestamp": base_time + timedelta(minutes=i),
        }

        executions.append(execution)

    return executions


def main():
    """Demo function for Slippage Monitor"""
    print("🚀 Real-Time Slippage Monitoring System")
    print("=" * 80)

    # Initialize monitor
    monitor = SlippageMonitor()

    # Generate and process demo executions
    demo_executions = generate_demo_executions()

    print(f"📊 Processing {len(demo_executions)} demo executions...")

    alerts_generated = 0
    for execution in demo_executions:
        result = monitor.record_execution(execution)
        if result["alerts"]:
            alerts_generated += len(result["alerts"])

    print(f"   ✅ Processed {len(demo_executions)} executions")
    print(f"   🚨 Generated {alerts_generated} alerts")

    # Get market conditions assessment
    conditions = monitor.assess_market_conditions()
    print(f"\n🌍 Market Conditions Assessment:")
    print(f"   Volatility Regime: {conditions.volatility_regime}")
    print(f"   Liquidity Score: {conditions.liquidity_score:.2f}")
    print(f"   Spread Environment: {conditions.spread_environment}")
    print(f"   Market Stress Level: {conditions.market_stress_level:.2f}")

    # Get adaptive execution parameters
    print(f"\n📋 Adaptive Execution Parameters:")
    for symbol in ["BTCUSDT", "ETHUSDT", "NVDA"]:
        params = monitor.get_adaptive_execution_params(symbol)
        print(
            f"   {symbol}: {params['order_type']}, aggressiveness={params['aggressiveness']:.1f}"
        )

    # Get dashboard data
    dashboard_data = monitor.get_monitoring_dashboard_data()
    print(f"\n📈 Performance Summary:")
    analysis = dashboard_data["performance_analysis"]
    if analysis.get("status") != "no_data":
        print(f"   Total Executions: {analysis['total_executions']}")
        print(f"   Average Slippage: {analysis['avg_slippage_bps']:.1f}bps")
        print(f"   High Slippage Rate: {analysis['high_slippage_rate']:.1%}")

    print("\n🎉 Slippage Monitoring Demo Complete!")
    print("✅ Real-time execution tracking")
    print("✅ Adaptive threshold management")
    print("✅ Market condition assessment")
    print("✅ Venue performance analysis")
    print("✅ Alert generation and routing")


if __name__ == "__main__":
    main()
