#!/usr/bin/env python3
"""
SOL Asset Onboarding
End-to-end onboarding of SOL with risk caps and monitoring
"""

import os
import sys
import json
import time
import logging
import asyncio
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("sol_onboarding")


class SOLAssetOnboarder:
    """SOL asset onboarding service."""

    def __init__(self):
        """Initialize SOL onboarding."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK", "")

        # SOL configuration
        self.sol_config = {
            "symbol": "SOL",
            "pairs": {"binance": "SOLUSDT", "coinbase": "SOL-USD"},
            "risk_caps": {
                "max_gross_pct": 0.15,  # 15% of equity
                "max_single_trade_var": 0.005,  # 0.5% VaR cap
                "max_position_size": 50000,  # $50k max position
                "max_daily_volume": 200000,  # $200k daily volume cap
                "concentration_limit": 0.10,  # 10% portfolio concentration
            },
            "features": {
                "ob_pressure": True,
                "ma_momentum": True,
                "news_sentiment": True,
                "funding_basis": True,
                "lstm_alpha": True,
                "onchain_alpha": True,
            },
            "execution": {
                "smart_routing": True,
                "rl_shadow": True,
                "paper_mode": True,  # Start in paper mode
                "live_mode": False,  # Promote via A/B gate
            },
            "monitoring": {
                "pnl_tracking": True,
                "slippage_tracking": True,
                "entropy_tracking": True,
                "risk_monitoring": True,
            },
        }

        logger.info("🪙 SOL Asset Onboarder initialized")
        logger.info(f"   Pairs: {self.sol_config['pairs']}")
        logger.info(
            f"   Max gross: {self.sol_config['risk_caps']['max_gross_pct']*100}%"
        )
        logger.info(f"   Paper mode: {self.sol_config['execution']['paper_mode']}")

    def setup_layer0_data_ingestion(self) -> bool:
        """Setup Layer 0 data ingestion for SOL."""
        try:
            logger.info("🔌 Setting up Layer 0 - Data Ingestion")

            # Enable SOL symbols in existing connectors
            enabled_symbols = []

            # Binance connector
            binance_symbols = self.redis.get("connectors:binance:symbols")
            if binance_symbols:
                symbols_list = json.loads(binance_symbols)
            else:
                symbols_list = ["BTCUSDT", "ETHUSDT"]

            if self.sol_config["pairs"]["binance"] not in symbols_list:
                symbols_list.append(self.sol_config["pairs"]["binance"])
                enabled_symbols.append(f"Binance:{self.sol_config['pairs']['binance']}")

            self.redis.set("connectors:binance:symbols", json.dumps(symbols_list))

            # Coinbase connector
            coinbase_symbols = self.redis.get("connectors:coinbase:symbols")
            if coinbase_symbols:
                cb_symbols_list = json.loads(coinbase_symbols)
            else:
                cb_symbols_list = ["BTC-USD", "ETH-USD"]

            if self.sol_config["pairs"]["coinbase"] not in cb_symbols_list:
                cb_symbols_list.append(self.sol_config["pairs"]["coinbase"])
                enabled_symbols.append(
                    f"Coinbase:{self.sol_config['pairs']['coinbase']}"
                )

            self.redis.set("connectors:coinbase:symbols", json.dumps(cb_symbols_list))

            # Setup FeatureBus inclusion
            feature_bus_assets = self.redis.get("feature_bus:assets")
            if feature_bus_assets:
                assets_list = json.loads(feature_bus_assets)
            else:
                assets_list = ["BTC", "ETH"]

            if "SOL" not in assets_list:
                assets_list.append("SOL")

            self.redis.set("feature_bus:assets", json.dumps(assets_list))

            # Mock some initial SOL market data
            self._setup_mock_sol_data()

            logger.info(f"✅ Layer 0 setup complete: {enabled_symbols}")
            return True

        except Exception as e:
            logger.error(f"Error setting up Layer 0: {e}")
            return False

    def setup_layer1_alpha_models(self) -> bool:
        """Setup Layer 1 alpha models for SOL."""
        try:
            logger.info("🧠 Setting up Layer 1 - Alpha Models")

            # Enable features for SOL
            for feature_name, enabled in self.sol_config["features"].items():
                if enabled:
                    feature_key = f"alpha:sol:{feature_name}:enabled"
                    self.redis.set(feature_key, 1)

            # Setup feature weights for SOL
            sol_feature_weights = {
                "ob_pressure": 0.15,
                "ma_momentum": 0.20,
                "news_sentiment": 0.10,
                "funding_basis": 0.12,
                "lstm_alpha": 0.18,
                "onchain_alpha": 0.25,
            }

            self.redis.hset("feature_weights:sol", mapping=sol_feature_weights)

            # Initialize SOL alpha signals in Redis
            self._initialize_sol_alpha_signals()

            logger.info(
                f"✅ Layer 1 setup complete: {len([f for f, e in self.sol_config['features'].items() if e])} features"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting up Layer 1: {e}")
            return False

    def setup_layer3_risk_management(self) -> bool:
        """Setup Layer 3 risk management for SOL."""
        try:
            logger.info("⚖️ Setting up Layer 3 - Risk Management")

            # Set SOL risk caps
            risk_caps = self.sol_config["risk_caps"]

            self.redis.hset(
                "risk_caps:sol",
                mapping={
                    "max_gross_pct": risk_caps["max_gross_pct"],
                    "max_single_trade_var": risk_caps["max_single_trade_var"],
                    "max_position_size": risk_caps["max_position_size"],
                    "max_daily_volume": risk_caps["max_daily_volume"],
                    "concentration_limit": risk_caps["concentration_limit"],
                },
            )

            # Initialize SOL risk metrics
            self.redis.hset(
                "risk_metrics:sol",
                mapping={
                    "current_exposure": 0.0,
                    "daily_volume": 0.0,
                    "var_contribution": 0.0,
                    "concentration": 0.0,
                    "last_update": time.time(),
                },
            )

            # Add SOL to portfolio risk calculations
            portfolio_assets = self.redis.get("portfolio:assets")
            if portfolio_assets:
                assets_list = json.loads(portfolio_assets)
            else:
                assets_list = ["BTC", "ETH"]

            if "SOL" not in assets_list:
                assets_list.append("SOL")

            self.redis.set("portfolio:assets", json.dumps(assets_list))

            logger.info(
                f"✅ Layer 3 setup complete: Max gross {risk_caps['max_gross_pct']*100}%"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting up Layer 3: {e}")
            return False

    def setup_layer4_execution(self) -> bool:
        """Setup Layer 4 execution for SOL."""
        try:
            logger.info("🚀 Setting up Layer 4 - Execution")

            # Whitelist SOL for Smart Order Router
            sor_whitelist = self.redis.get("sor:whitelist")
            if sor_whitelist:
                whitelist = json.loads(sor_whitelist)
            else:
                whitelist = ["BTC", "ETH"]

            if "SOL" not in whitelist:
                whitelist.append("SOL")

            self.redis.set("sor:whitelist", json.dumps(whitelist))

            # Setup SOL routing configuration
            sol_routing_config = {
                "primary_venue": "binance",
                "backup_venues": json.dumps(["coinbase"]),  # Convert list to JSON
                "min_size_usd": 100,
                "max_size_usd": self.sol_config["risk_caps"]["max_position_size"],
                "slippage_limit_bps": 20,
                "timeout_ms": 5000,
            }

            self.redis.hset("sor:config:sol", mapping=sol_routing_config)

            # Enable RL shadow trading for SOL
            if self.sol_config["execution"]["rl_shadow"]:
                self.redis.set("rl:shadow:sol:enabled", 1)

            # Set execution flags
            execution_flags = {
                "sol_paper_mode": (
                    1 if self.sol_config["execution"]["paper_mode"] else 0
                ),
                "sol_live_mode": 1 if self.sol_config["execution"]["live_mode"] else 0,
                "sol_smart_routing": (
                    1 if self.sol_config["execution"]["smart_routing"] else 0
                ),
                "sol_rl_shadow": 1 if self.sol_config["execution"]["rl_shadow"] else 0,
            }

            self.redis.hset("execution:flags", mapping=execution_flags)

            logger.info("✅ Layer 4 setup complete: Smart routing enabled")
            return True

        except Exception as e:
            logger.error(f"Error setting up Layer 4: {e}")
            return False

    def setup_monitoring_dashboards(self) -> bool:
        """Setup monitoring dashboards for SOL."""
        try:
            logger.info("📊 Setting up Monitoring")

            # Initialize SOL metrics
            sol_metrics = {
                "pnl_total": 0.0,
                "pnl_realized": 0.0,
                "pnl_unrealized": 0.0,
                "trades_count": 0,
                "volume_total": 0.0,
                "slippage_avg": 0.0,
                "slippage_worst": 0.0,
                "hit_rate": 0.0,
                "sharpe_1h": 0.0,
                "entropy": 0.0,
                "q_spread": 0.0,
                "last_update": time.time(),
            }

            self.redis.hset("metrics:sol", mapping=sol_metrics)

            # Add SOL to Prometheus metrics
            prometheus_assets = self.redis.get("prometheus:assets")
            if prometheus_assets:
                assets_list = json.loads(prometheus_assets)
            else:
                assets_list = ["BTC", "ETH"]

            if "SOL" not in assets_list:
                assets_list.append("SOL")

            self.redis.set("prometheus:assets", json.dumps(assets_list))

            # Initialize SOL-specific Prometheus metrics
            prometheus_metrics = [
                "sol_pnl_total",
                "sol_pnl_realized",
                "sol_position_size",
                "sol_slippage_bps",
                "sol_entropy",
                "sol_q_spread",
                "sol_trade_count",
            ]

            for metric in prometheus_metrics:
                self.redis.set(f"metric:{metric}", 0.0)

            logger.info(
                f"✅ Monitoring setup complete: {len(prometheus_metrics)} metrics"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting up monitoring: {e}")
            return False

    def _setup_mock_sol_data(self):
        """Setup mock SOL market data for testing."""
        try:
            # Mock SOL price and market data
            sol_price = 180.50  # Mock SOL price

            # Add to market data streams
            sol_tick = {
                "ts": int(time.time() * 1000),
                "price": sol_price,
                "qty": 100.0,
                "symbol": "SOLUSDT",
            }

            self.redis.rpush("market.raw.crypto.SOLUSDT", json.dumps(sol_tick))

            # Add SOL orderbook data
            sol_orderbook = {
                "bids": [[sol_price - 0.05, 500], [sol_price - 0.10, 1000]],
                "asks": [[sol_price + 0.05, 500], [sol_price + 0.10, 1000]],
                "timestamp": time.time(),
            }

            self.redis.set("orderbook:sol", json.dumps(sol_orderbook))

            # Add to price feeds
            self.redis.set("price:sol:usd", sol_price)
            self.redis.set("price:sol:btc", sol_price / 97500)  # SOL/BTC ratio

            logger.debug(f"Mock SOL data: ${sol_price:.2f}")

        except Exception as e:
            logger.error(f"Error setting up mock SOL data: {e}")

    def _initialize_sol_alpha_signals(self):
        """Initialize SOL alpha signals."""
        try:
            # Mock realistic alpha signals for SOL
            alpha_signals = {
                "ob_pressure": np.random.normal(
                    0.02, 0.15
                ),  # Slight bullish OB pressure
                "ma_momentum": np.random.normal(0.05, 0.20),  # Positive momentum
                "news_sentiment": np.random.normal(0.1, 0.25),  # Slightly positive news
                "funding_basis": np.random.normal(
                    -0.01, 0.10
                ),  # Small negative funding
                "lstm_alpha": np.random.normal(0.03, 0.18),  # LSTM bullish
                "onchain_alpha": np.random.normal(0.08, 0.30),  # Strong onchain signal
            }

            # Store alpha signals
            for signal, value in alpha_signals.items():
                self.redis.set(f"alpha:sol:{signal}", float(value))

            # Calculate composite alpha
            weights = [0.15, 0.20, 0.10, 0.12, 0.18, 0.25]  # Match feature weights
            composite_alpha = sum(
                alpha_signals[signal] * weight
                for signal, weight in zip(alpha_signals.keys(), weights)
            )

            self.redis.set("alpha:sol:composite", composite_alpha)

            logger.debug(f"Alpha signals: composite={composite_alpha:.4f}")

        except Exception as e:
            logger.error(f"Error initializing alpha signals: {e}")

    def validate_onboarding(self) -> Dict[str, bool]:
        """Validate SOL onboarding status."""
        try:
            validation_results = {}

            # Check Layer 0
            binance_symbols = self.redis.get("connectors:binance:symbols")
            validation_results["layer0_binance"] = (
                binance_symbols and "SOLUSDT" in json.loads(binance_symbols)
            )

            # Check Layer 1
            validation_results["layer1_features"] = bool(
                self.redis.get("alpha:sol:ob_pressure:enabled")
            )

            # Check Layer 3
            validation_results["layer3_risk_caps"] = bool(
                self.redis.hget("risk_caps:sol", "max_gross_pct")
            )

            # Check Layer 4
            validation_results["layer4_execution"] = bool(
                self.redis.hget("execution:flags", "sol_paper_mode")
            )

            # Check Monitoring
            validation_results["monitoring"] = bool(
                self.redis.hget("metrics:sol", "pnl_total")
            )

            # Check market data
            validation_results["market_data"] = bool(self.redis.get("price:sol:usd"))

            all_passed = all(validation_results.values())

            logger.info(
                f"🔍 Validation results: {sum(validation_results.values())}/{len(validation_results)} passed"
            )

            return {"all_passed": all_passed, "details": validation_results}

        except Exception as e:
            logger.error(f"Error validating onboarding: {e}")
            return {"all_passed": False, "error": str(e)}

    def get_sol_status_report(self) -> Dict:
        """Get comprehensive SOL status report."""
        try:
            # Get current metrics
            sol_metrics = self.redis.hgetall("metrics:sol")
            risk_metrics = self.redis.hgetall("risk_metrics:sol")
            execution_flags = self.redis.hgetall("execution:flags")

            # Get alpha signals
            alpha_signals = {}
            for feature in self.sol_config["features"]:
                value = self.redis.get(f"alpha:sol:{feature}")
                alpha_signals[feature] = float(value) if value else 0.0

            composite_alpha = self.redis.get("alpha:sol:composite")
            alpha_signals["composite"] = (
                float(composite_alpha) if composite_alpha else 0.0
            )

            # Get current price
            sol_price = self.redis.get("price:sol:usd")

            status = {
                "service": "sol_onboarding",
                "timestamp": time.time(),
                "asset": "SOL",
                "price_usd": float(sol_price) if sol_price else 0.0,
                "config": self.sol_config,
                "metrics": {k: float(v) if v else 0.0 for k, v in sol_metrics.items()},
                "risk_metrics": {
                    k: float(v) if v else 0.0 for k, v in risk_metrics.items()
                },
                "alpha_signals": alpha_signals,
                "execution_flags": {
                    k: bool(int(v)) if v and v.isdigit() else False
                    for k, v in execution_flags.items()
                    if k.startswith("sol_")
                },
                "validation": self.validate_onboarding(),
            }

            return status

        except Exception as e:
            return {"service": "sol_onboarding", "status": "error", "error": str(e)}

    def send_onboarding_notification(self, message: str) -> bool:
        """Send onboarding notification to Slack."""
        try:
            if not self.slack_webhook:
                return False

            payload = {
                "text": message,
                "username": "SOL Onboarding",
                "icon_emoji": ":coin:",
            }

            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("📱 Sent onboarding notification to Slack")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def run_complete_onboarding(self) -> bool:
        """Run complete SOL onboarding process."""
        try:
            logger.info("🚀 Starting complete SOL onboarding...")

            # Step-by-step onboarding
            steps = [
                ("Layer 0 - Data Ingestion", self.setup_layer0_data_ingestion),
                ("Layer 1 - Alpha Models", self.setup_layer1_alpha_models),
                ("Layer 3 - Risk Management", self.setup_layer3_risk_management),
                ("Layer 4 - Execution", self.setup_layer4_execution),
                ("Monitoring", self.setup_monitoring_dashboards),
            ]

            results = []
            for step_name, step_func in steps:
                logger.info(f"📋 Running: {step_name}")
                success = step_func()
                results.append((step_name, success))

                if not success:
                    logger.error(f"❌ Failed: {step_name}")
                    break
                else:
                    logger.info(f"✅ Completed: {step_name}")

            # Validate complete setup
            validation = self.validate_onboarding()
            all_successful = validation["all_passed"]

            if all_successful:
                logger.info("🎉 SOL onboarding completed successfully!")

                # Send success notification
                self.send_onboarding_notification(
                    "🪙 SOL Asset Onboarded Successfully!\n"
                    f"• All layers configured\n"
                    f"• Risk caps: {self.sol_config['risk_caps']['max_gross_pct']*100}% max gross\n"
                    f"• Features: {len([f for f, e in self.sol_config['features'].items() if e])}/6 enabled\n"
                    f"• Mode: Paper trading (A/B gate for live promotion)"
                )

                return True
            else:
                logger.error("❌ SOL onboarding validation failed")
                return False

        except Exception as e:
            logger.error(f"Error in complete onboarding: {e}")
            return False


def main():
    """Main entry point for SOL onboarding."""
    import argparse

    parser = argparse.ArgumentParser(description="SOL Asset Onboarding")
    parser.add_argument(
        "--onboard", action="store_true", help="Run complete SOL onboarding"
    )
    parser.add_argument("--status", action="store_true", help="Show SOL status report")
    parser.add_argument(
        "--validate", action="store_true", help="Validate SOL onboarding"
    )

    args = parser.parse_args()

    # Create onboarder
    onboarder = SOLAssetOnboarder()

    if args.status:
        # Show status report
        status = onboarder.get_sol_status_report()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.validate:
        # Run validation
        validation = onboarder.validate_onboarding()
        print(json.dumps(validation, indent=2, default=str))
        return

    if args.onboard:
        # Run complete onboarding
        success = onboarder.run_complete_onboarding()

        if success:
            logger.info("✅ SOL onboarding completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ SOL onboarding failed")
            sys.exit(1)

    # Default action - show help
    parser.print_help()


if __name__ == "__main__":
    main()
