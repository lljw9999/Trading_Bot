#!/usr/bin/env python3
"""
Portfolio Exposure Limiter
Runtime guard for portfolio-level risk controls before order execution
"""
import os
import json
import yaml
import datetime
import pathlib
import logging
from pathlib import Path
from datetime import timezone
from typing import Dict, Tuple, Any, Optional

from src.utils.dev_env import ensure_dev_cli_scripts


class ExposureLimiter:
    """
    Portfolio exposure and risk limiter for runtime order validation.

    Enforces:
    - Gross/net notional limits
    - Per-venue caps
    - Per-asset percentage caps
    - Exposure by asset class
    - VaR/ES budget limits
    """

    def __init__(self, config_path: str = "pilot/portfolio_pilot.yaml"):
        """
        Initialize exposure limiter with configuration.

        Args:
            config_path: Path to portfolio pilot configuration
        """
        self.logger = logging.getLogger("exposure_limiter")
        self.config = self._load_config(config_path)
        self.limits = self.config.get("portfolio_limits", {})
        self.assets = {
            asset["symbol"]: asset
            for asset in self.config.get("pilot", {}).get("assets", [])
        }

    def _load_config(self, config_path: str) -> dict:
        """Load portfolio configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def _write_audit(
        self, action: str, order_request: dict, reason: str, allowed: bool
    ):
        """Write WORM audit record for exposure check."""
        audit_record = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "action": action,
            "order_request": order_request,
            "reason": reason,
            "allowed": allowed,
            "limits_checked": {
                "gross_notional_usd": self.limits.get("max_gross_notional_usd"),
                "var_95_usd": self.limits.get("max_var_95_usd"),
                "venue_caps": self.limits.get("per_venue_notional_caps", {}),
            },
            "operator": "exposure_limiter",
        }

        audit_root = pathlib.Path("artifacts") / "audit"
        audit_root.mkdir(parents=True, exist_ok=True)
        ts = audit_record["timestamp"].replace(":", "_")
        audit_file = audit_root / f"{ts}_exposure_check.json"

        with open(audit_file, "w") as f:
            json.dump(audit_record, f, indent=2)

        self.logger.info(f"Audit written: {audit_file}")

        ensure_dev_cli_scripts(Path.cwd())

    def _get_portfolio_state(self) -> dict:
        """
        Get current portfolio state (stub implementation).

        In production, this would integrate with position management system.
        """
        # Stub portfolio state for testing
        return {
            "gross_notional_usd": 85000,
            "net_notional_usd": 42000,
            "positions": {
                "SOL-USD": {"notional_usd": 25000, "venue": "coinbase"},
                "BTC-USD": {"notional_usd": 35000, "venue": "binance"},
                "ETH-USD": {"notional_usd": 25000, "venue": "coinbase"},
            },
            "venue_exposure": {"coinbase": 50000, "binance": 35000, "alpaca": 0},
            "var_95_estimate_usd": 8500,
            "class_exposure": {"crypto": 85000, "equity": 0},
        }

    def _check_gross_notional_limit(
        self, order_request: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """Check gross notional limit."""
        max_gross = self.limits.get("max_gross_notional_usd", float("inf"))

        current_gross = portfolio_state.get("gross_notional_usd", 0)
        order_notional = abs(order_request.get("notional_usd", 0))
        projected_gross = current_gross + order_notional

        if projected_gross > max_gross:
            return (
                False,
                f"Gross notional limit exceeded: {projected_gross:,.0f} > {max_gross:,.0f} USD",
            )

        return True, "Gross notional limit OK"

    def _check_venue_limit(
        self, order_request: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """Check per-venue notional limits."""
        venue = order_request.get("venue", "unknown")
        venue_caps = self.limits.get("per_venue_notional_caps", {})

        if venue not in venue_caps:
            return True, f"No limit configured for venue {venue}"

        max_venue = venue_caps[venue]
        current_venue = portfolio_state.get("venue_exposure", {}).get(venue, 0)
        order_notional = abs(order_request.get("notional_usd", 0))
        projected_venue = current_venue + order_notional

        if projected_venue > max_venue:
            return (
                False,
                f"Venue limit exceeded for {venue}: {projected_venue:,.0f} > {max_venue:,.0f} USD",
            )

        return True, f"Venue limit OK for {venue}"

    def _check_asset_percentage_cap(
        self, order_request: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """Check per-asset percentage caps."""
        symbol = order_request.get("symbol", "")

        if symbol not in self.assets:
            return False, f"Asset {symbol} not configured in pilot"

        max_influence_pct = self.assets[symbol].get("max_influence_pct", 0)

        # For this check, we assume order represents the influence percentage directly
        order_influence_pct = order_request.get("influence_pct", 0)

        if order_influence_pct > max_influence_pct:
            return (
                False,
                f"Asset influence cap exceeded for {symbol}: {order_influence_pct}% > {max_influence_pct}%",
            )

        return True, f"Asset influence cap OK for {symbol}"

    def _check_var_budget(
        self, order_request: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """Check VaR budget limit."""
        max_var = self.limits.get("max_var_95_usd", float("inf"))

        current_var = portfolio_state.get("var_95_estimate_usd", 0)
        # Simple stub: assume order adds 10% to current VaR (would be proper calculation in production)
        order_var_contribution = current_var * 0.1
        projected_var = current_var + order_var_contribution

        if projected_var > max_var:
            return (
                False,
                f"VaR budget exceeded: {projected_var:,.0f} > {max_var:,.0f} USD",
            )

        return True, "VaR budget OK"

    def _check_class_exposure(
        self, order_request: dict, portfolio_state: dict
    ) -> Tuple[bool, str]:
        """Check asset class exposure limits."""
        symbol = order_request.get("symbol", "")

        if symbol not in self.assets:
            return False, f"Asset {symbol} not in configured assets"

        asset_class = self.assets[symbol].get("class", "unknown")

        # Simple check: ensure we're not over-concentrating in one class
        current_class_exposure = portfolio_state.get("class_exposure", {}).get(
            asset_class, 0
        )
        total_gross = portfolio_state.get("gross_notional_usd", 1)

        class_percentage = (
            (current_class_exposure / total_gross) * 100 if total_gross > 0 else 0
        )

        # Stub limit: no class should exceed 80% of portfolio
        if class_percentage > 80:
            return (
                False,
                f"Asset class {asset_class} exposure too high: {class_percentage:.1f}% > 80%",
            )

        return True, f"Asset class exposure OK for {asset_class}"

    def enforce(
        self, order_request: dict, portfolio_state: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Main enforcement function - check all limits.

        Args:
            order_request: Order to validate
            portfolio_state: Current portfolio state (optional, will fetch if None)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        try:
            if portfolio_state is None:
                portfolio_state = self._get_portfolio_state()

            # Run all checks
            checks = [
                ("gross_notional", self._check_gross_notional_limit),
                ("venue_limit", self._check_venue_limit),
                ("asset_percentage", self._check_asset_percentage_cap),
                ("var_budget", self._check_var_budget),
                ("class_exposure", self._check_class_exposure),
            ]

            failed_checks = []
            passed_checks = []

            for check_name, check_func in checks:
                try:
                    allowed, reason = check_func(order_request, portfolio_state)
                    if allowed:
                        passed_checks.append(f"{check_name}: {reason}")
                    else:
                        failed_checks.append(f"{check_name}: {reason}")

                except Exception as e:
                    failed_checks.append(f"{check_name}: Error - {e}")

            # Overall result
            overall_allowed = len(failed_checks) == 0

            if overall_allowed:
                reason = f"All checks passed ({len(passed_checks)} checks)"
                self.logger.info(
                    f"Order allowed: {order_request.get('symbol', 'unknown')} - {reason}"
                )
            else:
                reason = f"Failed checks: {'; '.join(failed_checks)}"
                self.logger.warning(
                    f"Order rejected: {order_request.get('symbol', 'unknown')} - {reason}"
                )

            # Write audit record
            self._write_audit("exposure_check", order_request, reason, overall_allowed)

            return overall_allowed, reason

        except Exception as e:
            error_reason = f"Exposure limiter error: {e}"
            self.logger.error(error_reason)
            self._write_audit("exposure_error", order_request, error_reason, False)
            return False, error_reason


def main():
    """CLI interface for testing exposure limiter."""
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio Exposure Limiter")
    parser.add_argument(
        "--config",
        "-c",
        default="pilot/portfolio_pilot.yaml",
        help="Portfolio configuration file",
    )
    parser.add_argument("--test", action="store_true", help="Run test scenario")
    args = parser.parse_args()

    limiter = ExposureLimiter(args.config)

    if args.test:
        # Test scenarios
        test_orders = [
            {
                "symbol": "SOL-USD",
                "notional_usd": 15000,
                "venue": "coinbase",
                "influence_pct": 20,
                "side": "buy",
            },
            {
                "symbol": "BTC-USD",
                "notional_usd": 200000,  # Should exceed gross limit
                "venue": "binance",
                "influence_pct": 15,
                "side": "buy",
            },
            {
                "symbol": "ETH-USD",
                "notional_usd": 80000,  # Should exceed coinbase venue limit
                "venue": "coinbase",
                "influence_pct": 18,
                "side": "buy",
            },
        ]

        for i, order in enumerate(test_orders, 1):
            print(f"\n🧪 Test {i}: {order['symbol']} @ {order['notional_usd']:,} USD")
            allowed, reason = limiter.enforce(order)
            status = "✅ ALLOWED" if allowed else "❌ REJECTED"
            print(f"{status}: {reason}")

    else:
        print("Use --test to run test scenarios")


if __name__ == "__main__":
    main()
