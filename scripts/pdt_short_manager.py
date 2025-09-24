#!/usr/bin/env python3
"""
PDT and Short Locate Manager

Manages Pattern Day Trading rules and short selling requirements:
- PDT account classification and monitoring
- Day trading buying power calculations
- Short locate requirements and inventory tracking
- Short sale rule (SSR) compliance
- Automated position size limits based on account type
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("pdt_short_manager")


class PDTShortManager:
    """
    Manages PDT rules and short selling compliance.
    Ensures regulatory compliance for equity trading operations.
    """

    def __init__(self):
        """Initialize PDT and short manager."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # PDT and short selling configuration
        self.config = {
            "pdt_rules": {
                "minimum_account_value": 25000,  # $25k minimum for PDT
                "day_trading_buying_power_ratio": 4.0,  # 4:1 intraday leverage
                "overnight_buying_power_ratio": 2.0,  # 2:1 overnight leverage
                "day_trade_limit_non_pdt": 3,  # 3 day trades per 5 business days
                "monitoring_period_days": 5,  # Rolling 5 business day window
            },
            "short_selling": {
                "locate_required": True,  # Require locate for short sales
                "ssr_compliance": True,  # Comply with Short Sale Rule
                "hard_to_borrow_threshold": 5.0,  # >5% borrow rate = HTB
                "max_short_position_pct": 0.30,  # Max 30% of portfolio short
                "locate_sources": ["prime_broker", "clearing_firm", "third_party"],
            },
            "account_types": {
                "cash": {"day_trading": False, "short_selling": False},
                "margin": {"day_trading": False, "short_selling": True},
                "pdt": {"day_trading": True, "short_selling": True},
            },
            "restricted_securities": {
                "no_short_list": ["IPO_STOCKS"],  # Recent IPOs typically not shortable
                "locate_exempt": [
                    "ETF",
                    "INDEX_FUNDS",
                ],  # Some ETFs have locate exemptions
                "enhanced_margin": [],  # Securities requiring enhanced margin
            },
        }

        logger.info("Initialized PDT and short selling manager")

    def classify_account_type(
        self, account_value: float, day_trades_5d: int
    ) -> Dict[str, any]:
        """
        Classify account type based on value and trading activity.

        Args:
            account_value: Current account value in USD
            day_trades_5d: Number of day trades in last 5 business days

        Returns:
            Account classification and capabilities
        """
        try:
            min_pdt_value = self.config["pdt_rules"]["minimum_account_value"]
            max_day_trades = self.config["pdt_rules"]["day_trade_limit_non_pdt"]

            # Determine account classification
            if account_value >= min_pdt_value:
                if day_trades_5d > max_day_trades:
                    account_type = "pdt"  # Pattern Day Trader
                else:
                    account_type = "margin"  # Eligible for PDT but not flagged
            else:
                if day_trades_5d > max_day_trades:
                    account_type = "restricted"  # PDT flagged but insufficient funds
                else:
                    account_type = "margin"  # Regular margin account

            capabilities = self.config["account_types"].get(account_type, {})

            # Calculate buying power
            if account_type == "pdt":
                day_bp_ratio = self.config["pdt_rules"][
                    "day_trading_buying_power_ratio"
                ]
                overnight_bp_ratio = self.config["pdt_rules"][
                    "overnight_buying_power_ratio"
                ]
            else:
                day_bp_ratio = self.config["pdt_rules"]["overnight_buying_power_ratio"]
                overnight_bp_ratio = self.config["pdt_rules"][
                    "overnight_buying_power_ratio"
                ]

            classification = {
                "account_type": account_type,
                "account_value": account_value,
                "day_trades_5d": day_trades_5d,
                "pdt_eligible": account_value >= min_pdt_value,
                "pdt_flagged": day_trades_5d > max_day_trades,
                "day_trading_allowed": capabilities.get("day_trading", False)
                and account_type != "restricted",
                "short_selling_allowed": capabilities.get("short_selling", False),
                "day_trading_buying_power": account_value * day_bp_ratio,
                "overnight_buying_power": account_value * overnight_bp_ratio,
                "remaining_day_trades": (
                    max(0, max_day_trades - day_trades_5d)
                    if account_type != "pdt"
                    else float("inf")
                ),
            }

            return classification

        except Exception as e:
            logger.error(f"Error classifying account type: {e}")
            return {"error": str(e), "account_type": "unknown"}

    def check_short_locate_availability(
        self, symbol: str, quantity: int
    ) -> Dict[str, any]:
        """
        Check short locate availability for a symbol.

        Args:
            symbol: Stock symbol to check
            quantity: Number of shares to locate

        Returns:
            Locate availability and borrowing costs
        """
        try:
            # Mock implementation - real version would:
            # 1. Query prime broker locate systems
            # 2. Check clearing firm inventory
            # 3. Contact third-party locate services
            # 4. Calculate borrowing costs and fees

            locate_result = {
                "symbol": symbol,
                "requested_quantity": quantity,
                "timestamp": datetime.now().isoformat(),
                "locate_available": False,
                "available_quantity": 0,
                "borrow_rate_annual": 0.0,
                "locate_fee": 0.0,
                "source": None,
                "expiration": None,
                "hard_to_borrow": False,
            }

            # Mock locate logic based on symbol characteristics
            import random

            # Simulate locate availability (80% success rate for most stocks)
            if random.random() < 0.8:
                available_qty = max(quantity, random.randint(1000, 10000))
                borrow_rate = random.uniform(0.5, 15.0)  # 0.5% to 15% annual

                locate_result.update(
                    {
                        "locate_available": True,
                        "available_quantity": available_qty,
                        "borrow_rate_annual": borrow_rate,
                        "locate_fee": quantity * 0.01,  # $0.01 per share locate fee
                        "source": random.choice(
                            self.config["short_selling"]["locate_sources"]
                        ),
                        "expiration": (
                            datetime.now() + timedelta(hours=8)
                        ).isoformat(),  # Locate good for trading day
                        "hard_to_borrow": borrow_rate
                        > self.config["short_selling"]["hard_to_borrow_threshold"],
                    }
                )

            # Store locate result in Redis
            if self.redis_client:
                locate_key = f"short_locate:{symbol}:{quantity}"
                self.redis_client.set(locate_key, json.dumps(locate_result))
                self.redis_client.expire(locate_key, 28800)  # 8 hours expiration

            if locate_result["locate_available"]:
                logger.info(
                    f"✅ Short locate available for {symbol}: {available_qty} shares at {borrow_rate:.1f}%"
                )
            else:
                logger.warning(f"❌ Short locate NOT available for {symbol}")

            return locate_result

        except Exception as e:
            logger.error(f"Error checking short locate for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "locate_available": False}

    def validate_short_sale_order(
        self, symbol: str, quantity: int, account_info: Dict
    ) -> Dict[str, any]:
        """
        Validate short sale order against all regulations.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to short
            account_info: Account information

        Returns:
            Validation results and any restrictions
        """
        try:
            validation = {
                "symbol": symbol,
                "quantity": quantity,
                "timestamp": datetime.now().isoformat(),
                "order_allowed": True,
                "restrictions": [],
                "warnings": [],
            }

            # Check account capabilities
            account_type = account_info.get("account_type", "unknown")
            short_allowed = account_info.get("short_selling_allowed", False)

            if not short_allowed:
                validation["order_allowed"] = False
                validation["restrictions"].append(
                    f"Account type '{account_type}' not authorized for short selling"
                )

            # Check short locate requirement
            if self.config["short_selling"]["locate_required"]:
                locate_result = self.check_short_locate_availability(symbol, quantity)

                if not locate_result.get("locate_available", False):
                    validation["order_allowed"] = False
                    validation["restrictions"].append("Short locate not available")
                elif locate_result.get("hard_to_borrow", False):
                    validation["warnings"].append(
                        f"Hard-to-borrow: {locate_result.get('borrow_rate_annual', 0):.1f}% borrow rate"
                    )

            # Check restricted securities
            if symbol in self.config["restricted_securities"]["no_short_list"]:
                validation["order_allowed"] = False
                validation["restrictions"].append("Symbol on no-short list")

            # Check SSR compliance
            if self.config["short_selling"]["ssr_compliance"]:
                ssr_check = self._check_ssr_compliance(symbol)
                if ssr_check.get("ssr_active", False):
                    validation["restrictions"].append(
                        "SSR active - short sales restricted"
                    )
                    # Note: In real SSR, shorts are only allowed on upticks

            # Check position size limits
            max_short_pct = self.config["short_selling"]["max_short_position_pct"]
            account_value = account_info.get("account_value", 100000)

            # Mock current short exposure
            current_short_value = account_value * 0.15  # Assume 15% currently short
            order_value = quantity * 100  # Assume $100/share for calculation

            if (current_short_value + order_value) > (account_value * max_short_pct):
                validation["order_allowed"] = False
                validation["restrictions"].append(
                    f"Would exceed maximum short exposure ({max_short_pct:.0%})"
                )

            return validation

        except Exception as e:
            logger.error(f"Error validating short sale order: {e}")
            return {"error": str(e), "order_allowed": False}

    def _check_ssr_compliance(self, symbol: str) -> Dict[str, any]:
        """Check if symbol is currently under Short Sale Rule."""
        try:
            # Mock SSR check - real version would check:
            # 1. Current SSR list from exchanges
            # 2. Stocks that declined >10% from previous close
            # 3. SSR activation/expiration times

            ssr_result = {"ssr_active": False, "trigger_time": None, "expires": None}

            # Check Redis for SSR simulation
            if self.redis_client:
                ssr_simulation = self.redis_client.get("market:ssr_simulation")
                if ssr_simulation == "active":
                    ssr_result["ssr_active"] = True
                    ssr_result["trigger_time"] = (
                        datetime.now() - timedelta(hours=2)
                    ).isoformat()
                    ssr_result["expires"] = (
                        datetime.now() + timedelta(hours=22)
                    ).isoformat()

            return ssr_result

        except Exception as e:
            return {"error": str(e), "ssr_active": False}

    def calculate_day_trading_requirements(
        self, account_info: Dict, trades_today: List[Dict]
    ) -> Dict[str, any]:
        """
        Calculate day trading buying power and requirements.

        Args:
            account_info: Account information
            trades_today: List of trades executed today

        Returns:
            Day trading calculations and limits
        """
        try:
            account_value = account_info.get("account_value", 0)
            account_type = account_info.get("account_type", "margin")
            day_trades_5d = account_info.get("day_trades_5d", 0)

            # Count day trades from today's trades
            day_trades_today = self._count_day_trades(trades_today)

            # Calculate buying power
            if account_type == "pdt":
                day_bp_ratio = self.config["pdt_rules"][
                    "day_trading_buying_power_ratio"
                ]
                max_day_trades = float("inf")
            else:
                day_bp_ratio = 2.0  # Standard margin
                max_day_trades = self.config["pdt_rules"]["day_trade_limit_non_pdt"]

            day_trading_bp = account_value * day_bp_ratio
            remaining_day_trades = max(
                0, max_day_trades - day_trades_5d - day_trades_today
            )

            # Calculate used buying power from open positions
            used_bp = self._calculate_used_buying_power(trades_today)
            available_bp = max(0, day_trading_bp - used_bp)

            requirements = {
                "account_type": account_type,
                "account_value": account_value,
                "day_trading_buying_power": day_trading_bp,
                "used_buying_power": used_bp,
                "available_buying_power": available_bp,
                "day_trades_5d": day_trades_5d,
                "day_trades_today": day_trades_today,
                "remaining_day_trades": remaining_day_trades,
                "day_trading_allowed": remaining_day_trades > 0
                or account_type == "pdt",
                "pdt_call_risk": account_value
                < self.config["pdt_rules"]["minimum_account_value"]
                and day_trades_5d > 3,
            }

            return requirements

        except Exception as e:
            logger.error(f"Error calculating day trading requirements: {e}")
            return {"error": str(e)}

    def _count_day_trades(self, trades: List[Dict]) -> int:
        """Count day trades from trade list."""
        # Mock implementation - real version would:
        # 1. Group trades by symbol and direction
        # 2. Identify opening and closing transactions on same day
        # 3. Count round-trip day trades

        day_trade_count = 0
        symbol_positions = {}

        for trade in trades:
            symbol = trade.get("symbol", "")
            side = trade.get("side", "")
            quantity = abs(trade.get("quantity", 0))

            if symbol not in symbol_positions:
                symbol_positions[symbol] = 0

            if side.lower() == "buy":
                symbol_positions[symbol] += quantity
            elif side.lower() == "sell":
                if symbol_positions[symbol] > 0:
                    # This is a closing trade - count as day trade
                    day_trade_count += 1
                symbol_positions[symbol] -= quantity

        return day_trade_count

    def _calculate_used_buying_power(self, trades: List[Dict]) -> float:
        """Calculate buying power used by current positions."""
        # Mock calculation
        total_position_value = sum(
            abs(trade.get("quantity", 0)) * trade.get("price", 100) for trade in trades
        )

        return total_position_value * 0.5  # Assume 2:1 margin usage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PDT and Short Manager")

    parser.add_argument(
        "--action",
        choices=["classify", "short-locate", "validate-short", "day-trading"],
        default="classify",
        help="Action to perform",
    )
    parser.add_argument(
        "--account-value",
        type=float,
        default=50000,
        help="Account value for classification",
    )
    parser.add_argument(
        "--day-trades", type=int, default=2, help="Number of day trades in last 5 days"
    )
    parser.add_argument(
        "--symbol", type=str, default="AAPL", help="Symbol for short selling checks"
    )
    parser.add_argument(
        "--quantity", type=int, default=100, help="Quantity for short selling"
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("🏦 Starting PDT and Short Manager")

    try:
        manager = PDTShortManager()

        if args.action == "classify":
            results = manager.classify_account_type(args.account_value, args.day_trades)
            print(f"\n🏦 ACCOUNT CLASSIFICATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "short-locate":
            results = manager.check_short_locate_availability(
                args.symbol, args.quantity
            )
            print(f"\n📍 SHORT LOCATE CHECK:")
            print(json.dumps(results, indent=2))

        elif args.action == "validate-short":
            # Create mock account info
            account_info = manager.classify_account_type(
                args.account_value, args.day_trades
            )
            results = manager.validate_short_sale_order(
                args.symbol, args.quantity, account_info
            )
            print(f"\n✅ SHORT SALE VALIDATION:")
            print(json.dumps(results, indent=2))

        elif args.action == "day-trading":
            account_info = manager.classify_account_type(
                args.account_value, args.day_trades
            )
            mock_trades = [
                {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 150},
                {"symbol": "AAPL", "side": "sell", "quantity": 100, "price": 152},
            ]
            results = manager.calculate_day_trading_requirements(
                account_info, mock_trades
            )
            print(f"\n📊 DAY TRADING REQUIREMENTS:")
            print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0 if not results.get("error") else 1

    except Exception as e:
        logger.error(f"Error in PDT and short manager: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
