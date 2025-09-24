#!/usr/bin/env python3
"""
Live P&L Tracker for Binance Spot Trading
Provides real-time portfolio monitoring and P&L calculations
"""
import os
import json
import time
import requests
from decimal import Decimal
from binance.spot import Spot as Client


class LivePnLTracker:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_TRADING_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_TRADING_SECRET_KEY", "")

        # Use mock mode if credentials are missing
        self.mock_mode = not (self.api_key and self.api_secret)

        if not self.mock_mode:
            self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        else:
            self.client = None

        # Track your actual trades (from successful execution)
        self.trades = {
            "BTC": {
                "amount": Decimal("0.00034965"),  # Your total BTC holdings
                "avg_cost": Decimal("117567.56"),  # Weighted average buy price
                "total_cost": Decimal("41.17"),  # Total USD spent on BTC
            },
            "ETH": {
                "amount": Decimal("0.00919080"),  # Your total ETH holdings
                "avg_cost": Decimal("4457.81"),  # Weighted average buy price
                "total_cost": Decimal("40.97"),  # Total USD spent on ETH
            },
        }

        self.initial_investment = Decimal("72.98")  # Your starting USDT

    def get_live_prices(self):
        """Get current BTC and ETH prices from Binance or fallback to CoinGecko"""
        if self.mock_mode:
            return self._get_fallback_prices()

        try:
            btc_ticker = self.client.ticker_24hr("BTCUSDT")
            eth_ticker = self.client.ticker_24hr("ETHUSDT")

            return {
                "BTC": {
                    "price": Decimal(btc_ticker["lastPrice"]),
                    "change_24h": Decimal(btc_ticker["priceChangePercent"]),
                },
                "ETH": {
                    "price": Decimal(eth_ticker["lastPrice"]),
                    "change_24h": Decimal(eth_ticker["priceChangePercent"]),
                },
            }
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return self._get_fallback_prices()

    def get_current_balances(self):
        """Get current account balances"""
        if self.mock_mode:
            return self._get_mock_balances()

        try:
            account = self.client.account()
            balances = {}

            for balance in account["balances"]:
                asset = balance["asset"]
                free = Decimal(balance["free"])
                locked = Decimal(balance["locked"])
                total = free + locked

                if total > 0:
                    balances[asset] = {"free": free, "locked": locked, "total": total}

            return balances
        except Exception as e:
            print(f"Error fetching balances: {e}")
            return {}

    def calculate_pnl(self):
        """Calculate real-time P&L"""
        prices = self.get_live_prices()
        balances = self.get_current_balances()

        if not prices:
            return None

        btc_price = prices["BTC"]["price"]
        eth_price = prices["ETH"]["price"]

        # Current values
        btc_value = self.trades["BTC"]["amount"] * btc_price
        eth_value = self.trades["ETH"]["amount"] * eth_price
        usdt_value = balances.get("USDT", {}).get("total", Decimal("0"))

        total_value = btc_value + eth_value + usdt_value

        # P&L calculations
        btc_pnl = btc_value - self.trades["BTC"]["total_cost"]
        eth_pnl = eth_value - self.trades["ETH"]["total_cost"]
        total_pnl = total_value - self.initial_investment

        # Returns
        btc_return = (btc_pnl / self.trades["BTC"]["total_cost"]) * 100
        eth_return = (eth_pnl / self.trades["ETH"]["total_cost"]) * 100
        total_return = (total_pnl / self.initial_investment) * 100

        return {
            "timestamp": time.time(),
            "prices": {"BTC": float(btc_price), "ETH": float(eth_price)},
            "holdings": {
                "BTC": {
                    "amount": float(self.trades["BTC"]["amount"]),
                    "value": float(btc_value),
                    "cost": float(self.trades["BTC"]["total_cost"]),
                    "pnl": float(btc_pnl),
                    "return_pct": float(btc_return),
                },
                "ETH": {
                    "amount": float(self.trades["ETH"]["amount"]),
                    "value": float(eth_value),
                    "cost": float(self.trades["ETH"]["total_cost"]),
                    "pnl": float(eth_pnl),
                    "return_pct": float(eth_return),
                },
                "USDT": {"amount": float(usdt_value), "value": float(usdt_value)},
            },
            "portfolio": {
                "total_value": float(total_value),
                "initial_investment": float(self.initial_investment),
                "total_pnl": float(total_pnl),
                "total_return_pct": float(total_return),
                "deployed_capital": float(
                    self.trades["BTC"]["total_cost"] + self.trades["ETH"]["total_cost"]
                ),
            },
            "price_changes_24h": {
                "BTC": float(prices["BTC"]["change_24h"]),
                "ETH": float(prices["ETH"]["change_24h"]),
            },
        }

    def display_pnl(self, pnl_data):
        """Display P&L in terminal"""
        if not pnl_data:
            print("❌ Unable to fetch P&L data")
            return

        portfolio = pnl_data["portfolio"]
        btc = pnl_data["holdings"]["BTC"]
        eth = pnl_data["holdings"]["ETH"]
        usdt = pnl_data["holdings"]["USDT"]

        # Clear screen and display header
        os.system("clear" if os.name == "posix" else "cls")
        print("🚀 LIVE TRADING P&L MONITOR")
        print("=" * 50)
        print(f"📊 Portfolio Value: ${portfolio['total_value']:.2f}")

        pnl_color = "🟢" if portfolio["total_pnl"] >= 0 else "🔴"
        print(
            f"{pnl_color} Total P&L: ${portfolio['total_pnl']:+.2f} ({portfolio['total_return_pct']:+.2f}%)"
        )
        print()

        # BTC Position
        btc_color = "🟢" if btc["pnl"] >= 0 else "🔴"
        print(f"₿ BTC Position:")
        print(f"  Amount: {btc['amount']:.8f} BTC")
        print(f"  Value: ${btc['value']:.2f}")
        print(f"  {btc_color} P&L: ${btc['pnl']:+.2f} ({btc['return_pct']:+.2f}%)")
        print(f"  Price: ${pnl_data['prices']['BTC']:,.2f}")
        print()

        # ETH Position
        eth_color = "🟢" if eth["pnl"] >= 0 else "🔴"
        print(f"Ξ ETH Position:")
        print(f"  Amount: {eth['amount']:.8f} ETH")
        print(f"  Value: ${eth['value']:.2f}")
        print(f"  {eth_color} P&L: ${eth['pnl']:+.2f} ({eth['return_pct']:+.2f}%)")
        print(f"  Price: ${pnl_data['prices']['ETH']:,.2f}")
        print()

        # Cash
        print(f"💵 USDT: ${usdt['value']:.2f}")
        print()

        # Summary
        print("📈 SUMMARY:")
        print(f"  Initial Investment: ${portfolio['initial_investment']:.2f}")
        print(f"  Deployed Capital: ${portfolio['deployed_capital']:.2f}")
        print(f"  Current Value: ${portfolio['total_value']:.2f}")
        print(f"  Cash Remaining: ${usdt['value']:.2f}")
        print()

        print(f"⏰ Last Update: {time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")

    def save_pnl_snapshot(self, pnl_data):
        """Save P&L snapshot to file"""
        if not pnl_data:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"artifacts/pnl_snapshot_{timestamp}.json"

        os.makedirs("artifacts", exist_ok=True)

        with open(filename, "w") as f:
            json.dump(pnl_data, f, indent=2)

        return filename

    def run_monitor(self, interval=10, save_snapshots=False):
        """Run continuous P&L monitoring"""
        print("🚀 Starting Live P&L Monitor...")
        print(f"📊 Monitoring every {interval} seconds")
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                pnl_data = self.calculate_pnl()
                self.display_pnl(pnl_data)

                if save_snapshots and pnl_data:
                    self.save_pnl_snapshot(pnl_data)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n👋 P&L monitoring stopped")

            if pnl_data:
                final_snapshot = self.save_pnl_snapshot(pnl_data)
                print(f"💾 Final snapshot saved: {final_snapshot}")

    def _get_fallback_prices(self):
        """Get prices from CoinGecko API as fallback"""
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
            )
            data = response.json()

            return {
                "BTC": {
                    "price": Decimal(str(data["bitcoin"]["usd"])),
                    "change_24h": Decimal(str(data["bitcoin"]["usd_24h_change"])),
                },
                "ETH": {
                    "price": Decimal(str(data["ethereum"]["usd"])),
                    "change_24h": Decimal(str(data["ethereum"]["usd_24h_change"])),
                },
            }
        except Exception as e:
            print(f"Error fetching fallback prices: {e}")
            # Return mock data if everything fails
            return {
                "BTC": {"price": Decimal("100000"), "change_24h": Decimal("0")},
                "ETH": {"price": Decimal("4000"), "change_24h": Decimal("0")},
            }

    def _get_mock_balances(self):
        """Return mock balances when API is unavailable"""
        return {
            "BTC": {
                "free": Decimal("0.00034965"),
                "locked": Decimal("0"),
                "total": Decimal("0.00034965"),
            },
            "ETH": {
                "free": Decimal("0.00919080"),
                "locked": Decimal("0"),
                "total": Decimal("0.00919080"),
            },
            "USDT": {
                "free": Decimal("10.00"),
                "locked": Decimal("0"),
                "total": Decimal("10.00"),
            },
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Live P&L Tracker for Binance Trading")
    parser.add_argument(
        "--interval", type=int, default=10, help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--save-snapshots", action="store_true", help="Save P&L snapshots to files"
    )
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    try:
        tracker = LivePnLTracker()

        if args.once:
            pnl_data = tracker.calculate_pnl()
            tracker.display_pnl(pnl_data)
            if args.save_snapshots:
                snapshot_file = tracker.save_pnl_snapshot(pnl_data)
                print(f"\n💾 Snapshot saved: {snapshot_file}")
        else:
            tracker.run_monitor(
                interval=args.interval, save_snapshots=args.save_snapshots
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
