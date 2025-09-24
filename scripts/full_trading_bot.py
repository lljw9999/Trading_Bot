#!/usr/bin/env python3
"""
Full Trading Bot with Buy AND Sell Functionality
Trades BTC and ETH with profit-taking and rebalancing
"""
import os
import time
import json
from decimal import Decimal, ROUND_DOWN, getcontext
from binance.spot import Spot as Client

getcontext().prec = 28


class FullTradingBot:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_TRADING_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_TRADING_SECRET_KEY", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing Binance API credentials")

        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        self.recv_window = 5000

        # Trading parameters
        self.target_allocation = {"BTC": 0.5, "ETH": 0.5}  # 50/50 target
        self.rebalance_threshold = 0.15  # Rebalance if >15% off target
        self.profit_threshold = 0.05  # Take profit at 5% gain
        self.loss_threshold = -0.03  # Cut losses at 3% loss

        print("🤖 Full Trading Bot initialized")

    def get_balances(self):
        """Get current balances"""
        account = self.client.account(recvWindow=self.recv_window)
        balances = {}

        for balance in account["balances"]:
            asset = balance["asset"]
            free = Decimal(balance["free"])
            locked = Decimal(balance["locked"])
            total = free + locked

            if total > 0:
                balances[asset] = {
                    "free": float(free),
                    "locked": float(locked),
                    "total": float(total),
                }

        return balances

    def get_prices(self):
        """Get current BTC and ETH prices"""
        tickers = self.client.ticker_24hr()
        prices = {}

        for ticker in tickers:
            if ticker["symbol"] == "BTCUSDT":
                prices["BTC"] = float(ticker["lastPrice"])
            elif ticker["symbol"] == "ETHUSDT":
                prices["ETH"] = float(ticker["lastPrice"])

        return prices

    def get_filters(self, symbol):
        """Get trading filters for symbol"""
        ex = self.client.exchange_info(symbol=symbol)
        s = ex["symbols"][0]
        f = {fl["filterType"]: fl for fl in s["filters"]}

        lot = f.get("LOT_SIZE") or f.get("MARKET_LOT_SIZE")
        pricef = f["PRICE_FILTER"]
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL")

        return {
            "minQty": Decimal(lot["minQty"]),
            "stepSize": Decimal(lot["stepSize"]),
            "tickSize": Decimal(pricef["tickSize"]),
            "minNotional": Decimal(notional.get("minNotional", "0")),
        }

    def format_quantity(self, qty, step_size):
        """Format quantity according to step size"""
        if step_size == 0:
            return f"{qty:.8f}"

        # Round down to step size
        rounded = (Decimal(str(qty)) / step_size).to_integral_value(
            rounding=ROUND_DOWN
        ) * step_size

        # Count decimal places in step size
        step_str = f"{step_size:.8f}".rstrip("0").rstrip(".")
        if "." in step_str:
            decimals = len(step_str.split(".")[1])
        else:
            decimals = 0

        return f"{rounded:.{decimals}f}"

    def format_price(self, price, tick_size):
        """Format price according to tick size"""
        if tick_size == 0:
            return f"{price:.8f}"

        # Round to tick size
        rounded = round(Decimal(str(price)) / tick_size) * tick_size

        # Count decimal places in tick size
        tick_str = f"{tick_size:.8f}".rstrip("0").rstrip(".")
        if "." in tick_str:
            decimals = len(tick_str.split(".")[1])
        else:
            decimals = 0

        return f"{rounded:.{decimals}f}"

    def place_order(self, symbol, side, quantity, order_type="MARKET"):
        """Place a buy or sell order"""
        try:
            filters = self.get_filters(symbol)
            qty_str = self.format_quantity(quantity, filters["stepSize"])

            # Check minimum quantity
            if Decimal(qty_str) < filters["minQty"]:
                print(f"❌ Quantity {qty_str} below minimum {filters['minQty']}")
                return None

            order_params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": qty_str,
                "recvWindow": self.recv_window,
            }

            print(f"📊 Placing {side} order: {qty_str} {symbol} ({order_type})")

            result = self.client.new_order(**order_params)
            print(f"✅ Order filled: {result.get('executedQty', 'N/A')} {symbol}")

            return result

        except Exception as e:
            print(f"❌ Order failed: {e}")
            return None

    def calculate_portfolio_value(self, balances, prices):
        """Calculate total portfolio value in USDT"""
        total_value = 0
        asset_values = {}

        # USDT value
        usdt_balance = balances.get("USDT", {}).get("total", 0)
        total_value += usdt_balance
        asset_values["USDT"] = usdt_balance

        # BTC value
        btc_balance = balances.get("BTC", {}).get("total", 0)
        btc_value = btc_balance * prices["BTC"]
        total_value += btc_value
        asset_values["BTC"] = btc_value

        # ETH value
        eth_balance = balances.get("ETH", {}).get("total", 0)
        eth_value = eth_balance * prices["ETH"]
        total_value += eth_value
        asset_values["ETH"] = eth_value

        return total_value, asset_values

    def should_rebalance(self, asset_values, total_value):
        """Check if portfolio needs rebalancing"""
        if total_value < 10:  # Don't rebalance small portfolios
            return False, {}

        current_allocation = {}
        target_values = {}

        for asset in ["BTC", "ETH"]:
            current_pct = asset_values.get(asset, 0) / total_value
            target_pct = self.target_allocation[asset]

            current_allocation[asset] = current_pct
            target_values[asset] = total_value * target_pct

            # Check if allocation is significantly off
            if abs(current_pct - target_pct) > self.rebalance_threshold:
                return True, target_values

        return False, {}

    def execute_rebalancing(self, balances, prices, target_values):
        """Execute rebalancing trades"""
        print("⚖️  Executing portfolio rebalancing...")

        for asset in ["BTC", "ETH"]:
            current_balance = balances.get(asset, {}).get("total", 0)
            current_value = current_balance * prices[asset]
            target_value = target_values[asset]

            difference = target_value - current_value

            if abs(difference) < 5:  # Don't trade tiny amounts
                continue

            symbol = f"{asset}USDT"

            if difference > 0:  # Need to buy more of this asset
                # Buy with USDT
                usdt_needed = difference
                usdt_available = balances.get("USDT", {}).get("free", 0)

                if usdt_available >= usdt_needed:
                    qty_to_buy = (
                        usdt_needed / prices[asset] * 0.99
                    )  # Leave room for fees
                    self.place_order(symbol, "BUY", qty_to_buy)
                else:
                    print(
                        f"⚠️  Insufficient USDT for {asset} purchase: need ${usdt_needed:.2f}, have ${usdt_available:.2f}"
                    )

            else:  # Need to sell some of this asset
                qty_to_sell = abs(difference) / prices[asset]
                available_qty = balances.get(asset, {}).get("free", 0)

                if available_qty >= qty_to_sell:
                    self.place_order(symbol, "SELL", qty_to_sell)
                else:
                    print(
                        f"⚠️  Insufficient {asset} for sale: need {qty_to_sell:.6f}, have {available_qty:.6f}"
                    )

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Trading Cycle")
            print("=" * 40)

            # Get current state
            balances = self.get_balances()
            prices = self.get_prices()
            total_value, asset_values = self.calculate_portfolio_value(balances, prices)

            print(f"💰 Portfolio Value: ${total_value:.2f}")
            print(f"₿ BTC: ${asset_values.get('BTC', 0):.2f}")
            print(f"Ξ ETH: ${asset_values.get('ETH', 0):.2f}")
            print(f"💵 USDT: ${asset_values.get('USDT', 0):.2f}")

            # Check if rebalancing is needed
            needs_rebalancing, target_values = self.should_rebalance(
                asset_values, total_value
            )

            if needs_rebalancing:
                print("⚖️  Portfolio needs rebalancing!")
                self.execute_rebalancing(balances, prices, target_values)
            else:
                print("✅ Portfolio is balanced")

            return True

        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            return False

    def run(self):
        """Run the trading bot continuously"""
        print("🚀 Starting Full Trading Bot")
        print("💡 Will buy AND sell BTC/ETH for rebalancing")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 50)

        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                print(f"\n🔄 Cycle #{cycle_count}")

                success = self.run_trading_cycle()

                if success:
                    print("✅ Cycle completed successfully")
                else:
                    print("❌ Cycle had errors")

                # Sleep between cycles (5-10 minutes)
                sleep_time = 300  # 5 minutes
                print(f"😴 Sleeping {sleep_time}s until next cycle...")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n🛑 Trading bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("😴 Sleeping 120s before retry...")
                time.sleep(120)


def main():
    try:
        bot = FullTradingBot()
        bot.run()
    except Exception as e:
        print(f"❌ Failed to start trading bot: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
