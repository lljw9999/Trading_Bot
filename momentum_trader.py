#!/usr/bin/env python3
"""
Momentum Profit Trading Bot
Buys on upward momentum, sells on downward momentum to make profit
"""
import os
import time
import json
from decimal import Decimal, ROUND_DOWN, getcontext
from binance.spot import Spot as Client

getcontext().prec = 28


class MomentumTrader:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_TRADING_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_TRADING_SECRET_KEY", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing Binance API credentials")

        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        self.recv_window = 5000

        # Trading parameters
        self.momentum_threshold = 0.005  # 0.5% price movement triggers trade
        self.profit_target = 0.03  # Take profit at 3%
        self.stop_loss = -0.015  # Stop loss at 1.5%
        self.position_size = 0.3  # Use 30% of available balance per trade

        # Price history for momentum calculation
        self.price_history = {"BTC": [], "ETH": []}
        self.positions = {}  # Track open positions

        print("🚀 Momentum Trader initialized")
        print(f"📈 Momentum threshold: {self.momentum_threshold*100:.1f}%")
        print(f"🎯 Profit target: {self.profit_target*100:.1f}%")
        print(f"🛑 Stop loss: {abs(self.stop_loss)*100:.1f}%")

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
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL")

        return {
            "minQty": Decimal(lot["minQty"]),
            "stepSize": Decimal(lot["stepSize"]),
            "minNotional": Decimal(notional.get("minNotional", "0")),
        }

    def format_quantity(self, qty, step_size):
        """Format quantity according to step size"""
        if step_size == 0:
            return f"{qty:.8f}"

        rounded = (Decimal(str(qty)) / step_size).to_integral_value(
            rounding=ROUND_DOWN
        ) * step_size

        step_str = f"{step_size:.8f}".rstrip("0").rstrip(".")
        if "." in step_str:
            decimals = len(step_str.split(".")[1])
        else:
            decimals = 0

        return f"{rounded:.{decimals}f}"

    def place_order(self, symbol, side, quantity, order_type="MARKET"):
        """Place a buy or sell order"""
        try:
            filters = self.get_filters(symbol)
            qty_str = self.format_quantity(quantity, filters["stepSize"])

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

            print(f"💹 {side} {qty_str} {symbol} at market price")

            result = self.client.new_order(**order_params)
            filled_qty = float(result.get("executedQty", "0"))
            avg_price = (
                float(result.get("fills", [{}])[0].get("price", "0"))
                if result.get("fills")
                else 0
            )

            print(f"✅ Filled: {filled_qty} {symbol} @ ${avg_price:.2f}")

            return {
                "symbol": symbol,
                "side": side,
                "quantity": filled_qty,
                "price": avg_price,
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"❌ Order failed: {e}")
            return None

    def calculate_momentum(self, asset, current_price):
        """Calculate momentum based on recent price history"""
        history = self.price_history[asset]

        # Keep last 10 price points
        history.append(current_price)
        if len(history) > 10:
            history.pop(0)

        if len(history) < 3:
            return 0  # Not enough data

        # Calculate momentum as % change from 3 periods ago
        momentum = (current_price - history[-3]) / history[-3]
        return momentum

    def check_profit_loss(self, asset, current_price):
        """Check if we should close positions for profit/loss"""
        if asset not in self.positions:
            return None

        position = self.positions[asset]
        entry_price = position["price"]
        side = position["side"]

        if side == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price

        # Check profit target
        if pnl_pct >= self.profit_target:
            print(f"🎯 Taking profit on {asset}: {pnl_pct*100:.2f}%")
            return "CLOSE_PROFIT"

        # Check stop loss
        if pnl_pct <= self.stop_loss:
            print(f"🛑 Stop loss triggered on {asset}: {pnl_pct*100:.2f}%")
            return "CLOSE_LOSS"

        return None

    def execute_momentum_strategy(self, asset, current_price, balances):
        """Execute momentum-based trading strategy"""
        momentum = self.calculate_momentum(asset, current_price)
        symbol = f"{asset}USDT"

        # Check if we should close existing position
        close_action = self.check_profit_loss(asset, current_price)
        if close_action and asset in self.positions:
            position = self.positions[asset]
            # Close position (opposite side)
            close_side = "SELL" if position["side"] == "BUY" else "BUY"
            close_qty = position["quantity"]

            result = self.place_order(symbol, close_side, close_qty)
            if result:
                del self.positions[asset]
                print(f"✅ Closed {asset} position ({close_action})")
            return

        # Don't open new position if we already have one
        if asset in self.positions:
            return

        # Check for momentum signals
        if momentum > self.momentum_threshold:
            # Strong upward momentum - BUY
            usdt_balance = balances.get("USDT", {}).get("free", 0)
            trade_amount = usdt_balance * self.position_size

            if trade_amount > 10:  # Minimum $10 trade
                qty_to_buy = trade_amount / current_price * 0.99  # Leave room for fees
                result = self.place_order(symbol, "BUY", qty_to_buy)

                if result:
                    self.positions[asset] = result
                    print(
                        f"📈 Opened BUY position on {asset} momentum: {momentum*100:.2f}%"
                    )

        elif momentum < -self.momentum_threshold:
            # Strong downward momentum - SELL (if we have the asset)
            asset_balance = balances.get(asset, {}).get("free", 0)

            if asset_balance > 0:
                trade_qty = asset_balance * self.position_size
                result = self.place_order(symbol, "SELL", trade_qty)

                if result:
                    self.positions[asset] = result
                    print(
                        f"📉 Opened SELL position on {asset} momentum: {momentum*100:.2f}%"
                    )

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Momentum Trading Cycle")
            print("=" * 50)

            # Get current state
            balances = self.get_balances()
            prices = self.get_prices()

            # Calculate total portfolio value
            total_value = balances.get("USDT", {}).get("total", 0)
            for asset in ["BTC", "ETH"]:
                balance = balances.get(asset, {}).get("total", 0)
                total_value += balance * prices[asset]

            print(f"💰 Portfolio Value: ${total_value:.2f}")
            print(f"💵 USDT: ${balances.get('USDT', {}).get('total', 0):.2f}")
            print(
                f"₿ BTC: {balances.get('BTC', {}).get('total', 0):.6f} (${balances.get('BTC', {}).get('total', 0) * prices['BTC']:.2f})"
            )
            print(
                f"Ξ ETH: {balances.get('ETH', {}).get('total', 0):.6f} (${balances.get('ETH', {}).get('total', 0) * prices['ETH']:.2f})"
            )

            # Execute momentum strategy for each asset
            for asset in ["BTC", "ETH"]:
                current_price = prices[asset]
                momentum = self.calculate_momentum(asset, current_price)

                print(f"{asset}: ${current_price:.2f} | Momentum: {momentum*100:.2f}%")

                self.execute_momentum_strategy(asset, current_price, balances)

            # Show open positions
            if self.positions:
                print("\n🔄 Open Positions:")
                for asset, pos in self.positions.items():
                    current_price = prices[asset]
                    entry_price = pos["price"]
                    if pos["side"] == "BUY":
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price

                    print(
                        f"  {asset}: {pos['side']} {pos['quantity']:.6f} @ ${entry_price:.2f} | PnL: {pnl_pct*100:.2f}%"
                    )

            return True

        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            return False

    def run(self):
        """Run the momentum trading bot continuously"""
        print("🚀 Starting Momentum Profit Trader")
        print("📈 Will buy on upward momentum, sell on downward momentum")
        print("💰 Target: Make profit from price movements")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 60)

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

                # Sleep between cycles (2-3 minutes for momentum trading)
                sleep_time = 120  # 2 minutes
                print(f"😴 Sleeping {sleep_time}s until next cycle...")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n🛑 Momentum trader stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("😴 Sleeping 60s before retry...")
                time.sleep(60)


def main():
    try:
        trader = MomentumTrader()
        trader.run()
    except Exception as e:
        print(f"❌ Failed to start momentum trader: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
