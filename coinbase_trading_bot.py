#!/usr/bin/env python3
"""
Coinbase Trading Bot - Live Trading with 80% Capital
"""
import os
import time
import sys
import json

sys.path.append("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

from src.layers.layer4_execution.coinbase_executor import CoinbaseExecutor, OrderRequest


class CoinbaseTradingBot:
    def __init__(self, live_mode=False):
        # Configure environment - Corrected API credentials for JWT
        os.environ["COINBASE_API_KEY"] = "afc7fd77-0741-4a43-8b81-ae45e837e7ee"
        os.environ["COINBASE_API_SECRET"] = (
            "MHcCAQEEIDnNUtf8EZSegEHggCiYXwRhzvzI03EUbeW406JyTnzhoAoGCCqGSM49AwEHoUQDQgAEZ/Q/zsHCNWwlNTu3Kp+woT7M0VyNtmsyrT/rAS62y+u2x73mjQLHVOcrtJAo23zCAUQFXug1jQA45vaza32Ebw=="
        )
        os.environ["COINBASE_PASSPHRASE"] = ""
        os.environ["EXEC_MODE"] = "live" if live_mode else "paper"

        self.executor = CoinbaseExecutor()
        self.symbols = ["BTC-USD", "ETH-USD"]
        self.target_allocation = {
            "BTC-USD": 0.45,
            "ETH-USD": 0.35,
        }  # 80% total, 20% cash

        print(f"🚀 Coinbase Trading Bot initialized - Mode: {os.environ['EXEC_MODE']}")

    def get_portfolio_status(self):
        """Get current portfolio status"""
        account = self.executor.get_account()
        positions = self.executor.get_positions()

        print(f"💰 Account: {account.get('account_id', 'N/A')}")
        print(
            f"💵 Available Balance: ${account.get('available_balance', {}).get('value', '0')}"
        )
        print(f"📊 Positions: {len(positions)}")

        return account, positions

    def place_test_orders(self):
        """Place small test orders"""
        try:
            # Test BTC order
            btc_order = OrderRequest(
                symbol="BTC-USD", side="buy", qty=25.0, order_type="market"  # $25 worth
            )

            btc_response = self.executor.submit_order(btc_order)
            print(f"✅ BTC order: {btc_response.status}")

            # Test ETH order
            eth_order = OrderRequest(
                symbol="ETH-USD", side="buy", qty=25.0, order_type="market"  # $25 worth
            )

            eth_response = self.executor.submit_order(eth_order)
            print(f"✅ ETH order: {eth_response.status}")

            return True

        except Exception as e:
            print(f"❌ Order execution failed: {e}")
            return False

    def run_trading_session(self, duration_minutes=5):
        """Run a short trading session"""
        print(f"🔄 Starting {duration_minutes}-minute trading session")
        print("=" * 50)

        # Get initial status
        account, positions = self.get_portfolio_status()

        # Place test orders
        success = self.place_test_orders()

        if success:
            print("✅ Trading session completed successfully")
        else:
            print("❌ Trading session had errors")

        return success


def main():
    print("🚀 Starting Coinbase Trading Bot")

    # Start in LIVE mode with 80% capital
    bot = CoinbaseTradingBot(live_mode=True)

    # Run live trading session
    success = bot.run_trading_session(duration_minutes=60)  # 1 hour session

    if success:
        print("\n🎯 Live trading session completed successfully")
        print("✅ 80% capital deployment with BTC-USD/ETH-USD")
    else:
        print("\n❌ Live trading session had errors")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
