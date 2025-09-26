#!/usr/bin/env python3
"""
Test Coinbase Live Trading
"""
import os
import sys

sys.path.append("/Users/yanzewu/PycharmProjects/NLP_Final_Project_D")

from src.layers.layer4_execution.coinbase_executor import CoinbaseExecutor, OrderRequest


def main():
    print("🚀 Testing Coinbase Live Trading")

    # Set environment
    os.environ["COINBASE_API_KEY"] = (
        "organizations/11c946be-1e43-4297-bd21-71ee665901f7/apiKeys/5bd7de07-22d7-4ab2-9f10-a82e49c826aa"
    )
    os.environ[
        "COINBASE_API_SECRET"
    ] = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIARlBET9lYEdVpc+ECA+sdGHlEVjmZ0r3uyv15Z92jY5oAoGCCqGSM49
AwEHoUQDQgAE2SkLE7GUExTZvoa7j6IQsHjq/dsdb3R4EsUnmhwG7GpR4tA4+GaY
ZdJoas7ytrLS4EOHnbcTe2tuwroz7aemtQ==
-----END EC PRIVATE KEY-----"""
    os.environ["COINBASE_PASSPHRASE"] = ""
    os.environ["EXEC_MODE"] = "live"

    # Initialize executor
    executor = CoinbaseExecutor()

    # Test account access
    print("🔍 Testing account access...")
    account = executor.get_account()
    print(f"Account: {account}")

    # Test positions
    print("📊 Testing positions...")
    positions = executor.get_positions()
    print(f"Positions: {positions}")

    # Test small order (paper mode for safety)
    os.environ["EXEC_MODE"] = "paper"
    executor = CoinbaseExecutor()

    print("📝 Testing order submission (paper mode)...")
    test_order = OrderRequest(
        symbol="BTC-USD", side="buy", qty=50.0, order_type="market"  # $50 worth
    )

    response = executor.submit_order(test_order)
    print(f"Order response: {response}")

    print("✅ Coinbase test completed!")


if __name__ == "__main__":
    main()
