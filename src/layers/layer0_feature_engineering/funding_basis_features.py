#!/usr/bin/env python3
"""
Funding & Basis Feature Builder
Integrates funding rate and spot-perp basis into state vector
"""

import redis
import logging
from typing import Dict, Any

logger = logging.getLogger("funding_basis_features")


class FundingBasisFeatures:
    """Feature builder for funding rate and basis signals."""

    def __init__(self, redis_client: redis.Redis = None):
        """Initialize funding basis features."""
        self.redis = redis_client or redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )
        self.funding_key = "funding:basis"

        # Default symbols (can be configured)
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def get_funding_basis_features(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """Get funding and basis features for a specific symbol."""
        try:
            # Get funding and basis data from Redis
            funding_ann = self.redis.hget(self.funding_key, f"{symbol}:funding")
            basis_bps = self.redis.hget(self.funding_key, f"{symbol}:basis")

            # Convert to float with defaults
            funding_ann = float(funding_ann) if funding_ann else 0.0
            basis_bps = float(basis_bps) if basis_bps else 0.0

            return {"funding_ann": funding_ann, "basis_bps": basis_bps}

        except Exception as e:
            logger.warning(f"Error getting funding/basis features for {symbol}: {e}")
            return {"funding_ann": 0.0, "basis_bps": 0.0}

    def get_cross_asset_features(self) -> Dict[str, float]:
        """Get cross-asset funding and basis features."""
        try:
            features = {}

            # Get data for all symbols
            btc_features = self.get_funding_basis_features("BTCUSDT")
            eth_features = self.get_funding_basis_features("ETHUSDT")
            sol_features = self.get_funding_basis_features("SOLUSDT")

            # Add individual asset features
            features.update(
                {
                    "btc_funding_ann": btc_features["funding_ann"],
                    "btc_basis_bps": btc_features["basis_bps"],
                    "eth_funding_ann": eth_features["funding_ann"],
                    "eth_basis_bps": eth_features["basis_bps"],
                    "sol_funding_ann": sol_features["funding_ann"],
                    "sol_basis_bps": sol_features["basis_bps"],
                }
            )

            # Calculate cross-asset metrics
            funding_values = [
                btc_features["funding_ann"],
                eth_features["funding_ann"],
                sol_features["funding_ann"],
            ]
            basis_values = [
                btc_features["basis_bps"],
                eth_features["basis_bps"],
                sol_features["basis_bps"],
            ]

            # Average funding across assets
            avg_funding = (
                sum(funding_values) / len(funding_values) if funding_values else 0.0
            )

            # Average basis across assets
            avg_basis = sum(basis_values) / len(basis_values) if basis_values else 0.0

            # Funding spread (max - min)
            funding_spread = (
                (max(funding_values) - min(funding_values)) if funding_values else 0.0
            )

            # Basis spread (max - min)
            basis_spread = (
                (max(basis_values) - min(basis_values)) if basis_values else 0.0
            )

            features.update(
                {
                    "avg_funding_ann": avg_funding,
                    "avg_basis_bps": avg_basis,
                    "funding_spread": funding_spread,
                    "basis_spread": basis_spread,
                }
            )

            return features

        except Exception as e:
            logger.error(f"Error getting cross-asset funding/basis features: {e}")
            return {}


def add_funding_basis_to_state(
    state: Dict[str, Any], symbol: str = "BTCUSDT", redis_client: redis.Redis = None
) -> Dict[str, Any]:
    """
    Add funding and basis features to existing state vector.

    This function can be called from your main state builder to integrate
    funding/basis features as specified in the requirements:

    state["funding_ann"] = float(R.hget("funding:basis", f"{sym}:funding") or 0)
    state["basis_bps"]   = float(R.hget("funding:basis", f"{sym}:basis") or 0)
    """
    try:
        feature_builder = FundingBasisFeatures(redis_client)

        # Get primary symbol features (as specified in requirements)
        primary_features = feature_builder.get_funding_basis_features(symbol)
        state["funding_ann"] = primary_features["funding_ann"]
        state["basis_bps"] = primary_features["basis_bps"]

        # Optionally add cross-asset features for enhanced alpha
        cross_features = feature_builder.get_cross_asset_features()
        state.update(cross_features)

        logger.debug(
            f"Added funding/basis features: funding={state['funding_ann']:.3f}, basis={state['basis_bps']:.1f}bps"
        )

        return state

    except Exception as e:
        logger.error(f"Error adding funding/basis features to state: {e}")
        # Add default values to prevent downstream errors
        state["funding_ann"] = 0.0
        state["basis_bps"] = 0.0
        return state


# Example integration function
def build_enhanced_state_with_funding_basis(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Example of how to integrate funding/basis features into your state vector.
    This would be called from your main state builder.
    """
    # Your existing state building logic would go here
    state = {
        # Example existing features
        "price": 50000.0,
        "volume": 1000.0,
        "rsi": 55.0,
        "volatility": 0.02,
        "momentum_5m": 0.001,
        "order_book_imbalance": 0.1,
        # ... other features
    }

    # Add funding/basis features as specified
    R = redis.Redis(host="localhost", port=6379, decode_responses=True)
    state = add_funding_basis_to_state(state, symbol, R)

    return state


if __name__ == "__main__":
    # Test the feature builder
    import json

    # Test individual symbol features
    feature_builder = FundingBasisFeatures()
    btc_features = feature_builder.get_funding_basis_features("BTCUSDT")
    print("BTC Features:", json.dumps(btc_features, indent=2))

    # Test cross-asset features
    cross_features = feature_builder.get_cross_asset_features()
    print("Cross-Asset Features:", json.dumps(cross_features, indent=2))

    # Test state integration
    enhanced_state = build_enhanced_state_with_funding_basis("BTCUSDT")
    print("Enhanced State (funding/basis only):")
    funding_basis_features = {
        k: v for k, v in enhanced_state.items() if "funding" in k or "basis" in k
    }
    print(json.dumps(funding_basis_features, indent=2))
