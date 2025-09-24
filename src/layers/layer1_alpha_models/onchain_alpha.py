"""
On-Chain Data Alpha Model

Leverages blockchain data to generate trading signals based on:
- Whale wallet movements
- Exchange inflows/outflows
- Network activity metrics
- SOPR (Spent Output Profit Ratio)
- Mining hash rate changes
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
from decimal import Decimal
import logging

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal


class OnChainDataProvider:
    """
    Fetches on-chain data from various blockchain APIs.

    Supports multiple data sources:
    - Whale Alert API for large transactions
    - Glassnode API for network metrics
    - CoinMetrics API for institutional data
    - Free APIs for basic metrics
    """

    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize on-chain data provider.

        Args:
            api_keys: Dictionary of API keys for different services
        """
        self.api_keys = api_keys or {}
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.logger = logging.getLogger("onchain_provider")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_whale_transactions(
        self, symbol: str, min_value: int = 1000000
    ) -> List[Dict]:
        """
        Get recent whale transactions for a symbol.

        Args:
            symbol: Cryptocurrency symbol (BTC, ETH)
            min_value: Minimum transaction value in USD

        Returns:
            List of whale transactions
        """
        cache_key = f"whale_{symbol}_{min_value}"

        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return data

        try:
            # Use free blockchain.info API for Bitcoin
            if symbol.upper() == "BTC":
                url = "https://blockchain.info/unconfirmed-transactions?format=json"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        transactions = []

                        for tx in data.get("txs", [])[
                            :10
                        ]:  # Get recent 10 transactions
                            # Calculate transaction value
                            value_btc = (
                                sum(out.get("value", 0) for out in tx.get("out", []))
                                / 1e8
                            )
                            value_usd = value_btc * 50000  # Approximate BTC price

                            if value_usd >= min_value:
                                transactions.append(
                                    {
                                        "hash": tx.get("hash"),
                                        "time": tx.get("time"),
                                        "value_usd": value_usd,
                                        "value_crypto": value_btc,
                                        "symbol": "BTC",
                                    }
                                )

                        # Cache result
                        self.cache[cache_key] = (time.time(), transactions)
                        return transactions

            # For other symbols, return empty list (would need paid APIs)
            return []

        except Exception as e:
            self.logger.error(f"Error fetching whale transactions for {symbol}: {e}")
            return []

    async def get_exchange_flows(self, symbol: str) -> Dict[str, float]:
        """
        Get exchange inflow/outflow data.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Dictionary with inflow/outflow metrics
        """
        cache_key = f"exchange_flows_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return data

        try:
            # Mock data for demonstration (would use real APIs like Glassnode)
            flows = {
                "net_inflow": np.random.randn() * 1000,  # Positive = inflow (bearish)
                "net_outflow": np.random.randn() * 1000,  # Positive = outflow (bullish)
                "exchange_reserves": 50000 + np.random.randn() * 5000,
                "flow_ratio": np.random.randn() * 0.1,  # Recent vs historical
            }

            # Cache result
            self.cache[cache_key] = (time.time(), flows)
            return flows

        except Exception as e:
            self.logger.error(f"Error fetching exchange flows for {symbol}: {e}")
            return {}

    async def get_network_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Get network activity metrics.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Dictionary with network metrics
        """
        cache_key = f"network_metrics_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return data

        try:
            # Mock network metrics (would use real APIs)
            metrics = {
                "active_addresses": 800000 + np.random.randn() * 50000,
                "transaction_count": 300000 + np.random.randn() * 20000,
                "hash_rate": 150e18 + np.random.randn() * 10e18,  # For Bitcoin
                "difficulty": 25e12 + np.random.randn() * 1e12,
                "mempool_size": 2000 + np.random.randn() * 500,
                "avg_fee": 0.0001 + np.random.randn() * 0.00005,
            }

            # Cache result
            self.cache[cache_key] = (time.time(), metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error fetching network metrics for {symbol}: {e}")
            return {}

    async def get_sopr(self, symbol: str) -> float:
        """
        Get Spent Output Profit Ratio (SOPR).

        SOPR = Realized Value / Value at Creation
        > 1: Coins moving at profit
        < 1: Coins moving at loss

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            SOPR value
        """
        cache_key = f"sopr_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return data

        try:
            # Mock SOPR calculation (would use real on-chain data)
            sopr = 1.0 + np.random.randn() * 0.1  # Around 1.0 with noise

            # Cache result
            self.cache[cache_key] = (time.time(), sopr)
            return sopr

        except Exception as e:
            self.logger.error(f"Error calculating SOPR for {symbol}: {e}")
            return 1.0


class OnChainAlpha:
    """
    On-Chain Alpha Model for generating signals from blockchain data.

    Signal Generation Logic:
    - Large whale outflows from exchanges = Bullish (hodling)
    - Large whale inflows to exchanges = Bearish (selling)
    - High SOPR = Potential selling pressure
    - Network congestion = Short-term bearish, long-term bullish
    - Hash rate increases = Long-term bullish
    """

    def __init__(self, symbol: str, lookback_hours: int = 24):
        """
        Initialize on-chain alpha model.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours of historical data to analyze
        """
        self.symbol = symbol
        self.lookback_hours = lookback_hours

        # Data storage
        self.whale_history = deque(maxlen=100)
        self.flow_history = deque(maxlen=100)
        self.network_history = deque(maxlen=100)
        self.sopr_history = deque(maxlen=100)

        # Data provider
        self.data_provider = OnChainDataProvider()

        # Signal parameters
        self.min_whale_threshold = 1000000  # $1M minimum whale transaction
        self.flow_sensitivity = 0.1  # Sensitivity to flow changes
        self.sopr_threshold = 1.05  # SOPR threshold for selling pressure

        # Performance tracking
        self.signals_generated = 0
        self.last_update = None

        self.logger = logging.getLogger(f"onchain_alpha.{symbol}")
        self.logger.info(f"Initialized OnChain Alpha for {symbol}")

    async def update_onchain_data(self) -> bool:
        """
        Update on-chain data from various sources.

        Returns:
            True if data was updated successfully
        """
        try:
            async with self.data_provider as provider:
                # Fetch whale transactions
                whale_txs = await provider.get_whale_transactions(
                    self.symbol, self.min_whale_threshold
                )

                # Fetch exchange flows
                flows = await provider.get_exchange_flows(self.symbol)

                # Fetch network metrics
                network = await provider.get_network_metrics(self.symbol)

                # Fetch SOPR
                sopr = await provider.get_sopr(self.symbol)

                # Store data with timestamp
                timestamp = datetime.now()

                if whale_txs:
                    self.whale_history.append((timestamp, whale_txs))

                if flows:
                    self.flow_history.append((timestamp, flows))

                if network:
                    self.network_history.append((timestamp, network))

                if sopr:
                    self.sopr_history.append((timestamp, sopr))

                self.last_update = timestamp
                return True

        except Exception as e:
            self.logger.error(f"Error updating on-chain data: {e}")
            return False

    def _analyze_whale_activity(self) -> Dict[str, float]:
        """
        Analyze whale transaction patterns.

        Returns:
            Analysis results with signals
        """
        if not self.whale_history:
            return {}

        recent_whales = []
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)

        for timestamp, whale_txs in self.whale_history:
            if timestamp >= cutoff_time:
                recent_whales.extend(whale_txs)

        if not recent_whales:
            return {}

        # Calculate metrics
        total_value = sum(tx["value_usd"] for tx in recent_whales)
        avg_value = total_value / len(recent_whales)
        transaction_count = len(recent_whales)

        # Generate signals
        # Large whale activity = higher volatility expected
        whale_intensity = min(transaction_count / 10.0, 1.0)  # Normalized
        whale_size_factor = min(avg_value / 5000000, 1.0)  # Normalized to $5M

        return {
            "whale_intensity": whale_intensity,
            "whale_size_factor": whale_size_factor,
            "total_whale_value": total_value,
            "whale_count": transaction_count,
        }

    def _analyze_exchange_flows(self) -> Dict[str, float]:
        """
        Analyze exchange flow patterns.

        Returns:
            Flow analysis results
        """
        if not self.flow_history:
            return {}

        # Get recent flows
        recent_flows = []
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)

        for timestamp, flows in self.flow_history:
            if timestamp >= cutoff_time:
                recent_flows.append(flows)

        if not recent_flows:
            return {}

        # Calculate net flows
        net_inflow = np.mean([f.get("net_inflow", 0) for f in recent_flows])
        net_outflow = np.mean([f.get("net_outflow", 0) for f in recent_flows])

        # Net flow signal: positive = bullish (outflow), negative = bearish (inflow)
        flow_signal = (net_outflow - net_inflow) / 10000  # Normalize

        # Exchange reserves trend
        reserves = [f.get("exchange_reserves", 0) for f in recent_flows]
        if len(reserves) > 1:
            reserve_trend = (reserves[-1] - reserves[0]) / reserves[0]
        else:
            reserve_trend = 0

        return {
            "flow_signal": flow_signal,
            "reserve_trend": reserve_trend,
            "net_inflow": net_inflow,
            "net_outflow": net_outflow,
        }

    def _analyze_network_activity(self) -> Dict[str, float]:
        """
        Analyze network activity patterns.

        Returns:
            Network analysis results
        """
        if not self.network_history:
            return {}

        # Get recent network data
        recent_network = []
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)

        for timestamp, network in self.network_history:
            if timestamp >= cutoff_time:
                recent_network.append(network)

        if not recent_network:
            return {}

        # Calculate trends
        hash_rates = [n.get("hash_rate", 0) for n in recent_network]
        active_addresses = [n.get("active_addresses", 0) for n in recent_network]
        transaction_counts = [n.get("transaction_count", 0) for n in recent_network]

        # Calculate percentage changes
        def pct_change(values):
            if len(values) < 2 or values[0] == 0:
                return 0
            return (values[-1] - values[0]) / values[0]

        hash_rate_trend = pct_change(hash_rates)
        activity_trend = pct_change(active_addresses)
        tx_trend = pct_change(transaction_counts)

        return {
            "hash_rate_trend": hash_rate_trend,
            "activity_trend": activity_trend,
            "transaction_trend": tx_trend,
            "network_strength": (hash_rate_trend + activity_trend) / 2,
        }

    def _analyze_sopr(self) -> Dict[str, float]:
        """
        Analyze SOPR (Spent Output Profit Ratio) trends.

        Returns:
            SOPR analysis results
        """
        if not self.sopr_history:
            return {}

        # Get recent SOPR data
        recent_sopr = []
        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)

        for timestamp, sopr in self.sopr_history:
            if timestamp >= cutoff_time:
                recent_sopr.append(sopr)

        if not recent_sopr:
            return {}

        current_sopr = recent_sopr[-1]
        avg_sopr = np.mean(recent_sopr)

        # SOPR signals
        # > 1.05: Strong selling pressure (bearish)
        # < 0.95: Capitulation, potential bottom (bullish)
        # 1.0: Neutral

        if current_sopr > self.sopr_threshold:
            sopr_signal = -(current_sopr - 1.0) * 2  # Bearish
        elif current_sopr < 0.95:
            sopr_signal = (1.0 - current_sopr) * 2  # Bullish
        else:
            sopr_signal = 0  # Neutral

        return {
            "sopr_signal": sopr_signal,
            "current_sopr": current_sopr,
            "avg_sopr": avg_sopr,
            "sopr_trend": current_sopr - avg_sopr,
        }

    async def generate_signal(
        self, feature_snapshot: FeatureSnapshot
    ) -> Optional[AlphaSignal]:
        """
        Generate on-chain alpha signal.

        Args:
            feature_snapshot: Current market feature snapshot

        Returns:
            AlphaSignal if data is available, None otherwise
        """
        # Update on-chain data (rate limited by caching)
        await self.update_onchain_data()

        # Analyze different components
        whale_analysis = self._analyze_whale_activity()
        flow_analysis = self._analyze_exchange_flows()
        network_analysis = self._analyze_network_activity()
        sopr_analysis = self._analyze_sopr()

        # Check if we have enough data
        if not any([whale_analysis, flow_analysis, network_analysis, sopr_analysis]):
            return None

        # Combine signals
        total_signal = 0
        signal_count = 0
        confidence_factors = []

        # Whale activity signal
        if whale_analysis:
            whale_signal = (
                whale_analysis.get("whale_intensity", 0) * 0.5
            )  # Neutral to slightly bullish
            total_signal += whale_signal
            signal_count += 1
            confidence_factors.append(whale_analysis.get("whale_size_factor", 0))

        # Flow signal
        if flow_analysis:
            flow_signal = flow_analysis.get("flow_signal", 0)
            total_signal += flow_signal
            signal_count += 1
            confidence_factors.append(abs(flow_signal))

        # Network activity signal
        if network_analysis:
            network_signal = network_analysis.get("network_strength", 0) * 0.3
            total_signal += network_signal
            signal_count += 1
            confidence_factors.append(abs(network_signal))

        # SOPR signal
        if sopr_analysis:
            sopr_signal = sopr_analysis.get("sopr_signal", 0)
            total_signal += sopr_signal
            signal_count += 1
            confidence_factors.append(abs(sopr_signal))

        if signal_count == 0:
            return None

        # Calculate final signal
        avg_signal = total_signal / signal_count
        edge_bps = avg_signal * 30  # Scale to basis points (±30 bps max)

        # Calculate confidence
        if confidence_factors:
            confidence = np.mean(confidence_factors)
            confidence = np.clip(
                confidence, 0.1, 0.8
            )  # On-chain signals are medium confidence
        else:
            confidence = 0.3

        # Create reasoning
        reasoning_parts = []
        if whale_analysis:
            reasoning_parts.append(
                f"whale_activity: {whale_analysis.get('whale_count', 0)} txs"
            )
        if flow_analysis:
            flow_dir = (
                "outflow" if flow_analysis.get("flow_signal", 0) > 0 else "inflow"
            )
            reasoning_parts.append(f"exchange_{flow_dir}")
        if network_analysis:
            net_trend = (
                "growing"
                if network_analysis.get("network_strength", 0) > 0
                else "declining"
            )
            reasoning_parts.append(f"network_{net_trend}")
        if sopr_analysis:
            sopr_val = sopr_analysis.get("current_sopr", 1.0)
            reasoning_parts.append(f"sopr: {sopr_val:.3f}")

        reasoning = f"OnChain: {', '.join(reasoning_parts)}"

        # Generate signal
        self.signals_generated += 1

        return AlphaSignal(
            model_name="onchain_alpha",
            symbol=self.symbol,
            timestamp=feature_snapshot.timestamp,
            edge_bps=edge_bps,
            confidence=confidence,
            signal_strength=abs(avg_signal),
            metadata={
                "whale_analysis": whale_analysis,
                "flow_analysis": flow_analysis,
                "network_analysis": network_analysis,
                "sopr_analysis": sopr_analysis,
                "reasoning": reasoning,
            },
        )

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "model_name": "onchain_alpha",
            "symbol": self.symbol,
            "signals_generated": self.signals_generated,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "data_points": {
                "whale_history": len(self.whale_history),
                "flow_history": len(self.flow_history),
                "network_history": len(self.network_history),
                "sopr_history": len(self.sopr_history),
            },
        }


# Factory function
def create_onchain_alpha(symbol: str, **kwargs) -> OnChainAlpha:
    """
    Factory function to create OnChain alpha model.

    Args:
        symbol: Trading symbol
        **kwargs: Additional model parameters

    Returns:
        Initialized OnChainAlpha instance
    """
    return OnChainAlpha(symbol=symbol, **kwargs)


# Example usage
if __name__ == "__main__":

    async def test_onchain_alpha():
        model = OnChainAlpha("BTCUSDT")

        # Create test feature snapshot
        feature_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            mid_price=Decimal("50000"),
            return_1m=0.001,
        )

        # Generate signal
        signal = await model.generate_signal(feature_snapshot)

        if signal:
            print(
                f"OnChain Signal: {signal.edge_bps:.2f} bps, confidence: {signal.confidence:.3f}"
            )
            print(f"Reasoning: {signal.metadata.get('reasoning', 'N/A')}")
        else:
            print("No signal generated")

        print("\nModel Stats:")
        print(model.get_model_stats())

    # Run test
    asyncio.run(test_onchain_alpha())
