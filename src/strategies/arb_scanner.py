#!/usr/bin/env python3
"""
Cross-Exchange Arbitrage Scanner

Real-time scanner for arbitrage opportunities across multiple crypto exchanges.
Computes risk-adjusted spreads accounting for fees, latency, and liquidity.
"""

import os
import sys
import time
import asyncio
import logging
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.utils.aredis import (
        get_redis,
        get_batch_writer,
        set_metric,
        get_config_value,
    )

    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False
    import redis

logger = logging.getLogger("arb_scanner")


class ExchangeType(Enum):
    """Exchange types for different asset classes."""

    SPOT = "spot"
    PERP = "perp"
    OPTIONS = "options"


@dataclass
class ExchangeConfig:
    """Exchange configuration and fee structure."""

    name: str
    exchange_type: ExchangeType
    maker_fee_bps: float
    taker_fee_bps: float
    withdrawal_fee_fixed: float  # Fixed withdrawal fee in USD
    withdrawal_fee_pct: float  # Percentage withdrawal fee
    min_trade_size: float
    max_trade_size: float
    latency_ms: float  # Expected latency
    reliability: float  # Reliability score (0-1)
    active: bool = True


@dataclass
class MarketData:
    """Market data snapshot."""

    symbol: str
    exchange: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid_price: float
    spread_bps: float
    timestamp: float
    valid: bool = True


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    max_size: float
    gross_spread_bps: float
    net_spread_bps: float  # After fees
    risk_adjusted_bps: float  # After fees, latency, reliability
    estimated_profit_usd: float
    timestamp: float
    confidence: float  # Confidence score (0-1)
    latency_risk_bps: float  # Latency-based risk adjustment


class ArbitrageScanner:
    """
    Real-time cross-exchange arbitrage scanner.

    Monitors multiple exchanges and identifies profitable arbitrage opportunities
    with risk adjustments for fees, latency, and market conditions.
    """

    def __init__(
        self, symbols: List[str] = None, exchanges: Dict[str, ExchangeConfig] = None
    ):
        """
        Initialize arbitrage scanner.

        Args:
            symbols: List of symbols to monitor (default: ["BTC", "ETH", "SOL"])
            exchanges: Exchange configurations
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL"]

        if exchanges is None:
            exchanges = self._get_default_exchanges()

        self.symbols = symbols
        self.exchanges = exchanges

        # Market data storage
        self.market_data: Dict[Tuple[str, str], MarketData] = (
            {}
        )  # (symbol, exchange) -> data
        self.price_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Arbitrage opportunities
        self.opportunities: Dict[str, List[ArbitrageOpportunity]] = defaultdict(list)
        self.opportunity_history: List[ArbitrageOpportunity] = deque(maxlen=10000)

        # Configuration
        self.config = {
            "min_spread_bps": 5.0,  # Minimum gross spread to consider
            "min_net_spread_bps": 2.0,  # Minimum net spread after fees
            "max_latency_risk_bps": 3.0,  # Maximum latency risk to accept
            "min_size_usd": 1000,  # Minimum trade size in USD
            "max_size_usd": 100000,  # Maximum trade size in USD
            "price_staleness_seconds": 30,  # Max age of price data
            "opportunity_ttl_seconds": 10,  # How long opportunities remain valid
            "volume_impact_threshold": 0.1,  # Volume impact threshold (10% of book)
            "confidence_threshold": 0.7,  # Minimum confidence to publish
        }

        # Performance metrics
        self.stats = {
            "total_scans": 0,
            "opportunities_found": 0,
            "opportunities_published": 0,
            "avg_scan_time_ms": 0.0,
            "best_spread_today": 0.0,
            "exchanges_active": len([e for e in exchanges.values() if e.active]),
        }

        logger.info(f"Initialized arbitrage scanner for {symbols}")
        logger.info(f"  Exchanges: {list(exchanges.keys())}")
        logger.info(
            f"  Min spreads: gross>{self.config['min_spread_bps']}bps, "
            f"net>{self.config['min_net_spread_bps']}bps"
        )

    def _get_default_exchanges(self) -> Dict[str, ExchangeConfig]:
        """Get default exchange configurations."""
        return {
            "binance_spot": ExchangeConfig(
                name="binance_spot",
                exchange_type=ExchangeType.SPOT,
                maker_fee_bps=10,  # 0.1%
                taker_fee_bps=10,
                withdrawal_fee_fixed=0,
                withdrawal_fee_pct=0,
                min_trade_size=10,
                max_trade_size=1000000,
                latency_ms=50,
                reliability=0.98,
            ),
            "coinbase_spot": ExchangeConfig(
                name="coinbase_spot",
                exchange_type=ExchangeType.SPOT,
                maker_fee_bps=50,  # 0.5%
                taker_fee_bps=60,  # 0.6%
                withdrawal_fee_fixed=0,
                withdrawal_fee_pct=0,
                min_trade_size=1,
                max_trade_size=500000,
                latency_ms=80,
                reliability=0.95,
            ),
            "binance_perp": ExchangeConfig(
                name="binance_perp",
                exchange_type=ExchangeType.PERP,
                maker_fee_bps=2,  # 0.02%
                taker_fee_bps=4,  # 0.04%
                withdrawal_fee_fixed=0,
                withdrawal_fee_pct=0,
                min_trade_size=5,
                max_trade_size=2000000,
                latency_ms=40,
                reliability=0.99,
            ),
            "deribit_perp": ExchangeConfig(
                name="deribit_perp",
                exchange_type=ExchangeType.PERP,
                maker_fee_bps=2.5,  # 0.025%
                taker_fee_bps=7.5,  # 0.075%
                withdrawal_fee_fixed=0,
                withdrawal_fee_pct=0,
                min_trade_size=1,
                max_trade_size=1000000,
                latency_ms=60,
                reliability=0.96,
            ),
        }

    async def update_market_data(
        self,
        symbol: str,
        exchange: str,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        timestamp: float = None,
    ):
        """Update market data for a symbol-exchange pair."""
        try:
            if timestamp is None:
                timestamp = time.time()

            # Validate inputs
            if bid <= 0 or ask <= 0 or bid >= ask:
                logger.warning(
                    f"Invalid market data: {exchange} {symbol} bid={bid} ask={ask}"
                )
                return

            if exchange not in self.exchanges:
                logger.warning(f"Unknown exchange: {exchange}")
                return

            # Calculate derived values
            mid_price = (bid + ask) / 2
            spread_bps = (ask - bid) / mid_price * 10000

            # Create market data
            market_data = MarketData(
                symbol=symbol,
                exchange=exchange,
                bid_price=bid,
                ask_price=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                mid_price=mid_price,
                spread_bps=spread_bps,
                timestamp=timestamp,
                valid=True,
            )

            # Store market data
            key = (symbol, exchange)
            self.market_data[key] = market_data

            # Store price history
            self.price_history[key].append((timestamp, mid_price))

            # Trigger arbitrage scan for this symbol
            await self._scan_arbitrage_for_symbol(symbol)

        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    async def _scan_arbitrage_for_symbol(self, symbol: str):
        """Scan for arbitrage opportunities for a specific symbol."""
        try:
            start_time = time.time()
            current_time = time.time()

            # Get all market data for this symbol
            symbol_data = {}
            for exchange in self.exchanges:
                key = (symbol, exchange)
                if key in self.market_data:
                    data = self.market_data[key]

                    # Check if data is fresh
                    age = current_time - data.timestamp
                    if age <= self.config["price_staleness_seconds"]:
                        symbol_data[exchange] = data

            if len(symbol_data) < 2:
                return  # Need at least 2 exchanges

            # Find arbitrage opportunities
            opportunities = []
            exchanges_list = list(symbol_data.keys())

            for i in range(len(exchanges_list)):
                for j in range(i + 1, len(exchanges_list)):
                    exchange_a = exchanges_list[i]
                    exchange_b = exchanges_list[j]

                    data_a = symbol_data[exchange_a]
                    data_b = symbol_data[exchange_b]

                    # Check both directions
                    opp_a_to_b = self._analyze_arbitrage(symbol, data_a, data_b)
                    opp_b_to_a = self._analyze_arbitrage(symbol, data_b, data_a)

                    for opp in [opp_a_to_b, opp_b_to_a]:
                        if opp and self._validate_opportunity(opp):
                            opportunities.append(opp)

            # Update opportunities
            self.opportunities[symbol] = opportunities

            # Publish top opportunities
            if opportunities:
                await self._publish_opportunities(symbol, opportunities)

            # Update stats
            self.stats["total_scans"] += 1
            self.stats["opportunities_found"] += len(opportunities)

            scan_time = (time.time() - start_time) * 1000
            self.stats["avg_scan_time_ms"] = (
                self.stats["avg_scan_time_ms"] * (self.stats["total_scans"] - 1)
                + scan_time
            ) / self.stats["total_scans"]

            # Update best spread
            if opportunities:
                best_spread = max(opp.risk_adjusted_bps for opp in opportunities)
                self.stats["best_spread_today"] = max(
                    self.stats["best_spread_today"], best_spread
                )

        except Exception as e:
            logger.error(f"Error scanning arbitrage for {symbol}: {e}")

    def _analyze_arbitrage(
        self, symbol: str, buy_data: MarketData, sell_data: MarketData
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analyze potential arbitrage between two exchanges.

        Args:
            symbol: Trading symbol
            buy_data: Market data for buy side
            sell_data: Market data for sell side
        """
        try:
            buy_exchange = buy_data.exchange
            sell_exchange = sell_data.exchange

            # Can't arbitrage same exchange
            if buy_exchange == sell_exchange:
                return None

            # Basic arbitrage check: buy low, sell high
            buy_price = buy_data.ask_price  # We buy at ask
            sell_price = sell_data.bid_price  # We sell at bid

            if sell_price <= buy_price:
                return None  # No arbitrage opportunity

            # Calculate gross spread
            gross_spread_bps = (sell_price - buy_price) / buy_price * 10000

            if gross_spread_bps < self.config["min_spread_bps"]:
                return None

            # Get exchange configs
            buy_config = self.exchanges[buy_exchange]
            sell_config = self.exchanges[sell_exchange]

            # Calculate fees
            buy_fee_bps = buy_config.taker_fee_bps  # Assume taker
            sell_fee_bps = sell_config.taker_fee_bps
            total_fee_bps = buy_fee_bps + sell_fee_bps

            # Net spread after fees
            net_spread_bps = gross_spread_bps - total_fee_bps

            if net_spread_bps < self.config["min_net_spread_bps"]:
                return None

            # Calculate maximum trade size based on liquidity
            max_size = min(
                buy_data.ask_size,
                sell_data.bid_size,
                buy_config.max_trade_size,
                sell_config.max_trade_size,
            )

            # Check minimum size in USD
            trade_value_usd = max_size * buy_price
            if trade_value_usd < self.config["min_size_usd"]:
                return None

            # Cap at maximum size
            if trade_value_usd > self.config["max_size_usd"]:
                max_size = self.config["max_size_usd"] / buy_price
                trade_value_usd = self.config["max_size_usd"]

            # Risk adjustments
            latency_risk_bps = self._calculate_latency_risk(buy_config, sell_config)
            reliability_adjustment = (
                buy_config.reliability + sell_config.reliability
            ) / 2

            # Risk-adjusted spread
            risk_adjusted_bps = net_spread_bps - latency_risk_bps

            if (
                risk_adjusted_bps < self.config["min_net_spread_bps"]
                or latency_risk_bps > self.config["max_latency_risk_bps"]
            ):
                return None

            # Estimate profit
            estimated_profit_usd = (risk_adjusted_bps / 10000) * trade_value_usd

            # Calculate confidence
            confidence = self._calculate_confidence(
                buy_data, sell_data, buy_config, sell_config, risk_adjusted_bps
            )

            if confidence < self.config["confidence_threshold"]:
                return None

            # Create opportunity
            opportunity = ArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                max_size=max_size,
                gross_spread_bps=gross_spread_bps,
                net_spread_bps=net_spread_bps,
                risk_adjusted_bps=risk_adjusted_bps,
                estimated_profit_usd=estimated_profit_usd,
                timestamp=time.time(),
                confidence=confidence,
                latency_risk_bps=latency_risk_bps,
            )

            return opportunity

        except Exception as e:
            logger.error(f"Error analyzing arbitrage: {e}")
            return None

    def _calculate_latency_risk(
        self, buy_config: ExchangeConfig, sell_config: ExchangeConfig
    ) -> float:
        """Calculate latency-based risk adjustment."""
        try:
            # Higher latency increases risk of prices moving
            avg_latency = (buy_config.latency_ms + sell_config.latency_ms) / 2

            # Risk increases non-linearly with latency
            # Base risk: 0.5 bps per 100ms of latency
            base_risk_bps = (avg_latency / 100) * 0.5

            # Additional risk for high latency
            if avg_latency > 100:
                extra_risk_bps = ((avg_latency - 100) / 100) * 1.0
                base_risk_bps += extra_risk_bps

            return base_risk_bps

        except Exception:
            return 1.0  # Default risk

    def _calculate_confidence(
        self,
        buy_data: MarketData,
        sell_data: MarketData,
        buy_config: ExchangeConfig,
        sell_config: ExchangeConfig,
        risk_adjusted_bps: float,
    ) -> float:
        """Calculate confidence score for arbitrage opportunity."""
        try:
            confidence = 1.0

            # Reduce confidence for wide spreads (less liquid)
            avg_spread_bps = (buy_data.spread_bps + sell_data.spread_bps) / 2
            if avg_spread_bps > 20:  # 20 bps
                confidence *= 0.8
            elif avg_spread_bps > 50:  # 50 bps
                confidence *= 0.6

            # Reduce confidence for low reliability exchanges
            avg_reliability = (buy_config.reliability + sell_config.reliability) / 2
            confidence *= avg_reliability

            # Reduce confidence for small opportunities
            if risk_adjusted_bps < 5:
                confidence *= 0.7
            elif risk_adjusted_bps < 10:
                confidence *= 0.9

            # Reduce confidence for stale data
            current_time = time.time()
            max_age = max(
                current_time - buy_data.timestamp, current_time - sell_data.timestamp
            )

            if max_age > 10:  # 10 seconds
                confidence *= 0.8
            elif max_age > 20:  # 20 seconds
                confidence *= 0.6

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5  # Default confidence

    def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate arbitrage opportunity."""
        try:
            # Check basic thresholds
            if opportunity.risk_adjusted_bps < self.config["min_net_spread_bps"]:
                return False

            if opportunity.confidence < self.config["confidence_threshold"]:
                return False

            if opportunity.latency_risk_bps > self.config["max_latency_risk_bps"]:
                return False

            # Check trade size
            trade_value = opportunity.max_size * opportunity.buy_price
            if trade_value < self.config["min_size_usd"]:
                return False

            # Check exchange activity
            buy_config = self.exchanges.get(opportunity.buy_exchange)
            sell_config = self.exchanges.get(opportunity.sell_exchange)

            if not (
                buy_config and sell_config and buy_config.active and sell_config.active
            ):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating opportunity: {e}")
            return False

    async def _publish_opportunities(
        self, symbol: str, opportunities: List[ArbitrageOpportunity]
    ):
        """Publish arbitrage opportunities to Redis."""
        try:
            if not opportunities:
                return

            # Sort by risk-adjusted spread (best first)
            opportunities.sort(key=lambda x: x.risk_adjusted_bps, reverse=True)

            # Prepare data for Redis ZSET (sorted set)
            opportunity_data = {}
            for i, opp in enumerate(opportunities[:10]):  # Top 10 opportunities
                key = f"{opp.buy_exchange}→{opp.sell_exchange}"
                score = opp.risk_adjusted_bps

                data = {
                    "symbol": opp.symbol,
                    "buy_exchange": opp.buy_exchange,
                    "sell_exchange": opp.sell_exchange,
                    "buy_price": opp.buy_price,
                    "sell_price": opp.sell_price,
                    "max_size": opp.max_size,
                    "gross_spread_bps": opp.gross_spread_bps,
                    "net_spread_bps": opp.net_spread_bps,
                    "risk_adjusted_bps": opp.risk_adjusted_bps,
                    "estimated_profit_usd": opp.estimated_profit_usd,
                    "confidence": opp.confidence,
                    "timestamp": opp.timestamp,
                }

                opportunity_data[key] = {"score": score, "data": json.dumps(data)}

            # Publish to Redis
            if ASYNC_REDIS_AVAILABLE and opportunity_data:
                redis = await get_redis()

                # Clear old opportunities
                arb_key = f"arb:opps:{symbol}"
                if hasattr(redis, "delete"):
                    await redis.delete(arb_key)

                # Add new opportunities to sorted set
                if hasattr(redis, "zadd"):
                    zadd_data = {}
                    for key, value in opportunity_data.items():
                        zadd_data[value["data"]] = value["score"]

                    if zadd_data:
                        await redis.zadd(arb_key, zadd_data)
                        # Set TTL
                        await redis.expire(
                            arb_key, self.config["opportunity_ttl_seconds"]
                        )

                # Update metrics
                best_spread = opportunities[0].risk_adjusted_bps
                await set_metric(f"arb_best_bps_{symbol.lower()}", best_spread)
                await set_metric(f"arb_opps_count_{symbol.lower()}", len(opportunities))

                # Count opportunities above threshold
                good_opps = len(
                    [opp for opp in opportunities if opp.risk_adjusted_bps >= 2.0]
                )
                await set_metric(f"arb_opps_gt_2bps_{symbol.lower()}", good_opps)

            # Store in history
            for opp in opportunities:
                self.opportunity_history.append(opp)

            self.stats["opportunities_published"] += len(opportunities)

            # Log best opportunity
            best = opportunities[0]
            logger.info(
                f"📊 Arbitrage: {symbol} {best.buy_exchange}→{best.sell_exchange} "
                f"{best.risk_adjusted_bps:.1f}bps (${best.estimated_profit_usd:.0f} profit, "
                f"conf={best.confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Error publishing opportunities: {e}")

    def get_top_opportunities(
        self, symbol: str, limit: int = 5
    ) -> List[ArbitrageOpportunity]:
        """Get top arbitrage opportunities for a symbol."""
        try:
            opps = self.opportunities.get(symbol, [])
            return sorted(opps, key=lambda x: x.risk_adjusted_bps, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error getting top opportunities: {e}")
            return []

    def get_scanner_status(self) -> Dict[str, Any]:
        """Get comprehensive scanner status."""
        try:
            current_time = time.time()

            status = {
                "symbols": self.symbols,
                "exchanges": {
                    name: asdict(config) for name, config in self.exchanges.items()
                },
                "config": self.config.copy(),
                "stats": self.stats.copy(),
                "market_data_age": {},
                "current_opportunities": {},
                "recent_history": [],
            }

            # Add market data freshness
            for symbol in self.symbols:
                status["market_data_age"][symbol] = {}
                for exchange in self.exchanges:
                    key = (symbol, exchange)
                    if key in self.market_data:
                        age = current_time - self.market_data[key].timestamp
                        status["market_data_age"][symbol][exchange] = age

            # Add current opportunities
            for symbol in self.symbols:
                opps = self.get_top_opportunities(symbol, 3)
                status["current_opportunities"][symbol] = [
                    {
                        "buy_exchange": opp.buy_exchange,
                        "sell_exchange": opp.sell_exchange,
                        "risk_adjusted_bps": opp.risk_adjusted_bps,
                        "estimated_profit_usd": opp.estimated_profit_usd,
                        "confidence": opp.confidence,
                    }
                    for opp in opps
                ]

            # Add recent history
            recent_opps = [
                opp
                for opp in self.opportunity_history
                if current_time - opp.timestamp < 300
            ]  # Last 5 minutes
            status["recent_history"] = [
                {
                    "symbol": opp.symbol,
                    "spread_bps": opp.risk_adjusted_bps,
                    "profit_usd": opp.estimated_profit_usd,
                    "timestamp": opp.timestamp,
                }
                for opp in recent_opps[-20:]  # Last 20
            ]

            return status

        except Exception as e:
            logger.error(f"Error getting scanner status: {e}")
            return {"error": str(e)}


async def main():
    """Test the arbitrage scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Exchange Arbitrage Scanner")
    parser.add_argument("--symbol", default="BTC", help="Symbol to scan")
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic data"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Start scanner in dry-run mode"
    )

    args = parser.parse_args()

    if args.test:
        # Test with synthetic market data
        scanner = ArbitrageScanner([args.symbol])

        base_price = 50000

        for i in range(50):
            # Generate synthetic market data with arbitrage opportunities
            exchanges = ["binance_spot", "coinbase_spot", "binance_perp"]

            for j, exchange in enumerate(exchanges):
                # Add some price variance between exchanges
                price_offset = (j - 1) * 10 + np.random.normal(0, 5)
                mid_price = base_price + price_offset

                # Create bid/ask spread
                spread_pct = 0.05 + np.random.uniform(0, 0.05)  # 0.05-0.1%
                spread = mid_price * spread_pct / 100

                bid = mid_price - spread / 2
                ask = mid_price + spread / 2

                # Random sizes
                bid_size = np.random.uniform(1, 10)
                ask_size = np.random.uniform(1, 10)

                await scanner.update_market_data(
                    args.symbol, exchange, bid, ask, bid_size, ask_size
                )

            # Occasionally create larger arbitrage gaps
            if i % 10 == 0:
                # Make coinbase more expensive
                await scanner.update_market_data(
                    args.symbol,
                    "coinbase_spot",
                    base_price + 50,
                    base_price + 70,
                    5.0,
                    5.0,
                )

            await asyncio.sleep(0.1)

            # Print status every 10 iterations
            if i % 10 == 0:
                opps = scanner.get_top_opportunities(args.symbol, 3)
                if opps:
                    best = opps[0]
                    print(
                        f"Iteration {i}: Best arbitrage: {best.buy_exchange}→{best.sell_exchange} "
                        f"{best.risk_adjusted_bps:.1f}bps (${best.estimated_profit_usd:.0f})"
                    )
                else:
                    print(f"Iteration {i}: No arbitrage opportunities found")

        # Final status
        final_status = scanner.get_scanner_status()
        print(f"\nScanner statistics:")
        print(f"  Total scans: {final_status['stats']['total_scans']}")
        print(f"  Opportunities found: {final_status['stats']['opportunities_found']}")
        print(
            f"  Best spread today: {final_status['stats']['best_spread_today']:.1f}bps"
        )

    elif args.dry_run:
        # Start scanner in dry-run mode
        scanner = ArbitrageScanner([args.symbol])
        logger.info(f"Starting arbitrage scanner for {args.symbol} in dry-run mode...")

        try:
            # In real implementation, this would connect to live market data feeds
            while True:
                await asyncio.sleep(1)

                # Mock some market data updates
                base_price = 50000
                for exchange in scanner.exchanges:
                    if scanner.exchanges[exchange].active:
                        price_noise = np.random.normal(0, 10)
                        mid_price = base_price + price_noise
                        spread = mid_price * 0.001  # 0.1% spread

                        await scanner.update_market_data(
                            args.symbol,
                            exchange,
                            mid_price - spread / 2,
                            mid_price + spread / 2,
                            5.0,
                            5.0,
                        )

        except KeyboardInterrupt:
            logger.info("Arbitrage scanner stopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
