#!/usr/bin/env python3
"""
Statistical Arbitrage - Pairs Trading and Cross-Asset Correlations
Implements mean-reverting strategies based on statistical relationships between assets
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatArbConfig:
    """Configuration for statistical arbitrage strategies"""

    lookback_days: int = 60  # Lookback period for analysis
    formation_period: int = 252  # Formation period for pair selection
    trading_period: int = 21  # Trading period length
    min_correlation: float = 0.7  # Minimum correlation for pair selection
    entry_threshold: float = 2.0  # Z-score entry threshold
    exit_threshold: float = 0.5  # Z-score exit threshold
    stop_loss: float = 3.0  # Stop loss Z-score threshold
    max_holding_period: int = 10  # Maximum holding period in days
    transaction_cost: float = 0.001  # Transaction cost (0.1%)
    min_half_life: int = 1  # Minimum half-life for mean reversion
    max_half_life: int = 30  # Maximum half-life for mean reversion


class CointegrationAnalyzer:
    """Analyze cointegration relationships between asset pairs"""

    def __init__(self, config: StatArbConfig):
        self.config = config

    def engle_granger_test(self, y: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
        """Perform Engle-Granger cointegration test"""
        try:
            # Step 1: Run regression y = α + β*x + ε
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # Step 2: Calculate residuals
            residuals = y - (beta[0] + beta[1] * x)

            # Step 3: ADF test on residuals
            adf_result = self.adf_test(residuals)

            # Calculate additional statistics
            durbin_watson = self.durbin_watson_stat(residuals)
            half_life = self.calculate_half_life(residuals)

            return {
                "cointegrated": adf_result["p_value"] < 0.05,
                "p_value": adf_result["p_value"],
                "adf_statistic": adf_result["statistic"],
                "critical_values": adf_result["critical_values"],
                "alpha": float(beta[0]),
                "beta": float(beta[1]),
                "residuals": residuals,
                "durbin_watson": durbin_watson,
                "half_life": half_life,
                "spread_std": float(np.std(residuals)),
                "spread_mean": float(np.mean(residuals)),
            }
        except Exception as e:
            logger.error(f"Error in Engle-Granger test: {e}")
            return {"cointegrated": False, "error": str(e)}

    def adf_test(self, series: np.ndarray, max_lags: int = 10) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test implementation"""
        n = len(series)
        y = series[1:]
        x = series[:-1]

        # Add lagged differences
        max_lags = min(max_lags, n // 4)
        for lag in range(1, max_lags + 1):
            if lag < len(series) - 1:
                diff_lag = (
                    np.diff(series, n=1)[:-lag] if lag > 0 else np.diff(series, n=1)
                )
                if len(diff_lag) == len(y):
                    x = np.column_stack([x[:-lag] if lag > 0 else x, diff_lag])

        # Add constant and trend
        X = np.column_stack([np.ones(len(y)), x])

        try:
            # Run regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta

            # Calculate t-statistic for unit root
            mse = np.sum(residuals**2) / (len(residuals) - len(beta))
            var_beta = mse * np.linalg.inv(X.T @ X)
            t_stat = beta[1] / np.sqrt(var_beta[1, 1])

            # Critical values (approximate)
            critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

            # Calculate p-value (approximation)
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

            return {
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "critical_values": critical_values,
            }
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "critical_values": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
            }

    def durbin_watson_stat(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        if len(residuals) < 2:
            return 2.0

        diff_residuals = np.diff(residuals)
        return float(np.sum(diff_residuals**2) / np.sum(residuals**2))

    def calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate half-life of mean reversion"""
        try:
            y = spread[1:]
            x = spread[:-1]

            # Run regression: y_t = α + β*y_{t-1} + ε
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # Half-life = -ln(2) / ln(β)
            if abs(beta[1]) < 1 and beta[1] > 0:
                half_life = -np.log(2) / np.log(beta[1])
                return float(half_life)
            else:
                return float("inf")
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return float("inf")


class PairsTradingStrategy:
    """Pairs trading strategy implementation"""

    def __init__(self, config: StatArbConfig):
        self.config = config
        self.coint_analyzer = CointegrationAnalyzer(config)

    def find_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Find cointegrated pairs from price data"""
        pairs = []
        assets = list(price_data.keys())

        logger.info(
            f"🔍 Searching for cointegrated pairs among {len(assets)} assets..."
        )

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:  # Only analyze upper triangle
                    try:
                        # Get price series
                        df1 = price_data[asset1].tail(self.config.formation_period)
                        df2 = price_data[asset2].tail(self.config.formation_period)

                        if len(df1) < 50 or len(df2) < 50:
                            continue

                        # Align dates
                        common_dates = df1.index.intersection(df2.index)
                        if len(common_dates) < 50:
                            continue

                        prices1 = df1.loc[common_dates, "close"].values
                        prices2 = df2.loc[common_dates, "close"].values

                        # Check correlation first
                        correlation = np.corrcoef(prices1, prices2)[0, 1]
                        if abs(correlation) < self.config.min_correlation:
                            continue

                        # Test cointegration
                        coint_result = self.coint_analyzer.engle_granger_test(
                            prices1, prices2
                        )

                        if coint_result.get("cointegrated", False):
                            # Check half-life constraint
                            half_life = coint_result.get("half_life", float("inf"))
                            if (
                                self.config.min_half_life
                                <= half_life
                                <= self.config.max_half_life
                            ):
                                pairs.append(
                                    {
                                        "asset1": asset1,
                                        "asset2": asset2,
                                        "correlation": float(correlation),
                                        "cointegration": coint_result,
                                        "formation_score": self._calculate_formation_score(
                                            coint_result, correlation
                                        ),
                                    }
                                )

                                logger.info(
                                    f"✅ Found cointegrated pair: {asset1}-{asset2} (ρ={correlation:.3f}, HL={half_life:.1f})"
                                )

                    except Exception as e:
                        logger.error(f"Error analyzing pair {asset1}-{asset2}: {e}")
                        continue

        # Sort pairs by formation score
        pairs.sort(key=lambda x: x["formation_score"], reverse=True)
        logger.info(f"🎯 Found {len(pairs)} viable trading pairs")

        return pairs[:10]  # Return top 10 pairs

    def _calculate_formation_score(
        self, coint_result: Dict[str, Any], correlation: float
    ) -> float:
        """Calculate formation period score for pair ranking"""
        score = 0.0

        # Cointegration strength (lower p-value is better)
        p_value = coint_result.get("p_value", 1.0)
        score += (1 - p_value) * 40  # 40 points max

        # Correlation strength
        score += abs(correlation) * 30  # 30 points max

        # Half-life (prefer moderate values)
        half_life = coint_result.get("half_life", float("inf"))
        if 1 <= half_life <= 10:
            score += 20  # 20 points for good half-life
        elif 10 < half_life <= 20:
            score += 10  # 10 points for acceptable half-life

        # Durbin-Watson (prefer values around 2)
        dw = coint_result.get("durbin_watson", 2.0)
        score += (2 - abs(2 - dw)) * 10  # 10 points max

        return score

    def generate_signals(
        self, pair: Dict[str, Any], recent_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate trading signals for a pair"""
        try:
            asset1 = pair["asset1"]
            asset2 = pair["asset2"]

            # Get recent price data
            df1 = recent_data[asset1].tail(self.config.lookback_days)
            df2 = recent_data[asset2].tail(self.config.lookback_days)

            # Align dates
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) < 10:
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "error": "Insufficient data",
                }

            prices1 = df1.loc[common_dates, "close"].values
            prices2 = df2.loc[common_dates, "close"].values

            # Calculate spread using cointegration parameters
            alpha = pair["cointegration"]["alpha"]
            beta = pair["cointegration"]["beta"]
            spread = prices1 - (alpha + beta * prices2)

            # Calculate z-score
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            current_z_score = (
                (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
            )

            # Generate signal
            signal_info = self._generate_signal_from_zscore(current_z_score, spread)

            # Add pair-specific information
            signal_info.update(
                {
                    "pair": f"{asset1}-{asset2}",
                    "current_spread": float(spread[-1]),
                    "z_score": float(current_z_score),
                    "spread_mean": float(spread_mean),
                    "spread_std": float(spread_std),
                    "prices": {asset1: float(prices1[-1]), asset2: float(prices2[-1])},
                    "hedge_ratio": float(beta),
                    "half_life": pair["cointegration"].get("half_life", 0),
                }
            )

            return signal_info

        except Exception as e:
            logger.error(
                f"Error generating signals for pair {pair.get('asset1', '')}-{pair.get('asset2', '')}: {e}"
            )
            return {"signal": "HOLD", "confidence": 0.0, "error": str(e)}

    def _generate_signal_from_zscore(
        self, z_score: float, spread_history: np.ndarray
    ) -> Dict[str, Any]:
        """Generate trading signal from z-score"""
        signal = "HOLD"
        confidence = 0.0
        reasoning = []

        # Entry signals
        if abs(z_score) >= self.config.entry_threshold:
            if z_score > 0:
                signal = "SHORT_SPREAD"  # Short asset1, Long asset2
                reasoning.append(f"Spread {z_score:.2f} std above mean - short spread")
            else:
                signal = "LONG_SPREAD"  # Long asset1, Short asset2
                reasoning.append(
                    f"Spread {abs(z_score):.2f} std below mean - long spread"
                )

            confidence = min(abs(z_score) / self.config.entry_threshold, 2.0) * 0.5

        # Exit signals
        elif abs(z_score) <= self.config.exit_threshold:
            signal = "EXIT"
            confidence = 0.3
            reasoning.append(f"Spread normalized (z={z_score:.2f}) - exit position")

        # Stop loss
        elif abs(z_score) >= self.config.stop_loss:
            signal = "STOP_LOSS"
            confidence = 0.8
            reasoning.append(f"Stop loss triggered (z={z_score:.2f})")

        # Additional confidence factors
        if len(spread_history) > 5:
            # Trend consistency
            recent_trend = np.polyfit(range(5), spread_history[-5:], 1)[0]
            if (signal == "LONG_SPREAD" and recent_trend < 0) or (
                signal == "SHORT_SPREAD" and recent_trend > 0
            ):
                confidence *= 1.2  # Boost confidence if trend supports signal

            # Volatility adjustment
            recent_vol = (
                np.std(spread_history[-10:])
                if len(spread_history) >= 10
                else np.std(spread_history)
            )
            historical_vol = np.std(spread_history)
            if recent_vol < historical_vol * 0.8:  # Low volatility environment
                confidence *= 1.1

        confidence = min(confidence, 1.0)  # Cap at 1.0

        return {
            "signal": signal,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "signal_strength": abs(z_score),
        }


class CrossAssetArbitrage:
    """Cross-asset arbitrage opportunities detector"""

    def __init__(self, config: StatArbConfig):
        self.config = config

    def find_arbitrage_opportunities(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Find cross-asset arbitrage opportunities"""
        opportunities = []

        # Multi-asset momentum arbitrage
        momentum_opps = self._momentum_arbitrage(price_data)
        opportunities.extend(momentum_opps)

        # Volatility arbitrage
        vol_opps = self._volatility_arbitrage(price_data)
        opportunities.extend(vol_opps)

        # Cross-exchange arbitrage (simulated)
        cross_exchange_opps = self._cross_exchange_arbitrage(price_data)
        opportunities.extend(cross_exchange_opps)

        return opportunities

    def _momentum_arbitrage(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Detect momentum-based arbitrage opportunities"""
        opportunities = []

        for asset, df in price_data.items():
            if len(df) < 20:
                continue

            # Calculate momentum indicators
            returns = df["close"].pct_change().dropna()
            momentum_5d = returns.rolling(5).mean().iloc[-1] if len(returns) >= 5 else 0
            momentum_20d = (
                returns.rolling(20).mean().iloc[-1] if len(returns) >= 20 else 0
            )

            # RSI calculation
            delta = returns
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

            # Momentum divergence signal
            if momentum_5d > 0.02 and momentum_20d < -0.01 and current_rsi < 30:
                opportunities.append(
                    {
                        "type": "momentum_reversal",
                        "asset": asset,
                        "signal": "LONG",
                        "strength": abs(momentum_5d - momentum_20d),
                        "rsi": float(current_rsi),
                        "confidence": 0.6,
                        "reasoning": f"Strong short-term momentum ({momentum_5d:.3f}) vs weak long-term ({momentum_20d:.3f}), oversold RSI",
                    }
                )
            elif momentum_5d < -0.02 and momentum_20d > 0.01 and current_rsi > 70:
                opportunities.append(
                    {
                        "type": "momentum_reversal",
                        "asset": asset,
                        "signal": "SHORT",
                        "strength": abs(momentum_5d - momentum_20d),
                        "rsi": float(current_rsi),
                        "confidence": 0.6,
                        "reasoning": f"Strong negative momentum ({momentum_5d:.3f}) vs positive long-term ({momentum_20d:.3f}), overbought RSI",
                    }
                )

        return opportunities

    def _volatility_arbitrage(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Detect volatility-based arbitrage opportunities"""
        opportunities = []

        for asset, df in price_data.items():
            if len(df) < 30:
                continue

            returns = df["close"].pct_change().dropna()

            # Calculate realized vs historical volatility
            realized_vol_5d = (
                returns.rolling(5).std().iloc[-1] * np.sqrt(252)
                if len(returns) >= 5
                else 0
            )
            historical_vol_30d = (
                returns.rolling(30).std().iloc[-1] * np.sqrt(252)
                if len(returns) >= 30
                else 0
            )

            # Volatility ratio
            vol_ratio = (
                realized_vol_5d / historical_vol_30d if historical_vol_30d > 0 else 1
            )

            # Volatility clustering signal
            if vol_ratio > 2.0:  # High recent volatility
                opportunities.append(
                    {
                        "type": "volatility_clustering",
                        "asset": asset,
                        "signal": "VOL_SHORT",  # Expect mean reversion
                        "vol_ratio": float(vol_ratio),
                        "realized_vol": float(realized_vol_5d),
                        "historical_vol": float(historical_vol_30d),
                        "confidence": min(vol_ratio / 3.0, 0.8),
                        "reasoning": f"Volatility spike: {vol_ratio:.2f}x historical - expect mean reversion",
                    }
                )
            elif vol_ratio < 0.5:  # Low recent volatility
                opportunities.append(
                    {
                        "type": "volatility_expansion",
                        "asset": asset,
                        "signal": "VOL_LONG",  # Expect volatility increase
                        "vol_ratio": float(vol_ratio),
                        "realized_vol": float(realized_vol_5d),
                        "historical_vol": float(historical_vol_30d),
                        "confidence": min((1 - vol_ratio) * 0.8, 0.7),
                        "reasoning": f"Low volatility: {vol_ratio:.2f}x historical - expect expansion",
                    }
                )

        return opportunities

    def _cross_exchange_arbitrage(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Detect simulated cross-exchange arbitrage opportunities"""
        opportunities = []

        # Simulate price differences between exchanges
        for asset, df in price_data.items():
            if len(df) == 0:
                continue

            current_price = df["close"].iloc[-1]

            # Simulate exchange price differences (demo)
            exchange_prices = {
                "Binance": current_price * (1 + np.random.normal(0, 0.001)),
                "Coinbase": current_price * (1 + np.random.normal(0, 0.0015)),
                "Kraken": current_price * (1 + np.random.normal(0, 0.0012)),
            }

            # Find arbitrage opportunities
            max_exchange = max(exchange_prices, key=exchange_prices.get)
            min_exchange = min(exchange_prices, key=exchange_prices.get)

            price_diff = exchange_prices[max_exchange] - exchange_prices[min_exchange]
            price_diff_pct = (price_diff / exchange_prices[min_exchange]) * 100

            # Account for transaction costs
            net_profit_pct = price_diff_pct - (2 * self.config.transaction_cost * 100)

            if net_profit_pct > 0.05:  # Minimum 0.05% profit after costs
                opportunities.append(
                    {
                        "type": "cross_exchange",
                        "asset": asset,
                        "buy_exchange": min_exchange,
                        "sell_exchange": max_exchange,
                        "buy_price": float(exchange_prices[min_exchange]),
                        "sell_price": float(exchange_prices[max_exchange]),
                        "profit_pct": float(net_profit_pct),
                        "confidence": min(net_profit_pct / 0.2, 0.9),
                        "reasoning": f"Price gap: {price_diff_pct:.3f}% between {min_exchange} and {max_exchange}",
                    }
                )

        return opportunities


class StatisticalArbitrageEngine:
    """Main engine for statistical arbitrage strategies"""

    def __init__(self, config: Optional[StatArbConfig] = None):
        self.config = config or StatArbConfig()
        self.pairs_strategy = PairsTradingStrategy(self.config)
        self.cross_asset = CrossAssetArbitrage(self.config)

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for statistical arbitrage")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate historical price data for arbitrage analysis"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Generate realistic crypto price data
        np.random.seed(hash(symbol) % 2**32)

        # Base prices for different cryptos
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "ADAUSDT": 0.5,
            "DOTUSDT": 8.0,
            "LINKUSDT": 15.0,
            "LTCUSDT": 100,
            "XRPUSDT": 0.6,
            "BCHUSDT": 200,
            "BNBUSDT": 300,
            "SOLUSDT": 100,
        }

        base_price = base_prices.get(symbol, 100)

        # Generate price series with mean reversion and cointegration patterns
        timestamps = pd.date_range(start=start_time, end=end_time, freq="1D")[:days]

        # Create price series with some mean reversion
        returns = []
        prices = [base_price]

        for i in range(1, days):
            # Add mean reversion component
            price_deviation = (prices[-1] - base_price) / base_price
            mean_revert = -0.1 * price_deviation  # Mean reversion strength

            # Add random component
            random_return = np.random.normal(0, 0.02)

            # Combine components
            total_return = mean_revert + random_return
            new_price = prices[-1] * (1 + total_return)
            prices.append(new_price)
            returns.append(total_return)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "close": prices,
                "volume": np.random.normal(1000000, 200000, len(timestamps)),
            },
            index=timestamps,
        )

        return df

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive statistical arbitrage analysis"""
        logger.info("🚀 Starting comprehensive statistical arbitrage analysis...")

        # Get historical data for multiple assets
        assets = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        price_data = {}

        for asset in assets:
            price_data[asset] = self.get_historical_data(
                asset, self.config.formation_period
            )

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": f"{self.config.formation_period} days",
            "assets_analyzed": assets,
            "pairs_trading": {},
            "cross_asset_arbitrage": {},
            "strategy_recommendations": [],
            "risk_metrics": {},
        }

        # 1. Pairs Trading Analysis
        logger.info("🔗 Analyzing pairs trading opportunities...")
        pairs = self.pairs_strategy.find_pairs(price_data)

        if pairs:
            analysis["pairs_trading"]["viable_pairs"] = len(pairs)
            analysis["pairs_trading"]["top_pairs"] = []

            # Generate signals for top pairs
            for pair in pairs[:5]:  # Top 5 pairs
                signals = self.pairs_strategy.generate_signals(pair, price_data)
                pair_info = {
                    "pair": f"{pair['asset1']}-{pair['asset2']}",
                    "correlation": pair["correlation"],
                    "formation_score": pair["formation_score"],
                    "cointegration_p_value": pair["cointegration"]["p_value"],
                    "half_life": pair["cointegration"]["half_life"],
                    "current_signal": signals,
                }
                analysis["pairs_trading"]["top_pairs"].append(pair_info)

                # Add to recommendations
                if signals["signal"] in ["LONG_SPREAD", "SHORT_SPREAD"]:
                    analysis["strategy_recommendations"].append(
                        {
                            "type": "pairs_trading",
                            "pair": pair_info["pair"],
                            "signal": signals["signal"],
                            "confidence": signals["confidence"],
                            "reasoning": " | ".join(signals.get("reasoning", [])),
                        }
                    )

        # 2. Cross-Asset Arbitrage Analysis
        logger.info("⚡ Analyzing cross-asset arbitrage opportunities...")
        arbitrage_opportunities = self.cross_asset.find_arbitrage_opportunities(
            price_data
        )

        analysis["cross_asset_arbitrage"] = {
            "total_opportunities": len(arbitrage_opportunities),
            "momentum_opportunities": len(
                [
                    op
                    for op in arbitrage_opportunities
                    if op["type"] == "momentum_reversal"
                ]
            ),
            "volatility_opportunities": len(
                [
                    op
                    for op in arbitrage_opportunities
                    if op["type"] in ["volatility_clustering", "volatility_expansion"]
                ]
            ),
            "cross_exchange_opportunities": len(
                [op for op in arbitrage_opportunities if op["type"] == "cross_exchange"]
            ),
            "opportunities": arbitrage_opportunities[:10],  # Top 10
        }

        # Add arbitrage recommendations
        for opp in arbitrage_opportunities[:5]:  # Top 5 opportunities
            analysis["strategy_recommendations"].append(
                {
                    "type": opp["type"],
                    "asset": opp.get("asset", "N/A"),
                    "signal": opp.get("signal", "HOLD"),
                    "confidence": opp.get("confidence", 0.0),
                    "reasoning": opp.get("reasoning", "No reasoning provided"),
                }
            )

        # 3. Risk Metrics
        analysis["risk_metrics"] = self._calculate_strategy_risk_metrics(
            price_data, pairs, arbitrage_opportunities
        )

        # Store results
        self.store_analysis(analysis)

        logger.info("✅ Statistical arbitrage analysis completed")
        return analysis

    def _calculate_strategy_risk_metrics(
        self,
        price_data: Dict[str, pd.DataFrame],
        pairs: List[Dict[str, Any]],
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate risk metrics for arbitrage strategies"""
        risk_metrics = {}

        # Portfolio correlation analysis
        correlations = []
        for i, asset1 in enumerate(price_data.keys()):
            for j, asset2 in enumerate(list(price_data.keys())[i + 1 :], i + 1):
                df1 = price_data[asset1]
                df2 = price_data[list(price_data.keys())[j]]

                if len(df1) > 30 and len(df2) > 30:
                    returns1 = df1["close"].pct_change().dropna()
                    returns2 = df2["close"].pct_change().dropna()

                    common_dates = returns1.index.intersection(returns2.index)
                    if len(common_dates) > 20:
                        corr = returns1.loc[common_dates].corr(
                            returns2.loc[common_dates]
                        )
                        correlations.append(corr)

        risk_metrics["average_correlation"] = (
            float(np.mean(correlations)) if correlations else 0.0
        )
        risk_metrics["max_correlation"] = (
            float(np.max(correlations)) if correlations else 0.0
        )

        # Pairs trading risk
        if pairs:
            half_lives = [
                pair["cointegration"]["half_life"]
                for pair in pairs
                if pair["cointegration"]["half_life"] != float("inf")
            ]
            risk_metrics["pairs_avg_half_life"] = (
                float(np.mean(half_lives)) if half_lives else 0.0
            )
            risk_metrics["pairs_risk_score"] = min(
                len(pairs) / 10.0, 1.0
            )  # More pairs = lower risk

        # Arbitrage opportunity risk
        high_confidence_opps = [op for op in opportunities if op["confidence"] > 0.7]
        risk_metrics["arbitrage_opportunity_count"] = len(opportunities)
        risk_metrics["high_confidence_opportunities"] = len(high_confidence_opps)

        return risk_metrics

    def store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis results in Redis"""
        if not self.redis_client:
            return

        try:
            # Store main analysis
            self.redis_client.setex(
                "statistical_arbitrage_analysis",
                3600,  # 1 hour expiry
                json.dumps(analysis, default=str),
            )

            # Store pairs trading signals
            if "pairs_trading" in analysis and "top_pairs" in analysis["pairs_trading"]:
                self.redis_client.setex(
                    "pairs_trading_signals",
                    1800,  # 30 minutes expiry
                    json.dumps(analysis["pairs_trading"]["top_pairs"], default=str),
                )

            # Store arbitrage opportunities
            if "cross_asset_arbitrage" in analysis:
                self.redis_client.setex(
                    "arbitrage_opportunities",
                    900,  # 15 minutes expiry
                    json.dumps(
                        analysis["cross_asset_arbitrage"]["opportunities"], default=str
                    ),
                )

            logger.info("💾 Statistical arbitrage analysis stored in Redis")

        except Exception as e:
            logger.error(f"Error storing statistical arbitrage analysis: {e}")

    def get_stored_analysis(self) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis from Redis"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get("statistical_arbitrage_analysis")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving statistical arbitrage analysis: {e}")

        return None


def main():
    """Demo function for statistical arbitrage"""
    print("🚀 Initializing Statistical Arbitrage Engine")
    print("=" * 80)

    # Initialize engine
    config = StatArbConfig(
        lookback_days=60,
        formation_period=252,
        min_correlation=0.7,
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    engine = StatisticalArbitrageEngine(config)

    # Run comprehensive analysis
    analysis = engine.run_comprehensive_analysis()

    if analysis:
        print("✅ Statistical Arbitrage Analysis Results:")

        # Pairs trading results
        pairs_data = analysis["pairs_trading"]
        print(f"📊 Pairs Trading:")
        print(f"   Viable pairs found: {pairs_data.get('viable_pairs', 0)}")

        if "top_pairs" in pairs_data:
            print(f"   Top pairs:")
            for pair_info in pairs_data["top_pairs"][:3]:
                print(
                    f"     • {pair_info['pair']}: correlation={pair_info['correlation']:.3f}, "
                    f"signal={pair_info['current_signal']['signal']}"
                )

        # Cross-asset arbitrage results
        arb_data = analysis["cross_asset_arbitrage"]
        print(f"\n⚡ Cross-Asset Arbitrage:")
        print(f"   Total opportunities: {arb_data['total_opportunities']}")
        print(f"   Momentum opportunities: {arb_data['momentum_opportunities']}")
        print(f"   Volatility opportunities: {arb_data['volatility_opportunities']}")
        print(
            f"   Cross-exchange opportunities: {arb_data['cross_exchange_opportunities']}"
        )

        # Strategy recommendations
        recommendations = analysis["strategy_recommendations"]
        if recommendations:
            print(f"\n💡 Strategy Recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                print(
                    f"   {i+1}. {rec['type']}: {rec.get('pair', rec.get('asset', ''))} - "
                    f"{rec['signal']} (confidence: {rec['confidence']:.2f})"
                )

        # Risk metrics
        risk_metrics = analysis["risk_metrics"]
        print(f"\n📊 Risk Metrics:")
        print(
            f"   Average correlation: {risk_metrics.get('average_correlation', 0):.3f}"
        )
        print(
            f"   High confidence opportunities: {risk_metrics.get('high_confidence_opportunities', 0)}"
        )

    print("\n🎉 Statistical Arbitrage Analysis Complete!")


if __name__ == "__main__":
    main()
