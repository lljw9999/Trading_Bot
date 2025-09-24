#!/usr/bin/env python3
"""
Expected Shortfall (ES) and Extreme Value Theory (EVT) Risk Measures
Advanced risk management using conditional value at risk and extreme value distributions
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
from scipy.stats import genpareto, gumbel_r, genextreme
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management models"""

    confidence_levels: List[float] = None
    lookback_days: int = 252  # 1 year of daily data
    evt_threshold_percentile: float = 95.0  # Use top 5% for EVT
    min_exceedances: int = 20  # Minimum number of threshold exceedances
    block_size: int = 21  # Block size for block maxima (3 weeks)
    simulation_runs: int = 10000  # Monte Carlo simulations

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99, 0.999]


class ExpectedShortfallCalculator:
    """Calculate Expected Shortfall (Conditional Value at Risk)"""

    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_historical_es(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate Historical Expected Shortfall"""
        if len(returns) == 0:
            return {"es": 0.0, "var": 0.0, "num_tail_observations": 0}

        # Calculate VaR at confidence level
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)

        # Expected Shortfall: mean of losses beyond VaR
        tail_losses = returns[returns <= var]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var

        return {
            "es": float(es),
            "var": float(var),
            "num_tail_observations": len(tail_losses),
            "confidence_level": confidence_level,
        }

    def calculate_parametric_es(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        distribution: str = "normal",
    ) -> Dict[str, float]:
        """Calculate Parametric Expected Shortfall"""
        if len(returns) == 0:
            return {"es": 0.0, "var": 0.0, "distribution": distribution}

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if distribution == "normal":
            # Normal distribution ES
            var_quantile = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
            # ES for normal distribution: μ + σ * φ(Φ^(-1)(α)) / α
            z_alpha = stats.norm.ppf(1 - confidence_level)
            es = mean_return + std_return * stats.norm.pdf(z_alpha) / (
                1 - confidence_level
            )

        elif distribution == "t":
            # Student's t-distribution ES
            # Fit t-distribution
            df_fit, loc_fit, scale_fit = stats.t.fit(returns)
            var_quantile = stats.t.ppf(1 - confidence_level, df_fit, loc_fit, scale_fit)

            # ES for t-distribution (approximation)
            t_alpha = stats.t.ppf(1 - confidence_level, df_fit)
            es_factor = stats.t.pdf(t_alpha, df_fit) / (1 - confidence_level)
            es = loc_fit + scale_fit * (df_fit + t_alpha**2) / (df_fit - 1) * es_factor

        else:
            # Fallback to historical
            return self.calculate_historical_es(returns, confidence_level)

        return {
            "es": float(es),
            "var": float(var_quantile),
            "distribution": distribution,
            "confidence_level": confidence_level,
            "parameters": {"mean": float(mean_return), "std": float(std_return)},
        }

    def calculate_cornish_fisher_es(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate Expected Shortfall using Cornish-Fisher expansion"""
        if len(returns) == 0:
            return {"es": 0.0, "var": 0.0, "method": "cornish_fisher"}

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        # Cornish-Fisher VaR adjustment
        z_alpha = stats.norm.ppf(1 - confidence_level)
        cf_adjustment = (
            (1 / 6) * (z_alpha**2 - 1) * skewness
            + (1 / 24) * (z_alpha**3 - 3 * z_alpha) * kurtosis
            - (1 / 36) * (2 * z_alpha**3 - 5 * z_alpha) * skewness**2
        )

        cf_quantile = z_alpha + cf_adjustment
        var_cf = mean_return + std_return * cf_quantile

        # Approximate ES using modified quantile
        # ES ≈ μ + σ * (z_α + CF_adjustment + φ(z_α)/α)
        es_cf = mean_return + std_return * (
            cf_quantile + stats.norm.pdf(z_alpha) / (1 - confidence_level)
        )

        return {
            "es": float(es_cf),
            "var": float(var_cf),
            "method": "cornish_fisher",
            "confidence_level": confidence_level,
            "moments": {
                "mean": float(mean_return),
                "std": float(std_return),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
            },
        }


class ExtremeValueTheory:
    """Extreme Value Theory implementation for tail risk modeling"""

    def __init__(self, config: RiskConfig):
        self.config = config

    def fit_peaks_over_threshold(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit Generalized Pareto Distribution using Peaks Over Threshold method"""
        if len(returns) == 0:
            return {"method": "POT", "fitted": False}

        # Calculate threshold (e.g., 95th percentile)
        threshold = np.percentile(returns, self.config.evt_threshold_percentile)

        # Extract exceedances (losses beyond threshold)
        exceedances = returns[returns > threshold] - threshold

        if len(exceedances) < self.config.min_exceedances:
            logger.warning(
                f"Insufficient exceedances: {len(exceedances)} < {self.config.min_exceedances}"
            )
            return {
                "method": "POT",
                "fitted": False,
                "reason": "insufficient_exceedances",
            }

        try:
            # Fit Generalized Pareto Distribution
            shape, loc, scale = genpareto.fit(exceedances, floc=0)  # loc=0 for POT

            # Calculate goodness of fit
            ks_stat, p_value = stats.kstest(
                exceedances, lambda x: genpareto.cdf(x, shape, loc, scale)
            )

            return {
                "method": "POT",
                "fitted": True,
                "threshold": float(threshold),
                "num_exceedances": len(exceedances),
                "parameters": {
                    "shape": float(shape),
                    "location": float(loc),
                    "scale": float(scale),
                },
                "goodness_of_fit": {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                },
                "exceedance_rate": len(exceedances) / len(returns),
            }

        except Exception as e:
            logger.error(f"Failed to fit GPD: {e}")
            return {"method": "POT", "fitted": False, "error": str(e)}

    def fit_block_maxima(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit Generalized Extreme Value Distribution using Block Maxima method"""
        if len(returns) < self.config.block_size:
            return {"method": "BM", "fitted": False, "reason": "insufficient_data"}

        # Create blocks and extract maxima
        num_blocks = len(returns) // self.config.block_size
        block_maxima = []

        for i in range(num_blocks):
            start_idx = i * self.config.block_size
            end_idx = (i + 1) * self.config.block_size
            block_max = np.max(returns[start_idx:end_idx])
            block_maxima.append(block_max)

        block_maxima = np.array(block_maxima)

        if len(block_maxima) < 10:  # Need sufficient blocks
            return {"method": "BM", "fitted": False, "reason": "insufficient_blocks"}

        try:
            # Fit Generalized Extreme Value Distribution
            shape, loc, scale = genextreme.fit(block_maxima)

            # Calculate goodness of fit
            ks_stat, p_value = stats.kstest(
                block_maxima, lambda x: genextreme.cdf(x, shape, loc, scale)
            )

            return {
                "method": "BM",
                "fitted": True,
                "block_size": self.config.block_size,
                "num_blocks": len(block_maxima),
                "parameters": {
                    "shape": float(shape),
                    "location": float(loc),
                    "scale": float(scale),
                },
                "goodness_of_fit": {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                },
            }

        except Exception as e:
            logger.error(f"Failed to fit GEV: {e}")
            return {"method": "BM", "fitted": False, "error": str(e)}

    def calculate_evt_var_es(
        self,
        pot_fit: Dict[str, Any],
        confidence_level: float = 0.95,
        num_observations: int = 252,
    ) -> Dict[str, float]:
        """Calculate VaR and ES using fitted EVT model"""
        if not pot_fit.get("fitted", False):
            return {"var": 0.0, "es": 0.0, "method": "EVT"}

        params = pot_fit["parameters"]
        threshold = pot_fit["threshold"]
        exceedance_rate = pot_fit["exceedance_rate"]

        # Calculate VaR using POT model
        # P(X > x) = P(X > u) * P(X > x | X > u) where u is threshold
        target_prob = 1 - confidence_level
        conditional_prob = target_prob / exceedance_rate

        if conditional_prob > 1:
            # Not enough extreme events for this confidence level
            return {
                "var": threshold,
                "es": threshold,
                "method": "EVT",
                "warning": "extrapolation",
            }

        # VaR from GPD
        shape = params["shape"]
        scale = params["scale"]

        if abs(shape) < 1e-6:  # Exponential case (shape ≈ 0)
            var_exceedance = -scale * np.log(conditional_prob)
        else:
            var_exceedance = (scale / shape) * (conditional_prob ** (-shape) - 1)

        var_evt = threshold + var_exceedance

        # Expected Shortfall from GPD
        if shape < 1:
            es_exceedance = (var_exceedance + scale - shape * threshold) / (1 - shape)
            es_evt = threshold + es_exceedance
        else:
            # For shape >= 1, ES may not exist; use approximation
            es_evt = var_evt * 1.5  # Conservative approximation

        return {
            "var": float(var_evt),
            "es": float(es_evt),
            "method": "EVT",
            "confidence_level": confidence_level,
            "threshold": threshold,
            "extrapolation_ratio": conditional_prob,
        }


class ComprehensiveRiskManager:
    """Comprehensive risk management combining ES and EVT"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.es_calculator = ExpectedShortfallCalculator(self.config)
        self.evt_analyzer = ExtremeValueTheory(self.config)

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for risk management")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate historical price data for risk analysis"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Generate realistic price data with fat tails
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
        }

        base_price = base_prices.get(symbol, 100)

        # Generate returns with fat tails (t-distribution)
        t_returns = np.random.standard_t(df=4, size=days) * 0.03  # 3% daily volatility
        normal_returns = np.random.normal(0, 0.02, days)  # Background normal returns

        # Mix distributions to create realistic crypto returns
        mixed_returns = 0.7 * normal_returns + 0.3 * t_returns

        # Add some trend and volatility clustering
        trend = np.cumsum(np.random.normal(0, 0.001, days))
        volatility = np.abs(np.random.normal(0.02, 0.01, days))

        returns = mixed_returns + trend + volatility * np.random.normal(0, 1, days)

        # Generate prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create DataFrame
        timestamps = pd.date_range(start=start_time, end=end_time, periods=days)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "returns": [0] + list(np.diff(np.log(prices))),  # Log returns
            }
        )

        return df

    def analyze_comprehensive_risk(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive risk analysis combining ES and EVT"""
        logger.info(f"📊 Starting comprehensive risk analysis for {symbol}")

        # Get historical data
        df = self.get_historical_data(symbol, self.config.lookback_days)
        returns = df["returns"].values[1:]  # Remove first NaN

        if len(returns) == 0:
            return {"error": "No returns data available"}

        # Convert to losses (negative returns)
        losses = -returns

        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data_period": f"{self.config.lookback_days} days",
            "analysis_methods": [
                "Historical ES",
                "Parametric ES",
                "Cornish-Fisher ES",
                "EVT",
            ],
            "expected_shortfall": {},
            "extreme_value_theory": {},
            "risk_metrics": {},
            "backtesting": {},
            "recommendations": [],
        }

        # 1. Expected Shortfall Analysis
        logger.info("📈 Calculating Expected Shortfall measures...")
        es_results = {}

        for confidence_level in self.config.confidence_levels:
            # Historical ES
            hist_es = self.es_calculator.calculate_historical_es(
                losses, confidence_level
            )

            # Parametric ES (Normal)
            param_normal_es = self.es_calculator.calculate_parametric_es(
                losses, confidence_level, "normal"
            )

            # Parametric ES (Student-t)
            param_t_es = self.es_calculator.calculate_parametric_es(
                losses, confidence_level, "t"
            )

            # Cornish-Fisher ES
            cf_es = self.es_calculator.calculate_cornish_fisher_es(
                losses, confidence_level
            )

            es_results[f"{confidence_level:.1%}"] = {
                "historical": hist_es,
                "parametric_normal": param_normal_es,
                "parametric_t": param_t_es,
                "cornish_fisher": cf_es,
            }

        analysis["expected_shortfall"] = es_results

        # 2. Extreme Value Theory Analysis
        logger.info("🎯 Performing Extreme Value Theory analysis...")

        # Peaks Over Threshold
        pot_fit = self.evt_analyzer.fit_peaks_over_threshold(losses)

        # Block Maxima
        bm_fit = self.evt_analyzer.fit_block_maxima(losses)

        # Calculate EVT-based VaR and ES
        evt_results = {}
        if pot_fit.get("fitted", False):
            for confidence_level in self.config.confidence_levels:
                evt_risk = self.evt_analyzer.calculate_evt_var_es(
                    pot_fit, confidence_level, len(losses)
                )
                evt_results[f"{confidence_level:.1%}"] = evt_risk

        analysis["extreme_value_theory"] = {
            "peaks_over_threshold": pot_fit,
            "block_maxima": bm_fit,
            "evt_risk_measures": evt_results,
        }

        # 3. Risk Metrics Summary
        current_price = df["close"].iloc[-1]
        daily_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility

        # Get 95% confidence level results for summary
        cl_95 = "95.0%"
        hist_95 = es_results.get(cl_95, {}).get("historical", {})
        evt_95 = evt_results.get(cl_95, {})

        analysis["risk_metrics"] = {
            "current_price": float(current_price),
            "daily_volatility": float(np.std(returns)),
            "annualized_volatility": float(daily_vol),
            "max_drawdown": float(np.min(returns)),
            "var_95_historical": float(hist_95.get("var", 0)) * current_price,
            "es_95_historical": float(hist_95.get("es", 0)) * current_price,
            "var_95_evt": (
                float(evt_95.get("var", 0)) * current_price if evt_95 else None
            ),
            "es_95_evt": float(evt_95.get("es", 0)) * current_price if evt_95 else None,
            "tail_risk_ratio": (
                float(hist_95.get("es", 0) / hist_95.get("var", 1))
                if hist_95.get("var", 0) != 0
                else 1.0
            ),
        }

        # 4. Risk-based Recommendations
        recommendations = []

        # Volatility-based recommendations
        if daily_vol > 0.8:  # High volatility
            recommendations.append(
                "⚠️ High volatility detected - consider reducing position size"
            )
        elif daily_vol < 0.3:
            recommendations.append(
                "✅ Low volatility environment - potential for position increase"
            )

        # ES vs VaR comparison
        tail_ratio = analysis["risk_metrics"]["tail_risk_ratio"]
        if tail_ratio > 1.5:
            recommendations.append(
                "📊 Significant tail risk - Expected Shortfall much higher than VaR"
            )

        # EVT-based recommendations
        if pot_fit.get("fitted", False):
            shape_param = pot_fit["parameters"]["shape"]
            if shape_param > 0.1:
                recommendations.append(
                    "🎯 Heavy-tailed distribution detected - extreme losses possible"
                )
            elif shape_param < -0.1:
                recommendations.append("📈 Light-tailed distribution - losses bounded")

        analysis["recommendations"] = recommendations

        # Store results
        self.store_risk_analysis(analysis)

        logger.info("✅ Comprehensive risk analysis completed")
        return analysis

    def store_risk_analysis(self, analysis: Dict[str, Any]):
        """Store risk analysis results in Redis"""
        if not self.redis_client:
            return

        try:
            # Store main analysis
            self.redis_client.setex(
                f"risk_analysis_{analysis['symbol']}",
                3600,  # 1 hour expiry
                json.dumps(analysis, default=str),
            )

            # Store risk metrics summary
            self.redis_client.setex(
                f"risk_metrics_{analysis['symbol']}",
                1800,  # 30 minutes expiry
                json.dumps(analysis["risk_metrics"], default=str),
            )

            logger.info("💾 Risk analysis stored in Redis")

        except Exception as e:
            logger.error(f"Error storing risk analysis: {e}")

    def get_stored_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored risk analysis from Redis"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"risk_analysis_{symbol}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving risk analysis: {e}")

        return None

    def calculate_portfolio_risk(
        self, portfolio_symbols: List[str], weights: List[float]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level risk measures"""
        if len(portfolio_symbols) != len(weights) or abs(sum(weights) - 1.0) > 1e-6:
            return {"error": "Invalid portfolio weights"}

        portfolio_analysis = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_composition": dict(zip(portfolio_symbols, weights)),
            "individual_risks": {},
            "portfolio_risk": {},
            "diversification_benefit": {},
        }

        # Get individual asset risks
        for symbol in portfolio_symbols:
            analysis = self.get_stored_analysis(symbol)
            if not analysis:
                analysis = self.analyze_comprehensive_risk(symbol)

            portfolio_analysis["individual_risks"][symbol] = analysis.get(
                "risk_metrics", {}
            )

        # Calculate portfolio risk (simplified approach)
        individual_vars = []
        individual_ess = []

        for symbol, weight in zip(portfolio_symbols, weights):
            metrics = portfolio_analysis["individual_risks"].get(symbol, {})
            var_95 = metrics.get("var_95_historical", 0)
            es_95 = metrics.get("es_95_historical", 0)

            individual_vars.append(weight * var_95)
            individual_ess.append(weight * es_95)

        # Portfolio risk (assuming independence - conservative)
        portfolio_var = np.sqrt(np.sum(np.array(individual_vars) ** 2))
        portfolio_es = np.sqrt(np.sum(np.array(individual_ess) ** 2))

        # Diversification benefit
        sum_individual_var = sum(individual_vars)
        sum_individual_es = sum(individual_ess)

        portfolio_analysis["portfolio_risk"] = {
            "portfolio_var_95": float(portfolio_var),
            "portfolio_es_95": float(portfolio_es),
            "sum_individual_var": float(sum_individual_var),
            "sum_individual_es": float(sum_individual_es),
        }

        portfolio_analysis["diversification_benefit"] = {
            "var_reduction": (
                float((sum_individual_var - portfolio_var) / sum_individual_var)
                if sum_individual_var > 0
                else 0
            ),
            "es_reduction": (
                float((sum_individual_es - portfolio_es) / sum_individual_es)
                if sum_individual_es > 0
                else 0
            ),
        }

        return portfolio_analysis


def main():
    """Demo function for risk management"""
    print("🚀 Initializing Expected Shortfall & Extreme Value Theory Risk Management")
    print("=" * 80)

    # Initialize risk manager
    config = RiskConfig(
        confidence_levels=[0.90, 0.95, 0.99, 0.999],
        lookback_days=252,
        evt_threshold_percentile=95.0,
    )

    risk_manager = ComprehensiveRiskManager(config)

    # Analyze risk for major crypto assets
    assets = ["BTCUSDT", "ETHUSDT"]

    for asset in assets:
        print(f"\n📊 Analyzing {asset}...")
        analysis = risk_manager.analyze_comprehensive_risk(asset)

        if "error" not in analysis:
            metrics = analysis["risk_metrics"]
            print(f"✅ Risk Analysis for {asset}:")
            print(f"   Current Price: ${metrics['current_price']:.2f}")
            print(f"   Daily Volatility: {metrics['daily_volatility']:.2%}")
            print(f"   VaR (95%): ${metrics['var_95_historical']:.2f}")
            print(f"   ES (95%): ${metrics['es_95_historical']:.2f}")
            print(f"   Tail Risk Ratio: {metrics['tail_risk_ratio']:.2f}")

            # Show EVT results if available
            if metrics.get("var_95_evt"):
                print(f"   EVT VaR (95%): ${metrics['var_95_evt']:.2f}")
                print(f"   EVT ES (95%): ${metrics['es_95_evt']:.2f}")

            # Show recommendations
            if analysis["recommendations"]:
                print(f"   Recommendations:")
                for rec in analysis["recommendations"][:2]:
                    print(f"     • {rec}")

    # Portfolio risk analysis
    print(f"\n💼 Portfolio Risk Analysis...")
    portfolio_risk = risk_manager.calculate_portfolio_risk(
        ["BTCUSDT", "ETHUSDT"], [0.6, 0.4]
    )

    if "error" not in portfolio_risk:
        print(f"✅ Portfolio Risk Results:")
        portfolio_metrics = portfolio_risk["portfolio_risk"]
        diversification = portfolio_risk["diversification_benefit"]

        print(f"   Portfolio VaR (95%): ${portfolio_metrics['portfolio_var_95']:.2f}")
        print(f"   Portfolio ES (95%): ${portfolio_metrics['portfolio_es_95']:.2f}")
        print(f"   VaR Diversification Benefit: {diversification['var_reduction']:.1%}")
        print(f"   ES Diversification Benefit: {diversification['es_reduction']:.1%}")

    print("\n🎉 Risk Management System Complete!")


if __name__ == "__main__":
    main()
