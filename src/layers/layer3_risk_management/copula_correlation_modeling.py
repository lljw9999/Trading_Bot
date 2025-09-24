#!/usr/bin/env python3
"""
Copula-based Correlation Modeling for Advanced Dependency Analysis
Implements various copula models to capture complex dependencies between crypto assets
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gamma
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
class CopulaConfig:
    """Configuration for copula modeling"""

    copula_types: List[str] = None
    lookback_days: int = 252  # 1 year of daily data
    confidence_levels: List[float] = None
    simulation_size: int = 10000  # Monte Carlo simulations
    optimization_method: str = "SLSQP"
    convergence_tolerance: float = 1e-6

    def __post_init__(self):
        if self.copula_types is None:
            self.copula_types = ["gaussian", "student_t", "clayton", "gumbel", "frank"]
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


class CopulaModels:
    """Implementation of various copula models"""

    @staticmethod
    def gaussian_copula_cdf(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
        """Gaussian copula CDF"""
        if abs(rho) >= 1:
            rho = np.sign(rho) * 0.999

        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)

        # Bivariate normal CDF
        rho2 = rho**2
        z = (x**2 - 2 * rho * x * y + y**2) / (2 * (1 - rho2))

        return np.exp(-z) / (2 * np.pi * np.sqrt(1 - rho2))

    @staticmethod
    def gaussian_copula_pdf(u: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
        """Gaussian copula PDF"""
        if abs(rho) >= 1:
            rho = np.sign(rho) * 0.999

        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)

        rho2 = rho**2
        numerator = -rho * (x**2 + y**2) + 2 * rho * x * y
        denominator = 2 * (1 - rho2)

        return (1 / np.sqrt(1 - rho2)) * np.exp(numerator / denominator)

    @staticmethod
    def student_t_copula_pdf(
        u: np.ndarray, v: np.ndarray, rho: float, nu: float
    ) -> np.ndarray:
        """Student-t copula PDF"""
        if abs(rho) >= 1:
            rho = np.sign(rho) * 0.999
        if nu <= 2:
            nu = 2.1

        x = stats.t.ppf(u, nu)
        y = stats.t.ppf(v, nu)

        # Bivariate t-distribution density
        rho2 = rho**2
        z = (x**2 - 2 * rho * x * y + y**2) / (1 - rho2)

        # Normalization constants
        gamma_term = gamma((nu + 2) / 2) / (gamma(nu / 2) * gamma(1))
        pi_term = 1 / (np.pi * np.sqrt(1 - rho2))

        # T-copula density
        numerator = gamma_term * pi_term
        denominator = (1 + z / nu) ** ((nu + 2) / 2)

        # Marginal t-densities
        marginal_x = stats.t.pdf(x, nu)
        marginal_y = stats.t.pdf(y, nu)

        return (numerator / denominator) / (marginal_x * marginal_y)

    @staticmethod
    def clayton_copula_pdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Clayton copula PDF"""
        if theta <= 0:
            theta = 0.01

        term1 = 1 + theta
        term2 = (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 2)
        term3 = u ** (-theta - 1) * v ** (-theta - 1)

        return term1 * term2 * term3

    @staticmethod
    def gumbel_copula_pdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Gumbel copula PDF"""
        if theta < 1:
            theta = 1.01

        # Gumbel copula CDF
        ln_u = -np.log(u)
        ln_v = -np.log(v)

        A = (ln_u**theta + ln_v**theta) ** (1 / theta)
        C = np.exp(-A)

        # Gumbel copula PDF
        term1 = C / (u * v)
        term2 = (A + theta - 1) / A ** (2 - 1 / theta)
        term3 = (ln_u * ln_v) ** (theta - 1)
        term4 = (ln_u**theta + ln_v**theta) ** (1 / theta - 2)

        return term1 * term2 * term3 * term4

    @staticmethod
    def frank_copula_pdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Frank copula PDF"""
        if abs(theta) < 1e-6:
            return np.ones_like(u)  # Independence case

        exp_theta = np.exp(theta)
        exp_theta_u = np.exp(theta * u)
        exp_theta_v = np.exp(theta * v)

        numerator = theta * exp_theta * (exp_theta - 1)
        denominator_part1 = (exp_theta - 1) + (exp_theta_u - 1) * (exp_theta_v - 1)
        denominator = denominator_part1**2

        return numerator / denominator


class CopulaFitter:
    """Fit various copula models to data"""

    def __init__(self, config: CopulaConfig):
        self.config = config
        self.models = CopulaModels()

    def empirical_cdf(self, data: np.ndarray) -> np.ndarray:
        """Convert data to uniform marginals using empirical CDF"""
        n = len(data)
        ranks = stats.rankdata(data, method="average")
        return ranks / (n + 1)

    def fit_gaussian_copula(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit Gaussian copula"""
        try:
            # Transform to normal scores
            x = stats.norm.ppf(u)
            y = stats.norm.ppf(v)

            # Calculate correlation
            rho = np.corrcoef(x, y)[0, 1]

            # Log-likelihood
            def neg_log_likelihood(params):
                rho_param = params[0]
                if abs(rho_param) >= 1:
                    return 1e10

                try:
                    pdf_vals = self.models.gaussian_copula_pdf(u, v, rho_param)
                    pdf_vals = np.clip(pdf_vals, 1e-10, None)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                [rho],
                bounds=[(-0.999, 0.999)],
                method=self.config.optimization_method,
            )

            optimal_rho = result.x[0]
            aic = 2 * 1 + 2 * result.fun  # 1 parameter
            bic = np.log(len(u)) * 1 + 2 * result.fun

            return {
                "type": "gaussian",
                "parameters": {"rho": float(optimal_rho)},
                "log_likelihood": float(-result.fun),
                "aic": float(aic),
                "bic": float(bic),
                "fitted": result.success,
            }
        except Exception as e:
            logger.error(f"Error fitting Gaussian copula: {e}")
            return {"type": "gaussian", "fitted": False, "error": str(e)}

    def fit_student_t_copula(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit Student-t copula"""
        try:
            # Initial estimates
            x = stats.norm.ppf(u)
            y = stats.norm.ppf(v)
            rho_init = np.corrcoef(x, y)[0, 1]

            def neg_log_likelihood(params):
                rho_param, nu_param = params
                if abs(rho_param) >= 1 or nu_param <= 2:
                    return 1e10

                try:
                    pdf_vals = self.models.student_t_copula_pdf(
                        u, v, rho_param, nu_param
                    )
                    pdf_vals = np.clip(pdf_vals, 1e-10, None)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                [rho_init, 5.0],
                bounds=[(-0.999, 0.999), (2.1, 30)],
                method=self.config.optimization_method,
            )

            optimal_rho, optimal_nu = result.x
            aic = 2 * 2 + 2 * result.fun  # 2 parameters
            bic = np.log(len(u)) * 2 + 2 * result.fun

            return {
                "type": "student_t",
                "parameters": {"rho": float(optimal_rho), "nu": float(optimal_nu)},
                "log_likelihood": float(-result.fun),
                "aic": float(aic),
                "bic": float(bic),
                "fitted": result.success,
            }
        except Exception as e:
            logger.error(f"Error fitting Student-t copula: {e}")
            return {"type": "student_t", "fitted": False, "error": str(e)}

    def fit_clayton_copula(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit Clayton copula"""
        try:

            def neg_log_likelihood(params):
                theta = params[0]
                if theta <= 0:
                    return 1e10

                try:
                    pdf_vals = self.models.clayton_copula_pdf(u, v, theta)
                    pdf_vals = np.clip(pdf_vals, 1e-10, None)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                [1.0],
                bounds=[(0.01, 20)],
                method=self.config.optimization_method,
            )

            optimal_theta = result.x[0]
            aic = 2 * 1 + 2 * result.fun  # 1 parameter
            bic = np.log(len(u)) * 1 + 2 * result.fun

            return {
                "type": "clayton",
                "parameters": {"theta": float(optimal_theta)},
                "log_likelihood": float(-result.fun),
                "aic": float(aic),
                "bic": float(bic),
                "fitted": result.success,
            }
        except Exception as e:
            logger.error(f"Error fitting Clayton copula: {e}")
            return {"type": "clayton", "fitted": False, "error": str(e)}

    def fit_gumbel_copula(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit Gumbel copula"""
        try:

            def neg_log_likelihood(params):
                theta = params[0]
                if theta < 1:
                    return 1e10

                try:
                    pdf_vals = self.models.gumbel_copula_pdf(u, v, theta)
                    pdf_vals = np.clip(pdf_vals, 1e-10, None)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                [1.5],
                bounds=[(1.01, 20)],
                method=self.config.optimization_method,
            )

            optimal_theta = result.x[0]
            aic = 2 * 1 + 2 * result.fun  # 1 parameter
            bic = np.log(len(u)) * 1 + 2 * result.fun

            return {
                "type": "gumbel",
                "parameters": {"theta": float(optimal_theta)},
                "log_likelihood": float(-result.fun),
                "aic": float(aic),
                "bic": float(bic),
                "fitted": result.success,
            }
        except Exception as e:
            logger.error(f"Error fitting Gumbel copula: {e}")
            return {"type": "gumbel", "fitted": False, "error": str(e)}

    def fit_frank_copula(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit Frank copula"""
        try:

            def neg_log_likelihood(params):
                theta = params[0]
                if abs(theta) < 1e-6:
                    return 1e10

                try:
                    pdf_vals = self.models.frank_copula_pdf(u, v, theta)
                    pdf_vals = np.clip(pdf_vals, 1e-10, None)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                [1.0],
                bounds=[(-20, 20)],
                method=self.config.optimization_method,
            )

            optimal_theta = result.x[0]
            aic = 2 * 1 + 2 * result.fun  # 1 parameter
            bic = np.log(len(u)) * 1 + 2 * result.fun

            return {
                "type": "frank",
                "parameters": {"theta": float(optimal_theta)},
                "log_likelihood": float(-result.fun),
                "aic": float(aic),
                "bic": float(bic),
                "fitted": result.success,
            }
        except Exception as e:
            logger.error(f"Error fitting Frank copula: {e}")
            return {"type": "frank", "fitted": False, "error": str(e)}

    def fit_all_copulas(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Fit all copula models and select the best one"""
        results = {}

        # Fit each copula type
        fitting_methods = {
            "gaussian": self.fit_gaussian_copula,
            "student_t": self.fit_student_t_copula,
            "clayton": self.fit_clayton_copula,
            "gumbel": self.fit_gumbel_copula,
            "frank": self.fit_frank_copula,
        }

        for copula_type in self.config.copula_types:
            if copula_type in fitting_methods:
                logger.info(f"Fitting {copula_type} copula...")
                results[copula_type] = fitting_methods[copula_type](u, v)

        # Select best copula based on AIC
        fitted_copulas = {k: v for k, v in results.items() if v.get("fitted", False)}

        if fitted_copulas:
            best_copula = min(
                fitted_copulas.keys(), key=lambda k: fitted_copulas[k]["aic"]
            )
            results["best_copula"] = best_copula
            results["model_comparison"] = {
                k: {
                    "aic": v["aic"],
                    "bic": v["bic"],
                    "log_likelihood": v["log_likelihood"],
                }
                for k, v in fitted_copulas.items()
            }
        else:
            results["best_copula"] = None
            results["model_comparison"] = {}

        return results


class CopulaAnalyzer:
    """Main class for copula-based dependency analysis"""

    def __init__(self, config: Optional[CopulaConfig] = None):
        self.config = config or CopulaConfig()
        self.fitter = CopulaFitter(self.config)

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for copula analysis")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate historical return data for copula analysis"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Generate realistic crypto return data
        np.random.seed(hash(symbol) % 2**32)

        # Base parameters for different cryptos
        base_params = {
            "BTCUSDT": {"vol": 0.04, "trend": 0.0002},
            "ETHUSDT": {"vol": 0.05, "trend": 0.0003},
            "ADAUSDT": {"vol": 0.07, "trend": 0.0001},
            "DOTUSDT": {"vol": 0.08, "trend": 0.0001},
        }

        params = base_params.get(symbol, {"vol": 0.06, "trend": 0.0001})

        # Generate returns with fat tails and volatility clustering
        returns = []
        volatility = params["vol"]

        for i in range(days):
            # Volatility clustering
            if i > 0 and abs(returns[-1]) > 2 * volatility:
                volatility *= 1.1  # Increase volatility after large moves
            else:
                volatility = 0.9 * volatility + 0.1 * params["vol"]  # Mean reversion

            # Fat-tailed returns using t-distribution
            if np.random.random() < 0.05:  # 5% chance of extreme move
                ret = np.random.standard_t(df=3) * volatility * 2
            else:
                ret = np.random.normal(params["trend"], volatility)

            returns.append(ret)

        timestamps = pd.date_range(start=start_time, end=end_time, periods=days)
        return pd.DataFrame({"timestamp": timestamps, "returns": returns})

    def analyze_pairwise_dependencies(self, asset1: str, asset2: str) -> Dict[str, Any]:
        """Analyze dependencies between two assets using copulas"""
        logger.info(f"🔗 Analyzing dependencies between {asset1} and {asset2}")

        # Get return data
        data1 = self.get_historical_data(asset1, self.config.lookback_days)
        data2 = self.get_historical_data(asset2, self.config.lookback_days)

        returns1 = data1["returns"].values
        returns2 = data2["returns"].values

        # Transform to uniform marginals
        u = self.fitter.empirical_cdf(returns1)
        v = self.fitter.empirical_cdf(returns2)

        # Fit copula models
        copula_results = self.fitter.fit_all_copulas(u, v)

        # Calculate dependency measures
        dependency_measures = self.calculate_dependency_measures(u, v, copula_results)

        # Monte Carlo simulations
        simulations = self.monte_carlo_simulations(copula_results, 1000)

        analysis = {
            "asset_pair": f"{asset1}-{asset2}",
            "timestamp": datetime.now().isoformat(),
            "data_period": f"{self.config.lookback_days} days",
            "copula_models": copula_results,
            "dependency_measures": dependency_measures,
            "simulations": simulations,
            "recommendations": self.generate_recommendations(
                copula_results, dependency_measures
            ),
        }

        # Store results
        self.store_analysis(analysis)

        logger.info("✅ Copula dependency analysis completed")
        return analysis

    def calculate_dependency_measures(
        self, u: np.ndarray, v: np.ndarray, copula_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate various dependency measures"""
        measures = {}

        # Pearson correlation
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        measures["pearson_correlation"] = float(np.corrcoef(x, y)[0, 1])

        # Spearman rank correlation
        measures["spearman_correlation"] = float(stats.spearmanr(u, v)[0])

        # Kendall's tau
        measures["kendall_tau"] = float(stats.kendalltau(u, v)[0])

        # Tail dependence (approximate)
        # Upper tail dependence
        threshold = 0.95
        upper_tail_u = u > threshold
        upper_tail_v = v > threshold
        upper_tail_dep = np.sum(upper_tail_u & upper_tail_v) / np.sum(upper_tail_u)
        measures["upper_tail_dependence"] = float(upper_tail_dep)

        # Lower tail dependence
        threshold = 0.05
        lower_tail_u = u < threshold
        lower_tail_v = v < threshold
        lower_tail_dep = np.sum(lower_tail_u & lower_tail_v) / np.sum(lower_tail_u)
        measures["lower_tail_dependence"] = float(lower_tail_dep)

        # Mutual information (approximate)
        try:
            from sklearn.feature_selection import mutual_info_regression

            mi = mutual_info_regression(u.reshape(-1, 1), v)[0]
            measures["mutual_information"] = float(mi)
        except:
            measures["mutual_information"] = 0.0

        return measures

    def monte_carlo_simulations(
        self, copula_results: Dict[str, Any], n_sims: int
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulations using fitted copula"""
        simulations = {}

        best_copula = copula_results.get("best_copula")
        if not best_copula or best_copula not in copula_results:
            return {"error": "No fitted copula available for simulation"}

        copula_data = copula_results[best_copula]
        if not copula_data.get("fitted", False):
            return {"error": "Best copula not properly fitted"}

        # Generate random samples based on copula type
        try:
            np.random.seed(42)  # For reproducibility

            if best_copula == "gaussian":
                rho = copula_data["parameters"]["rho"]
                # Generate bivariate normal samples
                samples = np.random.multivariate_normal(
                    [0, 0], [[1, rho], [rho, 1]], n_sims
                )
                u_sim = stats.norm.cdf(samples[:, 0])
                v_sim = stats.norm.cdf(samples[:, 1])

            elif best_copula == "student_t":
                rho = copula_data["parameters"]["rho"]
                nu = copula_data["parameters"]["nu"]
                # Generate bivariate t samples (simplified)
                samples = np.random.multivariate_normal(
                    [0, 0], [[1, rho], [rho, 1]], n_sims
                )
                chi_samples = np.random.chisquare(nu, n_sims)
                samples = samples / np.sqrt(chi_samples[:, None] / nu)
                u_sim = stats.t.cdf(samples[:, 0], nu)
                v_sim = stats.t.cdf(samples[:, 1], nu)

            else:
                # For other copulas, use independence as fallback
                u_sim = np.random.uniform(0, 1, n_sims)
                v_sim = np.random.uniform(0, 1, n_sims)

            # Calculate simulation statistics
            sim_pearson = float(
                np.corrcoef(stats.norm.ppf(u_sim), stats.norm.ppf(v_sim))[0, 1]
            )
            sim_spearman = float(stats.spearmanr(u_sim, v_sim)[0])

            simulations = {
                "copula_type": best_copula,
                "n_simulations": n_sims,
                "simulated_pearson": sim_pearson,
                "simulated_spearman": sim_spearman,
                "simulated_samples": {
                    "u_quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                    "u_values": [
                        float(np.percentile(u_sim, q * 100))
                        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
                    ],
                    "v_quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
                    "v_values": [
                        float(np.percentile(v_sim, q * 100))
                        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
                    ],
                },
            }

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            simulations = {"error": str(e)}

        return simulations

    def generate_recommendations(
        self, copula_results: Dict[str, Any], dependency_measures: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on copula analysis"""
        recommendations = []

        best_copula = copula_results.get("best_copula")
        if not best_copula:
            recommendations.append(
                "⚠️ No suitable copula model found - use caution in dependency modeling"
            )
            return recommendations

        # Correlation-based recommendations
        spearman = dependency_measures.get("spearman_correlation", 0)
        if abs(spearman) > 0.7:
            recommendations.append(
                f"🔗 Strong dependency detected (ρ={spearman:.3f}) - consider diversification"
            )
        elif abs(spearman) < 0.3:
            recommendations.append(
                f"✅ Low correlation (ρ={spearman:.3f}) - good diversification potential"
            )

        # Tail dependence recommendations
        upper_tail = dependency_measures.get("upper_tail_dependence", 0)
        lower_tail = dependency_measures.get("lower_tail_dependence", 0)

        if upper_tail > 0.3:
            recommendations.append(
                "📈 Strong upper tail dependence - assets move together in bull markets"
            )
        if lower_tail > 0.3:
            recommendations.append(
                "📉 Strong lower tail dependence - assets crash together"
            )

        # Copula-specific recommendations
        if best_copula == "clayton":
            recommendations.append("⬇️ Clayton copula: Lower tail dependence dominates")
        elif best_copula == "gumbel":
            recommendations.append("⬆️ Gumbel copula: Upper tail dependence dominates")
        elif best_copula == "student_t":
            recommendations.append(
                "📊 Student-t copula: Symmetric tail dependence detected"
            )
        elif best_copula == "gaussian":
            recommendations.append("📈 Gaussian copula: Linear dependence structure")
        elif best_copula == "frank":
            recommendations.append(
                "🔄 Frank copula: Symmetric dependence across all levels"
            )

        return recommendations

    def analyze_portfolio_dependencies(
        self, asset_symbols: List[str]
    ) -> Dict[str, Any]:
        """Analyze dependencies across multiple assets"""
        portfolio_analysis = {
            "timestamp": datetime.now().isoformat(),
            "assets": asset_symbols,
            "pairwise_analysis": {},
            "portfolio_summary": {},
        }

        # Analyze all pairs
        dependency_matrix = np.zeros((len(asset_symbols), len(asset_symbols)))

        for i, asset1 in enumerate(asset_symbols):
            for j, asset2 in enumerate(asset_symbols):
                if i < j:  # Only analyze upper triangle
                    pair_analysis = self.analyze_pairwise_dependencies(asset1, asset2)
                    pair_key = f"{asset1}-{asset2}"
                    portfolio_analysis["pairwise_analysis"][pair_key] = pair_analysis

                    # Extract correlation for matrix
                    spearman = pair_analysis["dependency_measures"].get(
                        "spearman_correlation", 0
                    )
                    dependency_matrix[i, j] = spearman
                    dependency_matrix[j, i] = spearman
                elif i == j:
                    dependency_matrix[i, j] = 1.0

        # Portfolio summary statistics
        portfolio_analysis["portfolio_summary"] = {
            "average_correlation": float(
                np.mean(dependency_matrix[np.triu_indices_from(dependency_matrix, k=1)])
            ),
            "max_correlation": float(
                np.max(dependency_matrix[np.triu_indices_from(dependency_matrix, k=1)])
            ),
            "min_correlation": float(
                np.min(dependency_matrix[np.triu_indices_from(dependency_matrix, k=1)])
            ),
            "dependency_matrix": dependency_matrix.tolist(),
        }

        return portfolio_analysis

    def store_analysis(self, analysis: Dict[str, Any]):
        """Store copula analysis results in Redis"""
        if not self.redis_client:
            return

        try:
            # Store main analysis
            asset_pair = analysis["asset_pair"]
            self.redis_client.setex(
                f"copula_analysis_{asset_pair}",
                3600,  # 1 hour expiry
                json.dumps(analysis, default=str),
            )

            logger.info("💾 Copula analysis stored in Redis")

        except Exception as e:
            logger.error(f"Error storing copula analysis: {e}")

    def get_stored_analysis(self, asset_pair: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored copula analysis from Redis"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"copula_analysis_{asset_pair}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving copula analysis: {e}")

        return None


def main():
    """Demo function for copula analysis"""
    print("🚀 Initializing Copula-based Correlation Modeling")
    print("=" * 80)

    # Initialize copula analyzer
    config = CopulaConfig(
        copula_types=["gaussian", "student_t", "clayton", "gumbel", "frank"],
        lookback_days=252,
        simulation_size=1000,
    )

    analyzer = CopulaAnalyzer(config)

    # Analyze pairwise dependencies
    print("🔗 Analyzing BTC-ETH dependency structure...")
    btc_eth_analysis = analyzer.analyze_pairwise_dependencies("BTCUSDT", "ETHUSDT")

    if btc_eth_analysis:
        print("✅ BTC-ETH Copula Analysis Results:")

        best_copula = btc_eth_analysis["copula_models"].get("best_copula")
        if best_copula:
            print(f"   Best copula model: {best_copula}")

            model_data = btc_eth_analysis["copula_models"][best_copula]
            print(f"   Parameters: {model_data['parameters']}")
            print(f"   Log-likelihood: {model_data['log_likelihood']:.2f}")
            print(f"   AIC: {model_data['aic']:.2f}")

        # Dependency measures
        deps = btc_eth_analysis["dependency_measures"]
        print(f"   Spearman correlation: {deps['spearman_correlation']:.3f}")
        print(f"   Kendall's tau: {deps['kendall_tau']:.3f}")
        print(f"   Upper tail dependence: {deps['upper_tail_dependence']:.3f}")
        print(f"   Lower tail dependence: {deps['lower_tail_dependence']:.3f}")

        # Recommendations
        if btc_eth_analysis["recommendations"]:
            print("   Recommendations:")
            for rec in btc_eth_analysis["recommendations"][:3]:
                print(f"     • {rec}")

    # Portfolio analysis
    print(f"\n💼 Portfolio dependency analysis...")
    portfolio_assets = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    portfolio_analysis = analyzer.analyze_portfolio_dependencies(portfolio_assets)

    if portfolio_analysis:
        summary = portfolio_analysis["portfolio_summary"]
        print(f"✅ Portfolio Dependency Summary:")
        print(f"   Average correlation: {summary['average_correlation']:.3f}")
        print(f"   Max correlation: {summary['max_correlation']:.3f}")
        print(f"   Min correlation: {summary['min_correlation']:.3f}")

    print("\n🎉 Copula Analysis Complete!")


if __name__ == "__main__":
    main()
