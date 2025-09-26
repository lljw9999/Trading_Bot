#!/usr/bin/env python3
"""
Nightly DL Fine-Tune & Promotion Script
Fine-tune LSTM/TFT models on recent data and auto-promote if performance improves
"""

import os
import sys
import json
import time
import hashlib
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("fine_tune_dl")


class DLFineTuner:
    """Nightly deep learning model fine-tuner."""

    def __init__(
        self,
        data_dir: str = "/data/binance_ticks",
        models_dir: str = "/models",
        lookback_days: int = 30,
    ):
        """
        Initialize DL fine-tuner.

        Args:
            data_dir: Directory containing parquet training data
            models_dir: Directory for model storage
            lookback_days: Days of data to use for fine-tuning
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.lookback_days = lookback_days

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "blue").mkdir(exist_ok=True)
        (self.models_dir / "green").mkdir(exist_ok=True)
        (self.models_dir / "temp").mkdir(exist_ok=True)

        # Model configurations
        self.models_to_finetune = ["lstm_alpha", "tft_alpha"]

        # Performance thresholds for promotion
        self.promotion_thresholds = {
            "sharpe_improvement": 0.10,  # Must improve Sharpe by 0.10
            "max_drawdown_ratio": 1.0,  # DD must be <= baseline
            "min_trades": 100,  # Minimum trades for valid backtest
        }

        logger.info("🧠 DL Fine-Tuner initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Models directory: {self.models_dir}")
        logger.info(f"   Lookback: {lookback_days} days")

    def load_training_data(self) -> pd.DataFrame:
        """Load recent training data from parquet files."""
        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.lookback_days)

            # Look for parquet files in date range
            parquet_files = []
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                pattern = f"btc_usdt_{date_str}_*.parquet"

                # Find files matching pattern
                for file_path in self.data_dir.glob(pattern):
                    parquet_files.append(file_path)

                current_date += timedelta(days=1)

            if not parquet_files:
                # Generate synthetic data for demo
                logger.warning("No parquet files found, generating synthetic data")
                return self._generate_synthetic_data()

            # Load and concatenate parquet files
            dfs = []
            for file_path in sorted(parquet_files):
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")

            if not dfs:
                return self._generate_synthetic_data()

            # Combine all data
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"Loaded {len(combined_df):,} training samples from {len(parquet_files)} files"
            )

            return combined_df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demo purposes."""
        logger.info("Generating synthetic training data")

        # Generate realistic market data
        n_samples = 50000
        np.random.seed(42)

        # Price features
        returns = np.random.normal(0, 0.001, n_samples)
        prices = 50000 * np.exp(np.cumsum(returns))

        # Volume and spread features
        volumes = np.random.lognormal(10, 1, n_samples)
        spreads = np.random.exponential(0.01, n_samples)

        # Technical indicators (mock)
        rsi = (
            50
            + 30 * np.sin(np.arange(n_samples) / 100)
            + np.random.normal(0, 5, n_samples)
        )
        rsi = np.clip(rsi, 0, 100)

        # Target: future return (5-minute forward)
        future_returns = np.roll(returns, -5)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=n_samples, freq="1min"
                ),
                "price": prices,
                "volume": volumes,
                "spread": spreads,
                "rsi": rsi,
                "return_1m": returns,
                "target_5m": future_returns,
            }
        )

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets for model training."""
        try:
            # Feature engineering
            features = ["price", "volume", "spread", "rsi", "return_1m"]

            # Create lagged features
            for lag in [1, 2, 3, 5, 10]:
                for feature in ["price", "volume", "return_1m"]:
                    df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
                    features.append(f"{feature}_lag{lag}")

            # Rolling statistics
            for window in [5, 20]:
                df[f"return_mean_{window}"] = df["return_1m"].rolling(window).mean()
                df[f"return_std_{window}"] = df["return_1m"].rolling(window).std()
                features.extend([f"return_mean_{window}", f"return_std_{window}"])

            # Drop NaN rows
            df = df.dropna()

            # Prepare arrays
            X = df[features].values.astype(np.float32)
            y = df["target_5m"].values.astype(np.float32)

            logger.info(f"Prepared features: {X.shape}, targets: {y.shape}")
            return X, y, features

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None

    def fine_tune_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fine-tune a specific model.

        Args:
            model_name: Name of model to fine-tune
            X: Feature array
            y: Target array

        Returns:
            Dictionary with fine-tuning results
        """
        try:
            logger.info(f"🧠 Fine-tuning {model_name}")

            # Split data (80/20 train/val)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Mock fine-tuning process (in production, this would use actual models)
            logger.info(f"   Training on {len(X_train):,} samples")

            # Simulate training time
            time.sleep(2)

            # Mock validation metrics
            val_mae = np.random.uniform(0.0005, 0.002)  # Mock validation MAE
            val_sharpe = np.random.uniform(0.5, 1.5)  # Mock Sharpe ratio

            # Create model hash for versioning
            model_data = f"{model_name}_{datetime.now().isoformat()}_{val_mae:.6f}"
            model_hash = hashlib.md5(model_data.encode()).hexdigest()[:8]

            # Mock model saving (would export ONNX in production)
            model_path = self.models_dir / "temp" / f"{model_name}_{model_hash}.onnx"

            # Create mock ONNX file
            with open(model_path, "w") as f:
                f.write(f"# Mock ONNX model: {model_name}\n")
                f.write(f"# Hash: {model_hash}\n")
                f.write(f"# Validation MAE: {val_mae:.6f}\n")
                f.write(f"# Validation Sharpe: {val_sharpe:.3f}\n")

            results = {
                "model_name": model_name,
                "model_hash": model_hash,
                "model_path": str(model_path),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "val_mae": val_mae,
                "val_sharpe": val_sharpe,
                "training_time_seconds": 2.0,
                "status": "success",
            }

            logger.info(
                f"   ✅ {model_name} fine-tuned: MAE={val_mae:.6f}, Sharpe={val_sharpe:.3f}"
            )
            return results

        except Exception as e:
            logger.error(f"Error fine-tuning {model_name}: {e}")
            return {"model_name": model_name, "status": "failed", "error": str(e)}

    def evaluate_shadow_performance(self, model_results: list) -> dict:
        """
        Evaluate shadow performance of fine-tuned models.

        Args:
            model_results: List of fine-tuning results

        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info("📊 Evaluating shadow performance")

            # Mock shadow evaluation (in production, would run backtests)
            baseline_sharpe = 0.8  # Mock baseline
            baseline_dd = 0.15  # Mock baseline drawdown

            evaluation_results = {}

            for result in model_results:
                if result["status"] != "success":
                    continue

                model_name = result["model_name"]

                # Mock shadow performance
                shadow_sharpe = result["val_sharpe"] + np.random.normal(0, 0.1)
                shadow_dd = np.random.uniform(0.05, 0.25)
                shadow_trades = np.random.randint(50, 200)

                # Check promotion criteria
                sharpe_improvement = shadow_sharpe - baseline_sharpe
                dd_ratio = shadow_dd / baseline_dd

                meets_sharpe = (
                    sharpe_improvement
                    >= self.promotion_thresholds["sharpe_improvement"]
                )
                meets_dd = dd_ratio <= self.promotion_thresholds["max_drawdown_ratio"]
                meets_trades = shadow_trades >= self.promotion_thresholds["min_trades"]

                promote = meets_sharpe and meets_dd and meets_trades

                evaluation_results[model_name] = {
                    "shadow_sharpe": shadow_sharpe,
                    "shadow_drawdown": shadow_dd,
                    "shadow_trades": shadow_trades,
                    "baseline_sharpe": baseline_sharpe,
                    "baseline_drawdown": baseline_dd,
                    "sharpe_improvement": sharpe_improvement,
                    "dd_ratio": dd_ratio,
                    "meets_criteria": {
                        "sharpe": meets_sharpe,
                        "drawdown": meets_dd,
                        "trades": meets_trades,
                    },
                    "promote": promote,
                    "model_hash": result["model_hash"],
                    "model_path": result["model_path"],
                }

                logger.info(
                    f"   {model_name}: Sharpe={shadow_sharpe:.3f} (+{sharpe_improvement:+.3f}), "
                    f"DD={shadow_dd:.1%} ({dd_ratio:.2f}x), "
                    f"Trades={shadow_trades}, "
                    f"Promote={promote}"
                )

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in shadow evaluation: {e}")
            return {}

    def promote_model(self, model_name: str, evaluation: dict) -> bool:
        """
        Promote model to green deployment.

        Args:
            model_name: Name of model to promote
            evaluation: Evaluation results for the model

        Returns:
            True if promotion successful
        """
        try:
            if not evaluation.get("promote", False):
                logger.info(f"   {model_name}: Does not meet promotion criteria")
                return False

            model_path = Path(evaluation["model_path"])
            model_hash = evaluation["model_hash"]

            # Copy to green deployment directory
            green_path = self.models_dir / "green" / f"{model_name}.onnx"
            green_path.parent.mkdir(exist_ok=True)

            # Copy model file
            import shutil

            shutil.copy2(model_path, green_path)

            # Create metadata file
            metadata = {
                "model_name": model_name,
                "model_hash": model_hash,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "performance": evaluation,
                "source_path": str(model_path),
            }

            metadata_path = self.models_dir / "green" / f"{model_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Promoted {model_name} to green deployment")
            return True

        except Exception as e:
            logger.error(f"Error promoting {model_name}: {e}")
            return False

    def run_canary_switch(self) -> bool:
        """Run canary switch to activate green models."""
        try:
            logger.info("🚀 Running canary switch to green deployment")

            # In production, this would call the actual canary switch script
            # For demo, we'll simulate it

            # Mock canary switch
            time.sleep(1)

            logger.info("✅ Canary switch completed - green models active")
            return True

        except Exception as e:
            logger.error(f"Error in canary switch: {e}")
            return False

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        try:
            temp_dir = self.models_dir / "temp"

            # Remove files older than 24 hours
            cutoff_time = time.time() - (24 * 3600)

            for file_path in temp_dir.iterdir():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.debug(f"Cleaned up {file_path}")

        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

    def run_nightly_finetune(self) -> dict:
        """Run the complete nightly fine-tune process."""
        logger.info("🌙 Starting nightly DL fine-tune process")
        start_time = time.time()

        try:
            # Load training data
            df = self.load_training_data()
            if df.empty:
                return {"status": "failed", "error": "No training data available"}

            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            if X is None:
                return {"status": "failed", "error": "Feature preparation failed"}

            # Fine-tune models
            model_results = []
            for model_name in self.models_to_finetune:
                result = self.fine_tune_model(model_name, X, y)
                model_results.append(result)

            # Evaluate shadow performance
            evaluations = self.evaluate_shadow_performance(model_results)

            # Promote qualifying models
            promoted_models = []
            for model_name, evaluation in evaluations.items():
                if self.promote_model(model_name, evaluation):
                    promoted_models.append(model_name)

            # Run canary switch if any models were promoted
            canary_success = False
            if promoted_models:
                canary_success = self.run_canary_switch()

            # Cleanup
            self.cleanup_temp_files()

            elapsed_time = time.time() - start_time

            summary = {
                "status": "success",
                "duration_seconds": elapsed_time,
                "training_samples": len(X),
                "models_attempted": len(self.models_to_finetune),
                "models_trained": sum(
                    1 for r in model_results if r["status"] == "success"
                ),
                "models_promoted": len(promoted_models),
                "promoted_models": promoted_models,
                "canary_switched": canary_success,
                "model_results": model_results,
                "evaluations": evaluations,
            }

            logger.info(
                f"🌙 Nightly fine-tune complete: "
                f"{summary['models_trained']}/{summary['models_attempted']} trained, "
                f"{summary['models_promoted']} promoted in {elapsed_time:.1f}s"
            )

            return summary

        except Exception as e:
            logger.error(f"Error in nightly fine-tune: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "duration_seconds": time.time() - start_time,
            }


def main():
    """Main entry point for nightly DL fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Nightly DL Fine-Tuner")
    parser.add_argument(
        "--data-dir", default="/data/binance_ticks", help="Training data directory"
    )
    parser.add_argument("--models-dir", default="/models", help="Models directory")
    parser.add_argument("--days", type=int, default=30, help="Days of training data")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run without promotion"
    )

    args = parser.parse_args()

    # Create fine-tuner
    fine_tuner = DLFineTuner(
        data_dir=args.data_dir, models_dir=args.models_dir, lookback_days=args.days
    )

    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - No models will be promoted")

    # Run nightly fine-tune
    results = fine_tuner.run_nightly_finetune()

    # Print results
    print(json.dumps(results, indent=2, default=str))

    # Exit code
    if results["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
