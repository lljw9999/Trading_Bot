#!/usr/bin/env python3
"""
Model Governance

Implements model lifecycle management and governance:
- Daily checkpoint snapshots with metrics bundles
- Promotion rules: +0.10 Sharpe, -10% maxDD, stability checks
- Rollback rules: quarantine models with 2+ P1 incidents in 7 days
- Out-of-sample validation and A/B testing framework
"""

import argparse
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import hashlib
import pickle
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import redis
    import numpy as np

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger("model_governance")


class ModelGovernanceManager:
    """
    Manages ML model lifecycle including checkpointing, validation,
    promotion/rollback decisions, and performance tracking.
    """

    def __init__(self):
        """Initialize model governance manager."""
        self.redis_client = None
        if DEPS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        # Model governance configuration
        self.governance_config = {
            "checkpoint_cadence_hours": 24,  # Daily checkpoints
            "promotion_thresholds": {
                "min_sharpe_improvement": 0.10,  # +0.10 Sharpe
                "max_drawdown_reduction": 0.10,  # -10% maxDD
                "min_sample_days": 5,  # 5-day out-of-sample
                "stability_tolerance": 0.20,  # Entropy variance within 20%
            },
            "rollback_conditions": {
                "max_incidents_7days": 2,  # 2+ P1 incidents
                "quarantine_days": 14,  # 14-day quarantine
            },
            "model_types": ["rl_policy", "alpha_ensemble", "execution_policy"],
            "metrics_tracked": [
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "entropy_mean",
                "entropy_variance",
                "q_spread_mean",
                "action_diversity",
            ],
        }

        # Model storage paths
        self.model_paths = {
            "checkpoints": Path("checkpoints"),
            "active_models": Path("models/active"),
            "candidate_models": Path("models/candidates"),
            "quarantined_models": Path("models/quarantined"),
        }

        # Ensure directories exist
        for path in self.model_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized model governance manager")

    def create_model_checkpoint(
        self, model_type: str, model_data: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Create daily model checkpoint with metrics bundle.

        Args:
            model_type: Type of model (rl_policy, alpha_ensemble, etc.)
            model_data: Model data to checkpoint (optional, will load from active)

        Returns:
            Checkpoint creation results
        """
        try:
            logger.info(f"📸 Creating checkpoint for {model_type}")

            checkpoint_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            checkpoint_results = {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "checkpoint_id": checkpoint_id,
                "status": "creating",
            }

            # Collect current model state
            if model_data is None:
                model_data = self._load_active_model(model_type)

            if not model_data:
                checkpoint_results["status"] = "no_active_model"
                return checkpoint_results

            # Collect performance metrics
            metrics_bundle = self._collect_performance_metrics(model_type)
            checkpoint_results["metrics"] = metrics_bundle

            # Create checkpoint package
            checkpoint_package = {
                "checkpoint_id": checkpoint_id,
                "model_type": model_type,
                "created_timestamp": datetime.now().isoformat(),
                "model_data": model_data,
                "metrics_bundle": metrics_bundle,
                "model_hash": self._calculate_model_hash(model_data),
                "governance_metadata": {
                    "eligible_for_promotion": False,  # Will be determined later
                    "incidents_count": 0,
                    "quarantine_status": "active",
                },
            }

            # Save checkpoint
            checkpoint_path = self.model_paths["checkpoints"] / f"{checkpoint_id}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_package, f)

            checkpoint_results.update(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "model_hash": checkpoint_package["model_hash"],
                    "metrics_summary": self._summarize_metrics(metrics_bundle),
                    "status": "completed",
                }
            )

            # Store checkpoint reference in Redis
            if self.redis_client:
                checkpoint_key = f"model_checkpoints:{model_type}"
                checkpoint_info = {
                    "checkpoint_id": checkpoint_id,
                    "timestamp": datetime.now().isoformat(),
                    "path": str(checkpoint_path),
                    "hash": checkpoint_package["model_hash"],
                }
                self.redis_client.lpush(checkpoint_key, json.dumps(checkpoint_info))

                # Keep only last 30 checkpoints
                self.redis_client.ltrim(checkpoint_key, 0, 29)

            logger.info(f"✅ Checkpoint created: {checkpoint_id}")
            return checkpoint_results

        except Exception as e:
            logger.error(f"Error creating model checkpoint: {e}")
            return {"error": str(e), "status": "failed"}

    def evaluate_promotion_candidate(
        self, model_type: str, candidate_checkpoint: str = None
    ) -> Dict[str, any]:
        """
        Evaluate model for promotion based on governance rules.

        Args:
            model_type: Type of model to evaluate
            candidate_checkpoint: Specific checkpoint to evaluate (optional)

        Returns:
            Promotion evaluation results
        """
        try:
            logger.info(f"🔍 Evaluating promotion candidate for {model_type}")

            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "candidate_checkpoint": candidate_checkpoint,
                "status": "evaluating",
            }

            # Load candidate model
            if candidate_checkpoint:
                candidate_model = self._load_checkpoint(candidate_checkpoint)
            else:
                candidate_model = self._get_latest_checkpoint(model_type)

            if not candidate_model:
                evaluation_results["status"] = "no_candidate"
                return evaluation_results

            evaluation_results["candidate_id"] = candidate_model["checkpoint_id"]

            # Load current active model for comparison
            active_model = self._load_active_model_checkpoint(model_type)

            if not active_model:
                # No active model - candidate becomes active if it meets minimum standards
                evaluation_results["comparison_type"] = "absolute_evaluation"
                promotion_decision = self._evaluate_absolute_performance(
                    candidate_model
                )
            else:
                # Compare candidate vs active model
                evaluation_results["comparison_type"] = "relative_evaluation"
                evaluation_results["active_model_id"] = active_model["checkpoint_id"]
                promotion_decision = self._evaluate_relative_performance(
                    candidate_model, active_model
                )

            evaluation_results.update(promotion_decision)

            # Check for disqualifying conditions
            disqualification_check = self._check_disqualifying_conditions(
                candidate_model
            )
            evaluation_results["disqualification_check"] = disqualification_check

            # Final promotion decision
            final_decision = (
                promotion_decision["meets_thresholds"]
                and not disqualification_check["disqualified"]
            )

            evaluation_results.update(
                {
                    "final_decision": "promote" if final_decision else "reject",
                    "promotion_eligible": final_decision,
                    "status": "completed",
                }
            )

            logger.info(
                f"✅ Promotion evaluation: {evaluation_results['final_decision']}"
            )
            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating promotion candidate: {e}")
            return {"error": str(e), "status": "failed"}

    def promote_model(
        self, model_type: str, candidate_checkpoint: str
    ) -> Dict[str, any]:
        """
        Promote candidate model to active status.

        Args:
            model_type: Type of model to promote
            candidate_checkpoint: Checkpoint ID to promote

        Returns:
            Model promotion results
        """
        try:
            logger.info(f"⬆️ Promoting model {candidate_checkpoint}")

            promotion_results = {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "candidate_checkpoint": candidate_checkpoint,
                "status": "promoting",
            }

            # Load candidate model
            candidate_model = self._load_checkpoint(candidate_checkpoint)
            if not candidate_model:
                promotion_results["status"] = "candidate_not_found"
                return promotion_results

            # Backup current active model
            backup_result = self._backup_active_model(model_type)
            promotion_results["backup_result"] = backup_result

            # Deploy candidate as new active model
            deployment_result = self._deploy_as_active(candidate_model, model_type)
            promotion_results["deployment_result"] = deployment_result

            # Update model registry
            registry_update = self._update_model_registry(
                model_type, candidate_checkpoint, "promoted"
            )
            promotion_results["registry_update"] = registry_update

            # Record promotion event
            self._record_promotion_event(model_type, candidate_checkpoint)

            promotion_results["status"] = (
                "completed" if deployment_result["success"] else "failed"
            )

            if promotion_results["status"] == "completed":
                logger.info(f"✅ Model promoted: {candidate_checkpoint} → active")
            else:
                logger.error(f"❌ Model promotion failed")

            return promotion_results

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return {"error": str(e), "status": "failed"}

    def quarantine_model(
        self, model_type: str, model_id: str, reason: str
    ) -> Dict[str, any]:
        """
        Quarantine model due to incidents or poor performance.

        Args:
            model_type: Type of model to quarantine
            model_id: Model/checkpoint ID to quarantine
            reason: Reason for quarantine

        Returns:
            Quarantine results
        """
        try:
            logger.warning(f"🚫 Quarantining model {model_id}: {reason}")

            quarantine_results = {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "model_id": model_id,
                "reason": reason,
                "status": "quarantining",
            }

            # Move model to quarantine directory
            quarantine_result = self._move_to_quarantine(model_type, model_id, reason)
            quarantine_results["quarantine_result"] = quarantine_result

            # Update model registry
            registry_update = self._update_model_registry(
                model_type, model_id, "quarantined"
            )
            quarantine_results["registry_update"] = registry_update

            # Set quarantine expiration
            quarantine_until = datetime.now() + timedelta(
                days=self.governance_config["rollback_conditions"]["quarantine_days"]
            )
            quarantine_results["quarantine_until"] = quarantine_until.isoformat()

            # Record quarantine event
            self._record_quarantine_event(model_type, model_id, reason)

            quarantine_results["status"] = "completed"

            logger.info(f"✅ Model quarantined until {quarantine_until}")
            return quarantine_results

        except Exception as e:
            logger.error(f"Error quarantining model: {e}")
            return {"error": str(e), "status": "failed"}

    def run_model_governance_cycle(self) -> Dict[str, any]:
        """
        Run complete model governance cycle for all model types.

        Returns:
            Complete governance cycle results
        """
        try:
            logger.info("🏛️ Running model governance cycle...")

            cycle_results = {
                "timestamp": datetime.now().isoformat(),
                "procedure": "model_governance_cycle",
                "model_types": {},
            }

            for model_type in self.governance_config["model_types"]:
                logger.info(f"Processing {model_type}...")

                model_results = {"model_type": model_type, "steps": {}}

                # Step 1: Create checkpoint
                checkpoint_result = self.create_model_checkpoint(model_type)
                model_results["steps"]["checkpoint"] = checkpoint_result

                # Step 2: Check for quarantine conditions
                quarantine_check = self._check_quarantine_conditions(model_type)
                model_results["steps"]["quarantine_check"] = quarantine_check

                if quarantine_check.get("should_quarantine", False):
                    # Quarantine current model
                    quarantine_result = self.quarantine_model(
                        model_type,
                        quarantine_check["model_id"],
                        quarantine_check["reason"],
                    )
                    model_results["steps"]["quarantine"] = quarantine_result

                # Step 3: Evaluate promotion candidates
                promotion_evaluation = self.evaluate_promotion_candidate(model_type)
                model_results["steps"]["promotion_evaluation"] = promotion_evaluation

                # Step 4: Execute promotion if eligible
                if promotion_evaluation.get("promotion_eligible", False):
                    promotion_result = self.promote_model(
                        model_type, promotion_evaluation["candidate_id"]
                    )
                    model_results["steps"]["promotion"] = promotion_result

                cycle_results["model_types"][model_type] = model_results

            # Overall cycle summary
            cycle_results["summary"] = self._summarize_governance_cycle(
                cycle_results["model_types"]
            )

            return cycle_results

        except Exception as e:
            logger.error(f"Error in model governance cycle: {e}")
            return {"error": str(e)}

    # Helper methods (simplified implementations)

    def _load_active_model(self, model_type: str) -> Optional[Dict]:
        """Load current active model."""
        # Mock implementation - would load actual model
        return {
            "model_type": model_type,
            "version": "1.0.0",
            "parameters": {"param1": 0.5, "param2": 1.0},
            "architecture": "mock_architecture",
        }

    def _collect_performance_metrics(self, model_type: str) -> Dict[str, any]:
        """Collect performance metrics for model."""
        import random

        # Mock performance metrics
        base_sharpe = random.uniform(0.8, 1.8)
        base_drawdown = random.uniform(0.05, 0.15)

        return {
            "collection_timestamp": datetime.now().isoformat(),
            "evaluation_period_days": 5,
            "sharpe_ratio": base_sharpe,
            "max_drawdown": base_drawdown,
            "win_rate": random.uniform(0.55, 0.75),
            "entropy_mean": random.uniform(0.3, 0.8),
            "entropy_variance": random.uniform(0.01, 0.05),
            "q_spread_mean": random.uniform(0.02, 0.08),
            "action_diversity": random.uniform(0.6, 0.9),
            "sample_count": random.randint(1000, 5000),
            "stability_metrics": {
                "entropy_cv": random.uniform(0.1, 0.3),
                "performance_consistency": random.uniform(0.7, 0.95),
            },
        }

    def _calculate_model_hash(self, model_data: Dict) -> str:
        """Calculate hash of model for versioning."""
        model_str = json.dumps(model_data, sort_keys=True)
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]

    def _summarize_metrics(self, metrics: Dict[str, any]) -> Dict[str, any]:
        """Summarize metrics for display."""
        return {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "sample_count": metrics.get("sample_count", 0),
        }

    def _get_latest_checkpoint(self, model_type: str) -> Optional[Dict]:
        """Get latest checkpoint for model type."""
        try:
            checkpoint_dir = self.model_paths["checkpoints"]
            checkpoints = list(checkpoint_dir.glob(f"{model_type}_*.pkl"))

            if not checkpoints:
                return None

            # Sort by creation time and get latest
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

            with open(latest_checkpoint, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            logger.error(f"Error loading latest checkpoint: {e}")
            return None

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Load specific checkpoint by ID."""
        try:
            checkpoint_path = self.model_paths["checkpoints"] / f"{checkpoint_id}.pkl"

            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None

    def _load_active_model_checkpoint(self, model_type: str) -> Optional[Dict]:
        """Load checkpoint of currently active model."""
        # Mock implementation
        return self._get_latest_checkpoint(model_type)

    def _evaluate_absolute_performance(self, candidate_model: Dict) -> Dict[str, any]:
        """Evaluate model performance against absolute thresholds."""
        metrics = candidate_model["metrics_bundle"]

        # Minimum performance thresholds for new models
        min_sharpe = 0.5
        max_allowed_drawdown = 0.20

        meets_thresholds = (
            metrics.get("sharpe_ratio", 0) >= min_sharpe
            and metrics.get("max_drawdown", 1) <= max_allowed_drawdown
        )

        return {
            "evaluation_type": "absolute",
            "thresholds": {
                "min_sharpe": min_sharpe,
                "max_drawdown": max_allowed_drawdown,
            },
            "candidate_metrics": self._summarize_metrics(metrics),
            "meets_thresholds": meets_thresholds,
        }

    def _evaluate_relative_performance(
        self, candidate_model: Dict, active_model: Dict
    ) -> Dict[str, any]:
        """Evaluate candidate vs active model performance."""
        candidate_metrics = candidate_model["metrics_bundle"]
        active_metrics = active_model["metrics_bundle"]

        # Calculate improvements
        sharpe_improvement = candidate_metrics.get(
            "sharpe_ratio", 0
        ) - active_metrics.get("sharpe_ratio", 0)
        drawdown_reduction = active_metrics.get(
            "max_drawdown", 0
        ) - candidate_metrics.get("max_drawdown", 0)

        # Relative improvement calculation
        drawdown_pct_reduction = drawdown_reduction / max(
            active_metrics.get("max_drawdown", 0.01), 0.01
        )

        # Check thresholds
        meets_thresholds = (
            sharpe_improvement
            >= self.governance_config["promotion_thresholds"]["min_sharpe_improvement"]
            and drawdown_pct_reduction
            >= self.governance_config["promotion_thresholds"]["max_drawdown_reduction"]
        )

        return {
            "evaluation_type": "relative",
            "active_metrics": self._summarize_metrics(active_metrics),
            "candidate_metrics": self._summarize_metrics(candidate_metrics),
            "improvements": {
                "sharpe_improvement": sharpe_improvement,
                "drawdown_reduction": drawdown_reduction,
                "drawdown_pct_reduction": drawdown_pct_reduction,
            },
            "thresholds": self.governance_config["promotion_thresholds"],
            "meets_thresholds": meets_thresholds,
        }

    def _check_disqualifying_conditions(self, candidate_model: Dict) -> Dict[str, any]:
        """Check for conditions that disqualify model from promotion."""
        # Check stability
        metrics = candidate_model["metrics_bundle"]
        entropy_cv = metrics.get("stability_metrics", {}).get("entropy_cv", 0)

        stability_disqualified = (
            entropy_cv
            > self.governance_config["promotion_thresholds"]["stability_tolerance"]
        )

        # Check incident history (mock)
        recent_incidents = 0  # Would check actual incident records
        incident_disqualified = (
            recent_incidents
            >= self.governance_config["rollback_conditions"]["max_incidents_7days"]
        )

        disqualified = stability_disqualified or incident_disqualified

        return {
            "disqualified": disqualified,
            "reasons": {
                "stability_issues": stability_disqualified,
                "incident_history": incident_disqualified,
            },
            "entropy_cv": entropy_cv,
            "recent_incidents": recent_incidents,
        }

    def _backup_active_model(self, model_type: str) -> Dict[str, any]:
        """Backup currently active model before promotion."""
        try:
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"models/backups/{model_type}")
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Mock backup
            return {
                "success": True,
                "backup_path": str(backup_dir / f"backup_{backup_timestamp}.pkl"),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _deploy_as_active(
        self, candidate_model: Dict, model_type: str
    ) -> Dict[str, any]:
        """Deploy candidate model as new active model."""
        try:
            active_path = self.model_paths["active_models"] / f"{model_type}.pkl"

            with open(active_path, "wb") as f:
                pickle.dump(candidate_model, f)

            # Update Redis registry
            if self.redis_client:
                self.redis_client.set(
                    f"active_model:{model_type}", candidate_model["checkpoint_id"]
                )

            return {"success": True, "active_path": str(active_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_model_registry(
        self, model_type: str, model_id: str, status: str
    ) -> Dict[str, any]:
        """Update model registry with new status."""
        try:
            if self.redis_client:
                registry_key = f"model_registry:{model_type}:{model_id}"
                registry_data = {
                    "status": status,
                    "updated_timestamp": datetime.now().isoformat(),
                    "model_type": model_type,
                    "model_id": model_id,
                }
                self.redis_client.set(registry_key, json.dumps(registry_data))

            return {"success": True, "status": status}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _record_promotion_event(self, model_type: str, checkpoint_id: str):
        """Record model promotion event."""
        event = {
            "event_type": "model_promotion",
            "model_type": model_type,
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
        }

        if self.redis_client:
            self.redis_client.lpush("events:model_governance", json.dumps(event))

    def _move_to_quarantine(
        self, model_type: str, model_id: str, reason: str
    ) -> Dict[str, any]:
        """Move model to quarantine directory."""
        try:
            quarantine_dir = self.model_paths["quarantined_models"]

            # Create quarantine record
            quarantine_record = {
                "model_id": model_id,
                "model_type": model_type,
                "reason": reason,
                "quarantined_timestamp": datetime.now().isoformat(),
                "quarantine_until": (datetime.now() + timedelta(days=14)).isoformat(),
            }

            quarantine_path = quarantine_dir / f"{model_id}_quarantine.json"
            with open(quarantine_path, "w") as f:
                json.dump(quarantine_record, f, indent=2)

            return {"success": True, "quarantine_path": str(quarantine_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _record_quarantine_event(self, model_type: str, model_id: str, reason: str):
        """Record model quarantine event."""
        event = {
            "event_type": "model_quarantine",
            "model_type": model_type,
            "model_id": model_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }

        if self.redis_client:
            self.redis_client.lpush("events:model_governance", json.dumps(event))

    def _check_quarantine_conditions(self, model_type: str) -> Dict[str, any]:
        """Check if current model should be quarantined."""
        # Mock incident checking
        return {
            "should_quarantine": False,
            "model_id": "current_active",
            "reason": None,
            "incident_count": 0,
        }

    def _summarize_governance_cycle(
        self, model_results: Dict[str, any]
    ) -> Dict[str, any]:
        """Summarize governance cycle results."""
        total_models = len(model_results)
        promotions = sum(1 for r in model_results.values() if "promotion" in r["steps"])
        quarantines = sum(
            1 for r in model_results.values() if "quarantine" in r["steps"]
        )
        checkpoints = sum(
            1
            for r in model_results.values()
            if r["steps"]["checkpoint"]["status"] == "completed"
        )

        return {
            "total_models_processed": total_models,
            "checkpoints_created": checkpoints,
            "promotions_executed": promotions,
            "quarantines_executed": quarantines,
            "overall_status": "completed",
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model Governance Manager")

    parser.add_argument(
        "--action",
        choices=["checkpoint", "evaluate", "promote", "quarantine", "governance"],
        default="governance",
        help="Action to perform",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rl_policy", "alpha_ensemble", "execution_policy"],
        help="Model type to operate on",
    )
    parser.add_argument("--checkpoint-id", type=str, help="Specific checkpoint ID")
    parser.add_argument("--reason", type=str, help="Reason for quarantine")
    parser.add_argument("--output", type=str, help="Output file for results")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("🏛️ Starting Model Governance Manager")

    try:
        manager = ModelGovernanceManager()

        if args.action == "checkpoint":
            if not args.model_type:
                logger.error("--model-type required for checkpoint action")
                return 1
            results = manager.create_model_checkpoint(args.model_type)
        elif args.action == "evaluate":
            if not args.model_type:
                logger.error("--model-type required for evaluate action")
                return 1
            results = manager.evaluate_promotion_candidate(
                args.model_type, args.checkpoint_id
            )
        elif args.action == "promote":
            if not args.model_type or not args.checkpoint_id:
                logger.error(
                    "--model-type and --checkpoint-id required for promote action"
                )
                return 1
            results = manager.promote_model(args.model_type, args.checkpoint_id)
        elif args.action == "quarantine":
            if not args.model_type or not args.checkpoint_id or not args.reason:
                logger.error(
                    "--model-type, --checkpoint-id, and --reason required for quarantine action"
                )
                return 1
            results = manager.quarantine_model(
                args.model_type, args.checkpoint_id, args.reason
            )
        else:  # governance
            results = manager.run_model_governance_cycle()

        print(f"\n🏛️ MODEL GOVERNANCE RESULTS:")
        print(json.dumps(results, indent=2))

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error in model governance: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
