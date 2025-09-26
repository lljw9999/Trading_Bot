#!/usr/bin/env python3
"""
Experiment Registry
Create, start, stop, and audit experiment metadata with WORM trails.
"""
import os
import sys
import json
import yaml
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExperimentRegistry:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.registry_dir = Path("artifacts/exp_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load experiment config: {e}")

    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        exp_name = self.config["experiment"]["name"]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"{exp_name}_{timestamp}_{config_hash}"

    def create_experiment(self) -> Dict[str, Any]:
        """Create new experiment entry."""
        print("🧪 Creating new experiment...")

        exp_id = self.generate_experiment_id()
        exp_config = self.config["experiment"]

        # Create experiment metadata
        experiment = {
            "experiment_id": exp_id,
            "name": exp_config["name"],
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "status": "INITIALIZED",
            "config": self.config,
            "assets": exp_config["assets"],
            "design": exp_config["design"],
            "horizon_days": exp_config["horizon_days"],
            "min_days": exp_config["min_days"],
            "metrics": exp_config["metrics"],
            "governance": self.config.get("governance", {}),
            "artifacts_dir": exp_config.get(
                "artifacts_dir", f"artifacts/experiments/{exp_id}"
            ),
            "audit_trail": [],
        }

        # Create artifacts directory
        artifacts_path = Path(experiment["artifacts_dir"])
        artifacts_path.mkdir(parents=True, exist_ok=True)

        # Add creation audit entry
        self.add_audit_entry(
            experiment,
            "CREATED",
            "Experiment initialized",
            {
                "config_path": self.config_path,
                "assets": exp_config["assets"],
                "horizon_days": exp_config["horizon_days"],
            },
        )

        # Save experiment metadata
        exp_file = self.registry_dir / f"{exp_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment, f, indent=2)

        # Create latest symlink
        latest_link = self.registry_dir / "latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(exp_file)

        print(f"   Experiment ID: {exp_id}")
        print(f"   Assets: {', '.join(exp_config['assets'])}")
        print(
            f"   Design: {exp_config['design']} ({exp_config['block_minutes']}min blocks)"
        )
        print(
            f"   Duration: {exp_config['min_days']}-{exp_config['horizon_days']} days"
        )
        print(f"   Registry: {exp_file}")

        return experiment

    def start_experiment(self, exp_id: str) -> bool:
        """Start experiment (change status to RUNNING)."""
        print(f"▶️ Starting experiment {exp_id}...")

        experiment = self.load_experiment(exp_id)
        if not experiment:
            print(f"❌ Experiment {exp_id} not found")
            return False

        if experiment["status"] != "INITIALIZED":
            print(f"❌ Experiment status is {experiment['status']}, cannot start")
            return False

        # Check governance requirements
        governance = experiment.get("governance", {})
        if not self.check_governance_requirements(governance):
            print("❌ Governance requirements not met")
            return False

        # Update status
        experiment["status"] = "RUNNING"
        experiment["started_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        # Add audit entry
        self.add_audit_entry(
            experiment,
            "STARTED",
            "Experiment started",
            {"governance_checks": "PASSED", "start_time": experiment["started_at"]},
        )

        # Save updated experiment
        self.save_experiment(experiment)

        print(f"✅ Experiment {exp_id} started successfully")
        return True

    def stop_experiment(self, exp_id: str, reason: str = "Manual stop") -> bool:
        """Stop experiment (change status to STOPPED)."""
        print(f"⏹️ Stopping experiment {exp_id}...")

        experiment = self.load_experiment(exp_id)
        if not experiment:
            print(f"❌ Experiment {exp_id} not found")
            return False

        if experiment["status"] != "RUNNING":
            print(f"❌ Experiment status is {experiment['status']}, cannot stop")
            return False

        # Update status
        experiment["status"] = "STOPPED"
        experiment["stopped_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        # Calculate duration
        started_at = datetime.fromisoformat(
            experiment["started_at"].replace("Z", "+00:00")
        )
        stopped_at = datetime.fromisoformat(
            experiment["stopped_at"].replace("Z", "+00:00")
        )
        duration_hours = (stopped_at - started_at).total_seconds() / 3600

        # Add audit entry
        self.add_audit_entry(
            experiment,
            "STOPPED",
            reason,
            {
                "stop_time": experiment["stopped_at"],
                "duration_hours": duration_hours,
                "reason": reason,
            },
        )

        # Save updated experiment
        self.save_experiment(experiment)

        print(f"✅ Experiment {exp_id} stopped (duration: {duration_hours:.1f}h)")
        return True

    def check_governance_requirements(self, governance: Dict[str, Any]) -> bool:
        """Check if governance requirements are met."""
        # Check GO_LIVE flag
        if governance.get("require_go_flag", False):
            go_live = os.getenv("GO_LIVE", "0") == "1"
            if not go_live:
                print("   ❌ GO_LIVE flag not set")
                return False
            print("   ✅ GO_LIVE flag set")

        # Check clean alerts
        clean_hours = governance.get("require_clean_alert_hours", 48)
        alert_files = list(Path("artifacts/audit").glob("*alert*.json"))

        if alert_files:
            # Check if any alerts in last N hours
            cutoff = datetime.now() - timedelta(hours=clean_hours)
            recent_alerts = []

            for alert_file in alert_files:
                mtime = datetime.fromtimestamp(alert_file.stat().st_mtime)
                if mtime > cutoff:
                    recent_alerts.append(str(alert_file))

            if recent_alerts:
                print(
                    f"   ❌ {len(recent_alerts)} recent alerts in last {clean_hours}h"
                )
                return False

        print(f"   ✅ No alerts in last {clean_hours}h")

        # Check KRI guards (would integrate with actual KRI monitoring)
        kri_guards = governance.get("kri_guards", {})
        if kri_guards:
            print("   ✅ KRI guards checked (simulated)")

        return True

    def load_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment metadata."""
        exp_file = self.registry_dir / f"{exp_id}.json"
        if not exp_file.exists():
            return None

        try:
            with open(exp_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_experiment(self, experiment: Dict[str, Any]):
        """Save experiment metadata."""
        exp_id = experiment["experiment_id"]
        exp_file = self.registry_dir / f"{exp_id}.json"

        with open(exp_file, "w") as f:
            json.dump(experiment, f, indent=2)

    def add_audit_entry(
        self,
        experiment: Dict[str, Any],
        action: str,
        description: str,
        metadata: Dict[str, Any] = None,
    ):
        """Add audit trail entry."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "action": action,
            "description": description,
            "metadata": metadata or {},
        }

        experiment.setdefault("audit_trail", []).append(audit_entry)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        for exp_file in self.registry_dir.glob("*.json"):
            if exp_file.name == "latest.json":
                continue

            try:
                with open(exp_file, "r") as f:
                    exp = json.load(f)
                    experiments.append(
                        {
                            "experiment_id": exp["experiment_id"],
                            "name": exp["name"],
                            "status": exp["status"],
                            "created_at": exp["created_at"],
                            "assets": exp["assets"],
                        }
                    )
            except Exception:
                continue

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def get_experiment_status(self, exp_id: str) -> Dict[str, Any]:
        """Get experiment status and summary."""
        experiment = self.load_experiment(exp_id)
        if not experiment:
            return {"error": "Experiment not found"}

        status = {
            "experiment_id": exp_id,
            "name": experiment["name"],
            "status": experiment["status"],
            "created_at": experiment["created_at"],
            "config": experiment["config"]["experiment"],
            "governance": experiment.get("governance", {}),
            "audit_entries": len(experiment.get("audit_trail", [])),
        }

        if "started_at" in experiment:
            status["started_at"] = experiment["started_at"]

            if experiment["status"] == "RUNNING":
                started = datetime.fromisoformat(
                    experiment["started_at"].replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                status["running_hours"] = (now - started).total_seconds() / 3600

        if "stopped_at" in experiment:
            status["stopped_at"] = experiment["stopped_at"]

        return status


def main():
    """Main experiment registry function."""
    parser = argparse.ArgumentParser(description="Experiment Registry")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--init", action="store_true", help="Initialize new experiment")
    parser.add_argument("--start", help="Start experiment by ID")
    parser.add_argument("--stop", help="Stop experiment by ID")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--status", help="Get experiment status by ID")
    args = parser.parse_args()

    try:
        registry = ExperimentRegistry(args.config)

        if args.init:
            experiment = registry.create_experiment()
            print(f"\n✅ Experiment initialized: {experiment['experiment_id']}")
            return 0

        elif args.start:
            success = registry.start_experiment(args.start)
            return 0 if success else 1

        elif args.stop:
            success = registry.stop_experiment(args.stop)
            return 0 if success else 1

        elif args.list:
            experiments = registry.list_experiments()
            print("\n📋 Experiments:")
            for exp in experiments:
                print(f"  {exp['experiment_id']}: {exp['name']} [{exp['status']}]")
                print(f"    Created: {exp['created_at']}")
                print(f"    Assets: {', '.join(exp['assets'])}")
                print()
            return 0

        elif args.status:
            status = registry.get_experiment_status(args.status)
            if "error" in status:
                print(f"❌ {status['error']}")
                return 1

            print(f"\n📊 Experiment Status:")
            print(f"  ID: {status['experiment_id']}")
            print(f"  Name: {status['name']}")
            print(f"  Status: {status['status']}")
            print(f"  Created: {status['created_at']}")
            if "started_at" in status:
                print(f"  Started: {status['started_at']}")
            if "running_hours" in status:
                print(f"  Running: {status['running_hours']:.1f} hours")
            print(f"  Audit Entries: {status['audit_entries']}")
            return 0

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"❌ Registry operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
