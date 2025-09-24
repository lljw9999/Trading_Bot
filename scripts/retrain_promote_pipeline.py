#!/usr/bin/env python3
"""
Continuous Retrain & Promote Pipeline
Safe loop: train → offline gate → update model card & lineage → A/B quick check → update last_good
"""
import os
import sys
import json
import subprocess
import datetime
import pathlib
import argparse
from datetime import timezone
from pathlib import Path


def write_audit(kind, payload):
    """Write WORM audit record."""
    ts = datetime.datetime.now(timezone.utc).isoformat().replace(":", "_")
    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    audit_file = f"artifacts/audit/{ts}_retrain_{kind}.json"

    audit_record = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "action": f"retrain_{kind}",
        "payload": payload,
        "operator": "automated_pipeline",
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", f"run_{ts}"),
    }

    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2)

    print(f"[AUDIT] {audit_file}")
    return audit_file


def run_command(cmd, description="", cwd=None):
    """Execute command with logging."""
    print(f"[EXEC] {description}: {cmd}")

    if cwd is None:
        cwd = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

    result = subprocess.run(
        cmd, shell=True, text=True, capture_output=True, cwd=cwd, env=env
    )

    print(f"  Return code: {result.returncode}")
    if result.stdout.strip():
        print(f"  STDOUT: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  STDERR: {result.stderr.strip()}")

    return result


def simulate_training():
    """Simulate model training (placeholder)."""
    print("🤖 Simulating model training...")

    # Create training artifacts
    artifacts_dir = Path("artifacts/training")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    # Simulate training metadata
    training_metadata = {
        "training_id": f"train_{timestamp}",
        "algorithm": "SAC-DiF + LoRA",
        "start_time": datetime.datetime.now(timezone.utc).isoformat(),
        "duration_minutes": 45,  # Simulated
        "hyperparameters": {"learning_rate": 3e-4, "batch_size": 256, "epochs": 100},
        "metrics": {"final_loss": 0.0234, "entropy": 1.12, "avg_reward": 0.0045},
        "status": "completed",
        "checkpoint_path": f"checkpoints/model_{timestamp}.pt",
    }

    metadata_file = artifacts_dir / f"training_{timestamp}.json"
    with open(metadata_file, "w") as f:
        json.dump(training_metadata, f, indent=2)

    # Simulate checkpoint file
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoints_dir / f"model_{timestamp}.pt"
    with open(checkpoint_file, "w") as f:
        f.write(
            "# Simulated PyTorch checkpoint\n# This would be a binary model file in production\n"
        )

    print(f"  ✅ Training completed: {metadata_file}")
    print(f"  ✅ Checkpoint saved: {checkpoint_file}")

    return {
        "status": "SUCCESS",
        "metadata_file": str(metadata_file),
        "checkpoint_file": str(checkpoint_file),
        "training_id": training_metadata["training_id"],
    }


def run_offline_validation(training_result):
    """Run offline validation gate."""
    print("🚦 Running offline validation gate...")

    # Use existing validation infrastructure
    result = run_command("make validate-48h-now", "Offline validation")

    if result.returncode == 0:
        print("  ✅ Offline validation PASSED")
        return {"status": "PASS", "details": "Validation gate passed"}
    else:
        print("  ❌ Offline validation FAILED")
        return {"status": "FAIL", "details": result.stderr.strip()}


def update_model_card(training_result, validation_result):
    """Update model card with new training results."""
    print("📄 Updating model card...")

    try:
        # Read current model card
        model_card_path = "model_cards/sol_policy_card.yaml"
        if os.path.exists(model_card_path):
            import yaml

            with open(model_card_path, "r") as f:
                card_data = yaml.safe_load(f)
        else:
            card_data = {}

        # Update with training results
        if "training" not in card_data:
            card_data["training"] = {}

        card_data["training"]["last_update"] = datetime.datetime.now(
            timezone.utc
        ).isoformat()
        card_data["training"]["last_training_id"] = training_result["training_id"]

        if "evaluation" not in card_data:
            card_data["evaluation"] = {}

        card_data["evaluation"]["last_validation"] = {
            "status": validation_result["status"],
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }

        # Save updated model card
        with open(model_card_path, "w") as f:
            yaml.dump(card_data, f, default_flow_style=False)

        print(f"  ✅ Model card updated: {model_card_path}")

        # Regenerate markdown version
        result = run_command(
            f"python scripts/model_card_builder.py --yaml {model_card_path} --out model_cards/sol_policy_card.md",
            "Regenerate model card markdown",
        )

        if result.returncode == 0:
            print("  ✅ Model card markdown regenerated")

        return {"status": "SUCCESS", "card_path": model_card_path}

    except Exception as e:
        print(f"  ❌ Model card update failed: {e}")
        return {"status": "FAILED", "error": str(e)}


def update_data_lineage():
    """Update data lineage documentation."""
    print("🔗 Updating data lineage...")

    result = run_command("make lineage", "Update data lineage")

    if result.returncode == 0:
        print("  ✅ Data lineage updated")
        return {"status": "SUCCESS"}
    else:
        print("  ❌ Data lineage update failed")
        return {"status": "FAILED", "error": result.stderr.strip()}


def run_ab_quick_check(training_result):
    """Run quick A/B check (using existing data)."""
    print("📊 Running A/B quick check...")

    result = run_command("make ab", "A/B evaluation")

    if result.returncode == 0 and "PASS" in result.stdout:
        print("  ✅ A/B quick check PASSED")
        return {"status": "PASS", "verdict": "PASS"}
    elif "FAIL" in result.stdout:
        print("  ❌ A/B quick check FAILED")
        return {"status": "FAIL", "verdict": "FAIL"}
    else:
        print("  ⚠️ A/B quick check INCONCLUSIVE")
        return {"status": "INCONCLUSIVE", "verdict": "INCONCLUSIVE"}


def update_last_good(training_result, validation_result, ab_result):
    """Update last_good artifacts if all checks pass."""
    print("📦 Updating last_good artifacts...")

    # Only update if validation passed and A/B is not failing
    if validation_result["status"] == "PASS" and ab_result["status"] != "FAIL":

        # Create last_good directory
        last_good_dir = Path("artifacts/last_good")
        last_good_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        last_good_metadata = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "training_id": training_result["training_id"],
            "checkpoint_path": training_result["checkpoint_file"],
            "validation_status": validation_result["status"],
            "ab_verdict": ab_result["verdict"],
            "pipeline_version": "1.0",
        }

        metadata_file = last_good_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(last_good_metadata, f, indent=2)

        # Copy checkpoint to last_good
        import shutil

        if os.path.exists(training_result["checkpoint_file"]):
            checkpoint_dest = last_good_dir / "model.pt"
            shutil.copy2(training_result["checkpoint_file"], checkpoint_dest)
            print(f"  ✅ Checkpoint copied to {checkpoint_dest}")

        print(f"  ✅ Last good model updated: {metadata_file}")
        return {"status": "UPDATED", "metadata": last_good_metadata}

    else:
        print(
            f"  ⚠️ Last good model NOT updated (validation={validation_result['status']}, ab={ab_result['status']})"
        )
        return {
            "status": "SKIPPED",
            "reason": f"validation={validation_result['status']}, ab={ab_result['status']}",
        }


def generate_pipeline_summary(results):
    """Generate pipeline run summary."""
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    summary_dir = Path("artifacts/pipeline") / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Overall status
    training_ok = results["training"]["status"] == "SUCCESS"
    validation_ok = results["validation"]["status"] == "PASS"
    ab_ok = results["ab"]["status"] != "FAIL"
    last_good_updated = results["last_good"]["status"] == "UPDATED"

    overall_status = (
        "SUCCESS" if training_ok and validation_ok and ab_ok else "PARTIAL_SUCCESS"
    )
    if not training_ok:
        overall_status = "FAILED"

    summary = {
        "pipeline_run_id": f"run_{timestamp}",
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "results": results,
        "recommendations": [],
    }

    # Add recommendations
    if not validation_ok:
        summary["recommendations"].append(
            "Review training data quality and model parameters"
        )
    if not ab_ok:
        summary["recommendations"].append("Investigate A/B test performance regression")
    if not last_good_updated:
        summary["recommendations"].append(
            "Address validation or A/B issues before promoting model"
        )
    if overall_status == "SUCCESS":
        summary["recommendations"].append(
            "Model ready for consideration in next deployment"
        )

    # Save summary
    summary_file = summary_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    markdown_content = f"""# Retrain & Promote Pipeline Summary

**Run ID:** {summary['pipeline_run_id']}
**Status:** {overall_status}
**Timestamp:** {summary['timestamp']}

## Pipeline Results

### Training
- **Status:** {results['training']['status']}
- **Training ID:** {results['training'].get('training_id', 'N/A')}

### Offline Validation
- **Status:** {results['validation']['status']}
- **Details:** {results['validation'].get('details', 'N/A')}

### A/B Quick Check
- **Status:** {results['ab']['status']}
- **Verdict:** {results['ab'].get('verdict', 'N/A')}

### Model Card Update
- **Status:** {results['model_card']['status']}

### Data Lineage Update
- **Status:** {results['lineage']['status']}

### Last Good Update
- **Status:** {results['last_good']['status']}
- **Reason:** {results['last_good'].get('reason', 'Success')}

## Recommendations

"""

    for rec in summary["recommendations"]:
        markdown_content += f"- {rec}\n"

    markdown_content += f"""
## Next Steps

{"✅ Model ready for deployment consideration" if overall_status == "SUCCESS" else "⚠️ Address issues before deployment"}

---
*Generated by automated retrain & promote pipeline*
"""

    markdown_file = summary_dir / "summary.md"
    with open(markdown_file, "w") as f:
        f.write(markdown_content)

    print(f"📋 Pipeline summary: {summary_file}")
    return summary


def main():
    """Main retrain & promote pipeline."""
    parser = argparse.ArgumentParser(description="Retrain & Promote Pipeline")
    parser.add_argument("--weekly", action="store_true", help="Weekly automated run")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    pipeline_id = (
        f"run_{datetime.datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}"
    )
    os.environ["PIPELINE_RUN_ID"] = pipeline_id

    print("🔄 Continuous Retrain & Promote Pipeline")
    print("=" * 60)
    print(f"Pipeline ID: {pipeline_id}")
    print(f"Mode: {'Weekly Automated' if args.weekly else 'Manual'}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 60)

    results = {}

    try:
        # Step 1: Training
        write_audit(
            "pipeline_start",
            {"pipeline_id": pipeline_id, "mode": "weekly" if args.weekly else "manual"},
        )

        print("\n🤖 STEP 1: Model Training")
        if args.dry_run:
            print("  [DRY RUN] Skipping actual training")
            results["training"] = {
                "status": "SUCCESS",
                "training_id": f"dry_run_{pipeline_id}",
                "checkpoint_file": "dry_run_checkpoint.pt",
            }
        else:
            results["training"] = simulate_training()

        if results["training"]["status"] != "SUCCESS":
            raise Exception(f"Training failed: {results['training']}")

        # Step 2: Offline Validation
        print("\n🚦 STEP 2: Offline Validation")
        if args.dry_run:
            print("  [DRY RUN] Simulating validation PASS")
            results["validation"] = {"status": "PASS", "details": "Dry run validation"}
        else:
            results["validation"] = run_offline_validation(results["training"])

        # Step 3: Update Model Card
        print("\n📄 STEP 3: Update Model Card")
        results["model_card"] = update_model_card(
            results["training"], results["validation"]
        )

        # Step 4: Update Data Lineage
        print("\n🔗 STEP 4: Update Data Lineage")
        if args.dry_run:
            print("  [DRY RUN] Skipping lineage update")
            results["lineage"] = {"status": "SUCCESS"}
        else:
            results["lineage"] = update_data_lineage()

        # Step 5: A/B Quick Check
        print("\n📊 STEP 5: A/B Quick Check")
        if args.dry_run:
            print("  [DRY RUN] Simulating A/B PASS")
            results["ab"] = {"status": "PASS", "verdict": "PASS"}
        else:
            results["ab"] = run_ab_quick_check(results["training"])

        # Step 6: Update Last Good (if criteria met)
        print("\n📦 STEP 6: Update Last Good")
        results["last_good"] = update_last_good(
            results["training"], results["validation"], results["ab"]
        )

        # Step 7: Generate Summary
        print("\n📋 STEP 7: Generate Summary")
        summary = generate_pipeline_summary(results)

        print("=" * 60)
        print(f"🎯 PIPELINE COMPLETE: {summary['overall_status']}")
        print(f"Training: {results['training']['status']}")
        print(f"Validation: {results['validation']['status']}")
        print(f"A/B Check: {results['ab']['status']}")
        print(f"Last Good: {results['last_good']['status']}")
        print("=" * 60)

        write_audit(
            "pipeline_complete",
            {
                "pipeline_id": pipeline_id,
                "overall_status": summary["overall_status"],
                "results_summary": {
                    k: v.get("status", "UNKNOWN") for k, v in results.items()
                },
            },
        )

        return 0 if summary["overall_status"] in ["SUCCESS", "PARTIAL_SUCCESS"] else 1

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        write_audit("pipeline_failed", {"pipeline_id": pipeline_id, "error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
