#!/usr/bin/env python3
"""
Model Risk Management (MRM) Gate
Verifies all required evidence exists and model card completeness
"""
import argparse
import json
import yaml
import sys
import os
import pathlib
import glob
from datetime import datetime

REQ = ["training", "evaluation", "risk", "governance"]


def check_model_card_completeness(card_data):
    """Check model card for required sections and completeness."""
    issues = []
    warnings = []

    # Check required top-level sections
    missing_sections = [section for section in REQ if section not in card_data]
    if missing_sections:
        issues.append(f"Missing required sections: {missing_sections}")

    # Check training section completeness
    if "training" in card_data:
        training = card_data["training"]
        required_training_fields = ["algo", "data_window", "env"]
        missing_training = [
            field for field in required_training_fields if field not in training
        ]
        if missing_training:
            issues.append(f"Training section missing fields: {missing_training}")

    # Check evaluation section completeness
    if "evaluation" in card_data:
        evaluation = card_data["evaluation"]
        if "offline_gate" not in evaluation:
            issues.append("Evaluation section missing offline_gate results")
        if "ab_test" not in evaluation:
            warnings.append("Evaluation section missing A/B test results")
        elif evaluation["ab_test"].get("verdict") == "PENDING":
            warnings.append(
                "A/B test results pending - not blocking but should be completed"
            )

    # Check risk section completeness
    if "risk" in card_data:
        risk = card_data["risk"]
        required_risk_fields = [
            "entropy_floor",
            "drawdown_day_max",
            "kill_switch_triggers",
        ]
        missing_risk = [field for field in required_risk_fields if field not in risk]
        if missing_risk:
            issues.append(f"Risk section missing fields: {missing_risk}")

        # Validate kill switch triggers
        if "kill_switch_triggers" in risk and len(risk["kill_switch_triggers"]) == 0:
            issues.append("Risk section has empty kill_switch_triggers list")

    # Check governance section completeness
    if "governance" in card_data:
        governance = card_data["governance"]
        required_gov_fields = ["worm_audit_paths", "controls", "monitoring"]
        missing_gov = [
            field for field in required_gov_fields if field not in governance
        ]
        if missing_gov:
            issues.append(f"Governance section missing fields: {missing_gov}")

        # Check audit paths exist
        if "worm_audit_paths" in governance:
            for path in governance["worm_audit_paths"]:
                if not os.path.exists(path):
                    warnings.append(f"Audit path does not exist: {path}")

    return issues, warnings


def check_artifact_evidence():
    """Check for required evidence artifacts."""
    evidence_checks = []

    # Check for validation artifacts
    validation_files = glob.glob("artifacts/*/validation_*.json")
    if validation_files:
        evidence_checks.append("✅ Validation artifacts found")

        # Check recent validations
        recent_validations = validation_files[-3:]  # Last 3
        pass_count = 0
        for file in recent_validations:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "PASS":
                        pass_count += 1
            except Exception:
                continue

        if pass_count >= 2:
            evidence_checks.append("✅ Recent validation passes found")
        else:
            evidence_checks.append("⚠️ Insufficient recent validation passes")
    else:
        evidence_checks.append("❌ No validation artifacts found")

    # Check for audit trail
    audit_files = glob.glob("artifacts/audit/*.json")
    if audit_files:
        evidence_checks.append("✅ Audit trail artifacts found")
    else:
        evidence_checks.append("❌ No audit trail found")

    # Check for pilot artifacts
    pilot_files = glob.glob("artifacts/pilot/*")
    if pilot_files:
        evidence_checks.append("✅ Pilot artifacts found")
    else:
        evidence_checks.append("⚠️ No pilot artifacts found")

    # Check for postmortem
    postmortem_files = glob.glob("artifacts/pilot/*/postmortem.json")
    if postmortem_files:
        evidence_checks.append("✅ Postmortem analysis found")
    else:
        evidence_checks.append("⚠️ No postmortem analysis found")

    # Check for A/B test results
    ab_files = glob.glob("artifacts/ab/*/ab_results.json")
    if ab_files:
        evidence_checks.append("✅ A/B test results found")
    else:
        evidence_checks.append("⚠️ No A/B test results found")

    return evidence_checks


def check_operational_readiness():
    """Check operational readiness indicators."""
    readiness_checks = []

    # Check for kill switch script
    kill_switch_path = "scripts/kill_switch.py"
    if os.path.exists(kill_switch_path):
        readiness_checks.append("✅ Kill switch script available")
    else:
        readiness_checks.append("❌ Kill switch script not found")

    # Check for pilot guard
    pilot_guard_path = "scripts/pilot_guard.py"
    if os.path.exists(pilot_guard_path):
        readiness_checks.append("✅ Pilot guard script available")
    else:
        readiness_checks.append("❌ Pilot guard script not found")

    # Check for monitoring configuration
    systemd_files = glob.glob("systemd/*.service") + glob.glob("systemd/*.timer")
    if systemd_files:
        readiness_checks.append("✅ Monitoring services configured")
    else:
        readiness_checks.append("⚠️ No monitoring services found")

    # Check for runbook
    runbook_path = "RUNBOOK.md"
    if os.path.exists(runbook_path):
        readiness_checks.append("✅ Operations runbook available")
    else:
        readiness_checks.append("⚠️ Operations runbook not found")

    return readiness_checks


def generate_mrm_checklist(
    card_data, evidence_checks, readiness_checks, issues, warnings
):
    """Generate MRM checklist report."""
    checklist = f"""# Model Risk Management (MRM) Checklist

**Generated:** {datetime.now().isoformat()}
**Model:** {card_data.get('model', 'N/A')} v{card_data.get('version', 'N/A')}

## Model Card Completeness

### Required Sections
"""

    for section in REQ:
        status = "✅" if section in card_data else "❌"
        checklist += f"- {status} {section.title()}\n"

    checklist += f"""
### Issues Found
"""
    if issues:
        for issue in issues:
            checklist += f"- ❌ {issue}\n"
    else:
        checklist += "- ✅ No critical issues found\n"

    checklist += f"""
### Warnings
"""
    if warnings:
        for warning in warnings:
            checklist += f"- ⚠️ {warning}\n"
    else:
        checklist += "- ✅ No warnings\n"

    checklist += f"""
## Evidence Artifacts
"""
    for check in evidence_checks:
        checklist += f"- {check}\n"

    checklist += f"""
## Operational Readiness
"""
    for check in readiness_checks:
        checklist += f"- {check}\n"

    # Overall assessment
    critical_issues = len(
        [issue for issue in issues if "Missing required sections" in issue]
    )
    missing_evidence = len([check for check in evidence_checks if "❌" in check])
    missing_ops = len([check for check in readiness_checks if "❌" in check])

    if critical_issues > 0:
        overall_status = "❌ FAIL - Critical model card issues"
    elif missing_evidence > 0:
        overall_status = "❌ FAIL - Missing required evidence"
    elif missing_ops > 0:
        overall_status = "❌ FAIL - Operational readiness issues"
    elif warnings:
        overall_status = "⚠️ CONDITIONAL PASS - Address warnings before production"
    else:
        overall_status = "✅ PASS - All requirements met"

    checklist += f"""
## Overall Assessment

**Status:** {overall_status}

### Summary
- **Critical Issues:** {critical_issues}
- **Missing Evidence:** {missing_evidence} 
- **Operational Issues:** {missing_ops}
- **Warnings:** {len(warnings)}

### Recommendations
"""

    if critical_issues > 0:
        checklist += "- Complete missing model card sections before proceeding\n"
    if missing_evidence > 0:
        checklist += "- Generate missing evidence artifacts\n"
    if missing_ops > 0:
        checklist += "- Deploy missing operational components\n"
    if warnings:
        checklist += "- Address warnings to improve model risk posture\n"
    if critical_issues == 0 and missing_evidence == 0 and missing_ops == 0:
        checklist += "- Model ready for production deployment\n"

    return checklist, overall_status


def main():
    """Main MRM gate check function."""
    ap = argparse.ArgumentParser(description="MRM gate check for model deployment")
    ap.add_argument("--card", required=True, help="Model card YAML file")
    ap.add_argument("--checklist", required=True, help="Output checklist file")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args()

    print(f"Running MRM gate check on: {args.card}")

    try:
        with open(args.card) as f:
            card_data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load model card: {e}")
        sys.exit(1)

    # Run all checks
    print("Checking model card completeness...")
    issues, warnings = check_model_card_completeness(card_data)

    print("Checking evidence artifacts...")
    evidence_checks = check_artifact_evidence()

    print("Checking operational readiness...")
    readiness_checks = check_operational_readiness()

    # Generate checklist
    print("Generating MRM checklist...")
    checklist_content, overall_status = generate_mrm_checklist(
        card_data, evidence_checks, readiness_checks, issues, warnings
    )

    # Write checklist
    pathlib.Path(args.checklist).parent.mkdir(parents=True, exist_ok=True)
    with open(args.checklist, "w") as f:
        f.write(checklist_content)

    print(f"✅ MRM checklist generated: {args.checklist}")

    # Determine exit status
    has_critical_issues = len(issues) > 0
    has_missing_evidence = any("❌" in check for check in evidence_checks)
    has_missing_ops = any("❌" in check for check in readiness_checks)

    if has_critical_issues or has_missing_evidence or has_missing_ops:
        print(f"❌ MRM_FAIL: {overall_status}")
        sys.exit(1)
    elif warnings and args.strict:
        print(f"⚠️ MRM_FAIL (strict mode): {overall_status}")
        sys.exit(1)
    else:
        print(f"✅ MRM_PASS: {overall_status}")
        sys.exit(0)


if __name__ == "__main__":
    main()
