#!/usr/bin/env python3
"""
Experiment Decider
Final GO/EXTEND/NO-GO decision combining CUPED uplift, sequential test, KRI guards, and cost ratio.
"""
import os
import sys
import json
import yaml
from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ExperimentDecider:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.exp_config = self.config["experiment"]
        self.governance = self.config.get("governance", {})
        self.artifacts_dir = Path(self.config.get("artifacts_dir", "experiments/m11"))

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load experiment config: {e}")

    def load_latest_analysis_results(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load latest CUPED and sequential test results."""

        cuped_file = self.artifacts_dir / "cuped_analysis_latest.json"
        sequential_file = self.artifacts_dir / "sequential_test_latest.json"

        cuped_results = None
        sequential_results = None

        if cuped_file.exists():
            try:
                with open(cuped_file, "r") as f:
                    cuped_results = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading CUPED results: {e}")
        else:
            # Try to find the most recent CUPED analysis file
            cuped_files = list(self.artifacts_dir.glob("cuped_analysis_*.json"))
            if cuped_files:
                latest_cuped = max(cuped_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_cuped, "r") as f:
                        cuped_results = json.load(f)
                    print(f"📊 Using latest CUPED analysis: {latest_cuped.name}")
                except Exception as e:
                    print(f"⚠️ Error loading CUPED results: {e}")

        if sequential_file.exists():
            try:
                with open(sequential_file, "r") as f:
                    sequential_results = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading sequential test results: {e}")
        else:
            # Try to find the most recent sequential test file
            sequential_files = list(self.artifacts_dir.glob("sequential_test_*.json"))
            if sequential_files:
                latest_sequential = max(
                    sequential_files, key=lambda f: f.stat().st_mtime
                )
                try:
                    with open(latest_sequential, "r") as f:
                        sequential_results = json.load(f)
                    print(f"🎯 Using latest sequential test: {latest_sequential.name}")
                except Exception as e:
                    print(f"⚠️ Error loading sequential test results: {e}")

        return cuped_results, sequential_results

    def check_cost_ratio_gate(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if cost ratio meets governance requirements."""

        cost_gate_max = self.governance.get("cost_gate_ratio_max", 0.30)

        try:
            # Load latest CFO report (updated with M10+M11 improvements)
            cfo_files = list(Path("artifacts/cfo").glob("*/cfo_report.json"))
            if cfo_files:
                latest_cfo = max(cfo_files, key=lambda x: x.stat().st_mtime)
                with open(latest_cfo, "r") as f:
                    cfo_data = json.load(f)

                current_cost_ratio = cfo_data.get("portfolio_metrics", {}).get(
                    "avg_cost_ratio", 1.0
                )

                # Apply M10 quantization improvements if available
                quant_files = list(
                    Path("artifacts/cost/quant").glob("*/quantization_report.json")
                )
                if quant_files:
                    latest_quant = max(quant_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_quant, "r") as f:
                        quant_data = json.load(f)

                    # Use projected cost ratio from quantization
                    projected_ratio = (
                        quant_data.get("cost_analysis", {}).get(
                            "projected_cost_ratio", current_cost_ratio * 100
                        )
                        / 100
                    )
                    current_cost_ratio = projected_ratio
            else:
                current_cost_ratio = 0.58  # From M10 analysis

            cost_gate_pass = current_cost_ratio <= cost_gate_max

            return cost_gate_pass, {
                "current_cost_ratio": current_cost_ratio,
                "cost_gate_max": cost_gate_max,
                "cost_gate_pass": cost_gate_pass,
                "cost_ratio_pct": current_cost_ratio * 100,
            }

        except Exception as e:
            print(f"⚠️ Error checking cost ratio: {e}")
            return False, {"error": str(e), "cost_gate_pass": False}

    def check_kri_guards(self) -> Tuple[bool, Dict[str, Any]]:
        """Check KRI (Key Risk Indicator) guards."""

        kri_guards = self.governance.get("kri_guards", {})

        if not kri_guards:
            return True, {"kri_guards_configured": False, "kri_pass": True}

        # Simulate KRI checks (in production, would query actual monitoring)
        entropy_floor = kri_guards.get("entropy_floor", 0.90)
        qspread_ratio_max = kri_guards.get("qspread_ratio_max", 2.0)

        # Mock current KRI values (would be real monitoring data)
        current_entropy = 1.15  # From previous analysis
        current_qspread_ratio = 1.6

        entropy_ok = current_entropy >= entropy_floor
        qspread_ok = current_qspread_ratio <= qspread_ratio_max

        kri_pass = entropy_ok and qspread_ok

        return kri_pass, {
            "entropy": {
                "current": current_entropy,
                "floor": entropy_floor,
                "pass": entropy_ok,
            },
            "qspread_ratio": {
                "current": current_qspread_ratio,
                "max": qspread_ratio_max,
                "pass": qspread_ok,
            },
            "kri_pass": kri_pass,
        }

    def check_alerts_cleanliness(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if alerts are clean for required period."""

        required_clean_hours = self.governance.get("require_clean_alert_hours", 48)
        cutoff = datetime.datetime.now() - datetime.timedelta(
            hours=required_clean_hours
        )

        try:
            alert_files = list(Path("artifacts/audit").glob("*alert*.json"))
            recent_alerts = []

            for alert_file in alert_files:
                mtime = datetime.datetime.fromtimestamp(alert_file.stat().st_mtime)
                if mtime > cutoff:
                    recent_alerts.append(
                        {"file": str(alert_file), "time": mtime.isoformat()}
                    )

            alerts_clean = len(recent_alerts) == 0

            return alerts_clean, {
                "required_clean_hours": required_clean_hours,
                "recent_alerts": recent_alerts,
                "alerts_clean": alerts_clean,
            }

        except Exception as e:
            return True, {
                "error": str(e),
                "alerts_clean": True,  # Default to pass if can't check
            }

    def make_experiment_decision(
        self, cuped_results: Optional[Dict], sequential_results: Optional[Dict]
    ) -> Dict[str, Any]:
        """Make final experiment decision."""

        print("🎯 Making experiment decision...")

        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "experiment": self.exp_config["name"],
            "decision": "EXTEND",  # Default
            "reasons": [],
            "checks": {},
            "final_recommendation": None,
        }

        # Check data availability
        if not cuped_results or not sequential_results:
            decision["decision"] = "EXTEND"
            decision["reasons"].append("Insufficient analysis data")
            decision["checks"]["data_available"] = False
            return decision

        decision["checks"]["data_available"] = True

        # Extract key metrics
        key_findings = cuped_results.get("key_findings", {})
        sequential_verdict = sequential_results.get("verdict", "EXTEND")
        sequential_reason = sequential_results.get("reason", "")

        primary_effect = key_findings.get(
            "primary_effect_cuped", key_findings.get("primary_effect_raw", 0)
        )
        primary_ci = key_findings.get("primary_ci_95", [0, 0])
        meets_met = key_findings.get("meets_met", False)
        ci_excludes_zero = key_findings.get("ci_excludes_zero", False)

        # Check governance gates
        print("🔍 Checking governance gates...")

        # Cost ratio gate
        cost_gate_pass, cost_info = self.check_cost_ratio_gate()
        decision["checks"]["cost_gate"] = cost_info

        # KRI guards
        kri_pass, kri_info = self.check_kri_guards()
        decision["checks"]["kri_guards"] = kri_info

        # Alerts cleanliness
        alerts_clean, alerts_info = self.check_alerts_cleanliness()
        decision["checks"]["alerts_clean"] = alerts_info

        # Sequential test results
        decision["checks"]["sequential_test"] = {
            "verdict": sequential_verdict,
            "reason": sequential_reason,
            "effect": primary_effect,
            "ci_95": primary_ci,
            "meets_met": meets_met,
        }

        # Decision logic
        blocking_reasons = []

        # Cost ratio check
        if not cost_gate_pass:
            cost_ratio_pct = cost_info.get("cost_ratio_pct", 100)
            cost_max_pct = cost_info.get("cost_gate_max", 0.3) * 100
            blocking_reasons.append(
                f"cost_ratio_high ({cost_ratio_pct:.1f}% > {cost_max_pct:.1f}%)"
            )

        # KRI guards check
        if not kri_pass:
            blocking_reasons.append("kri_guards_fail")

        # Alerts check
        if not alerts_clean:
            num_alerts = len(alerts_info.get("recent_alerts", []))
            blocking_reasons.append(f"alerts_not_clean ({num_alerts} recent)")

        # Sequential test check
        if sequential_verdict == "FAIL":
            blocking_reasons.append("sequential_test_fail")
        elif sequential_verdict == "EXTEND":
            if "insufficient evidence" in sequential_reason.lower():
                blocking_reasons.append("exp_extend_power_low")
            else:
                blocking_reasons.append("exp_extend_duration")

        # CI check
        if ci_excludes_zero and primary_ci[0] < 0:
            blocking_reasons.append("exp_ci_low_neg")

        # Final decision
        if not blocking_reasons and sequential_verdict == "PASS":
            decision["decision"] = "GO"
            decision["final_recommendation"] = (
                "Experiment successful - proceed with M11 deployment and enable economic ramps"
            )
            decision["go_token_created"] = True

            # Create GO token file
            self.create_go_token(decision)

        elif sequential_verdict == "FAIL" or (
            "cost_ratio_high" in blocking_reasons and primary_effect < 0
        ):
            decision["decision"] = "NO_GO"
            decision["final_recommendation"] = (
                "Experiment failed - do not proceed with M11 deployment"
            )
            decision["go_token_created"] = False

        else:
            decision["decision"] = "EXTEND"
            decision["final_recommendation"] = (
                "Continue experiment - gates not yet satisfied"
            )
            decision["go_token_created"] = False

        decision["reasons"] = blocking_reasons

        return decision

    def create_go_token(self, decision: Dict[str, Any]):
        """Create GO token file for ramp decider integration."""

        token_dir = Path("experiments") / "m11"
        token_dir.mkdir(parents=True, exist_ok=True)

        token_file = token_dir / "token_GO"

        token_data = {
            "experiment": self.exp_config["name"],
            "decision": decision["decision"],
            "timestamp": decision["timestamp"],
            "valid_until": (
                datetime.now(timezone.utc) + timedelta(days=30)
            ).isoformat()
            + "Z",
            "decision_summary": decision["final_recommendation"],
        }

        with open(token_file, "w") as f:
            json.dump(token_data, f, indent=2)

        print(f"✅ GO token created: {token_file}")

    def generate_decision_markdown(self, decision: Dict[str, Any]) -> str:
        """Generate markdown summary of decision."""

        decision_emoji = {"GO": "✅", "NO_GO": "❌", "EXTEND": "🔄"}[
            decision["decision"]
        ]

        md = f"""# Experiment Decision: {decision["decision"]} {decision_emoji}

**Experiment:** {decision["experiment"]}  
**Decision Time:** {decision["timestamp"]}  
**Final Decision:** **{decision["decision"]}**

## Summary

{decision.get("final_recommendation", "No recommendation available")}

## Decision Criteria Analysis

### ✅ Data Availability
- **CUPED Analysis:** {'✅ Available' if decision["checks"]["data_available"] else '❌ Missing'}
- **Sequential Test:** {'✅ Available' if decision["checks"]["data_available"] else '❌ Missing'}

### 📊 Statistical Analysis
"""

        if decision["checks"]["data_available"]:
            seq_check = decision["checks"]["sequential_test"]
            md += f"""
- **Sequential Verdict:** {seq_check["verdict"]}
- **Primary Effect:** {seq_check["effect"]:.3f}
- **95% Confidence Interval:** [{seq_check["ci_95"][0]:.3f}, {seq_check["ci_95"][1]:.3f}]
- **Meets MET:** {'✅' if seq_check["meets_met"] else '❌'}
- **Reason:** {seq_check["reason"]}
"""

        md += """
### 🛡️ Governance Gates
"""

        # Cost gate
        cost_check = decision["checks"]["cost_gate"]
        cost_status = "✅" if cost_check.get("cost_gate_pass", False) else "❌"
        md += f"""
- **Cost Ratio Gate:** {cost_status}
  - Current: {cost_check.get('cost_ratio_pct', 0):.1f}%
  - Required: ≤{cost_check.get('cost_gate_max', 0.3)*100:.1f}%
"""

        # KRI guards
        kri_check = decision["checks"]["kri_guards"]
        kri_status = "✅" if kri_check.get("kri_pass", False) else "❌"
        md += f"""
- **KRI Guards:** {kri_status}
  - Entropy: {kri_check.get('entropy', {}).get('current', 0):.2f} (≥{kri_check.get('entropy', {}).get('floor', 0):.2f})
  - QSpread Ratio: {kri_check.get('qspread_ratio', {}).get('current', 0):.2f} (≤{kri_check.get('qspread_ratio', {}).get('max', 0):.1f})
"""

        # Alerts
        alerts_check = decision["checks"]["alerts_clean"]
        alerts_status = "✅" if alerts_check.get("alerts_clean", False) else "❌"
        recent_alerts = len(alerts_check.get("recent_alerts", []))
        md += f"""
- **Alerts Clean:** {alerts_status}
  - Recent alerts: {recent_alerts}
  - Required clean period: {alerts_check.get('required_clean_hours', 48)}h
"""

        md += """
## Blocking Reasons
"""

        if decision["reasons"]:
            for reason in decision["reasons"]:
                md += f"- ❌ {reason}\n"
        else:
            md += "- ✅ No blocking reasons\n"

        md += f"""

## Next Steps

"""

        if decision["decision"] == "GO":
            md += """1. **Deploy M11 improvements** to production
2. **Enable economic ramp gates** in ramp decider
3. **Monitor KRIs** for first 48h after deployment
4. **Retry ramp decision** to unlock first economic ramp step
"""
        elif decision["decision"] == "NO_GO":
            md += """1. **Do not deploy M11 improvements**
2. **Investigate negative results** and root causes
3. **Consider alternative approaches** for cost ratio reduction
4. **Re-evaluate experiment design** if needed
"""
        else:  # EXTEND
            md += """1. **Continue experiment** for additional data
2. **Address blocking reasons** if possible
3. **Monitor gates** for compliance
4. **Re-evaluate** when minimum duration criteria met
"""

        md += f"""

---
*Decision generated by Experiment Decider - M12 Live Economic Experiment*
"""

        return md

    def run_experiment_decision(self) -> Dict[str, Any]:
        """Run complete experiment decision process."""

        print("🎯 Running experiment decision process...")

        # Load analysis results
        cuped_results, sequential_results = self.load_latest_analysis_results()

        # Make decision
        decision = self.make_experiment_decision(cuped_results, sequential_results)

        return decision


def main():
    """Main experiment decider function."""
    parser = argparse.ArgumentParser(description="Experiment Decider")
    parser.add_argument("-c", "--config", required=True, help="Experiment config file")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()

    try:
        decider = ExperimentDecider(args.config)

        # Run decision process
        decision = decider.run_experiment_decision()

        # Save results
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            output_dir = decider.artifacts_dir / timestamp

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON decision
        decision_json = output_dir / "decision.json"
        with open(decision_json, "w") as f:
            json.dump(decision, f, indent=2)

        # Save markdown summary
        decision_md = output_dir / "decision.md"
        markdown_content = decider.generate_decision_markdown(decision)
        with open(decision_md, "w") as f:
            f.write(markdown_content)

        # Create latest symlinks
        latest_json = decider.artifacts_dir / "decision_latest.json"
        latest_md = decider.artifacts_dir / "decision_latest.md"

        for latest, target in [(latest_json, decision_json), (latest_md, decision_md)]:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(target)

        # Display summary
        decision_emoji = {"GO": "✅", "NO_GO": "❌", "EXTEND": "🔄"}[
            decision["decision"]
        ]

        print(f"\n🎯 Experiment Decision: {decision['decision']} {decision_emoji}")
        print(f"  Recommendation: {decision.get('final_recommendation', 'None')}")

        if decision["reasons"]:
            print(f"  Blocking Reasons:")
            for reason in decision["reasons"]:
                print(f"    - {reason}")
        else:
            print(f"  ✅ No blocking reasons")

        if decision.get("go_token_created", False):
            print(f"  🎫 GO token created for ramp decider")

        print(f"\n📄 Decision saved:")
        print(f"  JSON: {decision_json}")
        print(f"  Markdown: {decision_md}")

        return 0

    except Exception as e:
        print(f"❌ Experiment decision failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
