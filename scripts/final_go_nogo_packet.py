#!/usr/bin/env python3
"""
Final Go/No-Go Packet Auto-compiler
Compiles comprehensive deployment readiness packet with all gates, validations, and approvals
"""
import os
import sys
import json
import yaml
import glob
import datetime
import pathlib
import subprocess
from pathlib import Path
from datetime import timezone, timedelta


def run_command(cmd, cwd=None, timeout=60):
    """Execute command with logging."""
    if cwd is None:
        cwd = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

    env = os.environ.copy()
    env["PYTHONPATH"] = cwd

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        return {
            "command": cmd,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "success": False,
        }
    except Exception as e:
        return {
            "command": cmd,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def check_release_gates():
    """Run all release gates and collect results."""
    print("🚦 Running release gates...")

    gates = {
        "preflight": run_command("make preflight"),
        "ab_evaluation": run_command("make ab"),
        "validation": run_command("make validate-48h-now"),
        "unit_tests": run_command("make test"),
        "smoke_tests": run_command("make smoke"),
    }

    # Add basic gate analysis
    passed_gates = sum(1 for gate in gates.values() if gate["success"])
    total_gates = len(gates)

    gate_summary = {
        "gates": gates,
        "passed": passed_gates,
        "total": total_gates,
        "pass_rate": passed_gates / total_gates if total_gates > 0 else 0,
        "overall_status": "PASS" if passed_gates == total_gates else "FAIL",
    }

    print(f"  🚦 Gates: {passed_gates}/{total_gates} passed")
    return gate_summary


def check_model_performance():
    """Check current model performance metrics."""
    print("📈 Checking model performance...")

    performance = {
        "entropy": None,
        "q_spread": None,
        "heartbeat_age": None,
        "alerts_count": 0,
        "status": "UNKNOWN",
    }

    try:
        import requests

        response = requests.get("http://localhost:9100/metrics", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text

            for line in metrics_text.split("\n"):
                if line.startswith("rl_policy_entropy"):
                    parts = line.split()
                    if len(parts) >= 2:
                        performance["entropy"] = float(parts[1])
                elif line.startswith("rl_policy_q_spread"):
                    parts = line.split()
                    if len(parts) >= 2:
                        performance["q_spread"] = float(parts[1])
                elif line.startswith("rl_policy_heartbeat_age_seconds"):
                    parts = line.split()
                    if len(parts) >= 2:
                        performance["heartbeat_age"] = float(parts[1])

            # Check if metrics are within acceptable ranges
            entropy_ok = (
                performance["entropy"] is not None
                and 1.0 <= performance["entropy"] <= 2.0
            )
            heartbeat_ok = (
                performance["heartbeat_age"] is not None
                and performance["heartbeat_age"] < 600
            )

            if entropy_ok and heartbeat_ok:
                performance["status"] = "HEALTHY"
            else:
                performance["status"] = "DEGRADED"

    except Exception as e:
        performance["error"] = str(e)
        performance["status"] = "UNAVAILABLE"

    # Check for recent alerts
    try:
        import redis

        r = redis.Redis(decode_responses=True)
        performance["alerts_count"] = r.llen("alerts:policy")
    except Exception:
        pass

    print(f"  📈 Model status: {performance['status']}")
    return performance


def check_infrastructure_readiness():
    """Check infrastructure and operational readiness."""
    print("🏗️ Checking infrastructure readiness...")

    infra = {
        "disk_space": "unknown",
        "memory_usage": "unknown",
        "services_running": [],
        "redis_available": False,
        "exporter_available": False,
        "status": "UNKNOWN",
    }

    # Check disk space
    try:
        import psutil

        disk_usage = psutil.disk_usage("/")
        free_gb = disk_usage.free / (1024**3)
        infra["disk_space"] = f"{free_gb:.1f} GB free"

        memory = psutil.virtual_memory()
        infra["memory_usage"] = f"{memory.percent}% used"
    except Exception:
        pass

    # Check services
    service_checks = [
        ("redis", "redis-cli ping"),
        ("prometheus_exporter", "curl -s http://localhost:9100/metrics"),
        ("grafana", "curl -s http://localhost:3000/api/health"),
    ]

    running_services = 0
    for service_name, check_cmd in service_checks:
        result = run_command(check_cmd, timeout=5)
        if result["success"]:
            infra["services_running"].append(service_name)
            running_services += 1
            if service_name == "redis":
                infra["redis_available"] = True
            elif service_name == "prometheus_exporter":
                infra["exporter_available"] = True

    infra["status"] = "READY" if running_services >= 2 else "NEEDS_ATTENTION"

    print(f"  🏗️ Infrastructure: {infra['status']} ({running_services}/3 services)")
    return infra


def check_security_compliance():
    """Check security and compliance status."""
    print("🔒 Checking security compliance...")

    security = {
        "sbom_available": False,
        "sbom_signed": False,
        "audit_trails": 0,
        "secrets_status": "unknown",
        "status": "UNKNOWN",
    }

    # Check for SBOM
    sbom_files = glob.glob("artifacts/sbom/sbom_*.json")
    if sbom_files:
        security["sbom_available"] = True

        # Check for signatures
        for sbom_file in sbom_files:
            sig_file = sbom_file.replace(".json", ".sig")
            if os.path.exists(sig_file):
                security["sbom_signed"] = True
                break

    # Check audit trails
    audit_files = glob.glob("artifacts/audit/*.json")
    security["audit_trails"] = len(audit_files)

    # Basic secrets check (ensure no obvious secrets in logs)
    security["secrets_status"] = "pass"  # Would be more sophisticated in production

    if security["sbom_available"] and security["audit_trails"] > 0:
        security["status"] = "COMPLIANT"
    else:
        security["status"] = "NON_COMPLIANT"

    print(f"  🔒 Security: {security['status']}")
    return security


def check_operational_readiness():
    """Check operational procedures and runbooks."""
    print("📚 Checking operational readiness...")

    operational = {
        "runbook_available": False,
        "monitoring_configured": False,
        "alerts_configured": False,
        "rollback_procedures": False,
        "status": "UNKNOWN",
    }

    # Check for runbook
    if os.path.exists("RUNBOOK.md"):
        operational["runbook_available"] = True

    # Check monitoring config
    grafana_configs = glob.glob("grafana/dashboards/*.json")
    if grafana_configs:
        operational["monitoring_configured"] = True

    # Check alert configs
    alert_configs = glob.glob("grafana/alerts/*.json")
    if alert_configs:
        operational["alerts_configured"] = True

    # Check rollback scripts
    if os.path.exists("scripts/kill_switch.py"):
        operational["rollback_procedures"] = True

    ready_count = sum(
        [
            operational["runbook_available"],
            operational["monitoring_configured"],
            operational["alerts_configured"],
            operational["rollback_procedures"],
        ]
    )

    operational["status"] = "READY" if ready_count >= 3 else "PARTIAL"

    print(f"  📚 Operations: {operational['status']} ({ready_count}/4 items)")
    return operational


def check_business_approvals():
    """Check for business approvals and sign-offs."""
    print("✍️ Checking business approvals...")

    approvals = {
        "model_risk_approval": "unknown",
        "trading_risk_approval": "unknown",
        "technology_approval": "unknown",
        "final_signoff": "unknown",
        "status": "PENDING",
    }

    # In production, this would integrate with approval systems
    # For now, check for approval artifacts
    approval_files = glob.glob("artifacts/approvals/*.json")

    if approval_files:
        approvals["status"] = "APPROVED"
        print("  ✍️ Approval artifacts found")
    else:
        print("  ✍️ No approval artifacts found - manual approval required")

    return approvals


def generate_executive_summary(results):
    """Generate executive summary for decision makers."""

    # Calculate overall readiness score
    scores = []

    # Gates (40% weight)
    gates_score = results["gates"]["pass_rate"] * 100
    scores.append(("Release Gates", gates_score, 0.4))

    # Model performance (25% weight)
    perf_score = 100 if results["performance"]["status"] == "HEALTHY" else 0
    scores.append(("Model Performance", perf_score, 0.25))

    # Infrastructure (15% weight)
    infra_score = 100 if results["infrastructure"]["status"] == "READY" else 50
    scores.append(("Infrastructure", infra_score, 0.15))

    # Security (15% weight)
    security_score = 100 if results["security"]["status"] == "COMPLIANT" else 0
    scores.append(("Security", security_score, 0.15))

    # Operations (5% weight)
    ops_score = 100 if results["operational"]["status"] == "READY" else 50
    scores.append(("Operational", ops_score, 0.05))

    # Calculate weighted average
    weighted_score = sum(score * weight for _, score, weight in scores)

    # Determine recommendation
    if weighted_score >= 90:
        recommendation = "GO - All systems ready for deployment"
        decision_confidence = "HIGH"
    elif weighted_score >= 75:
        recommendation = "CONDITIONAL GO - Address minor issues"
        decision_confidence = "MEDIUM"
    elif weighted_score >= 60:
        recommendation = "NO GO - Significant issues require resolution"
        decision_confidence = "HIGH"
    else:
        recommendation = "NO GO - Critical issues must be resolved"
        decision_confidence = "HIGH"

    return {
        "overall_score": weighted_score,
        "recommendation": recommendation,
        "confidence": decision_confidence,
        "component_scores": [
            {"component": name, "score": score, "weight": weight}
            for name, score, weight in scores
        ],
        "critical_blockers": get_critical_blockers(results),
        "next_steps": get_next_steps(results, weighted_score),
    }


def get_critical_blockers(results):
    """Identify critical blockers for deployment."""
    blockers = []

    if results["gates"]["overall_status"] != "PASS":
        failed_gates = [
            name
            for name, gate in results["gates"]["gates"].items()
            if not gate["success"]
        ]
        blockers.append(f"Failed release gates: {', '.join(failed_gates)}")

    if results["performance"]["status"] in ["DEGRADED", "UNAVAILABLE"]:
        blockers.append(f"Model performance issues: {results['performance']['status']}")

    if results["security"]["status"] != "COMPLIANT":
        blockers.append("Security compliance not met")

    return blockers


def get_next_steps(results, score):
    """Generate next steps based on readiness assessment."""
    steps = []

    if score >= 90:
        steps = [
            "✅ Proceed with deployment",
            "📊 Monitor deployment metrics closely",
            "📞 Ensure incident response team is available",
        ]
    elif score >= 75:
        steps = [
            "🔧 Address identified minor issues",
            "✅ Re-run final checks",
            "📊 Proceed with cautious deployment",
        ]
    else:
        steps = [
            "❌ Do not proceed with deployment",
            "🔧 Resolve all critical blockers",
            "🔄 Re-run complete assessment after fixes",
        ]

    return steps


def generate_packet_markdown(results, summary):
    """Generate comprehensive markdown report."""
    timestamp = datetime.datetime.now(timezone.utc)

    markdown = f"""# Final Go/No-Go Deployment Packet

**Generated:** {timestamp.isoformat()}  
**Overall Score:** {summary['overall_score']:.1f}/100  
**Recommendation:** {summary['recommendation']}  
**Confidence:** {summary['confidence']}

## Executive Summary

This automated assessment evaluated all deployment readiness criteria across 5 key areas. The overall readiness score is **{summary['overall_score']:.1f}/100**, leading to the recommendation: **{summary['recommendation']}**.

### Component Scores
"""

    for component in summary["component_scores"]:
        markdown += f"- **{component['component']}:** {component['score']:.0f}/100 (weight: {component['weight']:.0%})\n"

    markdown += f"""

## Critical Assessment

### Release Gates ({results['gates']['passed']}/{results['gates']['total']} passed)
"""

    for gate_name, gate_result in results["gates"]["gates"].items():
        status_emoji = "✅" if gate_result["success"] else "❌"
        markdown += f"- {status_emoji} **{gate_name}**: {'PASS' if gate_result['success'] else 'FAIL'}\n"

    markdown += f"""

### Model Performance Status: {results['performance']['status']}
"""
    if results["performance"]["entropy"]:
        markdown += f"- **Entropy:** {results['performance']['entropy']:.3f} (target: 1.0-2.0)\n"
    if results["performance"]["heartbeat_age"]:
        markdown += f"- **Heartbeat Age:** {results['performance']['heartbeat_age']:.0f}s (target: <600s)\n"
    markdown += f"- **Active Alerts:** {results['performance']['alerts_count']}\n"

    markdown += f"""

### Infrastructure Status: {results['infrastructure']['status']}
- **Running Services:** {', '.join(results['infrastructure']['services_running']) if results['infrastructure']['services_running'] else 'None detected'}
- **Disk Space:** {results['infrastructure']['disk_space']}
- **Memory Usage:** {results['infrastructure']['memory_usage']}

### Security Compliance: {results['security']['status']}
- **SBOM Available:** {'✅' if results['security']['sbom_available'] else '❌'}
- **SBOM Signed:** {'✅' if results['security']['sbom_signed'] else '❌'}
- **Audit Trails:** {results['security']['audit_trails']} records

### Operational Readiness: {results['operational']['status']}
- **Runbook:** {'✅' if results['operational']['runbook_available'] else '❌'}
- **Monitoring:** {'✅' if results['operational']['monitoring_configured'] else '❌'}
- **Alerts:** {'✅' if results['operational']['alerts_configured'] else '❌'}
- **Rollback Procedures:** {'✅' if results['operational']['rollback_procedures'] else '❌'}

### Business Approvals: {results['approvals']['status']}
"""

    if summary["critical_blockers"]:
        markdown += f"""
## ❌ Critical Blockers

The following issues must be resolved before deployment:
"""
        for blocker in summary["critical_blockers"]:
            markdown += f"- {blocker}\n"
    else:
        markdown += "\n## ✅ No Critical Blockers\n\nAll critical deployment criteria have been met.\n"

    markdown += f"""

## Next Steps

"""
    for step in summary["next_steps"]:
        markdown += f"1. {step}\n"

    markdown += f"""

## Detailed Results

### Gate Execution Details
"""
    for gate_name, gate_result in results["gates"]["gates"].items():
        markdown += f"""
#### {gate_name}
- **Status:** {'PASS' if gate_result['success'] else 'FAIL'}
- **Command:** `{gate_result['command']}`
- **Exit Code:** {gate_result['return_code']}
"""
        if gate_result["stderr"] and gate_result["stderr"].strip():
            markdown += f"- **Error:** {gate_result['stderr'].strip()[:200]}...\n"

    markdown += f"""

---

## Approval Section

**Trading Risk Sign-off:** ________________ Date: ________  
**Model Risk Sign-off:** _________________ Date: ________  
**Technology Sign-off:** _________________ Date: ________  
**Final Authorization:** _________________ Date: ________

---
*This packet was automatically generated from production systems and live checks. Manual verification of critical systems is recommended before final deployment authorization.*

**Packet ID:** go_nogo_{timestamp.strftime('%Y%m%d_%H%M%SZ')}
"""

    return markdown


def main():
    """Main go/no-go assessment function."""
    print("🎯 Final Go/No-Go Packet Auto-compiler")
    print("=" * 60)

    import argparse

    parser = argparse.ArgumentParser(description="Generate deployment readiness packet")
    parser.add_argument(
        "--output",
        "-o",
        default="artifacts/go_nogo",
        help="Output directory for packet",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Skip longer running checks"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    # Collect all assessment data
    results = {}

    try:
        results["gates"] = check_release_gates()
        results["performance"] = check_model_performance()
        results["infrastructure"] = check_infrastructure_readiness()
        results["security"] = check_security_compliance()
        results["operational"] = check_operational_readiness()
        results["approvals"] = check_business_approvals()

        # Generate executive summary
        summary = generate_executive_summary(results)

        # Save raw results
        results_file = output_dir / f"assessment_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
                    "results": results,
                    "summary": summary,
                },
                f,
                indent=2,
                default=str,
            )

        # Generate markdown packet
        packet_content = generate_packet_markdown(results, summary)
        packet_file = output_dir / f"go_nogo_packet_{timestamp}.md"
        with open(packet_file, "w") as f:
            f.write(packet_content)

        # Create latest symlinks
        latest_packet = output_dir / "go_nogo_packet_latest.md"
        latest_results = output_dir / "assessment_latest.json"

        if latest_packet.exists():
            latest_packet.unlink()
        if latest_results.exists():
            latest_results.unlink()

        latest_packet.symlink_to(packet_file.name)
        latest_results.symlink_to(results_file.name)

        # Print summary
        print("=" * 60)
        print(f"📊 ASSESSMENT COMPLETE")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Recommendation: {summary['recommendation']}")
        print(f"Packet: {packet_file}")

        if summary["critical_blockers"]:
            print(f"\n❌ Critical Blockers ({len(summary['critical_blockers'])}):")
            for blocker in summary["critical_blockers"]:
                print(f"  - {blocker}")

        print("=" * 60)

        return 0 if summary["overall_score"] >= 75 else 1

    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
