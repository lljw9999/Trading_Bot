#!/usr/bin/env python3
"""
Go-Live Day Orchestrator
Re-run gates → set influence to 10% (guarded) → start 2-hour canary watch → auto-rollback on breach
"""
import os
import subprocess
import time
import json
import pathlib
import datetime
import sys
from datetime import timezone

# Artifact directory for this go-live session
ART = (
    f"artifacts/golive/{datetime.datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}"
)
pathlib.Path(ART).mkdir(parents=True, exist_ok=True)


def run(cmd):
    """Execute shell command with capture."""
    print(f"[EXEC] {cmd}")
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        cwd="/Users/yanzewu/PycharmProjects/NLP_Final_Project_D",
    )


def write_audit(kind, payload):
    """Write WORM audit record."""
    ts = datetime.datetime.now(timezone.utc).isoformat().replace(":", "_")
    pathlib.Path("artifacts/audit").mkdir(parents=True, exist_ok=True)
    audit_file = f"artifacts/audit/{ts}_{kind}.json"

    audit_record = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "action": kind,
        "payload": payload,
        "operator": os.getenv("USER", "unknown"),
        "session_id": os.path.basename(ART),
    }

    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2)

    print(f"[AUDIT] {audit_file}")
    return audit_file


def check_prerequisites():
    """Check GO_LIVE flag and basic prerequisites."""
    print("🔒 Checking prerequisites...")

    if os.getenv("GO_LIVE", "0") != "1":
        print("❌ NO_GO: GO_LIVE environment variable not set to 1")
        print("   Set it with: export GO_LIVE=1")
        write_audit("go_live_blocked", {"reason": "GO_LIVE_flag_not_set"})
        sys.exit(1)

    print("✅ GO_LIVE flag is set")
    write_audit("prerequisites_check", {"go_live_flag": "set", "status": "PASS"})


def run_release_gates():
    """Re-run all release gates before deployment."""
    print("🚦 Running release gates...")

    gate_commands = ["make preflight", "make go-nogo"]

    for cmd in gate_commands:
        print(f"  Running: {cmd}")
        r = run(cmd)

        # Save gate output
        gate_log = f"{ART}/{cmd.replace(' ', '_').replace('make', '')}.log"
        with open(gate_log, "w") as f:
            f.write(f"Command: {cmd}\n")
            f.write(f"Return code: {r.returncode}\n")
            f.write(f"STDOUT:\n{r.stdout}\n")
            f.write(f"STDERR:\n{r.stderr}\n")

        if r.returncode != 0:
            print(f"❌ NO_GO: {cmd} failed (exit code: {r.returncode})")
            write_audit(
                "gate_failure",
                {
                    "command": cmd,
                    "return_code": r.returncode,
                    "stdout": r.stdout[:500],  # Truncate for audit
                    "stderr": r.stderr[:500],
                },
            )
            sys.exit(1)

        print(f"  ✅ {cmd} passed")

    write_audit("release_gates_passed", {"gates": gate_commands, "status": "ALL_PASS"})
    print("✅ All release gates passed")


def execute_guarded_ramp(target_pct):
    """Execute guarded ramp to target percentage."""
    print(f"📈 Executing guarded ramp to {target_pct}%...")

    # Set environment for pilot guard
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"
    env["REASON"] = f"Go-Live Day canary deployment to {target_pct}%"

    ramp_cmd = f"python scripts/pilot_guard.py --target-pct {target_pct}"
    r = subprocess.run(
        ramp_cmd,
        shell=True,
        text=True,
        capture_output=True,
        env=env,
        cwd="/Users/yanzewu/PycharmProjects/NLP_Final_Project_D",
    )

    # Save pilot guard output
    guard_log = f"{ART}/pilot_guard.log"
    with open(guard_log, "w") as f:
        f.write(f"Command: {ramp_cmd}\n")
        f.write(f"Return code: {r.returncode}\n")
        f.write(f"STDOUT:\n{r.stdout}\n")
        f.write(f"STDERR:\n{r.stderr}\n")

    if r.returncode != 0:
        print(f"❌ NO_GO: Pilot guard blocked ramp to {target_pct}%")
        print(f"   Reason: {r.stdout.strip()}")
        write_audit(
            "ramp_blocked",
            {
                "target_pct": target_pct,
                "reason": r.stdout.strip(),
                "guard_exit_code": r.returncode,
            },
        )
        sys.exit(1)

    print(f"✅ Successfully ramped to {target_pct}% influence")
    write_audit("go_live_ramp", {"pct": target_pct, "status": "SUCCESS"})


def run_canary_watch(hold_minutes, target_pct):
    """Run 2-hour canary watch with auto-rollback on breach."""
    print(f"👁️ Starting {hold_minutes}-minute canary watch...")

    start_time = time.time()
    breaches = 0
    kri_samples = 0
    snapshots = []

    # Create KRI monitoring log
    kri_log_file = f"{ART}/kri_tail.log"

    while time.time() - start_time < hold_minutes * 60:
        elapsed_minutes = int((time.time() - start_time) / 60)
        remaining_minutes = hold_minutes - elapsed_minutes

        print(
            f"  📊 Canary watch: {elapsed_minutes}m elapsed, {remaining_minutes}m remaining..."
        )

        # Run KRI monitor
        env = os.environ.copy()
        env["PYTHONPATH"] = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

        kri_result = subprocess.run(
            "python scripts/pilot_kri_monitor.py",
            shell=True,
            text=True,
            capture_output=True,
            env=env,
            cwd="/Users/yanzewu/PycharmProjects/NLP_Final_Project_D",
        )

        # Append to KRI log
        with open(kri_log_file, "a") as f:
            f.write(f"\n--- Sample {kri_samples + 1} at {elapsed_minutes}m ---\n")
            f.write(f"Return code: {kri_result.returncode}\n")
            f.write(f"STDOUT: {kri_result.stdout}\n")
            f.write(f"STDERR: {kri_result.stderr}\n")

        # Check for auto-rollback trigger
        if "AUTO_ROLLBACK" in kri_result.stdout.upper():
            breaches += 1
            print(f"🚨 KRI BREACH DETECTED! Auto-rollback triggered.")
            write_audit(
                "canary_breach",
                {
                    "breach_number": breaches,
                    "elapsed_minutes": elapsed_minutes,
                    "kri_output": kri_result.stdout,
                    "action": "AUTO_ROLLBACK",
                },
            )
            break

        # Record snapshot
        snapshot = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "elapsed_minutes": elapsed_minutes,
            "kri_status": "OK" if "KRI OK" in kri_result.stdout else "UNKNOWN",
            "kri_output": kri_result.stdout.strip(),
            "target_pct": target_pct,
        }
        snapshots.append(snapshot)

        kri_samples += 1

        # Wait 1 minute between samples (or exit if time is up)
        if remaining_minutes > 0:
            time.sleep(60)

    # Determine final status
    total_elapsed = int((time.time() - start_time) / 60)

    if breaches == 0:
        status = "PASS"
        print(f"✅ Canary watch completed successfully after {total_elapsed} minutes")
        print(f"   No KRI breaches detected in {kri_samples} samples")
    else:
        status = "ROLLBACK"
        print(f"🚨 Canary watch failed - {breaches} KRI breaches detected")
        # Execute emergency rollback
        execute_emergency_rollback()

    # Save snapshots
    snapshots_file = f"{ART}/snapshots.jsonl"
    with open(snapshots_file, "w") as f:
        for snapshot in snapshots:
            f.write(json.dumps(snapshot) + "\n")

    # Write summary
    summary_content = f"""# Go-Live Canary Watch Summary

**Session ID:** {os.path.basename(ART)}
**Status:** {status}
**Target Influence:** {target_pct}%
**Watch Duration:** {total_elapsed} minutes (planned: {hold_minutes} minutes)
**KRI Samples:** {kri_samples}
**Breaches Detected:** {breaches}

## Timeline
- **Start:** {datetime.datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()}
- **End:** {datetime.datetime.now(timezone.utc).isoformat()}
- **Elapsed:** {total_elapsed} minutes

## Results
{"✅ CANARY PASSED - No KRI breaches detected" if status == "PASS" else "🚨 CANARY FAILED - KRI breach triggered auto-rollback"}

## Next Steps
{"- Consider proceeding to next ramp phase" if status == "PASS" else "- Investigate root cause of KRI breach"}
- Review detailed logs in artifacts/golive/{os.path.basename(ART)}/
- Update stakeholders on canary results
"""

    summary_file = f"{ART}/summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_content)

    write_audit(
        "canary_watch_complete",
        {
            "status": status,
            "duration_minutes": total_elapsed,
            "kri_samples": kri_samples,
            "breaches": breaches,
            "target_pct": target_pct,
        },
    )

    return status


def execute_emergency_rollback():
    """Execute emergency rollback to 0% influence."""
    print("🚨 EXECUTING EMERGENCY ROLLBACK...")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/yanzewu/PycharmProjects/NLP_Final_Project_D"

    rollback_result = subprocess.run(
        "python scripts/kill_switch.py",
        shell=True,
        text=True,
        capture_output=True,
        env=env,
        cwd="/Users/yanzewu/PycharmProjects/NLP_Final_Project_D",
    )

    if rollback_result.returncode == 0:
        print("✅ Emergency rollback completed - influence set to 0%")
    else:
        print("❌ Emergency rollback failed!")
        print(f"   Output: {rollback_result.stdout}")
        print(f"   Error: {rollback_result.stderr}")

    write_audit(
        "emergency_rollback",
        {
            "trigger": "canary_breach",
            "rollback_exit_code": rollback_result.returncode,
            "rollback_output": rollback_result.stdout,
        },
    )


def main():
    """Main go-live day orchestration."""
    print("🚀 Go-Live Day Orchestrator")
    print("=" * 50)

    import argparse

    ap = argparse.ArgumentParser(description="Execute Go-Live Day with canary watch")
    ap.add_argument("--pct", type=int, default=10, help="Target influence percentage")
    ap.add_argument(
        "--hold-min", type=int, default=120, help="Canary watch duration in minutes"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (shorter watch)"
    )
    args = ap.parse_args()

    # Adjust for dry-run
    if args.dry_run:
        args.hold_min = 1
        print("🧪 DRY RUN MODE - Watch duration set to 1 minute")

    session_start = datetime.datetime.now(timezone.utc)
    print(f"Session: {os.path.basename(ART)}")
    print(f"Target: {args.pct}% influence")
    print(f"Watch: {args.hold_min} minutes")
    print(f"Started: {session_start.isoformat()}")
    print("=" * 50)

    try:
        # Step 1: Check prerequisites
        check_prerequisites()

        # Step 2: Run release gates
        run_release_gates()

        # Step 3: Execute guarded ramp
        execute_guarded_ramp(args.pct)

        # Step 4: Run canary watch
        status = run_canary_watch(args.hold_min, args.pct)

        # Step 5: Final status
        session_end = datetime.datetime.now(timezone.utc)
        duration = (session_end - session_start).total_seconds() / 60

        print("=" * 50)
        print(f"🎯 GO-LIVE DAY COMPLETE")
        print(f"Status: {status}")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Artifacts: {ART}")
        print("=" * 50)

        write_audit(
            "go_live_day_complete",
            {
                "final_status": status,
                "duration_minutes": duration,
                "target_pct": args.pct,
                "artifacts_dir": ART,
            },
        )

        print(status)  # For script return value parsing
        return 0 if status == "PASS" else 1

    except KeyboardInterrupt:
        print("\n🛑 Go-Live Day interrupted by user")
        execute_emergency_rollback()
        write_audit("go_live_interrupted", {"reason": "user_interrupt"})
        return 1
    except Exception as e:
        print(f"\n❌ Go-Live Day failed with error: {e}")
        execute_emergency_rollback()
        write_audit("go_live_error", {"error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
