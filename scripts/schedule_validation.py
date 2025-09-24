#!/usr/bin/env python3
"""
48h Offline Validation Scheduler
Runs offline gate, manages baseline artifacts, and sends Slack summaries
"""
import os, subprocess, time, json, sys
from datetime import datetime, timezone
from pathlib import Path


ART_ROOT = os.getenv("ART_ROOT", "artifacts")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cmd(cmd, cwd=None):
    """Run shell command and return result."""
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=cwd)


def send_slack_notification(message):
    """Send notification to Slack if webhook URL is configured."""
    if not SLACK_WEBHOOK_URL:
        print("ℹ️  No SLACK_WEBHOOK_URL configured, skipping notification")
        return False

    try:
        # Try using requests first
        try:
            import requests

            response = requests.post(
                SLACK_WEBHOOK_URL, json={"text": message}, timeout=10
            )
            if response.status_code == 200:
                print("✅ Slack notification sent via requests")
                return True
            else:
                print(f"⚠️  Slack API returned {response.status_code}")
        except ImportError:
            pass

        # Fallback to curl
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"text": message}, f)
            temp_file = f.name

        result = run_cmd(
            f'curl -X POST -H "Content-type: application/json" --data @{temp_file} {SLACK_WEBHOOK_URL}'
        )
        os.unlink(temp_file)

        if result.returncode == 0:
            print("✅ Slack notification sent via curl")
            return True
        else:
            print(f"⚠️  Slack notification failed: {result.stderr}")

    except Exception as e:
        print(f"⚠️  Slack notification error: {e}")

    return False


def find_latest_artifact_dir():
    """Find the most recent artifact directory."""
    try:
        result = run_cmd("ls -dt artifacts/*/rl 2>/dev/null | head -n1")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def main():
    print("🚀 Starting 48h RL Validation Cycle")
    print(f"📁 Artifact root: {ART_ROOT}")
    print(f"📅 Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Generate timestamped artifact directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    art_dir = f"{ART_ROOT}/{ts}/rl"
    os.makedirs(art_dir, exist_ok=True)
    print(f"📦 Created artifact directory: {art_dir}")

    # Change to project root for running scripts
    os.chdir(PROJECT_ROOT)
    print(f"📂 Working directory: {os.getcwd()}")

    # 1) Run offline gate evaluation
    print("\n🚦 Running offline gate evaluation...")
    gate_result = run_cmd("bash scripts/run_offline_gate.sh")
    print(f"Gate command exit code: {gate_result.returncode}")

    if gate_result.stdout:
        print("📋 Gate stdout:")
        print(gate_result.stdout[-1000:])  # Last 1000 chars
    if gate_result.stderr:
        print("⚠️  Gate stderr:")
        print(gate_result.stderr[-500:])  # Last 500 chars

    # 2) Find and read the gate report
    gate_md = ""
    gate_status = "UNKNOWN"
    latest_dir = find_latest_artifact_dir()

    if latest_dir:
        gate_report_path = f"{latest_dir}/gate_report.md"
        print(f"📄 Looking for gate report: {gate_report_path}")

        if os.path.exists(gate_report_path):
            try:
                with open(gate_report_path, "r") as f:
                    gate_md = f.read()
                print(f"✅ Loaded gate report ({len(gate_md)} chars)")

                # Determine pass/fail status
                if "PASS" in gate_md.upper() and "✅" in gate_md:
                    gate_status = "PASS"
                elif "FAIL" in gate_md.upper() and "❌" in gate_md:
                    gate_status = "FAIL"

            except Exception as e:
                print(f"❌ Failed to read gate report: {e}")
        else:
            print(f"⚠️  Gate report not found at expected path")
    else:
        print("⚠️  Could not find latest artifact directory")

    print(f"🏁 Gate evaluation result: {gate_status}")

    # 3) If PASS, update baseline artifacts
    baseline_updated = False
    if gate_status == "PASS":
        print("\n✅ Gate PASSED - updating baseline artifacts...")

        try:
            baseline_dir = "artifacts/last_good/rl"
            os.makedirs(baseline_dir, exist_ok=True)

            # Copy evaluation results to baseline
            if latest_dir:
                eval_json_src = f"{latest_dir}/eval.json"
                eval_json_dst = f"{baseline_dir}/eval.json"

                if os.path.exists(eval_json_src):
                    result = run_cmd(f"cp {eval_json_src} {eval_json_dst}")
                    if result.returncode == 0:
                        print(f"📋 Updated baseline eval.json")
                        baseline_updated = True
                    else:
                        print(f"⚠️  Failed to copy eval.json: {result.stderr}")

                # Copy checkpoint if it exists
                ckpt_src = "checkpoints/latest.pt"
                ckpt_dst = f"{baseline_dir}/latest.pt"

                if os.path.exists(ckpt_src):
                    result = run_cmd(f"cp {ckpt_src} {ckpt_dst}")
                    if result.returncode == 0:
                        print(f"📦 Updated baseline checkpoint")
                    else:
                        print(f"⚠️  Failed to copy checkpoint: {result.stderr}")

                # Update baseline timestamp
                with open(f"{baseline_dir}/timestamp.txt", "w") as f:
                    f.write(f"{ts}\n")
                print(f"🕒 Updated baseline timestamp")

        except Exception as e:
            print(f"❌ Failed to update baseline: {e}")
    else:
        print(f"❌ Gate FAILED - baseline unchanged")

    # 4) Send Slack summary
    print(f"\n📱 Preparing Slack notification...")

    status_emoji = "✅" if gate_status == "PASS" else "❌"
    baseline_note = " (baseline updated)" if baseline_updated else ""

    slack_message = f"""*48h RL Validation {status_emoji}*

**Result:** {gate_status}{baseline_note}
**Timestamp:** {ts}
**Artifacts:** `{latest_dir or art_dir}`

**Gate Report Summary:**
```
{gate_md[:800] if gate_md else "Gate report not available"}
```

**Actions:**
{f"• Baseline artifacts updated in `artifacts/last_good/rl/`" if baseline_updated else "• Baseline unchanged (gate failed)"}
• Full report: `{latest_dir}/gate_report.md` if available
• Policy influence remains at 0% (shadow mode)
"""

    slack_sent = send_slack_notification(slack_message)

    # 5) Write validation index
    print(f"\n📝 Writing validation index...")

    try:
        with open(f"{art_dir}/index.md", "w") as f:
            f.write(f"# 48h Validation Report - {ts}\n\n")
            f.write(f"**Status:** {gate_status} {status_emoji}\n\n")
            f.write(f"**Details:**\n")
            f.write(f"- Timestamp: {ts}\n")
            f.write(f"- Latest artifact dir: {latest_dir or 'not found'}\n")
            f.write(f"- Baseline updated: {baseline_updated}\n")
            f.write(
                f"- Slack notification: {'sent' if slack_sent else 'failed/disabled'}\n\n"
            )

            if gate_md:
                f.write(f"**Gate Report:**\n")
                f.write(f"```\n{gate_md}\n```\n")

        print(f"✅ Validation index written to: {art_dir}/index.md")

    except Exception as e:
        print(f"⚠️  Failed to write index: {e}")

    # 6) Final summary
    print(f"\n🏁 48h Validation Cycle Complete")
    print(f"   Status: {gate_status}")
    print(f"   Artifacts: {art_dir}")
    print(f"   Baseline: {'updated' if baseline_updated else 'unchanged'}")
    print(f"   Slack: {'✅' if slack_sent else '❌'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)
