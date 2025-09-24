#!/usr/bin/env python3
import os, time, yaml, subprocess, sys, datetime, json
from datetime import timezone

CFG = "pilot/pilot_run.yaml"


def run(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def main():
    cfg = yaml.safe_load(open(CFG))
    sched = cfg["pilot"]["schedule"]
    name = cfg["pilot"]["name"]
    # iterate schedule; for each step, attempt guarded ramp
    for step in sched:
        pct = step["pct"]
        hold = step["hold_min"]
        # attempt ramp with guard
        r = run(f"python scripts/pilot_guard.py --config {CFG} --target-pct {pct}")
        print(r.stdout.strip())
        if r.returncode != 0:
            print("Ramp blocked; stopping schedule.")
            sys.exit(1)
        # hold & monitor (lightweight)
        print(f"Holding at {pct}% for {hold} minutes...")
        time.sleep(1)  # replace with real scheduler; keep 1s for dry-run
    print("Schedule complete.")


if __name__ == "__main__":
    main()
