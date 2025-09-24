#!/usr/bin/env python3
"""
Offline Gate Checker
Reads eval.json and gate YAML, outputs hard PASS/FAIL with exit codes
"""

import json
import sys
import yaml
import pathlib
import argparse


def fail(msg):
    print("GATE_FAIL:", msg)
    sys.exit(1)


def ok(msg):
    print("GATE_PASS:", msg)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Offline Gate Checker")
    parser.add_argument("--eval", required=True, help="Path to eval.json")
    parser.add_argument("--gate", required=True, help="Path to gate YAML")
    parser.add_argument("--out-md", default=None, help="Output markdown report path")
    args = parser.parse_args()

    try:
        # Load evaluation results and gate configuration
        with open(args.eval, "r") as f:
            e = json.load(f)
        with open(args.gate, "r") as f:
            g = yaml.safe_load(f)

        print(f"🔍 Checking gate: {g.get('name', 'unnamed')}")
        print(f"📊 Eval results from: {args.eval}")

    except FileNotFoundError as err:
        fail(f"File not found: {err}")
    except Exception as err:
        fail(f"Error loading files: {err}")

    msgs = []

    def add(condition, message):
        if not condition:
            msgs.append(message)

    # Extract evaluation metrics
    ent = e.get("entropy_mean")
    ret = e.get("return_mean")
    qsp = e.get("q_spread_mean", None)
    gn95 = e.get("grad_norm_p95", None)
    has_nan = e.get("has_nan", True)
    episodes = e.get("episodes", 0)

    # Load baseline if available
    base = {}
    bp = g.get("baseline_json")
    if bp and pathlib.Path(bp).exists():
        try:
            with open(bp, "r") as f:
                base = json.load(f)
            print(f"📈 Using baseline from: {bp}")
        except Exception as err:
            print(f"⚠️  Warning: Could not load baseline {bp}: {err}")
    else:
        print(f"⚠️  No baseline found at: {bp}")

    ret_base = base.get("return_mean", 0.0)
    qsp_base = base.get("q_spread_mean", qsp or 1.0)

    # Apply gate rules
    print(f"\n🚦 Applying gate rules...")

    # Episodes minimum
    episodes_min = g.get("episodes_min", 1)
    add(episodes >= episodes_min, f"episodes_min failed: {episodes} < {episodes_min}")
    print(
        f"   Episodes: {episodes} >= {episodes_min} {'✅' if episodes >= episodes_min else '❌'}"
    )

    # Entropy range
    if ent is not None:
        lo, hi = g["entropy_range"]
        entropy_ok = lo <= ent <= hi
        add(entropy_ok, f"entropy {ent:.3f} not in [{lo},{hi}]")
        print(f"   Entropy: {ent:.3f} in [{lo},{hi}] {'✅' if entropy_ok else '❌'}")
    else:
        add(False, "entropy_mean missing")
        print("   Entropy: missing ❌")

    # NaN requirement
    nan_ok = not has_nan if g.get("require_no_nans", True) else True
    add(nan_ok, "has_nan true")
    print(f"   No NaNs: {not has_nan} {'✅' if nan_ok else '❌'}")

    # Gradient norm
    if gn95 is not None:
        max_grad = g.get("max_grad_norm_p95", float("inf"))
        grad_ok = gn95 <= max_grad
        add(grad_ok, f"grad_norm_p95 {gn95:.3f} > {max_grad}")
        print(f"   Grad Norm P95: {gn95:.3f} <= {max_grad} {'✅' if grad_ok else '❌'}")

    # Return vs baseline
    min_rel = g["min_return_vs_last_good"]
    if ret is not None:
        min_required = ret_base + min_rel
        return_ok = ret >= min_required
        add(return_ok, f"return_mean {ret:.6f} < baseline {ret_base:.6f} + {min_rel}")
        print(
            f"   Return: {ret:.6f} >= {min_required:.6f} {'✅' if return_ok else '❌'}"
        )
    else:
        add(False, "return_mean missing")
        print("   Return: missing ❌")

    # Q-spread sanity
    if qsp is not None and qsp_base:
        max_qsp = g["q_spread_max_ratio"] * qsp_base
        qspread_ok = qsp <= max_qsp
        add(
            qspread_ok,
            f"q_spread {qsp:.1f} too high vs base {qsp_base:.1f} * {g['q_spread_max_ratio']}",
        )
        print(f"   Q-Spread: {qsp:.1f} <= {max_qsp:.1f} {'✅' if qspread_ok else '❌'}")

    # Generate markdown report if requested
    if args.out_md:
        try:
            with open(args.out_md, "w") as f:
                f.write("# Offline Gate Report\n\n")
                f.write(f"**Gate:** {g.get('name', 'unnamed')}\n")
                f.write(f"**Timestamp:** {e.get('timestamp', 'unknown')}\n")
                f.write(f"**Checkpoint:** `{e.get('ckpt_path', 'unknown')}`\n\n")

                f.write("## Metrics\n\n")
                f.write(f"- **Episodes:** {episodes}\n")
                f.write(
                    f"- **Entropy Mean:** {ent:.3f}\n"
                    if ent
                    else "- **Entropy Mean:** missing\n"
                )
                f.write(
                    f"- **Return Mean:** {ret:.6f}\n"
                    if ret
                    else "- **Return Mean:** missing\n"
                )
                f.write(
                    f"- **Grad Norm P95:** {gn95:.3f}\n"
                    if gn95
                    else "- **Grad Norm P95:** not available\n"
                )
                f.write(
                    f"- **Q-Spread Mean:** {qsp:.1f}\n"
                    if qsp
                    else "- **Q-Spread Mean:** not available\n"
                )
                f.write(f"- **Has NaN:** {has_nan}\n\n")

                if base:
                    f.write("## Baseline Comparison\n\n")
                    f.write(f"- **Baseline Return:** {ret_base:.6f}\n")
                    f.write(f"- **Baseline Q-Spread:** {qsp_base:.1f}\n\n")

                if msgs:
                    f.write("## Failures ❌\n\n")
                    for msg in msgs:
                        f.write(f"- {msg}\n")
                    f.write("\n")
                else:
                    f.write("## Result ✅\n\n")
                    f.write("- **PASS** - All checks passed\n\n")

            print(f"📝 Report saved to: {args.out_md}")
        except Exception as err:
            print(f"⚠️  Warning: Could not write report: {err}")

    # Final result
    print(f"\n🏁 Final Result:")
    if msgs:
        failure_summary = "; ".join(msgs)
        print(f"   ❌ GATE_FAIL: {len(msgs)} check(s) failed")
        for msg in msgs:
            print(f"      - {msg}")
        fail(failure_summary)
    else:
        print(f"   ✅ GATE_PASS: All checks passed")
        ok("All checks passed.")


if __name__ == "__main__":
    main()
