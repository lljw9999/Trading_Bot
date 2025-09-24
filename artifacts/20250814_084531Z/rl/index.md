# 48h Validation Report - 20250814_084531Z

**Status:** FAIL ❌

**Details:**
- Timestamp: 20250814_084531Z
- Latest artifact dir: artifacts/20250814_164531/rl
- Baseline updated: False
- Slack notification: failed/disabled

**Gate Report:**
```
# Offline Gate Report

**Gate:** sol_offline_gate
**Timestamp:** 2025-08-14T08:45:33.315643Z
**Checkpoint:** `checkpoints/latest.pt`

## Metrics

- **Episodes:** 32
- **Entropy Mean:** 1.421
- **Return Mean:** -187.742876
- **Grad Norm P95:** 1.137
- **Q-Spread Mean:** 34.4
- **Has NaN:** False

## Failures ❌

- return_mean -187.742876 < baseline 0.000000 + -0.05


```
