# 48h Validation Report - 20250813_080435Z

**Status:** FAIL ❌

**Details:**
- Timestamp: 20250813_080435Z
- Latest artifact dir: artifacts/20250813_160435/rl
- Baseline updated: False
- Slack notification: failed/disabled

**Gate Report:**
```
# Offline Gate Report

**Gate:** sol_offline_gate
**Timestamp:** 2025-08-13T08:04:38.082949Z
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
