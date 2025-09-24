# Model Card: sol_rl_policy_v1

**Version:** 1.0.0  
**Date:** 2025-08-13  
**Owner:** rl-team@company

## Overview

Reinforcement Learning policy for SOL-USD trading using SAC with LoRA adaptation

## Model Details

### Architecture
- **Algorithm:** SAC-DiF + LoRA
- **Environment:** OrderBookEnv
- **Training Data Window:** 2025-05 → 2025-08
- **Training Duration:** 72 hours
- **Compute Resources:** 8x A100 GPUs

### Hyperparameters
- **Batch Size:** 256
- **Gamma:** 0.99
- **Learning Rate:** 3e-4
- **Lora Alpha:** 32
- **Lora Rank:** 16
- **Replay Buffer Size:** 1000000
- **Tau:** 0.005

## Performance Evaluation

### Offline Validation
- **Pass Rate:** 95%
- **Mean Entropy:** 1.05 ± 0.15
- **Mean Return:** 0.0024
- **Sharpe Ratio:** 1.8
- **Max Drawdown:** 1.2%

### A/B Test Results
- **PnL Delta:** TBD
- **95% Confidence Interval:** TBD
- **Verdict:** **PENDING**
- **Sample Size:** TBD
## Risk Management

### Risk Limits
- **Entropy Floor:** 0.9
- **Q-Spread Guard:** x2 of 24h baseline
- **Daily Drawdown Max:** 2%
- **Max Position Size:** 10% of daily volume

### Kill Switch Triggers
- Entropy < 0.9 for 30min
- Daily drawdown > 2%
- Q-spread > 2x baseline for 30min
- Manual emergency stop

### Influence Limits
- **Pilot Maximum:** 25%
- **Production Maximum:** 100%
- **Ramp Schedule:** 10% → 15% → 25%

## Explainability & Interpretability

### Input Features
- ob_imbalance
- vol
- returns
- spread
- depth

### Attribution Methods
- Gradient-based attribution
- Feature importance via permutation
- LIME for local explanations

### Tools
- ExplainMiddleware
## Governance & Controls

### Audit Trail
- `artifacts/audit/`
- `artifacts/pilot/`
- `artifacts/validation/`

### Operational Controls
- kill-switch
- error-budget-guard
- watchdog
- alerts
- pilot-kri-monitor
- ramp-guard

### Approval Chain
- Model Risk Committee
- Trading Risk Manager
- Head of Technology

### Monitoring Infrastructure
#### Prometheus Metrics
- `rl_policy_entropy`
- `rl_policy_q_spread`
- `rl_policy_heartbeat_age_seconds`
- `rl_policy_influence_pct`

#### Alert Rules
- RLPolicyStale
- RLPolicyEntropyLow
- RLPolicyCollapseRisk

## Compliance

- **Model Risk Tier:** Tier 1
- **Regulatory Framework:** MiFID II, MAS Guidelines
- **Validation Frequency:** Weekly offline, Daily KRI monitoring
- **Documentation Status:** COMPLETE
- **Audit Trail:** WORM-compliant

## Deployment

### Infrastructure
- **Environment:** Kubernetes on AWS
- **Scaling:** Horizontal pod autoscaling
- **Persistence:** Redis + InfluxDB

### Rollout Strategy
- **Blue/Green:** True
- **Canary Percentage:** 10%
- **Monitoring Period:** 2 hours

### Rollback Criteria
- KRI breach
- Alert severity >= critical
- Performance degradation > 5%

## Metadata

- **Created By:** rl-team@company
- **Reviewed By:** model-risk@company
- **Approved By:** trading-risk@company
- **Last Updated:** 2025-08-13T13:00:00Z
- **Next Review:** 2025-08-20T00:00:00Z

### Version History
- v1.0.0: Initial production release

---
*This model card was generated automatically on 2025-08-14T08:38:58.031544+00:00*
