# Go/No-Go Playbook - RL Policy Live Deployment

## Overview

This playbook defines the criteria and process for making Go/No-Go decisions for RL policy live deployment. **Default influence is 0% (shadow mode)** until all Go criteria are met.

## Go/No-Go Decision Criteria

### 🟢 GO Criteria

All of the following must be satisfied for a GO decision:

#### 1. Validation History ✅
- [ ] **Two consecutive 48h PASS validations** in `artifacts/validation/`
- [ ] Latest validation artifacts < 24h old
- [ ] No validation failures in last 72h

#### 2. Alerting & Monitoring ✅
- [ ] **No page alerts** in last 48h
- [ ] Warning alerts ≤ 5 and all resolved
- [ ] All monitoring services operational (exporter, prober, cost monitor)

#### 3. SLO Performance ✅
- [ ] **Error budget burn < 25%** month-to-date
- [ ] Policy heartbeat freshness SLO: ≥99.5%
- [ ] Exporter uptime SLO: ≥99.9%
- [ ] Validation cadence SLO: ≥95%

#### 4. Technical Readiness ✅
- [ ] **Preflight diff gate PASS** (blue/green consistency)
- [ ] Security scan PASS (no secret leaks, low-risk dependencies)
- [ ] **Kill-switch tested** in current build
- [ ] Ramp guard operational

#### 5. Operational Readiness ✅
- [ ] **On-call engineer ACK** for deployment window
- [ ] Ops bot commands tested and functional
- [ ] Runbooks updated and accessible
- [ ] Rollback procedures validated

#### 6. Policy Health ✅
- [ ] Policy entropy ≥ 0.9 (sufficient exploration)
- [ ] Q-value spread within 2x baseline
- [ ] No collapse risk indicators
- [ ] Training stability confirmed

### 🔴 NO-GO Criteria

Any of the following results in immediate NO-GO:

- ❌ **Active page alerts** or unresolved critical incidents
- ❌ **Error budget exhausted** (≥100% burn)
- ❌ **Policy collapse risk** = HIGH
- ❌ **Kill-switch non-functional**
- ❌ **Failed security scan** (secrets detected)
- ❌ **No on-call coverage** during deployment window
- ❌ **Cost spike >100%** WoW without explanation

## Decision Process

### 1. Automated Check
Run `python scripts/go_nogo_check.py` to get initial GO/NO_GO assessment.

### 2. Human Review
- **Release Captain** reviews automated results
- **On-call Engineer** provides operational ACK
- **Technical Lead** approves policy health metrics

### 3. Decision Recording
All Go/No-Go decisions are recorded in `artifacts/audit/` with:
- Timestamp and decision maker
- All criteria status (PASS/FAIL/WARN)
- Justification for decision
- Next review date if NO-GO

## Deployment Procedure (GO Only)

### Phase 1: Pre-Flight Checks (15 min)
```bash
# Run all pre-deployment gates
make release-gates
make preflight  
make go-nogo
```

**Proceed only if all return 0 (success)**

### Phase 2: Controlled Ramp (2 hours monitoring)
```bash
# Confirm 0% influence
make influence

# Execute ramp with safety checks
make ramp-10

# Monitor for 2 hours
# Watch dashboards: entropy, Q-spread, alerts
```

### Phase 3: Monitoring & Decision Points

**T+30min:** Check for any alerts
- If alerts → immediate `make kill-switch`

**T+60min:** Review metrics stability
- Entropy should remain ≥ 0.9
- Q-spread within acceptable bounds
- No degradation in shadow metrics

**T+120min:** Make continuation decision
- All metrics stable → can consider higher influence
- Any degradation → `make kill-switch` and post-incident

## Rollback Triggers

Execute immediate `make kill-switch` if ANY of:
- Page alert triggered
- Policy entropy < 0.8
- Q-value spread >3x baseline  
- Trading performance degradation
- Cost spike >50% from baseline
- Any technical anomaly

## Communication Plan

### Pre-Deployment
- [ ] Slack announcement in #ops-alerts 2h before
- [ ] Email to stakeholders with Go/No-Go result
- [ ] Runbook links shared with on-call

### During Deployment  
- [ ] Real-time updates in #deployment-live every 30min
- [ ] Dashboard links pinned in channel
- [ ] Kill-switch procedures visible

### Post-Deployment
- [ ] Success/failure summary in #ops-alerts
- [ ] Metrics analysis and lessons learned
- [ ] Update playbook based on experience

## Emergency Contacts

- **Release Captain:** @release-captain
- **On-Call Engineer:** @oncall-primary  
- **Technical Lead:** @tech-lead
- **Escalation:** @engineering-manager

## Key Commands Reference

```bash
# Status checks
make influence              # Check current influence
make go-nogo               # Run Go/No-Go assessment
make ramp-status           # Current ramp metrics

# Deployment actions  
make ramp-10               # Ramp to 10% with guards
make kill-switch           # Emergency 0% revert
make influence-set PCT=25  # Manual influence setting

# Monitoring
curl localhost:9108/metrics | grep influence
curl localhost:9110/metrics | grep probe
make budget-guard-now      # Check error budget
```

## Success Criteria

A successful deployment demonstrates:
1. **Stable operation** at target influence for ≥2 hours
2. **No degradation** in key metrics (entropy, Q-spread)
3. **Alert-free** monitoring period
4. **Controlled shadow→live** transition without incidents

## Failure Criteria

A failed deployment requires immediate rollback:
1. **Any page alert** during deployment window
2. **Policy metrics degradation** beyond thresholds
3. **Cost or performance** impact beyond acceptable bounds
4. **Technical failures** in monitoring or control systems

---

## Audit Trail

All Go/No-Go decisions are automatically recorded in:
- `artifacts/audit/*_go_nogo.json` - Decision records
- `artifacts/deployment/*` - Deployment artifacts  
- Slack #ops-alerts channel - Communication record

**Next Review:** This playbook should be reviewed after each deployment and updated based on lessons learned.