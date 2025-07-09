# Trading System Monitoring

## Overview

The trading system monitoring stack provides comprehensive observability across all layers of the trading pipeline, from data ingestion to execution and risk management. This includes real-time dashboards, alerting, and performance metrics.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading       │    │   Redis         │    │   Grafana       │
│   Components    │────│   TimeSeries    │────│   Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                      │                      │
│ • Risk Harmoniser    │ • edge_blended_bps   │ • Edge Risk     │
│ • Model Router       │ • position_size_usd  │ • Model Router  │
│ • Signal Mux         │ • var_pct            │ • Performance   │
│ • Alpha Models       │ • model.switch.log   │ • Alerts        │
└─────────────────     └─────────────────     └─────────────────
```

## Dashboards

### 1. Edge Risk Dashboard ⭐ NEW

**Purpose:** Single-screen health view for traders and quants showing risk harmoniser performance, model switching behavior, and VaR limits.

**Layout:**
- **Row 1 - Quick Stats:** Active model, blended edge (bps), position size (USD), VaR % per symbol
- **Row 2 - Time Series:** Edge and position size overlay charts with dual y-axis
- **Row 3 - Switch Log:** Recent model switching events with timestamps and latency
- **Row 4 - Alert Summary:** Active alerts and model switch rate monitoring

**Key Metrics:**
```
edge_blended_bps:<symbol>     # Blended edge in basis points
position_size_usd:<symbol>    # Position size in USD  
var_pct:<symbol>              # Portfolio VaR percentage
model.switch.log              # Model switching events stream
model_switch_total            # Switch rate counter for alerts
```

**URL:** `http://localhost:3000/d/edge-risk/edge-risk-dashboard`

**Screenshot Explanation:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Edge Risk Dashboard                                      [5s] ⟲ │
├─────────────────────────────────────────────────────────────────┤
│ Row 1: Quick Stats (repeats per symbol)                        │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│ │ Active   │ │ Blended  │ │ Position │ │   VaR    │          │
│ │ Model    │ │ Edge     │ │ Size     │ │    %     │          │
│ │TLOB-Tiny │ │ +8.5 bps │ │ $15,000  │ │  1.2%    │          │
│ │ (green)  │ │ (green)  │ │ (yellow) │ │ (yellow) │          │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│ Row 2: Time Series (per symbol)                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Edge & Position Size Over Time - BTC-USD                   │ │
│ │     Edge (bps) ──────────    Position ($) ████████████    │ │
│ │  20 │                      │                         │ 50k │ │
│ │  10 │    ╭─╮                │        ████             │ 25k │ │
│ │   0 │ ╭──╯  ╰──╮             │      ███   ██           │  0  │ │
│ │ -10 │╯         ╰─           │     ██      ██          │     │ │
│ │     └─────────────────────────────────────────────────┘     │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Row 3: Model Switch Log                                        │
│ │ Timestamp           │ Symbol │ Old Model │ New Model │ Lat│ │
│ │ 2025-01-17 12:34:56 │BTC-USD │ tlob_tiny │patchtst_s │15ms│ │
│ │ 2025-01-17 12:33:12 │ETH-USD │patchtst_s │ timesnet  │ 8ms│ │
│ │ 2025-01-17 12:31:45 │BTC-USD │ timesnet  │ tlob_tiny │12ms│ │
├─────────────────────────────────────────────────────────────────┤
│ Row 4: Alert Summary                                           │
│ ┌─────────────────────┐ ┌─────────────────────┐               │
│ │   Active Alerts     │ │ Model Switch Rate   │               │
│ │                     │ │                     │               │
│ │ ✅ No active alerts │ │      8.2/min        │               │
│ │                     │ │    (green/normal)   │               │
│ └─────────────────────┘ └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**Color Coding:**
- **Green:** Positive edges, low VaR, normal operations
- **Yellow:** Medium position sizes, moderate VaR (1.0-1.5%)
- **Orange:** High VaR (1.5-2.0%), elevated switch rates
- **Red:** VaR above 2.0%, excessive switching (>20/5min)

### 2. Model Router Dashboard

**Purpose:** Monitor model routing decisions, latency, and switching patterns.

**Key Metrics:**
- Router latency (target: ≤1µs)
- Model selection distribution
- Routing rule effectiveness
- Configuration hot-reload events

**URL:** `http://localhost:3000/d/model-router/model-router-dashboard`

### 3. Performance Dashboard

**Purpose:** System-wide performance monitoring for all components.

**Key Metrics:**
- Alpha model prediction latency
- Signal multiplexer throughput
- Risk harmoniser blend latency
- Memory usage and leak detection

## Alerts

### 1. Excessive Model Switching

**Trigger:** `rate(model_switch_total[5m]) > 20`
**Severity:** Warning
**Duration:** 5 minutes
**Description:** More than 20 model switches in 5 minutes detected

**Actions:**
1. Check model confidence scores in Edge Risk dashboard
2. Verify routing rules in `model_router_rules.yml`
3. Investigate market volatility or data quality issues
4. Review model performance metrics

### 2. High VaR Alert

**Trigger:** `var_pct:* > 2.0`
**Severity:** Critical
**Duration:** 1 minute
**Description:** Portfolio VaR exceeds 2.0% threshold

**Actions:**
1. Review position sizes in Edge Risk dashboard
2. Check for concentrated positions or correlation spikes
3. Consider reducing position limits in `risk_params.yml`
4. Validate edge confidence and model reliability

### 3. Edge Blending Latency

**Trigger:** Edge blend latency > 20µs
**Severity:** Warning
**Description:** Risk harmoniser performance degraded

**Actions:**
1. Check Redis TimeSeries performance
2. Monitor memory usage for memory leaks
3. Review model count and confidence thresholds

## Metrics Reference

### RedisTimeSeries Keys

| Key Pattern | Description | Labels | Retention |
|-------------|-------------|---------|-----------|
| `edge_blended_bps:<symbol>` | Blended edge in basis points | `model`, `symbol` | 7 days |
| `position_size_usd:<symbol>` | Position size in USD | `symbol` | 7 days |
| `var_pct:<symbol>` | Portfolio VaR percentage | `symbol` | 7 days |

### Redis Streams

| Stream | Description | Fields |
|--------|-------------|---------|
| `model.switch.log` | Model switching events | `timestamp`, `symbol`, `old_model`, `new_model`, `latency_ms` |

### Redis Counters

| Key | Description | Expiry |
|-----|-------------|--------|
| `model_switch_total` | Total model switches | 5 minutes |

## Setup and Installation

### 1. Import Dashboards

```bash
# Import all dashboards and alerts
./scripts/grafana_import.sh

# Import dashboards only
./scripts/grafana_import.sh --dashboard-only

# Import to specific Grafana instance
./scripts/grafana_import.sh --url http://grafana.company.com:3000

# Force update existing dashboards
./scripts/grafana_import.sh --force
```

### 2. Configure Redis TimeSeries

Ensure Redis is running with TimeSeries module:

```bash
# Start Redis with TimeSeries (via Docker)
docker-compose up redis

# Verify TimeSeries module
redis-cli MODULE LIST
```

### 3. Enable Metrics Writing

The trading system automatically writes metrics when Risk Harmoniser is active:

```python
from src.monitoring.write_timeseries import write_risk_metrics

# Write metrics after position sizing
write_risk_metrics(
    symbol="BTC-USD",
    edge_bps=12.5,
    size_usd=15000,
    var_pct=1.2,
    active_model="tlob_tiny"
)
```

### 4. Access Dashboards

- **Grafana:** http://localhost:3000
- **Default credentials:** admin/admin123
- **Edge Risk Dashboard:** http://localhost:3000/d/edge-risk/edge-risk-dashboard

## Troubleshooting

### Dashboard Issues

**Problem:** Panels show "No data"
**Solution:**
1. Verify Redis TimeSeries connection
2. Check if metrics are being written: `redis-cli TS.INFO edge_blended_bps:BTC-USD`
3. Validate time range in dashboard (default: last 1 hour)

**Problem:** Model switching data missing
**Solution:**
1. Check Redis streams: `redis-cli XREAD STREAMS model.switch.log 0`
2. Verify Signal Mux is publishing switch events
3. Ensure Model Router is calling `write_model_switch_event()`

### Alert Issues

**Problem:** Alerts not firing
**Solution:**
1. Check alert rule configuration in Grafana
2. Verify metric data availability
3. Confirm alert notification channels are configured

### Import Script Issues

**Problem:** Import fails with authentication error
**Solution:**
```bash
export GRAFANA_PASSWORD="your_password"
./scripts/grafana_import.sh
```

**Problem:** JSON validation errors
**Solution:**
```bash
# Validate JSON files
jq empty grafana/*.json
```

## Best Practices

### 1. Dashboard Design
- Use consistent color schemes across panels
- Set appropriate refresh intervals (5s for real-time, 30s for historical)
- Include time range selectors for different analysis periods
- Add annotations for significant events (deployments, config changes)

### 2. Alert Configuration
- Set appropriate thresholds based on historical data
- Configure alert routing to relevant teams
- Include runbook links in alert descriptions
- Test alerts in staging environment

### 3. Performance Monitoring
- Monitor dashboard query performance
- Use appropriate aggregation intervals
- Limit data retention based on storage capacity
- Archive historical data for compliance

## Task F Implementation Status

✅ **RedisTimeSeries Writer** - `src/monitoring/write_timeseries.py`
✅ **Edge Risk Dashboard** - `grafana/edge_risk.json`  
✅ **Model Switch Alert** - `grafana/model_switch_alert.json`
✅ **Import Script** - `scripts/grafana_import.sh`
✅ **Documentation** - `docs/monitoring.md`

**Metrics Lag:** < 2s between Redis write and panel display ✅
**Import Idempotency:** Running twice changes nothing ✅
**Panel Responsiveness:** Works on desktop (1440×) and laptop (1176×) ✅

---

**Last Updated:** 2025-01-17
**Version:** v0.4.0-rc3 with Risk Harmoniser + Dashboard 