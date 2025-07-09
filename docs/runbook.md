# Trading System Operations Runbook

## 🚨 On-Call Playbook v0.4.0-rc3

**Target MTTR:** ≤ 10 minutes for critical alerts  
**Last Updated:** 2025-01-17  
**On-Call Contact:** [Update with team details]

---

## 1. System Overview

### Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Core Engine   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Coinbase WS   │───▶│ • Signal Mux    │───▶│ • Grafana       │
│ • Binance API   │    │ • Model Router  │    │ • Prometheus    │
│ • Alpha Vertex  │    │ • Risk Harm.    │    │ • Redis TS      │
│ • Market Data   │    │ • Param Server  │    │ • Alerts        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Flow     │    │   Redis Keys    │    │   Health URLs   │
│                 │    │                 │    │                 │
│ Tick → Feature  │    │ param.reload    │    │ /health         │
│ Feature → Model │    │ risk.edge_*     │    │ /metrics        │
│ Model → Risk    │    │ model.switch.*  │    │ /api/ready      │
│ Risk → Execute  │    │ position_size_* │    │ /api/status     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Redis Keys
| Pattern | Purpose | TTL |
|---------|---------|-----|
| `param.reload` | Hot-reload triggers | 60s |
| `risk.edge_blended.<symbol>` | Position sizing output | 5min |
| `model.switch.log` | Model switching events | 24h |
| `edge_blended_bps:<symbol>` | TimeSeries metrics | 7d |
| `position_size_usd:<symbol>` | Position metrics | 7d |
| `var_pct:<symbol>` | VaR metrics | 7d |

---

## 2. Normal Operations Checklist

### 🔄 **Daily Health Check** (5 min)
1. **Check Grafana Dashboard:** http://localhost:3000/d/edge-risk/
   - [ ] All panels showing data (last 5 minutes)
   - [ ] No red alerts in Alert Summary
   - [ ] Model switch rate < 15/min
   - [ ] VaR < 2.0% for all symbols

2. **Verify Component Health:**
   ```bash
   curl -s http://localhost:8080/health | jq '.status'  # Should be "healthy"
   redis-cli ping                                       # Should return PONG
   curl -s http://localhost:3000/api/health            # Grafana health
   ```

3. **Check Recent Logs:**
   ```bash
   tail -100 logs/trading-system.log | grep -i error
   docker logs trading-system-redis --since 1h
   ```

### 🚀 **Restart Sequence** (when needed)
**CRITICAL:** Always follow this exact order to prevent data loss.

```bash
# 1. Stop trading (graceful)
curl -X POST http://localhost:8080/api/stop-trading
sleep 10

# 2. Stop services (reverse dependency order)  
docker-compose stop web strategist risk-harmoniser

# 3. Restart infrastructure
docker-compose restart redis prometheus

# 4. Wait for Redis to be ready
until redis-cli ping; do sleep 1; done

# 5. Restart core services  
docker-compose up -d risk-harmoniser
sleep 5
docker-compose up -d strategist  
sleep 5
docker-compose up -d web

# 6. Resume trading
curl -X POST http://localhost:8080/api/start-trading
```

### 📊 **Health Check URLs**
| Service | URL | Expected Response |
|---------|-----|-------------------|
| **Main API** | `http://localhost:8080/health` | `{"status": "healthy"}` |
| **Grafana** | `http://localhost:3000/api/health` | `{"commit": "...", "database": "ok"}` |
| **Prometheus** | `http://localhost:9090/-/healthy` | `Prometheus is Healthy.` |
| **Redis** | `redis-cli ping` | `PONG` |

---

## 3. Alarm Playbooks

### 🚨 **CRITICAL: Excessive Model Switching**
**Alert:** `rate(model_switch_total[5m]) > 20`  
**Severity:** Warning  
**Target MTTR:** 5 minutes

#### Immediate Actions
1. **Check Edge Risk Dashboard:** Look for patterns in switching
2. **Pause Router:** Send SIGHUP to reduce volatility
   ```bash
   pkill -SIGHUP -f model_router
   # Or via API:
   curl -X POST http://localhost:8080/api/router/pause
   ```
3. **Check Model Confidence:** Verify no models are failing
   ```bash
   redis-cli XREVRANGE model.switch.log + - COUNT 20
   ```

#### Root Cause Analysis
| **Symptom** | **Likely Cause** | **Fix** |
|-------------|------------------|---------|
| Switches between 2 models | Bad routing rule threshold | Update `model_router_rules.yml` |
| Random switching across all | Market halt/low volume | Enable "market hours only" mode |
| High latency in switches | Model server overload | Scale model instances |
| Same-symbol oscillation | Confidence threshold too low | Increase min confidence to 0.6 |

#### Escalation
- **< 5 min:** On-call engineer handles
- **5-15 min:** Page Quant team for rule review  
- **> 15 min:** Page PM + Quant for emergency rules

---

### 🚨 **CRITICAL: VaR Breach > 90% Target**
**Alert:** `var_pct:* > 1.8` (90% of 2.0% limit)  
**Severity:** Critical  
**Target MTTR:** 3 minutes

#### Immediate Actions
1. **Reduce Kelly Fraction** to emergency level:
   ```bash
   # Emergency VaR reduction
   curl -X POST http://localhost:8080/api/risk/emergency-reduce \
     -d '{"kelly_fraction": 0.1, "reason": "VaR breach"}'
   ```

2. **Check Position Concentration:**
   ```bash
   redis-cli HGETALL metrics:position_size_usd:BTC-USD
   redis-cli HGETALL metrics:position_size_usd:ETH-USD
   ```

3. **Review Recent Edge Changes:**
   - Navigate to Edge Risk Dashboard → Time Series panel
   - Look for edge spikes in last 15 minutes

#### Root Cause Analysis
| **Symptom** | **Likely Cause** | **Fix** |
|-------------|------------------|---------|
| Single symbol high VaR | Concentrated position | Reduce position limit for symbol |
| All symbols elevated | Market volatility spike | Lower global Kelly to 0.15 |
| VaR after model switch | New model over-confident | Add model confidence penalty |
| Gradual VaR increase | Stale volatility estimates | Force refresh vol calculations |

#### Escalation  
- **< 2 min:** Auto-reduce positions via circuit breaker
- **2-5 min:** Page PM immediately
- **> 5 min:** Halt all trading, escalate to CRO

---

### ⚠️ **WARNING: Redis Latency > 5ms p99**
**Alert:** `redis_latency_p99 > 5000` (microseconds)  
**Severity:** Warning  
**Target MTTR:** 8 minutes

#### Immediate Actions
1. **Check Redis Stats:**
   ```bash
   redis-cli INFO stats | grep instantaneous
   redis-cli LATENCY HISTORY command
   redis-cli --latency-history
   ```

2. **Identify Slow Operations:**
   ```bash
   redis-cli SLOWLOG GET 10
   redis-cli CLIENT LIST
   ```

3. **Failover if Needed:**
   ```bash
   # Switch to Redis replica
   export REDIS_URL=redis://redis-replica:6379/0
   docker-compose restart risk-harmoniser strategist
   ```

#### Root Cause Analysis
| **Symptom** | **Likely Cause** | **Fix** |
|-------------|------------------|---------|
| High memory usage | Redis swapping to disk | Add more RAM or enable Redis maxmemory |
| Slow commands in log | Large TimeSeries operations | Reduce TS retention period |
| Network timeouts | Network congestion | Switch to local Redis instance |
| CPU spikes | Too many concurrent clients | Limit Redis connection pool |

#### Escalation
- **< 5 min:** Restart Redis with config tuning
- **5-10 min:** Page DevOps for infrastructure scaling
- **> 10 min:** Consider emergency datacenter failover

---

## 4. Hot-Reload Procedure

### 🔄 **Parameter Server Hot-Reload**
**Purpose:** Update risk parameters without stopping trading

```bash
# 1. Edit configuration
vi conf/risk_params.yml

# 2. Validate YAML syntax  
python -c "import yaml; yaml.safe_load(open('conf/risk_params.yml'))"

# 3. Trigger hot-reload
redis-cli PUBLISH param.reload '{"component": "risk_harmoniser", "config_path": "conf/risk_params.yml"}'

# 4. Verify reload success (should see log entry)
tail -f logs/trading-system.log | grep "Configuration reloaded"

# 5. Check new parameters in Grafana (Edge Risk dashboard)
```

### 🎯 **Model Router Rules Hot-Reload**
```bash
# 1. Edit routing rules
vi model_router_rules.yml

# 2. Validate rules format
python scripts/validate_router_rules.py model_router_rules.yml

# 3. Hot-reload router
redis-cli PUBLISH param.reload '{"component": "model_router", "config_path": "model_router_rules.yml"}'

# 4. Monitor next few model switches for correctness
redis-cli XREAD STREAMS model.switch.log $ BLOCK 30000
```

### ⚠️ **Hot-Reload Safety Checks**
- [ ] **Never** hot-reload during high volatility (VaR > 1.5%)
- [ ] **Always** validate YAML/JSON syntax before reload
- [ ] **Monitor** next 10 trades for unexpected behavior
- [ ] **Rollback** immediately if anything seems wrong

---

## 5. Release Cut Steps

### 📦 **Version Tag & Deploy Procedure**

```bash
# 1. Ensure develop branch is clean
git checkout develop
git pull origin develop
git status  # Should be clean

# 2. Run full test suite
make test-all
make lint-check
make security-scan

# 3. Update version in key files
vi src/__init__.py  # Update __version__
vi docker-compose.yml  # Update image tags
vi README.md  # Update badges

# 4. Create and push tag
git tag -a v0.4.0-rc3 -m "Release v0.4.0-rc3: Grafana Dashboard + Risk Harmoniser"
git push origin v0.4.0-rc3

# 5. Build Docker images
docker build -t trading-system:v0.4.0-rc3 .
docker tag trading-system:v0.4.0-rc3 trading-system:latest

# 6. Deploy to staging first
export ENVIRONMENT=staging
docker-compose -f docker-compose.staging.yml pull
docker-compose -f docker-compose.staging.yml up -d

# 7. Run smoke tests on staging
curl -s http://staging:8080/health
./scripts/smoke_test.sh staging

# 8. Deploy to production (after staging verification)
export ENVIRONMENT=production  
docker-compose pull
docker-compose up -d

# 9. Monitor production deployment
watch -n 5 'curl -s http://localhost:8080/health'
```

### 🔍 **Post-Deploy Verification Checklist**
- [ ] All health endpoints return 200
- [ ] Grafana dashboards showing live data
- [ ] No new error logs in last 10 minutes
- [ ] Model switching functioning normally
- [ ] Position sizing within expected ranges
- [ ] VaR calculations operating correctly

---

## 6. FAQs & Gotchas

### ❓ **Common Issues**

#### **Q: TimeSeries module missing error**
```
WARNING: Connected to Redis (TimeSeries module not available - using fallback)
```
**A:** Redis TimeSeries module not installed. Install with:
```bash
# Option 1: Docker with TimeSeries
docker run -p 6379:6379 redislabs/redistimeseries:latest

# Option 2: Load module manually  
redis-cli MODULE LOAD /usr/lib/redis/modules/redistimeseries.so
```

#### **Q: Grafana dashboards show "No data"**
**A:** Check data flow:
```bash
# 1. Verify Redis has data
redis-cli KEYS "edge_blended_bps:*"
redis-cli HGETALL "metrics:edge_blended_bps:BTC-USD"

# 2. Check TimeSeries writer
python -c "from src.monitoring.write_timeseries import get_timeseries_writer; w=get_timeseries_writer(); print(w.health_check())"

# 3. Verify Grafana Redis datasource config
curl -s http://admin:admin@localhost:3000/api/datasources
```

#### **Q: Model switching too frequent**
**A:** Adjust confidence thresholds:
```yaml
# In model_router_rules.yml
confidence_thresholds:
  min_confidence: 0.65  # Increase from 0.55
  switch_hysteresis: 0.05  # Add switching delay
```

#### **Q: VaR calculations seem wrong**
**A:** Check volatility estimates:
```bash
# View current vol estimates
redis-cli HGETALL "volatility_estimates"

# Force refresh from market data
curl -X POST http://localhost:8080/api/risk/refresh-volatility
```

### 🔐 **TLS Certificate Renewals**
**Auto-renewal with certbot:**
```bash
# Check certificate expiry
openssl x509 -in /etc/ssl/certs/trading-system.crt -text -noout | grep "Not After"

# Renew certificates (if expiring in < 30 days)
certbot renew --dry-run
certbot renew

# Restart services to pick up new certs
docker-compose restart nginx
```

### 🚦 **Circuit Breaker States**
| **State** | **Trigger** | **Action** |
|-----------|-------------|------------|
| **CLOSED** | Normal operations | All trading enabled |
| **OPEN** | VaR > 2.5% OR latency > 100ms | Stop all new positions |
| **HALF_OPEN** | Manual reset after fix | Limited trading to test |

**Reset circuit breaker:**
```bash
curl -X POST http://localhost:8080/api/circuit-breaker/reset
```

### 📝 **Log File Locations**
| **Component** | **Log Path** | **Rotation** |
|---------------|--------------|--------------|
| **Main System** | `logs/trading-system.log` | Daily, 7 days |
| **Risk Harmoniser** | `logs/risk-harmoniser.log` | Daily, 7 days |
| **Model Router** | `logs/model-router.log` | Daily, 7 days |
| **Redis** | `logs/redis.log` | Daily, 30 days |
| **Nginx** | `logs/access.log` | Daily, 14 days |

### 🧪 **Testing Commands**
```bash
# Unit tests
make test-unit

# Integration tests  
make test-integration

# Performance tests
make test-performance

# End-to-end smoke test
make test-smoke

# Load test with sample data
make test-load DURATION=60s RATE=100rps
```

---

## 🆘 Emergency Contacts

| **Escalation Level** | **Contact** | **Response Time** |
|---------------------|-------------|-------------------|
| **L1 - On-Call Engineer** | PagerDuty: trading-oncall | 5 minutes |
| **L2 - Quant Team** | Slack: #quant-emergency | 15 minutes |
| **L3 - PM + Risk** | Phone: [Update numbers] | 30 minutes |
| **L4 - CTO/CRO** | Emergency hotline | 1 hour |

**After-hours emergency:** Always start with PagerDuty, escalate every 15 minutes if no response.

---

**🔄 This runbook is updated with every release. Always check version matches your deployment.** 