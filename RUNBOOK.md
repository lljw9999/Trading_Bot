# Trading System Runbook

**Quick Reference for Incident Response and System Operations**

## 🤖 RL Policy Operations

### Quick RL Status Check
```bash
# Policy heartbeat age (should be <1h)
redis-cli GET policy:last_update_ts | xargs -I {} date -d @{}

# Current influence (should be 0%)
redis-cli GET policy:allowed_influence_pct

# Gate status
ls -la artifacts/*/rl/gate_report.md | tail -1
```

### 🚨 RLPolicyStale Alert
**Response Time: <15 minutes**

1. **Check policy daemon:**
   ```bash
   systemctl status rl-policy
   journalctl -u rl-policy --since "1 hour ago"
   ```

2. **Restart if needed:**
   ```bash
   systemctl restart rl-policy
   ```

3. **Ensure shadow mode:**
   ```bash
   make promote-zero
   ```

### 📊 RL Monitoring Components
- **Exporter:** `curl localhost:9108/metrics`
- **Watchdog:** `systemctl status rl-watchdog.timer`
- **Validation:** `make validate-48h-now`
- **Gate Reports:** `artifacts/*/rl/gate_report.md`

## 🚨 Emergency Procedures

### Immediate Response (< 2 minutes)

1. **Stop Trading Activity**
   ```bash
   # Set system to halt mode
   /mode halt
   
   # Or via CLI
   python3 scripts/set_mode.py halt
   ```

2. **Capture Incident Snapshot**
   ```bash
   # Via Slack (recommended)
   /snapshot now
   
   # Or via CLI
   python3 scripts/capture_state.py --capture
   ```

3. **Check System Health**
   ```bash
   # Via Slack
   /health
   
   # Or check key metrics
   redis-cli get mode
   redis-cli get risk:capital_effective
   redis-cli hgetall risk:stats
   ```

### System Recovery (< 10 minutes)

4. **Identify Issue Scope**
   - Check logs: `tail -f /var/log/trader/*.log`
   - Check service status: `systemctl status trading_bot ops_bot`
   - Check Redis connectivity: `redis-cli ping`

5. **Rollback if Needed**
   ```bash
   # Switch to stable deployment
   /canary blue  # or green depending on stable version
   
   # Or restart services
   sudo systemctl restart trading_bot
   sudo systemctl restart ops_bot
   ```

6. **Resume Operations**
   ```bash
   # Gradually restore trading
   /mode paper    # Test mode first
   /mode auto     # Resume live trading when confident
   ```

---

## 🛡️ Safety Controls & Feature Flags

### Environment Toggles

**DRY_RUN Control (Executor Safety)**
```bash
# Default: DRY_RUN=1 (safe, no real orders)
export DRY_RUN=1

# For production trading (requires explicit setting)
export DRY_RUN=0  # ⚠️ ENABLES REAL MONEY ORDERS
export EXEC_MODE=live
```

**NOWNodes WebSocket Control**
```bash
# Default: USE_NOWNODES=0 (disabled, simulation mode)
export USE_NOWNODES=0

# To enable NOWNodes websocket connections
export USE_NOWNODES=1
```

**Integration Mocks**
```bash
# Default mocks used in CI
export OPENAI_MOCK=1   # Skip live OpenAI calls
export REDIS_MOCK=0    # Use real Redis when available
# Optional extras test suites
export RUN_ML_TESTS=1      # Enable ML ensemble tests
export RUN_ONNX_TESTS=1    # Enable ONNX latency benchmarks
```

### Safety Defaults
- **Executor Mode**: Defaults to `dry_run=True` to prevent accidental real orders
- **NOWNodes Connector**: Disabled by default to avoid websocket deprecation warnings
- **Live Trading**: Requires explicit `DRY_RUN=0` environment variable
- **Optional Extras**: Enable `RUN_ML_TESTS=1` or `RUN_ONNX_TESTS=1` when extras are installed

---

## 🧪 Testing Strategy

### Staged Test Matrix

**Stage 1: Unit Tests (Fast)**
```bash
# Run unit tests only (no external dependencies)
pytest -m "not integration and not soak" --maxfail=5 --durations=20

# Or use the CI helper
./run_ci_tests.sh unit
```

**Stage 2: Integration Tests**
```bash
# Requires Redis/API mocks (docker-compose.ci.yml)
docker compose -f docker-compose.ci.yml up -d
pytest -m "integration and not soak" --maxfail=5 --durations=20
docker compose -f docker-compose.ci.yml down
```

**Stage 3: Soak Tests (Nightly)**
```bash
# Long-running resilience tests (nightly CI cron)
pip install .[ml,onnx,bandits]
pytest -m "soak" --maxfail=5 --durations=20
```

The GitHub Actions workflow `.github/workflows/tests.yml` orchestrates these
stages automatically (unit → integration → soak-on-cron) and caches pip wheels
for repeatable runtimes.

### Test Categories
- **`integration`**: Requires external services (Redis, APIs)
- **`ml`**: Needs xgboost/lightgbm libraries
- **`onnx`**: Needs onnxruntime
- **`soak`**: Long-running tests
- **`nownodes`**: Requires USE_NOWNODES=1 and websockets library

### Coverage Testing
```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage (75% threshold)
pytest --cov=src --cov-fail-under=75 --cov-report=term-missing
```

---

## 🔧 System Architecture

### Core Components

- **Trading Bot** (`trading_bot.service`): Main trading engine
- **Ops Bot** (`ops_bot.service`): Slack integration & controls  
- **Risk Monitor** (`risk_monitor.service`): Risk management daemon
- **Redis**: State storage and pub/sub messaging
- **Grafana**: Monitoring and alerting dashboard

### Key File Locations

```
/opt/trader/                    # Main application directory
├── src/                        # Source code
├── scripts/                    # Operational scripts  
├── logs/                       # Application logs
├── reports/                    # Generated reports
├── grafana/                    # Dashboard configs
└── CLAUDE.md                   # System memory/context

/var/log/trader/               # System logs
/tmp/snapshot_*.tgz            # Incident snapshots
```

---

## 💻 Slack Commands

### System Control
- `/status` - Full system status report
- `/health` - Quick health check
- `/mode [auto|manual|paper|halt]` - Set trading mode

### Deployment Management
- `/canary [blue|green]` - Switch deployment color
- `/capital stage [10|20|30|50|70|100]` - Stage capital allocation

### Incident Response
- `/snapshot now` - Capture full system state

---

## 📊 Key Metrics & Thresholds

### Critical Alerts
- **P&L Drawdown** > -5%: Review positions
- **Fill Rate** < 80%: Check venue connectivity
- **Latency** > 1000ms: Check network/API health
- **SLO Tier C**: Capital staging disabled

### Normal Operating Ranges
- **Capital Effective**: 20-80%
- **Strategy Fill Rate**: 85-95%
- **Venue Latency**: 50-500ms
- **Redis Memory**: < 1GB

---

## 🔍 Troubleshooting

### Common Issues

**Trading Bot Not Responding**
```bash
# Check service status
sudo systemctl status trading_bot

# Check logs
tail -f /var/log/trader/trading_bot.log

# Restart if needed
sudo systemctl restart trading_bot
```

**Redis Connection Issues**
```bash
# Check Redis status
sudo systemctl status redis

# Test connectivity
redis-cli ping

# Check memory usage
redis-cli info memory
```

**High Latency/Timeouts**
```bash
# Check network connectivity
ping api.binance.com
ping api.coinbase.com

# Check DNS resolution
nslookup api.binance.com

# Monitor network traffic
sudo netstat -i
```

**Deployment Issues**
```bash
# Check current deployment
redis-cli get mode:active_color

# Switch to stable version
python3 scripts/canary_switch.py blue

# Verify switch
redis-cli get mode:active_color
```

---

## 📋 Daily Operations

### Morning Checklist (9 AM UTC)
- [ ] Check overnight P&L: `redis-cli get pnl:net:today`
- [ ] Review risk metrics: `/health` in Slack
- [ ] Check venue scorecards: Grafana TCA dashboard
- [ ] Verify capital allocation: `redis-cli get risk:capital_effective`

### End of Day (6 PM UTC)
- [ ] Generate daily P&L report (automatic at 23:59 UTC)
- [ ] Review execution quality metrics
- [ ] Check system resource usage
- [ ] Archive important logs if needed

### Weekly Review
- [ ] Review capital staging approvals
- [ ] Update venue weights based on TCA analysis
- [ ] Check incident snapshot retention
- [ ] Review and update runbook if needed

---

## 📞 Escalation Contacts

### Primary On-Call
- **Trading Operations**: trading-ops@company.com
- **Slack Channel**: #trading-alerts

### Secondary Contacts
- **Infrastructure**: infra-team@company.com  
- **Risk Management**: risk-team@company.com
- **Compliance**: compliance@company.com

### Vendor Support
- **Redis Enterprise**: [Support Portal](https://support.redis.com)
- **AWS Support**: [Console](https://support.console.aws.amazon.com)
- **Exchange APIs**: Check individual exchange support channels

---

## 📚 Additional Resources

### Documentation
- **System Architecture**: `/opt/trader/docs/architecture.md`
- **API Documentation**: `/opt/trader/docs/api/`
- **Deployment Guide**: `/opt/trader/docs/deployment.md`

### Monitoring Dashboards
- **Main Dashboard**: http://localhost:3000/d/trading-main
- **Risk Dashboard**: http://localhost:3000/d/risk-overview  
- **TCA Dashboard**: http://localhost:3000/d/tca-scorecard
- **Capital Staging**: http://localhost:3000/d/capital-staging

### Log Analysis Tools
```bash
# Search for errors in last hour
grep -i error /var/log/trader/*.log | grep $(date +%Y-%m-%d\ %H)

# Find high latency events
grep "latency.*[5-9][0-9][0-9]ms" /var/log/trader/*.log

# Check fill rates by venue
grep "fill_rate" /var/log/trader/*.log | tail -20
```

---

## ⚡ Quick Commands

```bash
# Emergency stop
redis-cli set mode halt

# Check critical metrics
redis-cli get risk:capital_effective
redis-cli get pnl:net:today  
redis-cli hget risk:stats drawdown

# View active positions
redis-cli get portfolio_positions

# Check service health
systemctl is-active trading_bot ops_bot redis

# Capture incident snapshot
python3 scripts/capture_state.py --capture
```

---

*Last Updated: 2025-01-12*  
*Runbook Version: 1.0.0*  
*Emergency Contact: trading-ops@company.com*
