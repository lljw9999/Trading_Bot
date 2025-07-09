# Model Router & Param Server v1 Documentation

## Overview

The Param Server v1 provides hot-reloading parameter management for the Model Router with sub-100ms latency. It enables dynamic routing rule updates without system restart, supporting file-watch triggers and Redis pub/sub notifications.

## 📊 Grafana Dashboard Integration

### Model Router Dashboard Panels

The Model Router integrates with Grafana for real-time monitoring and observability:

**Dashboard URL:** http://localhost:3000/d/model-router/model-router-dashboard

**Key Panels:**
- **Panel 1:** Active Model Distribution (pie chart)
- **Panel 2:** Routing Latency (time series, target: ≤50µs)  
- **Panel 3:** Model Switch Rate (rate gauge, alert: >20/5min)
- **Panel 4:** Configuration Reload Events (event log)
- **Panel 5:** Rule Match Performance (heatmap by asset class)
- **Panel 6:** Failover/Fallback Events (alert panel)

**Metrics Sources:**
```
model_router_active_model{asset_class, horizon_bucket}
model_router_latency_microseconds
model_router_reload_duration_ms
model_router_rule_matches_total{rule_index, model}
```

### Switch Log Integration

Model switching events are automatically logged to the **Edge Risk Dashboard Switch Log** panel:

- **Panel Location:** Edge Risk Dashboard → Row 3 (Switch Log)
- **Stream:** `model.switch.log` Redis stream
- **Fields:** `timestamp`, `symbol`, `old_model`, `new_model`, `latency_ms`
- **Real-time Updates:** Sub-2s latency from switch to display

### Alert Integration

**Excessive Model Switching Alert:**
- **Trigger:** `rate(model_switch_total[5m]) > 20`
- **Dashboard Panel:** Edge Risk Dashboard → Row 4 → Panel 2
- **Runbook:** See `docs/runbook.md` → Section 3 → Model Switching Playbook

### Screenshot Capture

Generate dashboard screenshots for documentation:
```bash
# Capture routing latency panel
./scripts/capture_grafana.sh model-router 2 docs/images/router-latency.png

# Capture full dashboard
./scripts/capture_grafana.sh model-router full docs/images/router-overview.png

# High-resolution for presentations
./scripts/capture_grafana.sh model-router 1 router-hires.png --width 1920 --height 1080
```

## YAML Configuration Specification

### Basic Structure

```yaml
model_router:
  rules:
    - match:
        asset_class: "crypto"
        horizon_ms: "<60000"
      model: "tlob_tiny"
    - match:
        asset_class: "crypto"
        horizon_ms: ">=60000 & <7200000"
      model: "patchtst_small"
  default_model: "timesnet_base"
```

### Asset Class Patterns

The `asset_class` field supports basic glob patterns:

- `"crypto"` - Exact match for crypto assets
- `"crypto*"` - Matches crypto, cryptos, cryptocurrency, etc.
- `"*stock"` - Matches us_stocks, a_shares, etc.
- `"*"` - Wildcard matches any asset class

### Horizon Range Syntax

The `horizon_ms` field supports comparison operators and ranges:

- `"<60000"` - Less than 60 seconds
- `">=60000"` - Greater than or equal to 60 seconds
- `">=60000 & <7200000"` - Range: 1 minute to 2 hours
- `">7200000"` - Greater than 2 hours
- `"*"` - Wildcard matches any horizon

### Complete Configuration Schema

```yaml
model_router:
  rules:
    - match:
        asset_class: "crypto"
        horizon_ms: "<60000"
      model: "tlob_tiny"
      priority: 10
      description: "High-frequency crypto microstructure"
    
    - match:
        asset_class: "us_stocks"
        horizon_ms: ">=14400000"
      model: "mamba_ts_small" 
      priority: 50
      description: "US equity overnight/swing trading"

  config:
    default_model: "tlob_tiny"
    cache_ttl_seconds: 300
    redis_url: "redis://localhost:6379/0"
    performance_logging: true
    max_latency_us: 50

  model_thresholds:
    tlob_tiny:
      max_latency_ms: 3.0
      min_accuracy: 0.52
    patchtst_small:
      max_latency_ms: 10.0
      min_accuracy: 0.54

  reload:
    enabled: true
    signal: "SIGHUP"
    validation: true
    backup_on_reload: true
```

## Hot-Reload Workflow

### Automatic File Watching

1. **File Change Detection**: Watchdog monitors `conf/model_router_rules.yml`
2. **Debouncing**: 100ms debounce to prevent rapid-fire reloads
3. **Validation**: Pydantic validates new configuration
4. **Atomic Swap**: Thread-safe replacement of active rules
5. **Notification**: Redis pub/sub message to `param.reload` channel

### Manual Reload Triggers

```bash
# Method 1: Send SIGHUP signal
kill -HUP <router_process_pid>

# Method 2: Touch configuration file
touch conf/model_router_rules.yml

# Method 3: Programmatic reload
param_server.reload_config()
```

### Performance Targets

- **Reload Latency**: < 100ms (target: ~50ms)
- **Router Latency**: ≤ 50µs per `select_model()` call
- **Memory Overhead**: Zero allocation fastpath after reload
- **Availability**: No downtime during configuration updates

## Integration Example

```python
from src.core.param_server import ParamServer, create_param_server

# Create param server with file watching
param_server = create_param_server(
    config_path="conf/model_router_rules.yml",
    redis_url="redis://localhost:6379/0"
)

# Start hot-reload watching
with param_server:
    # Get current rules (< 50µs)
    rules = param_server.get_rules()
    
    # Router will automatically use updated rules
    # when configuration file changes
```

## Failure Modes & Fallback Behavior

### Configuration Validation Failures

**Scenario**: Invalid YAML syntax or schema violations

**Behavior**: 
- Keep previous valid configuration active
- Log error with validation details  
- Publish failed reload event to Redis
- No disruption to active routing

**Example Error Handling**:
```python
# Invalid config is rejected, old rules preserved
success = param_server.reload_config()
if not success:
    # Previous rules still active
    rules = param_server.get_rules()  # Returns old valid rules
```

### File System Issues

**Scenario**: Configuration file deleted or permission denied

**Behavior**:
- Retain last successfully loaded configuration
- Continue operating with cached rules
- Log warnings about file access issues
- Resume watching when file becomes available

### Redis Connection Failures

**Scenario**: Redis unavailable or connection timeout

**Behavior**:
- Local hot-reload continues functioning
- No pub/sub notifications sent
- Fallback to local-only operation
- Automatic reconnection attempts

### Memory/Resource Exhaustion

**Scenario**: System under memory pressure

**Behavior**:
- Graceful degradation to previous configuration
- Skip non-essential features (performance logging)
- Force garbage collection of old rule sets
- Maintain core routing functionality

### Performance Degradation

**Scenario**: Reload latency exceeds 100ms target

**Monitoring**:
```python
stats = param_server.get_performance_stats()
if stats["avg_load_time_ms"] > 100:
    # Alert: Performance degradation detected
    logger.warning(f"Reload latency: {stats['avg_load_time_ms']:.1f}ms")
```

**Mitigation**:
- Reduce configuration complexity
- Optimize rule priority ordering
- Enable Redis-based rule caching
- Consider configuration partitioning

## Monitoring & Observability

### Performance Metrics

```python
stats = param_server.get_performance_stats()
{
    "load_count": 15,
    "avg_load_time_ms": 45.2,
    "current_rules_count": 8,
    "recent_reload_success_rate": 1.0,
    "recent_events": [...]
}
```

### Redis Pub/Sub Events

**Channel**: `param.reload`

**Message Format**:
```json
{
    "component": "router",
    "timestamp": 1640995200.123,
    "config_hash": "abc123def456",
    "rules_count": 8,
    "latency_ms": 47.3,
    "success": true
}
```

### Health Checks

```python
# Validate configuration without loading
is_valid, error = param_server.validate_config_file()
if not is_valid:
    logger.error(f"Config validation failed: {error}")

# Check reload event history
events = param_server.get_reload_events(limit=10)
failed_reloads = [e for e in events if not e.success]
```

## Production Deployment

### Configuration Management

1. **Version Control**: Store `model_router_rules.yml` in Git
2. **Deployment Pipeline**: Validate config before deployment
3. **Rollback Strategy**: Keep previous config versions
4. **Testing**: Validate rules against historical data

### Monitoring Setup

1. **Grafana Dashboard**: Track reload latency and success rate
2. **Alerting**: Alert on reload failures or performance degradation  
3. **Logging**: Structured logs for config changes and errors
4. **Health Checks**: Regular validation of active configuration

### Security Considerations

1. **File Permissions**: Restrict write access to config file
2. **Input Validation**: Pydantic schema prevents injection
3. **Resource Limits**: Monitor memory usage during reloads
4. **Audit Trail**: Log all configuration changes with timestamps

This completes the Param Server v1 implementation, providing robust hot-reloading capabilities for the Model Router with comprehensive error handling and monitoring. 