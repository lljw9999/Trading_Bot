#!/usr/bin/env bash
#
# Trading System Health Check Watchdog
# 
# Continuous monitoring loop that checks system health and exits with
# non-zero status to trigger PagerDuty alerts when issues are detected.
#
# Usage: ./scripts/healthcheck_loop.sh
# Exit codes: 0=healthy, 1=critical issue detected

set -euo pipefail

# Configuration
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"  # seconds between checks
MAX_FAILURES="${MAX_FAILURES:-3}"       # consecutive failures before alert
LOG_FILE="${LOG_FILE:-logs/healthcheck.log}"  # Use accessible logs directory

# Counters
failure_count=0
check_count=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log_message "INFO" "$1"
}

log_warn() {
    log_message "WARN" "${YELLOW}$1${NC}"
}

log_error() {
    log_message "ERROR" "${RED}$1${NC}"
}

log_success() {
    log_message "SUCCESS" "${GREEN}$1${NC}"
}

# Health check functions
check_docker_services() {
    local required_services=("trading_redis" "trading_grafana" "trading_prometheus")
    
    for service in "${required_services[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            log_error "Docker service not running: ${service}"
            return 1
        fi
    done
    
    log_info "✅ All required Docker services running"
    return 0
}

check_grafana_health() {
    local grafana_url="http://localhost:3000/api/health"
    
    if ! curl -sf "${grafana_url}" >/dev/null 2>&1; then
        log_error "Grafana health check failed"
        return 1
    fi
    
    log_info "✅ Grafana healthy"
    return 0
}

check_redis_health() {
    if ! docker exec trading_redis redis-cli ping | grep -q "PONG"; then
        log_error "Redis health check failed"
        return 1
    fi
    
    log_info "✅ Redis healthy"
    return 0
}

check_prometheus_health() {
    local prom_url="http://localhost:9090/-/healthy"
    
    if ! curl -sf "${prom_url}" >/dev/null 2>&1; then
        log_error "Prometheus health check failed"
        return 1
    fi
    
    log_info "✅ Prometheus healthy"
    return 0
}

check_metrics_freshness() {
    # Check if metrics are being written to Redis
    local latest_metrics=$(docker exec trading_redis redis-cli --scan --pattern "*:*" | wc -l)
    
    if [ "${latest_metrics}" -lt 5 ]; then
        log_warn "Low metrics count: ${latest_metrics} (expected: ≥5)"
        return 1
    fi
    
    log_info "✅ Metrics freshness OK (${latest_metrics} active metrics)"
    return 0
}

check_memory_usage() {
    # Check Docker container memory usage
    local high_memory_containers=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | \
        awk 'NR>1 {gsub(/[A-Za-z%]/, "", $2); if ($2 > 1000) print $1}')
    
    if [ -n "${high_memory_containers}" ]; then
        log_warn "High memory usage detected in containers: ${high_memory_containers}"
        return 1
    fi
    
    log_info "✅ Memory usage within limits"
    return 0
}

# Main health check function
run_health_checks() {
    local checks_passed=0
    local total_checks=6
    
    log_info "🔍 Starting health check cycle #${check_count}"
    
    # Run all health checks
    check_docker_services && ((checks_passed++)) || true
    check_grafana_health && ((checks_passed++)) || true
    check_redis_health && ((checks_passed++)) || true
    check_prometheus_health && ((checks_passed++)) || true
    check_metrics_freshness && ((checks_passed++)) || true
    check_memory_usage && ((checks_passed++)) || true
    
    log_info "Health check summary: ${checks_passed}/${total_checks} passed"
    
    # Return success if all critical checks pass (allow 1 warning)
    if [ "${checks_passed}" -ge $((total_checks - 1)) ]; then
        return 0
    else
        return 1
    fi
}

# Signal handlers
cleanup() {
    log_info "🛑 Healthcheck watchdog stopping (received signal)"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main monitoring loop
main() {
    log_info "🚀 Starting Trading System Healthcheck Watchdog"
    log_info "Check interval: ${CHECK_INTERVAL}s, Max failures: ${MAX_FAILURES}"
    
    # Start memory probe in background
    if [[ -f "scripts/mem_probe.py" ]]; then
        log_info "🔍 Starting memory probe in background..."
        python scripts/mem_probe.py --background &
        mem_probe_pid=$!
        log_info "Memory probe started with PID: ${mem_probe_pid}"
    else
        log_warn "Memory probe script not found, skipping memory monitoring"
    fi
    
    while true; do
        ((check_count++))
        RC=0
        
        if run_health_checks; then
            failure_count=0
            log_success "✅ System healthy (check #${check_count})"
        else
            ((failure_count++))
            RC=1
            log_error "❌ Health check failed (${failure_count}/${MAX_FAILURES})"
            
            if [ "${failure_count}" -ge "${MAX_FAILURES}" ]; then
                log_error "🚨 CRITICAL: ${MAX_FAILURES} consecutive failures detected!"
                log_error "Triggering PagerDuty alert and exiting..."
                
                # Send Slack alert for critical failure
                if command -v ./scripts/send_slack.sh >/dev/null 2>&1; then
                    ./scripts/send_slack.sh ":x: CRITICAL: ${MAX_FAILURES} consecutive healthcheck failures | Triggering PagerDuty"
                fi
                
                exit 1
            fi
        fi
        
        # Send Slack alert on any healthcheck failure
        if [[ $RC -ne 0 ]] && command -v ./scripts/send_slack.sh >/dev/null 2>&1; then
            ./scripts/send_slack.sh ":warning: Healthcheck fail RC=$RC | Check #${check_count} | Failures: ${failure_count}/${MAX_FAILURES}"
        fi
        
        sleep "${CHECK_INTERVAL}"
    done
}

# Run main function
main "$@" 