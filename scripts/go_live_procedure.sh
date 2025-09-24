#!/usr/bin/env bash
# Go-Live Procedure - Guarded Production Deployment
# Executes comprehensive checks and deploys model with safety guardrails

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INITIAL_INFLUENCE_PCT=10
MONITORING_PERIOD_MIN=120
AUDIT_DIR="$PROJECT_ROOT/artifacts/audit"

echo -e "${BLUE}🚀 SOL RL Policy Go-Live Procedure${NC}"
echo "================================================================"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Project Root: $PROJECT_ROOT"
echo "Initial Influence: $INITIAL_INFLUENCE_PCT%"
echo "================================================================"

# Function to log with timestamp
log() {
    echo -e "[$(date -u '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to create audit record
create_audit() {
    local action="$1"
    local details="$2"
    local timestamp=$(date -u '+%Y-%m-%dT%H-%M-%SZ')
    local audit_file="$AUDIT_DIR/${timestamp}_go_live_${action}.json"
    
    mkdir -p "$AUDIT_DIR"
    cat > "$audit_file" << EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')",
  "action": "go_live_$action",
  "details": $details,
  "operator": "${USER:-unknown}",
  "hostname": "$(hostname)",
  "git_sha": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "environment": {
    "go_live_flag": "${GO_LIVE:-unset}",
    "pwd": "$(pwd)"
  }
}
EOF
    echo "$audit_file"
}

# Check prerequisites
check_prerequisites() {
    log "${YELLOW}📋 Checking prerequisites...${NC}"
    
    # Check GO_LIVE environment variable
    if [[ "${GO_LIVE:-}" != "1" ]]; then
        log "${RED}❌ GO_LIVE environment variable must be set to 1${NC}"
        echo "Set it with: export GO_LIVE=1"
        exit 1
    fi
    log "${GREEN}✅ GO_LIVE flag is set${NC}"
    
    # Check we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/pilot/pilot_run.yaml" ]]; then
        log "${RED}❌ Not in correct project directory (missing pilot/pilot_run.yaml)${NC}"
        exit 1
    fi
    log "${GREEN}✅ Project directory validated${NC}"
    
    # Check required scripts exist
    local required_scripts=(
        "scripts/pilot_guard.py"
        "scripts/kill_switch.py"
        "scripts/go_nogo_check.py"
        "scripts/preflight_release_check.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$script" ]]; then
            log "${RED}❌ Missing required script: $script${NC}"
            exit 1
        fi
    done
    log "${GREEN}✅ All required scripts present${NC}"
}

# Run release gates
run_release_gates() {
    log "${YELLOW}🚦 Running release gates...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Run preflight checks
    log "  Running preflight release checks..."
    if ! PYTHONPATH="$PROJECT_ROOT" python scripts/preflight_release_check.py; then
        log "${RED}❌ Preflight release checks failed${NC}"
        create_audit "preflight_fail" '{"reason": "preflight_checks_failed"}'
        exit 1
    fi
    log "${GREEN}✅ Preflight checks passed${NC}"
    
    # Run Go/No-Go decision
    log "  Running Go/No-Go decision check..."
    if ! PYTHONPATH="$PROJECT_ROOT" python scripts/go_nogo_check.py; then
        log "${RED}❌ Go/No-Go check failed${NC}"
        create_audit "gonogo_fail" '{"reason": "gonogo_check_failed"}'
        exit 1
    fi
    log "${GREEN}✅ Go/No-Go check passed${NC}"
    
    create_audit "gates_passed" '{"preflight": "pass", "gonogo": "pass"}'
}

# Execute guarded ramp to initial influence
execute_ramp() {
    log "${YELLOW}📈 Executing guarded ramp to $INITIAL_INFLUENCE_PCT%...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Set environment for pilot guard
    export REASON="Go-live procedure: initial canary deployment"
    
    # Execute pilot guard with target percentage
    if ! PYTHONPATH="$PROJECT_ROOT" python scripts/pilot_guard.py --target-pct "$INITIAL_INFLUENCE_PCT"; then
        log "${RED}❌ Pilot guard blocked ramp to $INITIAL_INFLUENCE_PCT%${NC}"
        create_audit "ramp_blocked" "{\"target_pct\": $INITIAL_INFLUENCE_PCT, \"reason\": \"pilot_guard_blocked\"}"
        exit 1
    fi
    
    log "${GREEN}✅ Successfully ramped to $INITIAL_INFLUENCE_PCT% influence${NC}"
    create_audit "ramp_success" "{\"target_pct\": $INITIAL_INFLUENCE_PCT, \"status\": \"live_canary\"}"
}

# Display monitoring information
display_monitoring() {
    log "${YELLOW}📊 Live canary deployment active${NC}"
    
    echo ""
    echo "================================================================"
    echo -e "${GREEN}🎉 GO-LIVE SUCCESSFUL${NC}"
    echo "================================================================"
    echo -e "Status: ${GREEN}LIVE CANARY${NC} at ${INITIAL_INFLUENCE_PCT}% influence"
    echo "Monitoring Period: $MONITORING_PERIOD_MIN minutes"
    echo ""
    echo "📊 MONITORING DASHBOARDS:"
    echo "  • Grafana RL Policy:  http://localhost:3000/d/rl-policy"
    echo "  • Prometheus:         http://localhost:9090"
    echo "  • System Health:      http://localhost:9100/metrics"
    echo ""
    echo "🔧 OPERATIONAL COMMANDS:"
    echo -e "  • Status:      ${BLUE}make pilot-status${NC}"
    echo -e "  • KRI Check:   ${BLUE}make pilot-kri-test${NC}"
    echo -e "  • Kill Switch: ${RED}make kill-switch${NC}"
    echo -e "  • Digest:      ${BLUE}make pilot-digest${NC}"
    echo ""
    echo "📞 EMERGENCY CONTACTS:"
    echo "  • On-Call Engineer: [CONFIGURED IN RUNBOOK]"
    echo "  • Trading Desk:     [CONFIGURED IN RUNBOOK]"
    echo "  • Incident Command: [CONFIGURED IN RUNBOOK]"
    echo ""
    echo "⚠️  CRITICAL REMINDERS:"
    echo "  • Monitor continuously for next $MONITORING_PERIOD_MIN minutes"
    echo "  • Watch for KRI alerts and system health"
    echo "  • Execute kill-switch immediately if any concerns"
    echo "  • Document any issues in incident log"
    echo ""
    echo "================================================================"
    
    create_audit "monitoring_active" "{\"influence_pct\": $INITIAL_INFLUENCE_PCT, \"monitoring_period_min\": $MONITORING_PERIOD_MIN}"
}

# Main execution
main() {
    log "${BLUE}Starting Go-Live procedure...${NC}"
    
    # Create initial audit record
    create_audit "procedure_start" '{"operator": "'${USER:-unknown}'", "go_live_flag": "'${GO_LIVE:-unset}'"}'
    
    # Execute procedure steps
    check_prerequisites
    run_release_gates
    execute_ramp
    display_monitoring
    
    # Final audit record
    create_audit "procedure_complete" "{\"status\": \"success\", \"influence_pct\": $INITIAL_INFLUENCE_PCT}"
    
    log "${GREEN}✅ Go-Live procedure completed successfully${NC}"
    
    # Return appropriate exit code
    exit 0
}

# Error handling
trap 'log "${RED}❌ Go-Live procedure failed${NC}"; create_audit "procedure_error" "{\"exit_code\": $?, \"line\": $LINENO}"; exit 1' ERR

# Execute main function
main "$@"