#!/usr/bin/env bash
set -euo pipefail

echo "🚀 RL Policy Canary Ramp to 10%"
echo "⏰ Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Default reason if not provided
REASON=${REASON:-"M3 canary 10% after dual PASS and no alerts"}

echo "🛡️  Running ramp guard safety checks..."
if python scripts/ramp_guard.py; then
    echo "✅ Ramp guard PASSED - proceeding with 10% ramp"
    echo
else
    echo "❌ Ramp guard FAILED - aborting ramp"
    echo "🛑 Fix safety issues before attempting ramp"
    exit 1
fi

echo "🔄 Setting policy influence to 10%..."
echo "📋 Reason: $REASON"

# Set influence with audit trail
REASON="$REASON" python scripts/promote_policy.py --pct 10

if [ $? -eq 0 ]; then
    echo
    echo "✅ Canary ramp to 10% COMPLETE"
    echo
    echo "🔍 MONITORING REQUIRED:"
    echo "   - Watch Grafana dashboards for anomalies"
    echo "   - Monitor entropy and Q-spread metrics"
    echo "   - Check shadow trading performance"
    echo
    echo "🚨 EMERGENCY PROCEDURES:"
    echo "   - Kill-switch: python scripts/kill_switch.py"
    echo "   - Manual revert: make promote-zero"
    echo "   - Check status: make influence"
    echo
    echo "⏱️  Policy influence will auto-expire in 1 hour if not refreshed"
    echo "📊 Monitor: curl localhost:9108/metrics | grep influence"
else
    echo "❌ Failed to set 10% influence"
    exit 1
fi