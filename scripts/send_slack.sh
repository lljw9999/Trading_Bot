#!/usr/bin/env bash
#
# Slack Notification Script for Trading System
#
# Sends messages to #trading-ops Slack channel for health alerts
# Usage: ./scripts/send_slack.sh "message"
# Exit codes: 0=sent, 1=failed

set -euo pipefail

# Configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"  # Set in environment or CI secrets
SLACK_CHANNEL="${SLACK_CHANNEL:-#trading-ops}"
SLACK_USERNAME="${SLACK_USERNAME:-Trading-Bot}"
SLACK_ICON="${SLACK_ICON:-:robot_face:}"

# Colors for different alert levels
COLOR_SUCCESS="#36a64f"  # Green
COLOR_WARNING="#ff9900"  # Orange  
COLOR_DANGER="#ff0000"   # Red
COLOR_INFO="#0099cc"     # Blue
COLOR_YELLOW="#ffcc00"   # Yellow (for memory drift 5-10%)

# Message parsing and color assignment
get_message_color() {
    local message="$1"
    
    # Check for memory drift patterns first (more specific)
    if [[ "$message" =~ [0-9]+\.?[0-9]*% ]] && [[ "$message" =~ [Mm]emory|[Mm]em ]]; then
        # Extract percentage value
        local mem_pct=$(echo "$message" | grep -o '[0-9]*\.?[0-9]*%' | head -1 | sed 's/%//')
        
        if [[ -n "$mem_pct" ]]; then
            # Compare memory drift percentages
            if (( $(echo "$mem_pct >= 10" | bc -l) )); then
                echo "$COLOR_DANGER"   # Red: ≥10%
            elif (( $(echo "$mem_pct > 5" | bc -l) )); then
                echo "$COLOR_YELLOW"   # Yellow: >5% but <10%
            else
                echo "$COLOR_SUCCESS"  # Green: ≤5%
            fi
            return
        fi
    fi
    
    # Standard color logic
    if [[ "$message" =~ ":warning:" ]] || [[ "$message" =~ "WARN" ]]; then
        echo "$COLOR_WARNING"
    elif [[ "$message" =~ ":x:" ]] || [[ "$message" =~ "CRITICAL" ]] || [[ "$message" =~ "FAIL" ]]; then
        echo "$COLOR_DANGER"
    elif [[ "$message" =~ ":white_check_mark:" ]] || [[ "$message" =~ "SUCCESS" ]] || [[ "$message" =~ "PASS" ]]; then
        echo "$COLOR_SUCCESS"
    else
        echo "$COLOR_INFO"
    fi
}

# Main function to send Slack message
send_slack_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local hostname=$(hostname -s)
    local color=$(get_message_color "$message")
    
    # Check if webhook URL is configured
    if [[ -z "$SLACK_WEBHOOK_URL" ]]; then
        echo "⚠️  SLACK_WEBHOOK_URL not configured. Message would be:"
        echo "    Channel: $SLACK_CHANNEL"
        echo "    Message: $message"
        echo "    Time: $timestamp"
        echo "    Host: $hostname"
        return 0  # Don't fail if Slack not configured
    fi
    
    # Prepare JSON payload for Slack
    local json_payload=$(cat << EOF
{
    "channel": "$SLACK_CHANNEL",
    "username": "$SLACK_USERNAME",
    "icon_emoji": "$SLACK_ICON",
    "attachments": [
        {
            "color": "$color",
            "fields": [
                {
                    "title": "Trading System Alert",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Host",
                    "value": "$hostname",
                    "short": true
                },
                {
                    "title": "Time",
                    "value": "$timestamp",
                    "short": true
                }
            ],
            "footer": "v0.4.0-rc3 → GA",
            "ts": $(date +%s)
        }
    ]
}
EOF
)
    
    # Send to Slack
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$SLACK_WEBHOOK_URL")
    
    # Check response
    if [[ "$response" == "ok" ]]; then
        echo "✅ Slack message sent successfully"
        return 0
    else
        echo "❌ Failed to send Slack message: $response"
        return 1
    fi
}

# Enhanced notification function with retry logic
send_with_retry() {
    local message="$1"
    local max_retries=3
    local retry_delay=5
    
    for ((i=1; i<=max_retries; i++)); do
        if send_slack_message "$message"; then
            return 0
        else
            echo "Retry $i/$max_retries failed. Waiting ${retry_delay}s..."
            sleep $retry_delay
        fi
    done
    
    echo "❌ All $max_retries attempts failed"
    return 1
}

# Quick send functions for common alert types
send_healthcheck_alert() {
    local return_code="$1"
    send_with_retry ":warning: Healthcheck fail RC=$return_code | Host: $(hostname -s) | Time: $(date '+%H:%M')"
}

send_success_alert() {
    local message="$1"
    send_with_retry ":white_check_mark: $message"
}

send_warning_alert() {
    local message="$1"
    send_with_retry ":warning: $message"
}

send_critical_alert() {
    local message="$1"
    send_with_retry ":x: CRITICAL: $message"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] "message"

OPTIONS:
  -t, --type TYPE    Alert type: success, warning, critical, info (default: auto-detect)
  -c, --channel CH   Slack channel (default: #trading-ops)
  -r, --retry N      Number of retries (default: 3)
  -h, --help         Show this help

EXAMPLES:
  $0 "System healthy"
  $0 --type warning "VaR approaching limits"
  $0 --type critical "Container restart detected"
  
ENVIRONMENT VARIABLES:
  SLACK_WEBHOOK_URL  Slack incoming webhook URL (required)
  SLACK_CHANNEL      Default channel (default: #trading-ops)
  SLACK_USERNAME     Bot username (default: Trading-Bot)

INTEGRATION EXAMPLES:
  # In healthcheck_loop.sh:
  [[ \$RC -ne 0 ]] && ./scripts/send_slack.sh ":warning: Healthcheck fail \$RC"
  
  # In cron:
  0 */12 * * * ./scripts/send_slack.sh "\$(python scripts/report_pnl.py | grep FINAL_STATUS)"
EOF
}

# Main script logic
main() {
    local message=""
    local alert_type="auto"
    local max_retries=3
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                alert_type="$2"
                shift 2
                ;;
            -c|--channel)
                SLACK_CHANNEL="$2"
                shift 2
                ;;
            -r|--retry)
                max_retries="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                message="$1"
                shift
                ;;
        esac
    done
    
    # Check if message provided
    if [[ -z "$message" ]]; then
        echo "❌ Error: Message required"
        usage
        exit 1
    fi
    
    # Send based on alert type
    case "$alert_type" in
        success)
            send_success_alert "$message"
            ;;
        warning)
            send_warning_alert "$message"
            ;;
        critical)
            send_critical_alert "$message"
            ;;
        info|auto)
            send_with_retry "$message"
            ;;
        *)
            echo "❌ Unknown alert type: $alert_type"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 