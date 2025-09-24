#!/bin/bash
# Hourly validation script for GA monitoring
# As specified in Future_instruction.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
VALIDATOR_SCRIPT="scripts/validate_exit_criteria.py"
LOG_FILE="logs/hourly_validation.log"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"  # Set this environment variable

# Create logs directory if it doesn't exist
mkdir -p logs

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting hourly validation..." >> "$LOG_FILE"

# Run the validator in silent mode
if python "$VALIDATOR_SCRIPT" --silent >> "$LOG_FILE" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] ✅ Validator returned 0 - all criteria met" >> "$LOG_FILE"
else
    EXIT_CODE=$?
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ❌ Validator returned $EXIT_CODE - criteria failed" >> "$LOG_FILE"
    
    # Post to #trading-ops and Jira if configured
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 GA Validation Failed: Exit code $EXIT_CODE at $(date '+%Y-%m-%d %H:%M:%S'). Check logs for details.\"}" \
            "$SLACK_WEBHOOK" >> "$LOG_FILE" 2>&1
    fi
    
    # Write to alert file for monitoring
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EXIT_CODE: $EXIT_CODE" >> logs/validation_alerts.log
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Hourly validation completed" >> "$LOG_FILE" 