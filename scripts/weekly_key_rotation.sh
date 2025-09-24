#!/bin/bash
# Weekly API Key Rotation Script
# Rotates trading API keys and updates SSM parameters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔄 Weekly API Key Rotation${NC}"
echo "=================================="

# Check if we're in a production environment
if [ "${ENVIRONMENT}" != "production" ]; then
    echo -e "${YELLOW}⚠️  Not in production environment, running in dry-run mode${NC}"
    DRY_RUN="--dry-run"
else
    DRY_RUN=""
fi

# Log rotation start
echo "$(date): Starting weekly key rotation" >> /var/log/trading/key-rotation.log

# Run security hardener key rotation
echo -e "${GREEN}Running key rotation...${NC}"
PYTHONPATH="$PROJECT_ROOT" python3 "$PROJECT_ROOT/scripts/security_hardener.py" \
    --action rotate-keys \
    --output "/tmp/key_rotation_$(date +%Y%m%d_%H%M%S).json"

ROTATION_EXIT_CODE=$?

if [ $ROTATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Key rotation completed successfully${NC}"
    
    # Send success notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"✅ Weekly API key rotation completed successfully"}' \
            "$SLACK_WEBHOOK_URL"
    fi
    
    # Update Redis with last successful rotation
    if command -v redis-cli &> /dev/null; then
        redis-cli SET security:last_successful_rotation "$(date -Iseconds)"
    fi
    
    echo "$(date): Key rotation completed successfully" >> /var/log/trading/key-rotation.log
    
else
    echo -e "${RED}❌ Key rotation failed with exit code $ROTATION_EXIT_CODE${NC}"
    
    # Send failure notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"❌ Weekly API key rotation FAILED - manual intervention required"}' \
            "$SLACK_WEBHOOK_URL"
    fi
    
    echo "$(date): Key rotation FAILED with exit code $ROTATION_EXIT_CODE" >> /var/log/trading/key-rotation.log
    
    exit $ROTATION_EXIT_CODE
fi

# Run compliance scan after rotation
echo -e "${GREEN}Running post-rotation compliance scan...${NC}"
PYTHONPATH="$PROJECT_ROOT" python3 "$PROJECT_ROOT/scripts/security_hardener.py" \
    --action compliance-scan \
    --output "/tmp/compliance_scan_$(date +%Y%m%d_%H%M%S).json"

SCAN_EXIT_CODE=$?

if [ $SCAN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Compliance scan passed${NC}"
else
    echo -e "${YELLOW}⚠️  Compliance scan found issues - check logs${NC}"
fi

echo -e "${GREEN}Weekly key rotation process complete${NC}"