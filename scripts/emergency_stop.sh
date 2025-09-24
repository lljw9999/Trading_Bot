#!/bin/bash
# Emergency Stop Script - Quick panic button activation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}🚨 EMERGENCY STOP SYSTEM${NC}"
echo "=================================="

# Check if reason provided
REASON="${1:-Emergency stop activated via CLI}"

# Confirmation prompt
echo -e "${YELLOW}⚠️  WARNING: This will immediately halt all trading and flatten positions${NC}"
echo -e "Reason: ${REASON}"
echo
read -p "Type 'EMERGENCY' to confirm: " confirmation

if [ "$confirmation" != "EMERGENCY" ]; then
    echo -e "${GREEN}✅ Emergency stop cancelled${NC}"
    exit 0
fi

echo -e "${RED}🚨 EXECUTING EMERGENCY STOP...${NC}"

# Execute panic button
PYTHONPATH="$PROJECT_ROOT" python3 "$PROJECT_ROOT/scripts/panic_button.py" \
    --action panic \
    --reason "$REASON" \
    --output "/tmp/emergency_stop_$(date +%Y%m%d_%H%M%S).json"

echo -e "${GREEN}✅ Emergency stop sequence completed${NC}"
echo "Check logs and Redis for detailed results."