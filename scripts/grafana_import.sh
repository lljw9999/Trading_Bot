#!/bin/bash
# Grafana Dashboard and Alert Import Script
# 
# Imports dashboards and alerts to Grafana API idempotently.
# As specified in Task F: Running twice changes nothing.
#
# Usage: ./scripts/grafana_import.sh [options]

set -euo pipefail

# Configuration with defaults
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}" 
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRAFANA_DIR="${PROJECT_ROOT}/grafana"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -u, --url URL          Grafana URL (default: ${GRAFANA_URL})
    -U, --user USER        Grafana username (default: ${GRAFANA_USER})
    -P, --password PASS    Grafana password (default: from env or admin123)
    -d, --dashboard-only   Import dashboards only
    -a, --alerts-only      Import alerts only
    -f, --force           Force update existing dashboards/alerts
    -h, --help            Show this help message

ENVIRONMENT VARIABLES:
    GRAFANA_URL           Grafana server URL
    GRAFANA_USER          Grafana username  
    GRAFANA_PASSWORD      Grafana password
    GRAFANA_API_TOKEN     API token (alternative to user/password)

EXAMPLES:
    # Import all (default)
    $0
    
    # Import to specific Grafana instance
    $0 --url http://grafana.company.com:3000
    
    # Import dashboards only
    $0 --dashboard-only
    
    # Force update existing items
    $0 --force

EOF
}

# Parse command line arguments
DASHBOARD_ONLY=false
ALERTS_ONLY=false
FORCE_UPDATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            GRAFANA_URL="$2"
            shift 2
            ;;
        -U|--user)
            GRAFANA_USER="$2"
            shift 2
            ;;
        -P|--password)
            GRAFANA_PASSWORD="$2"
            shift 2
            ;;
        -d|--dashboard-only)
            DASHBOARD_ONLY=true
            shift
            ;;
        -a|--alerts-only)
            ALERTS_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_UPDATE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        exit 1
    fi
    
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed"
        exit 1
    fi
    
    # Check if grafana directory exists
    if [[ ! -d "${GRAFANA_DIR}" ]]; then
        log_error "Grafana directory not found: ${GRAFANA_DIR}"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Test Grafana connection
test_grafana_connection() {
    log_info "Testing Grafana connection to ${GRAFANA_URL}..."
    
    local auth=""
    if [[ -n "${GRAFANA_API_TOKEN:-}" ]]; then
        auth="Authorization: Bearer ${GRAFANA_API_TOKEN}"
    else
        auth="Authorization: Basic $(echo -n "${GRAFANA_USER}:${GRAFANA_PASSWORD}" | base64)"
    fi
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "${auth}" \
        "${GRAFANA_URL}/api/health")
    
    if [[ "${response}" != "200" ]]; then
        log_error "Failed to connect to Grafana (HTTP ${response})"
        log_error "Please check URL, credentials, and ensure Grafana is running"
        exit 1
    fi
    
    log_success "Connected to Grafana successfully"
}

# Import dashboard
import_dashboard() {
    local dashboard_file="$1"
    local dashboard_name
    dashboard_name=$(basename "${dashboard_file}" .json)
    
    log_info "Importing dashboard: ${dashboard_name}"
    
    if [[ ! -f "${dashboard_file}" ]]; then
        log_error "Dashboard file not found: ${dashboard_file}"
        return 1
    fi
    
    # Validate JSON
    if ! jq empty "${dashboard_file}" 2>/dev/null; then
        log_error "Invalid JSON in dashboard file: ${dashboard_file}"
        return 1
    fi
    
    # Prepare authentication
    local auth=""
    if [[ -n "${GRAFANA_API_TOKEN:-}" ]]; then
        auth="Authorization: Bearer ${GRAFANA_API_TOKEN}"
    else
        auth="Authorization: Basic $(echo -n "${GRAFANA_USER}:${GRAFANA_PASSWORD}" | base64)"
    fi
    
    # Check if dashboard already exists
    local existing_dashboard
    existing_dashboard=$(curl -s \
        -H "${auth}" \
        -H "Content-Type: application/json" \
        "${GRAFANA_URL}/api/search?query=${dashboard_name}" | jq -r '.[0].uid // empty')
    
    local import_payload
    if [[ -n "${existing_dashboard}" ]] && [[ "${FORCE_UPDATE}" == "false" ]]; then
        log_warning "Dashboard '${dashboard_name}' already exists (UID: ${existing_dashboard})"
        log_info "Use --force to update existing dashboard"
        return 0
    elif [[ -n "${existing_dashboard}" ]] && [[ "${FORCE_UPDATE}" == "true" ]]; then
        # Update existing dashboard by preserving UID
        log_info "Updating existing dashboard (UID: ${existing_dashboard})"
        import_payload=$(jq --arg uid "${existing_dashboard}" \
            '.dashboard.uid = $uid | {dashboard: .dashboard, overwrite: true}' \
            "${dashboard_file}")
    else
        # Create new dashboard
        import_payload=$(jq '{dashboard: .dashboard, overwrite: false}' "${dashboard_file}")
    fi
    
    # Import dashboard
    local response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "${auth}" \
        -H "Content-Type: application/json" \
        -d "${import_payload}" \
        "${GRAFANA_URL}/api/dashboards/db")
    
    local http_code
    http_code=$(echo "${response}" | tail -n1)
    local body
    body=$(echo "${response}" | head -n -1)
    
    if [[ "${http_code}" == "200" ]]; then
        local dashboard_url
        dashboard_url=$(echo "${body}" | jq -r '.url // "unknown"')
        log_success "Dashboard imported successfully: ${dashboard_url}"
        return 0
    else
        log_error "Failed to import dashboard (HTTP ${http_code})"
        log_error "Response: ${body}"
        return 1
    fi
}

# Import alert rule  
import_alert() {
    local alert_file="$1"
    local alert_name
    alert_name=$(basename "${alert_file}" .json)
    
    log_info "Importing alert rule: ${alert_name}"
    
    if [[ ! -f "${alert_file}" ]]; then
        log_error "Alert file not found: ${alert_file}"
        return 1
    fi
    
    # Validate JSON
    if ! jq empty "${alert_file}" 2>/dev/null; then
        log_error "Invalid JSON in alert file: ${alert_file}"
        return 1
    fi
    
    # Prepare authentication
    local auth=""
    if [[ -n "${GRAFANA_API_TOKEN:-}" ]]; then
        auth="Authorization: Bearer ${GRAFANA_API_TOKEN}"
    else
        auth="Authorization: Basic $(echo -n "${GRAFANA_USER}:${GRAFANA_PASSWORD}" | base64)"
    fi
    
    # Check if alert rule already exists
    local alert_uid
    alert_uid=$(jq -r '.alert.uid // empty' "${alert_file}")
    
    if [[ -n "${alert_uid}" ]]; then
        local existing_alert
        existing_alert=$(curl -s \
            -H "${auth}" \
            "${GRAFANA_URL}/api/v1/provisioning/alert-rules/${alert_uid}" \
            2>/dev/null | jq -r '.uid // empty')
        
        if [[ -n "${existing_alert}" ]] && [[ "${FORCE_UPDATE}" == "false" ]]; then
            log_warning "Alert rule '${alert_name}' already exists (UID: ${alert_uid})"
            log_info "Use --force to update existing alert"
            return 0
        fi
    fi
    
    # Import alert rule
    local response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "${auth}" \
        -H "Content-Type: application/json" \
        -d "$(cat "${alert_file}")" \
        "${GRAFANA_URL}/api/v1/provisioning/alert-rules")
    
    local http_code
    http_code=$(echo "${response}" | tail -n1)
    local body
    body=$(echo "${response}" | head -n -1)
    
    if [[ "${http_code}" == "201" ]] || [[ "${http_code}" == "200" ]]; then
        log_success "Alert rule imported successfully"
        return 0
    else
        log_error "Failed to import alert rule (HTTP ${http_code})"
        log_error "Response: ${body}"
        return 1
    fi
}

# Main import function
main() {
    log_info "🚀 Starting Grafana import process..."
    log_info "Target: ${GRAFANA_URL}"
    log_info "User: ${GRAFANA_USER}"
    
    check_prerequisites
    test_grafana_connection
    
    local dashboard_count=0
    local alert_count=0
    local errors=0
    
    # Import dashboards
    if [[ "${ALERTS_ONLY}" == "false" ]]; then
        log_info "📊 Importing dashboards..."
        
        for dashboard_file in "${GRAFANA_DIR}"/*.json; do
            if [[ ! -f "${dashboard_file}" ]]; then
                log_warning "No dashboard JSON files found in ${GRAFANA_DIR}"
                break
            fi
            
            # Skip alert files
            if [[ "${dashboard_file}" == *"alert"* ]]; then
                continue
            fi
            
            if import_dashboard "${dashboard_file}"; then
                ((dashboard_count++))
            else
                ((errors++))
            fi
        done
    fi
    
    # Import alerts
    if [[ "${DASHBOARD_ONLY}" == "false" ]]; then
        log_info "🚨 Importing alert rules..."
        
        for alert_file in "${GRAFANA_DIR}"/*alert*.json; do
            if [[ ! -f "${alert_file}" ]]; then
                log_warning "No alert JSON files found in ${GRAFANA_DIR}"
                break
            fi
            
            if import_alert "${alert_file}"; then
                ((alert_count++))
            else
                ((errors++))
            fi
        done
    fi
    
    # Summary
    echo
    log_info "📋 Import Summary:"
    log_info "  Dashboards imported: ${dashboard_count}"
    log_info "  Alert rules imported: ${alert_count}"
    log_info "  Errors: ${errors}"
    
    if [[ "${errors}" -eq 0 ]]; then
        log_success "✅ All imports completed successfully!"
        echo
        log_info "🌐 Access your dashboards at: ${GRAFANA_URL}/dashboards"
        log_info "🚨 View alerts at: ${GRAFANA_URL}/alerting/list"
    else
        log_error "❌ ${errors} import(s) failed"
        exit 1
    fi
}

# Run main function
main "$@" 