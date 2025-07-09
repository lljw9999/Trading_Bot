#!/usr/bin/env bash
#
# Grafana Dashboard Screenshot Capture Tool
# 
# Automated PNG generation for documentation embedding.
# Usage: ./capture_grafana.sh DASH_ID PANEL_ID OUT.png
#
# As specified in Task G: Documentation & Runbook Polish

set -euo pipefail

# Configuration with defaults
GF_HOST="${GF_HOST:-localhost:3000}"
GF_USER="${GF_USER:-admin}"
GF_PASS="${GF_PASS:-admin123}"
GF_THEME="${GF_THEME:-light}"
GF_WIDTH="${GF_WIDTH:-1440}"
GF_HEIGHT="${GF_HEIGHT:-900}"
GF_SCALE="${GF_SCALE:-1}"

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
Usage: $0 DASH_ID PANEL_ID OUT.png [OPTIONS]

ARGUMENTS:
    DASH_ID     Dashboard ID (e.g., edge-risk, model-router)
    PANEL_ID    Panel ID (integer, e.g., 1, 2, 3) or 'full' for entire dashboard
    OUT.png     Output PNG file path

OPTIONS:
    -h, --host HOST         Grafana host:port (default: ${GF_HOST})
    -u, --user USER         Grafana username (default: ${GF_USER})
    -p, --pass PASS         Grafana password (default: ${GF_PASS})
    -t, --theme THEME       Theme: light|dark (default: ${GF_THEME})
    -w, --width WIDTH       Image width in pixels (default: ${GF_WIDTH})
    -H, --height HEIGHT     Image height in pixels (default: ${GF_HEIGHT})
    -s, --scale SCALE       Scale factor (default: ${GF_SCALE})
    --from FROM             Time range start (default: now-6h)
    --to TO                 Time range end (default: now)
    --help                  Show this help message

ENVIRONMENT VARIABLES:
    GF_HOST                 Grafana host:port
    GF_USER                 Grafana username  
    GF_PASS                 Grafana password
    GF_API_TOKEN            API token (alternative to user/password)

EXAMPLES:
    # Capture Edge Risk dashboard panel 1
    $0 edge-risk 1 docs/images/edge-risk-stats.png
    
    # Capture full Model Router dashboard
    $0 model-router full docs/images/model-router-full.png
    
    # Capture with custom time range
    $0 edge-risk 2 edge-timeseries.png --from now-24h --to now
    
    # Dark theme with high resolution
    $0 edge-risk 1 panel.png --theme dark --width 1920 --height 1080
    
    # Using API token
    GF_API_TOKEN=xxx $0 edge-risk 1 output.png

PANEL IDs for Edge Risk Dashboard:
    1-4     Quick Stats (Active Model, Edge, Position, VaR)
    5-8     Time Series charts (per symbol)
    9       Switch Log table
    10-11   Alert Summary panels
    full    Entire dashboard

EOF
}

# Parse command line arguments
TIME_FROM="now-6h"
TIME_TO="now"

# Check minimum arguments
if [[ $# -lt 3 ]]; then
    log_error "Insufficient arguments"
    show_usage
    exit 1
fi

DASH_ID="$1"
PANEL_ID="$2"
OUTPUT_FILE="$3"
shift 3

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            GF_HOST="$2"
            shift 2
            ;;
        -u|--user)
            GF_USER="$2"
            shift 2
            ;;
        -p|--pass)
            GF_PASS="$2"
            shift 2
            ;;
        -t|--theme)
            GF_THEME="$2"
            shift 2
            ;;
        -w|--width)
            GF_WIDTH="$2"
            shift 2
            ;;
        -H|--height)
            GF_HEIGHT="$2"
            shift 2
            ;;
        -s|--scale)
            GF_SCALE="$2"
            shift 2
            ;;
        --from)
            TIME_FROM="$2"
            shift 2
            ;;
        --to)
            TIME_TO="$2"
            shift 2
            ;;
        --help)
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
    
    # Validate output directory
    output_dir=$(dirname "${OUTPUT_FILE}")
    if [[ ! -d "${output_dir}" ]]; then
        log_info "Creating output directory: ${output_dir}"
        mkdir -p "${output_dir}"
    fi
    
    log_success "Prerequisites check passed"
}

# Test Grafana connection
test_grafana_connection() {
    log_info "Testing Grafana connection to ${GF_HOST}..."
    
    local auth=""
    if [[ -n "${GF_API_TOKEN:-}" ]]; then
        auth="Authorization: Bearer ${GF_API_TOKEN}"
    else
        auth="Authorization: Basic $(echo -n "${GF_USER}:${GF_PASS}" | base64)"
    fi
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "${auth}" \
        "http://${GF_HOST}/api/health")
    
    if [[ "${response}" != "200" ]]; then
        log_error "Failed to connect to Grafana (HTTP ${response})"
        log_error "Please check host, credentials, and ensure Grafana is running"
        exit 1
    fi
    
    log_success "Connected to Grafana successfully"
}

# Capture screenshot
capture_screenshot() {
    log_info "Capturing screenshot..."
    log_info "Dashboard: ${DASH_ID}"
    log_info "Panel: ${PANEL_ID}"
    log_info "Output: ${OUTPUT_FILE}"
    log_info "Time range: ${TIME_FROM} to ${TIME_TO}"
    
    # Prepare authentication
    local auth_header=""
    if [[ -n "${GF_API_TOKEN:-}" ]]; then
        auth_header="Authorization: Bearer ${GF_API_TOKEN}"
    else
        auth_header="Authorization: Basic $(echo -n "${GF_USER}:${GF_PASS}" | base64)"
    fi
    
    # Build URL based on panel type
    local url=""
    if [[ "${PANEL_ID}" == "full" ]]; then
        # Full dashboard screenshot
        url="http://${GF_HOST}/render/d/${DASH_ID}?theme=${GF_THEME}&from=${TIME_FROM}&to=${TIME_TO}&width=${GF_WIDTH}&height=${GF_HEIGHT}&scale=${GF_SCALE}"
    else
        # Individual panel screenshot
        url="http://${GF_HOST}/render/d-solo/${DASH_ID}?panelId=${PANEL_ID}&theme=${GF_THEME}&from=${TIME_FROM}&to=${TIME_TO}&width=${GF_WIDTH}&height=${GF_HEIGHT}&scale=${GF_SCALE}"
    fi
    
    log_info "Requesting: ${url}"
    
    # Capture with timeout and progress
    local start_time
    start_time=$(date +%s)
    
    local response
    response=$(curl -s -w "\n%{http_code}" \
        -H "${auth_header}" \
        -H "Accept: image/png" \
        --max-time 30 \
        --output "${OUTPUT_FILE}" \
        "${url}")
    
    local http_code
    http_code=$(echo "${response}" | tail -n1)
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Check response
    if [[ "${http_code}" == "200" ]]; then
        # Verify PNG file was created
        if [[ -f "${OUTPUT_FILE}" ]] && file "${OUTPUT_FILE}" | grep -q "PNG image"; then
            local file_size
            file_size=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo "unknown")
            
            log_success "Screenshot captured successfully!"
            log_info "  File: ${OUTPUT_FILE}"
            log_info "  Size: ${file_size} bytes"
            log_info "  Duration: ${duration}s"
            log_info "  Dimensions: ${GF_WIDTH}×${GF_HEIGHT}"
            
            # Check if capture was fast enough (Task G requirement: ≤5s)
            if [[ ${duration} -le 5 ]]; then
                log_success "✅ Performance target met (≤5s)"
            else
                log_warning "⚠️  Capture took ${duration}s (target: ≤5s)"
            fi
            
            return 0
        else
            log_error "Invalid PNG file generated"
            return 1
        fi
    else
        log_error "Failed to capture screenshot (HTTP ${http_code})"
        
        # Try to show error response if file contains text
        if [[ -f "${OUTPUT_FILE}" ]] && file "${OUTPUT_FILE}" | grep -q "text"; then
            log_error "Error response:"
            head -5 "${OUTPUT_FILE}"
        fi
        
        return 1
    fi
}

# Validate dashboard and panel
validate_parameters() {
    log_info "Validating parameters..."
    
    # Validate theme
    if [[ "${GF_THEME}" != "light" && "${GF_THEME}" != "dark" ]]; then
        log_error "Invalid theme: ${GF_THEME} (must be 'light' or 'dark')"
        exit 1
    fi
    
    # Validate dimensions
    if [[ ! "${GF_WIDTH}" =~ ^[0-9]+$ ]] || [[ ${GF_WIDTH} -lt 400 ]]; then
        log_error "Invalid width: ${GF_WIDTH} (must be integer ≥400)"
        exit 1
    fi
    
    if [[ ! "${GF_HEIGHT}" =~ ^[0-9]+$ ]] || [[ ${GF_HEIGHT} -lt 300 ]]; then
        log_error "Invalid height: ${GF_HEIGHT} (must be integer ≥300)"
        exit 1
    fi
    
    # Validate panel ID (if not 'full')
    if [[ "${PANEL_ID}" != "full" ]] && [[ ! "${PANEL_ID}" =~ ^[0-9]+$ ]]; then
        log_error "Invalid panel ID: ${PANEL_ID} (must be integer or 'full')"
        exit 1
    fi
    
    # Validate output file extension
    if [[ "${OUTPUT_FILE}" != *.png ]]; then
        log_error "Output file must have .png extension: ${OUTPUT_FILE}"
        exit 1
    fi
    
    log_success "Parameter validation passed"
}

# Main function
main() {
    log_info "🖼️  Starting Grafana screenshot capture"
    log_info "Dashboard: ${DASH_ID} | Panel: ${PANEL_ID} | Output: ${OUTPUT_FILE}"
    
    validate_parameters
    check_prerequisites
    test_grafana_connection
    
    if capture_screenshot; then
        log_success "🎉 Screenshot capture completed successfully!"
        exit 0
    else
        log_error "❌ Screenshot capture failed"
        exit 1
    fi
}

# Run main function
main "$@" 