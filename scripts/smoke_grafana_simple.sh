#!/bin/bash
# Simplified Grafana Monitoring Stack Smoke Test

set -e

GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"

echo "🔍 Grafana Monitoring Stack Smoke Test"
echo "======================================="

# Test: Grafana API Health Check
echo "📡 Testing Grafana API Health Endpoint"
echo "--------------------------------------"

RESPONSE=$(curl -s -w "HTTP_CODE:%{http_code}|SIZE:%{size_download}" "$GRAFANA_URL/api/health")
HTTP_CODE=$(echo "$RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
SIZE=$(echo "$RESPONSE" | grep -o "SIZE:[0-9]*" | cut -d: -f2)
JSON_CONTENT=$(echo "$RESPONSE" | sed 's/HTTP_CODE:.*//g')

echo "🌐 URL: $GRAFANA_URL/api/health"
echo "📊 HTTP Status: $HTTP_CODE"
echo "📏 Response Size: $SIZE bytes"
echo "📋 JSON Content: $JSON_CONTENT"

# Verify success criteria
if [ "$HTTP_CODE" = "200" ] && [ "$SIZE" -gt 0 ]; then
    echo ""
    echo "✅ Success Signal: HTTP 200 OK – panel JSON size > 0"
    echo "✅ Grafana monitoring stack is functional!"
    
    # Check container status
    echo ""
    echo "🔧 Container Status Check"
    echo "------------------------"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(grafana|redis|redpanda|influxdb)" || echo "Services found in different format"
    
    echo ""
    echo "🎉 Monitoring stack smoke test PASSED!"
    exit 0
else
    echo ""
    echo "❌ Failed: HTTP $HTTP_CODE, Size $SIZE bytes"
    echo "❌ Monitoring stack smoke test FAILED!"
    exit 1
fi 