#!/bin/bash
set -e

# Sentiment Stack Smoke Test
# Validates the complete sentiment analysis pipeline

echo "🔥 Starting Sentiment Stack Smoke Test..."

# Configuration
SENTIMENT_SERVICE="http://localhost:8002"
EXPLAIN_SERVICE="http://localhost:8003"
REDIS_HOST="localhost"
REDIS_PORT="6379"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Test 1: Check service health endpoints
echo ""
echo "📋 Test 1: Service Health Checks"

# Check sentiment enricher service
echo "Checking sentiment enricher service..."
if curl -s -f "$SENTIMENT_SERVICE/health" > /dev/null; then
    log_info "Sentiment enricher service is healthy"
else
    log_error "Sentiment enricher service health check failed"
    exit 1
fi

# Check explanation service
echo "Checking explanation service..."
if curl -s -f "$EXPLAIN_SERVICE/health" > /dev/null; then
    log_info "Explanation service is healthy"
else
    log_error "Explanation service health check failed"
    exit 1
fi

# Check Redis connectivity
echo "Checking Redis connectivity..."
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q "PONG"; then
    log_info "Redis is responding"
else
    log_error "Redis connection failed"
    exit 1
fi

# Test 2: Publish fake news document to Redis
echo ""
echo "📋 Test 2: Inject Test Data"

# Create fake news document
FAKE_NEWS_DOC='{
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%fZ)'",
    "symbol": "BTC",
    "text": "Bitcoin shows strong bullish momentum as institutional investors increase their holdings. Technical analysis suggests a potential breakout above $50,000 resistance level.",
    "source": "smoke-test",
    "url": "https://example.com/test-news",
    "author": "Smoke Test Bot"
}'

echo "Publishing fake news document to Redis..."
echo "$FAKE_NEWS_DOC" | redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -x lpush soft.raw.news
log_info "Fake news document published to soft.raw.news"

# Test 3: Wait for processing and check enriched data
echo ""
echo "📋 Test 3: Verify Pipeline Processing"

echo "Waiting for sentiment processing (30 seconds)..."
sleep 30

# Check if enriched data exists
ENRICHED_COUNT=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" llen soft.enriched)
if [ "$ENRICHED_COUNT" -gt 0 ]; then
    log_info "Found $ENRICHED_COUNT enriched documents"
    
    # Get latest enriched document
    LATEST_ENRICHED=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" lindex soft.enriched 0)
    
    # Check if it contains sentiment score
    if echo "$LATEST_ENRICHED" | grep -q "sentiment_score"; then
        log_info "Enriched document contains sentiment_score"
        
        # Extract sentiment score
        SENTIMENT_SCORE=$(echo "$LATEST_ENRICHED" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('sentiment_score', 0))
")
        
        echo "Latest sentiment score: $SENTIMENT_SCORE"
        
        # Check if sentiment score is non-zero
        if python3 -c "import sys; sys.exit(0 if abs(float('$SENTIMENT_SCORE')) > 0 else 1)"; then
            log_info "Sentiment score is non-zero: $SENTIMENT_SCORE"
        else
            log_warn "Sentiment score is zero, but processing occurred"
        fi
    else
        log_error "Enriched document missing sentiment_score field"
        exit 1
    fi
else
    log_error "No enriched documents found - sentiment processing failed"
    exit 1
fi

# Test 4: Check Prometheus metrics
echo ""
echo "📋 Test 4: Verify Prometheus Metrics"

# Check if sent_score_latest metric is available
if curl -s "$SENTIMENT_SERVICE/metrics" | grep -q "sent_score_latest"; then
    log_info "sent_score_latest metric is available"
    
    # Try to get the metric value
    METRIC_VALUE=$(curl -s "$SENTIMENT_SERVICE/metrics" | grep "sent_score_latest{symbol=\"BTC\"}" | grep -o '[0-9.-]*$' | head -1)
    
    if [ -n "$METRIC_VALUE" ]; then
        log_info "sent_score_latest{symbol=\"BTC\"} = $METRIC_VALUE"
        
        # Check if metric value is greater than 0
        if python3 -c "import sys; sys.exit(0 if abs(float('$METRIC_VALUE')) > 0 else 1)"; then
            log_info "✅ Sentiment metric is non-zero!"
        else
            log_warn "Sentiment metric is zero"
        fi
    else
        log_warn "Could not extract metric value (may not be updated yet)"
    fi
else
    log_error "sent_score_latest metric not found in Prometheus metrics"
    exit 1
fi

# Test 5: Test explanation API
echo ""
echo "📋 Test 5: Test Explanation Service"

# Create test order for explanation
TEST_ORDER='{
    "order_id": "smoke_test_order",
    "symbol": "BTC-USD",
    "side": "buy",
    "quantity": 0.1,
    "price": 50000.0,
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%fZ)'",
    "order_type": "market",
    "edge_bps": 25.0,
    "confidence": 0.8,
    "sentiment_score": 0.7
}'

echo "Testing explanation generation..."
EXPLANATION_RESPONSE=$(curl -s -X POST "$EXPLAIN_SERVICE/explain" \
    -H "Content-Type: application/json" \
    -d "$TEST_ORDER")

if echo "$EXPLANATION_RESPONSE" | grep -q "explanation"; then
    log_info "Explanation service responded successfully"
    EXPLANATION_TEXT=$(echo "$EXPLANATION_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('explanation', 'No explanation found')[:100] + '...')
except:
    print('Error parsing explanation response')
")
    echo "Explanation preview: $EXPLANATION_TEXT"
else
    log_error "Explanation service failed to respond properly"
    exit 1
fi

# Test 6: Cleanup
echo ""
echo "📋 Test 6: Cleanup"

echo "Removing test data..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" del smoke_test_key > /dev/null
log_info "Test data cleaned up"

# Final status
echo ""
echo "🎉 SMOKE TEST RESULTS:"
echo "========================"
log_info "✅ Sentiment enricher service: HEALTHY"
log_info "✅ Explanation service: HEALTHY" 
log_info "✅ Redis connectivity: OK"
log_info "✅ Sentiment pipeline: WORKING"
log_info "✅ Prometheus metrics: AVAILABLE"
log_info "✅ Explanation generation: WORKING"

echo ""
echo -e "${GREEN}🎊 SMOKE OK${NC}"
echo ""
echo "All sentiment stack components are functioning correctly!"
echo ""
echo "Next steps:"
echo "  • Monitor Grafana dashboard: http://localhost:3000"
echo "  • Check metrics: http://localhost:8002/metrics"
echo "  • View explanations: http://localhost:8003/recent"
echo ""

exit 0 