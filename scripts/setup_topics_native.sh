#!/usr/bin/env bash
set -e

topics=(
  market.raw.crypto.coinbase
  market.raw.crypto.binance
  market.raw.stocks
  features.raw.crypto
  features.raw
  signals.meta
  orders.target
  metrics.execution
)

echo "🔧 Creating Redpanda topics..."
for topic in "${topics[@]}"; do
  echo "• Creating $topic"
  rpk topic create "$topic" --partitions 3 --replicas 1 || echo "  ⚠️ Topic may already exist"
done

echo "✅ Topics created successfully"
echo "📊 Current topic list:"
rpk topic list 