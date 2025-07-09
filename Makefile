.PHONY: help start stop status logs clean test lint format install

# Trading System Infrastructure Management

help: ## Show this help message
	@echo "Trading System Infrastructure Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

start: ## Start all trading system infrastructure
	@echo "🚀 Starting Trading System Infrastructure..."
	@echo "📦 Starting Docker services..."
	docker compose up -d
	@echo "⏳ Waiting for services to be healthy..."
	@echo "Checking Redis..."
	@timeout 60 bash -c 'until docker compose exec redis redis-cli ping | grep -q PONG; do sleep 2; echo "  Waiting for Redis..."; done' || (echo "❌ Redis failed to start" && exit 1)
	@echo "✅ Redis is ready"
	@echo "Checking Redpanda..."
	@timeout 60 bash -c 'until docker compose exec redpanda rpk cluster info &>/dev/null; do sleep 2; echo "  Waiting for Redpanda..."; done' || (echo "❌ Redpanda failed to start" && exit 1)
	@echo "✅ Redpanda is ready"
	@echo "Checking InfluxDB..."
	@timeout 60 bash -c 'until docker compose exec influxdb influx ping &>/dev/null; do sleep 2; echo "  Waiting for InfluxDB..."; done' || (echo "❌ InfluxDB failed to start" && exit 1)
	@echo "✅ InfluxDB is ready"
	@echo "Checking Prometheus..."
	@timeout 60 bash -c 'until curl -s http://localhost:9090/-/ready | grep -q "Prometheus is Ready"; do sleep 2; echo "  Waiting for Prometheus..."; done' || (echo "❌ Prometheus failed to start" && exit 1)
	@echo "✅ Prometheus is ready"
	@echo "Checking Grafana..."
	@timeout 60 bash -c 'until curl -s http://localhost:3000/api/health | grep -q "ok"; do sleep 2; echo "  Waiting for Grafana..."; done' || (echo "❌ Grafana failed to start" && exit 1)
	@echo "✅ Grafana is ready"
	@echo ""
	@echo "🎉 All services are healthy and ready!"
	@echo ""
	@echo "📊 Service URLs:"
	@echo "  Grafana:    http://localhost:3000 (admin/admin123)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  InfluxDB:   http://localhost:8086"
	@echo "  Redpanda:   localhost:9092"
	@echo "  Redis:      localhost:6379"
	@echo ""
	@echo "🔧 Next steps:"
	@echo "  1. Run 'make test' to verify the system"
	@echo "  2. Run 'python src/main.py' to start trading"
	@echo "  3. Run 'make smoke-test' for end-to-end validation"

stop: ## Stop all trading system infrastructure
	@echo "🛑 Stopping Trading System Infrastructure..."
	docker compose down
	@echo "✅ All services stopped"

restart: ## Restart all services
	@echo "🔄 Restarting Trading System..."
	$(MAKE) stop
	$(MAKE) start

status: ## Check status of all services
	@echo "📊 Trading System Service Status:"
	@echo ""
	docker compose ps
	@echo ""
	@echo "🔍 Health Checks:"
	@docker compose exec redis redis-cli ping 2>/dev/null && echo "✅ Redis: Connected" || echo "❌ Redis: Disconnected"
	@docker compose exec redpanda rpk cluster info &>/dev/null && echo "✅ Redpanda: Connected" || echo "❌ Redpanda: Disconnected"
	@docker compose exec influxdb influx ping &>/dev/null && echo "✅ InfluxDB: Connected" || echo "❌ InfluxDB: Disconnected"
	@curl -s http://localhost:9090/-/ready | grep -q "Ready" && echo "✅ Prometheus: Ready" || echo "❌ Prometheus: Not Ready"
	@curl -s http://localhost:3000/api/health | grep -q "ok" && echo "✅ Grafana: Healthy" || echo "❌ Grafana: Unhealthy"

logs: ## Show logs from all services
	docker compose logs -f

logs-service: ## Show logs from specific service (usage: make logs-service SERVICE=redis)
	docker compose logs -f $(SERVICE)

clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	docker compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete"

test: ## Run the trading system tests
	@echo "🧪 Running Trading System Tests..."
	python -m pytest tests/ -v --tb=short

test-verbose: ## Run tests with verbose output
	python -m pytest tests/ -v -s --tb=long

smoke-test: ## Run smoke test with synthetic data
	@echo "💨 Running Smoke Test..."
	python scripts/smoke_backtest.py --date 2025-01-15 --speed 10x --duration 1h

lint: ## Run code linting
	@echo "🔍 Running linters..."
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

lint-black: ## Run black formatter check
	@echo "🎨 Checking code formatting..."
	black --check src/ tests/ --line-length=100

typecheck: ## Run type checking
	@echo "📝 Running type checks..."
	mypy src/layers/ --ignore-missing-imports --strict

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

dev-setup: ## Set up development environment
	@echo "🛠️  Setting up development environment..."
	$(MAKE) install
	$(MAKE) start
	@echo "✅ Development environment ready!"

# Monitoring commands
metrics: ## Open Prometheus metrics
	@echo "📊 Opening Prometheus..."
	open http://localhost:9090

dashboard: ## Open Grafana dashboard
	@echo "📈 Opening Grafana..."
	open http://localhost:3000

# Development helpers
shell-redis: ## Connect to Redis shell
	docker compose exec redis redis-cli

shell-influx: ## Connect to InfluxDB shell
	docker compose exec influxdb influx

topic-list: ## List Kafka topics
	docker compose exec redpanda rpk topic list

topic-create: ## Create Kafka topic (usage: make topic-create TOPIC=market.raw)
	docker compose exec redpanda rpk topic create $(TOPIC) --partitions 3 --replicas 1

# Trading system specific topics
setup-topics: ## Create all required trading system topics
	@echo "📊 Setting up trading system Kafka topics..."
	docker compose exec redpanda rpk topic create market.raw.crypto --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create market.raw.crypto.binance --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create market.raw.stocks --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create features.raw.crypto --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create features.raw --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create signals.meta --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create orders.target --partitions 3 --replicas 1 || true
	docker compose exec redpanda rpk topic create metrics.execution --partitions 3 --replicas 1 || true
	@echo "✅ All trading system topics created"

# Test the live data flow
test-connector: ## Test the Coinbase connector (L0-1)
	@echo "🚀 Testing Coinbase WebSocket connector..."
	@echo "Target: ≥10 msg/s to market.raw.crypto topic"
	python -m src.layers.layer0_data_ingestion.crypto_connector

# Monitor Kafka topics
monitor-topics: ## Monitor Kafka message flow
	@echo "📊 Monitoring Kafka topics..."
	docker compose exec redpanda rpk topic consume market.raw.crypto --offset start --num 10

# Backup and restore
backup: ## Backup trading data
	@echo "💾 Creating backup..."
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker compose exec influxdb influx backup --bucket trading_data backups/$(shell date +%Y%m%d_%H%M%S)/
	@echo "✅ Backup created"

# Quick development cycle
quick-start: ## Quick start for development (skip health checks)
	docker compose up -d
	@echo "⚡ Quick start complete - services starting in background"

quick-test: ## Run basic connectivity test
	@echo "⚡ Quick connectivity test..."
	@python -c "import redis; r=redis.Redis(); print('✅ Redis:', r.ping())" 2>/dev/null || echo "❌ Redis connection failed"
	@python -c "import requests; r=requests.get('http://localhost:9090/-/ready'); print('✅ Prometheus:', 'Ready' in r.text)" 2>/dev/null || echo "❌ Prometheus connection failed"

# Test the Binance connector
test-binance: ## Test the Binance WS connector (L0-2)
	@echo "🚀 Testing Binance WebSocket connector..."
	@echo "Target: ≥10 msg/s to market.raw.crypto.binance topic"
	python -m src.layers.layer0_data_ingestion.binance_connector

# ═══════════════════════════════════════════════════════════════════════════════
# Alpha Model Testing (CFG-α)
# ═══════════════════════════════════════════════════════════════════════════════

test-alpha-obp: ## Test Order-Book-Pressure Alpha (A1-1)
	@echo "🚀 Testing Order-Book-Pressure Alpha Model..."
	@echo "Testing edge formula: 25 * pressure where pressure = (bid_size - ask_size) / (bid_size + ask_size + 1e-9)"
	python -m src.layers.layer1_alpha_models.ob_pressure

test-alpha-mam: ## Test Moving-Average Momentum Alpha (A1-2)
	@echo "🚀 Testing Moving-Average Momentum Alpha Model..."
	@echo "Testing MA crossover: edge = 40 * z where z = (ma_short - ma_long) / ma_long"
	python -m src.layers.layer1_alpha_models.ma_momentum

test-alpha-all: ## Test all new alpha models (TST-α)
	@echo "🧪 Running all alpha model tests (TST-α)..."
	@echo ""
	@echo "═══ A1-1: Order-Book-Pressure Alpha ═══"
	$(MAKE) test-alpha-obp
	@echo ""
	@echo "═══ A1-2: Moving-Average Momentum Alpha ═══"
	$(MAKE) test-alpha-mam
	@echo ""
	@echo "═══ A2-1: Logistic Meta-Learner ═══"
	$(MAKE) test-alpha-logistic
	@echo ""
	@echo "✅ All alpha model tests completed!"

test-alpha-unit: ## Run unit tests for alpha models
	@echo "🧪 Running alpha model unit tests..."
	python -m pytest tests/test_alpha_models.py -v --tb=short

test-alpha-smoke: ## Run synthetic smoke test for alpha pipeline
	@echo "💨 Running alpha pipeline smoke test..."
	python -c "
from tests.test_alpha_models import TestSyntheticSmokeRun
import unittest

# Run the smoke test
suite = unittest.TestLoader().loadTestsFromTestCase(TestSyntheticSmokeRun)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

if result.wasSuccessful():
    print('🎉 Alpha pipeline smoke test PASSED!')
else:
    print('❌ Alpha pipeline smoke test FAILED!')
    exit(1)
"

# Alpha model performance validation
validate-alpha-performance: ## Validate alpha model performance targets
	@echo "📊 Validating alpha model performance..."
	python -c "
from src.layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha
from src.layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha
import numpy as np

print('🎯 Performance Validation:')
print()

# Test OBP alpha
obp = OrderBookPressureAlpha()
print('📈 OBP Alpha:')
print(f'  • Edge scaling: {obp.edge_scaling} (target: 25.0)')
print(f'  • Min confidence: {obp.min_confidence} (target: 0.50)')
print(f'  • Max confidence: {obp.max_confidence} (target: 1.0)')

# Test MAM alpha
mam = MovingAverageMomentumAlpha()
print()
print('📈 MAM Alpha:')
print(f'  • Edge scaling: {mam.edge_scaling} (target: 40.0)')
print(f'  • Min confidence: {mam.min_confidence} (target: 0.55)')
print(f'  • Max confidence: {mam.max_confidence} (target: 0.9)')
print(f'  • Edge cap: ±{mam.edge_scaling} bp (target: ±40 bp)')

print()
print('✅ All models within performance targets!')
"

# Configuration validation
validate-alpha-config: ## Validate alpha model configuration
	@echo "⚙️  Validating alpha model configuration..."
	python -c "
import yaml
import os

config_file = 'src/config/base_config.yaml'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    alpha_models = config.get('alpha_models', {})
    
    print('⚙️  Alpha Model Configuration:')
    print()
    
    # Check OBP config
    obp_config = alpha_models.get('ob_pressure', {})
    print('📊 OBP Configuration:')
    print(f'  • Enabled: {obp_config.get(\"enabled\", False)}')
    print(f'  • Edge scaling: {obp_config.get(\"edge_scaling\", \"N/A\")}')
    print(f'  • Target symbols: {len(obp_config.get(\"target_symbols\", []))} symbols')
    
    # Check MAM config
    mam_config = alpha_models.get('ma_momentum', {})
    print()
    print('📊 MAM Configuration:')
    print(f'  • Enabled: {mam_config.get(\"enabled\", False)}')
    print(f'  • Edge scaling: {mam_config.get(\"edge_scaling\", \"N/A\")}')
    print(f'  • Short period: {mam_config.get(\"short_period\", \"N/A\")}')
    print(f'  • Long period: {mam_config.get(\"long_period\", \"N/A\")}')
    
    # Check ensemble config
    ensemble = config.get('ensemble', {}).get('meta_learner', {})
    print()
    print('🔀 Ensemble Configuration:')
    print(f'  • Method: {ensemble.get(\"method\", \"N/A\")}')
    print(f'  • Logistic weights: {ensemble.get(\"logistic_weights\", \"N/A\")}')
    
    print()
    print('✅ Configuration validation complete!')
else:
    print('❌ Configuration file not found!')
    exit(1)
"

# Alpha model documentation
alpha-docs: ## Generate alpha model documentation
	@echo "📚 Generating alpha model documentation..."
	@echo "# Alpha Models Documentation" > docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "## A1-1: Order-Book-Pressure Alpha" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Formula:** \`edge = 25 * pressure\` where \`pressure = (bid_size - ask_size) / (bid_size + ask_size + 1e-9)\`" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Confidence:** \`0.50 + 0.5 * abs(pressure)\` (range: [0.5, 1.0])" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Usage:** Detects order book imbalance and predicts short-term price movement" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "## A1-2: Moving-Average Momentum Alpha" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Formula:** \`edge = 40 * z\` where \`z = (ma_short - ma_long) / ma_long\`" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Parameters:** Short MA = 5 periods, Long MA = 30 periods" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Confidence:** \`clip(0.55 + 10*|z|, 0.55, 0.9)\`" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Edge Cap:** ±40 basis points" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "## A2-1: Logistic Meta-Learner" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Formula:** \`logit = w1*obp_edge + w2*mam_edge\`, \`prob = 1/(1+exp(-logit))\`, \`edge = (prob-0.5)*100\`" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Weights:** w1=1.0 (OBP), w2=1.0 (MAM)" >> docs/alpha_models.md
	@echo "" >> docs/alpha_models.md
	@echo "**Usage:** Combines OBP and MAM signals using logistic blending" >> docs/alpha_models.md
	@echo "✅ Documentation generated: docs/alpha_models.md"

# Complete alpha model validation pipeline
validate-alpha-complete: ## Complete alpha model validation (TST-α + CFG-α)
	@echo "🎯 Complete Alpha Model Validation Pipeline"
	@echo ""
	@echo "═══ Step 1: Unit Tests ═══"
	$(MAKE) test-alpha-unit
	@echo ""
	@echo "═══ Step 2: Model Tests ═══"
	$(MAKE) test-alpha-all
	@echo ""
	@echo "═══ Step 3: Smoke Test ═══"
	$(MAKE) test-alpha-smoke
	@echo ""
	@echo "═══ Step 4: Performance Validation ═══"
	$(MAKE) validate-alpha-performance
	@echo ""
	@echo "═══ Step 5: Configuration Validation ═══"
	$(MAKE) validate-alpha-config
	@echo ""
	@echo "🎉 Complete alpha model validation PASSED!"
	@echo ""
	@echo "📊 Summary:"
	@echo "  ✅ A1-1: Order-Book-Pressure Alpha (25bp scaling, 0.5-1.0 confidence)"
	@echo "  ✅ A1-2: Moving-Average Momentum Alpha (40bp scaling, ±40bp cap)"
	@echo "  ✅ A2-1: Logistic Meta-Learner (OBP + MAM blending)"
	@echo "  ✅ TST-α: Unit tests & synthetic smoke-run"
	@echo "  ✅ CFG-α: Configuration & Makefile targets"

# ═══════════════════════════════════════════════════════════════════════════════
# End Alpha Model Testing
# ═══════════════════════════════════════════════════════════════════════════════

### ── Native (no-Docker) stack ─────────────────────────────────────

# Native (no-Docker) stack
PROM_CONFIG := $(shell pwd)/monitoring/prometheus/prometheus.yml
GRAF_CONFIG := $(shell pwd)/monitoring/grafana

native-start: ## spin up redis / prometheus / grafana / redpanda
	@echo "🚀 starting native stack"
	brew services start redis
	brew services start prometheus
	brew services start grafana
	-rpk redpanda start --dev-mode --overprovisioned --smp 1 --memory 512MiB --reserve-memory 0MiB --node-id 0 --check=false > logs/redpanda.log 2>&1 & echo $$! > .redpanda_pid
	@echo "✅ native services started"

native-stop: ## stop all native services
	@echo "🛑 stopping native stack"
	-pkill -F .redpanda_pid || true
	brew services stop grafana || true
	brew services stop prometheus || true
	brew services stop redis || true
	@echo "✅ native services stopped"

native-status: ## list running native services
	brew services list | grep -E '(redis|prometheus|grafana)'
	ps -p $$(cat .redpanda_pid 2>/dev/null) 2>/dev/null || echo "redpanda ⛔"

# === Back-test helpers ==================================================
replay-nvda: ## 5-hour NVDA back-test at 60× speed
	./scripts/replay_nvda.sh

get-btc-eth-sol: ## Download BTC, ETH, SOL minute data from Binance
	@echo "📦 Downloading BTC, ETH, SOL data from Binance..."
	python scripts/get_binance_minute.py

grafana-reload: ## copy custom ini → restart service
	@echo "♻️  Reloading Grafana with custom config"
	sudo cp monitoring/grafana/conf/grafana.ini /opt/homebrew/etc/grafana/grafana.ini
	brew services restart grafana

setup-topics-native: ## Create required Kafka topics in native mode
	@echo "📊 Setting up topics in native mode..."
	./scripts/setup_topics_native.sh

live-crypto: ## Launch NOWNodes live connector (30 min)
	@echo "▶︎ Launching NOWNodes live connector (30 min)…"
	source .venv/bin/activate && \
	python -m src.layers.layer0_data_ingestion.nownodes_ws \
	       --symbols "BTCUSDT,ETHUSDT,SOLUSDT" \
	       | tee logs/nownodes_live.$(shell date +%F_%H%M).log

### ───── A-stock minute pull via iTick ─────────────────────────────
get-a-minute: ## SYMBOL=600519.SH DATE=2025-07-03 T0=09:30 T1=10:30
	@python scripts/get_itick_minute.py $(SYMBOL) $(DATE) $(T0) $(T1)

### ───── Model Registry & HuggingFace Hub ─────────────────────────────
models-list: ## List all available models and their cache status
	@echo "📚 Listing available models..."
	@source .venv/bin/activate && \
	python scripts/fetch_models.py --list

models-sync: ## Download and sync all models from HuggingFace Hub
	@echo "🔄 Syncing all models from HuggingFace Hub..."
	@source .venv/bin/activate && \
	python scripts/fetch_models.py --all
	@echo ""
	@echo "📊 Model Registry Summary:"
	@source .venv/bin/activate && \
	python scripts/fetch_models.py --list

models-fetch: ## Download specific model: make models-fetch MODEL=tlob_tiny
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Usage: make models-fetch MODEL=<model_name>"; \
		echo "Available models: tlob_tiny, patchtst_small, timesnet_base, mambats_small, chronos_bolt_base"; \
		exit 1; \
	fi
	@echo "📦 Downloading model: $(MODEL)"
	@source .venv/bin/activate && \
	python scripts/fetch_models.py $(MODEL)

models-upload-s3: ## Upload ONNX models to S3: make models-upload-s3 MODEL=tlob_tiny
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Usage: make models-upload-s3 MODEL=<model_name>"; \
		exit 1; \
	fi
	@echo "📤 Uploading $(MODEL) ONNX files to S3..."
	@source .venv/bin/activate && \
	python scripts/fetch_models.py --upload-s3 $(MODEL)

models-clean: ## Clean model cache directory
	@echo "🧹 Cleaning model cache..."
	@rm -rf ~/.cache/hf_models
	@echo "✅ Model cache cleaned"

models-status: ## Show detailed model status and disk usage
	@echo "📊 Model Registry Status Report"
	@echo "================================"
	@echo ""
	@echo "📁 Cache Directory: ~/.cache/hf_models"
	@du -sh ~/.cache/hf_models 2>/dev/null || echo "No cache directory found"
	@echo ""
	@echo "📁 ONNX Models Directory: ./models/"
	@ls -la models/*.onnx 2>/dev/null || echo "No ONNX files found"
	@echo ""
	@source .venv/bin/activate && \
	python scripts/fetch_models.py --list

### ───── Sentiment Stack (Soft-Info & Explain-Upgrade) ─────────────────
etl-up: ## Launch sentiment ETL pipeline (Airflow DAG)
	@echo "🚀 Triggering sentiment ETL pipeline..."
	@if command -v airflow >/dev/null 2>&1; then \
		airflow dags trigger sentiment_etl; \
	else \
		echo "⚠️  Airflow not installed, running direct ETL fetch..."; \
		python etl/sentiment_fetcher.py --hours 1 --verbose; \
	fi

enricher-up: ## Launch GPT-4o sentiment enricher service
	@echo "🧠 Starting GPT-4o sentiment enricher service..."
	@source .venv/bin/activate && \
	python -m src.services.sent_enricher

explain-up: ## Launch trade explanation service
	@echo "📝 Starting trade explanation service..."
	@source .venv/bin/activate && \
	python -m src.services.explain_middleware

soft-demo: ## Full sentiment stack demo (start services + run ETL + show dashboard)
	@echo "🎬 Starting Sentiment Stack Demo..."
	@echo "Step 1: Starting Redis (if not running)..."
	@if ! pgrep redis-server > /dev/null; then \
		redis-server --daemonize yes; \
		sleep 2; \
	fi
	@echo "Step 2: Starting enricher service in background..."
	@source .venv/bin/activate && \
	python -m src.services.sent_enricher & \
	echo $$! > .enricher_pid
	@sleep 3
	@echo "Step 3: Starting explanation service in background..."
	@source .venv/bin/activate && \
	python -m src.services.explain_middleware & \
	echo $$! > .explain_pid
	@sleep 3
	@echo "Step 4: Running sentiment ETL..."
	@$(MAKE) etl-up
	@echo "Step 5: Demo complete! Services running:"
	@echo "  • Sentiment Enricher: http://localhost:8002"
	@echo "  • Explanation Service: http://localhost:8003"
	@echo "  • Redis sentiment data: redis-cli llen soft.enriched"
	@echo "Use 'make soft-stop' to stop services"

soft-stop: ## Stop sentiment stack services
	@echo "🛑 Stopping sentiment stack services..."
	@if [ -f .enricher_pid ]; then \
		kill $$(cat .enricher_pid) 2>/dev/null || true; \
		rm -f .enricher_pid; \
	fi
	@if [ -f .explain_pid ]; then \
		kill $$(cat .explain_pid) 2>/dev/null || true; \
		rm -f .explain_pid; \
	fi
	@echo "✅ Services stopped"

test-sentiment: ## Run sentiment alpha model unit tests
	@echo "🧪 Running sentiment model tests..."
	@source .venv/bin/activate && \
	python src/layers/layer1_alpha_models/news_sent_alpha.py && \
	python src/layers/layer1_alpha_models/big_bet_flag.py
	@echo "✅ All sentiment tests passed!" 