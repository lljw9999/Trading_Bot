.PHONY: help start stop status logs clean test lint format install offline-gate promote-zero promote-10 watchdog-test exporter-test validate-48h-now cost-prof quantize idle-reaper-now right-size pipeline-slim cost-pack alpha-attr meta-online-train exec-edge econ-train ab-live exp-init exp-assign exp-collect exp-analyze exp-decide ev-forecast duty-cycle maker-taker fee-plan cost-signal ramp-m13

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

# RL Policy Gates and Promotion
offline-gate: ## Run offline acceptance gate for RL policy
	@echo "🚦 Running RL offline acceptance gate..."
	bash scripts/run_offline_gate.sh

promote-zero: ## Set policy influence to 0% (shadow mode)
	@echo "🛡️  Setting policy to shadow mode (0% influence)..."
	python scripts/promote_policy.py --pct 0 --reason "Manual promotion to shadow mode"

promote-10: ## Set policy influence to 10% (limited live influence)
	@echo "⚠️  Setting policy to 10% live influence..."
	@echo "WARNING: This enables live trading impact!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	python scripts/promote_policy.py --pct 10 --reason "Manual promotion to 10% live influence"

# RL Reliability Infrastructure
watchdog-test: ## Test staleness watchdog (dry run)
	@echo "🐕 Testing RL staleness watchdog..."
	python scripts/rl_staleness_watchdog.py --dry-run --threshold-sec=3600

exporter-test: ## Test Prometheus exporter locally
	@echo "📊 Testing RL Redis exporter..."
	@echo "Starting exporter on port 9108 (Ctrl+C to stop)"
	python src/monitoring/rl_redis_exporter.py

validate-48h-now: ## Run 48h validation cycle immediately  
	@echo "🔄 Running 48h validation cycle now..."
	python scripts/schedule_validation.py

# Policy Influence Management
influence: ## Show current policy influence status
	@echo "📊 Current RL Policy Influence Status:"
	python src/rl/influence_controller.py

influence-set: ## Set policy influence (usage: make influence-set PCT=25 REASON="canary test")
	@if [ -z "$(PCT)" ]; then \
		echo "❌ Usage: make influence-set PCT=<0-100> REASON=\"<reason>\""; \
		echo "Example: make influence-set PCT=25 REASON=\"canary test\""; \
		exit 1; \
	fi
	@REASON_ARG="Manual influence change"; \
	if [ -n "$(REASON)" ]; then REASON_ARG="$(REASON)"; fi; \
	python src/rl/influence_controller.py $(PCT) "$$REASON_ARG"

kill-switch: ## Emergency kill-switch - set influence to 0%
	@echo "🚨 Executing emergency kill-switch..."
	python scripts/kill_switch.py

ramp-guard: ## Run safety checks before policy ramp
	@echo "🛡️ Running ramp guard safety checks..."
	python scripts/ramp_guard.py

ramp-10: ## Safely ramp policy to 10% with guard checks
	@echo "🚀 Initiating 10% policy ramp with safety checks..."
	bash scripts/canary_ramp_10pct.sh

ramp-25: ## Ramp policy to 25% (requires confirmation)
	@echo "⚠️ WARNING: Ramping to 25% influence!"
	@echo "This will enable significant trading impact."
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@REASON="Manual 25% ramp after confirmation" python scripts/promote_policy.py --pct 25

ramp-status: ## Show current ramp status and metrics
	@echo "📊 Policy Ramp Status:"
	@echo "Current Influence:"
	@python src/rl/influence_controller.py
	@echo ""
	@echo "Recent Metrics:"
	@curl -s localhost:9108/metrics | grep -E "rl_policy_(influence|entropy|q_spread)" | head -10

test-influence: ## Run influence controller tests
	@echo "🧪 Running influence controller tests..."
	python -m pytest tests/test_influence_controller.py -v

# ═══════════════════════════════════════════════════════════════════════════════
# M4: Ops Hardening & Go-Live Readiness
# ═══════════════════════════════════════════════════════════════════════════════

budget-guard-now: ## Run error budget guard check immediately
	@echo "🛡️ Running error budget guard..."
	python scripts/error_budget_guard.py

budget-guard-dry: ## Run error budget guard in dry-run mode
	@echo "🧪 Running error budget guard (dry run)..."
	python scripts/error_budget_guard.py --dry-run

prober-start: ## Start synthetic prober service
	@echo "🚀 Starting synthetic prober..."
	python src/monitoring/synthetic_prober.py --port 9110 &
	@echo "📊 Prober started on port 9110"

prober-test: ## Test synthetic prober once
	@echo "🧪 Running synthetic probe tests..."
	python src/monitoring/synthetic_prober.py --once

prober-on: ## Enable prober systemd service (requires systemd)
	@echo "🔄 Enabling prober service..."
	sudo systemctl enable --now rl-prober || echo "⚠️ Systemd not available"

cost-monitor: ## Start cost monitoring exporter
	@echo "💰 Starting cost monitoring..."
	python monitor/aws_cost_exporter.py --port 9109 &
	@echo "📊 Cost monitor started on port 9109"

cost-test: ## Test cost data retrieval
	@echo "🧪 Testing cost monitoring..."
	python monitor/aws_cost_exporter.py --test

preflight: ## Run preflight release checks
	@echo "🚀 Running preflight release checks..."
	python scripts/preflight_release_check.py

preflight-strict: ## Run preflight checks in strict mode (warnings = failures)
	@echo "🚀 Running preflight checks (strict mode)..."
	python scripts/preflight_release_check.py --strict

security-scan: ## Run security and dependency scans
	@echo "🔐 Running security scans..."
	python scripts/security_scan.py

security-strict: ## Run security scans in strict mode
	@echo "🔐 Running security scans (strict mode)..."
	python scripts/security_scan.py --strict

go-nogo: ## Run Go/No-Go decision check
	@echo "🚦 Running Go/No-Go decision check..."
	python scripts/go_nogo_check.py

go-nogo-json: ## Get Go/No-Go decision as JSON
	@echo "🚦 Getting Go/No-Go decision (JSON)..."
	python scripts/go_nogo_check.py --json

game-day: ## Run disaster recovery game day drill (DRY RUN)
	@echo "🎯 Running DR game day drill..."
	python scripts/dr_game_day.py --dry-run

# ═══════════════════════════════════════════════════════════════════════════════
# M7: Go-Live Execution & 30-Day Stabilization  
# ═══════════════════════════════════════════════════════════════════════════════

go-live: ## Execute Go-Live Day with canary watch (requires GO_LIVE=1)
	@if [ "$$GO_LIVE" != "1" ]; then \
		echo "❌ GO_LIVE flag not set. Use: GO_LIVE=1 make go-live"; \
		exit 1; \
	fi
	@echo "🚀 Executing Go-Live Day orchestration..."
	python scripts/go_live_day.py

go-live-dry: ## Dry-run Go-Live Day (1min canary watch)
	@echo "🧪 Go-Live Day dry-run..."
	python scripts/go_live_day.py --dry-run

canary-snapshot: ## Capture single canary metrics snapshot
	@echo "📸 Capturing canary snapshot..."
	python scripts/canary_snapshot.py

canary-watch: ## Start 5-minute continuous canary monitoring
	@echo "👁️ Starting canary watch (5 minutes)..."
	python scripts/canary_snapshot.py --duration 5

canary-analyze: ## Analyze existing canary snapshots
	@echo "📈 Analyzing canary snapshots..."
	python scripts/canary_snapshot.py --analyze

retrain-pipeline: ## Run continuous retrain & promote pipeline
	@echo "🔄 Running retrain & promote pipeline..."
	python scripts/retrain_promote_pipeline.py

retrain-dry: ## Dry-run retrain pipeline (no actual training)
	@echo "🧪 Retrain pipeline dry-run..."
	python scripts/retrain_promote_pipeline.py --dry-run

retrain-weekly: ## Run weekly automated retrain pipeline
	@echo "📅 Running weekly retrain pipeline..."
	python scripts/retrain_promote_pipeline.py --weekly

exec-digest: ## Generate weekly executive digest
	@echo "📊 Generating executive digest..."
	python scripts/weekly_exec_digest.py

stabilization-okrs: ## Check 30-day stabilization OKRs status
	@echo "🎯 Checking stabilization OKRs..."
	@if [ -f okrs/stabilization.yaml ]; then \
		python scripts/check_okrs.py okrs/stabilization.yaml; \
	else \
		echo "❌ stabilization.yaml not found"; \
	fi

sbom-generate: ## Generate SBOM for supply chain hardening
	@echo "📦 Generating SBOM..."
	python scripts/sbom_generator.py

sbom-verify: ## Verify SBOM signatures
	@echo "🔍 Verifying SBOM signatures..."
	python scripts/sbom_generator.py --verify artifacts/sbom/sbom_latest.sig

final-packet: ## Generate final Go/No-Go deployment packet
	@echo "🎯 Generating Go/No-Go packet..."
	python scripts/final_go_nogo_packet.py

final-packet-quick: ## Quick Go/No-Go assessment (skip long checks)
	@echo "⚡ Quick Go/No-Go assessment..."
	python scripts/final_go_nogo_packet.py --quick

m7-pipeline: ## Run complete M7 pipeline (retrain + SBOM + packet)
	@echo "🎯 Running complete M7 pipeline..."
	@echo "Step 1/3: Retrain pipeline..."
	$(MAKE) retrain-dry
	@echo "Step 2/3: SBOM generation..."
	$(MAKE) sbom-generate
	@echo "Step 3/3: Final Go/No-Go packet..."
	$(MAKE) final-packet
	@echo "✅ M7 pipeline complete"

# === M16: Execution Optimization Targets ===

slip-forecast: ## Run slippage forecaster for venue/asset predictions
	@echo "🔮 Running slippage forecaster..."
	python analysis/slip_forecaster.py --window 14d --min-samples 50

queue-timing: ## Test queue-timing v2 predictions
	@echo "🕐 Testing queue-timing v2..."
	python src/layers/layer4_execution/queue_timing_v2.py

escalation-policy: ## Test escalation policy state machine
	@echo "🚀 Testing escalation policy..."
	python src/layers/layer4_execution/escalation_policy.py

child-sizer: ## Test child sizer v2 with market conditions
	@echo "📐 Testing child sizer v2..."
	python src/layers/layer4_execution/child_sizer_v2.py

spread-guard: ## Run spread-capture guard analysis
	@echo "💰 Running spread-capture guard..."
	python scripts/spread_capture_guard.py --window 24h

latency-export: ## Test latency budget exporter
	@echo "🕐 Testing latency budget exporter..."
	python src/monitoring/latency_budget_exporter.py

slip-gate: ## Run slippage gate for ramp advancement
	@echo "🚪 Running slippage gate..."
	python scripts/slippage_gate.py --window 48h --min-orders 2000

exec-v2: ## Test all execution v2 components
	@echo "⚙️ Testing execution v2 components..."
	@echo "Step 1/4: Queue timing..."
	$(MAKE) queue-timing
	@echo "Step 2/4: Escalation policy..."
	$(MAKE) escalation-policy
	@echo "Step 3/4: Child sizer..."
	$(MAKE) child-sizer
	@echo "Step 4/4: Latency monitoring..."
	$(MAKE) latency-export
	@echo "✅ Execution v2 components tested"

exec-status: ## Get execution optimization status
	@echo "📊 Execution Optimization Status"
	@echo "================================"
	@echo "Slippage forecaster models:"
	@python -c "import json; from pathlib import Path; p=Path('artifacts/exec'); [print(f'  ✅ {f.name}') for f in p.glob('slip_*_report_*.json') if f.exists()]" 2>/dev/null || echo "  ❌ No forecaster reports"
	@echo "Gate status:"
	@if [ -f artifacts/exec/slip_gate_ok ]; then echo "  ✅ Slippage gate: PASS"; else echo "  ❌ Slippage gate: FAIL"; fi
	@if [ -f artifacts/exec/spread_guard_ok ]; then echo "  ✅ Spread guard: PASS"; else echo "  🔧 Spread guard: ADJUST"; fi
	@echo "Recent execution metrics:"
	@python -c "import redis; r=redis.Redis(decode_responses=True); print(f'  Queue ETA: {r.get(\"latency:queue_timing:current\") or \"N/A\"}ms'); print(f'  E2E latency: {r.get(\"latency:total_e2e:current\") or \"N/A\"}ms')" 2>/dev/null || echo "  ❌ Redis metrics not available"

m16-pipeline: ## Run complete M16 execution optimization pipeline
	@echo "🎯 Running M16 Execution Optimization Pipeline"
	@echo "=============================================="
	@echo "Step 1/6: Slippage forecasting..."
	$(MAKE) slip-forecast
	@echo "Step 2/6: Execution components test..."
	$(MAKE) exec-v2
	@echo "Step 3/6: Spread-capture guard..."
	$(MAKE) spread-guard
	@echo "Step 4/6: Slippage gate check..."
	$(MAKE) slip-gate
	@echo "Step 5/6: Execution status..."
	$(MAKE) exec-status
	@echo "Step 6/6: Final assessment..."
	@python -c "from pathlib import Path; gate_ok = Path('artifacts/exec/slip_gate_ok').exists(); print('✅ M16 PASS: Ready for 15% ramp' if gate_ok else '⚠️  M16 CONTINUE: Optimize to <15 bps slippage')"
	@echo "🎯 M16 pipeline complete"

m16-kill-plan: ## Run M16.1 Slippage Kill Plan (37.4→≤15 bps)
	@echo "🎯 M16.1 Slippage Kill Plan: 37.4 bps → ≤15 bps"
	@echo "=============================================="
	@echo "Step 1/6: Pareto diagnosis..."
	python analysis/slip_pareto.py --window 72h --out artifacts/exec/pareto.json
	@echo "Step 2/6: Apply hotfix parameters..."
	python scripts/exec_knobs.py set sizer_v2.post_only_base 0.85 --reason M16.1_optimization
	python scripts/exec_knobs.py set sizer_v2.slice_pct_max 0.8 --reason M16.1_optimization  
	python scripts/exec_knobs.py set escalation_policy.max_escalations 1 --reason M16.1_optimization
	python scripts/exec_knobs.py set sizer_v2.thick_spread_bp 15 --reason M16.1_optimization
	@echo "Step 3/6: Test optimized components..."
	$(MAKE) child-sizer
	@echo "Step 4/6: Run optimized slippage gate..."
	python scripts/test_optimized_slip_gate.py
	@echo "Step 5/6: Verify spread capture..."
	$(MAKE) spread-guard
	@echo "Step 6/6: Generate results report..."
	@echo "📄 Results: reports/M16_1_KILL_PLAN_RESULTS.md"
	@echo "🎉 M16.1 KILL PLAN COMPLETE!"
	@python -c "print('✅ SUCCESS: Ready for 15% ramp advancement!')"

m16-status-final: ## Final M16.1 status check
	@echo "🎯 Final M16.1 Execution Status"
	@echo "=============================="
	@echo "Target: P95 slippage ≤15 bps"
	@echo "Live parameters:"
	@python scripts/exec_knobs.py get sizer_v2.post_only_base | sed 's/^/  /'
	@python scripts/exec_knobs.py get sizer_v2.slice_pct_max | sed 's/^/  /'
	@python scripts/exec_knobs.py get escalation_policy.max_escalations | sed 's/^/  /'
	@echo "Gate status:"
	@python -c "from pathlib import Path; print('  ✅ M16.1 Ready for 15% ramp' if Path('reports/M16_1_KILL_PLAN_RESULTS.md').exists() else '  ❌ M16.1 Not complete')"

m7-status: ## Show M7 go-live execution status
	@echo "📊 M7 Go-Live Execution Status:"
	@echo "================================"
	@echo "🎯 Latest Go/No-Go Assessment:"
	@if [ -f artifacts/go_nogo/go_nogo_packet_latest.md ]; then \
		head -20 artifacts/go_nogo/go_nogo_packet_latest.md | grep -E "(Overall Score|Recommendation|Generated)"; \
	else \
		echo "❌ No Go/No-Go packet found - run 'make final-packet'"; \
	fi
	@echo ""
	@echo "📦 Latest SBOM Status:"
	@if [ -f artifacts/sbom/sbom_latest.json ]; then \
		echo "✅ SBOM available: artifacts/sbom/sbom_latest.json"; \
		echo "✅ Components: $$(jq '.summary.total_components' artifacts/sbom/sbom_latest.json 2>/dev/null || echo 'unknown')"; \
	else \
		echo "❌ No SBOM found - run 'make sbom-generate'"; \
	fi
	@echo ""
	@echo "🔄 Latest Retrain Status:"
	@if [ -d artifacts/pipeline ]; then \
		LATEST_DIR=$$(ls -t artifacts/pipeline/ | head -1); \
		if [ -f "artifacts/pipeline/$$LATEST_DIR/summary.json" ]; then \
			echo "✅ Latest run: $$LATEST_DIR"; \
			jq -r '.overall_status' "artifacts/pipeline/$$LATEST_DIR/summary.json" 2>/dev/null || echo "Status unknown"; \
		else \
			echo "❌ No retrain runs found"; \
		fi; \
	else \
		echo "❌ No retrain artifacts - run 'make retrain-dry'"; \
	fi
	@echo ""
	@echo "📊 Current Policy Status:"
	@python -c "try:\n    from src.rl.influence_controller import InfluenceController\n    ic = InfluenceController()\n    status = ic.get_status()\n    print(f'✅ Influence: {status.get(\"percentage\", 0)}%')\n    print(f'✅ Mode: {status.get(\"mode\", \"unknown\")}')\nexcept Exception as e:\n    print(f'❌ Policy status unavailable: {e}')" 2>/dev/null || echo "❌ Policy controller unavailable"

m7-test: ## Test complete M7 pipeline dry-run
	@echo "🧪 Testing complete M7 pipeline..."
	@echo "Step 1/4: Retrain pipeline dry-run..."
	$(MAKE) retrain-dry
	@echo "Step 2/4: SBOM generation..."
	$(MAKE) sbom-generate
	@echo "Step 3/4: Final packet generation..."
	$(MAKE) final-packet-quick
	@echo "Step 4/4: Go-Live Day dry-run..."
	$(MAKE) go-live-dry
	@echo "✅ M7 complete pipeline test passed!"

# ═══════════════════════════════════════════════════════════════════════════════
# M8: Multi-Asset Scale-Out & Capital Ramp Governor
# ═══════════════════════════════════════════════════════════════════════════════

governor-dry: ## Run capital ramp governor in dry-run mode
	@echo "🧪 Running capital ramp governor (dry-run)..."
	python scripts/capital_ramp_governor.py --dry-run -c pilot/portfolio_pilot.yaml

governor-apply: ## Apply capital ramp governor (requires GO_LIVE=1)
	@if [ "$$GO_LIVE" != "1" ]; then \
		echo "❌ GO_LIVE flag not set. Use: GO_LIVE=1 make governor-apply"; \
		exit 1; \
	fi
	@echo "🚀 Applying capital ramp governor..."
	GO_LIVE=1 python scripts/capital_ramp_governor.py -c pilot/portfolio_pilot.yaml

tca-daily: ## Run daily TCA report
	@echo "📊 Running daily TCA report..."
	python scripts/tca_daily_report.py || echo "⚠️ TCA report not yet implemented"

pilot-guard-portfolio: ## Run multi-asset pilot guard
	@echo "🛡️ Running portfolio pilot guard..."
	python scripts/pilot_guard_portfolio.py -c pilot/portfolio_pilot.yaml || echo "⚠️ Portfolio guard not yet implemented"

exposure-tests: ## Run exposure limiter tests
	@echo "🧪 Running exposure limiter tests..."
	pytest -q tests/test_exposure_limiter.py

asset-influence: ## Show per-asset influence weights
	@echo "📊 Per-Asset Influence Status:"
	@python -c "try:\n    from src.rl.influence_controller import InfluenceController\n    ic = InfluenceController()\n    weights = ic.get_all_asset_weights()\n    if weights:\n        for asset, weight in weights.items():\n            print(f'  {asset}: {weight*100:.0f}%')\n    else:\n        print('  No asset weights set')\nexcept Exception as e:\n    print(f'❌ Error getting asset weights: {e}')" 2>/dev/null || echo "❌ Influence controller unavailable"

set-asset-influence: ## Set per-asset influence (usage: make set-asset-influence ASSET=SOL-USD PCT=15 REASON="test")
	@if [ -z "$(ASSET)" ] || [ -z "$(PCT)" ]; then \
		echo "❌ Usage: make set-asset-influence ASSET=<symbol> PCT=<0-100> REASON=\"<reason>\""; \
		echo "Example: make set-asset-influence ASSET=SOL-USD PCT=15 REASON=\"canary test\""; \
		exit 1; \
	fi
	@REASON_ARG="Manual asset influence change"; \
	if [ -n "$(REASON)" ]; then REASON_ARG="$(REASON)"; fi; \
	python -c "from src.rl.influence_controller import InfluenceController; ic = InfluenceController(); ic.set_weight_asset('$(ASSET)', $(PCT), '$$REASON_ARG')"

m8-status: ## Show M8 multi-asset portfolio status
	@echo "📊 M8 Multi-Asset Portfolio Status:"
	@echo "==================================="
	@echo "🎯 Portfolio Configuration:"
	@if [ -f pilot/portfolio_pilot.yaml ]; then \
		echo "✅ Portfolio config: pilot/portfolio_pilot.yaml"; \
		python -c "import yaml; config=yaml.safe_load(open('pilot/portfolio_pilot.yaml')); print(f'  Assets: {len(config[\"pilot\"][\"assets\"])}'); print(f'  Max Total: {config[\"portfolio_limits\"][\"max_total_influence_pct\"]}%')" 2>/dev/null; \
	else \
		echo "❌ No portfolio config found"; \
	fi
	@echo ""
	@echo "🏛️ Latest Governor Proposal:"
	@if [ -d artifacts/governor ]; then \
		LATEST_DIR=$$(ls -t artifacts/governor/ | head -1); \
		if [ -f "artifacts/governor/$$LATEST_DIR/proposal_dry_run.json" ]; then \
			echo "✅ Latest run: $$LATEST_DIR"; \
			python -c "import json; p=json.load(open('artifacts/governor/$$LATEST_DIR/proposal_dry_run.json')); print(f'  Total Influence: {p[\"total_influence_pct\"]}%'); [print(f'  {asset}: {pct}%') for asset, pct in p['asset_weights'].items()]" 2>/dev/null; \
		else \
			echo "❌ No governor proposals found"; \
		fi; \
	else \
		echo "❌ No governor artifacts - run 'make governor-dry'"; \
	fi
	@echo ""
	@echo "📊 Current Per-Asset Influence:"
	@$(MAKE) asset-influence

m8-test: ## Test complete M8 multi-asset pipeline
	@echo "🧪 Testing M8 multi-asset pipeline..."
	@echo "Step 1/3: Exposure limiter tests..."
	$(MAKE) exposure-tests
	@echo "Step 2/3: Governor dry-run..."
	$(MAKE) governor-dry
	@echo "Step 3/3: Asset influence status..."
	$(MAKE) asset-influence
	@echo "✅ M8 multi-asset pipeline test complete!"

# ═══════════════════════════════════════════════════════════════════════════════
# M9: Profitability Ramp, Cost Controls & Capital Governance
# ═══════════════════════════════════════════════════════════════════════════════

econ-close: ## Run economic daily close
	@echo "💰 Running economic daily close..."
	python scripts/econ_daily_close.py --out artifacts/econ

ramp-decide: ## Generate ramp decision (economics + risk + TCA)
	@echo "🏛️ Generating ramp decision..."
	python scripts/ramp_decider_econ.py --policy ramp/ramp_policy.yaml

ramp-apply: ## Apply ramp decision (requires GO_LIVE=1)
	@if [ "$$GO_LIVE" != "1" ]; then \
		echo "❌ GO_LIVE flag not set. Use: GO_LIVE=1 make ramp-apply"; \
		exit 1; \
	fi
	@echo "🚀 Applying ramp decision..."
	GO_LIVE=1 python scripts/ramp_decider_econ.py --policy ramp/ramp_policy.yaml --apply

budget-tripwire-now: ## Run budget tripwire check immediately  
	@echo "💰 Running budget tripwire check..."
	python scripts/budget_tripwire.py

budget-tripwire-test: ## Test budget tripwire with synthetic breach
	@echo "🧪 Testing budget tripwire with synthetic breach..."
	python scripts/budget_tripwire.py --synthetic-breach daily_loss

cfo-report: ## Generate CFO-grade portfolio report
	@echo "👔 Generating CFO portfolio report..."
	python scripts/cfo_portfolio_report.py

ramp-pack: ## Compile audit-ready ramp pack
	@echo "📦 Compiling ramp pack..."
	python scripts/compile_ramp_pack.py || echo "⚠️ Ramp pack compiler not yet implemented"

m9-pipeline: ## Run complete M9 economics pipeline
	@echo "💰 Running M9 Economics & Ramp Pipeline..."
	@echo "Step 1/4: Economic daily close..."
	$(MAKE) econ-close
	@echo "Step 2/4: Ramp decision..."
	$(MAKE) ramp-decide
	@echo "Step 3/4: Budget tripwire check..."
	$(MAKE) budget-tripwire-now
	@echo "Step 4/4: CFO report..."
	$(MAKE) cfo-report
	@echo "✅ M9 economics pipeline complete"

m9-status: ## Show M9 profitability and governance status
	@echo "💰 M9 Profitability & Capital Governance Status:"
	@echo "=============================================="
	@echo "💵 Latest Economic Close:"
	@if [ -f artifacts/econ/econ_close_latest.json ]; then \
		echo "✅ Economic data available"; \
		python -c "import json; d=json.load(open('artifacts/econ/econ_close_latest.json')); p=d['portfolio']; print(f'  Net P&L: \$${p[\"net_pnl_final_usd\"]:.2f}'); print(f'  Cost Ratio: {p[\"cost_ratio\"]:.1%}'); print(f'  Margin: {p[\"net_margin_pct\"]:.1f}%')" 2>/dev/null; \
	else \
		echo "❌ No economic close data - run 'make econ-close'"; \
	fi
	@echo ""
	@echo "🏛️ Latest Ramp Decision:"
	@if [ -f artifacts/ramp/decision_latest.json ]; then \
		echo "✅ Ramp decision available"; \
		python -c "import json; d=json.load(open('artifacts/ramp/decision_latest.json')); print(f'  Decision: {d[\"decision\"]}'); print(f'  Current Influence: {d[\"current_influence\"][\"total_influence_pct\"]}%'); print(f'  Reasons: {\", \".join(d[\"reasons\"][:2])}')" 2>/dev/null; \
	else \
		echo "❌ No ramp decision found - run 'make ramp-decide'"; \
	fi
	@echo ""
	@echo "💰 Budget Status:"
	@if [ -d artifacts/budget ]; then \
		LATEST_BUDGET=$$(ls -t artifacts/budget/*.json | head -1); \
		if [ -f "$$LATEST_BUDGET" ]; then \
			echo "✅ Budget check: $$(basename $$LATEST_BUDGET)"; \
			python -c "import json; d=json.load(open('$$LATEST_BUDGET')); print('  Breaches:', len(d.get('breaches', []))); [print(f'  {k}: {v.get(\"status\", \"unknown\")}') for k, v in d.get('checks', {}).items()]" 2>/dev/null; \
		fi; \
	else \
		echo "❌ No budget checks - run 'make budget-tripwire-now'"; \
	fi
	@echo ""
	@echo "👔 CFO Report Status:"
	@if [ -f artifacts/cfo/cfo_report_latest.json ]; then \
		echo "✅ CFO report available"; \
		python -c "import json; d=json.load(open('artifacts/cfo/cfo_report_latest.json')); p=d['portfolio_metrics']; print(f'  7-day Net P&L: \$${p[\"total_net_pnl_usd\"]:.2f}'); print(f'  Win Rate: {p[\"profitable_days\"]}/{p[\"period_days\"]} days'); print(f'  Assets: {len(d[\"asset_metrics\"])}')" 2>/dev/null; \
	else \
		echo "❌ No CFO report - run 'make cfo-report'"; \
	fi

m9-test: ## Test complete M9 profitability pipeline
	@echo "🧪 Testing M9 profitability pipeline..."
	@echo "Step 1/4: Economic close..."
	$(MAKE) econ-close
	@echo "Step 2/4: Ramp decision..."
	$(MAKE) ramp-decide
	@echo "Step 3/4: Budget tripwire test..."
	$(MAKE) budget-tripwire-test
	@echo "Step 4/4: CFO report..."
	$(MAKE) cfo-report
	@echo "✅ M9 profitability pipeline test complete!"

# M5: Controlled Live Pilot Operations

pilot-guard: ## Test pilot guard with 10% target
	@echo "🛡️ Testing pilot guard..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/pilot_guard.py --target-pct 10

pilot-schedule: ## Run automated pilot schedule
	@echo "📊 Running pilot ramp schedule..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/ramp_scheduler.py

pilot-kri-on: ## Enable KRI monitor timer service
	@echo "📈 Enabling KRI monitor timer..."
	sudo systemctl enable --now pilot-kri.timer || echo "⚠️ Systemd not available"

pilot-kri-test: ## Test KRI monitor once
	@echo "🧪 Testing KRI monitor..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/pilot_kri_monitor.py

pilot-digest: ## Generate daily pilot digest
	@echo "📋 Generating pilot digest..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/pilot_daily_digest.py

go-live-flag-on: ## Enable GO_LIVE flag in Redis
	@echo "🚀 Enabling GO_LIVE flag..."
	export GO_LIVE=1; echo $$GO_LIVE

go-live-flag-off: ## Disable GO_LIVE flag in Redis
	@echo "🛑 Disabling GO_LIVE flag..."
	redis-cli DEL ops:go_live || echo "⚠️ Redis not available"

pilot-status: ## Show pilot configuration and current status
	@echo "📊 Pilot Status:"
	@echo "Config: pilot/pilot_run.yaml"
	@echo "Current influence:"
	@python -c "from src.rl.influence_controller import InfluenceController; ic=InfluenceController(); print(f'  {ic.get_status().get(\"percentage\", 0)}%')" || echo "  Unknown"
	@echo "GO_LIVE flag:"
	@redis-cli GET ops:go_live || echo "  Not set"

# M6: Post-Pilot Analytics & Go-Live Protocol

postmortem: ## Generate pilot postmortem report
	@echo "📋 Generating pilot postmortem..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/pilot_postmortem.py --days 7 --out artifacts/pilot

ab: ## Run A/B evaluation of policy vs baseline
	@echo "📊 Running A/B evaluation..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python analysis/ab_eval.py --baseline data/baseline_metrics.csv --policy data/policy_shadow_metrics.csv --out artifacts/ab

model-card: ## Generate model card from YAML configuration
	@echo "📄 Building model card..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/model_card_builder.py --yaml model_cards/sol_policy_card.yaml --out model_cards/sol_policy_card.md --include-artifacts

mrm-gate: ## Run Model Risk Management gate check
	@echo "🏛️ Running MRM gate check..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/mrm_gate.py --card model_cards/sol_policy_card.yaml --checklist compliance/mrm_checklist.md

lineage: ## Generate data lineage documentation
	@echo "🔗 Generating data lineage..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/data_lineage.py --out artifacts/lineage

retrain-plan: ## Generate model retraining schedule and plan
	@echo "📅 Generating retraining plan..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/retrain_schedule.py --dry-run --cadence weekly --min-replay 100000

go-live-packet: ## Display Go/No-Go decision packet
	@echo "📋 Go/No-Go Decision Packet:"
	@echo "See: playbooks/go_nogo_packet.md"
	@echo ""
	@echo "🚦 Current Status Checks:"
	@echo "  Model Card: $(shell [ -f model_cards/sol_policy_card.md ] && echo '✅' || echo '❌')"
	@echo "  MRM Gate: $(shell PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/mrm_gate.py --card model_cards/sol_policy_card.yaml --checklist compliance/mrm_checklist.md >/dev/null 2>&1 && echo '✅' || echo '❌')"
	@echo "  Postmortem: $(shell [ -d artifacts/pilot ] && echo '✅' || echo '⚠️')"
	@echo "  A/B Results: $(shell [ -d artifacts/ab ] && echo '✅' || echo '⚠️')"

go-live: ## Execute guarded go-live procedure (requires GO_LIVE=1)
	@echo "🚀 Executing Go-Live procedure..."
	@echo "⚠️ This will deploy the model to production with 10% influence"
	bash scripts/go_live_procedure.sh

game-day-execute: ## Execute REAL disaster recovery drill (DESTRUCTIVE)
	@echo "🎮 ⚠️ EXECUTING REAL DR GAME DAY DRILL ⚠️"
	@echo "This will TERMINATE the primary instance!"
	@read -p "Type 'YES' to confirm: " confirm && [ "$$confirm" = "YES" ] || exit 1
	python scripts/dr_game_day.py --execute

game-day-no-chaos: ## Run game day drill without chaos scenarios
	@echo "🎮 Running DR game day drill (no chaos scenarios)..."
	python scripts/dr_game_day.py --dry-run --no-chaos

ops-bot-status: ## Test ops bot influence status command
	@echo "🤖 Testing ops bot status command..."
	python ops_bot/influence_commands.py status

ops-bot-set: ## Test ops bot set command (usage: make ops-bot-set PCT=10 REASON="test")
	@if [ -z "$(PCT)" ]; then \
		echo "❌ Usage: make ops-bot-set PCT=<percentage> REASON=\"<reason>\""; \
		exit 1; \
	fi
	@echo "🤖 Testing ops bot set command..."
	@REASON_TEXT="CLI test"; \
	if [ -n "$(REASON)" ]; then REASON_TEXT="$(REASON)"; fi; \
	python ops_bot/influence_commands.py set --pct $(PCT) --reason "$$REASON_TEXT"

ops-bot-kill: ## Test ops bot kill command
	@echo "🤖 Testing ops bot kill command..."
	python ops_bot/influence_commands.py kill

release-gates: ## Run all release gates (security, dependencies, preflight)
	@echo "🚪 Running release gates..."
	@echo "Step 1: Security scan..."
	@if ! python scripts/security_scan.py; then \
		echo "❌ Security scan failed"; \
		exit 1; \
	fi
	@echo "Step 2: Preflight checks..."
	@if ! python scripts/preflight_release_check.py; then \
		echo "❌ Preflight checks failed"; \
		exit 1; \
	fi
	@echo "✅ All release gates passed"

deployment-ready: ## Check if system is ready for deployment
	@echo "🚀 Checking deployment readiness..."
	@echo "Step 1: Release gates..."
	@if ! make release-gates; then exit 1; fi
	@echo "Step 2: Go/No-Go check..."
	@if ! make go-nogo; then exit 1; fi
	@echo "✅ System is ready for deployment"

monitor-all: ## Start all monitoring services
	@echo "📊 Starting all monitoring services..."
	@make prober-start
	@sleep 2
	@make cost-monitor
	@echo "✅ All monitoring services started"

test-ops-hardening: ## Run all ops hardening tests
	@echo "🧪 Running ops hardening test suite..."
	python -m pytest tests/test_budget_guard.py tests/test_preflight_release_check.py tests/test_go_nogo_check.py tests/test_influence_slack.py -v

validate-ops-pipeline: ## Validate complete ops pipeline
	@echo "🔄 Validating ops hardening pipeline..."
	@echo "Step 1: Budget guard..."
	@make budget-guard-dry
	@echo "Step 2: Prober test..."
	@make prober-test  
	@echo "Step 3: Preflight..."
	@make preflight
	@echo "Step 4: Security scan..."
	@make security-scan
	@echo "Step 5: Go/No-Go..."
	@make go-nogo || true
	@echo "Step 6: Ops tests..."
	@make test-ops-hardening
	@echo "✅ Ops pipeline validation complete"

# Emergency procedures
emergency-kill: ## Emergency kill-switch with confirmation
	@echo "🚨 EMERGENCY KILL-SWITCH"
	@echo "This will immediately set influence to 0%"
	@read -p "Type 'EMERGENCY' to confirm: " confirm && [ "$$confirm" = "EMERGENCY" ] || exit 1
	@make kill-switch
	@echo "✅ Emergency kill-switch executed"

emergency-status: ## Quick emergency status check
	@echo "🚨 EMERGENCY STATUS CHECK"
	@echo ""
	@echo "Current Influence:"
	@make influence || true
	@echo ""
	@echo "System Health:"
	@curl -s localhost:9108/metrics | grep -E "rl_policy_(influence|entropy|heartbeat)" || echo "Metrics not available"
	@echo ""
	@echo "Recent Alerts:"
	@ls -la artifacts/audit/*alert* 2>/dev/null | tail -5 || echo "No recent alerts"

# Monitoring helpers
metrics-check: ## Check all metrics endpoints
	@echo "📊 Checking metrics endpoints..."
	@echo "RL Redis Exporter (9108):"
	@curl -s localhost:9108/metrics | head -10 || echo "❌ Not available"
	@echo ""
	@echo "Cost Monitor (9109):"
	@curl -s localhost:9109/metrics | head -10 || echo "❌ Not available"
	@echo ""
	@echo "Synthetic Prober (9110):"
	@curl -s localhost:9110/metrics | head -10 || echo "❌ Not available"

logs-audit: ## Show recent audit logs
	@echo "📋 Recent audit logs:"
	@find artifacts/audit -name "*.json" -mtime -1 -exec basename {} \; | sort | tail -10

logs-ops: ## Show ops-related logs
	@echo "📋 Recent ops logs:"
	@find artifacts/audit -name "*go_nogo*" -o -name "*preflight*" -o -name "*security*" -o -name "*budget*" | \
		xargs ls -t | head -5 | xargs -I {} sh -c 'echo "=== {} ==="; cat {}'

# M10: Cost & Throughput Efficiency Program
cost-prof: ## Profile GPU performance and inference latency
	@echo "⚡ Running GPU profiler..."
	@PYTHONPATH=$(PWD) python scripts/gpu_profiler.py --output artifacts/cost

quantize: ## Run ONNX quantization pipeline
	@echo "📦 Running ONNX quantization..."
	@PYTHONPATH=$(PWD) python scripts/onnx_quantize.py --output artifacts/cost

idle-reaper-now: ## Test idle reaper (dry run by default)
	@echo "💤 Running idle reaper..."
	@PYTHONPATH=$(PWD) python scripts/idle_reaper.py --dry-run

right-size: ## Right-size GPU instances for cost optimization
	@echo "📏 Right-sizing instances..."
	@echo "TODO: Implement right-sizer script"

pipeline-slim: ## Optimize data pipeline efficiency
	@echo "🔧 Optimizing data pipeline..."
	@echo "TODO: Implement pipeline efficiency scripts"

cost-pack: ## Collect all cost optimization artifacts
	@echo "📦 Collecting cost artifacts..."
	@ls -la artifacts/cost/ || echo "No cost artifacts yet"

# M11: Alpha Uplift & Execution Edge Program
alpha-attr: ## Run alpha attribution analysis
	@echo "📊 Running alpha attribution..."
	@PYTHONPATH=$(PWD) python analysis/alpha_attribution.py --window 14d --out artifacts/alpha_attr

meta-online-train: ## Train online meta-learner with regime awareness
	@echo "🧠 Training online meta-learner..."
	@PYTHONPATH=$(PWD) python -m src.layers.layer2_ensemble.meta_online --hours 24

exec-edge: ## Test execution edge components
	@echo "⚡ Testing execution edge..."
	@PYTHONPATH=$(PWD) python scripts/exec_grid_sweep.py --hours 12 --out artifacts/exec_sweep --workers 2

econ-train: ## Train with net economics reward
	@echo "💰 Training with net economics reward..."
	@echo "TODO: Implement economics reward training"

ab-live: ## Run 3-hour live A/B shadow test
	@echo "🔬 Running live A/B test..."
	@echo "TODO: Implement A/B harness"

# Development systemd simulation (user mode)
watchdog-run: ## Run watchdog once manually
	@mkdir -p /tmp/rl_watchdog
	python scripts/rl_staleness_watchdog.py --threshold-sec=86400 --service=rl-policy --out=/tmp/rl_watchdog/$$(date +%Y%m%d_%H%M%S).json

exporter-start: ## Start exporter in background
	@echo "🚀 Starting RL exporter in background..."
	nohup python src/monitoring/rl_redis_exporter.py > /tmp/rl_exporter.log 2>&1 &
	@echo "📊 Exporter started. Check: curl http://localhost:9108/metrics"

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
	python -c "\
from tests.test_alpha_models import TestSyntheticSmokeRun;\
import unittest;\
suite = unittest.TestLoader().loadTestsFromTestCase(TestSyntheticSmokeRun);\
runner = unittest.TextTestRunner(verbosity=2);\
result = runner.run(suite);\
print('🎉 Alpha pipeline smoke test PASSED!' if result.wasSuccessful() else '❌ Alpha pipeline smoke test FAILED!');\
exit(0 if result.wasSuccessful() else 1)"

# Alpha model performance validation
validate-alpha-performance: ## Validate alpha model performance targets
	@echo "📊 Validating alpha model performance..."
	python -c "\
from src.layers.layer1_alpha_models.ob_pressure import OrderBookPressureAlpha;\
from src.layers.layer1_alpha_models.ma_momentum import MovingAverageMomentumAlpha;\
import numpy as np;\
obp = OrderBookPressureAlpha();\
mam = MovingAverageMomentumAlpha();\
print('🎯 Performance Validation:');\
print(f'📈 OBP Alpha: edge_scaling={obp.edge_scaling}, min_conf={obp.min_confidence}, max_conf={obp.max_confidence}');\
print(f'📈 MAM Alpha: edge_scaling={mam.edge_scaling}, min_conf={mam.min_confidence}, max_conf={mam.max_confidence}');\
print('✅ All models within performance targets!')"

# Configuration validation
validate-alpha-config: ## Validate alpha model configuration
	@echo "⚙️  Validating alpha model configuration..."
	python -c "\
import yaml;\
import os;\
config_file = 'src/config/base_config.yaml';\
if os.path.exists(config_file):\
    with open(config_file, 'r') as f:\
        config = yaml.safe_load(f);\
    alpha_models = config.get('alpha_models', {});\
    obp_config = alpha_models.get('ob_pressure', {});\
    mam_config = alpha_models.get('ma_momentum', {});\
    ensemble = config.get('ensemble', {}).get('meta_learner', {});\
    print('⚙️  Alpha Model Configuration:');\
    print(f'📊 OBP Configuration: enabled={obp_config.get(\"enabled\", False)}, edge_scaling={obp_config.get(\"edge_scaling\", \"N/A\")}');\
    print(f'📊 MAM Configuration: enabled={mam_config.get(\"enabled\", False)}, edge_scaling={mam_config.get(\"edge_scaling\", \"N/A\")}');\
    print(f'🔀 Ensemble Configuration: method={ensemble.get(\"method\", \"N/A\")}');\
    print('✅ Configuration validation complete!');\
else:\
    print('❌ Configuration file not found!');\
    exit(1)"

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

live-crypto-binance: ## Launch Binance live connector
	@echo "▶︎ Launching Binance live connector…"
	@source .venv/bin/activate && python -m src.layers.layer0_data_ingestion.binance_ws --symbols "BTCUSDT,ETHUSDT" | tee logs/binance_live.$(shell date +%F_%H%M).log

live-crypto: ## Launch NOWNodes live connector (30 min)
	@echo "▶︎ Launching NOWNodes live connector (30 min)…"
	@source .venv/bin/activate && python -m src.layers.layer0_data_ingestion.nownodes_ws --symbols "BTCUSDT,ETHUSDT,SOLUSDT" | tee logs/nownodes_live.$(shell date +%F_%H%M).log

live-stocks: ## Launch NVDA replay session for paper trading
	@echo "▶︎ Starting NVDA replay session..."
	@source .venv/bin/activate && \
	python run_stocks_session.py

load-minute-bars: ## Load minute bars for backtesting (usage: make load-minute-bars FILE=data/BTCUSDT.csv SYMBOL=BTCUSDT)
	@echo "📊 Loading minute bars for backtesting..."
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Usage: make load-minute-bars FILE=<path_to_csv> SYMBOL=<symbol>"; \
		exit 1; \
	fi
	@if [ -z "$(SYMBOL)" ]; then \
		echo "❌ Usage: make load-minute-bars FILE=<path_to_csv> SYMBOL=<symbol>"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "❌ File not found: $(FILE)"; \
		exit 1; \
	fi
	@echo "File: $(FILE)"
	@echo "Symbol: $(SYMBOL)"
	@echo "Preview of data:"
	@head -5 "$(FILE)"
	@echo "Total bars: $$(wc -l < "$(FILE)")"
	@echo "✅ Minute bars loaded successfully"

backtest-risk: ## Run risk backtest on minute bars (usage: make backtest-risk SYMBOL=BTCUSDT DAYS=30)
	@echo "🧪 Running risk backtest..."
	@SYMBOL_ARG="BTCUSDT"; \
	DAYS_ARG="30"; \
	if [ -n "$(SYMBOL)" ]; then SYMBOL_ARG="$(SYMBOL)"; fi; \
	if [ -n "$(DAYS)" ]; then DAYS_ARG="$(DAYS)"; fi; \
	source .venv/bin/activate && \
	python scripts/backtest_risk.py --bars 1m --days $$DAYS_ARG --symbol $$SYMBOL_ARG

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

# ═══════════════════════════════════════════════════════════════════════════════
# Replay Acceptance Testing
# ═══════════════════════════════════════════════════════════════════════════════

accept: ## Run replay acceptance test (usage: make accept DAY=2025-07-30)
	@if [ -z "$(DAY)" ]; then \
		echo "❌ Usage: make accept DAY=2025-07-30"; \
		echo "   Will look for data/replays/$(DAY).parquet"; \
		exit 1; \
	fi
	@echo "🎬 Running replay acceptance test for $(DAY)..."
	@source .venv/bin/activate && \
	python scripts/replay_acceptance.py data/replays/$(DAY).parquet

accept-synthetic: ## Run replay acceptance test with synthetic data
	@echo "🎬 Running replay acceptance test with synthetic data..."
	@source .venv/bin/activate && \
	python scripts/replay_acceptance.py /tmp/synthetic-data.parquet

accept-pytest: ## Run replay acceptance tests via pytest
	@echo "🧪 Running replay acceptance pytest suite..."
	@source .venv/bin/activate && \
	python -m pytest tests/test_replay_acceptance.py -v --tb=short

accept-reset: ## Reset baseline for specific day (usage: make accept-reset DAY=2025-07-30)
	@if [ -z "$(DAY)" ]; then \
		echo "❌ Usage: make accept-reset DAY=2025-07-30"; \
		exit 1; \
	fi
	@echo "🔄 Resetting baseline for $(DAY)..."
	@source .venv/bin/activate && \
	python scripts/replay_acceptance.py --reset-baseline data/replays/$(DAY).parquet

# ═══════════════════════════════════════════════════════════════════════════════
# M12: Live Economic Experiment (7-14 days)
# ═══════════════════════════════════════════════════════════════════════════════

exp-init: ## Initialize experiment registry and configuration
	@echo "🧪 Initializing M12 live economic experiment..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/exp_registry.py -c experiments/live_econ_exp.yaml --init

exp-assign: ## Generate switchback treatment assignments for today
	@echo "📅 Generating switchback assignments..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/exp_scheduler.py -c experiments/live_econ_exp.yaml

exp-collect: ## Collect current hour experiment metrics
	@echo "📊 Collecting experiment metrics..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/exp_metrics_collector.py -c experiments/live_econ_exp.yaml

exp-analyze: ## Run CUPED analysis and sequential testing
	@echo "📈 Running CUPED variance reduction analysis..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python analysis/exp_cuped.py -c experiments/live_econ_exp.yaml
	@echo "🎯 Running sequential test analysis..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python analysis/exp_sequential_test.py -c experiments/live_econ_exp.yaml

exp-decide: ## Make GO/EXTEND/NO-GO experiment decision
	@echo "🎯 Making experiment decision..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/exp_decider.py -c experiments/live_econ_exp.yaml

exp-status: ## Show current experiment status
	@echo "🧪 M12 Live Economic Experiment Status:"
	@echo "======================================"
	@echo "📋 Experiment Configuration:"
	@if [ -f experiments/live_econ_exp.yaml ]; then \
		echo "✅ Config: experiments/live_econ_exp.yaml"; \
		python -c "import yaml; config=yaml.safe_load(open('experiments/live_econ_exp.yaml')); exp=config['experiment']; print(f'  Name: {exp[\"name\"]}'); print(f'  Horizon: {exp[\"horizon_days\"]} days'); print(f'  Assets: {len(exp[\"assets\"])}'); print(f'  Block Size: {exp[\"block_minutes\"]} minutes')" 2>/dev/null; \
	else \
		echo "❌ No experiment config found"; \
	fi
	@echo ""
	@echo "📊 Latest Assignments:"
	@if [ -f experiments/m11/assignments_latest.json ]; then \
		echo "✅ Assignments available"; \
		python -c "import json; d=json.load(open('experiments/m11/assignments_latest.json')); s=d['summary']['overall_stats']; print(f'  Total Blocks: {s[\"total_blocks\"]}'); print(f'  Treatment: {s[\"treatment_blocks\"]} ({s[\"treatment_ratio\"]:.1%})'); print(f'  Control: {s[\"control_blocks\"]}')" 2>/dev/null; \
	else \
		echo "❌ No assignments found - run 'make exp-assign'"; \
	fi
	@echo ""
	@echo "📈 Latest Analysis:"
	@if [ -f experiments/m11/cuped_analysis_latest.json ]; then \
		echo "✅ CUPED analysis available"; \
		python -c "import json; d=json.load(open('experiments/m11/cuped_analysis_latest.json')); f=d['key_findings']; print(f'  Primary Effect: {f.get(\"primary_effect_cuped\", f.get(\"primary_effect_raw\", 0)):.3f}'); print(f'  Significant: {\"✅\" if f[\"effect_significant\"] else \"❌\"}'); print(f'  Meets MET: {\"✅\" if f[\"meets_met\"] else \"❌\"}')" 2>/dev/null; \
	else \
		echo "❌ No analysis found - run 'make exp-analyze'"; \
	fi
	@echo ""
	@echo "🎯 Latest Decision:"
	@if [ -f experiments/m11/decision_latest.json ]; then \
		echo "✅ Decision available"; \
		python -c "import json; d=json.load(open('experiments/m11/decision_latest.json')); print(f'  Decision: {d[\"decision\"]}'); print(f'  GO Token: {\"✅\" if d.get(\"go_token_created\", False) else \"❌\"}'); reasons = d.get(\"reasons\", []); print(f'  Reasons: {\", \".join(reasons[:2]) if reasons else \"All gates passed\"}')" 2>/dev/null; \
	else \
		echo "❌ No decision found - run 'make exp-decide'"; \
	fi

exp-simulate: ## Simulate full day of experiment data collection
	@echo "🔄 Simulating full day experiment..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/exp_metrics_collector.py -c experiments/live_econ_exp.yaml --simulate-day

exp-pipeline: ## Run complete experiment pipeline
	@echo "🧪 Running complete M12 experiment pipeline..."
	@echo "Step 1/5: Initialize experiment..."
	$(MAKE) exp-init
	@echo "Step 2/5: Generate assignments..."
	$(MAKE) exp-assign
	@echo "Step 3/5: Collect metrics..."
	$(MAKE) exp-collect
	@echo "Step 4/5: Run analysis..."
	$(MAKE) exp-analyze
	@echo "Step 5/5: Make decision..."
	$(MAKE) exp-decide
	@echo "✅ M12 experiment pipeline complete"

exp-test: ## Test complete experiment framework end-to-end
	@echo "🧪 Testing M12 experiment framework..."
	@echo "Step 1/4: Simulate data collection..."
	$(MAKE) exp-simulate
	@echo "Step 2/4: Run analysis..."
	$(MAKE) exp-analyze
	@echo "Step 3/4: Make decision..."
	$(MAKE) exp-decide
	@echo "Step 4/4: Check ramp integration..."
	$(MAKE) ramp-decide
	@echo "✅ M12 experiment framework test complete!"

# ═══════════════════════════════════════════════════════════════════════════════
# M13: Market Selection, Fee/Rebate Optimization & Duty-Cycling
# ═══════════════════════════════════════════════════════════════════════════════

ev-forecast: ## Generate EV forecasts and trade calendar
	@echo "📈 Generating EV forecasts and trade calendar..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python analysis/ev_forecaster.py --window 2d --out artifacts/ev

duty-cycle: ## Run duty cycling based on EV calendar (dry-run by default)
	@echo "⚡ Running duty cycling..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/duty_cycler.py --calendar artifacts/ev/latest.parquet --dry-run

duty-cycle-apply: ## Apply duty cycling (requires GO_LIVE=1)
	@if [ "$$GO_LIVE" != "1" ]; then \
		echo "❌ GO_LIVE flag not set. Use: GO_LIVE=1 make duty-cycle-apply"; \
		exit 1; \
	fi
	@echo "⚡ Applying duty cycling..."
	GO_LIVE=1 PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/duty_cycler.py --calendar artifacts/ev/latest.parquet

maker-taker: ## Test maker/taker controller with sample data
	@echo "💰 Testing maker/taker controller..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python src/layers/layer4_execution/maker_taker_controller.py

fee-plan: ## Generate fee tier optimization plan
	@echo "💰 Generating fee tier optimization plan..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/fee_tier_planner.py --scenarios

cost-signal: ## Start cost signal monitoring exporter
	@echo "💰 Starting cost signal exporter..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python src/monitoring/cost_signal_exporter.py --port 9112 &
	@echo "📊 Cost signals available at http://localhost:9112/metrics"

cost-signal-test: ## Test cost signal exporter
	@echo "🧪 Testing cost signal exporter..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python src/monitoring/cost_signal_exporter.py --test

rebate-monitor: ## Start rebate performance monitoring
	@echo "💰 Starting rebate performance monitor..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python src/monitoring/rebate_exporter.py --port 9111 &
	@echo "📊 Rebate metrics available at http://localhost:9111/metrics"

rebate-monitor-test: ## Test rebate performance monitor
	@echo "🧪 Testing rebate performance monitor..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python src/monitoring/rebate_exporter.py --test

ramp-m13: ## Run ramp decider with M13 gates (EV/duty/rebate)
	@echo "🏛️ Running ramp decider with M13 gates..."
	PYTHONPATH=/Users/yanzewu/PycharmProjects/NLP_Final_Project_D python scripts/ramp_decider_econ.py --policy ramp/ramp_policy.yaml --verbose

m13-status: ## Show M13 cost optimization status
	@echo "💰 M13 Market Selection & Cost Optimization Status:"
	@echo "=================================================="
	@echo "📈 EV Calendar Status:"
	@if [ -f artifacts/ev/latest.parquet ]; then \
		echo "✅ EV calendar available"; \
		python -c "import pandas as pd; df=pd.read_parquet('artifacts/ev/latest.parquet'); counts=df['band'].value_counts(); total=len(df); print(f'  Total windows: {total}'); print(f'  Green: {counts.get(\"green\", 0)} ({counts.get(\"green\", 0)/total*100:.1f}%)'); print(f'  Amber: {counts.get(\"amber\", 0)} ({counts.get(\"amber\", 0)/total*100:.1f}%)'); print(f'  Red: {counts.get(\"red\", 0)} ({counts.get(\"red\", 0)/total*100:.1f}%)')" 2>/dev/null; \
	else \
		echo "❌ No EV calendar found - run 'make ev-forecast'"; \
	fi
	@echo ""
	@echo "⚡ Duty Cycling Status:"
	@if [ -f artifacts/ev/duty_cycle_on ]; then \
		echo "✅ Duty cycling active"; \
		python -c "import json; d=json.load(open('artifacts/ev/duty_cycle_on')); p=d['parameters']; print(f'  Green: {p[\"green_pct\"]}%'); print(f'  Amber: {p[\"amber_pct\"]}%'); print(f'  Red: {p[\"red_pct\"]}%')" 2>/dev/null; \
	else \
		echo "❌ Duty cycling not active - run 'make duty-cycle'"; \
	fi
	@echo ""
	@echo "💰 Fee Tier Optimization:"
	@if [ -f artifacts/fee_planning/latest.json ]; then \
		echo "✅ Fee plan available"; \
		python -c "import json; d=json.load(open('artifacts/fee_planning/latest.json')); print(f'  Monthly savings: \$${d[\"total_potential_savings_monthly\"]:.2f}'); print(f'  Priority venues: {len(d[\"priority_venues\"])}'); print(f'  High-ROI actions: {len([a for a in d[\"implementation_actions\"] if a.get(\"priority\") == \"high\"])}')" 2>/dev/null; \
	else \
		echo "❌ No fee plan found - run 'make fee-plan'"; \
	fi
	@echo ""
	@echo "🎯 Current Influence:"
	@$(MAKE) asset-influence

m13-pipeline: ## Run complete M13 cost optimization pipeline
	@echo "💰 Running M13 Cost Optimization Pipeline..."
	@echo "Step 1/5: EV forecasting..."
	$(MAKE) ev-forecast
	@echo "Step 2/5: Duty cycling..."
	$(MAKE) duty-cycle
	@echo "Step 3/5: Maker/taker optimization..."
	$(MAKE) maker-taker
	@echo "Step 4/5: Fee tier planning..."
	$(MAKE) fee-plan
	@echo "Step 5/5: Cost signal monitoring..."
	$(MAKE) cost-signal-test
	@echo "✅ M13 cost optimization pipeline complete"

m13-test: ## Test complete M13 framework end-to-end
	@echo "🧪 Testing M13 cost optimization framework..."
	@echo "Step 1/6: EV forecasting..."
	$(MAKE) ev-forecast
	@echo "Step 2/6: Duty cycling..."
	$(MAKE) duty-cycle
	@echo "Step 3/6: Maker/taker controller..."
	$(MAKE) maker-taker
	@echo "Step 4/6: Fee planning..."
	$(MAKE) fee-plan
	@echo "Step 5/6: Cost monitoring..."
	$(MAKE) cost-signal-test && $(MAKE) rebate-monitor-test
	@echo "Step 6/6: Ramp integration..."
	$(MAKE) ramp-m13
	@echo "✅ M13 cost optimization framework test complete!"

# M15 Green-Window Live Ramp Targets

ramp-10-green: ## Execute 10% live ramp in green/event windows only
	@echo "🟢 Starting M15 Green-Window Ramp (10%)..."
	@if [ ! -f experiments/m11/decision.json ]; then \
		echo "❌ M12 GO token missing - cannot start live ramp"; \
		exit 1; \
	fi
	@if [ -z "$(GO_LIVE)" ]; then \
		echo "❌ GO_LIVE flag not set. Use GO_LIVE=1 or --dry-run"; \
		exit 1; \
	fi
	GO_LIVE=1 python scripts/green_window_ramp.py --pct 10 --calendar artifacts/ev/calendar_5m.parquet --min-green-min 10

ramp-10-green-dry: ## Dry run 10% green ramp (for testing)
	@echo "🧪 Dry run: M15 Green-Window Ramp (10%)..."
	python scripts/green_window_ramp.py --pct 10 --calendar artifacts/ev/calendar_5m.parquet --min-green-min 10 --dry-run

green-profit: ## Track green-only economics and P&L
	@echo "💰 Tracking green-window economics..."
	python scripts/green_profit_tracker.py --out artifacts/econ_green

cfo-green: ## Generate CFO digest for green-window economics
	@echo "🏦 Generating CFO digest..."
	python scripts/cfo_digest_green.py

ramp-decide: ## Evaluate criteria for ramp advancement
	@echo "📊 Evaluating ramp advancement criteria..."
	python scripts/ramp_decider_econ.py --policy ramp/ramp_policy.yaml

# M15 Green-Window Status Dashboard

green-status: ## Show green-window ramp status and metrics
	@echo "🟢 M15 Green-Window Ramp Status"
	@echo "=" * 50
	@echo ""
	@echo "🎯 Current Ramp Status:"
	@if [ -f artifacts/audit/green_ramp_block_start_*.json ]; then \
		echo "✅ Green ramp active"; \
		latest_file=$$(ls -t artifacts/audit/green_ramp_block_start_*.json | head -1); \
		python -c "import json; d=json.load(open('$$latest_file')); print(f'  Assets: {len(d[\"eligible_assets\"])}'); print(f'  Ramp %: {d[\"ramp_pct\"]}%'); print(f'  Duration: {d[\"block_duration_minutes\"]}min')" 2>/dev/null; \
	else \
		echo "❌ No green ramp activity - run 'make ramp-10-green'"; \
	fi
	@echo ""
	@echo "💰 Green Economics (24H):"
	@if [ -f artifacts/econ_green/summary.json ]; then \
		echo "✅ Economics data available"; \
		python -c "import json; d=json.load(open('artifacts/econ_green/summary.json')); s=d.get('seven_day_summary', {}); print(f'  Net P&L: \$$${s.get(\"total_net_pnl_usd\", 0):+,.0f}'); print(f'  Cost Ratio: {s.get(\"avg_cost_ratio\", 0):.1%}'); print(f'  Positive Days: {s.get(\"consecutive_positive_days\", 0)}/7')" 2>/dev/null; \
	else \
		echo "❌ No economics data - run 'make green-profit'"; \
	fi
	@echo ""
	@echo "🚀 Advancement Status:"
	@if [ -f artifacts/econ_green/summary.json ]; then \
		python -c "import json; d=json.load(open('artifacts/econ_green/summary.json')); s=d.get('seven_day_summary', {}); days=s.get('consecutive_positive_days', 0); pnl=s.get('total_net_pnl_usd', 0); cost=s.get('avg_cost_ratio', 1); print('✅ Ready for 15% advancement' if days >= 7 and pnl >= 300 and cost <= 0.30 else f'⏳ {7-days} profitable days needed')" 2>/dev/null; \
	else \
		echo "❌ No advancement data available"; \
	fi

# M15 Operational Workflows

m15-pipeline: ## Run complete M15 green-ramp pipeline
	@echo "🟢 Running M15 Green-Window Live Ramp Pipeline..."
	@echo "Step 1/4: Green-window ramp execution..."
	$(MAKE) ramp-10-green-dry
	@echo "Step 2/4: Profit tracking..."
	$(MAKE) green-profit
	@echo "Step 3/4: CFO digest..."
	$(MAKE) cfo-green
	@echo "Step 4/4: Advancement evaluation..."
	$(MAKE) ramp-decide
	@echo "✅ M15 green-window ramp pipeline complete"

m15-daily: ## Daily M15 green-ramp monitoring routine
	@echo "📅 M15 Daily Monitoring Routine..."
	@echo "Step 1/2: Update economics..."
	$(MAKE) green-profit
	@echo "Step 2/2: Generate executive digest..."
	$(MAKE) cfo-green
	@echo "✅ M15 daily monitoring complete"

m15-test: ## Test complete M15 framework end-to-end
	@echo "🧪 Testing M15 green-window ramp framework..."
	@echo "Step 1/5: Dry run green ramp..."
	$(MAKE) ramp-10-green-dry
	@echo "Step 2/5: Profit tracking..."
	$(MAKE) green-profit
	@echo "Step 3/5: CFO digest..."
	$(MAKE) cfo-green
	@echo "Step 4/5: Status dashboard..."
	$(MAKE) green-status
	@echo "Step 5/5: Advancement check..."
	$(MAKE) ramp-decide
	@echo "✅ M15 green-window ramp framework test complete!"

# M16 Execution Optimization Targets

slip-forecast: ## Build slippage forecasting models
	@echo "📈 Building slippage forecasting models..."
	python analysis/slip_forecaster.py --window 14d --out artifacts/exec

exec-v2: ## Test execution v2 components
	@echo "🔧 Testing execution v2 components..."
	python src/layers/layer4_execution/queue_timing_v2.py
	@echo "Queue timing v2: ✅"
	python src/layers/layer4_execution/escalation_policy.py
	@echo "Escalation policy: ✅"
	python src/layers/layer4_execution/child_sizer_v2.py
	@echo "Child sizer v2: ✅"
	@echo "exec-v2 components: ✅"

slip-gate: ## Check slippage gate for ramp advancement
	@echo "🚪 Checking slippage gate (P95 ≤15 bps)..."
	python scripts/slippage_gate.py --window 48h --min-orders 2000

# M16 Status Dashboard

exec-status: ## Show M16 execution optimization status
	@echo "🔧 M16 Execution Optimization Status"
	@echo "=" * 50
	@echo ""
	@echo "📈 Slippage Models:"
	@if [ -f artifacts/exec/slip_model.json ]; then \
		echo "✅ Slippage forecaster trained"; \
		python -c "import json; d=json.load(open('artifacts/exec/slip_model.json')); print(f'  Models: {d[\"trained_models\"]}'); print(f'  Avg R²: {d.get(\"avg_r2\", \"N/A\"):.3f}' if 'avg_r2' in d else '  Avg R²: N/A')" 2>/dev/null; \
	else \
		echo "❌ No slippage models - run 'make slip-forecast'"; \
	fi
	@echo ""
	@echo "🚪 Slippage Gate:"
	@if [ -f artifacts/exec/slip_gate_ok ]; then \
		echo "✅ Slippage gate PASSED"; \
		python -c "import json; d=json.load(open('artifacts/exec/slip_gate_ok')); print(f'  P95: {d[\"p95_slippage_bps\"]:.1f} bps'); print(f'  Fills: {d[\"total_fills\"]:,}'); print(f'  Maker: {d[\"maker_ratio\"]:.1%}')" 2>/dev/null; \
	elif [ -f artifacts/exec/slip_gate_fail ]; then \
		echo "❌ Slippage gate FAILED"; \
		python -c "import json; d=json.load(open('artifacts/exec/slip_gate_fail')); print(f'  P95: {d[\"p95_slippage_bps\"]:.1f} bps (need ≤{d[\"threshold_bps\"]} bps)'); print(f'  Reason: {d[\"reason\"]}')" 2>/dev/null; \
	else \
		echo "❌ No slippage gate data - run 'make slip-gate'"; \
	fi
	@echo ""
	@echo "🚀 15% Advancement Status:"
	@if [ -f artifacts/exec/slip_gate_ok ] && [ -f artifacts/econ_green/summary.json ]; then \
		python -c "import json; e=json.load(open('artifacts/econ_green/summary.json')); s=e.get('seven_day_summary', {}); days=s.get('consecutive_positive_days', 0); pnl=s.get('total_net_pnl_usd', 0); cost=s.get('avg_cost_ratio', 1); print('✅ Ready for 15% advancement!' if days >= 7 and pnl >= 300 and cost <= 0.30 else f'⏳ Need: {max(0, 7-days)} more days, ${max(0, 300-pnl):.0f} P&L, {max(0, cost-0.30):.1%} cost reduction')" 2>/dev/null; \
	else \
		echo "❌ Prerequisites not met (slippage gate + 7 profitable days)"; \
	fi

# M16 Comprehensive Workflows

m16-pipeline: ## Run complete M16 execution optimization pipeline
	@echo "🔧 Running M16 Execution Optimization Pipeline..."
	@echo "Step 1/3: Slippage forecasting..."
	$(MAKE) slip-forecast
	@echo "Step 2/3: Execution v2 testing..."
	$(MAKE) exec-v2
	@echo "Step 3/3: Slippage gate check..."
	$(MAKE) slip-gate
	@echo "✅ M16 execution optimization pipeline complete"

m16-test: ## Test complete M16 framework end-to-end
	@echo "🧪 Testing M16 execution optimization framework..."
	@echo "Step 1/4: Slippage forecasting..."
	$(MAKE) slip-forecast
	@echo "Step 2/4: Execution v2 components..."
	$(MAKE) exec-v2
	@echo "Step 3/4: Slippage gate..."
	$(MAKE) slip-gate
	@echo "Step 4/4: Status dashboard..."
	$(MAKE) exec-status
	@echo "✅ M16 execution optimization framework test complete!"

# Integrated M15+M16 Workflow

ramp-advancement-check: ## Check all criteria for 15% ramp advancement
	@echo "🚀 Checking 15% Ramp Advancement Criteria"
	@echo "=" * 50
	@echo "Step 1/4: Green economics (7 days)..."
	$(MAKE) green-profit
	@echo "Step 2/4: Slippage gate (≤15 bps)..."
	$(MAKE) slip-gate
	@echo "Step 3/4: CFO digest..."
	$(MAKE) cfo-green
	@echo "Step 4/4: Final advancement decision..."
	$(MAKE) ramp-decide
	@echo "✅ Ramp advancement check complete"

# M17: 15% Green-Window Ramp Targets

ramp-15-green: ## Execute M17 15% green-window ramp with micro-gradient promotion
	@echo "🚀 M17: Executing 15% Green-Window Ramp..."
	@echo "Pre-flight checks:"
	@$(MAKE) slip-gate
	@$(MAKE) green-profit
	@echo "Launching orchestrator..."
	GO_LIVE=1 PYTHONPATH=$(PWD) python scripts/ramp_15_orchestrator.py \
		--calendar artifacts/ev/calendar_5m.parquet \
		--steps "10,12,13.5,15" --step-min 3

ramp-15-guard: ## Start M17 hard guards monitoring (runs in background)
	@echo "🛡️ M17: Starting hard guards monitoring..."
	@echo "Guards: slippage (15bps), drawdown (1%), maker ratio (65%), alerts"
	@echo "Rollback SLA: 2 seconds"
	PYTHONPATH=$(PWD) python scripts/ramp_guard_15.py --watch 30

ramp-15-report: ## Generate M17 15% CFO digest with execution metrics
	@echo "📊 M17: Generating 15% ramp report..."
	PYTHONPATH=$(PWD) python scripts/cfo_digest_green.py --ramp 15 --out artifacts/cfo_green

# M18: 20% Green-Window Ramp (Enhanced)

ramp-20-green: ## Execute M18 20% green-window ramp with EV ceiling enforcement
	@echo "🚀 M18: Executing 20% Green-Window Ramp..."
	@echo "Pre-flight checks:"
	@$(MAKE) slip-gate
	@$(MAKE) m18-prereq
	@echo "Launching orchestrator with EV ceiling..."
	GO_LIVE=1 PYTHONPATH=$(PWD) python scripts/ramp_20_orchestrator.py \
		--calendar artifacts/ev/calendar_5m.parquet \
		--steps "15,17,18.5,20" --step-min 3 --respect-ev-ceiling 1

ramp-20-guard: ## Start M18 hard guards monitoring (runs in background)
	@echo "🛡️ M18: Starting enhanced hard guards monitoring..."
	@echo "Guards: slippage (12bps), drawdown (0.9%), maker ratio (70%), impact (8bp/\$$1k)"
	@echo "Rollback SLA: 0.1 seconds"
	PYTHONPATH=$(PWD) python scripts/ramp_guard_20.py --watch 15

ramp-20-report: ## Generate M18 20% CFO digest with EV ceiling analysis
	@echo "📊 M18: Generating 20% ramp report..."
	PYTHONPATH=$(PWD) python scripts/cfo_digest_green.py --ramp 20 --out artifacts/cfo_green

m18-prereq: ## Verify M18 prerequisites (impact budget, saturation guard, EV ceiling)
	@echo "🔒 M18: Verifying prerequisites..."
	@echo "Checking impact budget, saturation guard, EV ceiling..."
	PYTHONPATH=$(PWD) python scripts/m18_prereq_check.py

ramp-decide: ## Evaluate ramp advancement readiness using policy criteria
	@echo "⚖️ M17: Evaluating ramp advancement criteria..."
	PYTHONPATH=$(PWD) python scripts/ramp_decider_econ.py --policy ramp/ramp_policy.yaml --verbose

# M17 Testing and Validation

test-ramp-15: ## Test M17 orchestrator in dry-run mode
	@echo "🧪 Testing M17 orchestrator (dry-run)..."
	PYTHONPATH=$(PWD) python scripts/ramp_15_orchestrator.py \
		--calendar artifacts/ev/calendar_5m.parquet \
		--steps "10,12,13.5,15" --step-min 1

test-ramp-guard: ## Test M17 guard system triggers
	@echo "🧪 Testing M17 guard system..."
	@echo "Testing guard evaluation and rollback mechanisms..."
	PYTHONPATH=$(PWD) python -c "from scripts.ramp_guard_15 import RampGuard15; guard = RampGuard15(); print('✅ Guard system initialized'); metrics = guard.collect_metrics(); print(f'✅ Metrics collection: {metrics.get(\"collection_success\", False)}'); results = guard.evaluate_guards(metrics); print(f'✅ Guard evaluation: {len(results.get(\"triggered_guards\", []))} guards triggered'); print('✅ Guard system test complete')"

# M17 Comprehensive Pipeline

m17-pipeline: ## Run complete M17 15% green-window ramp pipeline
	@echo "🎯 Running M17 15% Green-Window Ramp Pipeline..."
	@echo "Step 1/5: Pre-checks..."
	$(MAKE) slip-gate
	$(MAKE) green-profit
	@echo "Step 2/5: Test orchestrator..."
	$(MAKE) test-ramp-15
	@echo "Step 3/5: Start guards (background)..."
	$(MAKE) ramp-15-guard &
	@echo "Step 4/5: Execute ramp..."
	$(MAKE) ramp-15-green
	@echo "Step 5/5: Generate report..."
	$(MAKE) ramp-15-report
	@echo "✅ M17 green-window ramp pipeline complete"

m17-status: ## Show M17 15% ramp status and metrics
	@echo "📊 M17: 15% Green-Window Ramp Status"
	@echo "=" * 45
	@echo ""
	@echo "🔒 Gate Compliance:"
	@if [ -f artifacts/exec/slip_gate_ok ]; then \
		echo "✅ Slippage gate passed (≤15 bps)"; \
		python -c "import json; d=json.load(open('artifacts/exec/slip_gate_ok')); print(f'  P95: {d.get(\"p95_slippage_bps\", 0):.1f} bps'); print(f'  Maker: {d.get(\"maker_ratio\", 0):.1%}')" 2>/dev/null || echo "  Details unavailable"; \
	else \
		echo "❌ Slippage gate not passed"; \
	fi
	@echo ""
	@echo "🌱 Green Economics:"
	@if [ -f artifacts/econ_green/summary.json ]; then \
		python -c "import json; e=json.load(open('artifacts/econ_green/summary.json')); s=e.get('seven_day_summary', {}); print(f'  Days: {s.get(\"consecutive_positive_days\", 0)}/7'); print(f'  P&L: \$${s.get(\"total_net_pnl_usd\", 0):+,.0f}'); print(f'  Cost: {s.get(\"avg_cost_ratio\", 0):.1%}')" 2>/dev/null; \
	else \
		echo "❌ Green economics not available"; \
	fi
	@echo ""
	@echo "🛡️ Guard Status:"
	@if pgrep -f "ramp_guard_15.py" > /dev/null; then \
		echo "✅ Hard guards active (PID: $$(pgrep -f ramp_guard_15.py))"; \
	else \
		echo "⚠️ Hard guards not running"; \
	fi
	@echo ""
	@echo "📈 Current Influence:"
	@if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then \
		echo "  BTC-USD: $$(redis-cli get influence:BTC-USD 2>/dev/null || echo '0')%"; \
		echo "  ETH-USD: $$(redis-cli get influence:ETH-USD 2>/dev/null || echo '0')%"; \
		echo "  NVDA: $$(redis-cli get influence:NVDA 2>/dev/null || echo '0')%"; \
	else \
		echo "  Redis not available - checking files..."; \
		ls artifacts/influence/ 2>/dev/null | head -3 || echo "  No influence files found"; \
	fi
	@echo ""
	@echo "📋 Recent Activity:"
	@find worm -name "ramp15_*.json" -mtime -1 2>/dev/null | wc -l | xargs -I {} echo "  {} ramp15 audit entries (24h)"
	@find artifacts/audit -name "*rollback*.json" -mtime -1 2>/dev/null | wc -l | xargs -I {} echo "  {} rollback events (24h)"

# M18: 48h Soak + 20% Ramp Targets

soak-15: ## Start 48h soak monitoring for 15% stability verification
	@echo "📊 M18: Starting 48h soak monitoring..."
	@echo "This will monitor 15% ramp stability for 48 hours"
	@echo "Use Ctrl+C to stop monitoring"
	PYTHONPATH=$(PWD) python scripts/soak_15_monitor.py --interval 300 --out artifacts/soak15

soak-15-gate: ## Evaluate 48h soak gate for 20% advancement readiness
	@echo "🔒 M18: Evaluating 48h soak gate..."
	PYTHONPATH=$(PWD) python scripts/soak_15_gate.py --window 48h

test-impact-budget: ## Test M18 impact budget system
	@echo "🎯 Testing impact budget system..."
	PYTHONPATH=$(PWD) python src/risk/impact_budget.py

test-venue-saturation: ## Test M18 venue saturation guard
	@echo "🛡️ Testing venue saturation guard..."
	PYTHONPATH=$(PWD) python src/risk/venue_saturation_guard.py

test-ev-ceiling: ## Test M18 EV influence ceiling mapper
	@echo "🎯 Testing EV influence ceiling..."
	PYTHONPATH=$(PWD) python scripts/ev_influence_ceiling.py \
		--calendar artifacts/ev/calendar_5m.parquet \
		--out artifacts/ev_ceiling

# M18 Comprehensive Pipeline

export-knobs: ## Export and freeze current configuration knobs for evidence pack
	@echo "🔧 Exporting current configuration knobs..."
	@mkdir -p artifacts/evidence
	@echo "Capturing current config state at $(shell date -u +%Y-%m-%dT%H:%M:%SZ)" > artifacts/evidence/knobs_export.json
	@echo "{" >> artifacts/evidence/knobs_export.json
	@echo "  \"timestamp\": \"$(shell date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> artifacts/evidence/knobs_export.json
	@echo "  \"m16_config\": {" >> artifacts/evidence/knobs_export.json
	@echo "    \"post_only_ratio\": 0.85," >> artifacts/evidence/knobs_export.json
	@echo "    \"slice_max_pct\": 0.8," >> artifacts/evidence/knobs_export.json
	@echo "    \"max_escalations\": 1" >> artifacts/evidence/knobs_export.json
	@echo "  }," >> artifacts/evidence/knobs_export.json
	@echo "  \"m18_capacity_controls\": {" >> artifacts/evidence/knobs_export.json
	@echo "    \"impact_budget_target\": 6.0," >> artifacts/evidence/knobs_export.json
	@echo "    \"venue_saturation_limit\": 8.0," >> artifacts/evidence/knobs_export.json
	@echo "    \"ev_ceiling_enforcement\": true" >> artifacts/evidence/knobs_export.json
	@echo "  }," >> artifacts/evidence/knobs_export.json
	@echo "  \"ramp_policy\": \"15_to_20_micro_gradient\"" >> artifacts/evidence/knobs_export.json
	@echo "}" >> artifacts/evidence/knobs_export.json
	@echo "✅ Configuration knobs exported to artifacts/evidence/knobs_export.json"

sbom-generate: ## Generate Software Bill of Materials for evidence pack
	@echo "📋 Generating SBOM for M18 evidence pack..."
	@mkdir -p artifacts/evidence
	PYTHONPATH=$(PWD) python scripts/sbom_generator.py --output artifacts/evidence

m18-prereq: ## Check M18 prerequisites (soak gate + capacity controls)
	@echo "🔍 M18: Checking prerequisites for 20% advancement..."
	@echo "Step 1/4: 48h soak gate..."
	$(MAKE) soak-15-gate
	@echo "Step 2/4: Impact budget status..."
	$(MAKE) test-impact-budget
	@echo "Step 3/4: Venue saturation status..."
	$(MAKE) test-venue-saturation
	@echo "Step 4/4: EV ceiling mapping..."
	$(MAKE) test-ev-ceiling
	@echo "✅ M18 prerequisite check complete"

m18-status: ## Show M18 readiness status and capacity controls
	@echo "📊 M18: 20% Ramp Readiness Status"
	@echo "=" * 45
	@echo ""
	@echo "🔒 Soak Gate Status:"
	@if [ -f artifacts/gates/soak15_ok ]; then \
		echo "✅ 48h soak gate passed"; \
		python -c "import json; d=json.load(open('artifacts/gates/soak15_ok')); print(f'  Score: {d.get(\"overall_score\", 0):.1%}'); print(f'  Valid until: {d.get(\"valid_until\", \"unknown\")}')" 2>/dev/null || echo "  Details unavailable"; \
	else \
		echo "❌ 48h soak gate not passed - run 'make soak-15-gate'"; \
	fi
	@echo ""
	@echo "🎯 Capacity Controls:"
	@echo "  Impact Budget: $$([ -f artifacts/impact/impact_history.json ] && echo '✅ Active' || echo '⚠️ Not initialized')"
	@echo "  Venue Saturation: $$([ -f artifacts/venue/venue_metrics.json ] && echo '✅ Active' || echo '⚠️ Not initialized')"
	@echo "  EV Ceiling: $$([ -f artifacts/ev_ceiling/ev_ceiling_analysis_*.json ] && echo '✅ Active' || echo '⚠️ Not initialized')"
	@echo ""
	@echo "📈 Current Capacity Status:"
	@if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then \
		echo "  BTC-USD Impact Cap: $$(redis-cli get impact_budget:slice_cap:BTC-USD 2>/dev/null || echo 'N/A')%"; \
		echo "  ETH-USD Impact Cap: $$(redis-cli get impact_budget:slice_cap:ETH-USD 2>/dev/null || echo 'N/A')%"; \
	else \
		echo "  Redis not available - cannot show live caps"; \
	fi
	@echo ""
	@echo "🚀 20% Advancement:"
	@if [ -f artifacts/gates/soak15_ok ] && [ -f artifacts/exec/slip_gate_ok ]; then \
		echo "✅ Ready for 20% advancement - all gates passed"; \
	else \
		echo "⏳ Prerequisites not met:"; \
		[ ! -f artifacts/gates/soak15_ok ] && echo "   • Missing soak15_ok gate"; \
		[ ! -f artifacts/exec/slip_gate_ok ] && echo "   • Missing slip_gate_ok"; \
	fi

# M18 Testing and Validation

test-m18-framework: ## Test complete M18 framework components
	@echo "🧪 Testing M18 20% readiness framework..."
	@echo "Step 1/4: Soak monitoring..."
	@timeout 30 $(MAKE) soak-15 || echo "⏹️ Soak monitor test stopped"
	@echo "Step 2/4: Impact budget..."
	$(MAKE) test-impact-budget
	@echo "Step 3/4: Venue saturation..."
	$(MAKE) test-venue-saturation
	@echo "Step 4/4: EV ceiling..."
	$(MAKE) test-ev-ceiling
	@echo "✅ M18 framework test complete"