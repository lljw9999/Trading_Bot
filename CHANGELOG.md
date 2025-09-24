# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0-rc1] – 2025-09-23

### Added
- **GitHub Actions test matrix** (`.github/workflows/tests.yml`) with staged jobs:
  - `unit`: `pytest -m "not integration and not soak"`
  - `integration`: spins up `docker-compose.ci.yml` services (Redis, API mock)
  - `soak` (scheduled cron): installs extras via `pip install .[ml,onnx,bandits]` and runs `pytest -m "soak"`
- `docker-compose.ci.yml` to provision CI redis + mock API services.
- `pyproject.toml` with optional dependency groups (`ml`, `onnx`, `bandits`).

### Changed
- **Executor safety**: `AlpacaExecutor` defaults to `dry_run=True` and refuses live mode unless `DRY_RUN=0` is explicitly set.
- **NOWNodes connector**: gated by `USE_NOWNODES=1` and silences legacy `websockets` warnings (pinned to `websockets==11.0.3`).
- **Layer 0 package init** tolerates disabled NOWNodes without import explosions.
- Updated core version constants and README badge to `v0.9.0-rc1`.
- Refreshed RUNBOOK with staged test matrix, environment toggles (`DRY_RUN`, `OPENAI_MOCK`, `REDIS_MOCK`, `USE_NOWNODES`).

### Fixed
- Clean CI logs by suppressing legacy `websockets` deprecation noise when NOWNodes is disabled.
- Extras install job prevents `ImportError` for optional ML/bandit/onnx modules.

---

## [0.4.1] – 2025-09-23

### Added - Stabilization Sprint (Future_instruction.txt Phase 2)

#### 🛡️ Safety & Reliability Improvements
- **Websockets Deprecation Handling**: Added `USE_NOWNODES` environment flag to gate websockets imports
  - Graceful fallback to simulation mode when websockets disabled
  - Pytest markers for NOWNodes tests with conditional skipping  
  - Eliminated deprecation warnings in local/CI runs by default
- **Executor Safety Audit**: Enhanced AlpacaExecutor with explicit prod guards
  - **BREAKING**: Now defaults to `dry_run=True` to prevent accidental real orders
  - Requires explicit `DRY_RUN=0` environment variable for live trading
  - Added warning messages for live trading mode activation

#### 🧪 CI Test Strategy Enhancement  
- Implemented staged pytest matrix as specified in Future_instruction.txt:
  - **Stage 1**: Unit tests only (`pytest -m "not integration"`)
  - **Stage 2**: Full integration suite (Redis, API, connectors)  
  - **Stage 3**: Soak/long-running tests (nightly builds)
- Added timeout tuning and coverage thresholds to `pytest.ini`
- Created `run_ci_tests.sh` script for automated CI execution
- Enhanced pytest configuration with `--maxfail=5`, `--durations=20`, `--tb=short`

#### 📚 Documentation Updates
- Updated **RUNBOOK.md** with comprehensive safety controls section
- Added testing strategy documentation with staged test matrix
- Documented environment toggles (`USE_NOWNODES`, `DRY_RUN`)
- Added coverage testing instructions and pytest markers reference

### Technical Implementation Details

#### New Environment Variables
- `USE_NOWNODES=1`: Enable NOWNodes websocket connections (default: 0)
- `DRY_RUN=0`: Enable live trading (default: 1, requires explicit setting)

#### New Pytest Markers  
- `nownodes`: Tests requiring USE_NOWNODES=1 and websockets library
- Enhanced existing markers for better CI categorization

#### Safety Features
- Prevents accidental real money orders in tests/dev environments
- Clear warnings for live trading mode activation
- Isolated test execution from live trading functionality

### Previous Sprint Achievements (Phase 1)
- ✅ External service hardening with OpenAI/Redis mocks
- ✅ Dependency gating for ML libraries (xgboost/lightgbm/onnx)
- ✅ L2 order book depth with VWAP calculations and Kafka schema
- ✅ Advanced risk management (VaR/CVaR + exchange haircuts + WORM audit)
- ✅ Real prometheus metrics with proper histogram buckets and SLO monitoring

---

## [0.4.0-rc3] – 2025-01-17

### Added
- **Grafana Edge-Risk dashboard** (#145)
  - Single-screen health view for traders and quants
  - 4-row layout: Quick Stats, Time Series, Switch Log, Alert Summary
  - Real-time metrics display with color-coded risk thresholds
  - Responsive design for desktop (1440×) and laptop (1176×) screens
- Redis TimeSeries writer for blended edge, position size, and VaR metrics
  - `edge_blended_bps:<symbol>` - Blended edge in basis points
  - `position_size_usd:<symbol>` - Position size in USD  
  - `var_pct:<symbol>` - Portfolio VaR percentage
  - Fallback support when TimeSeries module unavailable
- Model switch alert rule for excessive switching detection
  - Triggers on >20 switches per 5 minutes
  - Includes runbook URL and escalation procedures
- Grafana import script (`scripts/grafana_import.sh`)
  - Idempotent dashboard and alert imports
  - Authentication support (user/password or API token)
  - Force update option for existing dashboards
- **Risk Harmoniser v1** - Mathematical edge blending and position sizing
  - Edge blending with decay weights and Bayesian shrinkage
  - Kelly criterion position sizing with VaR constraints
  - Asset-class specific risk limits (crypto: 20%, stocks: 25%)
  - Sub-microsecond performance (≤20µs edge blend, ≤50µs sizing)
- **Parameter Server v1** - Hot-reloading configuration management
  - Thread-safe YAML configuration loading
  - Redis pub/sub integration for live updates
  - Performance monitoring with sub-microsecond response times
- Comprehensive monitoring documentation (`docs/monitoring.md`)
  - Dashboard screenshots and explanations
  - Troubleshooting guides and best practices
  - Metrics reference and setup instructions
- Operations runbook (`docs/runbook.md`)
  - On-call playbook with ≤10min MTTR procedures
  - Alarm playbooks for 3 major alert types
  - Hot-reload procedures and release cut steps

### Changed
- Param Server hot-reload message format (`param.reload` JSON now includes `component` field)
- Signal Mux enhanced with model switching event logging
- Position Sizer integrated with TimeSeries metrics writing
- Enhanced error handling across all Risk Harmoniser components

### Fixed
- Decimal/float precision bug in VaR impact calculation (#142)
- Memory leak detection and prevention in edge blending
- Import conflicts with existing `yaml.py` file (renamed to `yaml_custom.py`)
- Configuration file path handling for hot-reload operations

### Security
- Input validation for all YAML configuration files
- Redis connection authentication and error handling
- Circuit breaker patterns for system resilience

## [0.3.2] – 2025-01-15

### Added
- Model Router dynamic routing based on instrument and horizon
- Feature Bus integration for multi-model predictions
- Basic Grafana monitoring setup

### Fixed
- WebSocket connection stability improvements
- Alpha model prediction pipeline optimization

## [0.3.1] – 2025-01-10

### Added
- Multi-model alpha prediction pipeline
- Basic risk management constraints
- Docker containerization

### Changed
- Switched to Redis for inter-component communication
- Improved logging and error handling

## [0.3.0] – 2025-01-05

### Added
- Initial alpha model integration
- Basic position sizing logic
- Market data ingestion pipeline

## [0.2.0] – 2024-12-20

### Added
- Core trading system architecture
- WebSocket market data feeds
- Basic signal processing

## [0.1.0] – 2024-12-15

### Added
- Initial project structure
- Development environment setup
- Basic configuration management

---

## Versioning Strategy

- **Major** (x.0.0): Breaking changes to API or architecture
- **Minor** (0.x.0): New features, backwards compatible
- **Patch** (0.0.x): Bug fixes, documentation updates
- **RC** (0.x.0-rcN): Release candidates for testing

## Release Notes

### v0.4.0-rc3 Highlights

This release represents a major milestone in the trading system evolution, introducing:

1. **Complete Risk Management Pipeline** - From edge blending to position sizing with mathematical rigor
2. **Production-Ready Monitoring** - Real-time Grafana dashboards with comprehensive alerting  
3. **Operational Excellence** - Hot-reload capabilities and detailed runbooks for 24/7 operations
4. **Performance Excellence** - Sub-microsecond latencies across all critical path components

The system is now ready for 48-hour paper trading validation before full production deployment.

**Breaking Changes:** None (backwards compatible with v0.3.x)

**Migration Required:** Update Grafana datasource configuration for new Redis TimeSeries keys

**Next Release:** v0.4.0 (production) planned after successful paper trading validation 
