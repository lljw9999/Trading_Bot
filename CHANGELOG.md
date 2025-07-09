# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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