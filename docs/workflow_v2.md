# Trading System Workflow v2

A comprehensive guide to the multi-layer trading system architecture and operational workflows.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)  
3. [Model Registry](#model-registry)
4. [Alpha Development Workflow](#alpha-development-workflow)
5. [Deployment Pipeline](#deployment-pipeline)
6. [Monitoring & Operations](#monitoring--operations)

## System Architecture

The trading system follows a 6-layer architecture designed for real-time processing, scalability, and risk management:

### Layer 0: Data Ingestion
- **Purpose**: Real-time market data collection from multiple exchanges
- **Components**: WebSocket connectors for Coinbase, Binance, Alpaca, NOWNodes
- **Output**: Standardized market ticks to Kafka topics

### Layer 1: Alpha Models
- **Purpose**: Generate trading signals from market data
- **Components**: Transformer models, statistical models, sentiment analysis
- **Models**: TLOB-Tiny, PatchTST-Small, Order Book Pressure, MA Momentum

### Layer 2: Ensemble Learning
- **Purpose**: Combine multiple alpha signals intelligently
- **Components**: Logistic meta-learner, weighted averaging, signal fusion
- **Output**: Unified trading signals with confidence scores

### Layer 3: Position Sizing
- **Purpose**: Determine optimal position sizes based on signals and risk
- **Components**: Kelly criterion, risk budgeting, portfolio optimization
- **Features**: Big-bet detection, volatility targeting, concentration limits

### Layer 4: Execution
- **Purpose**: Execute trades efficiently with minimal market impact
- **Components**: Smart order routing, TWAP/VWAP algorithms, slippage minimization
- **Features**: Multi-exchange execution, latency optimization

### Layer 5: Risk Management
- **Purpose**: Monitor and control portfolio risk in real-time
- **Components**: Position limits, VaR calculation, circuit breakers
- **Features**: Drawdown protection, correlation monitoring, emergency stops

---

## Data Flow

### Real-time Pipeline
```
Exchange APIs → Layer 0 (Connectors) → Kafka → Feature Bus → Layer 1 (Alphas) 
                                                                      ↓
Layer 5 (Risk) ← Layer 4 (Execution) ← Layer 3 (Sizing) ← Layer 2 (Ensemble)
```

### Message Topics
- `market.raw.crypto`: Raw crypto market data
- `market.raw.stocks`: Raw stock market data  
- `features.raw`: Computed technical indicators
- `signals.alpha`: Individual alpha model outputs
- `signals.ensemble`: Combined ensemble signals
- `orders.target`: Target position changes
- `metrics.execution`: Trade execution metrics

---

## Model Registry

The Model Registry provides centralized management of transformer models for the trading system, enabling hot-swapping of models and efficient model versioning.

### Architecture Overview

```
HuggingFace Hub → Local Cache → S3 Storage → Triton Server → Live Trading
     ↓              ↓              ↓              ↓              ↓
Model Source → `fetch_models.py` → Model Store → Inference → Redis Signals
```

### fetch_models.py Utility

The model fetch utility (`scripts/fetch_models.py`) handles downloading, caching, and deployment of transformer models:

**Key Features:**
- **Smart Caching**: Downloads to `~/.cache/hf_models/<model_name>` with SHA256 verification
- **S3 Integration**: Optional upload to `s3://trading-models/<model_name>/` for production
- **Progress Tracking**: Real-time download progress with file size and ETA
- **Error Handling**: Graceful handling of network issues and missing repositories

**Model Registry:**
```python
MODELS = {
    'tlob_tiny': {
        'repo_id': 'LeonardoBerti00/TLOB-FI-2010',
        'key_files': ['tlob_tiny.pt', 'config.json'],
        'description': 'TLOB Transformer for FI-2010 dataset',
        'size_mb': 45
    },
    'patchtst_small': {
        'repo_id': 'ibm-granite/granite-timeseries-patchtst', 
        'key_files': ['pytorch_model.bin', 'config.json'],
        'description': 'Granite PatchTST for time series forecasting',
        'size_mb': 120
    },
    # ... additional models
}
```

**Usage Examples:**
```bash
# List all available models
python scripts/fetch_models.py --list

# Download specific model
python scripts/fetch_models.py tlob_tiny

# Download all models
python scripts/fetch_models.py --all

# Download and upload to S3
python scripts/fetch_models.py --upload-s3 patchtst_small
```

### S3 Storage Layout

The S3 model bucket follows a structured layout for efficient model management:

```
s3://trading-models/
├── tlob_tiny/
│   ├── model.onnx              # Quantized ONNX model
│   ├── config.json             # Model configuration
│   ├── metadata.json           # Model metadata (version, SHA256, etc.)
│   └── versions/
│       ├── v1.0/              # Version history
│       └── v1.1/
├── patchtst_small/
│   ├── model.onnx
│   ├── config.json
│   └── metadata.json
└── chronos_bolt_base/
    ├── model.onnx
    ├── config.json
    └── metadata.json
```

**Metadata Structure:**
```json
{
    "model_name": "tlob_tiny",
    "version": "1.0",
    "created_at": "2025-01-17T10:30:00Z",
    "sha256": "a1b2c3d4e5f6...",
    "file_size_mb": 45.2,
    "onnx_optimized": true,
    "quantization": "INT8",
    "performance": {
        "latency_ms": 0.19,
        "throughput_qps": 5263
    }
}
```

### Hot-Swap via Redis

The system supports hot-swapping of models without downtime using Redis as a coordination mechanism:

**Redis Keys:**
- `model:registry:active`: Currently active model configuration
- `model:registry:pending`: Pending model configuration for hot-swap
- `model:performance:<model_name>`: Real-time performance metrics
- `model:signals:<model_name>`: Latest model predictions

**Hot-Swap Process:**
1. **Upload New Model**: New model uploaded to S3 with updated metadata
2. **Stage Model**: Model configuration updated in `model:registry:pending`
3. **Health Check**: System validates new model performance on test data
4. **Atomic Switch**: `model:registry:active` updated atomically
5. **Cleanup**: Old model moved to archive, metrics updated

**Configuration Example:**
```json
{
    "active_models": {
        "high_frequency": "tlob_tiny_v1.1",
        "medium_frequency": "patchtst_small_v2.0", 
        "long_horizon": "chronos_bolt_base_v1.0"
    },
    "routing_rules": {
        "symbols": ["BTC", "ETH"],
        "horizon_minutes": [1, 5, 15, 60],
        "sequence_length": [32, 64, 128]
    },
    "fallback_model": "tlob_tiny_v1.0"
}
```

**Hot-Swap Commands:**
```bash
# Check current model status
redis-cli HGETALL model:registry:active

# Stage new model for deployment
redis-cli HSET model:registry:pending high_frequency tlob_tiny_v1.2

# Monitor model performance
redis-cli HGETALL model:performance:tlob_tiny_v1.2
```

### Makefile Integration

Model management is integrated into the development workflow via Makefile targets:

```bash
# Model registry operations
make models-list          # List all models and cache status
make models-sync          # Download all models
make models-fetch MODEL=tlob_tiny     # Download specific model
make models-upload-s3 MODEL=tlob_tiny # Upload to S3
make models-clean         # Clear local cache
make models-status        # Detailed status report
```

### CI/CD Integration

The model registry integrates with CI/CD pipelines for automated model deployment:

**GitHub Actions Workflow:**
```yaml
- name: Cache HuggingFace Models
  uses: actions/cache@v3
  with:
    path: ~/.cache/huggingface/hub
    key: hf-models-${{ hashFiles('scripts/fetch_models.py') }}

- name: Download Models
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python scripts/fetch_models.py --all
    
- name: Upload to S3
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: |
    python scripts/fetch_models.py --upload-s3 --all
```

---

## Alpha Development Workflow

### 1. Research Phase
- **Data Analysis**: Explore market patterns and anomalies
- **Feature Engineering**: Design predictive features from raw market data
- **Backtesting**: Validate alpha performance on historical data
- **Risk Assessment**: Analyze correlations and regime dependencies

### 2. Implementation Phase
- **Model Development**: Implement alpha logic in standardized framework
- **Unit Testing**: Comprehensive testing of model components
- **Integration Testing**: End-to-end pipeline validation
- **Performance Optimization**: Latency and throughput optimization

### 3. Deployment Phase
- **Staging Environment**: Deploy to staging for live data testing
- **A/B Testing**: Compare new alpha against existing models
- **Gradual Rollout**: Incremental exposure to live trading
- **Performance Monitoring**: Continuous tracking of model performance

### 4. Monitoring Phase
- **Signal Quality**: Track signal-to-noise ratio and accuracy
- **Model Drift**: Monitor for concept drift and performance decay
- **Risk Metrics**: Ensure risk characteristics remain stable
- **Performance Attribution**: Analyze contribution to overall returns

---

## Deployment Pipeline

### Development Environment
```bash
# Start development stack
make dev-setup

# Run tests
make test

# Start trading simulation
python run_crypto_session.py --symbols BTC-USD ETH-USD --duration 30
```

### Staging Environment
```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run integration tests
make test-integration

# Monitor system health
make status
```

### Production Environment
```bash
# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Start trading with real money
export PAPER_TRADING=false
python run_crypto_session.py --symbols BTC-USD ETH-USD --duration 240
```

---

## Monitoring & Operations

### Key Metrics
- **Latency**: End-to-end processing latency < 100ms
- **Throughput**: Message processing rate > 1000 msg/s
- **Accuracy**: Signal accuracy > 55%
- **Risk**: Maximum drawdown < 3%
- **Uptime**: System availability > 99.5%

### Dashboards
- **Trading Dashboard**: Real-time P&L, positions, signals
- **System Health**: Component status, latencies, error rates
- **Model Performance**: Alpha accuracy, ensemble weights
- **Risk Monitor**: Exposure, VaR, correlation matrix

### Alerting
- **Critical Alerts**: System failures, risk limit breaches
- **Warning Alerts**: Performance degradation, model drift
- **Info Alerts**: Successful deployments, daily summaries

### Incident Response
1. **Detection**: Automated monitoring and alerting
2. **Assessment**: Rapid impact analysis and root cause identification
3. **Mitigation**: Emergency procedures and circuit breakers
4. **Recovery**: System restoration and performance validation
5. **Post-Mortem**: Incident analysis and prevention measures

---

## Best Practices

### Development
- **Code Quality**: Comprehensive testing, code reviews, documentation
- **Version Control**: Git flow with feature branches and pull requests
- **Configuration Management**: Environment-specific configurations
- **Dependency Management**: Pinned versions and virtual environments

### Operations
- **Infrastructure as Code**: Docker containers, Kubernetes manifests
- **Monitoring**: Comprehensive metrics, logging, and alerting
- **Backup & Recovery**: Regular backups, disaster recovery procedures
- **Security**: API key management, network security, access controls

### Risk Management
- **Position Limits**: Per-asset and portfolio-level limits
- **Diversification**: Correlation monitoring and sector limits
- **Circuit Breakers**: Automated stop-loss and volatility halts
- **Manual Overrides**: Emergency stop mechanisms and manual controls

---

## Conclusion

The trading system workflow provides a robust framework for developing, testing, and deploying quantitative trading strategies. The modular architecture enables rapid iteration while maintaining production stability and risk controls.

For additional information, see:
- [API Documentation](api.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md) 