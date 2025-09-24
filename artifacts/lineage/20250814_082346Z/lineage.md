# Data Lineage Documentation

**Generated:** 2025-08-14T08:23:46.146947+00:00
**Lineage Version:** 1.0

## Summary

- **Data Sources:** 43 files
- **Feature Artifacts:** 4 files  
- **Model Artifacts:** 5 files
- **Training Runs:** 25 artifacts

## Data Flow Architecture

```
Raw Market Data ──→ Feature Engineering ──→ Model Training ──→ Trained Model
Order Book Data ──┘                                           ├──→ Validation
                                                               └──→ Deployment
```

## Data Sources

Recent data source files (showing most recent 10):

| File | Type | Size | Last Modified | Hash |
|------|------|------|---------------|------|
| `policy_shadow_metrics.csv` | dataset | 0.0MB | 2025-08-14T08:22:59 | c2a1aa6f |
| `baseline_metrics.csv` | dataset | 0.0MB | 2025-08-14T08:22:59 | 1b03f7f9 |
| `slo_history.db` | other | 0.0MB | 2025-08-09T15:51:29 | ac3dbba7 |
| `ab_history.db` | other | 0.0MB | 2025-08-09T01:22:07 | 5cff8022 |
| `baseline_state_sample.parquet` | dataset | 0.7MB | 2025-08-08T13:36:31 | afe5d2b7 |
| `00000000` | other | 0.0MB | 2025-07-09T09:23:44 | 7a821c70 |
| `00000008` | other | 0.0MB | 2025-07-09T09:23:44 | 1802a34a |
| `00000007` | other | 0.0MB | 2025-07-09T09:23:44 | 64938803 |
| `00000006` | other | 0.0MB | 2025-07-09T09:23:44 | 0c4516f6 |
| `NVDA.csv` | dataset | 0.0MB | 2025-07-09T09:23:44 | d6ec976d |

## Feature Engineering

Recent feature engineering artifacts:

| File | Type | Size | Last Modified |
|------|------|------|---------------|
| `funding_basis_features.py` | feature_code | 6.8KB | 2025-08-09T01:56:49 |
| `feature_builder.py` | feature_code | 7.6KB | 2025-08-08T13:27:32 |
| `graph_neural_networks.py` | feature_code | 26.1KB | 2025-07-19T13:24:19 |
| `temporal_fusion_transformer.py` | feature_code | 34.8KB | 2025-07-19T13:07:49 |

## Model Artifacts

Recent model checkpoints and artifacts:

| File | Type | Size | Last Modified |
|------|------|------|---------------|
| `latest.pt` | model_checkpoint | 0.0MB | 2025-08-13T07:52:23 |
| `tlob_tiny_int8.onnx` | model_checkpoint | 0.1MB | 2025-07-09T09:23:44 |
| `tlob_tiny.onnx` | model_checkpoint | 0.3MB | 2025-07-09T09:23:44 |
| `patchtst_small_int8.onnx` | model_checkpoint | 0.1MB | 2025-07-09T09:23:44 |
| `patchtst_small.onnx` | model_checkpoint | 0.3MB | 2025-07-09T09:23:44 |

## Dependencies

### External Python Packages
- numpy==2.3.1
- pandas==2.3.1
- pydantic==2.11.7
- redis==6.2.0
- PyYAML==6.0.2
- torch==2.7.1
- transformers==4.53.1
- scikit-learn
- lightgbm==4.6.0
- huggingface-hub>=0.33.2
- fastapi==0.116.0
- uvicorn==0.35.0
- httpx==0.28.1
- sqlalchemy==1.4.54
- alembic==1.16.3

### Internal Modules

## Data Lineage Graph

### Nodes
- **raw_market_data** (data_source): Raw market data feed
- **order_book_data** (data_source): Order book snapshots
- **feature_engineering** (process): Feature extraction and engineering
- **model_training** (process): RL model training process
- **trained_model** (model): Trained RL policy
- **validation_results** (output): Model validation outputs
- **deployment** (process): Model deployment to production

### Data Flow
- raw_market_data → feature_engineering
- order_book_data → feature_engineering
- feature_engineering → model_training
- model_training → trained_model
- trained_model → validation_results
- trained_model → deployment

## Integrity Checks

- **Total Files Scanned:** 77
- **Scan Timestamp:** 2025-08-14T08:23:46.146952+00:00
- **Hash Algorithm:** SHA256
- **Lineage Coverage:** Data sources, features, models, training logs

## Recommendations

### Data Quality
- Implement automated data validation checks
- Add data schema validation for incoming feeds
- Monitor data freshness and completeness

### Model Governance
- Maintain model versioning with semantic versioning
- Implement model metadata tracking
- Add automated model performance monitoring

### Lineage Monitoring
- Schedule regular lineage scans (weekly)
- Alert on significant data source changes
- Track model drift relative to training data

---
*This lineage documentation was generated automatically by the Data Lineage Tracker.*
