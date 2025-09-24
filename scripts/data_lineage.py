#!/usr/bin/env python3
"""
Data Lineage Tracker
Scans datasets, feature snapshots, training runs and emits lineage documentation
"""
import os
import sys
import json
import glob
import hashlib
import argparse
import pathlib
import datetime
from datetime import timezone
from pathlib import Path


def compute_file_hash(file_path):
    """Compute SHA256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def scan_data_sources():
    """Scan for data source files and compute lineage."""
    data_sources = []

    # Common data directories to scan
    data_dirs = ["data/", "datasets/", "features/", "raw/", "processed/"]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for file_path in glob.glob(f"{data_dir}**/*", recursive=True):
                if os.path.isfile(file_path):
                    file_info = {
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "size_bytes": os.path.getsize(file_path),
                        "modified_time": datetime.datetime.fromtimestamp(
                            os.path.getmtime(file_path), tz=timezone.utc
                        ).isoformat(),
                        "hash": compute_file_hash(file_path),
                        "type": (
                            "dataset"
                            if any(
                                ext in file_path
                                for ext in [".csv", ".parquet", ".json", ".h5"]
                            )
                            else "other"
                        ),
                    }
                    data_sources.append(file_info)

    return sorted(data_sources, key=lambda x: x["modified_time"], reverse=True)


def scan_feature_engineering():
    """Scan for feature engineering artifacts."""
    features = []

    # Look for feature engineering code and outputs
    feature_patterns = [
        "src/layers/layer0_feature_engineering/**/*.py",
        "src/layers/layer2_feature_engineering/**/*.py",
        "features/**/*",
        "artifacts/features/**/*",
    ]

    for pattern in feature_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                feature_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size_bytes": os.path.getsize(file_path),
                    "modified_time": datetime.datetime.fromtimestamp(
                        os.path.getmtime(file_path), tz=timezone.utc
                    ).isoformat(),
                    "hash": compute_file_hash(file_path),
                    "type": (
                        "feature_code" if file_path.endswith(".py") else "feature_data"
                    ),
                }
                features.append(feature_info)

    return sorted(features, key=lambda x: x["modified_time"], reverse=True)


def scan_model_artifacts():
    """Scan for model training artifacts and checkpoints."""
    models = []

    # Common model artifact patterns
    model_patterns = [
        "checkpoints/**/*",
        "models/**/*",
        "artifacts/training/**/*",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.pkl",
    ]

    for pattern in model_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                model_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size_bytes": os.path.getsize(file_path),
                    "modified_time": datetime.datetime.fromtimestamp(
                        os.path.getmtime(file_path), tz=timezone.utc
                    ).isoformat(),
                    "hash": compute_file_hash(file_path),
                    "type": (
                        "model_checkpoint"
                        if any(ext in file_path for ext in [".pt", ".pth", ".onnx"])
                        else "model_artifact"
                    ),
                }
                models.append(model_info)

    return sorted(models, key=lambda x: x["modified_time"], reverse=True)


def scan_training_runs():
    """Scan for training run logs and metadata."""
    training_runs = []

    # Look for training logs and metadata
    training_patterns = [
        "logs/**/*.log",
        "runs/**/*",
        "artifacts/training/**/*.json",
        "tensorboard/**/*",
        "wandb/**/*",
    ]

    for pattern in training_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                run_info = {
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size_bytes": os.path.getsize(file_path),
                    "modified_time": datetime.datetime.fromtimestamp(
                        os.path.getmtime(file_path), tz=timezone.utc
                    ).isoformat(),
                    "hash": compute_file_hash(file_path),
                    "type": (
                        "training_log"
                        if file_path.endswith(".log")
                        else "training_metadata"
                    ),
                }
                training_runs.append(run_info)

    return sorted(training_runs, key=lambda x: x["modified_time"], reverse=True)


def analyze_dependencies():
    """Analyze code dependencies and data flow."""
    dependencies = {
        "python_packages": [],
        "internal_modules": [],
        "data_dependencies": [],
    }

    # Scan requirements.txt for external dependencies
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            dependencies["python_packages"] = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]

    # Scan for internal module imports
    python_files = glob.glob("src/**/*.py", recursive=True)
    for py_file in python_files[:10]:  # Sample first 10 files
        try:
            with open(py_file, "r") as f:
                content = f.read()
                # Simple regex to find imports (basic implementation)
                import re

                imports = re.findall(
                    r"^from\s+([^\s]+)\s+import|^import\s+([^\s]+)",
                    content,
                    re.MULTILINE,
                )
                for imp_tuple in imports:
                    imp = imp_tuple[0] or imp_tuple[1]
                    if (
                        imp.startswith("src.")
                        and imp not in dependencies["internal_modules"]
                    ):
                        dependencies["internal_modules"].append(imp)
        except Exception:
            continue

    return dependencies


def compute_lineage_graph():
    """Compute simplified data lineage graph."""
    graph = {"nodes": [], "edges": []}

    # Add data source nodes
    graph["nodes"].extend(
        [
            {
                "id": "raw_market_data",
                "type": "data_source",
                "description": "Raw market data feed",
            },
            {
                "id": "order_book_data",
                "type": "data_source",
                "description": "Order book snapshots",
            },
            {
                "id": "feature_engineering",
                "type": "process",
                "description": "Feature extraction and engineering",
            },
            {
                "id": "model_training",
                "type": "process",
                "description": "RL model training process",
            },
            {
                "id": "trained_model",
                "type": "model",
                "description": "Trained RL policy",
            },
            {
                "id": "validation_results",
                "type": "output",
                "description": "Model validation outputs",
            },
            {
                "id": "deployment",
                "type": "process",
                "description": "Model deployment to production",
            },
        ]
    )

    # Add edges showing data flow
    graph["edges"].extend(
        [
            {"from": "raw_market_data", "to": "feature_engineering"},
            {"from": "order_book_data", "to": "feature_engineering"},
            {"from": "feature_engineering", "to": "model_training"},
            {"from": "model_training", "to": "trained_model"},
            {"from": "trained_model", "to": "validation_results"},
            {"from": "trained_model", "to": "deployment"},
        ]
    )

    return graph


def generate_lineage_json(
    data_sources, features, models, training_runs, dependencies, lineage_graph
):
    """Generate machine-readable lineage JSON."""
    return {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "lineage_version": "1.0",
        "summary": {
            "data_sources": len(data_sources),
            "feature_artifacts": len(features),
            "model_artifacts": len(models),
            "training_runs": len(training_runs),
        },
        "data_sources": data_sources[:20],  # Limit to most recent 20
        "feature_engineering": features[:10],
        "model_artifacts": models[:10],
        "training_runs": training_runs[:10],
        "dependencies": dependencies,
        "lineage_graph": lineage_graph,
        "integrity": {
            "total_files_scanned": len(data_sources)
            + len(features)
            + len(models)
            + len(training_runs),
            "scan_timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        },
    }


def generate_lineage_markdown(lineage_data):
    """Generate human-readable lineage documentation."""
    summary = lineage_data["summary"]

    markdown = f"""# Data Lineage Documentation

**Generated:** {lineage_data['timestamp']}
**Lineage Version:** {lineage_data['lineage_version']}

## Summary

- **Data Sources:** {summary['data_sources']} files
- **Feature Artifacts:** {summary['feature_artifacts']} files  
- **Model Artifacts:** {summary['model_artifacts']} files
- **Training Runs:** {summary['training_runs']} artifacts

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
"""

    for ds in lineage_data["data_sources"][:10]:
        size_mb = ds["size_bytes"] / (1024 * 1024)
        hash_short = ds["hash"][:8] if ds["hash"] else "N/A"
        modified = ds["modified_time"][:19]  # Remove microseconds
        markdown += f"| `{ds['name']}` | {ds['type']} | {size_mb:.1f}MB | {modified} | {hash_short} |\n"

    markdown += f"""
## Feature Engineering

Recent feature engineering artifacts:

| File | Type | Size | Last Modified |
|------|------|------|---------------|
"""

    for feat in lineage_data["feature_engineering"][:10]:
        size_kb = feat["size_bytes"] / 1024
        modified = feat["modified_time"][:19]
        markdown += (
            f"| `{feat['name']}` | {feat['type']} | {size_kb:.1f}KB | {modified} |\n"
        )

    markdown += f"""
## Model Artifacts

Recent model checkpoints and artifacts:

| File | Type | Size | Last Modified |
|------|------|------|---------------|
"""

    for model in lineage_data["model_artifacts"][:10]:
        size_mb = model["size_bytes"] / (1024 * 1024)
        modified = model["modified_time"][:19]
        markdown += (
            f"| `{model['name']}` | {model['type']} | {size_mb:.1f}MB | {modified} |\n"
        )

    markdown += f"""
## Dependencies

### External Python Packages
"""
    for pkg in lineage_data["dependencies"]["python_packages"][:15]:
        markdown += f"- {pkg}\n"

    markdown += f"""
### Internal Modules
"""
    for mod in lineage_data["dependencies"]["internal_modules"][:10]:
        markdown += f"- {mod}\n"

    markdown += f"""
## Data Lineage Graph

### Nodes
"""
    for node in lineage_data["lineage_graph"]["nodes"]:
        markdown += f"- **{node['id']}** ({node['type']}): {node['description']}\n"

    markdown += f"""
### Data Flow
"""
    for edge in lineage_data["lineage_graph"]["edges"]:
        markdown += f"- {edge['from']} → {edge['to']}\n"

    markdown += f"""
## Integrity Checks

- **Total Files Scanned:** {lineage_data['integrity']['total_files_scanned']}
- **Scan Timestamp:** {lineage_data['integrity']['scan_timestamp']}
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
"""

    return markdown


def main():
    """Main data lineage scanning function."""
    parser = argparse.ArgumentParser(description="Generate data lineage documentation")
    parser.add_argument("--out", default="artifacts/lineage", help="Output directory")
    parser.add_argument(
        "--scan-depth", type=int, default=3, help="Directory scan depth"
    )
    args = parser.parse_args()

    print("🔍 Scanning data lineage...")

    print("  📊 Scanning data sources...")
    data_sources = scan_data_sources()

    print("  🔧 Scanning feature engineering...")
    features = scan_feature_engineering()

    print("  🤖 Scanning model artifacts...")
    models = scan_model_artifacts()

    print("  📈 Scanning training runs...")
    training_runs = scan_training_runs()

    print("  🔗 Analyzing dependencies...")
    dependencies = analyze_dependencies()

    print("  📊 Computing lineage graph...")
    lineage_graph = compute_lineage_graph()

    # Create output directory
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output_dir = Path(args.out) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  📝 Generating lineage documentation...")

    # Generate JSON lineage
    lineage_data = generate_lineage_json(
        data_sources, features, models, training_runs, dependencies, lineage_graph
    )

    json_file = output_dir / "lineage.json"
    with open(json_file, "w") as f:
        json.dump(lineage_data, f, indent=2)

    # Generate markdown documentation
    markdown_content = generate_lineage_markdown(lineage_data)
    markdown_file = output_dir / "lineage.md"
    with open(markdown_file, "w") as f:
        f.write(markdown_content)

    print(f"✅ Data lineage generated:")
    print(f"   📊 Data Sources: {len(data_sources)}")
    print(f"   🔧 Feature Files: {len(features)}")
    print(f"   🤖 Model Files: {len(models)}")
    print(f"   📈 Training Artifacts: {len(training_runs)}")
    print(f"   📄 JSON: {json_file}")
    print(f"   📖 Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
