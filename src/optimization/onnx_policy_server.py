#!/usr/bin/env python3
"""
ONNX Policy Server
Export live policy to ONNX and run via Triton (or torch.compile + autocast FP16) to cut inference p99
"""

import os
import sys
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("onnx_policy_server")


class PolicyModel(nn.Module):
    """Simple policy model for demonstration."""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 3
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class ONNXPolicyServer:
    """High-performance policy inference server using ONNX."""

    def __init__(self):
        """Initialize ONNX policy server."""
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Configuration
        self.config = {
            "model_path": "/opt/trader/models/policy_model.onnx",
            "pytorch_model_path": "/opt/trader/models/policy_model.pth",
            "input_dim": 128,
            "output_dim": 3,  # [hold, buy, sell]
            "batch_size": 32,
            "max_batch_wait_ms": 50,  # Max wait time to collect batch
            "inference_timeout_ms": 100,
            "use_fp16": True,
            "use_cuda": torch.cuda.is_available(),
            "thread_pool_size": 4,
        }

        # Performance tracking
        self.metrics = {
            "requests_total": 0,
            "requests_batched": 0,
            "inference_times": [],
            "batch_sizes": [],
            "errors_total": 0,
        }

        # ONNX session
        self.ort_session = None
        self.pytorch_model = None
        self.batch_queue = []
        self.batch_futures = []

        # Thread pool for inference
        self.executor = ThreadPoolExecutor(max_workers=self.config["thread_pool_size"])

        logger.info("🚀 ONNX Policy Server initialized")

    def export_pytorch_to_onnx(
        self, pytorch_model_path: str, onnx_model_path: str
    ) -> bool:
        """Export PyTorch model to ONNX format."""
        try:
            logger.info("📦 Exporting PyTorch model to ONNX...")

            # Load PyTorch model
            device = torch.device("cuda" if self.config["use_cuda"] else "cpu")
            model = PolicyModel(
                input_dim=self.config["input_dim"], output_dim=self.config["output_dim"]
            )

            if Path(pytorch_model_path).exists():
                checkpoint = torch.load(pytorch_model_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"📥 Loaded PyTorch model from {pytorch_model_path}")
            else:
                logger.warning(
                    "⚠️ PyTorch model not found, using randomly initialized weights"
                )

            model.eval()
            model = model.to(device)

            # Create dummy input
            dummy_input = torch.randn(1, self.config["input_dim"], device=device)

            if self.config["use_fp16"]:
                model = model.half()
                dummy_input = dummy_input.half()

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Verify ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)

            logger.info(f"✅ Successfully exported ONNX model to {onnx_model_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            return False

    def load_onnx_model(self) -> bool:
        """Load ONNX model for inference."""
        try:
            if not Path(self.config["model_path"]).exists():
                logger.info("📦 ONNX model not found, exporting from PyTorch...")
                if not self.export_pytorch_to_onnx(
                    self.config["pytorch_model_path"], self.config["model_path"]
                ):
                    return False

            # Create ONNX Runtime session
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.config["use_cuda"]
                else ["CPUExecutionProvider"]
            )

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.config["thread_pool_size"]
            session_options.inter_op_num_threads = self.config["thread_pool_size"]
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self.ort_session = ort.InferenceSession(
                self.config["model_path"],
                sess_options=session_options,
                providers=providers,
            )

            logger.info(f"✅ Loaded ONNX model: {self.config['model_path']}")
            logger.info(f"🔧 Providers: {self.ort_session.get_providers()}")

            return True

        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return False

    def load_pytorch_model(self) -> bool:
        """Load PyTorch model with torch.compile optimization."""
        try:
            device = torch.device("cuda" if self.config["use_cuda"] else "cpu")

            model = PolicyModel(
                input_dim=self.config["input_dim"], output_dim=self.config["output_dim"]
            )

            if Path(self.config["pytorch_model_path"]).exists():
                checkpoint = torch.load(
                    self.config["pytorch_model_path"], map_location=device
                )
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(
                    f"📥 Loaded PyTorch model from {self.config['pytorch_model_path']}"
                )
            else:
                logger.warning(
                    "⚠️ PyTorch model not found, using randomly initialized weights"
                )

            model.eval()
            model = model.to(device)

            if self.config["use_fp16"]:
                model = model.half()

            # Use torch.compile for optimization (if available)
            try:
                if hasattr(torch, "compile"):
                    model = torch.compile(model, mode="max-autotune")
                    logger.info("🚀 Enabled torch.compile optimization")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile not available: {e}")

            self.pytorch_model = model
            logger.info("✅ PyTorch model loaded and optimized")

            return True

        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False

    def preprocess_features(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Preprocess input features for inference."""
        try:
            # Convert features to numpy array
            batch_size = len(features)
            input_array = np.zeros(
                (batch_size, self.config["input_dim"]), dtype=np.float32
            )

            for i, feature_dict in enumerate(features):
                # Extract numeric features (mock implementation)
                feature_vector = []

                # Price features
                feature_vector.extend(
                    [
                        feature_dict.get("price", 0.0),
                        feature_dict.get("volume", 0.0),
                        feature_dict.get("spread", 0.0),
                        feature_dict.get("volatility", 0.0),
                    ]
                )

                # Technical indicators
                feature_vector.extend(
                    [
                        feature_dict.get("rsi", 50.0),
                        feature_dict.get("macd", 0.0),
                        feature_dict.get("bollinger_upper", 0.0),
                        feature_dict.get("bollinger_lower", 0.0),
                    ]
                )

                # Market microstructure
                feature_vector.extend(
                    [
                        feature_dict.get("order_book_imbalance", 0.0),
                        feature_dict.get("trade_intensity", 0.0),
                        feature_dict.get("bid_ask_spread", 0.0),
                        feature_dict.get("market_impact", 0.0),
                    ]
                )

                # Pad or truncate to input_dim
                while len(feature_vector) < self.config["input_dim"]:
                    feature_vector.append(0.0)
                feature_vector = feature_vector[: self.config["input_dim"]]

                input_array[i] = feature_vector

            return input_array

        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            raise

    def run_onnx_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference using ONNX model."""
        try:
            start_time = time.time()

            # Run inference
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
            outputs = self.ort_session.run(None, ort_inputs)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics["inference_times"].append(inference_time)

            return outputs[0]

        except Exception as e:
            logger.error(f"Error in ONNX inference: {e}")
            self.metrics["errors_total"] += 1
            raise

    def run_pytorch_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference using PyTorch model."""
        try:
            start_time = time.time()

            device = torch.device("cuda" if self.config["use_cuda"] else "cpu")

            # Convert to tensor
            input_tensor = torch.from_numpy(input_data).to(device)

            if self.config["use_fp16"]:
                input_tensor = input_tensor.half()

            # Run inference
            with torch.no_grad():
                if self.config["use_fp16"]:
                    with torch.autocast(
                        device_type="cuda" if self.config["use_cuda"] else "cpu"
                    ):
                        outputs = self.pytorch_model(input_tensor)
                else:
                    outputs = self.pytorch_model(input_tensor)

            # Convert back to numpy
            outputs_np = outputs.cpu().float().numpy()

            inference_time = (time.time() - start_time) * 1000
            self.metrics["inference_times"].append(inference_time)

            return outputs_np

        except Exception as e:
            logger.error(f"Error in PyTorch inference: {e}")
            self.metrics["errors_total"] += 1
            raise

    def predict_batch(
        self, features_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict actions for a batch of features."""
        try:
            # Preprocess features
            input_data = self.preprocess_features(features_batch)

            # Run inference (prefer ONNX if available)
            if self.ort_session:
                outputs = self.run_onnx_inference(input_data)
            elif self.pytorch_model:
                outputs = self.run_pytorch_inference(input_data)
            else:
                raise Exception("No model loaded")

            # Post-process outputs
            results = []
            action_names = ["hold", "buy", "sell"]

            for i, output in enumerate(outputs):
                action_probs = output.tolist()
                action_idx = np.argmax(output)

                results.append(
                    {
                        "action": action_names[action_idx],
                        "confidence": float(action_probs[action_idx]),
                        "probabilities": {
                            "hold": float(action_probs[0]),
                            "buy": float(action_probs[1]),
                            "sell": float(action_probs[2]),
                        },
                    }
                )

            # Update metrics
            self.metrics["requests_total"] += len(features_batch)
            self.metrics["batch_sizes"].append(len(features_batch))

            return results

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            self.metrics["errors_total"] += len(features_batch)
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            if not self.metrics["inference_times"]:
                return {"status": "no_data"}

            inference_times = self.metrics["inference_times"][
                -1000:
            ]  # Last 1000 requests
            batch_sizes = self.metrics["batch_sizes"][-1000:]

            metrics = {
                "requests_total": self.metrics["requests_total"],
                "requests_batched": self.metrics["requests_batched"],
                "errors_total": self.metrics["errors_total"],
                "error_rate": self.metrics["errors_total"]
                / max(self.metrics["requests_total"], 1),
                "inference_p50_ms": float(np.percentile(inference_times, 50)),
                "inference_p95_ms": float(np.percentile(inference_times, 95)),
                "inference_p99_ms": float(np.percentile(inference_times, 99)),
                "avg_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0,
                "throughput_rps": (
                    len(inference_times) / 60 if inference_times else 0
                ),  # Requests per second (approximate)
                "model_type": "onnx" if self.ort_session else "pytorch",
                "fp16_enabled": self.config["use_fp16"],
                "cuda_enabled": self.config["use_cuda"],
            }

            # Store metrics in Redis
            self.redis.set("policy:inference_p50_ms", metrics["inference_p50_ms"])
            self.redis.set("policy:inference_p99_ms", metrics["inference_p99_ms"])
            self.redis.set("policy:throughput_rps", metrics["throughput_rps"])
            self.redis.set("policy:error_rate", metrics["error_rate"])

            return metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}


# Global server instance
policy_server = ONNXPolicyServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting ONNX Policy Server...")

    # Try to load ONNX model first, fallback to PyTorch
    if not policy_server.load_onnx_model():
        logger.warning("⚠️ ONNX model loading failed, falling back to PyTorch")
        if not policy_server.load_pytorch_model():
            logger.error("❌ Both ONNX and PyTorch model loading failed")
            raise Exception("No model could be loaded")

    yield

    # Shutdown would go here if needed
    logger.info("🔽 ONNX Policy Server shutting down")


# FastAPI app
app = FastAPI(
    title="ONNX Policy Server",
    description="High-performance policy inference using ONNX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": policy_server.ort_session is not None
        or policy_server.pytorch_model is not None,
        "model_type": "onnx" if policy_server.ort_session else "pytorch",
    }


@app.post("/predict")
async def predict(features: Dict[str, Any]):
    """Single prediction endpoint."""
    try:
        results = policy_server.predict_batch([features])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(features_batch: List[Dict[str, Any]]):
    """Batch prediction endpoint."""
    try:
        if len(features_batch) > policy_server.config["batch_size"]:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(features_batch)} exceeds maximum {policy_server.config['batch_size']}",
            )

        results = policy_server.predict_batch(features_batch)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return policy_server.get_performance_metrics()


@app.get("/config")
async def get_config():
    """Get server configuration."""
    return policy_server.config


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Policy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export PyTorch model to ONNX and exit",
    )
    parser.add_argument("--pytorch-model", type=str, help="Path to PyTorch model")
    parser.add_argument("--onnx-model", type=str, help="Path to ONNX model")

    args = parser.parse_args()

    if args.pytorch_model:
        policy_server.config["pytorch_model_path"] = args.pytorch_model
    if args.onnx_model:
        policy_server.config["model_path"] = args.onnx_model

    if args.export_onnx:
        success = policy_server.export_pytorch_to_onnx(
            policy_server.config["pytorch_model_path"],
            policy_server.config["model_path"],
        )
        sys.exit(0 if success else 1)

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
