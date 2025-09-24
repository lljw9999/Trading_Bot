#!/usr/bin/env python3
"""
ONNX Runtime Model Runner

High-performance ONNX model inference with automatic fallback to PyTorch.
Supports CPU and GPU execution with session caching and batch processing.
"""

import os
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ONNXModelRunner:
    """High-performance ONNX model runner with fallback support."""

    def __init__(self):
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.fallback_models: Dict[str, Any] = {}  # PyTorch fallback models
        self.inference_stats: Dict[str, Dict[str, float]] = {}

        # Initialize providers based on availability
        self.providers = self._get_available_providers()
        logger.info(f"ONNX Runtime providers: {self.providers}")

    def _get_available_providers(self) -> List[str]:
        """Get available execution providers."""
        if not ONNXRUNTIME_AVAILABLE:
            return []

        available = ort.get_available_providers()

        # Prefer GPU providers
        preferred_order = [
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "DmlExecutionProvider",  # DirectML for Windows
            "CPUExecutionProvider",
        ]

        providers = []
        for provider in preferred_order:
            if provider in available:
                providers.append(provider)

        if not providers:
            providers = ["CPUExecutionProvider"]

        return providers

    def load_model(
        self,
        name: str,
        model_path: Union[str, Path],
        fallback_model: Optional[Any] = None,
    ) -> bool:
        """Load ONNX model with optional PyTorch fallback."""
        try:
            if not ONNXRUNTIME_AVAILABLE:
                logger.warning(f"ONNX Runtime not available for {name}, using fallback")
                if fallback_model is not None:
                    self.fallback_models[name] = fallback_model
                    return True
                return False

            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                if fallback_model is not None:
                    self.fallback_models[name] = fallback_model
                    logger.info(f"Using fallback model for {name}")
                    return True
                return False

            # Create session with optimized settings
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Enable parallel execution for CPU
            if "CPUExecutionProvider" in self.providers:
                session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                session_options.inter_op_num_threads = 1

            session = ort.InferenceSession(
                str(model_path), session_options, providers=self.providers
            )

            self.sessions[name] = session

            # Store model metadata
            input_info = [
                (inp.name, inp.shape, inp.type) for inp in session.get_inputs()
            ]
            output_info = [
                (out.name, out.shape, out.type) for out in session.get_outputs()
            ]

            self.model_info[name] = {
                "path": str(model_path),
                "inputs": input_info,
                "outputs": output_info,
                "providers": session.get_providers(),
                "loaded_at": time.time(),
            }

            # Initialize stats
            self.inference_stats[name] = {
                "total_calls": 0,
                "total_time": 0.0,
                "avg_latency_ms": 0.0,
                "error_count": 0,
            }

            logger.info(f"Loaded ONNX model '{name}' from {model_path}")
            logger.info(f"  Inputs: {[f'{n}:{s}' for n,s,_ in input_info]}")
            logger.info(f"  Outputs: {[f'{n}:{s}' for n,s,_ in output_info]}")
            logger.info(f"  Providers: {session.get_providers()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load ONNX model '{name}': {e}")

            # Try fallback model
            if fallback_model is not None:
                self.fallback_models[name] = fallback_model
                logger.info(f"Using fallback model for {name}")
                return True

            return False

    def run_inference(
        self, name: str, inputs: Dict[str, np.ndarray], timeout: float = 5.0
    ) -> Optional[Dict[str, np.ndarray]]:
        """Run model inference with automatic fallback."""
        if name not in self.sessions and name not in self.fallback_models:
            logger.error(f"Model '{name}' not loaded")
            return None

        start_time = time.time()

        try:
            # Use ONNX model if available
            if name in self.sessions:
                return self._run_onnx_inference(name, inputs, timeout)

            # Fall back to PyTorch model
            elif name in self.fallback_models:
                return self._run_fallback_inference(name, inputs)

        except Exception as e:
            logger.error(f"Inference failed for '{name}': {e}")

            # Update error stats
            if name in self.inference_stats:
                self.inference_stats[name]["error_count"] += 1

            # Try fallback if ONNX failed
            if name in self.sessions and name in self.fallback_models:
                logger.warning(f"ONNX inference failed for '{name}', trying fallback")
                return self._run_fallback_inference(name, inputs)

        finally:
            # Update timing stats
            if name in self.inference_stats:
                duration = time.time() - start_time
                stats = self.inference_stats[name]
                stats["total_calls"] += 1
                stats["total_time"] += duration
                stats["avg_latency_ms"] = (
                    stats["total_time"] / stats["total_calls"]
                ) * 1000

        return None

    def _run_onnx_inference(
        self, name: str, inputs: Dict[str, np.ndarray], timeout: float
    ) -> Dict[str, np.ndarray]:
        """Run ONNX inference."""
        session = self.sessions[name]

        # Prepare inputs
        feed_dict = {}
        for input_name, input_data in inputs.items():
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.detach().cpu().numpy()
            elif not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data)

            feed_dict[input_name] = input_data

        # Run inference
        outputs = session.run(None, feed_dict)

        # Convert to dictionary
        output_names = [out.name for out in session.get_outputs()]
        result = {name: output for name, output in zip(output_names, outputs)}

        return result

    def _run_fallback_inference(
        self, name: str, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Run PyTorch fallback inference."""
        model = self.fallback_models[name]

        if TORCH_AVAILABLE and hasattr(model, "__call__"):
            # Convert inputs to tensors
            torch_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, np.ndarray):
                    torch_inputs[key] = torch.from_numpy(value)
                else:
                    torch_inputs[key] = torch.tensor(value)

            # Run model
            with torch.no_grad():
                if hasattr(model, "forward"):
                    # Single input case
                    if len(torch_inputs) == 1:
                        output = model(list(torch_inputs.values())[0])
                    else:
                        output = model(**torch_inputs)
                else:
                    output = model(torch_inputs)

            # Convert output to numpy
            if isinstance(output, torch.Tensor):
                return {"output": output.detach().cpu().numpy()}
            elif isinstance(output, dict):
                return {
                    k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in output.items()
                }
            else:
                return {"output": np.array(output)}

        return {}

    def batch_inference(
        self, name: str, batch_inputs: List[Dict[str, np.ndarray]]
    ) -> List[Optional[Dict[str, np.ndarray]]]:
        """Run batch inference."""
        results = []

        for inputs in batch_inputs:
            result = self.run_inference(name, inputs)
            results.append(result)

        return results

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return self.model_info.get(name)

    def get_inference_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get inference statistics."""
        return self.inference_stats.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all inference statistics."""
        return self.inference_stats.copy()

    def is_model_loaded(self, name: str) -> bool:
        """Check if model is loaded."""
        return name in self.sessions or name in self.fallback_models

    def unload_model(self, name: str) -> bool:
        """Unload model to free memory."""
        unloaded = False

        if name in self.sessions:
            del self.sessions[name]
            unloaded = True

        if name in self.fallback_models:
            del self.fallback_models[name]
            unloaded = True

        if name in self.model_info:
            del self.model_info[name]

        if name in self.inference_stats:
            del self.inference_stats[name]

        if unloaded:
            logger.info(f"Unloaded model '{name}'")

        return unloaded

    def list_models(self) -> List[str]:
        """List loaded models."""
        onnx_models = set(self.sessions.keys())
        fallback_models = set(self.fallback_models.keys())
        return list(onnx_models.union(fallback_models))


# Global runner instance
_runner: Optional[ONNXModelRunner] = None


def get_runner() -> ONNXModelRunner:
    """Get global ONNX runner instance."""
    global _runner
    if _runner is None:
        _runner = ONNXModelRunner()
    return _runner


def load_model(
    name: str, path: Union[str, Path], fallback_model: Optional[Any] = None
) -> bool:
    """Load model into global runner."""
    runner = get_runner()
    return runner.load_model(name, path, fallback_model)


def run_inference(
    name: str,
    inputs: Dict[str, Union[np.ndarray, torch.Tensor, List]],
    timeout: float = 5.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Run inference using global runner."""
    runner = get_runner()

    # Convert inputs to numpy arrays
    np_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            np_inputs[key] = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)):
            np_inputs[key] = np.array(value)
        else:
            np_inputs[key] = np.asarray(value)

    return runner.run_inference(name, np_inputs, timeout)


def get_model_stats(name: str) -> Optional[Dict[str, Any]]:
    """Get model statistics."""
    runner = get_runner()
    return runner.get_inference_stats(name)


def get_all_model_stats() -> Dict[str, Dict[str, Any]]:
    """Get all model statistics."""
    runner = get_runner()
    return runner.get_all_stats()


# Convenience functions for specific model types
def load_patchtst_model(
    path: str = "models/patchtst_small.onnx", fallback_model: Optional[Any] = None
) -> bool:
    """Load PatchTST model."""
    return load_model("patchtst", path, fallback_model)


def load_tlob_model(
    path: str = "models/tlob_tiny.onnx", fallback_model: Optional[Any] = None
) -> bool:
    """Load TLOB (Time-aware LOB) model."""
    return load_model("tlob", path, fallback_model)


def run_patchtst_inference(inputs: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Run PatchTST inference."""
    return run_inference("patchtst", inputs)


def run_tlob_inference(inputs: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Run TLOB inference."""
    return run_inference("tlob", inputs)


# Model discovery and auto-loading
def discover_and_load_models(
    model_dir: str = "models/", patterns: List[str] = None
) -> Dict[str, bool]:
    """Discover and load ONNX models from directory."""
    if patterns is None:
        patterns = ["*.onnx"]

    model_dir = Path(model_dir)
    results = {}

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return results

    for pattern in patterns:
        for model_path in model_dir.glob(pattern):
            model_name = model_path.stem
            success = load_model(model_name, model_path)
            results[model_name] = success

            if success:
                logger.info(f"Auto-loaded model: {model_name}")
            else:
                logger.warning(f"Failed to auto-load model: {model_name}")

    return results


# Health check and diagnostics
def health_check() -> Dict[str, Any]:
    """Perform health check on ONNX runner."""
    runner = get_runner()

    health = {
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "providers": runner.providers if ONNXRUNTIME_AVAILABLE else [],
        "loaded_models": runner.list_models(),
        "model_count": len(runner.list_models()),
        "total_inference_calls": sum(
            stats.get("total_calls", 0) for stats in runner.inference_stats.values()
        ),
        "avg_latency_ms": (
            np.mean(
                [
                    stats.get("avg_latency_ms", 0)
                    for stats in runner.inference_stats.values()
                ]
            )
            if runner.inference_stats
            else 0.0
        ),
        "error_rate": sum(
            stats.get("error_count", 0) for stats in runner.inference_stats.values()
        )
        / max(
            1,
            sum(
                stats.get("total_calls", 0) for stats in runner.inference_stats.values()
            ),
        ),
    }

    return health
