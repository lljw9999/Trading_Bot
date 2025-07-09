#!/usr/bin/env python3
"""
ONNX Model Runner for Alpha Diversification

Provides a high-performance ONNX inference runner for transformer models
with optimized loading, warm-up, and inference capabilities.

Key Features:
- Model loading with provider optimization (CPU/GPU)
- Warm-up for consistent latency
- Batched and single inference
- Memory-efficient session management
- Latency monitoring and metrics
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import threading

import numpy as np
import onnxruntime as ort

# Setup logging
logger = logging.getLogger(__name__)


class ONNXRunner:
    """
    High-performance ONNX model runner optimized for trading inference.
    
    Supports multiple models with efficient session management and 
    sub-millisecond inference latency.
    """
    
    def __init__(self, 
                 providers: Optional[List[str]] = None,
                 session_options: Optional[ort.SessionOptions] = None):
        """
        Initialize ONNX runner with optimization settings.
        
        Args:
            providers: List of execution providers (e.g., ['CPUExecutionProvider'])
            session_options: Custom session options for optimization
        """
        self.models: Dict[str, ort.InferenceSession] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.warm_up_cache: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        
        # Configure execution providers
        if providers is None:
            # Auto-detect best providers
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers
        
        # Configure session options for optimal performance
        if session_options is None:
            self.session_options = ort.SessionOptions()
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            self.session_options.inter_op_num_threads = 1
            self.session_options.enable_mem_pattern = True
            self.session_options.enable_cpu_mem_arena = True
        else:
            self.session_options = session_options
        
        logger.info(f"ONNXRunner initialized with providers: {self.providers}")
    
    def load_model(self, 
                   model_path: str, 
                   model_name: str,
                   warm_up: bool = True) -> bool:
        """
        Load an ONNX model into memory with optimization.
        
        Args:
            model_path: Path to the .onnx model file
            model_name: Unique identifier for the model
            warm_up: Whether to perform warm-up inference
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            with self._lock:
                if model_name in self.models:
                    logger.warning(f"Model {model_name} already loaded, skipping")
                    return True
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    return False
                
                logger.info(f"Loading ONNX model: {model_name} from {model_path}")
                start_time = time.time()
                
                # Create inference session
                session = ort.InferenceSession(
                    model_path,
                    sess_options=self.session_options,
                    providers=self.providers
                )
                
                # Extract model metadata
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()[0]
                
                metadata = {
                    'path': model_path,
                    'input_name': input_info.name,
                    'input_shape': input_info.shape,
                    'input_type': input_info.type,
                    'output_name': output_info.name,
                    'output_shape': output_info.shape,
                    'output_type': output_info.type,
                    'load_time': time.time() - start_time
                }
                
                self.models[model_name] = session
                self.model_metadata[model_name] = metadata
                
                # Estimate memory usage (rough approximation)
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                logger.info(f"✅ Model {model_name} loaded successfully:")
                logger.info(f"   - Load time: {metadata['load_time']:.3f}s")
                logger.info(f"   - Input: {metadata['input_name']} {metadata['input_shape']}")
                logger.info(f"   - Output: {metadata['output_name']} {metadata['output_shape']}")
                logger.info(f"   - Size: {model_size_mb:.2f} MB")
                
                # Perform warm-up
                if warm_up:
                    self.warm_up_model(model_name)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def warm_up_model(self, model_name: str, num_iterations: int = 5) -> bool:
        """
        Warm up a model with dummy inference to optimize JIT compilation.
        
        Args:
            model_name: Name of the model to warm up
            num_iterations: Number of warm-up iterations
        
        Returns:
            True if warm-up successful, False otherwise
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not loaded")
                return False
            
            metadata = self.model_metadata[model_name]
            session = self.models[model_name]
            
            # Create dummy input based on model input shape
            input_shape = metadata['input_shape']
            
            # Handle dynamic batch size and convert string dimensions to integers
            fixed_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim == -1 or dim is None:
                    # Dynamic dimension - use default size
                    if len(fixed_shape) == 0:  # Batch dimension
                        fixed_shape.append(1)
                    else:
                        # For other dynamic dimensions, use model-specific defaults
                        if model_name == "tlob_tiny":
                            if len(fixed_shape) == 1:  # Sequence length
                                fixed_shape.append(32)
                            else:  # Features
                                fixed_shape.append(10)
                        elif model_name == "patchtst_small":
                            if len(fixed_shape) == 1:  # Sequence length  
                                fixed_shape.append(96)
                            else:  # Features
                                fixed_shape.append(5)
                        else:
                            fixed_shape.append(10)  # Generic default
                else:
                    fixed_shape.append(int(dim))
            
            dummy_input = np.random.randn(*fixed_shape).astype(np.float32)
            self.warm_up_cache[model_name] = dummy_input
            
            logger.info(f"Warming up model {model_name} with {num_iterations} iterations...")
            
            warm_up_times = []
            for i in range(num_iterations):
                start_time = time.time()
                
                # Run inference
                ort_inputs = {metadata['input_name']: dummy_input}
                _ = session.run(None, ort_inputs)
                
                latency = (time.time() - start_time) * 1000  # Convert to ms
                warm_up_times.append(latency)
            
            avg_latency = np.mean(warm_up_times)
            min_latency = np.min(warm_up_times)
            
            logger.info(f"✅ Warm-up complete for {model_name}:")
            logger.info(f"   - Average latency: {avg_latency:.2f} ms")
            logger.info(f"   - Best latency: {min_latency:.2f} ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to warm up model {model_name}: {e}")
            return False
    
    def predict(self, 
                model_name: str, 
                input_data: np.ndarray,
                return_latency: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Run inference on a loaded model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input numpy array
            return_latency: Whether to return inference latency
        
        Returns:
            Model output (and optionally latency in ms)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        try:
            session = self.models[model_name]
            metadata = self.model_metadata[model_name]
            
            # Prepare input
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            ort_inputs = {metadata['input_name']: input_data}
            
            # Run inference with timing
            start_time = time.time()
            
            outputs = session.run(None, ort_inputs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            if return_latency:
                return outputs[0], latency_ms
            else:
                return outputs[0]
                
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            raise
    
    def predict_batch(self, 
                      model_name: str, 
                      input_batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run batch inference on multiple inputs.
        
        Args:
            model_name: Name of the model to use
            input_batch: List of input arrays
        
        Returns:
            List of model outputs
        """
        if not input_batch:
            return []
        
        # Stack inputs into batch
        batch_input = np.stack(input_batch, axis=0)
        
        # Run inference
        batch_output = self.predict(model_name, batch_input)
        
        # Split outputs back into list
        return [batch_output[i] for i in range(len(input_batch))]
    
    def benchmark_model(self, 
                       model_name: str, 
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            model_name: Name of the model to benchmark
            num_iterations: Number of benchmark iterations
        
        Returns:
            Dictionary with performance statistics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        if model_name not in self.warm_up_cache:
            self.warm_up_model(model_name)
        
        dummy_input = self.warm_up_cache[model_name]
        latencies = []
        
        logger.info(f"Benchmarking {model_name} with {num_iterations} iterations...")
        
        for i in range(num_iterations):
            _, latency = self.predict(model_name, dummy_input, return_latency=True)
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        stats = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'std_ms': float(np.std(latencies))
        }
        
        logger.info(f"📊 Benchmark results for {model_name}:")
        logger.info(f"   - Mean: {stats['mean_ms']:.2f} ms")
        logger.info(f"   - Median: {stats['median_ms']:.2f} ms")
        logger.info(f"   - P95: {stats['p95_ms']:.2f} ms")
        logger.info(f"   - Min: {stats['min_ms']:.2f} ms")
        logger.info(f"   - Max: {stats['max_ms']:.2f} ms")
        
        return stats
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a loaded model."""
        return self.model_metadata.get(model_name)
    
    def list_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        
        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            with self._lock:
                if model_name not in self.models:
                    logger.warning(f"Model {model_name} not loaded")
                    return False
                
                del self.models[model_name]
                del self.model_metadata[model_name]
                
                if model_name in self.warm_up_cache:
                    del self.warm_up_cache[model_name]
                
                logger.info(f"✅ Model {model_name} unloaded")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def clear_all_models(self) -> None:
        """Unload all models from memory."""
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.unload_model(model_name)


class ModelManager:
    """
    High-level model manager for alpha models.
    
    Provides a simplified interface for loading and using
    TLOB-Tiny and PatchTST-Small models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory containing ONNX model files
        """
        self.models_dir = Path(models_dir)
        self.runner = ONNXRunner()
        self.loaded_models = set()
        
        logger.info(f"ModelManager initialized with models_dir: {models_dir}")
    
    def load_alpha_models(self, quantized: bool = True) -> bool:
        """
        Load all alpha models (TLOB-Tiny and PatchTST-Small).
        
        Args:
            quantized: Whether to load quantized (INT8) versions
        
        Returns:
            True if all models loaded successfully
        """
        suffix = "_int8.onnx" if quantized else ".onnx"
        
        models_to_load = [
            ("tlob_tiny", f"tlob_tiny{suffix}"),
            ("patchtst_small", f"patchtst_small{suffix}")
        ]
        
        success_count = 0
        
        for model_name, filename in models_to_load:
            model_path = self.models_dir / filename
            
            if self.runner.load_model(str(model_path), model_name):
                self.loaded_models.add(model_name)
                success_count += 1
            else:
                logger.error(f"Failed to load {model_name}")
        
        logger.info(f"Loaded {success_count}/{len(models_to_load)} alpha models")
        return success_count == len(models_to_load)
    
    def predict_order_book_pressure(self, features: np.ndarray) -> float:
        """
        Predict order book pressure using TLOB-Tiny model.
        
        Args:
            features: Order book features array (seq_len, n_features)
        
        Returns:
            Pressure score between -1 and 1
        """
        if "tlob_tiny" not in self.loaded_models:
            raise RuntimeError("TLOB-Tiny model not loaded")
        
        # Add batch dimension if needed
        if features.ndim == 2:
            features = features[np.newaxis, :]
        
        pressure = self.runner.predict("tlob_tiny", features)
        return float(pressure[0, 0])  # Extract scalar value
    
    def predict_price_forecast(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Predict price forecast using PatchTST-Small model.
        
        Args:
            timeseries: Time series data (seq_len, n_features)
        
        Returns:
            Forecast array (pred_len, n_features)
        """
        if "patchtst_small" not in self.loaded_models:
            raise RuntimeError("PatchTST-Small model not loaded")
        
        # Add batch dimension if needed
        if timeseries.ndim == 2:
            timeseries = timeseries[np.newaxis, :]
        
        forecast = self.runner.predict("patchtst_small", timeseries)
        return forecast[0]  # Remove batch dimension
    
    def benchmark_all_models(self) -> Dict[str, Dict[str, float]]:
        """Benchmark all loaded models."""
        results = {}
        for model_name in self.loaded_models:
            results[model_name] = self.runner.benchmark_model(model_name)
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all loaded models."""
        return {
            'loaded_models': list(self.loaded_models),
            'total_models': len(self.loaded_models),
            'models_dir': str(self.models_dir),
            'model_info': {name: self.runner.get_model_info(name) 
                          for name in self.loaded_models}
        }


# Global model manager instance
_model_manager = None

def get_model_manager(models_dir: str = "models") -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(models_dir)
    return _model_manager 