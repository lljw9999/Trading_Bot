#!/usr/bin/env python3
"""
Triton Python Backend Router Model

Implements intelligent routing between TLOB-Tiny and PatchTST-Small models
based on symbol characteristics and time horizon requirements.

Routing Logic:
- TLOB-Tiny: High-frequency symbols (BTC, ETH) with short horizons (<= 5 minutes)
- PatchTST-Small: All symbols with medium/long horizons (> 5 minutes)
- Fallback: TLOB-Tiny for unknown symbols
"""

import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os # Added for file existence check

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    # For development/testing outside Triton
    pb_utils = None

# Model routing configuration
ROUTING_CONFIG = {
    # High-frequency symbols prefer TLOB for short horizons
    "hf_symbols": {"BTC-USD", "ETH-USD", "BTC", "ETH"},
    
    # Time horizon thresholds (in minutes)
    "short_horizon_threshold": 5,
    "medium_horizon_threshold": 60,
    
    # Model capabilities
    "tlob_max_sequence": 32,
    "patchtst_max_sequence": 96,
    
    # Confidence thresholds
    "high_confidence_threshold": 0.8,
    "medium_confidence_threshold": 0.6
}


class TritonPythonModel:
    """Triton Python backend router model."""
    
    def initialize(self, args: Dict[str, Any]) -> None:
        """
        Initialize the router model.
        
        Args:
            args: Dictionary containing initialization arguments
        """
        self.model_config = args['model_config']
        self.model_instance_kind = args['model_instance_kind']
        self.model_instance_device_id = args['model_instance_device_id']
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize ONNX models
        self._load_onnx_models()
        
        # Routing statistics
        self.routing_stats = {
            "total_requests": 0,
            "tlob_requests": 0,
            "patchtst_requests": 0,
            "routing_latency_ms": []
        }
        
        self.logger.info("Router model initialized successfully")
    
    def _load_onnx_models(self) -> None:
        """Load ONNX models for inference."""
        try:
            import onnxruntime as ort
            
            # Model paths (adjust based on deployment environment)
            # Try multiple possible paths
            model_path_candidates = [
                "/models/tlob_tiny_int8.onnx",  # Triton deployment path
                "models/tlob_tiny_int8.onnx",   # Local development path
                "/onnx_models/tlob_tiny_int8.onnx"  # Docker volume mount
            ]
            
            tlob_path = None
            patchtst_path = None
            
            for base_path in model_path_candidates:
                tlob_candidate = base_path
                patchtst_candidate = base_path.replace("tlob_tiny", "patchtst_small")
                
                if os.path.exists(tlob_candidate) and os.path.exists(patchtst_candidate):
                    tlob_path = tlob_candidate
                    patchtst_path = patchtst_candidate
                    break
            
            if tlob_path is None or patchtst_path is None:
                self.logger.warning("ONNX model files not found, using mock inference")
                self.tlob_session = None
                self.patchtst_session = None
                return
            
            # Create ONNX sessions
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.tlob_session = ort.InferenceSession(tlob_path, sess_options=session_options)
            self.patchtst_session = ort.InferenceSession(patchtst_path, sess_options=session_options)
            
            self.logger.info(f"ONNX models loaded successfully from {tlob_path} and {patchtst_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX models: {e}")
            # Fallback to mock inference for development
            self.tlob_session = None
            self.patchtst_session = None
    
    def execute(self, requests: List[Any]) -> List[Any]:
        """
        Execute inference requests with routing logic.
        
        Args:
            requests: List of inference requests
        
        Returns:
            List of inference responses
        """
        responses = []
        
        for request in requests:
            try:
                routing_start = time.time()
                
                # Extract inputs
                symbol = self._get_input_tensor(request, "symbol")
                features = self._get_input_tensor(request, "features")
                time_horizon = self._get_input_tensor(request, "time_horizon")
                
                # Route request to appropriate model
                model_choice, prediction, confidence = self._route_and_predict(
                    symbol, features, time_horizon
                )
                
                # Update statistics
                self._update_stats(model_choice, time.time() - routing_start)
                
                # Create response
                response = self._create_response(model_choice, prediction, confidence)
                responses.append(response)
                
            except Exception as e:
                self.logger.error(f"Request execution failed: {e}")
                # Return error response
                error_response = self._create_error_response(str(e))
                responses.append(error_response)
        
        return responses
    
    def _route_and_predict(self, 
                          symbol: str, 
                          features: np.ndarray, 
                          time_horizon: int) -> Tuple[str, np.ndarray, float]:
        """
        Route request to appropriate model and generate prediction.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            features: Input feature array
            time_horizon: Prediction time horizon in minutes
        
        Returns:
            Tuple of (model_choice, prediction, confidence)
        """
        # Apply routing logic
        model_choice = self._select_model(symbol, time_horizon, features.shape)
        
        # Generate prediction based on model choice
        if model_choice == "tlob_tiny":
            prediction, confidence = self._predict_tlob(features)
        elif model_choice == "patchtst_small":
            prediction, confidence = self._predict_patchtst(features)
        else:
            # Fallback prediction
            prediction = np.array([0.0], dtype=np.float32)
            confidence = 0.5
        
        # Update statistics
        self._update_stats(model_choice, 0.0)  # 0 latency for internal calls
        
        # Ensure confidence is Python float
        confidence = float(confidence)
        
        return model_choice, prediction, confidence
    
    def _select_model(self, symbol: str, time_horizon: int, feature_shape: Tuple) -> str:
        """
        Select the best model based on symbol and time horizon.
        
        Args:
            symbol: Trading symbol
            time_horizon: Time horizon in minutes
            feature_shape: Shape of input features
        
        Returns:
            Model name to use for prediction
        """
        symbol_clean = symbol.upper().strip()
        sequence_length = feature_shape[0] if len(feature_shape) >= 2 else 32
        
        # Long sequences or long horizons → PatchTST (check this first)
        if (sequence_length > ROUTING_CONFIG["tlob_max_sequence"] or
            time_horizon > ROUTING_CONFIG["medium_horizon_threshold"]):
            return "patchtst_small"
        
        # High-frequency symbols with short horizons → TLOB
        if (symbol_clean in ROUTING_CONFIG["hf_symbols"] and 
            time_horizon <= ROUTING_CONFIG["short_horizon_threshold"]):
            return "tlob_tiny"
        
        # Medium horizons → PatchTST (better for forecasting)
        if time_horizon > ROUTING_CONFIG["short_horizon_threshold"]:
            return "patchtst_small"
        
        # Default to TLOB for short-term, high-frequency trading
        return "tlob_tiny"
    
    def _predict_tlob(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate prediction using TLOB-Tiny model.
        
        Args:
            features: Input features
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if self.tlob_session is None:
                # Mock prediction for development
                pressure = np.random.uniform(-0.5, 0.5)
                return np.array([pressure], dtype=np.float32), 0.7
            
            # Prepare input for TLOB (expects shape: [batch, seq_len, features])
            if len(features.shape) == 2:
                features = features[np.newaxis, :, :]  # Add batch dimension
            
            # Truncate or pad to expected sequence length
            target_seq_len = ROUTING_CONFIG["tlob_max_sequence"]
            if features.shape[1] > target_seq_len:
                features = features[:, -target_seq_len:, :]  # Take last N timesteps
            elif features.shape[1] < target_seq_len:
                # Pad with zeros
                pad_length = target_seq_len - features.shape[1]
                padding = np.zeros((features.shape[0], pad_length, features.shape[2]), dtype=np.float32)
                features = np.concatenate([padding, features], axis=1)
            
            # Ensure correct feature count (TLOB expects 10 features)
            if features.shape[2] != 10:
                if features.shape[2] > 10:
                    features = features[:, :, :10]  # Take first 10 features
                else:
                    # Pad features
                    pad_features = 10 - features.shape[2]
                    padding = np.zeros((features.shape[0], features.shape[1], pad_features), dtype=np.float32)
                    features = np.concatenate([features, padding], axis=2)
            
            # Run inference
            input_name = self.tlob_session.get_inputs()[0].name
            ort_inputs = {input_name: features}
            outputs = self.tlob_session.run(None, ort_inputs)
            
            pressure = outputs[0][0, 0]  # Extract scalar pressure value
            confidence = min(0.9, 0.6 + 0.3 * abs(pressure))  # Higher confidence for stronger signals
            
            return np.array([pressure], dtype=np.float32), float(confidence)
            
        except Exception as e:
            self.logger.error(f"TLOB prediction failed: {e}")
            return np.array([0.0], dtype=np.float32), 0.3
    
    def _predict_patchtst(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate prediction using PatchTST-Small model.
        
        Args:
            features: Input features
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            if self.patchtst_session is None:
                # Mock prediction for development
                forecast = np.random.uniform(-0.1, 0.1, size=8)
                return forecast.astype(np.float32), 0.6
            
            # Prepare input for PatchTST (expects shape: [batch, seq_len, features])
            if len(features.shape) == 2:
                features = features[np.newaxis, :, :]  # Add batch dimension
            
            # Truncate or pad to expected sequence length
            target_seq_len = ROUTING_CONFIG["patchtst_max_sequence"]
            if features.shape[1] > target_seq_len:
                features = features[:, -target_seq_len:, :]  # Take last N timesteps
            elif features.shape[1] < target_seq_len:
                # Pad with zeros
                pad_length = target_seq_len - features.shape[1]
                padding = np.zeros((features.shape[0], pad_length, features.shape[2]), dtype=np.float32)
                features = np.concatenate([padding, features], axis=1)
            
            # Ensure correct feature count (PatchTST expects 5 features)
            if features.shape[2] != 5:
                if features.shape[2] > 5:
                    features = features[:, :, :5]  # Take first 5 features
                else:
                    # Pad features
                    pad_features = 5 - features.shape[2]
                    padding = np.zeros((features.shape[0], features.shape[1], pad_features), dtype=np.float32)
                    features = np.concatenate([features, padding], axis=2)
            
            # Run inference
            input_name = self.patchtst_session.get_inputs()[0].name
            ort_inputs = {input_name: features}
            outputs = self.patchtst_session.run(None, ort_inputs)
            
            forecast = outputs[0][0]  # Shape: [pred_len, n_features] -> [8, 5]
            
            # Flatten forecast and compute confidence based on variance
            forecast_flat = forecast.flatten()
            forecast_var = np.var(forecast_flat)
            confidence = max(0.4, min(0.9, 0.8 - forecast_var))
            
            return forecast_flat.astype(np.float32), float(confidence)
            
        except Exception as e:
            self.logger.error(f"PatchTST prediction failed: {e}")
            return np.zeros(40, dtype=np.float32), 0.3  # 8 * 5 = 40 values
    
    def _get_input_tensor(self, request: Any, name: str) -> Any:
        """Extract input tensor from request."""
        if pb_utils is None:
            # Mock for development
            if name == "symbol":
                return "BTC-USD"
            elif name == "time_horizon":
                return 3
            elif name == "features":
                return np.random.randn(32, 10).astype(np.float32)
        
        input_tensor = pb_utils.get_input_tensor_by_name(request, name)
        if name == "symbol":
            return input_tensor.as_numpy()[0].decode('utf-8')
        elif name == "time_horizon":
            return int(input_tensor.as_numpy()[0])
        else:
            return input_tensor.as_numpy()
    
    def _create_response(self, 
                        model_choice: str, 
                        prediction: np.ndarray, 
                        confidence: float) -> Any:
        """Create inference response."""
        if pb_utils is None:
            # Mock for development
            return {
                "model_choice": model_choice,
                "prediction": prediction.tolist(),
                "confidence": confidence
            }
        
        # Create output tensors
        model_choice_tensor = pb_utils.Tensor("model_choice", 
                                            np.array([model_choice], dtype=object))
        prediction_tensor = pb_utils.Tensor("prediction", prediction)
        confidence_tensor = pb_utils.Tensor("confidence", 
                                          np.array([confidence], dtype=np.float32))
        
        return pb_utils.InferenceResponse(output_tensors=[
            model_choice_tensor, prediction_tensor, confidence_tensor
        ])
    
    def _create_error_response(self, error_msg: str) -> Any:
        """Create error response."""
        if pb_utils is None:
            return {"error": error_msg}
        
        return pb_utils.InferenceResponse(
            error=pb_utils.TritonError(error_msg)
        )
    
    def _update_stats(self, model_choice: str, latency_ms: float) -> None:
        """Update routing statistics."""
        self.routing_stats["total_requests"] += 1
        if model_choice == "tlob_tiny":
            self.routing_stats["tlob_requests"] += 1
        elif model_choice == "patchtst_small":
            self.routing_stats["patchtst_requests"] += 1
        
        self.routing_stats["routing_latency_ms"].append(latency_ms * 1000)
        
        # Keep only last 1000 latency measurements
        if len(self.routing_stats["routing_latency_ms"]) > 1000:
            self.routing_stats["routing_latency_ms"] = self.routing_stats["routing_latency_ms"][-1000:]
    
    def finalize(self) -> None:
        """Clean up resources."""
        # Log final statistics
        total = self.routing_stats["total_requests"]
        if total > 0:
            tlob_pct = 100 * self.routing_stats["tlob_requests"] / total
            patchtst_pct = 100 * self.routing_stats["patchtst_requests"] / total
            
            if self.routing_stats["routing_latency_ms"]:
                avg_latency = np.mean(self.routing_stats["routing_latency_ms"])
                p95_latency = np.percentile(self.routing_stats["routing_latency_ms"], 95)
            else:
                avg_latency = p95_latency = 0
            
            self.logger.info(f"Router statistics:")
            self.logger.info(f"  Total requests: {total}")
            self.logger.info(f"  TLOB routing: {tlob_pct:.1f}%")
            self.logger.info(f"  PatchTST routing: {patchtst_pct:.1f}%")
            self.logger.info(f"  Avg latency: {avg_latency:.2f}ms")
            self.logger.info(f"  P95 latency: {p95_latency:.2f}ms")
        
        self.logger.info("Router model finalized") 