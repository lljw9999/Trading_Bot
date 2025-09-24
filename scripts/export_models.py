#!/usr/bin/env python3
"""
Model Export Script for Alpha Diversification

Exports TLOB-Tiny & PatchTST-Small transformer models to ONNX format
with INT8 quantization for production inference.

Usage:
    python scripts/export_models.py --model tlob_tiny --quant int8
    python scripts/export_models.py --model patchtst_small --quant int8
    python scripts/export_models.py --all --quant int8
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic as onnx_quantize_dynamic,
    QuantType,
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TLOBTinyModel(nn.Module):
    """
    TLOB-Tiny: Transformer for Limit Order Book prediction.

    A lightweight transformer model designed for high-frequency trading
    order book pressure prediction with minimal latency.
    """

    def __init__(
        self,
        seq_len: int = 32,
        n_features: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Simple self-attention layers (ONNX-compatible)
        self.attention_layers = nn.ModuleList(
            [SimpleAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output head for order book pressure prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # Single value: pressure score [-1, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for order book pressure prediction.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Features: [bid_price, ask_price, bid_size, ask_size, mid, spread, ...]

        Returns:
            pressure: Tensor of shape (batch_size, 1) with pressure scores
        """
        batch_size, seq_len, _ = x.shape

        # Input projection and positional encoding
        x = self.input_projection(x)  # (B, T, d_model)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)  # Add positional encoding

        # Self-attention layers
        for layer in self.attention_layers:
            x = layer(x)

        # Global average pooling over sequence dimension
        pooled = x.mean(dim=1)  # (B, d_model)

        # Output prediction
        pressure = self.output_head(pooled)  # (B, 1)
        pressure = torch.tanh(pressure)  # Constrain to [-1, 1]

        return pressure


class SimpleAttentionLayer(nn.Module):
    """ONNX-compatible self-attention layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query, Key, Value projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)

        # Multi-head attention (simplified for ONNX)
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        attn_output = self.w_o(attn_output)
        x = residual + self.dropout(attn_output)

        # Feed forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


class PatchTSTSmallModel(nn.Module):
    """
    PatchTST-Small: Patching Time Series Transformer for forecasting.

    A patch-based transformer for multi-horizon time series forecasting
    optimized for financial market prediction.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 8,
        n_features: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_len: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(self.n_patches, d_model))

        # Simple attention layers (ONNX-compatible)
        self.attention_layers = nn.ModuleList(
            [SimpleAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * n_features),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time series forecasting.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
               Features: [price, volume, volatility, momentum, sentiment]

        Returns:
            forecast: Tensor of shape (batch_size, pred_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape

        # Create patches: (B, seq_len, n_features) -> (B, n_patches, patch_len * n_features)
        x_patches = x.view(batch_size, self.n_patches, self.patch_len * n_features)

        # Patch embedding
        embedded = self.patch_embedding(x_patches)  # (B, n_patches, d_model)
        embedded = embedded + self.pos_encoding.unsqueeze(0)  # Add positional encoding

        # Self-attention layers
        for layer in self.attention_layers:
            embedded = layer(embedded)

        # Global average pooling
        pooled = embedded.mean(dim=1)  # (B, d_model)

        # Forecast generation
        forecast_flat = self.forecast_head(pooled)  # (B, pred_len * n_features)
        forecast = forecast_flat.view(batch_size, self.pred_len, n_features)

        return forecast


def create_model(model_name: str) -> nn.Module:
    """Create a model instance based on the model name."""
    if model_name == "tlob_tiny":
        return TLOBTinyModel()
    elif model_name == "patchtst_small":
        return PatchTSTSmallModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def export_to_onnx(
    model: nn.Module,
    model_name: str,
    output_dir: str = "models",
    quantization: str = "int8",
) -> str:
    """
    Export PyTorch model to ONNX format with optional quantization.

    Args:
        model: PyTorch model to export
        model_name: Name of the model for file naming
        output_dir: Directory to save the ONNX model
        quantization: Quantization type ('int8' or 'fp16')

    Returns:
        Path to the exported ONNX model
    """
    logger.info(f"Exporting {model_name} to ONNX...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input based on model type
    if model_name == "tlob_tiny":
        dummy_input = torch.randn(1, 32, 10)  # (batch_size, seq_len, n_features)
        input_names = ["orderbook_features"]
        output_names = ["pressure_score"]
        dynamic_axes = {
            "orderbook_features": {0: "batch_size"},
            "pressure_score": {0: "batch_size"},
        }
    elif model_name == "patchtst_small":
        dummy_input = torch.randn(1, 96, 5)  # (batch_size, seq_len, n_features)
        input_names = ["timeseries_features"]
        output_names = ["forecast"]
        dynamic_axes = {
            "timeseries_features": {0: "batch_size"},
            "forecast": {0: "batch_size"},
        }
    else:
        raise ValueError(f"Unknown model for dummy input: {model_name}")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

    logger.info(f"✅ ONNX export complete: {onnx_path}")

    # Apply quantization if requested
    if quantization == "int8":
        quantized_path = os.path.join(output_dir, f"{model_name}_int8.onnx")
        logger.info(f"Applying INT8 quantization...")

        onnx_quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Add", "Mul"],
        )

        logger.info(f"✅ Quantized model saved: {quantized_path}")

        # Verify quantized model
        verify_onnx_model(quantized_path, dummy_input)

        return quantized_path
    else:
        # Verify original model
        verify_onnx_model(onnx_path, dummy_input)
        return onnx_path


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor) -> None:
    """Verify that the ONNX model can be loaded and run inference."""
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Test inference
        ort_session = ort.InferenceSession(onnx_path)

        # Prepare input
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.numpy()}

        # Run inference
        start_time = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        output_shape = ort_outputs[0].shape
        logger.info(f"✅ Model verification passed:")
        logger.info(f"   - Input shape: {dummy_input.shape}")
        logger.info(f"   - Output shape: {output_shape}")
        logger.info(f"   - Inference time: {inference_time:.2f} ms")

        return True

    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False


def get_model_info(model: nn.Module) -> dict:
    """Get model parameter count and size information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Export transformer models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tlob_tiny", "patchtst_small"],
        help="Model to export",
    )
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument(
        "--quant",
        type=str,
        default="int8",
        choices=["int8", "fp16", "none"],
        help="Quantization type",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX models",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which models to export
    models_to_export = []
    if args.all:
        models_to_export = ["tlob_tiny", "patchtst_small"]
    elif args.model:
        models_to_export = [args.model]
    else:
        parser.error("Must specify either --model or --all")

    logger.info("🚀 Starting model export process...")
    logger.info(f"Models to export: {models_to_export}")
    logger.info(f"Quantization: {args.quant}")
    logger.info(f"Output directory: {args.output_dir}")

    exported_models = []

    for model_name in models_to_export:
        try:
            logger.info(f"\n📦 Processing {model_name}...")

            # Create model
            model = create_model(model_name)

            # Print model info
            info = get_model_info(model)
            logger.info(
                f"Model info: {info['total_params']:,} params, "
                f"{info['model_size_mb']:.2f} MB"
            )

            # Export to ONNX
            exported_path = export_to_onnx(
                model, model_name, args.output_dir, args.quant
            )

            exported_models.append((model_name, exported_path))

        except Exception as e:
            logger.error(f"❌ Failed to export {model_name}: {e}")
            continue

    # Summary
    logger.info(f"\n🎉 Export Summary:")
    logger.info(f"Successfully exported {len(exported_models)} models:")
    for model_name, path in exported_models:
        logger.info(f"  ✅ {model_name}: {path}")

    if len(exported_models) > 0:
        logger.info("\n🎯 Next steps:")
        logger.info("1. Upload models to S3 bucket")
        logger.info("2. Test with src/models/onnx_runner.py")
        logger.info("3. Benchmark latency with real data")

    return len(exported_models) == len(models_to_export)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
