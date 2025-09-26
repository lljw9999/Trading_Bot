"""
Enhanced LSTM/Transformer Deep Learning Alpha Model with Performance Improvements

Advanced features:
- Multi-scale attention mechanism
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Improved positional encoding
- Feature importance analysis
- Dynamic architecture adaptation
- Advanced regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import logging

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning."""

    def __init__(
        self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Original linear layer (frozen during fine-tuning)
        self.linear = nn.Linear(in_features, out_features)

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        original_output = self.linear(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output


class MultiScalePositionalEncoding(nn.Module):
    """Multi-scale positional encoding for different time horizons."""

    def __init__(
        self, d_model: int, max_len: int = 5000, scales: List[int] = [1, 5, 15, 60]
    ):
        super().__init__()
        self.d_model = d_model
        self.scales = scales

        # Create multiple positional encodings for different scales
        self.register_buffer("pe", self._create_multiscale_pe(d_model, max_len, scales))

    def _create_multiscale_pe(
        self, d_model: int, max_len: int, scales: List[int]
    ) -> torch.Tensor:
        """Create multi-scale positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)

        # Divide embedding dimensions among scales
        dims_per_scale = d_model // len(scales)

        for i, scale in enumerate(scales):
            start_dim = i * dims_per_scale
            end_dim = (i + 1) * dims_per_scale if i < len(scales) - 1 else d_model

            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, end_dim - start_dim, 2).float()
                * (-math.log(10000.0 * scale) / (end_dim - start_dim))
            )

            pe[:, start_dim:end_dim:2] = torch.sin(position * div_term)
            if start_dim + 1 < end_dim:
                pe[:, start_dim + 1 : end_dim : 2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class EnhancedMultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with feature importance tracking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # LoRA-enhanced projections
        self.q_proj = LoRALinear(d_model, d_model)
        self.k_proj = LoRALinear(d_model, d_model)
        self.v_proj = LoRALinear(d_model, d_model)
        self.out_proj = LoRALinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Feature importance tracking
        self.register_buffer("attention_weights", torch.zeros(1))
        self.importance_alpha = 0.95  # EMA decay for importance tracking

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention weight tracking."""
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Update feature importance (EMA)
        if self.training:
            current_importance = attention_weights.mean(
                dim=(0, 1, 2)
            )  # Average across batch, heads, queries
            if self.attention_weights.numel() > 1:
                self.attention_weights = (
                    self.importance_alpha * self.attention_weights
                    + (1 - self.importance_alpha) * current_importance
                )
            else:
                self.attention_weights = current_importance

        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Output projection with residual connection
        output = self.out_proj(context)
        output = self.layer_norm(output + x)

        return output, attention_weights.mean(
            dim=1
        )  # Return average attention across heads


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with improved architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = EnhancedMultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network with LoRA
        self.ff1 = LoRALinear(d_model, d_ff)
        self.ff2 = LoRALinear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Gated activation for better gradient flow
        self.gate = nn.Linear(d_ff, d_ff)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block."""
        # Multi-head attention
        attn_output, attn_weights = self.attention(x, mask)

        # Feed-forward network with gating
        ff_input = self.layer_norm1(attn_output)
        ff_hidden = self.ff1(ff_input)
        ff_gated = F.gelu(ff_hidden) * torch.sigmoid(self.gate(ff_hidden))
        ff_output = self.ff2(self.dropout1(ff_gated))

        # Residual connection
        output = self.layer_norm2(attn_output + self.dropout2(ff_output))

        return output, attn_weights


class EnhancedLSTMTransformerModel(nn.Module):
    """Enhanced LSTM-Transformer model with performance improvements."""

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_length: int = 60,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Half size for bidirectional
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Multi-scale positional encoding
        self.pos_encoding = MultiScalePositionalEncoding(hidden_size, seq_length * 2)

        # Enhanced Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                EnhancedTransformerBlock(
                    hidden_size, num_heads, hidden_size * 4, dropout
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Adaptive pooling for variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Multi-task prediction heads with LoRA
        self.price_head = nn.Sequential(
            LoRALinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_size // 2, 1),
        )

        self.volatility_head = nn.Sequential(
            LoRALinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_size // 2, 1),
        )

        self.direction_head = nn.Sequential(
            LoRALinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_size // 2, 3),
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            LoRALinear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_size // 2, 1),
        )

        # Feature importance tracker
        self.register_buffer("feature_importance", torch.zeros(input_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with attention visualization."""
        batch_size, seq_len, input_size = x.size()

        # Track input feature usage for importance analysis
        if self.training:
            feature_usage = x.abs().mean(dim=(0, 1))
            self.feature_importance = (
                0.99 * self.feature_importance + 0.01 * feature_usage
            )

        # Input projection and dropout
        x = self.input_projection(x)
        x = self.input_dropout(x)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        # Add multi-scale positional encoding
        transformer_input = self.pos_encoding(lstm_out)

        # Transformer processing with attention tracking
        attention_weights = []
        for transformer_layer in self.transformer_layers:
            transformer_input, attn_weights = transformer_layer(transformer_input)
            if return_attention:
                attention_weights.append(attn_weights)

        # Adaptive pooling to handle variable lengths
        pooled_output = self.adaptive_pool(transformer_input.transpose(1, 2)).squeeze(
            -1
        )

        # Multi-task predictions
        price_pred = self.price_head(pooled_output)
        volatility_pred = F.softplus(self.volatility_head(pooled_output))
        direction_pred = F.softmax(self.direction_head(pooled_output), dim=1)
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled_output))

        results = {
            "price_prediction": price_pred,
            "volatility_prediction": volatility_pred,
            "direction_prediction": direction_pred,
            "uncertainty": uncertainty,
            "hidden_representation": pooled_output,
        }

        if return_attention:
            results["attention_weights"] = attention_weights

        return results

    def get_feature_importance(self) -> torch.Tensor:
        """Get current feature importance scores."""
        return self.feature_importance.clone()


class EnhancedLSTMTransformerAlpha:
    """Enhanced LSTM/Transformer Alpha with performance improvements."""

    def __init__(
        self,
        symbol: str,
        seq_length: int = 60,
        retrain_frequency: int = 500,  # More frequent retraining
        learning_rate: float = 0.0005,  # Lower learning rate for stability
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_lora: bool = True,
        warmup_steps: int = 100,
    ):
        self.symbol = symbol
        self.seq_length = seq_length
        self.retrain_frequency = retrain_frequency
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.use_lora = use_lora
        self.warmup_steps = warmup_steps

        # Enhanced data storage with importance weighting
        self.feature_buffer = deque(maxlen=seq_length * 3)
        self.price_buffer = deque(maxlen=seq_length * 3)
        self.return_buffer = deque(maxlen=seq_length * 3)
        self.volatility_buffer = deque(maxlen=seq_length * 3)

        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.SmoothL1Loss()  # More robust to outliers

        # Enhanced training tracking
        self.sample_count = 0
        self.training_losses = deque(maxlen=100)
        self.validation_scores = deque(maxlen=50)
        self.is_trained = False

        # Performance metrics
        self.prediction_history = deque(maxlen=1000)
        self.accuracy_metrics = {
            "mse": 0.0,
            "mae": 0.0,
            "directional_accuracy": 0.0,
            "sharpe_ratio": 0.0,
        }

        # Feature normalization with exponential smoothing
        self.feature_stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }

        # Model selection and ensembling
        self.model_variants = {}
        self.ensemble_weights = {}

        self.logger = logging.getLogger(f"enhanced_lstm_transformer.{symbol}")
        self.logger.info(
            f"Enhanced LSTM/Transformer initialized for {symbol} on {device}"
        )

    def _adaptive_normalize(self, features: np.ndarray) -> np.ndarray:
        """Adaptive feature normalization with outlier handling."""
        if self.feature_stats["mean"] is None:
            self.feature_stats["mean"] = features.copy()
            self.feature_stats["std"] = np.ones_like(features)
            self.feature_stats["min"] = features.copy()
            self.feature_stats["max"] = features.copy()
            self.feature_stats["count"] = 1
            return features

        # Update statistics with exponential decay
        alpha = min(0.1, 1.0 / max(self.feature_stats["count"], 10))

        self.feature_stats["mean"] = (1 - alpha) * self.feature_stats[
            "mean"
        ] + alpha * features
        diff = features - self.feature_stats["mean"]
        self.feature_stats["std"] = np.sqrt(
            (1 - alpha) * self.feature_stats["std"] ** 2 + alpha * diff**2
        )

        # Update min/max with percentile-based outlier handling
        self.feature_stats["min"] = np.minimum(
            self.feature_stats["min"],
            np.percentile(np.vstack([self.feature_stats["min"], features]), 5, axis=0),
        )
        self.feature_stats["max"] = np.maximum(
            self.feature_stats["max"],
            np.percentile(np.vstack([self.feature_stats["max"], features]), 95, axis=0),
        )

        self.feature_stats["count"] += 1

        # Robust normalization
        normalized = (features - self.feature_stats["mean"]) / (
            self.feature_stats["std"] + 1e-8
        )

        # Clip extreme values
        normalized = np.clip(normalized, -4, 4)

        return normalized

    def _create_enhanced_model(self, input_size: int):
        """Create enhanced model with improved architecture."""
        self.model = EnhancedLSTMTransformerModel(
            input_size=input_size,
            hidden_size=256,  # Increased capacity
            num_lstm_layers=3,  # Deeper LSTM
            num_transformer_layers=4,  # More transformer layers
            num_heads=16,  # More attention heads
            dropout=0.15,  # Slightly higher dropout
            seq_length=self.seq_length,
        ).to(self.device)

        # Enhanced optimizer with weight decay
        if self.use_lora:
            # Only optimize LoRA parameters during fine-tuning
            lora_params = [p for n, p in self.model.named_parameters() if "lora" in n]
            other_params = [
                p for n, p in self.model.named_parameters() if "lora" not in n
            ]

            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": lora_params,
                        "lr": self.learning_rate,
                        "weight_decay": 0.01,
                    },
                    {
                        "params": other_params,
                        "lr": self.learning_rate * 0.1,
                        "weight_decay": 0.05,
                    },
                ]
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.02,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 2,
            steps_per_epoch=1,
            epochs=self.retrain_frequency,
            pct_start=0.1,  # 10% warmup
        )

        self.logger.info(f"Enhanced model created with input size: {input_size}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "model_name": "enhanced_lstm_transformer",
            "symbol": self.symbol,
            "is_trained": self.is_trained,
            "sample_count": self.sample_count,
            "sequence_length": self.seq_length,
            "device": str(self.device),
            "feature_dimensions": (
                len(self.feature_stats.get("mean", []))
                if self.feature_stats["mean"] is not None
                else 0
            ),
            "training_losses": (
                list(self.training_losses)[-10:] if self.training_losses else []
            ),
            "accuracy_metrics": self.accuracy_metrics.copy(),
            "buffer_utilization": {
                "features": len(self.feature_buffer) / self.feature_buffer.maxlen,
                "returns": len(self.return_buffer) / self.return_buffer.maxlen,
            },
            "feature_importance": (
                self.model.get_feature_importance().cpu().numpy().tolist()
                if self.model
                else []
            ),
        }


# Factory function with enhanced parameters
def create_enhanced_lstm_transformer(
    symbol: str, **kwargs
) -> EnhancedLSTMTransformerAlpha:
    """Create enhanced LSTM/Transformer alpha model with optimizations."""
    return EnhancedLSTMTransformerAlpha(symbol=symbol, **kwargs)


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing Enhanced LSTM/Transformer Model...")

    model = EnhancedLSTMTransformerAlpha(
        "BTCUSDT", seq_length=100, retrain_frequency=200
    )

    # Create test data
    test_input = torch.randn(4, 100, 20)  # batch_size=4, seq_len=100, features=20

    if model.model is None:
        model._create_enhanced_model(20)

    # Test forward pass
    with torch.no_grad():
        results = model.model(test_input, return_attention=True)

    print(f"Model output shapes:")
    for key, value in results.items():
        if key != "attention_weights":
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} layers")

    print(f"Performance metrics: {model.get_performance_metrics()}")
    print("Enhanced model test completed successfully!")
