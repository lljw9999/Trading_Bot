"""
LSTM/Transformer Deep Learning Alpha Model

Advanced time series prediction using LSTM and Transformer architectures
for capturing long-term dependencies and non-linear patterns in market data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import logging

from ..layer0_data_ingestion.schemas import FeatureSnapshot, AlphaSignal


class LSTMTransformerModel(nn.Module):
    """
    Hybrid LSTM-Transformer model for time series prediction.

    Architecture:
    - LSTM layers for capturing sequential dependencies
    - Transformer encoder for attention-based pattern recognition
    - Multi-head attention for different time horizons
    - Residual connections and layer normalization
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_length: int = 60,
    ):
        """
        Initialize the hybrid model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            seq_length: Sequence length for training
        """
        super(LSTMTransformerModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,  # bidirectional LSTM
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Multi-horizon prediction heads
        self.price_head = nn.Linear(hidden_size * 2, 1)  # Price prediction
        self.volatility_head = nn.Linear(hidden_size * 2, 1)  # Volatility prediction
        self.direction_head = nn.Linear(hidden_size * 2, 3)  # Direction (up/down/flat)

        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_size * 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
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
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Tuple of (price_pred, volatility_pred, direction_pred, confidence)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size * 2)

        # Transformer processing
        # Apply positional encoding implicitly through attention
        transformer_out = self.transformer(
            lstm_out
        )  # (batch_size, seq_length, hidden_size * 2)

        # Use the last timestep output
        last_hidden = transformer_out[:, -1, :]  # (batch_size, hidden_size * 2)

        # Apply layer normalization and dropout
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Multi-head predictions
        price_pred = self.price_head(last_hidden)  # Price change prediction
        volatility_pred = F.softplus(
            self.volatility_head(last_hidden)
        )  # Volatility (positive)
        direction_pred = F.softmax(
            self.direction_head(last_hidden), dim=1
        )  # Direction probabilities
        confidence = torch.sigmoid(
            self.confidence_head(last_hidden)
        )  # Confidence score

        return price_pred, volatility_pred, direction_pred, confidence


class LSTMTransformerAlpha:
    """
    LSTM/Transformer Alpha Model for generating trading signals.

    Features:
    - Deep learning-based pattern recognition
    - Multi-horizon predictions (price, volatility, direction)
    - Adaptive confidence scoring
    - Online learning capability
    - GPU acceleration support
    """

    def __init__(
        self,
        symbol: str,
        seq_length: int = 60,
        retrain_frequency: int = 1000,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the LSTM/Transformer alpha model.

        Args:
            symbol: Trading symbol
            seq_length: Sequence length for model input
            retrain_frequency: Number of samples between retraining
            learning_rate: Learning rate for optimization
            device: Device for computation (cuda/cpu)
        """
        self.symbol = symbol
        self.seq_length = seq_length
        self.retrain_frequency = retrain_frequency
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # Data storage
        self.feature_buffer = deque(maxlen=seq_length * 2)  # Store more for training
        self.price_buffer = deque(maxlen=seq_length * 2)
        self.return_buffer = deque(maxlen=seq_length * 2)

        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # Training tracking
        self.sample_count = 0
        self.is_trained = False

        # Performance tracking
        self.predictions = deque(maxlen=100)
        self.prediction_accuracy = 0.0

        # Feature normalization
        self.feature_stats = {}

        self.logger = logging.getLogger(f"lstm_transformer_alpha.{symbol}")
        self.logger.info(f"Initialized LSTM/Transformer Alpha for {symbol} on {device}")

    def _extract_features(self, feature_snapshot: FeatureSnapshot) -> np.ndarray:
        """
        Extract and normalize features from FeatureSnapshot.

        Args:
            feature_snapshot: Input feature snapshot

        Returns:
            Normalized feature vector
        """
        features = []

        # Price features
        if feature_snapshot.mid_price:
            features.append(float(feature_snapshot.mid_price))
        else:
            features.append(0.0)

        # Spread features
        features.append(feature_snapshot.spread_bps or 0.0)

        # Return features
        features.extend(
            [
                feature_snapshot.return_1m or 0.0,
                feature_snapshot.return_5m or 0.0,
                feature_snapshot.return_15m or 0.0,
                feature_snapshot.return_1h or 0.0,
            ]
        )

        # Volatility features
        features.extend(
            [
                feature_snapshot.volatility_5m or 0.0,
                feature_snapshot.volatility_15m or 0.0,
                feature_snapshot.volatility_1h or 0.0,
            ]
        )

        # Order book features
        features.extend(
            [
                feature_snapshot.order_book_imbalance or 0.0,
                feature_snapshot.order_book_pressure or 0.0,
            ]
        )

        # Volume features
        features.extend(
            [
                (
                    float(feature_snapshot.volume_1m)
                    if feature_snapshot.volume_1m
                    else 0.0
                ),
                (
                    float(feature_snapshot.volume_5m)
                    if feature_snapshot.volume_5m
                    else 0.0
                ),
                feature_snapshot.volume_ratio or 0.0,
            ]
        )

        # Technical indicators
        features.extend(
            [
                float(feature_snapshot.sma_5) if feature_snapshot.sma_5 else 0.0,
                float(feature_snapshot.sma_20) if feature_snapshot.sma_20 else 0.0,
                float(feature_snapshot.sma_50) if feature_snapshot.sma_50 else 0.0,
                float(feature_snapshot.ema_5) if feature_snapshot.ema_5 else 0.0,
                float(feature_snapshot.ema_20) if feature_snapshot.ema_20 else 0.0,
                feature_snapshot.rsi_14 or 0.0,
            ]
        )

        # Sentiment features
        features.append(feature_snapshot.sent_score or 0.0)

        return np.array(features, dtype=np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using running statistics.

        Args:
            features: Raw feature vector

        Returns:
            Normalized feature vector
        """
        if not self.feature_stats:
            # Initialize with first observation
            self.feature_stats = {
                "mean": features.copy(),
                "std": np.ones_like(features),
                "count": 1,
            }
            return features

        # Update running statistics
        self.feature_stats["count"] += 1
        alpha = 1.0 / min(self.feature_stats["count"], 1000)  # Decay factor

        # Update mean
        self.feature_stats["mean"] = (1 - alpha) * self.feature_stats[
            "mean"
        ] + alpha * features

        # Update std
        diff = features - self.feature_stats["mean"]
        self.feature_stats["std"] = np.sqrt(
            (1 - alpha) * self.feature_stats["std"] ** 2 + alpha * diff**2
        )

        # Normalize
        normalized = (features - self.feature_stats["mean"]) / (
            self.feature_stats["std"] + 1e-8
        )

        # Clip to prevent extreme values
        normalized = np.clip(normalized, -3, 3)

        return normalized

    def _initialize_model(self, input_size: int):
        """Initialize the model with given input size."""
        self.model = LSTMTransformerModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            seq_length=self.seq_length,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        self.logger.info(f"Model initialized with input size: {input_size}")

    def _prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from buffers.

        Returns:
            Tuple of (input_sequences, target_returns)
        """
        if len(self.feature_buffer) < self.seq_length + 1:
            return None, None

        # Convert to numpy arrays
        features = np.array(list(self.feature_buffer))
        returns = np.array(list(self.return_buffer))

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(features) - self.seq_length):
            seq = features[i : i + self.seq_length]
            target = returns[i + self.seq_length]  # Next return

            sequences.append(seq)
            targets.append(target)

        if len(sequences) == 0:
            return None, None

        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        y = torch.FloatTensor(np.array(targets)).to(self.device)

        return X, y

    def _train_model(self):
        """Train the model on recent data."""
        if not self.model:
            return

        X, y = self._prepare_training_data()
        if X is None or len(X) < 10:  # Need minimum samples
            return

        self.model.train()

        # Training loop
        for epoch in range(5):  # Quick online learning
            self.optimizer.zero_grad()

            # Forward pass
            price_pred, vol_pred, dir_pred, confidence = self.model(X)

            # Calculate loss (focus on price prediction)
            loss = self.criterion(price_pred.squeeze(), y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        self.is_trained = True
        self.logger.debug(
            f"Model retrained on {len(X)} samples, final loss: {loss.item():.6f}"
        )

    def update_features(
        self, feature_snapshot: FeatureSnapshot
    ) -> Optional[AlphaSignal]:
        """
        Update model with new feature snapshot and generate signal.

        Args:
            feature_snapshot: New feature snapshot

        Returns:
            AlphaSignal if conditions are met, None otherwise
        """
        # Extract and normalize features
        features = self._extract_features(feature_snapshot)
        normalized_features = self._normalize_features(features)

        # Store in buffer
        self.feature_buffer.append(normalized_features)

        # Calculate return (if we have price history)
        if feature_snapshot.mid_price and len(self.price_buffer) > 0:
            prev_price = self.price_buffer[-1]
            current_price = float(feature_snapshot.mid_price)
            return_1tick = (current_price - prev_price) / prev_price
            self.return_buffer.append(return_1tick)
        else:
            self.return_buffer.append(0.0)

        # Store price
        if feature_snapshot.mid_price:
            self.price_buffer.append(float(feature_snapshot.mid_price))

        # Initialize model if needed
        if not self.model:
            self._initialize_model(len(features))

        # Increment sample count
        self.sample_count += 1

        # Retrain periodically
        if self.sample_count % self.retrain_frequency == 0:
            self._train_model()

        # Generate signal if model is trained and we have enough data
        if not self.is_trained or len(self.feature_buffer) < self.seq_length:
            return None

        return self._generate_signal(feature_snapshot)

    def _generate_signal(self, feature_snapshot: FeatureSnapshot) -> AlphaSignal:
        """
        Generate trading signal using the trained model.

        Args:
            feature_snapshot: Current feature snapshot

        Returns:
            AlphaSignal with prediction and confidence
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare input sequence
            sequence = np.array(list(self.feature_buffer)[-self.seq_length :])
            X = (
                torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            )  # Add batch dimension

            # Get predictions
            price_pred, vol_pred, dir_pred, confidence = self.model(X)

            # Extract values
            price_change = price_pred.squeeze().cpu().item()
            volatility = vol_pred.squeeze().cpu().item()
            direction_probs = dir_pred.squeeze().cpu().numpy()
            conf_score = confidence.squeeze().cpu().item()

            # Calculate edge in basis points
            # Scale price change prediction to basis points
            edge_bps = price_change * 10000  # Convert to basis points

            # Apply direction probability weighting
            # direction_probs: [down, flat, up]
            directional_bias = (
                direction_probs[2] - direction_probs[0]
            )  # up_prob - down_prob
            edge_bps *= directional_bias

            # Enhance confidence with volatility prediction
            vol_adjusted_conf = conf_score * (1 + np.tanh(volatility))

            # Clip values
            edge_bps = np.clip(edge_bps, -50, 50)  # ±50 bps max
            final_confidence = np.clip(vol_adjusted_conf, 0.1, 0.95)

            # Create reasoning
            direction_name = ["bearish", "neutral", "bullish"][
                np.argmax(direction_probs)
            ]
            reasoning = (
                f"LSTM/Transformer: {direction_name} prediction "
                f"(price_change: {price_change:.4f}, "
                f"direction_prob: {direction_probs[np.argmax(direction_probs)]:.3f}, "
                f"volatility: {volatility:.4f})"
            )

            return AlphaSignal(
                model_name="lstm_transformer",
                symbol=self.symbol,
                timestamp=feature_snapshot.timestamp,
                edge_bps=edge_bps,
                confidence=final_confidence,
                signal_strength=abs(price_change),
                metadata={
                    "price_change_pred": price_change,
                    "volatility_pred": volatility,
                    "direction_probs": direction_probs.tolist(),
                    "raw_confidence": conf_score,
                    "reasoning": reasoning,
                },
            )

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        return {
            "model_name": "lstm_transformer",
            "symbol": self.symbol,
            "is_trained": self.is_trained,
            "sample_count": self.sample_count,
            "buffer_size": len(self.feature_buffer),
            "device": str(self.device),
            "prediction_accuracy": self.prediction_accuracy,
            "feature_dimensions": len(self.feature_stats.get("mean", [])),
            "seq_length": self.seq_length,
        }


# Factory function for easy integration
def create_lstm_transformer_alpha(symbol: str, **kwargs) -> LSTMTransformerAlpha:
    """
    Factory function to create LSTM/Transformer alpha model.

    Args:
        symbol: Trading symbol
        **kwargs: Additional model parameters

    Returns:
        Initialized LSTMTransformerAlpha instance
    """
    return LSTMTransformerAlpha(symbol=symbol, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    import time
    from decimal import Decimal

    # Test the model
    model = LSTMTransformerAlpha("BTCUSDT")

    # Create synthetic feature snapshots
    for i in range(100):
        feature_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            mid_price=Decimal(str(50000 + i * 10 + np.random.randn() * 100)),
            spread_bps=2.5 + np.random.randn() * 0.5,
            return_1m=np.random.randn() * 0.001,
            return_5m=np.random.randn() * 0.003,
            volatility_5m=0.02 + np.random.randn() * 0.005,
            order_book_imbalance=np.random.randn() * 0.1,
            volume_1m=Decimal(str(100 + np.random.randn() * 20)),
            rsi_14=50 + np.random.randn() * 15,
            sent_score=np.random.randn() * 0.3,
        )

        signal = model.update_features(feature_snapshot)
        if signal:
            print(
                f"Signal {i}: edge_bps={signal.edge_bps:.2f}, confidence={signal.confidence:.3f}"
            )

    print("\nModel Stats:")
    print(model.get_model_stats())
