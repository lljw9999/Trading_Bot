#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) Implementation

Advanced attention-based architecture for multi-horizon time series forecasting
with static covariates and interpretable attention mechanisms.

Key Features:
- Multi-horizon forecasting
- Static and time-varying covariates
- Gated residual networks
- Variable selection networks
- Interpretable attention mechanisms
- Quantile forecasting for uncertainty estimation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""

    # Model architecture
    hidden_size: int = 160
    lstm_layers: int = 1
    dropout: float = 0.3
    attention_head_size: int = 4

    # Data configuration
    max_prediction_length: int = 6  # Forecast horizon (6 hours)
    max_encoder_length: int = 24  # Historical lookback (24 hours)

    # Training configuration
    batch_size: int = 64
    learning_rate: float = 0.03
    max_epochs: int = 100
    patience: int = 10

    # Feature configuration
    static_categoricals: List[str] = None
    static_reals: List[str] = None
    time_varying_known_categoricals: List[str] = None
    time_varying_known_reals: List[str] = None
    time_varying_unknown_categoricals: List[str] = None
    time_varying_unknown_reals: List[str] = None

    def __post_init__(self):
        if self.static_categoricals is None:
            self.static_categoricals = []
        if self.static_reals is None:
            self.static_reals = []
        if self.time_varying_known_categoricals is None:
            self.time_varying_known_categoricals = ["hour", "day_of_week", "month"]
        if self.time_varying_known_reals is None:
            self.time_varying_known_reals = ["time_idx"]
        if self.time_varying_unknown_categoricals is None:
            self.time_varying_unknown_categoricals = []
        if self.time_varying_unknown_reals is None:
            self.time_varying_unknown_reals = ["volume", "volatility", "rsi", "macd"]


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for TFT."""

    def __init__(self, input_size, hidden_size=None, dropout=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for TFT."""

    def __init__(
        self, input_size, hidden_size, output_size, dropout=0.3, context_size=None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.skip_connection = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

    def forward(self, x, context=None):
        # Main pathway
        residual = x
        x = self.fc1(x)

        if context is not None:
            x = x + self.context_fc(context)

        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)

        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        x = x + residual

        return self.layer_norm(x)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT."""

    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.input_size = input_size

        self.flattened_grn = GatedResidualNetwork(
            num_inputs * input_size, hidden_size, num_inputs, dropout
        )

        self.single_variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
                for _ in range(num_inputs)
            ]
        )

    def forward(self, flattened_embedding):
        # Flatten for variable selection
        sparse_weights = self.flattened_grn(flattened_embedding)
        sparse_weights = F.softmax(sparse_weights, dim=-1)

        # Apply variable selection
        var_outputs = []
        for i, grn in enumerate(self.single_variable_grns):
            # Extract variable
            var_embedding = flattened_embedding[
                ..., i * self.input_size : (i + 1) * self.input_size
            ]
            var_output = grn(var_embedding)
            var_outputs.append(var_output)

        # Weight and combine
        var_outputs = torch.stack(
            var_outputs, dim=-1
        )  # [batch, time, hidden_size, num_inputs]
        sparse_weights = sparse_weights.unsqueeze(-2)  # [batch, time, 1, num_inputs]

        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(dim=-1)  # [batch, time, hidden_size]

        return outputs, sparse_weights.squeeze(-2)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention for TFT."""

    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_h = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        batch_size, seq_len, _ = queries.size()

        # Linear transformations
        Q = self.w_q(queries)
        K = self.w_k(keys)
        V = self.w_v(values)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)

        # Concatenate heads
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear transformation
        output = self.w_h(attended_values)

        # Return average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)

        return output, avg_attention


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for multi-horizon forecasting."""

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Embedding dimensions
        self.categorical_embedding_sizes = {}
        self.numerical_input_size = len(config.time_varying_unknown_reals) + len(
            config.time_varying_known_reals
        )

        # Variable selection networks
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hidden_size,
            num_inputs=self.numerical_input_size,
            hidden_size=self.hidden_size,
            dropout=config.dropout,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hidden_size,
            num_inputs=len(config.time_varying_known_reals),
            hidden_size=self.hidden_size,
            dropout=config.dropout,
        )

        # Static variable selection
        if len(config.static_reals) > 0:
            self.static_variable_selection = VariableSelectionNetwork(
                input_size=self.hidden_size,
                num_inputs=len(config.static_reals),
                hidden_size=self.hidden_size,
                dropout=config.dropout,
            )

        # Input embeddings
        self.input_embeddings = nn.ModuleList(
            [nn.Linear(1, self.hidden_size) for _ in range(self.numerical_input_size)]
        )

        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, config.dropout
        )

        # LSTM encoder-decoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Gated skip connection
        self.gate_encoder = GatedLinearUnit(
            self.hidden_size, self.hidden_size, config.dropout
        )
        self.gate_decoder = GatedLinearUnit(
            self.hidden_size, self.hidden_size, config.dropout
        )

        # Self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            self.hidden_size, config.attention_head_size, config.dropout
        )

        # Post-attention gate
        self.post_attention_gate = GatedLinearUnit(
            self.hidden_size, self.hidden_size, config.dropout
        )

        # Position-wise feed forward
        self.pos_wise_ff = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, config.dropout
        )

        # Output projection
        self.output_layer = nn.Linear(
            self.hidden_size, 3
        )  # For quantile forecasting (10%, 50%, 90%)

    def forward(self, x_encoder, x_decoder, static_vars=None):
        batch_size = x_encoder.size(0)

        # Embed numerical inputs
        embedded_encoder = []
        for i, embedding_layer in enumerate(self.input_embeddings):
            if i < x_encoder.size(-1):
                embedded_encoder.append(embedding_layer(x_encoder[..., i : i + 1]))

        if embedded_encoder:
            embedded_encoder = torch.cat(embedded_encoder, dim=-1)

            # Variable selection for encoder
            selected_encoder, encoder_weights = self.encoder_variable_selection(
                embedded_encoder
            )
        else:
            selected_encoder = torch.zeros(
                batch_size, x_encoder.size(1), self.hidden_size, device=x_encoder.device
            )
            encoder_weights = None

        # Static enrichment
        if static_vars is not None and hasattr(self, "static_variable_selection"):
            static_embedding = []
            for i, var in enumerate(static_vars.split(1, dim=-1)):
                static_embedding.append(self.input_embeddings[0](var))
            static_embedding = torch.cat(static_embedding, dim=-1)

            selected_static, static_weights = self.static_variable_selection(
                static_embedding
            )

            # Expand static to match sequence length
            selected_static = selected_static.unsqueeze(1).expand(
                -1, selected_encoder.size(1), -1
            )
            selected_encoder = self.static_enrichment(selected_encoder, selected_static)

        # LSTM encoder
        encoder_output, encoder_state = self.encoder_lstm(selected_encoder)

        # Prepare decoder input
        embedded_decoder = []
        for i in range(
            min(len(self.config.time_varying_known_reals), x_decoder.size(-1))
        ):
            embedded_decoder.append(self.input_embeddings[i](x_decoder[..., i : i + 1]))

        if embedded_decoder:
            embedded_decoder = torch.cat(embedded_decoder, dim=-1)
            selected_decoder, decoder_weights = self.decoder_variable_selection(
                embedded_decoder
            )
        else:
            selected_decoder = torch.zeros(
                batch_size, x_decoder.size(1), self.hidden_size, device=x_decoder.device
            )
            decoder_weights = None

        # LSTM decoder
        decoder_output, _ = self.decoder_lstm(selected_decoder, encoder_state)

        # Gated skip connections
        gated_encoder = self.gate_encoder(encoder_output)
        gated_decoder = self.gate_decoder(decoder_output)

        # Combine encoder and decoder
        combined_sequence = torch.cat([gated_encoder, gated_decoder], dim=1)

        # Self-attention
        attended_sequence, attention_weights = self.self_attention(
            combined_sequence, combined_sequence, combined_sequence
        )

        # Post-attention gate
        gated_attention = self.post_attention_gate(attended_sequence)

        # Add residual connection
        attended_sequence = attended_sequence + gated_attention

        # Position-wise feed forward
        final_sequence = self.pos_wise_ff(attended_sequence)

        # Extract decoder portion
        decoder_length = x_decoder.size(1)
        final_decoder = final_sequence[:, -decoder_length:, :]

        # Output projection
        outputs = self.output_layer(final_decoder)

        return {
            "prediction": outputs,
            "attention_weights": attention_weights[:, -decoder_length:, :],
            "encoder_variable_weights": encoder_weights,
            "decoder_variable_weights": decoder_weights,
        }


class TFTDataset(Dataset):
    """Dataset for TFT training."""

    def __init__(self, data, config: TFTConfig, mode="train"):
        self.data = data
        self.config = config
        self.mode = mode

        # Calculate total sequence length
        self.total_length = config.max_encoder_length + config.max_prediction_length

        # Prepare sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        data_length = len(self.data)

        for i in range(
            self.config.max_encoder_length,
            data_length - self.config.max_prediction_length + 1,
        ):
            # Encoder data (historical)
            encoder_start = i - self.config.max_encoder_length
            encoder_end = i
            encoder_data = self.data.iloc[encoder_start:encoder_end]

            # Decoder data (future known variables)
            decoder_start = i
            decoder_end = i + self.config.max_prediction_length
            decoder_data = self.data.iloc[decoder_start:decoder_end]

            # Target (what we want to predict)
            target_data = self.data.iloc[decoder_start:decoder_end]

            sequences.append(
                {
                    "encoder": encoder_data,
                    "decoder": decoder_data,
                    "target": target_data,
                    "encoder_length": self.config.max_encoder_length,
                    "decoder_length": self.config.max_prediction_length,
                }
            )

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Extract encoder features
        encoder_features = []
        for col in (
            self.config.time_varying_unknown_reals
            + self.config.time_varying_known_reals
        ):
            if col in sequence["encoder"].columns:
                encoder_features.append(sequence["encoder"][col].values)

        if encoder_features:
            encoder_tensor = torch.FloatTensor(np.column_stack(encoder_features))
        else:
            encoder_tensor = torch.zeros(self.config.max_encoder_length, 1)

        # Extract decoder features (known future inputs)
        decoder_features = []
        for col in self.config.time_varying_known_reals:
            if col in sequence["decoder"].columns:
                decoder_features.append(sequence["decoder"][col].values)

        if decoder_features:
            decoder_tensor = torch.FloatTensor(np.column_stack(decoder_features))
        else:
            decoder_tensor = torch.zeros(self.config.max_prediction_length, 1)

        # Extract target
        if "close" in sequence["target"].columns:
            target_tensor = torch.FloatTensor(sequence["target"]["close"].values)
        else:
            target_tensor = torch.zeros(self.config.max_prediction_length)

        return {
            "encoder": encoder_tensor,
            "decoder": decoder_tensor,
            "target": target_tensor,
        }


class TFTTrainer:
    """Trainer for Temporal Fusion Transformer."""

    def __init__(self, config: TFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = TemporalFusionTransformer(config).to(self.device)

        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Quantile loss for uncertainty estimation
        self.quantiles = [0.1, 0.5, 0.9]

    def quantile_loss(self, predictions, targets, quantiles):
        """Quantile loss for uncertainty estimation."""
        losses = []
        for i, q in enumerate(quantiles):
            pred = predictions[..., i]
            diff = targets - pred
            loss = torch.max(q * diff, (q - 1) * diff)
            losses.append(loss.mean())
        return sum(losses) / len(losses)

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            encoder = batch["encoder"].to(self.device)
            decoder = batch["decoder"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(encoder, decoder)
            predictions = outputs["prediction"]

            # Calculate loss
            loss = self.quantile_loss(predictions, target, self.quantiles)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                encoder = batch["encoder"].to(self.device)
                decoder = batch["decoder"].to(self.device)
                target = batch["target"].to(self.device)

                outputs = self.model(encoder, decoder)
                predictions = outputs["prediction"]

                loss = self.quantile_loss(predictions, target, self.quantiles)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader=None):
        """Train the model."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.scheduler.step(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), "best_tft_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - Train Loss: {train_loss:.6f}"
                )

        # Load best model
        if val_loader is not None and os.path.exists("best_tft_model.pth"):
            self.model.load_state_dict(torch.load("best_tft_model.pth"))
            logger.info("Loaded best model")

    def predict(self, encoder_data, decoder_data):
        """Make predictions."""
        self.model.eval()

        with torch.no_grad():
            encoder_tensor = (
                torch.FloatTensor(encoder_data).unsqueeze(0).to(self.device)
            )
            decoder_tensor = (
                torch.FloatTensor(decoder_data).unsqueeze(0).to(self.device)
            )

            outputs = self.model(encoder_tensor, decoder_tensor)
            predictions = outputs["prediction"].cpu().numpy()[0]
            attention_weights = (
                outputs["attention_weights"].cpu().numpy()[0]
                if outputs["attention_weights"] is not None
                else None
            )

            return {
                "predictions": predictions,
                "attention_weights": attention_weights,
                "quantile_10": predictions[:, 0],
                "quantile_50": predictions[:, 1],
                "quantile_90": predictions[:, 2],
            }


class TFTCryptoPredictor:
    """TFT-based cryptocurrency price predictor."""

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.config = TFTConfig()
        self.trainer = TFTTrainer(self.config)

        # Redis connection for real-time data
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for TFT predictions")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

        # Data scalers
        self.scalers = {}

    def prepare_crypto_data(self, symbol="BTCUSDT", lookback_hours=168):
        """Prepare cryptocurrency data for TFT training."""
        # Generate sample crypto data with features
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=lookback_hours),
            end=datetime.now(),
            freq="H",
        )

        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        base_price = 117800 if symbol == "BTCUSDT" else 3580
        prices = []
        volumes = []

        for i, ts in enumerate(timestamps):
            # Add time-based patterns
            hour_factor = np.sin(2 * np.pi * ts.hour / 24) * 0.02
            day_factor = np.sin(2 * np.pi * ts.dayofweek / 7) * 0.05
            trend = i * 0.001
            noise = np.random.normal(0, 0.02)

            price_change = hour_factor + day_factor + trend + noise
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + price_change)

            prices.append(price)
            volumes.append(np.random.uniform(1000, 5000))

        # Create DataFrame
        df = pd.DataFrame({"timestamp": timestamps, "close": prices, "volume": volumes})

        # Add time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["time_idx"] = range(len(df))

        # Add technical indicators
        df["volatility"] = df["close"].rolling(window=24).std().fillna(0)
        df["rsi"] = self._calculate_rsi(df["close"])
        df["macd"] = self._calculate_macd(df["close"])

        # Fill NaN values
        df = df.fillna(method="bfill").fillna(0)

        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def train_model(self, symbol="BTCUSDT"):
        """Train TFT model on cryptocurrency data."""
        logger.info(f"Training TFT model for {symbol}")

        # Prepare data
        df = self.prepare_crypto_data(symbol)

        # Scale features
        feature_columns = ["close", "volume", "volatility", "rsi", "macd"]
        for col in feature_columns:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler

        # Create dataset
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        val_data = df[train_size:]

        train_dataset = TFTDataset(train_data, self.config, mode="train")
        val_dataset = TFTDataset(val_data, self.config, mode="val")

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Train model
        self.trainer.train(train_loader, val_loader)

        logger.info(f"✅ TFT model trained for {symbol}")

        # Save model and scalers
        self._save_model(symbol)

    def _save_model(self, symbol):
        """Save trained model and scalers."""
        model_path = f"tft_model_{symbol.lower()}.pth"
        scalers_path = f"tft_scalers_{symbol.lower()}.json"

        # Save model
        torch.save(self.trainer.model.state_dict(), model_path)

        # Save scalers (convert to serializable format)
        scalers_data = {}
        for name, scaler in self.scalers.items():
            scalers_data[name] = {
                "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            }

        with open(scalers_path, "w") as f:
            json.dump(scalers_data, f)

        logger.info(f"✅ Model and scalers saved for {symbol}")

    def predict_future(self, symbol="BTCUSDT", hours_ahead=6):
        """Predict future prices using TFT."""
        try:
            # Prepare recent data
            df = self.prepare_crypto_data(symbol, lookback_hours=48)  # More recent data

            # Scale features using saved scalers
            feature_columns = ["close", "volume", "volatility", "rsi", "macd"]
            for col in feature_columns:
                if col in self.scalers:
                    df[col] = (
                        self.scalers[col]
                        .transform(df[col].values.reshape(-1, 1))
                        .flatten()
                    )

            # Prepare encoder data (last 24 hours)
            encoder_data = (
                df[["volume", "volatility", "rsi", "macd", "time_idx"]]
                .iloc[-24:]
                .values
            )

            # Prepare decoder data (future time indices)
            last_time_idx = df["time_idx"].iloc[-1]
            future_time_indices = np.arange(
                last_time_idx + 1, last_time_idx + hours_ahead + 1
            )
            decoder_data = future_time_indices.reshape(-1, 1)

            # Make prediction
            results = self.trainer.predict(encoder_data, decoder_data)

            # Inverse transform predictions
            if "close" in self.scalers:
                predictions = (
                    self.scalers["close"]
                    .inverse_transform(results["quantile_50"].reshape(-1, 1))
                    .flatten()
                )

                lower_bound = (
                    self.scalers["close"]
                    .inverse_transform(results["quantile_10"].reshape(-1, 1))
                    .flatten()
                )

                upper_bound = (
                    self.scalers["close"]
                    .inverse_transform(results["quantile_90"].reshape(-1, 1))
                    .flatten()
                )
            else:
                predictions = results["quantile_50"]
                lower_bound = results["quantile_10"]
                upper_bound = results["quantile_90"]

            # Create timestamps for predictions
            last_timestamp = df["timestamp"].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(hours=i + 1) for i in range(hours_ahead)
            ]

            return {
                "symbol": symbol,
                "timestamps": [ts.isoformat() for ts in future_timestamps],
                "predictions": predictions.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "confidence_interval": 80,  # 10th to 90th percentile
                "attention_weights": (
                    results["attention_weights"].tolist()
                    if results["attention_weights"] is not None
                    else None
                ),
                "model_type": "TFT",
                "prediction_horizon": hours_ahead,
            }

        except Exception as e:
            logger.error(f"Error in TFT prediction: {e}")
            return None

    def store_predictions(self, predictions):
        """Store TFT predictions in Redis."""
        if not self.redis_client or not predictions:
            return

        try:
            # Store current predictions
            key = f"tft_predictions:{predictions['symbol']}"
            self.redis_client.setex(key, 3600, json.dumps(predictions))  # 1 hour expiry

            # Store in time series for historical tracking
            ts_key = f"tft_history:{predictions['symbol']}"
            timestamp = datetime.now().timestamp()
            self.redis_client.zadd(ts_key, {json.dumps(predictions): timestamp})

            # Keep only last 24 hours of predictions
            cutoff = timestamp - 86400
            self.redis_client.zremrangebyscore(ts_key, 0, cutoff)

            logger.info(f"✅ TFT predictions stored for {predictions['symbol']}")

        except Exception as e:
            logger.error(f"Error storing TFT predictions: {e}")

    def get_latest_predictions(self, symbol="BTCUSDT"):
        """Get latest TFT predictions from Redis."""
        if not self.redis_client:
            return None

        try:
            key = f"tft_predictions:{symbol}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting TFT predictions: {e}")

        return None


# Example usage and testing
async def main():
    """Main function for testing TFT implementation."""
    print("🚀 Testing Temporal Fusion Transformer (TFT)")
    print("=" * 60)

    # Initialize TFT predictor
    predictor = TFTCryptoPredictor()

    # Train model for BTC
    print("📊 Training TFT model for BTCUSDT...")
    predictor.train_model("BTCUSDT")

    # Make predictions
    print("🔮 Generating TFT predictions...")
    btc_predictions = predictor.predict_future("BTCUSDT", hours_ahead=6)

    if btc_predictions:
        print(f"✅ TFT Predictions for {btc_predictions['symbol']}:")
        for i, (ts, pred, lower, upper) in enumerate(
            zip(
                btc_predictions["timestamps"],
                btc_predictions["predictions"],
                btc_predictions["lower_bound"],
                btc_predictions["upper_bound"],
            )
        ):
            print(
                f"  {i+1}h: ${pred:.2f} [{lower:.2f}-{upper:.2f}] at {ts.strftime('%H:%M')}"
            )

        # Store predictions
        predictor.store_predictions(btc_predictions)
        print("💾 Predictions stored in Redis")

    # Train and predict for ETH
    print("\n📊 Training TFT model for ETHUSDT...")
    predictor.train_model("ETHUSDT")

    eth_predictions = predictor.predict_future("ETHUSDT", hours_ahead=6)
    if eth_predictions:
        print(f"✅ TFT Predictions for {eth_predictions['symbol']}:")
        for i, (ts, pred, lower, upper) in enumerate(
            zip(
                eth_predictions["timestamps"],
                eth_predictions["predictions"],
                eth_predictions["lower_bound"],
                eth_predictions["upper_bound"],
            )
        ):
            print(
                f"  {i+1}h: ${pred:.2f} [{lower:.2f}-{upper:.2f}] at {ts.strftime('%H:%M')}"
            )

        predictor.store_predictions(eth_predictions)

    print("\n🎉 TFT Implementation Complete!")
    print("✅ Multi-horizon forecasting with uncertainty estimation")
    print("✅ Interpretable attention mechanisms")
    print("✅ Variable selection networks")
    print("✅ Quantile forecasting for risk assessment")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
