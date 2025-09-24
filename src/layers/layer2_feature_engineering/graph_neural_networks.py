#!/usr/bin/env python3
"""
Graph Neural Networks (GCNs) for Cross-Asset Relationships
Implements GCN architecture for modeling complex relationships between cryptocurrencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GCNConfig:
    """Configuration for Graph Convolutional Network"""

    num_assets: int = 10  # Number of crypto assets
    input_features: int = 20  # Features per asset (price, volume, technical indicators)
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    lookback_steps: int = 24  # Hours to look back
    prediction_horizon: int = 6  # Hours to predict
    edge_threshold: float = 0.3  # Correlation threshold for graph edges


class GraphConvolutionLayer(nn.Module):
    """Graph Convolution Layer for processing asset relationships"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution

        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Output features [batch_size, num_nodes, out_features]
        """
        # Linear transformation: XW
        support = torch.matmul(x, self.weight)

        # Graph convolution: AXW
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output += self.bias

        return output


class TemporalGraphAttention(nn.Module):
    """Temporal attention mechanism for time-dependent graph relationships"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(TemporalGraphAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, hidden_dim]
        Returns:
            Attention-weighted features
        """
        batch_size, seq_len, num_nodes, _ = x.size()

        # Reshape for multi-head attention
        q = self.query(x).view(
            batch_size, seq_len, num_nodes, self.num_heads, self.head_dim
        )
        k = self.key(x).view(
            batch_size, seq_len, num_nodes, self.num_heads, self.head_dim
        )
        v = self.value(x).view(
            batch_size, seq_len, num_nodes, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        q = q.transpose(2, 3)  # [batch, seq, heads, nodes, head_dim]
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, num_nodes, self.hidden_dim)

        return self.out_proj(attn_output)


class CrossAssetGCN(nn.Module):
    """Graph Convolutional Network for Cross-Asset Relationship Modeling"""

    def __init__(self, config: GCNConfig):
        super(CrossAssetGCN, self).__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_features, config.hidden_dim)

        # Graph convolution layers
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvolutionLayer(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_layers)
            ]
        )

        # Temporal attention
        self.temporal_attention = TemporalGraphAttention(config.hidden_dim)

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
        )

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.output_proj = nn.Linear(config.hidden_dim, config.prediction_horizon)

        # Asset relationship predictor
        self.relationship_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GCN

        Args:
            x: Input features [batch_size, seq_len, num_nodes, input_features]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Dictionary with predictions and relationship scores
        """
        batch_size, seq_len, num_nodes, _ = x.size()

        # Project input features
        x = self.input_proj(x)  # [batch, seq, nodes, hidden]

        # Apply graph convolutions at each time step
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # [batch, nodes, hidden]

            # Apply GCN layers
            for gcn_layer in self.gcn_layers:
                x_t = F.relu(gcn_layer(x_t, adj))
                x_t = self.dropout(x_t)

            gcn_outputs.append(x_t)

        # Stack temporal outputs
        gcn_output = torch.stack(gcn_outputs, dim=1)  # [batch, seq, nodes, hidden]

        # Apply temporal attention
        attended_output = self.temporal_attention(gcn_output)

        # Process with LSTM for each node
        predictions = []
        node_embeddings = []

        for node_idx in range(num_nodes):
            node_seq = attended_output[:, :, node_idx, :]  # [batch, seq, hidden]

            # LSTM forward pass
            lstm_out, _ = self.lstm(node_seq)
            final_hidden = lstm_out[:, -1, :]  # [batch, hidden]

            # Prediction
            pred = self.output_proj(final_hidden)  # [batch, pred_horizon]
            predictions.append(pred)
            node_embeddings.append(final_hidden)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch, nodes, pred_horizon]
        node_embeddings = torch.stack(node_embeddings, dim=1)  # [batch, nodes, hidden]

        # Compute pairwise relationship scores
        relationship_scores = self._compute_relationships(node_embeddings)

        return {
            "predictions": predictions,
            "node_embeddings": node_embeddings,
            "relationship_scores": relationship_scores,
        }

    def _compute_relationships(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise relationship scores between assets"""
        batch_size, num_nodes, hidden_dim = embeddings.size()

        relationships = torch.zeros(batch_size, num_nodes, num_nodes)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Concatenate embeddings
                pair_embedding = torch.cat([embeddings[:, i], embeddings[:, j]], dim=1)

                # Predict relationship strength
                relationship = self.relationship_predictor(pair_embedding).squeeze(-1)
                relationships[:, i, j] = relationship
                relationships[:, j, i] = relationship  # Symmetric

        return relationships


class GCNCryptoAnalyzer:
    """Main class for GCN-based crypto cross-asset analysis"""

    def __init__(self, config: Optional[GCNConfig] = None):
        self.config = config or GCNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = CrossAssetGCN(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.scaler = StandardScaler()

        # Crypto assets to analyze
        self.assets = [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "XRPUSDT",
            "BCHUSDT",
            "BNBUSDT",
            "SOLUSDT",
        ]

        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for GCN analysis")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def build_correlation_graph(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> torch.Tensor:
        """Build graph structure based on price correlations"""
        correlations = np.zeros((len(self.assets), len(self.assets)))

        # Get returns for correlation calculation
        returns_data = {}
        for asset in self.assets:
            if asset in price_data and len(price_data[asset]) > 1:
                returns = price_data[asset]["close"].pct_change().dropna()
                returns_data[asset] = returns

        # Calculate pairwise correlations
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i == j:
                    correlations[i, j] = 1.0
                elif asset1 in returns_data and asset2 in returns_data:
                    corr = returns_data[asset1].corr(returns_data[asset2])
                    correlations[i, j] = abs(corr) if not np.isnan(corr) else 0.0

        # Create adjacency matrix with threshold
        adj_matrix = (correlations > self.config.edge_threshold).astype(float)

        # Add self-loops
        np.fill_diagonal(adj_matrix, 1.0)

        # Normalize adjacency matrix (symmetric normalization)
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix + 1e-6))
        normalized_adj = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt

        return torch.FloatTensor(normalized_adj).to(self.device)

    def prepare_features(self, price_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Prepare node features for each asset"""
        features_list = []

        for asset in self.assets:
            if (
                asset in price_data
                and len(price_data[asset]) >= self.config.lookback_steps
            ):
                df = price_data[asset].tail(self.config.lookback_steps).copy()

                # Technical indicators
                df["rsi"] = self._calculate_rsi(df["close"])
                df["macd"] = self._calculate_macd(df["close"])
                df["bb_upper"], df["bb_lower"] = self._calculate_bollinger_bands(
                    df["close"]
                )
                df["volume_sma"] = df["volume"].rolling(5).mean()
                df["price_sma"] = df["close"].rolling(5).mean()

                # Price features
                df["returns"] = df["close"].pct_change()
                df["volatility"] = df["returns"].rolling(5).std()
                df["high_low_ratio"] = df["high"] / df["low"]
                df["volume_price_ratio"] = df["volume"] / df["close"]

                # On-chain indicators (simulated)
                df["whale_activity"] = np.random.normal(0.5, 0.1, len(df))
                df["exchange_flow"] = np.random.normal(0.3, 0.15, len(df))

                # Select features
                feature_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "rsi",
                    "macd",
                    "bb_upper",
                    "bb_lower",
                    "volume_sma",
                    "price_sma",
                    "returns",
                    "volatility",
                    "high_low_ratio",
                    "volume_price_ratio",
                    "whale_activity",
                    "exchange_flow",
                ]

                # Pad if needed
                if len(feature_cols) < self.config.input_features:
                    missing = self.config.input_features - len(feature_cols)
                    for i in range(missing):
                        df[f"feature_{i}"] = 0.0
                        feature_cols.append(f"feature_{i}")

                asset_features = (
                    df[feature_cols[: self.config.input_features]].fillna(0).values
                )
            else:
                # Create dummy features if no data
                asset_features = np.zeros(
                    (self.config.lookback_steps, self.config.input_features)
                )

            features_list.append(asset_features)

        # Stack all asset features
        features = np.stack(features_list, axis=1)  # [seq_len, num_assets, features]

        # Normalize features
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.fit_transform(features_flat)
        features = features_normalized.reshape(original_shape)

        # Add batch dimension and convert to tensor
        features = (
            torch.FloatTensor(features).unsqueeze(0).to(self.device)
        )  # [1, seq, assets, features]

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    def get_historical_data(self, symbol: str, hours: int = 168) -> pd.DataFrame:
        """Generate historical price data (simulated for demo)"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)

        # Base prices for different cryptos
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "ADAUSDT": 0.5,
            "DOTUSDT": 8.0,
            "LINKUSDT": 15.0,
            "LTCUSDT": 100,
            "XRPUSDT": 0.6,
            "BCHUSDT": 200,
            "BNBUSDT": 300,
            "SOLUSDT": 100,
        }

        base_price = base_prices.get(symbol, 100)

        # Generate price walk
        timestamps = pd.date_range(start=start_time, end=end_time, freq="1H")
        returns = np.random.normal(0, 0.02, len(timestamps))

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLCV data
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.normal(1000000, 200000, len(timestamps)),
            }
        )

        return df

    def analyze_cross_asset_relationships(self) -> Dict[str, Any]:
        """Perform comprehensive cross-asset relationship analysis"""
        logger.info("🔗 Starting GCN cross-asset relationship analysis...")

        # Get historical data for all assets
        price_data = {}
        for asset in self.assets:
            try:
                df = self.get_historical_data(
                    asset, hours=self.config.lookback_steps + 24
                )
                price_data[asset] = df
            except Exception as e:
                logger.error(f"Error getting data for {asset}: {e}")

        if not price_data:
            logger.error("No price data available")
            return {}

        # Build graph structure
        adj_matrix = self.build_correlation_graph(price_data)

        # Prepare features
        features = self.prepare_features(price_data)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features, adj_matrix)

        # Process outputs
        predictions = outputs["predictions"].cpu().numpy()[0]  # Remove batch dimension
        relationships = outputs["relationship_scores"].cpu().numpy()[0]

        # Create analysis results
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "Graph Convolutional Network",
            "assets": self.assets,
            "predictions": {},
            "relationships": {},
            "graph_metrics": {},
            "market_insights": {},
        }

        # Asset predictions
        for i, asset in enumerate(self.assets):
            if i < len(predictions):
                pred_values = predictions[i].tolist()
                current_price = (
                    price_data[asset]["close"].iloc[-1] if asset in price_data else 100
                )

                analysis["predictions"][asset] = {
                    "current_price": current_price,
                    "predictions": pred_values,
                    "prediction_hours": list(
                        range(1, self.config.prediction_horizon + 1)
                    ),
                    "expected_return": (
                        (pred_values[-1] - current_price) / current_price
                        if pred_values
                        else 0
                    ),
                    "volatility_forecast": np.std(pred_values) if pred_values else 0,
                }

        # Relationship analysis
        strong_relationships = []
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i < j and i < len(relationships) and j < len(relationships[0]):
                    strength = relationships[i][j]
                    if strength > 0.7:  # Strong relationship threshold
                        strong_relationships.append(
                            {
                                "asset1": asset1,
                                "asset2": asset2,
                                "strength": float(strength),
                                "relationship_type": (
                                    "positive" if strength > 0.8 else "moderate"
                                ),
                            }
                        )

        analysis["relationships"]["strong_correlations"] = strong_relationships
        analysis["relationships"]["average_connectivity"] = float(
            np.mean(relationships)
        )

        # Graph metrics
        G = nx.from_numpy_array(adj_matrix.cpu().numpy())
        analysis["graph_metrics"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "clustering_coefficient": nx.average_clustering(G),
            "centrality_scores": {
                asset: float(centrality)
                for asset, centrality in zip(
                    self.assets, nx.degree_centrality(G).values()
                )
            },
        }

        # Market insights
        analysis["market_insights"] = {
            "most_connected_asset": self.assets[
                np.argmax(np.sum(relationships, axis=1))
            ],
            "market_cohesion": float(np.mean(relationships)),
            "volatility_cluster": [
                asset
                for asset in self.assets
                if analysis["predictions"].get(asset, {}).get("volatility_forecast", 0)
                > 0.05
            ],
            "growth_leaders": [
                asset
                for asset in self.assets
                if analysis["predictions"].get(asset, {}).get("expected_return", 0)
                > 0.02
            ],
        }

        # Store results
        self.store_analysis(analysis)

        logger.info("✅ GCN cross-asset analysis completed")
        return analysis

    def store_analysis(self, analysis: Dict[str, Any]):
        """Store analysis results in Redis"""
        if not self.redis_client:
            return

        try:
            # Store main analysis
            self.redis_client.setex(
                "gcn_analysis", 3600, json.dumps(analysis, default=str)  # 1 hour expiry
            )

            # Store relationship matrix
            if "relationships" in analysis:
                self.redis_client.setex(
                    "asset_relationships",
                    3600,
                    json.dumps(analysis["relationships"], default=str),
                )

            # Store predictions
            if "predictions" in analysis:
                self.redis_client.setex(
                    "gcn_predictions",
                    1800,  # 30 minutes expiry
                    json.dumps(analysis["predictions"], default=str),
                )

            logger.info("💾 GCN analysis stored in Redis")

        except Exception as e:
            logger.error(f"Error storing GCN analysis: {e}")

    def get_stored_analysis(self) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis from Redis"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get("gcn_analysis")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving GCN analysis: {e}")

        return None

    def predict_portfolio_relationships(
        self, portfolio_assets: List[str]
    ) -> Dict[str, Any]:
        """Analyze relationships for a specific portfolio"""
        analysis = self.get_stored_analysis()
        if not analysis:
            analysis = self.analyze_cross_asset_relationships()

        portfolio_relationships = {}

        for asset1 in portfolio_assets:
            for asset2 in portfolio_assets:
                if asset1 != asset2:
                    # Find relationship strength
                    relationships = analysis.get("relationships", {}).get(
                        "strong_correlations", []
                    )

                    strength = 0.0
                    for rel in relationships:
                        if (rel["asset1"] == asset1 and rel["asset2"] == asset2) or (
                            rel["asset1"] == asset2 and rel["asset2"] == asset1
                        ):
                            strength = rel["strength"]
                            break

                    if strength > 0.5:
                        portfolio_relationships[f"{asset1}-{asset2}"] = {
                            "strength": strength,
                            "diversification_benefit": 1.0 - strength,
                            "recommendation": (
                                "reduce_weight" if strength > 0.8 else "maintain"
                            ),
                        }

        return {
            "portfolio_assets": portfolio_assets,
            "relationships": portfolio_relationships,
            "diversification_score": (
                np.mean(
                    [
                        rel["diversification_benefit"]
                        for rel in portfolio_relationships.values()
                    ]
                )
                if portfolio_relationships
                else 1.0
            ),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """Demo function for GCN analysis"""
    print("🚀 Initializing Graph Neural Networks for Cross-Asset Analysis")
    print("=" * 80)

    # Initialize GCN analyzer
    config = GCNConfig(
        num_assets=10,
        hidden_dim=64,
        num_layers=3,
        lookback_steps=24,
        prediction_horizon=6,
    )

    analyzer = GCNCryptoAnalyzer(config)

    # Run analysis
    analysis = analyzer.analyze_cross_asset_relationships()

    if analysis:
        print("✅ GCN Analysis Results:")
        print(f"📊 Assets analyzed: {len(analysis['assets'])}")
        print(
            f"🔗 Strong relationships found: {len(analysis['relationships']['strong_correlations'])}"
        )
        print(
            f"📈 Market cohesion: {analysis['market_insights']['market_cohesion']:.3f}"
        )
        print(
            f"🎯 Most connected asset: {analysis['market_insights']['most_connected_asset']}"
        )

        # Test portfolio analysis
        test_portfolio = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        portfolio_analysis = analyzer.predict_portfolio_relationships(test_portfolio)

        print(f"\n💼 Portfolio Analysis for {test_portfolio}:")
        print(
            f"📊 Diversification score: {portfolio_analysis['diversification_score']:.3f}"
        )
        print(f"🔗 Portfolio relationships: {len(portfolio_analysis['relationships'])}")

    print("\n🎉 GCN Cross-Asset Analysis Complete!")


if __name__ == "__main__":
    main()
