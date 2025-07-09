#!/usr/bin/env python3
"""
Signal Multiplexer/Demultiplexer for Dynamic Model Routing

Wraps the existing Feature Bus to route incoming ticks to the active model's 
predict() function and tags output edges with model_id before Risk sizing.

This enables seamless switching between different alpha models based on
instrument type and trading horizon.

Enhanced with model switching event logging for monitoring.
"""

import time
import logging
import json
import asyncio
import redis
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .router import ModelRouter, create_model_router

# Import TimeSeries writer for model switch events
try:
    from ..monitoring.write_timeseries import get_timeseries_writer
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    logging.warning("TimeSeries writer not available - switch events won't be logged")

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Model prediction with metadata for tracking."""
    symbol: str
    model_id: str
    edge_bps: float
    confidence: float
    timestamp: float
    horizon_ms: int
    features: Dict[str, Any]
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "model_id": self.model_id,
            "edge_bps": self.edge_bps,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "horizon_ms": self.horizon_ms,
            "features": self.features,
            "latency_ms": self.latency_ms
        }


@dataclass
class TickData:
    """Standardized tick data for model input."""
    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    @classmethod
    def from_market_data(cls, symbol: str, data: Dict[str, Any]) -> 'TickData':
        """Create TickData from raw market data."""
        return cls(
            symbol=symbol,
            price=float(data.get('price', data.get('last', 0))),
            volume=float(data.get('volume', 0)),
            timestamp=data.get('timestamp', time.time()),
            bid=float(data['bid']) if data.get('bid') else None,
            ask=float(data['ask']) if data.get('ask') else None,
            bid_size=float(data['bid_size']) if data.get('bid_size') else None,
            ask_size=float(data['ask_size']) if data.get('ask_size') else None
        )


class ModelRegistry:
    """Registry of available alpha models for dynamic routing."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
    def register_model(self, model_id: str, model_instance: Any, metadata: Dict[str, Any]):
        """Register a model with the registry."""
        self.models[model_id] = model_instance
        self.model_metadata[model_id] = metadata
        logger.info(f"Registered model {model_id}: {metadata.get('description', 'N/A')}")
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model instance by ID."""
        return self.models.get(model_id)
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID."""
        return self.model_metadata.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())


class SignalMux:
    """
    Signal Multiplexer/Demultiplexer for dynamic model routing.
    
    Wraps the Feature Bus to:
    1. Route incoming ticks to the appropriate model based on routing rules
    2. Execute model predictions with performance tracking  
    3. Tag output with model_id for downstream processing
    4. Publish tagged signals to Redis channels
    """
    
    def __init__(self, 
                 router: ModelRouter,
                 model_registry: ModelRegistry,
                 redis_client: Optional[redis.Redis] = None,
                 feature_bus: Optional[Any] = None):
        """Initialize Signal Mux with router and model registry."""
        self.router = router
        self.model_registry = model_registry
        self.redis_client = redis_client or redis.Redis.from_url("redis://localhost:6379/0")
        self.feature_bus = feature_bus
        
        # Performance tracking
        self.prediction_count = 0
        self.total_latency_ms = 0.0
        self.model_usage_stats: Dict[str, int] = {}
        self.switching_events: List[Dict[str, Any]] = []
        
        # Current model per symbol (for switching detection)
        self.current_models: Dict[str, str] = {}
        
        logger.info("SignalMux initialized with dynamic model routing")
    
    async def process_tick(self, tick_data: TickData, horizon_ms: int) -> Optional[ModelPrediction]:
        """
        Process incoming tick through the appropriate model.
        
        Args:
            tick_data: Standardized tick data
            horizon_ms: Trading horizon in milliseconds
            
        Returns:
            ModelPrediction with tagged model_id or None if processing fails
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Route to appropriate model
            model_id = self.router.select_model(tick_data.symbol, horizon_ms)
            
            # Step 2: Detect model switching
            previous_model = self.current_models.get(tick_data.symbol)
            if previous_model and previous_model != model_id:
                switching_event = {
                    "symbol": tick_data.symbol,
                    "from_model": previous_model,
                    "to_model": model_id,
                    "horizon_ms": horizon_ms,
                    "timestamp": tick_data.timestamp,
                    "price": tick_data.price
                }
                self.switching_events.append(switching_event)
                logger.info(f"Model switch for {tick_data.symbol}: {previous_model} → {model_id}")
                
                # Publish switching event
                await self._publish_switching_event(switching_event)
            
            self.current_models[tick_data.symbol] = model_id
            
            # Step 3: Get model instance
            model = self.model_registry.get_model(model_id)
            if not model:
                logger.warning(f"Model {model_id} not found in registry for {tick_data.symbol}")
                return None
            
            # Step 4: Prepare features (integrate with Feature Bus if available)
            features = await self._prepare_features(tick_data, horizon_ms)
            
            # Step 5: Execute model prediction
            prediction_start = time.perf_counter()
            edge_bps, confidence = await self._execute_model_prediction(
                model, model_id, tick_data, features, horizon_ms
            )
            prediction_latency_ms = (time.perf_counter() - prediction_start) * 1000
            
            # Step 6: Create tagged prediction
            prediction = ModelPrediction(
                symbol=tick_data.symbol,
                model_id=model_id,
                edge_bps=edge_bps,
                confidence=confidence,
                timestamp=tick_data.timestamp,
                horizon_ms=horizon_ms,
                features=features,
                latency_ms=prediction_latency_ms
            )
            
            # Step 7: Publish tagged signal
            await self._publish_tagged_signal(prediction)
            
            # Step 8: Update performance metrics
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(model_id, total_latency_ms)
            
            # Notify router of successful selection
            self.router.publish_selection(
                tick_data.symbol, 
                model_id, 
                {
                    "horizon_ms": horizon_ms,
                    "edge_bps": edge_bps,
                    "confidence": confidence,
                    "latency_ms": total_latency_ms
                }
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Signal processing failed for {tick_data.symbol}: {e}")
            return None
    
    async def _prepare_features(self, tick_data: TickData, horizon_ms: int) -> Dict[str, Any]:
        """Prepare features for model input, integrating with Feature Bus if available."""
        features = {
            "price": tick_data.price,
            "volume": tick_data.volume,
            "timestamp": tick_data.timestamp,
            "horizon_ms": horizon_ms
        }
        
        # Add bid/ask data if available
        if tick_data.bid and tick_data.ask:
            features.update({
                "bid": tick_data.bid,
                "ask": tick_data.ask,
                "spread": tick_data.ask - tick_data.bid,
                "mid_price": (tick_data.bid + tick_data.ask) / 2
            })
            
            if tick_data.bid_size and tick_data.ask_size:
                features.update({
                    "bid_size": tick_data.bid_size,
                    "ask_size": tick_data.ask_size,
                    "imbalance": (tick_data.bid_size - tick_data.ask_size) / (tick_data.bid_size + tick_data.ask_size)
                })
        
        # Integrate with Feature Bus if available
        if self.feature_bus:
            try:
                fb_features = await self._get_feature_bus_data(tick_data.symbol, horizon_ms)
                features.update(fb_features)
            except Exception as e:
                logger.debug(f"Feature Bus integration failed for {tick_data.symbol}: {e}")
        
        return features
    
    async def _get_feature_bus_data(self, symbol: str, horizon_ms: int) -> Dict[str, Any]:
        """Get additional features from Feature Bus."""
        # This would integrate with the existing Feature Bus
        # For now, return empty dict as placeholder
        return {}
    
    async def _execute_model_prediction(self, 
                                       model: Any, 
                                       model_id: str, 
                                       tick_data: TickData, 
                                       features: Dict[str, Any],
                                       horizon_ms: int) -> tuple[float, float]:
        """Execute model prediction with proper error handling."""
        try:
            # Check if model has async predict method
            if hasattr(model, 'predict_async'):
                result = await model.predict_async(features)
            elif hasattr(model, 'predict'):
                # Run sync predict in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, model.predict, features)
            else:
                # Fallback: use dummy prediction
                logger.warning(f"Model {model_id} has no predict method, using dummy prediction")
                return 0.0, 0.5
            
            # Extract edge and confidence from result
            if isinstance(result, tuple):
                edge_bps, confidence = result
            elif isinstance(result, dict):
                edge_bps = result.get('edge_bps', 0.0)
                confidence = result.get('confidence', 0.5)
            else:
                # Single value result
                edge_bps = float(result)
                confidence = 0.6  # Default confidence
            
            return float(edge_bps), float(confidence)
            
        except Exception as e:
            logger.error(f"Model {model_id} prediction failed: {e}")
            # Return None tuple to signal failure
            raise Exception(f"Model prediction failed: {e}")  # Re-raise to trigger process_tick failure
    
    async def _publish_tagged_signal(self, prediction: ModelPrediction):
        """Publish tagged signal to Redis channels."""
        try:
            # Publish to symbol-specific channel
            symbol_channel = f"alpha.selected.{prediction.symbol}"
            message = json.dumps(prediction.to_dict())
            self.redis_client.publish(symbol_channel, message)
            
            # Publish to model-specific channel for monitoring
            model_channel = f"alpha.model.{prediction.model_id}"
            self.redis_client.publish(model_channel, message)
            
            logger.debug(f"Published tagged signal for {prediction.symbol} via {prediction.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish tagged signal: {e}")
    
    async def _publish_switching_event(self, switching_event: Dict[str, Any]):
        """Publish model switching event for monitoring and write to TimeSeries."""
        try:
            channel = "alpha.switching.events"
            message = json.dumps(switching_event)
            self.redis_client.publish(channel, message)
            
            # Write to TimeSeries for Grafana Switch Log panel (Task F)
            if TIMESERIES_AVAILABLE:
                ts_writer = get_timeseries_writer()
                ts_writer.write_model_switch_event(
                    symbol=switching_event['symbol'],
                    old_model=switching_event['from_model'],
                    new_model=switching_event['to_model'],
                    latency_ms=switching_event['latency_ms']
                )
                
                logger.debug(f"Wrote model switch event to TimeSeries: "
                           f"{switching_event['symbol']} "
                           f"{switching_event['from_model']} → {switching_event['to_model']}")
            
            logger.info(f"Published switching event: {switching_event['symbol']} "
                       f"{switching_event['from_model']} → {switching_event['to_model']}")
            
        except Exception as e:
            logger.error(f"Failed to publish switching event: {e}")
    
    def _update_performance_metrics(self, model_id: str, latency_ms: float):
        """Update performance tracking metrics."""
        self.prediction_count += 1
        self.total_latency_ms += latency_ms
        self.model_usage_stats[model_id] = self.model_usage_stats.get(model_id, 0) + 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_latency = self.total_latency_ms / max(1, self.prediction_count)
        
        return {
            "prediction_count": self.prediction_count,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self.total_latency_ms,
            "model_usage_stats": self.model_usage_stats.copy(),
            "switching_events_count": len(self.switching_events),
            "active_symbols": len(self.current_models),
            "router_stats": self.router.get_performance_stats()
        }
    
    def get_switching_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent model switching events."""
        return self.switching_events[-limit:]
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.prediction_count = 0
        self.total_latency_ms = 0.0
        self.model_usage_stats.clear()
        self.switching_events.clear()
        self.router.reset_performance_stats()


# Factory function for easy setup
def create_signal_mux(redis_url: str = "redis://localhost:6379/0",
                     router_config_path: Optional[str] = None,
                     feature_bus: Optional[Any] = None) -> SignalMux:
    """Factory function to create SignalMux with all dependencies."""
    
    # Create router
    router = create_model_router(redis_url, router_config_path)
    
    # Create model registry
    model_registry = ModelRegistry()
    
    # Register default models (placeholder - would be actual model instances)
    default_models = {
        "tlob_tiny": {
            "instance": DummyModel("tlob_tiny"),
            "metadata": {
                "description": "TLOB-Tiny for high-frequency crypto microstructure",
                "max_latency_ms": 3.0,
                "typical_accuracy": 0.52
            }
        },
        "patchtst_small": {
            "instance": DummyModel("patchtst_small"),
            "metadata": {
                "description": "PatchTST-Small for medium-frequency patterns",
                "max_latency_ms": 10.0,
                "typical_accuracy": 0.54
            }
        },
        "timesnet_base": {
            "instance": DummyModel("timesnet_base"),
            "metadata": {
                "description": "TimesNet-Base for intraday patterns",
                "max_latency_ms": 15.0,
                "typical_accuracy": 0.53
            }
        },
        "mamba_ts_small": {
            "instance": DummyModel("mamba_ts_small"),
            "metadata": {
                "description": "MambaTS-Small for long-term regime analysis",
                "max_latency_ms": 20.0,
                "typical_accuracy": 0.55
            }
        },
        "chronos_bolt_base": {
            "instance": DummyModel("chronos_bolt_base"),
            "metadata": {
                "description": "Chronos-Bolt-Base for macro-sensitive analysis",
                "max_latency_ms": 25.0,
                "typical_accuracy": 0.56
            }
        }
    }
    
    for model_id, model_data in default_models.items():
        model_registry.register_model(
            model_id, 
            model_data["instance"], 
            model_data["metadata"]
        )
    
    # Create Redis client
    redis_client = redis.Redis.from_url(redis_url)
    
    return SignalMux(router, model_registry, redis_client, feature_bus)


class DummyModel:
    """Dummy model for testing and development."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.call_count = 0
    
    def predict(self, features: Dict[str, Any]) -> tuple[float, float]:
        """Dummy prediction with model-specific characteristics."""
        self.call_count += 1
        
        # Simulate different model behaviors
        base_edge = features.get('price', 100) * 0.0001  # 1bp per $100
        
        if "tlob" in self.model_id:
            # High frequency, small edges
            edge_bps = base_edge * 0.5 + (hash(str(features.get('timestamp', 0))) % 100 - 50) * 0.1
            confidence = 0.52
        elif "patchtst" in self.model_id:
            # Medium frequency, moderate edges
            edge_bps = base_edge * 1.2 + (hash(str(features.get('timestamp', 0))) % 100 - 50) * 0.2
            confidence = 0.54
        elif "timesnet" in self.model_id:
            # Intraday patterns
            edge_bps = base_edge * 1.0 + (hash(str(features.get('timestamp', 0))) % 100 - 50) * 0.15
            confidence = 0.53
        elif "mamba" in self.model_id:
            # Long-term, higher edges
            edge_bps = base_edge * 1.8 + (hash(str(features.get('timestamp', 0))) % 100 - 50) * 0.3
            confidence = 0.55
        else:
            # Chronos default
            edge_bps = base_edge * 1.5 + (hash(str(features.get('timestamp', 0))) % 100 - 50) * 0.25
            confidence = 0.56
        
        return edge_bps, confidence
    
    async def predict_async(self, features: Dict[str, Any]) -> tuple[float, float]:
        """Async version of predict."""
        return self.predict(features) 