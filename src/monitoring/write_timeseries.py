#!/usr/bin/env python3
"""
RedisTimeSeries Writer for Trading System

Writes time series data for Grafana dashboards as specified in Task F:
- edge_blended_bps:<symbol> - Blended edge in basis points
- position_size_usd:<symbol> - Position size in USD
- var_pct:<symbol> - VaR percentage

Integrates with Risk Harmoniser to capture real-time risk metrics.

Falls back to regular Redis operations if TimeSeries module not available.
"""

import time
import logging
import json
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from decimal import Decimal

import redis
import redis.exceptions

logger = logging.getLogger(__name__)


class TimeSeriesWriter:
    """RedisTimeSeries writer for trading system metrics with fallback support."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize TimeSeries writer with Redis connection."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.has_timeseries = False
        self._connect_redis()
        
        # Performance tracking
        self.writes_count = 0
        self.failed_writes = 0
        self.last_write_timestamp = 0.0
        
        # Buffer for batching writes
        self._write_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._max_buffer_size = 100
        
        logger.info("TimeSeriesWriter initialized")
    
    def _connect_redis(self):
        """Connect to Redis and check for TimeSeries module."""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            
            # Check if TimeSeries module is available
            try:
                modules = self.redis_client.execute_command("MODULE", "LIST")
                self.has_timeseries = any("timeseries" in str(module).lower() for module in modules)
            except:
                self.has_timeseries = False
            
            if self.has_timeseries:
                logger.info(f"Connected to Redis TimeSeries at {self.redis_url}")
            else:
                logger.warning(f"Connected to Redis at {self.redis_url} (TimeSeries module not available - using fallback)")
                
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _write_timeseries_metric(self, key: str, value: float, labels: Dict[str, str] = None):
        """Write metric using TimeSeries commands."""
        if not self.redis_client or not self.has_timeseries:
            return False
            
        try:
            timestamp = int(time.time() * 1000)
            
            # Create timeseries if it doesn't exist
            try:
                cmd = ["TS.CREATE", key]
                if labels:
                    cmd.append("LABELS")
                    for k, v in labels.items():
                        cmd.extend([k, v])
                self.redis_client.execute_command(*cmd)
            except redis.exceptions.ResponseError as e:
                if "key already exists" not in str(e).lower():
                    raise
            
            # Add data point
            cmd = ["TS.ADD", key, timestamp, value]
            if labels:
                cmd.append("LABELS")
                for k, v in labels.items():
                    cmd.extend([k, v])
            self.redis_client.execute_command(*cmd)
            return True
            
        except Exception as e:
            logger.debug(f"TimeSeries write failed for {key}: {e}")
            return False
    
    def _write_fallback_metric(self, key: str, value: float, labels: Dict[str, str] = None):
        """Fallback to regular Redis operations."""
        if not self.redis_client:
            return False
            
        try:
            timestamp = int(time.time())
            
            # Store as hash with timestamp
            hash_key = f"metrics:{key}"
            metric_data = {
                "value": value,
                "timestamp": timestamp
            }
            if labels:
                metric_data.update(labels)
            
            # Store current value
            self.redis_client.hset(hash_key, mapping=metric_data)
            self.redis_client.expire(hash_key, 7 * 24 * 3600)  # 7 days expiry
            
            # Also store in sorted set for time-based queries
            ts_key = f"ts:{key}"
            self.redis_client.zadd(ts_key, {json.dumps({"value": value, **labels}): timestamp})
            
            # Keep only recent data (last 24 hours)
            cutoff = timestamp - 24 * 3600
            self.redis_client.zremrangebyscore(ts_key, 0, cutoff)
            
            return True
            
        except Exception as e:
            logger.debug(f"Fallback write failed for {key}: {e}")
            return False
    
    def _write_metric(self, key: str, value: float, labels: Dict[str, str] = None):
        """Write metric using TimeSeries or fallback method."""
        # Try TimeSeries first, then fallback
        if self._write_timeseries_metric(key, value, labels):
            return True
        return self._write_fallback_metric(key, value, labels)
    
    def write_edge_blended(self, symbol: str, edge_bps: float, active_model: str):
        """Write blended edge metric as specified in Task F."""
        if not self.redis_client:
            return False
        
        try:
            key = f"edge_blended_bps:{symbol}"
            labels = {"symbol": symbol, "metric": "edge_bps", "model": active_model}
            
            success = self._write_metric(key, edge_bps, labels)
            
            if success:
                self.writes_count += 1
                self.last_write_timestamp = time.time()
                logger.debug(f"Wrote edge_blended_bps:{symbol} = {edge_bps:.2f} (model: {active_model})")
            else:
                self.failed_writes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to write edge_blended for {symbol}: {e}")
            self.failed_writes += 1
            return False
    
    def write_position_size(self, symbol: str, size_usd: float):
        """Write position size metric as specified in Task F."""
        if not self.redis_client:
            return False
        
        try:
            key = f"position_size_usd:{symbol}"
            labels = {"symbol": symbol, "metric": "position_usd"}
            
            success = self._write_metric(key, size_usd, labels)
            
            if success:
                self.writes_count += 1
                logger.debug(f"Wrote position_size_usd:{symbol} = ${size_usd:.0f}")
            else:
                self.failed_writes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to write position_size for {symbol}: {e}")
            self.failed_writes += 1
            return False
    
    def write_var_pct(self, symbol: str, var_pct: float):
        """Write VaR percentage metric as specified in Task F."""
        if not self.redis_client:
            return False
        
        try:
            key = f"var_pct:{symbol}"
            labels = {"symbol": symbol, "metric": "var_pct"}
            
            success = self._write_metric(key, var_pct, labels)
            
            if success:
                self.writes_count += 1
                logger.debug(f"Wrote var_pct:{symbol} = {var_pct:.2f}%")
            else:
                self.failed_writes += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to write var_pct for {symbol}: {e}")
            self.failed_writes += 1
            return False
    
    def write_risk_metrics_batch(self, symbol: str, edge_bps: float, 
                                size_usd: float, var_pct: float, 
                                active_model: str) -> bool:
        """Write all risk metrics in a single batch for efficiency."""
        if not self.redis_client:
            return False
        
        try:
            success_count = 0
            
            # Write each metric individually for better error handling
            if self.write_edge_blended(symbol, edge_bps, active_model):
                success_count += 1
            if self.write_position_size(symbol, size_usd):
                success_count += 1
            if self.write_var_pct(symbol, var_pct):
                success_count += 1
            
            # Consider batch successful if at least 2/3 metrics written
            batch_success = success_count >= 2
            
            if batch_success:
                logger.debug(f"Batch wrote risk metrics for {symbol}: "
                           f"edge={edge_bps:.2f}bps, size=${size_usd:.0f}, var={var_pct:.2f}%")
            else:
                logger.warning(f"Batch write partially failed for {symbol}: {success_count}/3 metrics written")
            
            return batch_success
            
        except Exception as e:
            logger.error(f"Failed to batch write risk metrics for {symbol}: {e}")
            self.failed_writes += 3
            return False
    
    def write_model_switch_event(self, symbol: str, old_model: str, new_model: str, latency_ms: float):
        """Write model switch event to Redis stream for switch log panel."""
        if not self.redis_client:
            return False
        
        try:
            stream_name = "model.switch.log"
            event_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "old_model": old_model,
                "new_model": new_model,
                "latency_ms": latency_ms
            }
            
            # Add to Redis stream
            self.redis_client.xadd(stream_name, event_data)
            
            # Also increment counter for alert rule
            counter_key = "model_switch_total"
            self.redis_client.incr(counter_key)
            self.redis_client.expire(counter_key, 300)  # 5 minute expiry
            
            logger.info(f"Recorded model switch: {symbol} {old_model} → {new_model} ({latency_ms:.1f}ms)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write model switch event: {e}")
            return False
    
    def get_timeseries_data(self, key: str, from_timestamp: int, to_timestamp: int) -> List[tuple]:
        """Retrieve timeseries data for debugging/validation."""
        if not self.redis_client:
            return []
        
        try:
            result = self.redis_client.execute_command(
                "TS.RANGE", key, from_timestamp, to_timestamp
            )
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve timeseries data for {key}: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get writer performance statistics."""
        return {
            "writes_count": self.writes_count,
            "failed_writes": self.failed_writes,
            "success_rate": (self.writes_count / max(1, self.writes_count + self.failed_writes)) * 100,
            "last_write_timestamp": self.last_write_timestamp,
            "redis_connected": self.redis_client is not None
        }
    
    def health_check(self) -> bool:
        """Check if writer is healthy and can write to Redis."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global instance for easy import
_writer_instance: Optional[TimeSeriesWriter] = None


def get_timeseries_writer(redis_url: str = "redis://localhost:6379/0") -> TimeSeriesWriter:
    """Get or create global TimeSeries writer instance."""
    global _writer_instance
    
    if _writer_instance is None:
        _writer_instance = TimeSeriesWriter(redis_url)
    
    return _writer_instance


def write_risk_metrics(symbol: str, edge_bps: float, size_usd: float, 
                      var_pct: float, active_model: str) -> bool:
    """Convenience function to write risk metrics."""
    writer = get_timeseries_writer()
    return writer.write_risk_metrics_batch(symbol, edge_bps, size_usd, var_pct, active_model)


if __name__ == "__main__":
    # Test the TimeSeries writer
    import random
    
    logging.basicConfig(level=logging.DEBUG)
    writer = TimeSeriesWriter()
    
    # Test writes
    symbols = ["BTC-USD", "ETH-USD", "AAPL"]
    models = ["tlob_tiny", "patchtst_small", "timesnet_base"]
    
    for symbol in symbols:
        edge_bps = random.uniform(-20, 20)
        size_usd = random.uniform(1000, 50000)
        var_pct = random.uniform(0.5, 3.0)
        model = random.choice(models)
        
        success = writer.write_risk_metrics_batch(symbol, edge_bps, size_usd, var_pct, model)
        print(f"Wrote {symbol}: success={success}")
    
    # Test model switch
    writer.write_model_switch_event("BTC-USD", "tlob_tiny", "patchtst_small", 15.2)
    
    print(f"Performance stats: {writer.get_performance_stats()}") 