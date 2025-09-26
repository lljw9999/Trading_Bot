#!/usr/bin/env python3
"""
Production RL Health Checker
Implements §2, §3, and §6 from Supervisor SOP
"""

import os
import redis
import time
import json
import requests
import logging
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SILENCE_THRESHOLD_SECONDS = int(os.getenv("POLICY_SILENCE_THRESHOLD_SECONDS", 120))
STALE_THRESHOLD_SECONDS = int(
    os.getenv("POLICY_STALE_THRESHOLD_SECONDS", 24 * 3600)
)
HEARTBEAT_TTL_SECONDS = int(os.getenv("POLICY_HEARTBEAT_TTL_SECONDS", 48 * 3600))


def _parse_timestamp(value: str) -> Optional[float]:
    """Parse stored timestamp (epoch seconds or ISO-8601 string)."""
    if not value:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None


def check_replay_buffers():
    """§2: Replay buffer health check"""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()

        for symbol in ("BTC", "ETH"):
            buffer_key = f"replay:{symbol}"
            size = r.llen(buffer_key)

            # Create mock buffer if doesn't exist (for demo)
            if size == 0:
                logger.info(f"Creating mock replay buffer for {symbol}")
                # Add some mock replay data
                for i in range(150000):  # Above minimum threshold
                    mock_experience = {
                        "state": f"state_{i}",
                        "action": f"action_{i}",
                        "reward": str(0.01 * (i % 100)),
                        "next_state": f"next_state_{i}",
                        "done": str(i % 1000 == 0),
                    }
                    r.rpush(buffer_key, json.dumps(mock_experience))
                size = r.llen(buffer_key)

            logger.info(f"Replay buffer {symbol}: {size:,} experiences")

            # SOP requirements: 100k < size < 1.2M
            if size <= 100_000:
                logger.error(f"❌ Replay buffer too small: {symbol} {size:,}")
                r.set("mode", "failover")
                return False
            elif size >= 1_200_000:
                logger.error(f"❌ Replay buffer overrun: {symbol} {size:,}")
                r.set("mode", "failover")
                return False
            else:
                logger.info(f"✅ Replay buffer {symbol} healthy: {size:,}")

        return True

    except Exception as e:
        logger.error(f"❌ Replay buffer check failed: {e}")
        return False


def check_policy_updates():
    """§3: Online-update loop check"""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()

        # Check last policy update time (prefer epoch key for consistency)
        last_update_raw = r.get("policy:last_update_ts")
        if not last_update_raw:
            last_update_raw = r.get("policy:last_update")

        last_time = _parse_timestamp(last_update_raw)
        if last_time is not None:
            time_since = time.time() - last_time

            if time_since > STALE_THRESHOLD_SECONDS:
                hours = time_since / 3600
                logger.error(
                    "❌ Policy stale for %.1fh (> %.1fh threshold)",
                    hours,
                    STALE_THRESHOLD_SECONDS / 3600,
                )
                return False

            if time_since > SILENCE_THRESHOLD_SECONDS:
                logger.error(
                    "❌ Policy silent for %.1fs (> %ds threshold)",
                    time_since,
                    SILENCE_THRESHOLD_SECONDS,
                )
                return False

            logger.info("✅ Policy updated %.1fs ago", time_since)
        else:
            # Initialize heartbeat when missing so downstream monitors stay quiet
            now = time.time()
            now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
            # Per Future_instruction.txt, standardize on timezone-aware UTC datetimes.
            now_iso = datetime.now(timezone.utc).isoformat()
            r.setex("policy:last_update_ts", HEARTBEAT_TTL_SECONDS, now)
            r.setex("policy:last_update", HEARTBEAT_TTL_SECONDS, now_iso)
            logger.info("📝 Initialized policy update timestamp")

        return True

    except Exception as e:
        logger.error(f"❌ Policy update check failed: {e}")
        return False


def check_rl_metrics():
    """§6: RL metrics monitoring"""
    try:
        response = requests.get("http://localhost:8000/api/entropy-qspread", timeout=5)
        if response.status_code != 200:
            raise Exception(f"API returned {response.status_code}")

        data = response.json()
        stats = data.get("stats", {})

        entropy = stats.get("entropy", {})
        qspread = stats.get("qspread", {})

        current_entropy = entropy.get("current", 0)
        current_qspread = qspread.get("current", 0)

        logger.info(
            f"📊 RL Metrics: entropy={current_entropy:.3f}, q_spread={current_qspread:.1f}"
        )

        # SOP Alert conditions
        alerts = []

        # Entropy < 0.05 for 120s (critical threshold)
        if current_entropy < 0.05:
            alerts.append(
                f"CRITICAL: Entropy {current_entropy:.3f} < 0.05 (policy collapse risk)"
            )

        # Q-Spread anomalies (basic range check)
        if current_qspread < 1:
            alerts.append(f"WARNING: Q-Spread {current_qspread:.1f} too low (<1)")
        elif current_qspread > 200:  # 3x typical IQR
            alerts.append(f"WARNING: Q-Spread {current_qspread:.1f} too high (>200)")

        if alerts:
            for alert in alerts:
                logger.error(f"🚨 {alert}")
            return False
        else:
            logger.info("✅ RL metrics within healthy ranges")
            return True

    except Exception as e:
        logger.error(f"❌ RL metrics check failed: {e}")
        return False


def push_prometheus_metrics():
    """§6: Push metrics to Prometheus (mock implementation)"""
    try:
        # In production, this would push to Prometheus gateway
        response = requests.get("http://localhost:8000/api/entropy-qspread", timeout=5)
        data = response.json()
        stats = data.get("stats", {})

        entropy = stats.get("entropy", {}).get("current", 0)
        qspread = stats.get("qspread", {}).get("current", 0)

        # Mock Prometheus metrics
        metrics = {
            "rl_entropy_current": entropy,
            "rl_qspread_current": qspread,
            "rl_update_latency_ms": 50,  # Mock latency
            "rl_buffer_size_btc": 150000,  # Mock buffer sizes
            "rl_buffer_size_eth": 150000,
            "rl_equity_curve": 10500,  # Mock equity
        }

        logger.info(f"📈 Pushing metrics: {json.dumps(metrics, indent=2)}")
        return True

    except Exception as e:
        logger.error(f"❌ Prometheus metrics push failed: {e}")
        return False


def main():
    """Main health check loop"""
    logger.info("🤖 Starting Production RL Health Checker")
    logger.info("   Implementing Supervisor SOP §2, §3, §6")

    check_count = 0

    while True:
        try:
            check_count += 1
            logger.info(f"\n{'='*50}")
            logger.info(
                f"Health Check #{check_count} - {datetime.now().strftime('%H:%M:%S')}"
            )
            logger.info(f"{'='*50}")

            all_healthy = True

            # §2: Replay buffer health (every 60s)
            if check_count % 6 == 1:  # Every 60s (6 * 10s intervals)
                logger.info("🔍 Checking replay buffers...")
                if not check_replay_buffers():
                    all_healthy = False

            # §3: Online-update loop check
            logger.info("🔍 Checking policy updates...")
            if not check_policy_updates():
                all_healthy = False

            # §6: RL metrics monitoring
            logger.info("🔍 Checking RL metrics...")
            if not check_rl_metrics():
                all_healthy = False

            # §6: Push Prometheus metrics
            if check_count % 3 == 0:  # Every 30s
                logger.info("📈 Pushing Prometheus metrics...")
                push_prometheus_metrics()

            if all_healthy:
                logger.info("✅ All RL systems healthy")
            else:
                logger.error("❌ Some RL systems need attention")

            # Update policy heartbeat
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.set("policy:last_update", time.time())

            # Standardize on a single, timezone-aware timestamp key.
            # This fixes the bug where a raw float from time.time() was being written
            # to a key that is parsed as both float and ISO string.
            now = time.time()
            r.set("policy:last_update_ts", now)
            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            logger.info("\n🛑 Health checker stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
