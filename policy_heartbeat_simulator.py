#!/usr/bin/env python3
"""
Policy Daemon Heartbeat Simulator
Simulates SAC-DiF policy daemon by setting heartbeat every 1-2 seconds
"""

import redis
import time
import random
from datetime import datetime


def run_policy_heartbeat_simulator():
    """Run policy heartbeat simulator"""
    try:
        # Connect to Redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connected for policy heartbeat simulator")

        print("🤖 Starting Policy Daemon Heartbeat Simulator...")
        print("   This will set 'policy:ping' key every 1-2 seconds")
        print("   Press Ctrl+C to stop")

        counter = 0
        while True:
            try:
                # Set heartbeat with 2 second expiry (SETEX policy:ping 2 ok)
                r.setex("policy:ping", 2, "ok")

                # Also simulate some policy activity
                if counter % 5 == 0:  # Every 5th heartbeat
                    policy_data = {
                        "timestamp": int(time.time()),
                        "entropy": round(random.uniform(0.8, 1.8), 3),
                        "q_spread": round(random.uniform(30, 80), 1),
                        "action_count": counter,
                        "status": "active",
                    }
                    r.xadd("policy:actions", policy_data, maxlen=100)

                counter += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 💓 Policy heartbeat #{counter} sent")

                # Random interval between 1-2 seconds
                time.sleep(random.uniform(1.0, 2.0))

            except KeyboardInterrupt:
                print("\n🛑 Stopping policy heartbeat simulator...")
                break
            except Exception as e:
                print(f"❌ Error in heartbeat: {e}")
                time.sleep(1)

    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")


if __name__ == "__main__":
    run_policy_heartbeat_simulator()
