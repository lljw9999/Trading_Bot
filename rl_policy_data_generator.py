#!/usr/bin/env python3
"""
RL Policy Data Generator
Generates realistic entropy and Q-spread data for the RL Policy Health dashboard
"""

import redis
import time
import random
import json
from datetime import datetime


def generate_rl_policy_data():
    """Generate and store RL policy monitoring data"""
    try:
        # Connect to Redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connected for RL policy data generation")

        print("🤖 Starting RL Policy Data Generator...")
        print("   Generating entropy and Q-spread data every 5-15 seconds")
        print("   Press Ctrl+C to stop")

        counter = 0
        while True:
            try:
                # Generate realistic entropy values (0.1 to 2.0)
                # Lower values indicate policy collapse risk
                entropy_value = max(0.1, random.gauss(1.2, 0.4))  # Mean 1.2, std 0.4
                entropy_value = min(2.0, entropy_value)  # Cap at 2.0

                # Generate Q-value spread (20 to 100)
                # Higher spread indicates more diverse Q-values (good exploration)
                qspread_value = max(20, random.gauss(55, 15))  # Mean 55, std 15
                qspread_value = min(100, qspread_value)  # Cap at 100

                timestamp = int(time.time() * 1000)  # Milliseconds

                # Store entropy data
                entropy_data = {
                    "timestamp": timestamp,
                    "value": round(entropy_value, 3),
                    "policy_id": f"policy_{counter % 5}",  # Simulate multiple policies
                    "action_count": random.randint(50, 200),
                }

                # Store Q-spread data
                qspread_data = {
                    "timestamp": timestamp,
                    "value": round(qspread_value, 1),
                    "q_min": round(random.uniform(0, 20), 2),
                    "q_max": round(random.uniform(80, 120), 2),
                    "q_mean": round(random.uniform(40, 70), 2),
                }

                # Add to Redis streams (maxlen to prevent memory bloat)
                r.xadd("policy:entropy", entropy_data, maxlen=100)
                r.xadd("policy:qspread", qspread_data, maxlen=100)

                # Also store current values for quick access
                r.hset(
                    "policy:current",
                    mapping={
                        "entropy": entropy_value,
                        "qspread": qspread_value,
                        "timestamp": timestamp,
                        "collapse_risk": (
                            "HIGH"
                            if entropy_value < 0.3
                            else "MEDIUM" if entropy_value < 0.8 else "LOW"
                        ),
                    },
                )

                counter += 1
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp_str}] 📊 Policy Data #{counter}:")
                print(
                    f"   - Entropy: {entropy_value:.3f} ({'⚠️  HIGH RISK' if entropy_value < 0.3 else '✅ OK'})"
                )
                print(f"   - Q-Spread: {qspread_value:.1f}")

                # Random interval between 5-15 seconds
                sleep_time = random.uniform(5, 15)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n🛑 Stopping RL policy data generator...")
                break
            except Exception as e:
                print(f"❌ Error generating policy data: {e}")
                time.sleep(5)

    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")


if __name__ == "__main__":
    generate_rl_policy_data()
