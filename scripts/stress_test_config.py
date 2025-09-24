"""
Configuration script for adversarial stress testing
Adds --adversary flag support for offline back-tests
"""

import argparse
import json
from typing import Dict, Any


def create_stress_test_config() -> Dict[str, Any]:
    """Create configuration for stress testing with ghost trader."""

    parser = argparse.ArgumentParser(description="SAC-DiF Stress Testing Configuration")
    parser.add_argument(
        "--adversary",
        type=bool,
        default=False,
        help="Enable adversarial ghost trader for stress testing",
    )
    parser.add_argument(
        "--ghost-prob",
        type=float,
        default=0.05,
        help="Ghost trader manipulation probability (default: 0.05)",
    )
    parser.add_argument(
        "--spoof-ticks",
        type=int,
        default=2,
        help="Number of ticks for price spoofing (default: 2)",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=100,
        help="Number of episodes for stress testing",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="stress_test_results.json",
        help="Output file for test results",
    )

    args = parser.parse_args()

    config = {
        "adversary_enabled": args.adversary,
        "ghost_trader": {
            "probability": args.ghost_prob,
            "spoof_ticks": args.spoof_ticks,
        },
        "test_parameters": {
            "episodes": args.test_episodes,
            "output_file": args.output_file,
        },
    }

    if config["adversary_enabled"]:
        print("🏴‍☠️ Adversarial stress testing enabled")
        print(
            f"   Ghost trader probability: {config['ghost_trader']['probability']:.1%}"
        )
        print(f"   Spoof ticks: {config['ghost_trader']['spoof_ticks']}")
        print(f"   Test episodes: {config['test_parameters']['episodes']}")
    else:
        print("📊 Standard testing mode")

    return config


if __name__ == "__main__":
    config = create_stress_test_config()
    print(f"\nConfig: {json.dumps(config, indent=2)}")
