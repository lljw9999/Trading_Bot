#!/usr/bin/env python3
"""Test script for alpha impact learner."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.alpha_impact_learner import OnlineAlphaImpactLearner


async def test_learning():
    """Test the learning process."""
    learner = OnlineAlphaImpactLearner()

    print("🧪 Testing alpha impact learner...")
    print(f"Initial status: {learner.sample_count} samples")

    # Simulate 150 learning cycles
    for i in range(150):
        result = await learner.run_learning_cycle()

        if i % 25 == 0:
            print(
                f"Cycle {i}: {result.get('status', 'unknown')} - MAE: {result.get('mae', 0):.4f}"
            )

    # Show final status
    status = learner.get_learning_status()
    print(f"\n📊 Final status:")
    print(f"  Samples: {status['sample_count']}")
    print(f"  MAE: {status['performance']['mae']:.4f}")
    print(f"  Gated features: {status['feature_stats']['gated_features']}")
    print(
        f"  Top features: {[(k, f'{v:+.4f}') for k, v in status['top_features'][:3]]}"
    )

    # Test scheduled update
    print("\n⏰ Running scheduled update...")
    learner.run_scheduled_update()

    print("✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(test_learning())
