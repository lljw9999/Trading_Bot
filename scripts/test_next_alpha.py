#!/usr/bin/env python3
"""
Next-Alpha Sprint Integration Test
Test all three components: GPT sentiment, RL execution, contextual bandits
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_gpt_sentiment():
    """Test GPT sentiment microservice."""
    print("🧪 Testing GPT Sentiment Microservice...")

    try:
        from services.llm_sentiment_service import LLMSentimentService

        service = LLMSentimentService()
        stats = service.get_service_stats()

        assert stats["status"] == "active"
        assert stats["model"] == "gpt-4o-mini"

        print("✅ GPT Sentiment Service: PASS")
        print(f"   Model: {stats['model']}")
        print(f"   Status: {stats['status']}")
        return True

    except Exception as e:
        print(f"❌ GPT Sentiment Service: FAIL - {e}")
        return False


def test_orderbook_env():
    """Test RL orderbook environment."""
    print("\n🧪 Testing RL OrderBook Environment...")

    try:
        from envs.orderbook_env import OrderBookEnv, LiveExecEnv

        # Test training environment
        env = OrderBookEnv(mode="test")
        obs, _ = env.reset()

        assert obs.shape == (16,), f"Expected obs shape (16,), got {obs.shape}"
        assert env.action_space.shape == (
            3,
        ), f"Expected action shape (3,), got {env.action_space.shape}"

        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            if done:
                break

        # Test live environment
        live_env = LiveExecEnv()
        state = live_env.get_state()
        assert state.shape == (
            16,
        ), f"Expected live state shape (16,), got {state.shape}"

        print("✅ RL OrderBook Environment: PASS")
        print(f"   Training env obs shape: {obs.shape}")
        print(f"   Live env state shape: {state.shape}")
        return True

    except Exception as e:
        print(f"❌ RL OrderBook Environment: FAIL - {e}")
        return False


def test_bandit_blender():
    """Test contextual bandit blender."""
    print("\n🧪 Testing Contextual Bandit Blender...")

    try:
        from src.layers.layer2_ensemble.bandit_blender import BanditBlender

        blender = BanditBlender()

        # Test feature extraction
        mock_state = {
            "vol_20": 0.15,
            "sent_bull": 1,
            "sent_bear": 0,
            "iv_slope": 0.02,
            "rsi": 65.0,
            "volume_ratio": 1.2,
            "spread_pct": 5.0,
            "market_cap_flow": 1000000,
            "funding_rate": 0.001,
            "oi_change": 0.05,
        }

        context = blender.extract_context_features(mock_state)
        assert len(context) == 10, f"Expected 10 features, got {len(context)}"

        # Test weight selection
        weights = blender.choose_weights(context)
        assert len(weights) == len(
            blender.arms
        ), f"Expected {len(blender.arms)} weights"
        assert all(
            isinstance(w, float) for w in weights.values()
        ), "All weights should be floats"

        # Test bandit update
        reward_vector = np.random.randn(len(blender.arms)) * 0.01
        blender.update_bandit(context, reward_vector)

        # Test performance stats
        stats = blender.get_performance_stats()
        assert "total_rounds" in stats
        assert stats["total_rounds"] >= 1

        print("✅ Contextual Bandit Blender: PASS")
        print(f"   Arms: {blender.arms}")
        print(f"   Total rounds: {stats['total_rounds']}")
        print(f"   Avg reward: {stats['average_reward']:.4f}")
        return True

    except Exception as e:
        print(f"❌ Contextual Bandit Blender: FAIL - {e}")
        return False


def test_meta_learner_integration():
    """Test meta-learner integration with bandit."""
    print("\n🧪 Testing Meta-Learner Bandit Integration...")

    try:
        from src.layers.layer2_ensemble.meta_learner import MetaLearner
        from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot

        # Create meta-learner
        meta = MetaLearner()

        # Check if bandit is initialized
        assert hasattr(
            meta, "bandit_blender"
        ), "Meta-learner should have bandit_blender"
        if meta.use_bandit_weights:
            assert (
                meta.bandit_blender is not None
            ), "Bandit blender should be initialized"
            print("✅ Bandit integration enabled in meta-learner")
        else:
            print("⚠️  Bandit integration disabled, using fallback")

        # Test prediction with mock signals
        alpha_signals = {
            "ob_pressure": (25.0, 0.8),  # 25bp edge, 80% confidence
            "ma_momentum": (-15.0, 0.6),  # -15bp edge, 60% confidence
            "lstm_alpha": (10.0, 0.7),  # 10bp edge, 70% confidence
            "news_alpha": (5.0, 0.4),  # 5bp edge, 40% confidence
            "onchain_alpha": (-5.0, 0.3),  # -5bp edge, 30% confidence
        }

        # Mock market features
        market_features = FeatureSnapshot(
            timestamp=time.time(),
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_5m=0.02,
            volume_ratio=1.1,
            order_book_imbalance=0.1,
            return_1m=0.0005,
        )

        # Make prediction
        edge, confidence = meta.predict(alpha_signals, market_features)

        assert isinstance(edge, float), "Edge should be float"
        assert isinstance(confidence, float), "Confidence should be float"
        assert (
            0.0 <= confidence <= 1.0
        ), f"Confidence should be in [0,1], got {confidence}"

        print("✅ Meta-Learner Bandit Integration: PASS")
        print(f"   Ensemble edge: {edge:.2f}bp")
        print(f"   Ensemble confidence: {confidence:.3f}")
        print(f"   Using bandit weights: {meta.use_bandit_weights}")
        return True

    except Exception as e:
        print(f"❌ Meta-Learner Bandit Integration: FAIL - {e}")
        return False


def test_shadow_daemon():
    """Test shadow execution daemon (without model)."""
    print("\n🧪 Testing Shadow Execution Daemon...")

    try:
        from scripts.exec_shadow_daemon import ExecShadowDaemon

        # Test daemon initialization
        daemon = ExecShadowDaemon(model_path="/models/nonexistent", update_interval=0.1)

        # Test state retrieval
        state = daemon.get_current_state()
        assert state.shape == (16,), f"Expected state shape (16,), got {state.shape}"

        # Test prediction (should return neutral action without model)
        action = daemon.predict_action(state)
        assert action.shape == (3,), f"Expected action shape (3,), got {action.shape}"
        assert np.allclose(
            action, [0.0, 0.0, 0.0]
        ), "Should return neutral action without model"

        # Test stats
        stats = daemon.get_daemon_stats()
        assert "service" in stats
        assert stats["model_loaded"] is False  # No model file exists

        print("✅ Shadow Execution Daemon: PASS")
        print(f"   State shape: {state.shape}")
        print(f"   Action shape: {action.shape}")
        print(f"   Model loaded: {stats['model_loaded']}")
        return True

    except Exception as e:
        print(f"❌ Shadow Execution Daemon: FAIL - {e}")
        return False


def test_redis_integration():
    """Test Redis integration for all components."""
    print("\n🧪 Testing Redis Integration...")

    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Test basic connectivity
        r.ping()

        # Test bandit state storage
        from src.layers.layer2_ensemble.bandit_blender import BanditBlender

        blender = BanditBlender()

        # Generate a bandit decision and check Redis
        context = np.random.randn(10)
        weights = blender.choose_weights(context)

        # Check if weights are stored in Redis
        redis_weights = r.hgetall("ensemble:weights")
        assert len(redis_weights) > 0, "Weights should be stored in Redis"

        # Check bandit metadata
        meta = r.hgetall("ensemble:bandit_meta")
        assert "chosen_arm" in meta, "Bandit metadata should be stored"

        print("✅ Redis Integration: PASS")
        print(f"   Weights stored: {len(redis_weights)} arms")
        print(f"   Chosen arm: {meta.get('chosen_model', 'unknown')}")
        return True

    except Exception as e:
        print(f"❌ Redis Integration: FAIL - {e}")
        return False


def main():
    """Run all Next-Alpha component tests."""
    print("🚀 Next-Alpha Sprint Integration Test Suite")
    print("=" * 50)

    test_results = []

    # Run all tests
    test_results.append(("GPT Sentiment", test_gpt_sentiment()))
    test_results.append(("OrderBook Environment", test_orderbook_env()))
    test_results.append(("Contextual Bandits", test_bandit_blender()))
    test_results.append(("Meta-Learner Integration", test_meta_learner_integration()))
    test_results.append(("Shadow Daemon", test_shadow_daemon()))
    test_results.append(("Redis Integration", test_redis_integration()))

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Next-Alpha components are working correctly!")
        print("\nNext steps:")
        print("1. Train RL execution model: python rl/exec_agent.py --quick-test")
        print(
            "2. Start LLM sentiment service: python services/llm_sentiment_service.py"
        )
        print("3. Start shadow daemon: python scripts/exec_shadow_daemon.py")
        print("4. Monitor performance in Grafana")
    else:
        print("⚠️  Some components need attention before deployment")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
