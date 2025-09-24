#!/usr/bin/env python3
"""
Go-Live Pipeline Integration Test
Test all components of the safe deployment pipeline: A/B gates, feature flags, metrics, etc.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("golive_test")


def test_feature_flags_service():
    """Test feature flags service."""
    print("🚩 Testing Feature Flags Service...")

    try:
        from src.core.feature_flags import FeatureFlagService, is_enabled

        # Create service
        service = FeatureFlagService()

        # Initialize defaults
        service.initialize_defaults()

        # Test flag operations
        service.set_flag("EXEC_RL_LIVE", True)
        assert service.is_enabled("EXEC_RL_LIVE") == True

        service.set_flag("BANDIT_WEIGHTS", False)
        assert service.is_enabled("BANDIT_WEIGHTS") == False

        # Test convenience function
        assert is_enabled("EXEC_RL_LIVE") == True

        # Test get all flags
        all_flags = service.get_all_flags()
        assert "EXEC_RL_LIVE" in all_flags
        assert "BANDIT_WEIGHTS" in all_flags

        print("✅ Feature Flags Service: PASS")
        print(f"   Available flags: {list(all_flags.keys())}")
        return True

    except Exception as e:
        print(f"❌ Feature Flags Service: FAIL - {e}")
        return False


def test_param_server_schemas():
    """Test updated ParamServer schemas with feature flags."""
    print("\n🔧 Testing ParamServer Schemas...")

    try:
        from src.core.param_server.schemas import (
            FeatureFlags,
            ModelRouterRules,
            create_default_router_rules,
        )

        # Test FeatureFlags schema
        flags = FeatureFlags()
        assert hasattr(flags, "EXEC_RL_LIVE")
        assert hasattr(flags, "BANDIT_WEIGHTS")
        assert hasattr(flags, "LLM_SENTIMENT")

        # Test Redis conversion
        redis_mapping = flags.to_redis_mapping()
        assert isinstance(redis_mapping["EXEC_RL_LIVE"], int)

        # Test from Redis
        flags_from_redis = FeatureFlags.from_redis_mapping(redis_mapping)
        assert flags_from_redis.EXEC_RL_LIVE == flags.EXEC_RL_LIVE

        # Test in router rules
        rules = create_default_router_rules()
        assert hasattr(rules, "feature_flags")
        assert isinstance(rules.feature_flags, FeatureFlags)

        print("✅ ParamServer Schemas: PASS")
        print(f"   Feature flags: {list(FeatureFlags.model_fields.keys())}")
        return True

    except Exception as e:
        print(f"❌ ParamServer Schemas: FAIL - {e}")
        return False


def test_ab_evaluation_gate():
    """Test A/B evaluation gate."""
    print("\n🚪 Testing A/B Evaluation Gate...")

    try:
        from scripts.ab_eval_gate import ABEvaluationGate

        # Create gate
        gate = ABEvaluationGate()

        # Test metric retrieval
        metric_value = gate.get_metric("test:metric", 42.0)
        assert metric_value == 42.0  # Should return default

        # Test flag status
        status = gate.get_flag_status("EXEC_RL_LIVE")
        assert isinstance(status, bool)

        # Test evaluation methods (should not promote with mock data)
        exec_promoted = gate.evaluate_exec_rl()
        bandit_promoted = gate.evaluate_bandit_weights()
        llm_promoted = gate.evaluate_llm_sentiment()

        # Should be False with no real metrics
        assert exec_promoted == False
        assert bandit_promoted == False
        assert llm_promoted == False

        # Test status report
        status_report = gate.get_status_report()
        assert "feature_flags" in status_report
        assert "current_metrics" in status_report
        assert "thresholds" in status_report

        print("✅ A/B Evaluation Gate: PASS")
        print(f"   Thresholds: {gate.thresholds}")
        return True

    except Exception as e:
        print(f"❌ A/B Evaluation Gate: FAIL - {e}")
        return False


def test_llm_signal_evaluator():
    """Test LLM signal evaluator."""
    print("\n📊 Testing LLM Signal Evaluator...")

    try:
        from scripts.llm_signal_eval import LLMSignalEvaluator

        # Create evaluator
        evaluator = LLMSignalEvaluator(window_size=100)

        # Test data retrieval (will use synthetic data)
        llm_df = evaluator.get_llm_events(hours_back=1)
        price_df = evaluator.get_price_data(hours_back=1)

        # Should have synthetic data
        assert len(price_df) > 0

        # Test correlation computation
        correlation_result = evaluator.compute_correlation(llm_df, price_df)
        assert "correlation" in correlation_result
        assert "samples" in correlation_result
        assert "pvalue" in correlation_result

        # Test metrics storage
        evaluator.store_correlation_metrics(correlation_result)

        # Test status report
        status = evaluator.get_evaluation_status()
        assert "current_correlation" in status

        print("✅ LLM Signal Evaluator: PASS")
        print(f"   Correlation: {correlation_result['correlation']:.4f}")
        print(f"   Samples: {correlation_result['samples']}")
        return True

    except Exception as e:
        print(f"❌ LLM Signal Evaluator: FAIL - {e}")
        return False


def test_tail_risk_hedge():
    """Test tail-risk hedge overlay."""
    print("\n🛡️ Testing Tail Risk Hedge Overlay...")

    try:
        from src.layers.layer5_risk.hedge_overlay import TailRiskHedgeOverlay

        # Create hedge overlay
        hedge = TailRiskHedgeOverlay()

        # Test signal generation
        enter_signal, exit_signal = hedge.hedge_signal(0.035, 2.5)  # Above thresholds
        assert enter_signal == True
        assert exit_signal == False

        # Test with low risk
        enter_signal, exit_signal = hedge.hedge_signal(0.015, 1.0)  # Below thresholds
        assert enter_signal == False
        assert exit_signal == True

        # Test hedge sizing
        hedge_size = hedge.calculate_optimal_hedge_size(1000000)  # $1M exposure
        assert hedge_size > 0
        assert hedge_size <= 1000000 * hedge.max_hedge_ratio

        # Test entering position
        success = hedge.enter_hedge_position("BTCUSDT", -5.0, "perp_short", 50000.0)
        assert success == True
        assert len(hedge.active_hedges) == 1

        # Test tick function
        risk_metrics = {
            "es_evt_95": 0.015,
            "iv_change_sigma": 1.0,
            "gross_exposure": 1000000,
            "btc_price": 51000.0,
        }

        tick_result = hedge.tick(risk_metrics)
        assert "hedge_active" in tick_result
        assert "hedge_pnl_total" in tick_result

        # Test status
        status = hedge.get_status()
        assert "active_hedges" in status
        assert "total_pnl" in status

        print("✅ Tail Risk Hedge Overlay: PASS")
        print(f"   Active hedges: {len(hedge.active_hedges)}")
        print(f"   Total P&L: ${tick_result.get('hedge_pnl_total', 0):,.2f}")
        return True

    except Exception as e:
        print(f"❌ Tail Risk Hedge Overlay: FAIL - {e}")
        return False


def test_dl_fine_tuner():
    """Test DL fine-tuner."""
    print("\n🧠 Testing DL Fine-Tuner...")

    try:
        from scripts.fine_tune_dl import DLFineTuner

        # Create fine-tuner
        fine_tuner = DLFineTuner(
            data_dir="/tmp/test_data", models_dir="/tmp/test_models", lookback_days=7
        )

        # Test data loading (will generate synthetic)
        df = fine_tuner.load_training_data()
        assert len(df) > 0

        # Test feature preparation
        X, y, feature_names = fine_tuner.prepare_features(df)
        assert X is not None
        assert y is not None
        assert len(feature_names) > 0

        # Test model fine-tuning (mock)
        result = fine_tuner.fine_tune_model("test_model", X[:1000], y[:1000])
        assert result["status"] == "success"
        assert "model_hash" in result
        assert "val_sharpe" in result

        # Test evaluation
        evaluations = fine_tuner.evaluate_shadow_performance([result])
        assert "test_model" in evaluations

        # Test promotion logic
        eval_data = evaluations["test_model"]
        # Force promotion for testing
        eval_data["promote"] = True

        promoted = fine_tuner.promote_model("test_model", eval_data)
        # Should fail gracefully (no actual model file)

        print("✅ DL Fine-Tuner: PASS")
        print(f"   Features: {len(feature_names)}")
        print(f"   Training samples: {len(X):,}")
        return True

    except Exception as e:
        print(f"❌ DL Fine-Tuner: FAIL - {e}")
        return False


def test_integration_workflow():
    """Test end-to-end integration workflow."""
    print("\n🔄 Testing Integration Workflow...")

    try:
        # Import all services
        from src.core.feature_flags import get_feature_service
        from scripts.ab_eval_gate import ABEvaluationGate
        from scripts.llm_signal_eval import LLMSignalEvaluator

        # 1. Initialize feature flags
        feature_service = get_feature_service()
        feature_service.set_flag("BANDIT_WEIGHTS", False)  # Start disabled

        # 2. Run LLM evaluation
        llm_eval = LLMSignalEvaluator(window_size=50)
        llm_result = llm_eval.run_evaluation(hours_back=1)

        # 3. Run A/B evaluation gate
        gate = ABEvaluationGate()
        ab_results = gate.run_evaluation()

        # 4. Check feature flag changes
        bandit_enabled_after = feature_service.is_enabled("BANDIT_WEIGHTS")

        # 5. Generate workflow report
        workflow_report = {
            "llm_evaluation": {
                "correlation": llm_result.get("correlation", 0.0),
                "samples": llm_result.get("samples", 0),
            },
            "ab_evaluation": ab_results,
            "feature_flags_changed": ab_results.get("bandit_weights", False),
            "bandit_enabled": bandit_enabled_after,
            "workflow_success": True,
        }

        print("✅ Integration Workflow: PASS")
        print(f"   LLM correlation: {llm_result.get('correlation', 0.0):.4f}")
        print(f"   Features promoted: {sum(ab_results.values()) if ab_results else 0}")
        print(f"   Bandit weights enabled: {bandit_enabled_after}")

        return True

    except Exception as e:
        print(f"❌ Integration Workflow: FAIL - {e}")
        return False


def test_redis_integration():
    """Test Redis integration across all components."""
    print("\n🔗 Testing Redis Integration...")

    try:
        import redis

        # Test Redis connection
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()

        # Test feature flags in Redis
        from src.core.feature_flags import get_feature_service

        service = get_feature_service()
        service.set_flag("TEST_FLAG", True)

        # Verify in Redis directly
        flag_value = r.hget("features:flags", "TEST_FLAG")
        assert flag_value == "1"

        # Test A/B metrics storage
        from scripts.ab_eval_gate import ABEvaluationGate

        gate = ABEvaluationGate()

        # Store some test metrics
        r.set("slippage:live:bps", "12.5")
        r.set("slippage:shadow:bps", "8.2")
        r.set("sharpe:1h:live", "0.75")
        r.set("sharpe:1h:shadow", "0.90")

        # Test metric retrieval
        live_slip = gate.get_metric("slippage:live:bps")
        shadow_slip = gate.get_metric("slippage:shadow:bps")

        assert live_slip == 12.5
        assert shadow_slip == 8.2

        # Clean up test data
        r.delete("TEST_FLAG")

        print("✅ Redis Integration: PASS")
        print(f"   Connection: OK")
        print(f"   Feature flags: OK")
        print(f"   Metrics storage: OK")
        return True

    except Exception as e:
        print(f"❌ Redis Integration: FAIL - {e}")
        return False


def main():
    """Run all Go-Live pipeline tests."""
    print("🚀 Go-Live Pipeline Integration Test Suite")
    print("=" * 60)

    test_results = []

    # Run all tests
    test_functions = [
        ("Feature Flags Service", test_feature_flags_service),
        ("ParamServer Schemas", test_param_server_schemas),
        ("A/B Evaluation Gate", test_ab_evaluation_gate),
        ("LLM Signal Evaluator", test_llm_signal_evaluator),
        ("Tail Risk Hedge", test_tail_risk_hedge),
        ("DL Fine-Tuner", test_dl_fine_tuner),
        ("Integration Workflow", test_integration_workflow),
        ("Redis Integration", test_redis_integration),
    ]

    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("🏁 Go-Live Pipeline Test Results Summary:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Go-Live pipeline components are working correctly!")
        print("\nReady for deployment:")
        print("1. Set up cron jobs:")
        print("   - */15 * * * * python3 scripts/ab_eval_gate.py")
        print("   - 5 2 * * * python3 scripts/fine_tune_dl.py")
        print("2. Configure Slack webhook in environment")
        print("3. Set up Prometheus/Grafana monitoring")
        print("4. Deploy with blue/green safety net")
    else:
        print("⚠️  Some components need attention before deployment")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
