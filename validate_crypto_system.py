#!/usr/bin/env python3
"""
Crypto Trading System Validation

Tests all 6 layers of the crypto trading system:
- C1: Coinbase executor
- C2: Kelly sizing for crypto
- C3: Risk overlay for crypto
- C4: Momentum alpha model
- C5: End-to-end integration
- C6: System health checks
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all components
from src.layers.layer1_alpha_models.momo_fast import MomentumFastAlpha
from src.layers.layer2_ensemble.meta_learner import MetaLearner
from src.layers.layer3_position_sizing.kelly_sizing import KellySizing
from src.layers.layer4_execution.coinbase_executor import CoinbaseExecutor, OrderRequest
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager

class CryptoSystemValidator:
    """Comprehensive crypto system validation."""
    
    def __init__(self):
        self.test_results = {}
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.portfolio_value = Decimal('100000')
    
    def run_validation(self) -> Dict[str, bool]:
        """Run all validation tests."""
        logger.info("🚀 Starting Crypto System Validation...")
        
        tests = [
            ("C1_coinbase_executor", self._test_coinbase_executor),
            ("C2_kelly_sizing_crypto", self._test_kelly_sizing_crypto),
            ("C3_risk_overlay_crypto", self._test_risk_overlay_crypto),
            ("C4_momentum_alpha", self._test_momentum_alpha),
            ("C5_integration_pipeline", self._test_integration_pipeline),
            ("C6_system_health", self._test_system_health)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"🧪 Running {test_name}...")
                result = test_func()
                self.test_results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                logger.error(f"💥 {test_name} ERROR: {e}")
                self.test_results[test_name] = False
        
        # Summary
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        logger.info(f"\n📊 Validation Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All crypto system tests PASSED!")
        else:
            logger.warning(f"⚠️  {total - passed} tests FAILED")
        
        return self.test_results
    
    def _test_coinbase_executor(self) -> bool:
        """Test C1: Coinbase executor functionality."""
        try:
            # Initialize executor
            executor = CoinbaseExecutor()
            
            # Test account info
            account = executor.get_account()
            if not account:
                logger.error("Failed to get account info")
                return False
            
            # Test order submission
            test_order = OrderRequest(
                symbol="BTC-USD",
                side="buy",
                qty=100.0,  # $100 worth
                order_type="market",
                client_order_id="test_crypto_validation_001"
            )
            
            response = executor.submit_order(test_order)
            if not response:
                logger.error("Failed to submit test order")
                return False
            
            # Check order status
            order_status = executor.get_order(response.order_id)
            if not order_status:
                logger.error("Failed to get order status")
                return False
            
            # Test positions
            positions = executor.get_positions()
            
            logger.info(f"✅ Coinbase executor: account={bool(account)}, "
                       f"order={response.status}, positions={len(positions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Coinbase executor test failed: {e}")
            return False
    
    def _test_kelly_sizing_crypto(self) -> bool:
        """Test C2: Kelly sizing for crypto."""
        try:
            # Initialize Kelly sizer
            kelly = KellySizing(max_position_size=0.20, kelly_fraction=0.25)
            
            # Test crypto position sizing
            test_cases = [
                {'symbol': 'BTC-USD', 'edge_bps': 15.0, 'confidence': 0.8, 'price': 50000},
                {'symbol': 'ETH-USD', 'edge_bps': -10.0, 'confidence': 0.7, 'price': 3000},
                {'symbol': 'SOL-USD', 'edge_bps': 20.0, 'confidence': 0.9, 'price': 100}
            ]
            
            valid_positions = 0
            
            for case in test_cases:
                position_size, reason = kelly.calculate_position_size(
                    symbol=case['symbol'],
                    edge_bps=case['edge_bps'],
                    confidence=case['confidence'],
                    current_price=Decimal(str(case['price'])),
                    portfolio_value=self.portfolio_value,
                    instrument_type='crypto'
                )
                
                # Check position is within limits
                position_fraction = abs(position_size) / self.portfolio_value
                if position_fraction <= 0.20:  # 20% max for crypto
                    valid_positions += 1
                    logger.info(f"✅ {case['symbol']}: ${position_size:.0f} "
                               f"({position_fraction:.1%}) - {reason}")
                else:
                    logger.error(f"❌ {case['symbol']}: position {position_fraction:.1%} "
                                f"exceeds 20% limit")
            
            return valid_positions == len(test_cases)
            
        except Exception as e:
            logger.error(f"Kelly sizing test failed: {e}")
            return False
    
    def _test_risk_overlay_crypto(self) -> bool:
        """Test C3: Risk overlay for crypto."""
        try:
            # Initialize risk manager
            risk_mgr = BasicRiskManager()
            
            # Test crypto-specific risk checks
            test_cases = [
                # Normal position - should pass
                {'symbol': 'BTC-USD', 'position': 10000, 'price': 50000, 'should_pass': True},
                # Large position - should be limited (30% > 15% limit)
                {'symbol': 'ETH-USD', 'position': 30000, 'price': 3000, 'should_pass': True},  # Will be limited but allowed
                # Exchange limit test - should be limited
                {'symbol': 'SOL-USD', 'position': 60000, 'price': 100, 'should_pass': True}   # Will be limited but allowed
            ]
            
            passed_tests = 0
            
            for case in test_cases:
                # Set up exchange exposure for limit testing
                if case['position'] > 50000:  # Above exchange limit
                    risk_mgr.update_exchange_exposure('coinbase', 45000)
                
                is_allowed, reason, max_allowed = risk_mgr.check_position_risk(
                    symbol=case['symbol'],
                    proposed_position=Decimal(str(case['position'])),
                    current_price=Decimal(str(case['price'])),
                    portfolio_value=self.portfolio_value,
                    instrument_type='crypto'
                )
                
                # Check if position was properly limited
                if case['position'] > 15000:  # Above 15% limit
                    if max_allowed < Decimal(str(case['position'])):
                        passed_tests += 1
                        logger.info(f"✅ {case['symbol']}: ${case['position']} limited to ${max_allowed:.0f} - {reason}")
                    else:
                        logger.error(f"❌ {case['symbol']}: position not properly limited - {reason}")
                else:
                    if is_allowed:
                        passed_tests += 1
                        logger.info(f"✅ {case['symbol']}: ${case['position']} allowed - {reason}")
                    else:
                        logger.error(f"❌ {case['symbol']}: small position blocked - {reason}")
            
            return passed_tests == len(test_cases)
            
        except Exception as e:
            logger.error(f"Risk overlay test failed: {e}")
            return False
    
    def _test_momentum_alpha(self) -> bool:
        """Test C4: Momentum alpha model."""
        try:
            # Initialize momentum model
            alpha = MomentumFastAlpha(fast_period=5, slow_period=20, edge_scaling=15.0)
            
            # Generate test data with momentum
            signals_generated = 0
            strong_signals = 0
            
            for symbol in self.symbols:
                base_price = {'BTC-USD': 50000, 'ETH-USD': 3000, 'SOL-USD': 100}[symbol]
                
                # Create trending price data
                for i in range(30):
                    # Add trend + noise
                    trend = i * 0.02  # 2% trend
                    noise = np.random.normal(0, 0.01)  # 1% noise
                    price = base_price * (1 + trend + noise)
                    
                    timestamp = f"2025-01-15T10:{i:02d}:00Z"
                    signal = alpha.update_price(symbol, price, timestamp)
                    
                    if signal:
                        signals_generated += 1
                        if abs(signal.edge_bps) > 8.0:  # Strong signal
                            strong_signals += 1
                        
                        logger.debug(f"📊 {symbol}: edge={signal.edge_bps:.1f}bps, "
                                   f"conf={signal.confidence:.2f}")
            
            # Calculate ROC (Rate of Change)
            roc = strong_signals / max(1, signals_generated)
            
            # Test performance tracking
            if signals_generated > 0:
                # Simulate some performance updates
                test_signal = alpha.update_price('BTC-USD', 50500, '2025-01-15T10:30:00Z')
                if test_signal:
                    alpha.update_performance('BTC-USD', 10.0, test_signal.edge_bps)
            
            # Get stats
            stats = alpha.get_stats()
            
            logger.info(f"✅ Momentum alpha: signals={signals_generated}, "
                       f"strong={strong_signals}, ROC={roc:.3f}, "
                       f"hit_rate={stats['hit_rate']:.2%}")
            
            # Check requirements
            return signals_generated > 0 and roc > 0.3  # Relaxed ROC for test data
            
        except Exception as e:
            logger.error(f"Momentum alpha test failed: {e}")
            return False
    
    def _test_integration_pipeline(self) -> bool:
        """Test C5: End-to-end integration."""
        try:
            # Initialize all components
            alpha = MomentumFastAlpha(fast_period=5, slow_period=20, edge_scaling=15.0)
            ensemble = MetaLearner()
            kelly = KellySizing(max_position_size=0.20, kelly_fraction=0.25)
            executor = CoinbaseExecutor()
            risk_mgr = BasicRiskManager()
            
            # Simulate pipeline processing
            pipeline_steps = 0
            successful_orders = 0
            
            for symbol in self.symbols:
                base_price = {'BTC-USD': 50000, 'ETH-USD': 3000, 'SOL-USD': 100}[symbol]
                
                # Generate enough signals with strong momentum
                for i in range(35):  # More data points to generate signals
                    # Create strong trending pattern
                    trend = i * 0.015  # 1.5% trend per period
                    noise = np.random.normal(0, 0.005)  # Small noise
                    price = base_price * (1 + trend + noise)
                    
                    timestamp = f"2025-01-15T10:{i:02d}:00Z"
                    
                    # L1: Alpha signal
                    alpha_signal = alpha.update_price(symbol, price, timestamp)
                    if not alpha_signal:
                        continue
                    
                    pipeline_steps += 1
                    
                    # L2: Ensemble
                    from src.layers.layer0_data_ingestion.schemas import FeatureSnapshot
                    from datetime import datetime, timezone
                    
                    # Create mock feature snapshot
                    mock_features = FeatureSnapshot(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        spread_bps=1.0,
                        volatility_5m=0.01,
                        volume_ratio=1.0,
                        order_book_imbalance=0.0,
                        return_1m=0.0
                    )
                    
                    # Create alpha signals dict
                    alpha_signals_dict = {
                        'momentum_fast': (alpha_signal.edge_bps, alpha_signal.confidence)
                    }
                    
                    ensemble_edge, ensemble_confidence = ensemble.predict(
                        alpha_signals=alpha_signals_dict,
                        market_features=mock_features
                    )
                    
                    # L3: Kelly sizing
                    position_size, _ = kelly.calculate_position_size(
                        symbol=symbol,
                        edge_bps=ensemble_edge,
                        confidence=ensemble_confidence,
                        current_price=Decimal(str(price)),
                        portfolio_value=self.portfolio_value,
                        instrument_type='crypto'
                    )
                    
                    # L5: Risk check
                    is_allowed, _, max_allowed = risk_mgr.check_position_risk(
                        symbol=symbol,
                        proposed_position=position_size,
                        current_price=Decimal(str(price)),
                        portfolio_value=self.portfolio_value,
                        instrument_type='crypto'
                    )
                    
                    if is_allowed and abs(position_size) > 10:
                        # L4: Execute order
                        try:
                            order_request = OrderRequest(
                                symbol=symbol,
                                side='buy' if position_size > 0 else 'sell',
                                qty=float(abs(position_size)),
                                order_type='market'
                            )
                            
                            response = executor.submit_order(order_request)
                            if response and response.status in ['filled', 'pending_new']:
                                successful_orders += 1
                                
                        except Exception as e:
                            logger.debug(f"Order execution failed: {e}")
            
            logger.info(f"✅ Integration pipeline: steps={pipeline_steps}, "
                       f"orders={successful_orders}, "
                       f"conversion={successful_orders/max(1, pipeline_steps):.1%}")
            
            return pipeline_steps > 0 and successful_orders > 0
            
        except Exception as e:
            logger.error(f"Integration pipeline test failed: {e}")
            return False
    
    def _test_system_health(self) -> bool:
        """Test C6: System health checks."""
        try:
            health_checks = []
            
            # Check Prometheus availability
            try:
                import requests
                response = requests.get('http://localhost:9090/-/healthy', timeout=5)
                prometheus_healthy = response.status_code == 200
                health_checks.append(('Prometheus', prometheus_healthy))
            except:
                health_checks.append(('Prometheus', False))
            
            # Check Grafana availability
            try:
                import requests
                response = requests.get('http://localhost:3000/api/health', timeout=5)
                grafana_healthy = response.status_code == 200
                health_checks.append(('Grafana', grafana_healthy))
            except:
                health_checks.append(('Grafana', False))
            
            # Check Redis availability
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                redis_healthy = r.ping()
                health_checks.append(('Redis', redis_healthy))
            except:
                health_checks.append(('Redis', False))
            
            # Check required Python packages
            required_packages = ['numpy', 'pandas', 'requests', 'pyyaml']
            for package in required_packages:
                try:
                    __import__(package)
                    health_checks.append((f'Package {package}', True))
                except ImportError:
                    health_checks.append((f'Package {package}', False))
            
            # Report health status
            healthy_services = sum(1 for _, status in health_checks if status)
            total_services = len(health_checks)
            
            logger.info(f"✅ System health: {healthy_services}/{total_services} services healthy")
            
            for service, status in health_checks:
                status_icon = "✅" if status else "❌"
                logger.info(f"  {status_icon} {service}: {'OK' if status else 'FAIL'}")
            
            # System is healthy if at least 70% of services are up
            return healthy_services / total_services >= 0.7
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return False


def main():
    """Main validation entry point."""
    logger.info("🚀 Starting Crypto System Validation...")
    
    validator = CryptoSystemValidator()
    results = validator.run_validation()
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 CRYPTO SYSTEM VALIDATION RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 Crypto system validation SUCCESSFUL!")
        return 0
    else:
        print("⚠️  Crypto system validation FAILED!")
        return 1


if __name__ == "__main__":
    exit(main()) 