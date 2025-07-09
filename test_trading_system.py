#!/usr/bin/env python3
"""
Comprehensive Trading System Test

Tests all layers of the trading system with synthetic data to validate
the implementation matches the roadmap specifications.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.layers.layer0_data_ingestion.schemas import MarketTick, FeatureSnapshot, AlphaSignal, TradingDecision
from src.layers.layer0_data_ingestion.feature_bus import FeatureBus
from src.layers.layer1_alpha_models.order_book_pressure import OrderBookPressure
from src.layers.layer1_alpha_models.ma_momentum import MAMomentum
from src.layers.layer2_ensemble.meta_learner import MetaLearner
from src.layers.layer3_position_sizing.kelly_sizing import KellySizing
from src.layers.layer4_execution.market_order_executor import MarketOrderExecutor
from src.layers.layer5_risk.basic_risk_manager import BasicRiskManager, RiskLimits
from src.utils.logger import get_logger


class TradingSystemTest:
    """Comprehensive test suite for the trading system."""
    
    def __init__(self):
        """Initialize test environment."""
        self.logger = get_logger("test_trading_system")
        
        # Initialize all system components
        self.feature_bus = FeatureBus()
        self.alpha_models = {
            'order_book_pressure': OrderBookPressure(),
            'ma_momentum': MAMomentum()
        }
        self.meta_learner = MetaLearner()
        self.kelly_sizing = KellySizing()
        self.executor = MarketOrderExecutor(paper_trading=True)
        self.risk_manager = BasicRiskManager()
        
        # Test configuration
        self.test_symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'MSFT']
        self.test_duration_minutes = 60
        
        self.logger.info("Trading system test initialized")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of all system layers."""
        self.logger.info("Starting comprehensive trading system test...")
        
        try:
            # Test 1: Layer 0 - Data Ingestion and Feature Computation
            await self.test_layer0_feature_computation()
            
            # Test 2: Layer 1 - Alpha Models
            await self.test_layer1_alpha_models()
            
            # Test 3: Layer 2 - Ensemble Meta-Learner
            await self.test_layer2_ensemble()
            
            # Test 4: Layer 3 - Position Sizing
            await self.test_layer3_position_sizing()
            
            # Test 5: Layer 4 - Execution
            await self.test_layer4_execution()
            
            # Test 6: Layer 5 - Risk Management
            await self.test_layer5_risk_management()
            
            # Test 7: End-to-End Trading Pipeline
            await self.test_end_to_end_pipeline()
            
            # Test 8: Performance and Metrics
            await self.test_performance_metrics()
            
            self.logger.info("All tests completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            raise
    
    async def test_layer0_feature_computation(self):
        """Test Layer 0: Data ingestion and feature computation."""
        self.logger.info("Testing Layer 0: Feature computation...")
        
        # Generate synthetic market data
        base_price = Decimal('50000')
        features_computed = 0
        
        for i in range(100):
            for symbol in self.test_symbols[:2]:  # Test with crypto symbols
                # Create realistic tick data
                price_change = np.random.normal(0, 0.001) * float(base_price)
                current_price = base_price + Decimal(str(price_change))
                
                tick = MarketTick(
                    symbol=symbol,
                    exchange="test_exchange",
                    asset_type="crypto",
                    timestamp=datetime.utcnow(),
                    bid=current_price - Decimal('5'),
                    ask=current_price + Decimal('5'),
                    mid=current_price,
                    volume=Decimal(str(abs(np.random.normal(100, 20)))),
                    bid_size=Decimal(str(abs(np.random.normal(1.0, 0.2)))),
                    ask_size=Decimal(str(abs(np.random.normal(1.0, 0.2))))
                )
                
                # Process tick through feature bus
                features = await self.feature_bus.process_tick(tick)
                
                if features:
                    features_computed += 1
                    
                    # Validate feature computation
                    assert features.symbol == symbol
                    assert features.mid_price == current_price
                    assert features.spread_bps is not None
                    
                    if i > 20:  # After some history is built
                        assert features.return_1m is not None or features.volatility_5m is not None
        
        # Check performance
        stats = self.feature_bus.get_stats()
        assert stats['features_computed'] == features_computed
        assert stats['performance_target_met']  # Should be under 300µs
        
        self.logger.info(f"✓ Layer 0: Computed {features_computed} features, avg time: {stats['avg_computation_time_us']:.1f}µs")
    
    async def test_layer1_alpha_models(self):
        """Test Layer 1: Alpha models."""
        self.logger.info("Testing Layer 1: Alpha models...")
        
        # Create test feature snapshot
        features = FeatureSnapshot(
            symbol="BTC-USD",
            timestamp=datetime.utcnow(),
            mid_price=Decimal('50000'),
            spread_bps=5.0,
            return_1m=0.001,
            return_5m=0.002,
            volatility_5m=0.02,
            order_book_imbalance=0.1,
            order_book_pressure=0.05,
            volume_ratio=1.2,
            sma_5=Decimal('49900'),
            sma_20=Decimal('49800'),
            rsi_14=65.0
        )
        
        # Test each alpha model
        alpha_signals = {}
        
        for model_name, model in self.alpha_models.items():
            edge_bps, confidence = model.predict(features)
            alpha_signals[model_name] = (edge_bps, confidence)
            
            # Validate predictions
            assert isinstance(edge_bps, float)
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            assert -100 <= edge_bps <= 100  # Reasonable range
            
            # Get model stats
            stats = model.get_stats()
            assert stats['model_name'] == model_name
            assert stats['prediction_count'] > 0
        
        self.logger.info(f"✓ Layer 1: Generated signals from {len(alpha_signals)} models")
        for model_name, (edge, conf) in alpha_signals.items():
            self.logger.info(f"  {model_name}: edge={edge:.2f}bps, confidence={conf:.3f}")
    
    async def test_layer2_ensemble(self):
        """Test Layer 2: Ensemble meta-learner."""
        self.logger.info("Testing Layer 2: Ensemble meta-learner...")
        
        # Create test signals
        alpha_signals = {
            'order_book_pressure': (5.2, 0.7),
            'ma_momentum': (-2.1, 0.6)
        }
        
        features = FeatureSnapshot(
            symbol="BTC-USD",
            timestamp=datetime.utcnow(),
            mid_price=Decimal('50000'),
            spread_bps=5.0,
            volatility_5m=0.02,
            volume_ratio=1.1,
            order_book_imbalance=0.05,
            return_1m=0.001
        )
        
        # Test ensemble prediction
        ensemble_edge, ensemble_confidence = self.meta_learner.predict(alpha_signals, features)
        
        # Validate ensemble output
        assert isinstance(ensemble_edge, float)
        assert isinstance(ensemble_confidence, float)
        assert 0 <= ensemble_confidence <= 1
        
        # Since model is not trained, should use simple ensemble
        stats = self.meta_learner.get_stats()
        assert not stats['is_trained']
        assert stats['prediction_count'] > 0
        
        self.logger.info(f"✓ Layer 2: Ensemble edge={ensemble_edge:.2f}bps, confidence={ensemble_confidence:.3f}")
    
    async def test_layer3_position_sizing(self):
        """Test Layer 3: Position sizing with Kelly Criterion."""
        self.logger.info("Testing Layer 3: Position sizing...")
        
        # Test position sizing
        portfolio_value = Decimal('100000')
        current_price = Decimal('50000')
        edge_bps = 5.0
        confidence = 0.7
        
        position_size, reasoning = self.kelly_sizing.calculate_position_size(
            symbol="BTC-USD",
            edge_bps=edge_bps,
            confidence=confidence,
            current_price=current_price,
            portfolio_value=portfolio_value
        )
        
        # Validate position sizing
        assert isinstance(position_size, Decimal)
        assert isinstance(reasoning, str)
        assert abs(position_size) <= portfolio_value * Decimal('0.1')  # Max 10% position
        
        # Test with negative edge
        negative_position, _ = self.kelly_sizing.calculate_position_size(
            symbol="BTC-USD",
            edge_bps=-3.0,
            confidence=0.6,
            current_price=current_price,
            portfolio_value=portfolio_value
        )
        
        assert negative_position < 0  # Should be short position
        
        # Test risk limits
        risk_metrics = self.kelly_sizing.get_portfolio_risk_metrics()
        assert 'total_exposure' in risk_metrics
        assert 'win_rate' in risk_metrics
        
        self.logger.info(f"✓ Layer 3: Position size=${position_size:.0f} ({reasoning})")
    
    async def test_layer4_execution(self):
        """Test Layer 4: Order execution."""
        self.logger.info("Testing Layer 4: Order execution...")
        
        # Test order execution
        target_position = Decimal('5000')  # $5k position
        current_price = Decimal('50000')
        
        order = await self.executor.execute_order(
            symbol="BTC-USD",
            target_position=target_position,
            current_price=current_price
        )
        
        # Validate order execution
        assert order is not None
        assert order.symbol == "BTC-USD"
        assert order.status.value == "filled"
        
        # Check position tracking
        position_value = self.executor.get_position_value("BTC-USD", current_price)
        assert abs(position_value - target_position) < Decimal('50')  # Allow for slippage/commission
        
        # Test portfolio value calculation
        current_prices = {"BTC-USD": current_price}
        portfolio_value = self.executor.get_portfolio_value(current_prices)
        assert portfolio_value > Decimal('95000')  # Should be close to starting value
        
        # Test performance metrics
        performance = self.executor.get_performance_metrics()
        assert performance['total_trades'] > 0
        assert 'avg_slippage_bps' in performance
        
        self.logger.info(f"✓ Layer 4: Executed order, position=${position_value:.0f}, portfolio=${portfolio_value:.0f}")
    
    async def test_layer5_risk_management(self):
        """Test Layer 5: Risk management."""
        self.logger.info("Testing Layer 5: Risk management...")
        
        # Test position risk check
        portfolio_value = Decimal('100000')
        proposed_position = Decimal('20000')  # 20% position
        current_price = Decimal('50000')
        
        is_allowed, reason, max_allowed = self.risk_manager.check_position_risk(
            symbol="BTC-USD",
            proposed_position=proposed_position,
            current_price=current_price,
            portfolio_value=portfolio_value,
            volatility=0.3  # 30% volatility
        )
        
        # Should be limited due to size/volatility
        assert max_allowed < proposed_position
        
        # Test portfolio risk assessment
        current_positions = {"BTC-USD": Decimal('0.1')}  # 0.1 shares
        current_prices = {"BTC-USD": current_price}
        
        self.risk_manager.update_portfolio_value(portfolio_value)
        risk_metrics = self.risk_manager.check_portfolio_risk(
            current_positions=current_positions,
            current_prices=current_prices,
            portfolio_value=portfolio_value
        )
        
        # Validate risk metrics
        assert risk_metrics.portfolio_value == portfolio_value
        assert 0 <= risk_metrics.risk_score <= 100
        assert isinstance(risk_metrics.violations, list)
        
        # Test trading allowance
        trading_allowed, trading_status = self.risk_manager.is_trading_allowed(risk_metrics)
        assert isinstance(trading_allowed, bool)
        assert isinstance(trading_status, str)
        
        self.logger.info(f"✓ Layer 5: Risk score={risk_metrics.risk_score:.1f}, trading_allowed={trading_allowed}")
    
    async def test_end_to_end_pipeline(self):
        """Test end-to-end trading pipeline."""
        self.logger.info("Testing end-to-end pipeline...")
        
        portfolio_value = Decimal('100000')
        trades_executed = 0
        
        # Simulate multiple trading cycles
        for cycle in range(10):
            for symbol in self.test_symbols[:2]:
                # 1. Generate market data
                base_price = Decimal('50000') if 'BTC' in symbol else Decimal('150')
                price_change = np.random.normal(0, 0.005) * float(base_price)
                current_price = base_price + Decimal(str(price_change))
                
                tick = MarketTick(
                    symbol=symbol,
                    exchange="test",
                    asset_type="crypto" if 'BTC' in symbol or 'ETH' in symbol else "stock",
                    timestamp=datetime.utcnow(),
                    bid=current_price - Decimal('1'),
                    ask=current_price + Decimal('1'),
                    mid=current_price,
                    volume=Decimal('100'),
                    bid_size=Decimal('1.0'),
                    ask_size=Decimal('1.2')
                )
                
                # 2. Compute features
                features = await self.feature_bus.process_tick(tick)
                if not features:
                    continue
                
                # 3. Generate alpha signals
                alpha_signals = {}
                for model_name, model in self.alpha_models.items():
                    edge_bps, confidence = model.predict(features)
                    alpha_signals[model_name] = (edge_bps, confidence)
                
                # 4. Ensemble prediction
                ensemble_edge, ensemble_confidence = self.meta_learner.predict(alpha_signals, features)
                
                # 5. Position sizing
                position_size, reasoning = self.kelly_sizing.calculate_position_size(
                    symbol=symbol,
                    edge_bps=ensemble_edge,
                    confidence=ensemble_confidence,
                    current_price=current_price,
                    portfolio_value=portfolio_value
                )
                
                # 6. Risk check
                is_allowed, risk_reason, max_allowed = self.risk_manager.check_position_risk(
                    symbol=symbol,
                    proposed_position=position_size,
                    current_price=current_price,
                    portfolio_value=portfolio_value
                )
                
                # 7. Execute if allowed
                if is_allowed and abs(max_allowed) > Decimal('50'):
                    order = await self.executor.execute_order(
                        symbol=symbol,
                        target_position=max_allowed,
                        current_price=current_price
                    )
                    
                    if order:
                        trades_executed += 1
                        
                        # Update portfolio value
                        current_prices = {symbol: current_price}
                        portfolio_value = self.executor.get_portfolio_value(current_prices)
                        self.risk_manager.update_portfolio_value(portfolio_value)
        
        self.logger.info(f"✓ End-to-end: Executed {trades_executed} trades, final portfolio=${portfolio_value:.0f}")
    
    async def test_performance_metrics(self):
        """Test performance metrics and system stats."""
        self.logger.info("Testing performance metrics...")
        
        # Collect stats from all components
        stats = {
            'feature_bus': self.feature_bus.get_stats(),
            'alpha_models': {name: model.get_stats() for name, model in self.alpha_models.items()},
            'meta_learner': self.meta_learner.get_stats(),
            'kelly_sizing': self.kelly_sizing.get_stats(),
            'executor': self.executor.get_stats(),
            'risk_manager': self.risk_manager.get_stats()
        }
        
        # Validate key performance metrics
        assert stats['feature_bus']['performance_target_met']  # <300µs feature computation
        assert stats['executor']['performance']['total_trades'] > 0
        
        # Print summary
        self.logger.info("✓ Performance metrics collected:")
        self.logger.info(f"  Features computed: {stats['feature_bus']['features_computed']}")
        self.logger.info(f"  Alpha predictions: {sum(m['prediction_count'] for m in stats['alpha_models'].values())}")
        self.logger.info(f"  Ensemble predictions: {stats['meta_learner']['prediction_count']}")
        self.logger.info(f"  Trades executed: {stats['executor']['performance']['total_trades']}")
        self.logger.info(f"  Avg computation time: {stats['feature_bus']['avg_computation_time_us']:.1f}µs")
    
    def generate_synthetic_tick(self, symbol: str, base_price: Decimal) -> MarketTick:
        """Generate realistic synthetic market tick."""
        price_change = np.random.normal(0, 0.002) * float(base_price)
        current_price = base_price + Decimal(str(price_change))
        
        return MarketTick(
            symbol=symbol,
            exchange="test_exchange",
            asset_type="crypto" if symbol.endswith('-USD') else "stock",
            timestamp=datetime.utcnow(),
            bid=current_price - Decimal('1'),
            ask=current_price + Decimal('1'),
            mid=current_price,
            last=current_price + Decimal(str(np.random.normal(0, 0.1))),
            volume=Decimal(str(abs(np.random.lognormal(4, 1)))),
            bid_size=Decimal(str(abs(np.random.lognormal(0, 0.5)))),
            ask_size=Decimal(str(abs(np.random.lognormal(0, 0.5))))
        )


async def main():
    """Main test function."""
    test_system = TradingSystemTest()
    
    try:
        await test_system.run_comprehensive_test()
        print("\n🎉 All tests passed! Trading system implementation is working correctly.")
        return 0
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 