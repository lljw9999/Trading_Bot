#!/usr/bin/env python3
"""
Unit tests for explain middleware API.

Tests the GPT-4o powered trade explanation service with mocked OpenAI responses.
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.services import explain_middleware
from src.services.explain_middleware import ExplainService, OrderEvent, TradeExplanation

explain_middleware.USE_OPENAI_MOCK = False


class TestExplainAPI(unittest.TestCase):
    """Test explanation API functionality with mocked OpenAI."""

    def setUp(self):
        """Set up test fixtures."""
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        self.explain_service = ExplainService()
        self.explain_service.redis_client = AsyncMock()

        # Mock order event
        self.test_order = OrderEvent(
            order_id="test_order_123",
            symbol="BTC-USD",
            side="buy",
            quantity=0.1,
            price=50000.0,
            timestamp=datetime.now().isoformat(),
            order_type="market",
            edge_bps=25.5,
            confidence=0.8,
            portfolio_value=100000.0,
            position_size_pct=0.05,
            sentiment_score=0.7,
            technical_signal="MA_crossover_bullish",
            big_bet_flag=False,
            risk_metrics={"volatility": 0.03, "correlation": 0.2},
        )

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    def test_successful_explanation_generation(self, mock_openai_class):
        """Test successful generation of trade explanation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Strong bullish momentum detected with 25.5bp edge. "
            "Positive sentiment (0.7) and MA crossover signal support "
            "the 5% position size. Risk: Low-moderate volatility."
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Generate explanation
        result = asyncio.run(self.explain_service.generate_explanation(self.test_order))

        # Verify result
        self.assertIsInstance(result, TradeExplanation)
        self.assertEqual(result.order_id, "test_order_123")
        self.assertEqual(result.symbol, "BTC-USD")
        self.assertIn("bullish momentum", result.explanation)
        self.assertEqual(result.confidence_level, "High")  # 0.8 confidence
        self.assertTrue(
            any("positive sentiment" in factor.lower() for factor in result.key_factors)
        )
        self.assertGreater(result.processing_time_ms, 0)

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    def test_openai_api_error_fallback(self, mock_openai_class):
        """Test fallback explanation when OpenAI API fails."""
        # Mock OpenAI API error
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )
        mock_openai_class.return_value = mock_client

        # Generate explanation
        result = asyncio.run(self.explain_service.generate_explanation(self.test_order))

        # Verify fallback result
        self.assertIsInstance(result, TradeExplanation)
        self.assertEqual(result.order_id, "test_order_123")
        self.assertIn("25.5bp edge", result.explanation)
        self.assertIn("80% confidence", result.explanation)
        self.assertIn("API rate limit exceeded", result.explanation)
        self.assertEqual(result.key_factors, ["Analysis Error"])

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    def test_explanation_with_big_bet_flag(self, mock_openai_class):
        """Test explanation generation for big bet trades."""
        # Mock OpenAI response for big bet
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "HIGH-CONFIDENCE BIG BET: Exceptional opportunity with "
            "strong sentiment convergence and earnings surprise. "
            "40bp edge justifies aggressive 15% position."
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create big bet order
        big_bet_order = OrderEvent(
            order_id="big_bet_456",
            symbol="NVDA",
            side="buy",
            quantity=10,
            price=800.0,
            timestamp=datetime.now().isoformat(),
            order_type="market",
            edge_bps=40.0,
            confidence=0.9,
            portfolio_value=100000.0,
            position_size_pct=0.15,
            sentiment_score=0.85,
            technical_signal="earnings_surprise",
            big_bet_flag=True,
            risk_metrics={"volatility": 0.05},
        )

        result = asyncio.run(self.explain_service.generate_explanation(big_bet_order))

        # Verify big bet characteristics
        self.assertEqual(result.confidence_level, "Very High")  # 0.9 confidence
        self.assertIn("High-confidence big bet", result.key_factors)
        self.assertIn("Large position", result.key_factors)
        self.assertEqual(result.risk_assessment, "High Risk")  # 15% position + big bet

    def test_confidence_level_determination(self):
        """Test confidence level classification logic."""
        test_cases = [
            (0.9, "Very High"),
            (0.7, "High"),
            (0.5, "Moderate"),
            (0.3, "Low"),
            (None, "Low"),
        ]

        for confidence, expected_level in test_cases:
            result = self.explain_service._determine_confidence_level(confidence)
            self.assertEqual(result, expected_level)

    def test_risk_level_assessment(self):
        """Test risk level assessment logic."""
        # Low risk order
        low_risk_order = OrderEvent(
            order_id="low_risk",
            symbol="BTC-USD",
            side="buy",
            quantity=0.01,
            price=50000.0,
            timestamp=datetime.now().isoformat(),
            order_type="limit",
            confidence=0.8,
            position_size_pct=0.02,
            big_bet_flag=False,
        )

        risk = self.explain_service._assess_risk_level(low_risk_order)
        self.assertEqual(risk, "Very Low Risk")

        # High risk order
        high_risk_order = OrderEvent(
            order_id="high_risk",
            symbol="SOL-USD",
            side="buy",
            quantity=100,
            price=200.0,
            timestamp=datetime.now().isoformat(),
            order_type="market",
            confidence=0.3,
            position_size_pct=0.25,
            big_bet_flag=True,
        )

        risk = self.explain_service._assess_risk_level(high_risk_order)
        self.assertEqual(risk, "High Risk")

    def test_key_factors_extraction(self):
        """Test extraction of key trading factors."""
        factors = self.explain_service._extract_key_factors(self.test_order)

        # Should include strong edge and moderate sentiment
        self.assertIn("Strong edge (26bp)", factors)  # 25.5 rounds to 26
        self.assertTrue(any("positive sentiment" in f.lower() for f in factors))
        self.assertEqual(len(factors), 3)  # Limited to top 3

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    def test_explanation_with_negative_sentiment(self, mock_openai_class):
        """Test explanation for negative sentiment trade."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "SHORT opportunity based on strong negative sentiment (-0.8) "
            "and technical breakdown. 30bp edge supports defensive position."
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create negative sentiment order
        negative_order = OrderEvent(
            order_id="short_123",
            symbol="TSLA",
            side="sell",
            quantity=5,
            price=250.0,
            timestamp=datetime.now().isoformat(),
            order_type="limit",
            edge_bps=-30.0,
            confidence=0.7,
            sentiment_score=-0.8,
            technical_signal="breakdown_pattern",
        )

        result = asyncio.run(self.explain_service.generate_explanation(negative_order))

        # Verify negative sentiment handling
        factors = self.explain_service._extract_key_factors(negative_order)
        self.assertTrue(
            any("negative sentiment" in factor.lower() for factor in factors)
        )

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    def test_explanation_prompt_formatting(self, mock_openai_class):
        """Test that explanation prompt is properly formatted."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test explanation"))]
        )
        mock_openai_class.return_value = mock_client

        # Generate explanation
        asyncio.run(self.explain_service.generate_explanation(self.test_order))

        # Verify prompt was called with correct parameters
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-4")
        self.assertEqual(call_args[1]["max_tokens"], 100)
        self.assertEqual(call_args[1]["temperature"], 0.3)

        # Check prompt contains order details
        messages = call_args[1]["messages"]
        prompt_content = messages[1]["content"]
        self.assertIn("BTC-USD", prompt_content)
        self.assertIn("buy", prompt_content)
        self.assertIn("25.5", prompt_content)  # edge_bps
        self.assertIn("0.7", prompt_content)  # sentiment_score

    def test_redis_storage(self):
        """Test explanation storage in Redis."""
        explanation = TradeExplanation(
            order_id="test_store",
            symbol="BTC-USD",
            explanation="Test explanation",
            confidence_level="High",
            key_factors=["Factor 1", "Factor 2"],
            risk_assessment="Low Risk",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=150.0,
        )

        # Store explanation
        asyncio.run(self.explain_service.store_explanation(explanation))

        # Verify Redis calls
        self.assertTrue(self.explain_service.redis_client.setex.called)
        self.assertTrue(self.explain_service.redis_client.lpush.called)
        self.assertTrue(self.explain_service.redis_client.ltrim.called)


class TestExplainAPIIntegration(unittest.TestCase):
    """Integration tests for explanation API."""

    def setUp(self):
        """Set up integration test fixtures."""
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        self.explain_service = ExplainService()
        self.explain_service.redis_client = AsyncMock()

    @patch("src.services.explain_middleware.openai.AsyncOpenAI")
    @patch("src.services.explain_middleware.redis.Redis")
    def test_end_to_end_explanation_flow(self, mock_redis_class, mock_openai_class):
        """Test complete explanation generation and storage flow."""
        # Mock OpenAI
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Complete explanation test"

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_client

        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis_class.return_value = mock_redis
        self.explain_service.redis_client = mock_redis

        # Create test order
        order = OrderEvent(
            order_id="integration_test",
            symbol="ETH-USD",
            side="buy",
            quantity=1.0,
            price=3000.0,
            timestamp=datetime.now().isoformat(),
            order_type="market",
        )

        # Run end-to-end flow
        explanation = asyncio.run(self.explain_service.generate_explanation(order))
        asyncio.run(self.explain_service.store_explanation(explanation))

        # Verify explanation was generated
        self.assertIsInstance(explanation, TradeExplanation)
        self.assertEqual(explanation.order_id, "integration_test")

        # Verify storage calls
        self.assertTrue(mock_redis.setex.called)
        self.assertTrue(mock_redis.lpush.called)


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestExplainAPI))
    suite.addTest(unittest.makeSuite(TestExplainAPIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All explanation API tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
