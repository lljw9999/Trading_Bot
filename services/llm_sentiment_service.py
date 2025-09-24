#!/usr/bin/env python3
"""
GPT-4 Sentiment Analysis Microservice
Real-time news sentiment analysis using OpenAI's GPT-4o-mini for crypto headlines
"""

import os
import json
import time
import redis
import tenacity
import openai
import logging
from typing import Dict, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("llm_sentiment")


class LLMSentimentService:
    """GPT-powered sentiment analysis microservice for crypto news."""

    def __init__(self):
        """Initialize LLM sentiment service."""
        # OpenAI configuration
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_KEY", "sk-test-key")
        )
        self.model = "gpt-4o-mini"  # 128k context, cost-effective

        # Redis connection
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Service configuration
        self.max_retries = 3
        self.timeout = 15
        self.rate_limit_delay = 1

        logger.info(f"🤖 LLM Sentiment Service initialized (model: {self.model})")

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError)
        ),
    )
    def _score_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Score sentiment using GPT-4o-mini with retry logic.

        Args:
            text: News headline or article text

        Returns:
            Dictionary with sentiment and impact score
        """
        try:
            prompt = f"""Classify sentiment (bullish, bearish, neutral) and 0-1 impact for crypto headline:

"{text}"

Respond JSON: {{"sentiment":"", "impact":0.0}}

Guidelines:
- sentiment: "bullish" (positive for crypto), "bearish" (negative), "neutral"
- impact: 0.0-1.0 (0=no impact, 1=maximum market impact)
- Consider regulatory news, adoption, technical developments, market events"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                temperature=0.1,  # Low temperature for consistent responses
                max_tokens=100,  # Short responses
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result = json.loads(content)

                # Validate and clean response
                sentiment = result.get("sentiment", "neutral").lower()
                if sentiment not in ["bullish", "bearish", "neutral"]:
                    sentiment = "neutral"

                impact = float(result.get("impact", 0.0))
                impact = max(0.0, min(1.0, impact))  # Clamp to [0, 1]

                return {
                    "sentiment": sentiment,
                    "impact": impact,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                }

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse GPT response: {content}, error: {e}")
                return {
                    "sentiment": "neutral",
                    "impact": 0.0,
                    "model": self.model,
                    "error": "parse_error",
                }

        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit, backing off...")
            raise
        except openai.APITimeoutError:
            logger.warning("OpenAI API timeout, retrying...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in sentiment scoring: {e}")
            return {
                "sentiment": "neutral",
                "impact": 0.0,
                "model": self.model,
                "error": str(e),
            }

    def process_news_headline(self, headline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single news headline for sentiment.

        Args:
            headline_data: News headline with title, source, etc.

        Returns:
            Enhanced headline data with sentiment scores
        """
        try:
            title = headline_data.get("title", "")
            if not title:
                logger.warning("Empty title in headline data")
                return headline_data

            # Get sentiment scores from GPT
            sentiment_result = self._score_sentiment(title)

            # Create enhanced payload
            enhanced_payload = {
                **headline_data,
                **sentiment_result,
                "ts": int(time.time()),
                "processing_service": "llm_sentiment",
            }

            # Store in Redis streams
            self.redis.xadd("event:news_llm", enhanced_payload, maxlen=5000)

            # Update latest sentiment features for state builder
            sentiment_features = {
                "impact": sentiment_result["impact"],
                "bull": int(sentiment_result["sentiment"] == "bullish"),
                "bear": int(sentiment_result["sentiment"] == "bearish"),
                "neutral": int(sentiment_result["sentiment"] == "neutral"),
                "last_update": int(time.time()),
            }

            self.redis.hset("sentiment:latest", mapping=sentiment_features)

            logger.info(
                f"LLM sentiment: {sentiment_result['sentiment']} (impact: {sentiment_result['impact']:.2f}) - '{title[:60]}...'"
            )

            return enhanced_payload

        except Exception as e:
            logger.error(f"Error processing headline: {e}")
            return headline_data

    def stream_news_processing(self):
        """
        Main processing loop - listen to news:raw and process with sentiment.
        """
        logger.info("🚀 Starting LLM sentiment news processing stream")

        try:
            # Subscribe to raw news feed
            pubsub = self.redis.pubsub()
            pubsub.subscribe("news:raw")

            logger.info("📰 Subscribed to news:raw channel")

            for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                try:
                    # Parse incoming news data
                    news_data = json.loads(message["data"])

                    # Process with sentiment analysis
                    processed_data = self.process_news_headline(news_data)

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in news message: {e}")
                except Exception as e:
                    logger.error(f"Error processing news message: {e}")

        except KeyboardInterrupt:
            logger.info("🛑 LLM sentiment service stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error in news processing: {e}")
            raise
        finally:
            try:
                pubsub.close()
            except:
                pass

    def test_sentiment_analysis(self):
        """Test sentiment analysis with sample headlines."""
        test_headlines = [
            "Bitcoin surges to new all-time high as institutional adoption accelerates",
            "Major crypto exchange hacked, $100M stolen in security breach",
            "Federal Reserve announces digital dollar pilot program",
            "Ethereum upgrade successfully reduces gas fees by 90%",
            "SEC approves spot Bitcoin ETF applications from major banks",
            "China bans cryptocurrency mining operations nationwide",
        ]

        logger.info("🧪 Testing LLM sentiment analysis...")

        for headline in test_headlines:
            result = self._score_sentiment(headline)
            logger.info(
                f"Test: '{headline[:50]}...' → {result['sentiment']} (impact: {result['impact']:.2f})"
            )
            time.sleep(1)  # Rate limiting

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            # Get latest sentiment state
            latest_sentiment = self.redis.hgetall("sentiment:latest")

            # Get recent news events count
            recent_events = self.redis.xlen("event:news_llm")

            return {
                "service": "llm_sentiment",
                "status": "active",
                "latest_sentiment": latest_sentiment,
                "processed_events": recent_events,
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"service": "llm_sentiment", "status": "error", "error": str(e)}


def inject_test_news():
    """Inject test news for development/testing."""
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    test_news = [
        {
            "title": "Bitcoin breaks $50K as institutional demand surges",
            "source": "CryptoNews",
            "timestamp": int(time.time()),
            "url": "https://example.com/news1",
        },
        {
            "title": "Major exchange reports security vulnerability",
            "source": "BlockchainToday",
            "timestamp": int(time.time()),
            "url": "https://example.com/news2",
        },
        {
            "title": "New DeFi protocol launches with $100M TVL",
            "source": "DeFiTimes",
            "timestamp": int(time.time()),
            "url": "https://example.com/news3",
        },
    ]

    for news in test_news:
        redis_client.publish("news:raw", json.dumps(news))
        logger.info(f"Published test news: {news['title']}")
        time.sleep(2)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Sentiment Analysis Service")
    parser.add_argument(
        "--test", action="store_true", help="Run sentiment analysis tests"
    )
    parser.add_argument("--inject-news", action="store_true", help="Inject test news")
    parser.add_argument("--stats", action="store_true", help="Show service statistics")

    args = parser.parse_args()

    service = LLMSentimentService()

    if args.test:
        service.test_sentiment_analysis()
        return

    if args.inject_news:
        inject_test_news()
        return

    if args.stats:
        stats = service.get_service_stats()
        print(json.dumps(stats, indent=2))
        return

    # Default: run processing stream
    service.stream_news_processing()


if __name__ == "__main__":
    main()
