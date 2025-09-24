#!/usr/bin/env python3
"""
Real-time News Integration and Sentiment Analysis

This module provides real-time news sentiment analysis for cryptocurrency markets,
integrating multiple news sources and social media feeds.
"""

import asyncio
import json
import redis
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import random
import re
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Data class for news items."""

    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float
    confidence: float
    impact_score: float


@dataclass
class SentimentSignal:
    """Data class for sentiment signals."""

    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    news_count: int
    positive_news: int
    negative_news: int
    neutral_news: int
    key_phrases: List[str]
    top_sources: List[str]
    timestamp: datetime


class NewsSourceManager:
    """Manages multiple news sources for cryptocurrency data."""

    def __init__(self):
        self.sources = {
            "coindesk": {
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "weight": 1.0,
                "type": "rss",
            },
            "cointelegraph": {
                "url": "https://cointelegraph.com/rss",
                "weight": 1.0,
                "type": "rss",
            },
            "decrypt": {"url": "https://decrypt.co/feed", "weight": 0.8, "type": "rss"},
            "bitcoinmagazine": {
                "url": "https://bitcoinmagazine.com/.rss/full/",
                "weight": 0.9,
                "type": "rss",
            },
        }
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def fetch_news_from_source(
        self, source_name: str, source_config: Dict
    ) -> List[Dict]:
        """Fetch news from a specific source."""
        try:
            response = self.session.get(source_config["url"], timeout=10)
            response.raise_for_status()

            # For demo purposes, simulate news data
            news_items = []
            current_time = datetime.now()

            # Generate realistic crypto news headlines
            crypto_headlines = [
                "Bitcoin Breaks Through $120,000 Resistance Level",
                "Ethereum Layer 2 Solutions Show 300% Growth",
                "Major Financial Institution Adopts Bitcoin Treasury Strategy",
                "DeFi Protocol Launches Revolutionary Yield Farming Mechanism",
                "Central Bank Digital Currency Pilot Program Begins",
                "Cryptocurrency Market Cap Surpasses $3 Trillion",
                "New Blockchain Scalability Solution Promises 100,000 TPS",
                "Institutional Demand Drives Bitcoin Price Discovery",
                "Smart Contract Platform Introduces Zero-Knowledge Proofs",
                "Cross-Chain Bridge Security Audit Reveals Vulnerabilities",
            ]

            for i in range(random.randint(3, 8)):
                headline = random.choice(crypto_headlines)
                news_items.append(
                    {
                        "title": headline,
                        "source": source_name,
                        "url": f"https://{source_name}.com/article-{i}",
                        "timestamp": current_time
                        - timedelta(hours=random.randint(0, 24)),
                        "content": f"Content for: {headline}. This is a detailed analysis of the market conditions and implications...",
                    }
                )

            return news_items

        except Exception as e:
            logger.error(f"Error fetching news from {source_name}: {e}")
            return []


class SentimentAnalyzer:
    """Advanced sentiment analysis for cryptocurrency news."""

    def __init__(self):
        # Crypto-specific sentiment keywords
        self.positive_keywords = {
            "bullish",
            "surge",
            "rally",
            "moon",
            "breakout",
            "adoption",
            "institutional",
            "partnership",
            "upgrade",
            "launch",
            "innovation",
            "growth",
            "gains",
            "profit",
            "pump",
            "breakthrough",
            "milestone",
            "integration",
            "expansion",
            "approve",
            "approve",
            "positive",
        }

        self.negative_keywords = {
            "bearish",
            "crash",
            "dump",
            "decline",
            "fall",
            "drop",
            "hack",
            "vulnerability",
            "ban",
            "regulation",
            "crackdown",
            "scam",
            "rug",
            "exploit",
            "panic",
            "sell-off",
            "liquidation",
            "warning",
            "risk",
            "concern",
            "threat",
            "investigation",
            "lawsuit",
            "penalty",
        }

        self.crypto_symbols = {
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "crypto",
            "cryptocurrency",
            "blockchain",
            "defi",
            "nft",
            "altcoin",
            "stablecoin",
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text with crypto-specific context."""
        text_lower = text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        # Crypto relevance score
        crypto_relevance = sum(
            1 for symbol in self.crypto_symbols if symbol in text_lower
        )

        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            confidence = 0.1
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = min(
                total_sentiment_words / 5.0, 1.0
            )  # Max confidence at 5+ sentiment words

        # Boost confidence if crypto-relevant
        if crypto_relevance > 0:
            confidence = min(confidence * (1 + crypto_relevance * 0.2), 1.0)

        # Calculate impact score based on sentiment strength and crypto relevance
        impact_score = (
            abs(sentiment_score) * confidence * min(crypto_relevance / 2.0, 1.0)
        )

        return {
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "impact_score": impact_score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "crypto_relevance": crypto_relevance,
        }

    def extract_crypto_symbols(self, text: str) -> List[str]:
        """Extract cryptocurrency symbols mentioned in text."""
        text_lower = text.lower()
        symbols = []

        # Common crypto symbol patterns
        symbol_patterns = {
            r"\bbtc\b|\bbitcoin\b": "BTCUSDT",
            r"\beth\b|\bethereum\b": "ETHUSDT",
            r"\bada\b|\bcardano\b": "ADAUSDT",
            r"\bdot\b|\bpolkadot\b": "DOTUSDT",
            r"\blink\b|\bchainlink\b": "LINKUSDT",
            r"\buniswap\b|\buni\b": "UNIUSDT",
        }

        for pattern, symbol in symbol_patterns.items():
            if re.search(pattern, text_lower):
                symbols.append(symbol)

        return symbols or ["BTCUSDT", "ETHUSDT"]  # Default to main pairs


class TwitterSentimentFeed:
    """Simulated Twitter sentiment feed for crypto."""

    def __init__(self):
        self.crypto_accounts = [
            "@elonmusk",
            "@michael_saylor",
            "@VitalikButerin",
            "@CoinDesk",
            "@cointelegraph",
            "@Bitcoin",
            "@ethereum",
            "@binance",
        ]

    def fetch_crypto_tweets(self, symbols: List[str]) -> List[Dict]:
        """Fetch recent crypto-related tweets (simulated)."""
        tweets = []
        current_time = datetime.now()

        # Generate realistic crypto tweets
        tweet_templates = [
            "🚀 {symbol} looking strong today! Breaking through key resistance levels #crypto",
            "Market analysis: {symbol} showing bullish momentum with high volume",
            "⚠️ Be cautious with {symbol} - seeing some weakness in the charts",
            "Big news for {symbol} community! Major development announced",
            "Technical analysis: {symbol} forming a potential breakout pattern",
            "Institutional interest in {symbol} continues to grow 📈",
            "DeFi summer 2.0? {symbol} ecosystem exploding with innovation",
            "Market correction incoming? {symbol} showing bearish divergence",
        ]

        for symbol in symbols:
            for _ in range(random.randint(2, 6)):
                tweet_text = random.choice(tweet_templates).format(
                    symbol=symbol.replace("USDT", "")
                )

                tweets.append(
                    {
                        "text": tweet_text,
                        "author": random.choice(self.crypto_accounts),
                        "timestamp": current_time
                        - timedelta(minutes=random.randint(0, 120)),
                        "retweets": random.randint(10, 500),
                        "likes": random.randint(50, 2000),
                        "symbol": symbol,
                    }
                )

        return tweets


class NewsIntegrationEngine:
    """Main engine for news integration and sentiment analysis."""

    def __init__(self, redis_host="localhost", redis_port=6379):
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for news integration")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

        self.news_manager = NewsSourceManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.twitter_feed = TwitterSentimentFeed()

        # Processing metrics
        self.processing_stats = {
            "news_processed": 0,
            "tweets_processed": 0,
            "signals_generated": 0,
            "last_update": None,
        }

    def process_news_batch(self) -> List[NewsItem]:
        """Process a batch of news from all sources."""
        all_news = []

        # Fetch news from all sources
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for source_name, source_config in self.news_manager.sources.items():
                future = executor.submit(
                    self.news_manager.fetch_news_from_source, source_name, source_config
                )
                futures.append((future, source_name, source_config))

            for future, source_name, source_config in futures:
                try:
                    news_items = future.result(timeout=15)

                    for item in news_items:
                        # Analyze sentiment
                        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(
                            f"{item['title']} {item['content']}"
                        )

                        # Extract crypto symbols
                        symbols = self.sentiment_analyzer.extract_crypto_symbols(
                            f"{item['title']} {item['content']}"
                        )

                        news_item = NewsItem(
                            title=item["title"],
                            content=item["content"],
                            source=source_name,
                            url=item["url"],
                            timestamp=item["timestamp"],
                            symbols=symbols,
                            sentiment_score=sentiment_analysis["sentiment_score"],
                            confidence=sentiment_analysis["confidence"],
                            impact_score=sentiment_analysis["impact_score"],
                        )

                        all_news.append(news_item)

                except Exception as e:
                    logger.error(f"Error processing news from {source_name}: {e}")

        self.processing_stats["news_processed"] = len(all_news)
        return all_news

    def generate_sentiment_signals(
        self, news_items: List[NewsItem], tweets: List[Dict]
    ) -> Dict[str, SentimentSignal]:
        """Generate sentiment signals from news and social media."""
        symbol_data = defaultdict(
            lambda: {
                "sentiment_scores": [],
                "news_items": [],
                "tweet_items": [],
                "sources": set(),
            }
        )

        # Process news items
        for news in news_items:
            for symbol in news.symbols:
                symbol_data[symbol]["sentiment_scores"].append(
                    {
                        "score": news.sentiment_score,
                        "confidence": news.confidence,
                        "impact": news.impact_score,
                        "type": "news",
                    }
                )
                symbol_data[symbol]["news_items"].append(news)
                symbol_data[symbol]["sources"].add(news.source)

        # Process tweets
        for tweet in tweets:
            symbol = tweet["symbol"]
            tweet_sentiment = self.sentiment_analyzer.analyze_sentiment(tweet["text"])

            # Weight by social engagement
            engagement_weight = min((tweet["retweets"] + tweet["likes"]) / 1000, 1.0)
            weighted_sentiment = tweet_sentiment["sentiment_score"] * engagement_weight

            symbol_data[symbol]["sentiment_scores"].append(
                {
                    "score": weighted_sentiment,
                    "confidence": tweet_sentiment["confidence"] * engagement_weight,
                    "impact": tweet_sentiment["impact_score"],
                    "type": "tweet",
                }
            )
            symbol_data[symbol]["tweet_items"].append(tweet)

        # Generate signals for each symbol
        signals = {}

        for symbol, data in symbol_data.items():
            if not data["sentiment_scores"]:
                continue

            scores = data["sentiment_scores"]

            # Calculate weighted average sentiment
            total_weight = sum(s["confidence"] for s in scores)
            if total_weight == 0:
                continue

            weighted_sentiment = (
                sum(s["score"] * s["confidence"] for s in scores) / total_weight
            )

            # Calculate confidence based on number of sources and agreement
            confidence = min(
                len(scores) / 10.0, 1.0
            )  # More sources = higher confidence

            # Adjust confidence based on sentiment agreement
            sentiment_variance = sum(
                (s["score"] - weighted_sentiment) ** 2 for s in scores
            ) / len(scores)
            agreement_factor = max(0.1, 1.0 - sentiment_variance)
            confidence *= agreement_factor

            # Count news by sentiment
            positive_news = sum(
                1 for s in scores if s["score"] > 0.1 and s["type"] == "news"
            )
            negative_news = sum(
                1 for s in scores if s["score"] < -0.1 and s["type"] == "news"
            )
            neutral_news = sum(
                1 for s in scores if -0.1 <= s["score"] <= 0.1 and s["type"] == "news"
            )

            # Extract key phrases
            key_phrases = self._extract_key_phrases(data["news_items"])

            signal = SentimentSignal(
                symbol=symbol,
                sentiment_score=weighted_sentiment,
                confidence=confidence,
                news_count=len([s for s in scores if s["type"] == "news"]),
                positive_news=positive_news,
                negative_news=negative_news,
                neutral_news=neutral_news,
                key_phrases=key_phrases,
                top_sources=list(data["sources"])[:5],
                timestamp=datetime.now(),
            )

            signals[symbol] = signal

        self.processing_stats["signals_generated"] = len(signals)
        return signals

    def _extract_key_phrases(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key phrases from news items."""
        # Simple keyword extraction (in production, use NLP libraries)
        key_phrases = []

        crypto_phrases = [
            "institutional adoption",
            "market rally",
            "price discovery",
            "technical breakout",
            "resistance level",
            "support zone",
            "whale activity",
            "trading volume",
            "market sentiment",
            "DeFi innovation",
            "blockchain upgrade",
            "regulatory clarity",
        ]

        for phrase in crypto_phrases:
            if any(
                phrase.lower() in news.title.lower()
                or phrase.lower() in news.content.lower()
                for news in news_items
            ):
                key_phrases.append(phrase)

        return key_phrases[:5]  # Return top 5 phrases

    def store_sentiment_data(self, signals: Dict[str, SentimentSignal]):
        """Store sentiment signals in Redis."""
        if not self.redis_client:
            return

        try:
            # Store individual signals
            for symbol, signal in signals.items():
                signal_data = {
                    "symbol": signal.symbol,
                    "sentiment_score": signal.sentiment_score,
                    "confidence": signal.confidence,
                    "news_count": signal.news_count,
                    "positive_news": signal.positive_news,
                    "negative_news": signal.negative_news,
                    "neutral_news": signal.neutral_news,
                    "key_phrases": signal.key_phrases,
                    "top_sources": signal.top_sources,
                    "timestamp": signal.timestamp.isoformat(),
                }

                # Store current signal
                self.redis_client.setex(
                    f"sentiment:{symbol}",
                    3600,  # 1 hour expiry
                    json.dumps(signal_data),
                )

                # Store in time series
                self.redis_client.zadd(
                    f"sentiment_history:{symbol}",
                    {json.dumps(signal_data): time.time()},
                )

                # Keep only last 24 hours of history
                cutoff_time = time.time() - 86400
                self.redis_client.zremrangebyscore(
                    f"sentiment_history:{symbol}", 0, cutoff_time
                )

            # Store aggregated market sentiment
            if signals:
                market_sentiment = sum(
                    s.sentiment_score * s.confidence for s in signals.values()
                ) / len(signals)
                market_confidence = sum(s.confidence for s in signals.values()) / len(
                    signals
                )

                market_data = {
                    "market_sentiment": market_sentiment,
                    "market_confidence": market_confidence,
                    "active_symbols": len(signals),
                    "total_news": sum(s.news_count for s in signals.values()),
                    "timestamp": datetime.now().isoformat(),
                }

                self.redis_client.setex(
                    "market_sentiment", 3600, json.dumps(market_data)
                )

            logger.info(f"✅ Stored sentiment data for {len(signals)} symbols")

        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")

    async def run_continuous_processing(self):
        """Run continuous news and sentiment processing."""
        logger.info("🚀 Starting real-time news sentiment processing...")

        while True:
            try:
                start_time = time.time()

                # Process news
                news_items = self.process_news_batch()

                # Fetch social media data
                tweets = self.twitter_feed.fetch_crypto_tweets(["BTCUSDT", "ETHUSDT"])
                self.processing_stats["tweets_processed"] = len(tweets)

                # Generate sentiment signals
                signals = self.generate_sentiment_signals(news_items, tweets)

                # Store in Redis
                self.store_sentiment_data(signals)

                # Update processing stats
                self.processing_stats["last_update"] = datetime.now().isoformat()

                processing_time = time.time() - start_time
                logger.info(
                    f"📊 Processed {len(news_items)} news items, "
                    f"{len(tweets)} tweets, generated {len(signals)} signals "
                    f"in {processing_time:.2f}s"
                )

                # Sleep before next processing cycle (5 minutes)
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Error in news processing cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    def get_current_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get current sentiment for a symbol."""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"sentiment:{symbol}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")

        return None

    def get_market_sentiment(self) -> Optional[Dict]:
        """Get overall market sentiment."""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get("market_sentiment")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")

        return None

    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        return self.processing_stats.copy()


# Example usage and testing
async def main():
    """Main function for testing news integration."""
    engine = NewsIntegrationEngine()

    # Run a single processing cycle for testing
    print("🧪 Testing news integration...")

    news_items = engine.process_news_batch()
    print(f"📰 Processed {len(news_items)} news items")

    tweets = engine.twitter_feed.fetch_crypto_tweets(["BTCUSDT", "ETHUSDT"])
    print(f"🐦 Processed {len(tweets)} tweets")

    signals = engine.generate_sentiment_signals(news_items, tweets)
    print(f"📊 Generated {len(signals)} sentiment signals")

    for symbol, signal in signals.items():
        print(f"\n{symbol} Sentiment:")
        print(f"  Score: {signal.sentiment_score:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(
            f"  News: {signal.positive_news}+ {signal.negative_news}- {signal.neutral_news}="
        )
        print(f"  Key phrases: {', '.join(signal.key_phrases[:3])}")

    engine.store_sentiment_data(signals)
    print("✅ Sentiment data stored successfully")


if __name__ == "__main__":
    asyncio.run(main())
