#!/usr/bin/env python3
"""
CryptoPanic News API Client

Implementation following Future_instruction.txt specifications for news integration.
Provides aggregated crypto news with sentiment tags from >2000 sources.
"""

import os
import requests
import redis
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
import hashlib

logger = logging.getLogger(__name__)


class CryptoPanicClient:
    """CryptoPanic API client for crypto news with sentiment analysis."""

    def __init__(
        self, api_key: Optional[str] = None, redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize CryptoPanic client.

        Args:
            api_key: CryptoPanic API key (defaults to env var CryptoPanic_API)
            redis_client: Redis client for caching/streaming news
        """
        self.api_key = api_key or os.getenv("CryptoPanic_API")
        if not self.api_key:
            raise ValueError(
                "CryptoPanic API key not found. Set CryptoPanic_API env var."
            )

        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.redis_client = redis_client or redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )

        # Rate limiting (CryptoPanic allows 1000 requests per hour)
        self.last_request = 0
        self.min_interval = 3.6  # seconds between requests

    def _rate_limit(self):
        """Ensure rate limiting compliance."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

    def get_news(
        self,
        currencies: str = "BTC,ETH",
        kind: str = "news",
        limit: int = 50,
        filter_type: str = "hot",
        regions: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Get latest crypto news from CryptoPanic.

        Args:
            currencies: Comma-separated currency symbols (e.g., "BTC,ETH")
            kind: Type of posts ("news", "media", or "all")
            limit: Number of posts to retrieve (max 100)
            filter_type: Filter posts by type ("hot", "trending", "latest")
            regions: Language filter ("en" for English)

        Returns:
            List of news posts with sentiment data
        """
        self._rate_limit()

        params = {
            "auth_token": self.api_key,
            "currencies": currencies,
            "kind": kind,
            "public": "true",
            "filter": filter_type,
            "regions": regions,
            "page_size": limit,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            posts = data.get("results", [])

            # Clean and enrich posts
            enriched_posts = []
            for post in posts:
                enriched_post = self._enrich_post(post)
                enriched_posts.append(enriched_post)

            logger.info(f"Retrieved {len(enriched_posts)} news posts from CryptoPanic")
            return enriched_posts

        except requests.RequestException as e:
            logger.error(f"Error fetching news from CryptoPanic: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_news: {e}")
            return []

    def _enrich_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich news post with additional metadata.

        Args:
            post: Raw post from CryptoPanic API

        Returns:
            Enriched post with standardized fields
        """
        # Extract sentiment information
        votes = post.get("votes", {})
        # Try vote-based sentiment first, fallback to title analysis
        sentiment = self._calculate_sentiment(votes)
        if sentiment == "neutral":
            sentiment = self._analyze_title_sentiment(post.get("title", ""))

        # Create standardized post structure
        enriched = {
            "id": post.get("id"),
            "title": post.get("title", ""),
            "url": post.get("url", ""),
            "source": self._extract_source(post),
            "published_at": post.get("published_at"),
            "created_at": post.get("created_at"),
            "currencies": [c.get("code") for c in post.get("currencies", [])],
            "kind": post.get("kind"),
            "domain": post.get("domain", ""),
            "sentiment": sentiment,
            "votes": {
                "negative": votes.get("negative", 0),
                "positive": votes.get("positive", 0),
                "important": votes.get("important", 0),
                "liked": votes.get("liked", 0),
                "disliked": votes.get("disliked", 0),
                "lol": votes.get("lol", 0),
                "toxic": votes.get("toxic", 0),
                "saved": votes.get("saved", 0),
                "comments": votes.get("comments", 0),
            },
            "impact_score": self._calculate_impact_score(votes),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "hash": self._generate_hash(post.get("url", "") + post.get("title", "")),
        }

        return enriched

    def _extract_source(self, post: Dict[str, Any]) -> str:
        """Extract clean source name from post."""
        source = post.get("source", {})
        if isinstance(source, dict):
            return source.get("title", source.get("domain", "Unknown"))
        return str(source) if source else post.get("domain", "Unknown")

    def _calculate_sentiment(self, votes: Dict[str, int]) -> str:
        """
        Calculate overall sentiment from votes using CryptoPanic scoring thresholds.

        Args:
            votes: Vote counts from CryptoPanic

        Returns:
            Sentiment label: "bullish", "bearish", or "neutral"
        """
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)
        total_votes = positive + negative + votes.get("important", 0)

        if total_votes == 0:
            return "neutral"

        # Calculate raw sentiment score (CryptoPanic range: -2 to +2)
        raw_score = (positive - negative) / max(total_votes, 1) * 2

        # Apply Future_instruction.txt thresholds: >+0.3 → bullish, <-0.3 → bearish
        if raw_score > 0.3:
            return "bullish"
        elif raw_score < -0.3:
            return "bearish"
        else:
            return "neutral"

    def _analyze_title_sentiment(self, title: str) -> str:
        """
        Analyze sentiment from news title using keyword matching.

        Args:
            title: News headline text

        Returns:
            Sentiment label: "bullish", "bearish", or "neutral"
        """
        title_lower = title.lower()

        # Bullish keywords
        bullish_keywords = [
            "surge",
            "soar",
            "rally",
            "pump",
            "moon",
            "breakout",
            "bullish",
            "gains",
            "profits",
            "record high",
            "all-time high",
            "ath",
            "institutional adoption",
            "etf approval",
            "partnerships",
            "upgrade",
            "explosion",
            "massive gains",
            "skyrocket",
            "explode",
        ]

        # Bearish keywords
        bearish_keywords = [
            "crash",
            "dump",
            "plunge",
            "collapse",
            "bearish",
            "decline",
            "losses",
            "sell-off",
            "panic",
            "fear",
            "uncertainty",
            "regulatory",
            "ban",
            "crackdown",
            "investigation",
            "hack",
            "exploit",
            "scam",
            "bubble",
            "correction",
            "downturn",
            "bear market",
        ]

        bullish_count = sum(1 for word in bullish_keywords if word in title_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in title_lower)

        if bullish_count > bearish_count and bullish_count > 0:
            return "bullish"
        elif bearish_count > bullish_count and bearish_count > 0:
            return "bearish"
        else:
            return "neutral"

    def _calculate_impact_score(self, votes: Dict[str, int]) -> float:
        """
        Calculate impact score from votes.

        Args:
            votes: Vote counts from CryptoPanic

        Returns:
            Impact score (0.0 to 1.0)
        """
        important = votes.get("important", 0)
        liked = votes.get("liked", 0)
        comments = votes.get("comments", 0)
        saved = votes.get("saved", 0)

        # Weight different types of engagement
        raw_score = important * 3.0 + liked * 1.0 + comments * 2.0 + saved * 1.5

        # Normalize to 0-1 range (max observed scores around 100-200)
        normalized = min(raw_score / 200.0, 1.0)

        return round(normalized, 3)

    def _generate_hash(self, content: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def stream_to_redis(
        self,
        currencies: str = "BTC,ETH",
        stream_key: str = "news:cryptopanic",
        max_items: int = 1000,
    ) -> int:
        """
        Stream news to Redis for downstream consumption.

        Args:
            currencies: Currencies to monitor
            stream_key: Redis key for news stream
            max_items: Max items to keep in stream

        Returns:
            Number of new items added
        """
        try:
            news_posts = self.get_news(currencies=currencies)

            if not news_posts:
                return 0

            new_items = 0

            for post in news_posts:
                # Check if already exists (dedupe by hash)
                existing = self.redis_client.hget(f"{stream_key}:seen", post["hash"])
                if existing:
                    continue

                # Add to stream (convert to strings for Redis)
                redis_post = {
                    k: str(v) if not isinstance(v, (str, int, float)) else v
                    for k, v in post.items()
                }
                self.redis_client.xadd(stream_key, redis_post)

                # Mark as seen
                self.redis_client.hset(f"{stream_key}:seen", post["hash"], "1")

                # Also store in simple key for latest
                self.redis_client.lpush(f"{stream_key}:latest", json.dumps(post))

                new_items += 1

            # Trim streams to prevent memory issues
            self.redis_client.xtrim(stream_key, maxlen=max_items)
            self.redis_client.ltrim(f"{stream_key}:latest", 0, max_items - 1)

            # Clean old seen hashes (keep last 24h worth)
            seen_count = self.redis_client.hlen(f"{stream_key}:seen")
            if seen_count > max_items * 2:
                # This is a simple cleanup - in production you'd want timestamp-based cleanup
                keys = list(self.redis_client.hscan_iter(f"{stream_key}:seen"))
                for key, _ in keys[:-max_items]:
                    self.redis_client.hdel(f"{stream_key}:seen", key)

            logger.info(f"Added {new_items} new news items to {stream_key}")
            return new_items

        except Exception as e:
            logger.error(f"Error streaming news to Redis: {e}")
            return 0

    def get_latest_from_redis(
        self, stream_key: str = "news:cryptopanic", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get latest news from Redis cache.

        Args:
            stream_key: Redis key for news stream
            limit: Number of items to retrieve

        Returns:
            List of latest news posts
        """
        try:
            items = self.redis_client.lrange(f"{stream_key}:latest", 0, limit - 1)
            return [json.loads(item) for item in items]
        except Exception as e:
            logger.error(f"Error retrieving news from Redis: {e}")
            return []

    def get_sentiment_summary(
        self, currencies: str = "BTC,ETH", hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment summary for specified currencies.

        Args:
            currencies: Currencies to analyze
            hours: Time window in hours

        Returns:
            Sentiment summary statistics
        """
        try:
            # Get recent news
            news_posts = self.get_latest_from_redis(limit=100)

            # Filter by currencies and time
            currency_list = [c.strip().upper() for c in currencies.split(",")]
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            filtered_posts = []
            for post in news_posts:
                try:
                    published_at = post.get("published_at", "")
                    if published_at:
                        # Handle various datetime formats
                        if published_at.endswith("Z"):
                            published_at = published_at.replace("Z", "+00:00")

                        post_time = datetime.fromisoformat(published_at)

                        # Convert both times to UTC for comparison
                        if post_time.tzinfo is None:
                            post_time = post_time.replace(tzinfo=timezone.utc)
                        if cutoff_time.tzinfo is None:
                            cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)

                        if post_time >= cutoff_time:
                            post_currencies = [
                                c.upper() for c in post.get("currencies", [])
                            ]
                            title_lower = post.get("title", "").lower()

                            # Check if currencies match OR if title contains currency names
                            currency_match = any(
                                c in currency_list for c in post_currencies
                            ) or any(
                                curr.lower() in title_lower
                                for curr in ["bitcoin", "btc", "ethereum", "eth"]
                            )

                            if currency_match:
                                filtered_posts.append(post)
                    else:
                        # If no published_at, include in analysis (likely recent)
                        post_currencies = [
                            c.upper() for c in post.get("currencies", [])
                        ]
                        title_lower = post.get("title", "").lower()

                        currency_match = any(
                            c in currency_list for c in post_currencies
                        ) or any(
                            curr.lower() in title_lower
                            for curr in ["bitcoin", "btc", "ethereum", "eth"]
                        )

                        if currency_match:
                            filtered_posts.append(post)

                except Exception as e:
                    logger.debug(f"Error parsing date for post: {e}")
                    # Include post anyway if we can't parse date
                    post_currencies = [c.upper() for c in post.get("currencies", [])]
                    title_lower = post.get("title", "").lower()

                    currency_match = any(
                        c in currency_list for c in post_currencies
                    ) or any(
                        curr.lower() in title_lower
                        for curr in ["bitcoin", "btc", "ethereum", "eth"]
                    )

                    if currency_match:
                        filtered_posts.append(post)

            # Calculate sentiment statistics
            total_posts = len(filtered_posts)
            if total_posts == 0:
                return {
                    "total_posts": 0,
                    "bullish_pct": 0,
                    "bearish_pct": 0,
                    "neutral_pct": 0,
                    "avg_impact": 0,
                    "top_sources": [],
                }

            sentiments = [post.get("sentiment", "neutral") for post in filtered_posts]
            bullish_count = sentiments.count("bullish")
            bearish_count = sentiments.count("bearish")
            neutral_count = sentiments.count("neutral")

            avg_impact = (
                sum(post.get("impact_score", 0) for post in filtered_posts)
                / total_posts
            )

            # Top sources
            sources = {}
            for post in filtered_posts:
                source = post.get("source", "Unknown")
                sources[source] = sources.get(source, 0) + 1
            top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "total_posts": total_posts,
                "bullish_pct": round(bullish_count / total_posts * 100, 1),
                "bearish_pct": round(bearish_count / total_posts * 100, 1),
                "neutral_pct": round(neutral_count / total_posts * 100, 1),
                "avg_impact": round(avg_impact, 3),
                "top_sources": [
                    {"source": src, "count": cnt} for src, cnt in top_sources
                ],
                "time_window_hours": hours,
                "currencies": currencies,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating sentiment summary: {e}")
            return {
                "total_posts": 0,
                "bullish_pct": 0,
                "bearish_pct": 0,
                "neutral_pct": 0,
                "avg_impact": 0,
                "top_sources": [],
                "error": str(e),
            }


# Convenience function for easy integration
def get_crypto_news(
    currencies: str = "BTC,ETH", limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Convenience function to get crypto news.

    Args:
        currencies: Comma-separated currency symbols
        limit: Number of posts to retrieve

    Returns:
        List of news posts
    """
    try:
        client = CryptoPanicClient()
        return client.get_news(currencies=currencies, limit=limit)
    except Exception as e:
        logger.error(f"Error in get_crypto_news: {e}")
        return []


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = CryptoPanicClient()

    # Test basic news retrieval
    news = client.get_news(currencies="BTC,ETH", limit=5)
    print(f"\n📰 Retrieved {len(news)} news items:")

    for item in news[:3]:
        print(f"  • {item['title'][:60]}...")
        print(
            f"    Source: {item['source']} | Sentiment: {item['sentiment']} | Impact: {item['impact_score']}"
        )

    # Test Redis streaming
    print(f"\n🔄 Streaming to Redis...")
    new_items = client.stream_to_redis()
    print(f"Added {new_items} new items to Redis")

    # Test sentiment summary
    print(f"\n📊 Sentiment Summary:")
    summary = client.get_sentiment_summary()
    print(f"  Total posts: {summary['total_posts']}")
    print(
        f"  Bullish: {summary['bullish_pct']}% | Bearish: {summary['bearish_pct']}% | Neutral: {summary['neutral_pct']}%"
    )
    print(f"  Avg impact: {summary['avg_impact']}")
