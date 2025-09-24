#!/usr/bin/env python3
"""
Sentiment Data Fetcher for Trading System ETL Pipeline

Fetches news and social sentiment data from multiple sources:
- RSS feeds (financial news)
- Reddit (social sentiment)
- Finnhub (company news)

Data is structured as: {timestamp, symbol, text, source}
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import feedparser
import praw
import redis
from dataclasses import dataclass, asdict
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class NewsDocument:
    """Structured news document for sentiment pipeline."""

    timestamp: str
    symbol: str
    text: str
    source: str
    url: Optional[str] = None
    author: Optional[str] = None


class SentimentFetcher:
    """Multi-source sentiment data fetcher."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.logger = logging.getLogger(__name__)

        # API credentials from environment
        self.finnhub_token = os.getenv("FINNHUB_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv(
            "REDDIT_USER_AGENT", "trading-sentiment-bot/1.0"
        )

        # Target symbols for sentiment analysis
        self.target_symbols = [
            "BTC",
            "ETH",
            "SOL",
            "NVDA",
            "TSLA",
            "AAPL",
            "MSFT",
            "GOOGL",
        ]

        # RSS feeds for financial news
        self.rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]

    def fetch_rss_news(self, hours_back: int = 1) -> List[NewsDocument]:
        """Fetch news from RSS feeds."""
        documents = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"Fetching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # Parse publish time
                    try:
                        pub_time = datetime(*entry.published_parsed[:6])
                        if pub_time < cutoff_time:
                            continue
                    except:
                        pub_time = datetime.now()

                    # Extract relevant content
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}".strip()

                    # Check if any target symbols mentioned
                    text_upper = text.upper()
                    for symbol in self.target_symbols:
                        if (
                            symbol in text_upper
                            or symbol.replace("BTC", "BITCOIN").replace(
                                "ETH", "ETHEREUM"
                            )
                            in text_upper
                        ):
                            doc = NewsDocument(
                                timestamp=pub_time.isoformat(),
                                symbol=symbol,
                                text=text,
                                source=f"RSS-{feed_url.split('/')[2]}",
                                url=entry.get("link"),
                                author=entry.get("author"),
                            )
                            documents.append(doc)
                            break  # Avoid duplicates for multi-symbol mentions

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.error(f"RSS fetch error for {feed_url}: {e}")

        self.logger.info(f"Fetched {len(documents)} RSS documents")
        return documents

    def fetch_reddit_sentiment(self, hours_back: int = 1) -> List[NewsDocument]:
        """Fetch sentiment from Reddit financial subreddits."""
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            self.logger.warning("Reddit credentials not configured")
            return []

        documents = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent,
            )

            # Target subreddits for financial sentiment
            subreddits = [
                "wallstreetbets",
                "cryptocurrency",
                "investing",
                "stocks",
                "SecurityAnalysis",
            ]

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)

                    # Get hot posts from last hour
                    for submission in subreddit.hot(limit=50):
                        created_time = datetime.fromtimestamp(submission.created_utc)
                        if created_time < cutoff_time:
                            continue

                        # Check for symbol mentions
                        full_text = f"{submission.title} {submission.selftext}".upper()
                        for symbol in self.target_symbols:
                            symbol_variants = [symbol]
                            if symbol == "BTC":
                                symbol_variants.extend(["BITCOIN", "$BTC"])
                            elif symbol == "ETH":
                                symbol_variants.extend(["ETHEREUM", "$ETH"])

                            if any(variant in full_text for variant in symbol_variants):
                                doc = NewsDocument(
                                    timestamp=created_time.isoformat(),
                                    symbol=symbol,
                                    text=f"{submission.title}. {submission.selftext}"[
                                        :500
                                    ],  # Truncate
                                    source=f"Reddit-{sub_name}",
                                    url=f"https://reddit.com{submission.permalink}",
                                    author=str(submission.author),
                                )
                                documents.append(doc)
                                break

                    time.sleep(1)  # Reddit rate limiting

                except Exception as e:
                    self.logger.error(f"Reddit subreddit error {sub_name}: {e}")

        except Exception as e:
            self.logger.error(f"Reddit API error: {e}")

        self.logger.info(f"Fetched {len(documents)} Reddit documents")
        return documents

    def fetch_finnhub_news(self, hours_back: int = 1) -> List[NewsDocument]:
        """Fetch company news from Finnhub API."""
        if not self.finnhub_token:
            self.logger.warning("Finnhub API key not configured")
            return []

        documents = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # Map crypto symbols to traditional stock symbols for Finnhub
        finnhub_symbols = {
            "BTC": "MSTR",  # MicroStrategy (Bitcoin proxy)
            "ETH": "COIN",  # Coinbase
            "NVDA": "NVDA",
            "TSLA": "TSLA",
            "AAPL": "AAPL",
            "MSFT": "MSFT",
            "GOOGL": "GOOGL",
        }

        for target_symbol, finnhub_symbol in finnhub_symbols.items():
            try:
                url = "https://finnhub.io/api/v1/company-news"
                params = {
                    "symbol": finnhub_symbol,
                    "from": start_time.strftime("%Y-%m-%d"),
                    "to": end_time.strftime("%Y-%m-%d"),
                    "token": self.finnhub_token,
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                news_data = response.json()

                for article in news_data[:10]:  # Limit to 10 most recent
                    article_time = datetime.fromtimestamp(article["datetime"])
                    if article_time < start_time:
                        continue

                    doc = NewsDocument(
                        timestamp=article_time.isoformat(),
                        symbol=target_symbol,
                        text=f"{article['headline']}. {article.get('summary', '')}",
                        source="Finnhub",
                        url=article.get("url"),
                        author=article.get("source"),
                    )
                    documents.append(doc)

                time.sleep(0.2)  # Finnhub rate limiting

            except Exception as e:
                self.logger.error(f"Finnhub fetch error for {target_symbol}: {e}")

        self.logger.info(f"Fetched {len(documents)} Finnhub documents")
        return documents

    def push_to_redis(
        self, documents: List[NewsDocument], channel: str = "soft.raw.news"
    ) -> int:
        """Push documents to Redis for processing."""
        pushed_count = 0

        for doc in documents:
            try:
                doc_json = json.dumps(asdict(doc))
                self.redis_client.lpush(channel, doc_json)
                pushed_count += 1
            except Exception as e:
                self.logger.error(f"Redis push error: {e}")

        self.logger.info(f"Pushed {pushed_count} documents to Redis channel: {channel}")
        return pushed_count

    def fetch_all_sources(self, hours_back: int = 1) -> Dict[str, int]:
        """Fetch from all sources and push to Redis."""
        results = {}

        # Fetch from all sources
        rss_docs = self.fetch_rss_news(hours_back)
        reddit_docs = self.fetch_reddit_sentiment(hours_back)
        finnhub_docs = self.fetch_finnhub_news(hours_back)

        # Combine all documents
        all_docs = rss_docs + reddit_docs + finnhub_docs

        # Push to Redis
        total_pushed = self.push_to_redis(all_docs)

        results = {
            "rss_count": len(rss_docs),
            "reddit_count": len(reddit_docs),
            "finnhub_count": len(finnhub_docs),
            "total_pushed": total_pushed,
        }

        self.logger.info(f"Sentiment fetch complete: {results}")
        return results


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch sentiment data")
    parser.add_argument("--hours", type=int, default=1, help="Hours of data to fetch")
    parser.add_argument(
        "--source", choices=["rss", "reddit", "finnhub", "all"], default="all"
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    fetcher = SentimentFetcher()

    if args.source == "all":
        results = fetcher.fetch_all_sources(args.hours)
        print(f"✅ Sentiment fetch complete: {results}")
    else:
        if args.source == "rss":
            docs = fetcher.fetch_rss_news(args.hours)
        elif args.source == "reddit":
            docs = fetcher.fetch_reddit_sentiment(args.hours)
        elif args.source == "finnhub":
            docs = fetcher.fetch_finnhub_news(args.hours)

        pushed = fetcher.push_to_redis(docs)
        print(f"✅ {args.source} fetch: {len(docs)} docs, {pushed} pushed to Redis")


if __name__ == "__main__":
    main()
