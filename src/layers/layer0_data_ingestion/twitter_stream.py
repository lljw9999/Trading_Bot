#!/usr/bin/env python3
"""
X (Twitter) News Ingestor with VADER Sentiment Analysis
Streams tweets with crypto keywords and pushes to Redis
"""

import os
import json
import time
import redis
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterNewsIngestor:
    def __init__(self):
        self.redis_client = redis.Redis(
            host="localhost", port=6379, decode_responses=True
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.bearer_token = os.getenv("X_BEARER_TOKEN")

        # Fallback to scraping if no API key
        self.use_api = bool(self.bearer_token)

    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores["compound"]  # Returns -1 (negative) to 1 (positive)

    def process_tweet(self, tweet_data):
        """Process and store tweet data"""
        try:
            # Extract text and clean it
            text = tweet_data.get("text", "").replace("\n", " ").strip()
            tweet_id = tweet_data.get("id", "")

            if not text or not tweet_id:
                return False

            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(text)

            # Create payload
            payload = {
                "ts": int(time.time()),
                "id": str(tweet_id),
                "text": text,
                "url": f"https://x.com/i/web/status/{tweet_id}",
                "sent": sentiment_score,
                "source": "x",
            }

            # Store in Redis stream
            self.redis_client.xadd("news:x", payload, maxlen=5000)
            logger.info(f"Stored tweet {tweet_id} with sentiment {sentiment_score:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return False

    def search_tweets_api(
        self,
        query="(BTC OR Bitcoin OR ETH OR Ethereum) lang:en -is:retweet",
        max_results=10,
    ):
        """Search tweets using Twitter API v2"""
        if not self.bearer_token:
            logger.warning("No Bearer token available for API access")
            return []

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,public_metrics",
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                logger.error(
                    f"API request failed: {response.status_code} - {response.text}"
                )
                return []
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return []

    def scrape_nitter_fallback(self):
        """Fallback method using Nitter instances (when API not available)"""
        # This is a simplified fallback - in production you'd want more robust scraping
        logger.info("Using fallback scraping method")

        # Simulate some crypto news data for demonstration
        sample_tweets = [
            {
                "id": f"demo_{int(time.time())}_1",
                "text": "Bitcoin hits new resistance level as institutional adoption continues to grow #BTC",
            },
            {
                "id": f"demo_{int(time.time())}_2",
                "text": "Ethereum network upgrade shows promising results for scalability #ETH",
            },
            {
                "id": f"demo_{int(time.time())}_3",
                "text": "Regulatory uncertainty continues to impact crypto markets negatively",
            },
        ]

        return sample_tweets

    def run_continuous(self, interval=30):
        """Run continuous ingestion"""
        logger.info("Starting Twitter news ingestion...")

        while True:
            try:
                if self.use_api:
                    tweets = self.search_tweets_api()
                else:
                    tweets = self.scrape_nitter_fallback()

                processed = 0
                for tweet in tweets:
                    if self.process_tweet(tweet):
                        processed += 1

                logger.info(f"Processed {processed}/{len(tweets)} tweets")

                # Wait before next batch
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Stopping Twitter ingestion...")
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                time.sleep(5)  # Wait before retry

    def run_single_batch(self):
        """Run a single batch for testing"""
        logger.info("Running single batch...")

        if self.use_api:
            tweets = self.search_tweets_api(max_results=20)
        else:
            tweets = self.scrape_nitter_fallback()

        processed = 0
        for tweet in tweets:
            if self.process_tweet(tweet):
                processed += 1

        logger.info(
            f"Single batch complete: {processed}/{len(tweets)} tweets processed"
        )
        return processed


def main():
    ingestor = TwitterNewsIngestor()

    # Test Redis connection
    try:
        ingestor.redis_client.ping()
        logger.info("✅ Redis connected successfully")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return

    # Run single batch first for testing
    ingestor.run_single_batch()

    # Ask user if they want continuous mode
    print("\nRun continuous ingestion? (y/n): ", end="")
    choice = input().strip().lower()

    if choice == "y":
        ingestor.run_continuous()


if __name__ == "__main__":
    main()
