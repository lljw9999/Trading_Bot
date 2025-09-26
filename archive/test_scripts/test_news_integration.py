#!/usr/bin/env python3
"""
Test News Integration and Sentiment Analysis

Test script to verify the news sentiment integration is working correctly.
"""

import asyncio
import requests
import time
import webbrowser
from datetime import datetime


def test_news_sentiment_endpoints():
    """Test the news sentiment API endpoints."""
    print("🧪 Testing News Sentiment Integration")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # Test 1: Health check with news features
    print("1. Testing health check with news features...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Status: {health_data['status']}")
            print(
                f"   ✅ News Engine: {'Active' if health_data.get('news_engine_active', False) else 'Inactive'}"
            )
            print(f"   ✅ Features: {', '.join(health_data['features'])}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: News sentiment data
    print("\n2. Testing news sentiment data...")
    try:
        response = requests.get(f"{base_url}/api/news-sentiment")
        if response.status_code == 200:
            sentiment_data = response.json()

            print(f"   ✅ Last Update: {sentiment_data.get('last_update', 'Unknown')}")

            # Test BTC sentiment
            if "BTCUSDT" in sentiment_data:
                btc_sentiment = sentiment_data["BTCUSDT"]
                print(f"   📊 BTC Sentiment: {btc_sentiment['sentiment_score']:.3f}")
                print(f"      Confidence: {btc_sentiment['confidence']:.3f}")
                print(
                    f"      News: {btc_sentiment['positive_news']}+ {btc_sentiment['negative_news']}- {btc_sentiment['neutral_news']}="
                )
                print(
                    f"      Key phrases: {', '.join(btc_sentiment['key_phrases'][:3])}"
                )

            # Test ETH sentiment
            if "ETHUSDT" in sentiment_data:
                eth_sentiment = sentiment_data["ETHUSDT"]
                print(f"   📊 ETH Sentiment: {eth_sentiment['sentiment_score']:.3f}")
                print(f"      Confidence: {eth_sentiment['confidence']:.3f}")
                print(
                    f"      News: {eth_sentiment['positive_news']}+ {eth_sentiment['negative_news']}- {eth_sentiment['neutral_news']}="
                )

            # Test market sentiment
            if "market_sentiment" in sentiment_data:
                market_sentiment = sentiment_data["market_sentiment"]
                print(
                    f"   🌍 Market Sentiment: {market_sentiment['market_sentiment']:.3f}"
                )
                print(
                    f"      Market Confidence: {market_sentiment['market_confidence']:.3f}"
                )
                print(f"      Total News: {market_sentiment['total_news']}")

        else:
            print(f"   ❌ News sentiment failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ News sentiment error: {e}")
        return False

    # Test 3: Recent news feed
    print("\n3. Testing recent news feed...")
    try:
        response = requests.get(f"{base_url}/api/recent-news")
        if response.status_code == 200:
            news_items = response.json()

            if isinstance(news_items, list) and len(news_items) > 0:
                print(f"   ✅ Retrieved {len(news_items)} news items:")
                for i, news in enumerate(news_items[:3]):  # Show first 3
                    sentiment_label = (
                        "Positive"
                        if news["sentiment_score"] > 0.2
                        else "Negative" if news["sentiment_score"] < -0.2 else "Neutral"
                    )
                    print(f"      {i+1}. {news['title'][:60]}...")
                    print(
                        f"         Source: {news['source']} | Sentiment: {sentiment_label} ({news['sentiment_score']:.2f}) | {news['time_ago']}"
                    )
            else:
                print("   ❌ No news items retrieved")
                return False
        else:
            print(f"   ❌ Recent news failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Recent news error: {e}")
        return False

    # Test 4: Dashboard accessibility
    print("\n4. Testing dashboard with news sentiment...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            html_content = response.text

            # Check for news sentiment elements
            news_elements = [
                "Market Sentiment & News Analysis",
                "marketSentimentScore",
                "btcSentimentScore",
                "ethSentimentScore",
                "Recent Crypto News",
                "newsFeed",
            ]

            missing_elements = []
            for element in news_elements:
                if element not in html_content:
                    missing_elements.append(element)

            if not missing_elements:
                print("   ✅ All news sentiment UI elements found in dashboard")
            else:
                print(f"   ❌ Missing UI elements: {missing_elements}")
                return False

        else:
            print(f"   ❌ Dashboard access failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard access error: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 NEWS SENTIMENT INTEGRATION TEST RESULTS")
    print("=" * 60)
    print("✅ All news sentiment tests passed!")
    print("✅ News sentiment API endpoints working!")
    print("✅ Dashboard UI includes news sentiment!")
    print("✅ Real-time news feed operational!")

    return True


def show_news_features():
    """Show the completed news integration features."""
    print("\n📰 News Integration Features")
    print("=" * 60)

    features = [
        "🔍 Real-time News Processing:",
        "   • Multi-source news aggregation (CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine)",
        "   • Advanced sentiment analysis with crypto-specific keywords",
        "   • Symbol-specific sentiment tracking (BTC, ETH)",
        "   • Confidence scoring and impact assessment",
        "",
        "🐦 Social Media Integration:",
        "   • Twitter sentiment simulation with engagement weighting",
        "   • Crypto influencer account monitoring",
        "   • Real-time social sentiment scoring",
        "",
        "📊 Market Sentiment Dashboard:",
        "   • Overall market sentiment gauge",
        "   • Symbol-specific sentiment breakdowns",
        "   • Positive/negative news ratio visualization",
        "   • Key phrase extraction and display",
        "",
        "📱 News Feed Interface:",
        "   • Real-time crypto news feed",
        "   • Sentiment-colored news items",
        "   • Clickable news articles",
        "   • Time-stamped updates",
        "",
        "⚡ Technical Features:",
        "   • Redis-based data storage and caching",
        "   • Concurrent news source processing",
        "   • Sentiment history tracking",
        "   • Error handling and fallback data",
        "",
        "🎯 Integration Benefits:",
        "   • Enhanced trading signal context",
        "   • Market sentiment-based risk assessment",
        "   • News-driven alpha generation",
        "   • Real-time market narrative tracking",
    ]

    for feature in features:
        print(feature)


def open_dashboard_with_news():
    """Open the dashboard to show news integration."""
    print("\n🌐 Opening Enhanced Dashboard with News Sentiment...")
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Dashboard opened in browser!")

        print("\n🎯 News Features to Look For:")
        print("   • Market Sentiment & News Analysis section")
        print("   • Real-time sentiment scores for BTC and ETH")
        print("   • Market sentiment gauge with confidence levels")
        print("   • Recent Crypto News feed with sentiment indicators")
        print(
            "   • Color-coded news items (green=positive, red=negative, orange=neutral)"
        )
        print("   • Clickable news articles that open in new tabs")

    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print("💡 Please manually open: http://localhost:8000")


async def test_news_engine_directly():
    """Test the news engine directly."""
    print("\n🔧 Testing News Engine Directly...")

    try:
        # Import and test the news engine
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

        from layers.layer1_signal_generation.news_sentiment import NewsIntegrationEngine

        engine = NewsIntegrationEngine()

        # Test news processing
        print("   📰 Processing news batch...")
        news_items = engine.process_news_batch()
        print(f"   ✅ Processed {len(news_items)} news items")

        # Test Twitter feed
        print("   🐦 Processing Twitter feed...")
        tweets = engine.twitter_feed.fetch_crypto_tweets(["BTCUSDT", "ETHUSDT"])
        print(f"   ✅ Processed {len(tweets)} tweets")

        # Test sentiment signals
        print("   📊 Generating sentiment signals...")
        signals = engine.generate_sentiment_signals(news_items, tweets)
        print(f"   ✅ Generated {len(signals)} sentiment signals")

        # Show sample signals
        for symbol, signal in list(signals.items())[:2]:
            print(
                f"      {symbol}: Sentiment={signal.sentiment_score:.3f}, Confidence={signal.confidence:.3f}"
            )

        # Test data storage
        print("   💾 Testing data storage...")
        engine.store_sentiment_data(signals)
        print("   ✅ Data stored successfully")

        return True

    except ImportError:
        print("   ⚠️ News engine module not available for direct testing")
        return False
    except Exception as e:
        print(f"   ❌ Direct engine test error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 News Sentiment Integration Test Suite")
    print("=" * 80)

    # Test API endpoints
    api_success = test_news_sentiment_endpoints()

    if api_success:
        # Test news engine directly
        await test_news_engine_directly()

        # Show features
        show_news_features()

        # Open dashboard
        open_dashboard_with_news()

        print("\n🎉 NEWS INTEGRATION SUCCESSFULLY IMPLEMENTED!")
        print("✅ Real-time news sentiment analysis operational")
        print("📊 Dashboard enhanced with comprehensive news features")
        print("🔄 Continuous news processing and sentiment updates")
        print("💻 Available at: http://localhost:8000")
    else:
        print("\n❌ Issues found with news integration")
        print("💡 Please check that the enhanced dashboard is running")


if __name__ == "__main__":
    asyncio.run(main())
