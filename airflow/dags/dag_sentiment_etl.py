#!/usr/bin/env python3
"""
Sentiment ETL DAG for Trading System

Orchestrates parallel sentiment data collection from:
- RSS feeds (financial news)
- Reddit (social sentiment) 
- Finnhub (company news)

Runs every 15 minutes and pushes structured data to Redis.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from etl.sentiment_fetcher import SentimentFetcher

# DAG configuration
DEFAULT_ARGS = {
    "owner": "trading-system",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}

# Create DAG
dag = DAG(
    "sentiment_etl",
    default_args=DEFAULT_ARGS,
    description="Multi-source sentiment data ETL pipeline",
    schedule_interval=timedelta(minutes=15),  # Every 15 minutes
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "etl", "trading"],
)


# Initialize fetcher instance
def get_fetcher() -> SentimentFetcher:
    """Get configured sentiment fetcher instance."""
    redis_host = Variable.get("REDIS_HOST", default_var="localhost")
    redis_port = int(Variable.get("REDIS_PORT", default_var="6379"))
    return SentimentFetcher(redis_host=redis_host, redis_port=redis_port)


def fetch_rss_task(**context) -> Dict[str, Any]:
    """Airflow task to fetch RSS news data."""
    logging.info("Starting RSS news fetch task")

    try:
        fetcher = get_fetcher()
        hours_back = 1  # Fetch last hour of data

        # Fetch RSS documents
        documents = fetcher.fetch_rss_news(hours_back)

        # Push to Redis
        pushed_count = fetcher.push_to_redis(documents, "soft.raw.news.rss")

        result = {
            "source": "rss",
            "documents_fetched": len(documents),
            "documents_pushed": pushed_count,
            "execution_time": context["ts"],
            "success": True,
        }

        logging.info(f"RSS task completed: {result}")
        return result

    except Exception as e:
        logging.error(f"RSS fetch task failed: {e}")
        raise


def fetch_reddit_task(**context) -> Dict[str, Any]:
    """Airflow task to fetch Reddit sentiment data."""
    logging.info("Starting Reddit sentiment fetch task")

    try:
        fetcher = get_fetcher()
        hours_back = 1

        # Fetch Reddit documents
        documents = fetcher.fetch_reddit_sentiment(hours_back)

        # Push to Redis
        pushed_count = fetcher.push_to_redis(documents, "soft.raw.news.reddit")

        result = {
            "source": "reddit",
            "documents_fetched": len(documents),
            "documents_pushed": pushed_count,
            "execution_time": context["ts"],
            "success": True,
        }

        logging.info(f"Reddit task completed: {result}")
        return result

    except Exception as e:
        logging.error(f"Reddit fetch task failed: {e}")
        raise


def fetch_finnhub_task(**context) -> Dict[str, Any]:
    """Airflow task to fetch Finnhub news data."""
    logging.info("Starting Finnhub news fetch task")

    try:
        fetcher = get_fetcher()
        hours_back = 1

        # Fetch Finnhub documents
        documents = fetcher.fetch_finnhub_news(hours_back)

        # Push to Redis
        pushed_count = fetcher.push_to_redis(documents, "soft.raw.news.finnhub")

        result = {
            "source": "finnhub",
            "documents_fetched": len(documents),
            "documents_pushed": pushed_count,
            "execution_time": context["ts"],
            "success": True,
        }

        logging.info(f"Finnhub task completed: {result}")
        return result

    except Exception as e:
        logging.error(f"Finnhub fetch task failed: {e}")
        raise


def merge_and_push_task(**context) -> Dict[str, Any]:
    """Merge all source data and push to unified Redis channel."""
    logging.info("Starting merge and push task")

    try:
        fetcher = get_fetcher()
        redis_client = fetcher.redis_client

        # Collect all documents from source-specific channels
        all_documents = []
        source_channels = [
            "soft.raw.news.rss",
            "soft.raw.news.reddit",
            "soft.raw.news.finnhub",
        ]

        total_merged = 0
        for channel in source_channels:
            # Move all documents from source channel to unified channel
            while True:
                doc_json = redis_client.rpop(channel)
                if not doc_json:
                    break
                redis_client.lpush("soft.raw.news", doc_json)
                total_merged += 1

        # Get task results from XCom for summary
        task_instance = context["task_instance"]
        rss_result = task_instance.xcom_pull(task_ids="fetch_rss")
        reddit_result = task_instance.xcom_pull(task_ids="fetch_reddit")
        finnhub_result = task_instance.xcom_pull(task_ids="fetch_finnhub")

        # Compile summary
        result = {
            "execution_time": context["ts"],
            "total_documents_merged": total_merged,
            "source_results": {
                "rss": rss_result,
                "reddit": reddit_result,
                "finnhub": finnhub_result,
            },
            "success": True,
        }

        # Set Redis key with execution summary
        summary_key = f"sentiment_etl_summary:{context['ds']}:{context['ts']}"
        redis_client.setex(summary_key, 86400, str(result))  # 24h expiry

        logging.info(f"Merge task completed: {result}")
        return result

    except Exception as e:
        logging.error(f"Merge task failed: {e}")
        raise


def cleanup_task(**context) -> Dict[str, Any]:
    """Cleanup old data and maintain Redis hygiene."""
    logging.info("Starting cleanup task")

    try:
        fetcher = get_fetcher()
        redis_client = fetcher.redis_client

        # Keep only last 1000 documents in main channel
        list_length = redis_client.llen("soft.raw.news")
        if list_length > 1000:
            # Trim to keep most recent 1000
            redis_client.ltrim("soft.raw.news", 0, 999)
            trimmed = list_length - 1000
        else:
            trimmed = 0

        # Clean up old summary keys (keep last 7 days)
        pattern = "sentiment_etl_summary:*"
        old_keys = []
        for key in redis_client.scan_iter(match=pattern):
            if redis_client.ttl(key) < 0:  # No expiry set
                old_keys.append(key)

        if old_keys:
            redis_client.delete(*old_keys)

        result = {
            "documents_trimmed": trimmed,
            "old_keys_cleaned": len(old_keys),
            "execution_time": context["ts"],
            "success": True,
        }

        logging.info(f"Cleanup task completed: {result}")
        return result

    except Exception as e:
        logging.error(f"Cleanup task failed: {e}")
        raise


# Define task dependencies
start_task = DummyOperator(
    task_id="start_sentiment_etl",
    dag=dag,
)

# Parallel fetch tasks
rss_task = PythonOperator(
    task_id="fetch_rss",
    python_callable=fetch_rss_task,
    dag=dag,
)

reddit_task = PythonOperator(
    task_id="fetch_reddit",
    python_callable=fetch_reddit_task,
    dag=dag,
)

finnhub_task = PythonOperator(
    task_id="fetch_finnhub",
    python_callable=fetch_finnhub_task,
    dag=dag,
)

# Merge and cleanup tasks
merge_task = PythonOperator(
    task_id="merge_and_push",
    python_callable=merge_and_push_task,
    dag=dag,
)

cleanup_task_op = PythonOperator(
    task_id="cleanup",
    python_callable=cleanup_task,
    dag=dag,
)

end_task = DummyOperator(
    task_id="end_sentiment_etl",
    dag=dag,
)

# Set task dependencies
start_task >> [rss_task, reddit_task, finnhub_task]
[rss_task, reddit_task, finnhub_task] >> merge_task
merge_task >> cleanup_task_op >> end_task

# Export DAG for Airflow discovery
if __name__ == "__main__":
    dag.cli()
