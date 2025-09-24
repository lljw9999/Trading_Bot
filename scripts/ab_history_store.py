#!/usr/bin/env python3
"""
A/B History Store
Keep a tamper-proof log of why/when features were promoted; surface in UI
"""

import os
import sys
import json
import time
import sqlite3
import logging
import pathlib
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("ab_history_store")


class ABHistoryStore:
    """A/B testing history storage and retrieval service."""

    def __init__(self, db_path: str = "data/ab_history.db"):
        """Initialize A/B history store."""
        self.db_path = Path(db_path)

        # Create data directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.init_database()

        # Redis connection for pubsub
        self.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)

        logger.info(f"📚 A/B History Store initialized")
        logger.info(f"   Database: {self.db_path}")

    def init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create main A/B decisions table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ab_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        feature TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        metrics TEXT,
                        evaluator TEXT,
                        reason TEXT,
                        previous_state INTEGER DEFAULT 0,
                        new_state INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create index for efficient queries
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ab_timestamp 
                    ON ab_decisions(timestamp DESC)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ab_feature 
                    ON ab_decisions(feature, timestamp DESC)
                """
                )

                # Create metrics summary table for quick stats
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ab_metrics_summary (
                        date TEXT PRIMARY KEY,
                        total_decisions INTEGER DEFAULT 0,
                        promotions INTEGER DEFAULT 0,
                        rollbacks INTEGER DEFAULT 0,
                        features_affected TEXT,
                        summary_json TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()

            logger.info("✅ Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def store_ab_decision(self, decision_data: dict) -> bool:
        """Store an A/B decision in the database."""
        try:
            # Extract fields from decision data
            timestamp = decision_data.get("timestamp", time.time())
            feature = decision_data.get("feature", "unknown")
            decision = decision_data.get("decision", "unknown")
            metrics_json = json.dumps(decision_data.get("metrics", {}))
            evaluator = decision_data.get("evaluator", "unknown")
            reason = decision_data.get("reason", "")
            previous_state = decision_data.get("previous_state", 0)
            new_state = decision_data.get("new_state", 0)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ab_decisions 
                    (timestamp, feature, decision, metrics, evaluator, reason, previous_state, new_state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        feature,
                        decision,
                        metrics_json,
                        evaluator,
                        reason,
                        previous_state,
                        new_state,
                    ),
                )

                conn.commit()

            # Update daily summary
            self.update_daily_summary(timestamp, feature, decision)

            logger.info(f"📝 Stored A/B decision: {feature} -> {decision}")
            return True

        except Exception as e:
            logger.error(f"Error storing A/B decision: {e}")
            return False

    def update_daily_summary(self, timestamp: float, feature: str, decision: str):
        """Update daily metrics summary."""
        try:
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

            with sqlite3.connect(self.db_path) as conn:
                # Get existing summary
                cursor = conn.execute(
                    "SELECT total_decisions, promotions, rollbacks, features_affected FROM ab_metrics_summary WHERE date = ?",
                    (date_str,),
                )

                row = cursor.fetchone()

                if row:
                    # Update existing summary
                    total_decisions, promotions, rollbacks, features_affected_str = row

                    # Parse features
                    features_affected = (
                        set(features_affected_str.split(","))
                        if features_affected_str
                        else set()
                    )
                    features_affected.add(feature)

                    # Update counts
                    total_decisions += 1
                    if decision in ["promoted", "enabled"]:
                        promotions += 1
                    elif decision in ["rollback", "disabled"]:
                        rollbacks += 1

                    conn.execute(
                        """
                        UPDATE ab_metrics_summary 
                        SET total_decisions = ?, promotions = ?, rollbacks = ?, 
                            features_affected = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE date = ?
                    """,
                        (
                            total_decisions,
                            promotions,
                            rollbacks,
                            ",".join(features_affected),
                            date_str,
                        ),
                    )

                else:
                    # Create new summary
                    total_decisions = 1
                    promotions = 1 if decision in ["promoted", "enabled"] else 0
                    rollbacks = 1 if decision in ["rollback", "disabled"] else 0
                    features_affected = feature

                    conn.execute(
                        """
                        INSERT INTO ab_metrics_summary 
                        (date, total_decisions, promotions, rollbacks, features_affected)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            date_str,
                            total_decisions,
                            promotions,
                            rollbacks,
                            features_affected,
                        ),
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating daily summary: {e}")

    def get_ab_history(
        self, limit: int = 200, feature: str = None, since: float = None
    ) -> list:
        """Get A/B decision history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Return rows as dictionaries

                # Build query
                query = """
                    SELECT id, timestamp, feature, decision, metrics, evaluator, 
                           reason, previous_state, new_state, created_at
                    FROM ab_decisions
                """
                params = []

                conditions = []
                if feature:
                    conditions.append("feature = ?")
                    params.append(feature)

                if since:
                    conditions.append("timestamp >= ?")
                    params.append(since)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                history = []
                for row in rows:
                    decision_record = {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "datetime": datetime.fromtimestamp(
                            row["timestamp"]
                        ).isoformat(),
                        "feature": row["feature"],
                        "decision": row["decision"],
                        "evaluator": row["evaluator"],
                        "reason": row["reason"],
                        "previous_state": bool(row["previous_state"]),
                        "new_state": bool(row["new_state"]),
                        "created_at": row["created_at"],
                    }

                    # Parse metrics JSON
                    try:
                        decision_record["metrics"] = (
                            json.loads(row["metrics"]) if row["metrics"] else {}
                        )
                    except json.JSONDecodeError:
                        decision_record["metrics"] = {}

                    history.append(decision_record)

                return history

        except Exception as e:
            logger.error(f"Error getting A/B history: {e}")
            return []

    def get_daily_summaries(self, limit: int = 30) -> list:
        """Get daily A/B metrics summaries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    """
                    SELECT date, total_decisions, promotions, rollbacks, 
                           features_affected, updated_at
                    FROM ab_metrics_summary
                    ORDER BY date DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                summaries = []
                for row in cursor.fetchall():
                    summary = {
                        "date": row["date"],
                        "total_decisions": row["total_decisions"],
                        "promotions": row["promotions"],
                        "rollbacks": row["rollbacks"],
                        "features_affected": (
                            row["features_affected"].split(",")
                            if row["features_affected"]
                            else []
                        ),
                        "updated_at": row["updated_at"],
                    }
                    summaries.append(summary)

                return summaries

        except Exception as e:
            logger.error(f"Error getting daily summaries: {e}")
            return []

    def get_feature_stats(self, feature: str = None) -> dict:
        """Get statistics for a specific feature or all features."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if feature:
                    # Stats for specific feature
                    cursor = conn.execute(
                        """
                        SELECT 
                            COUNT(*) as total_decisions,
                            SUM(CASE WHEN decision IN ('promoted', 'enabled') THEN 1 ELSE 0 END) as promotions,
                            SUM(CASE WHEN decision IN ('rollback', 'disabled') THEN 1 ELSE 0 END) as rollbacks,
                            MIN(timestamp) as first_decision,
                            MAX(timestamp) as last_decision
                        FROM ab_decisions
                        WHERE feature = ?
                    """,
                        (feature,),
                    )

                    row = cursor.fetchone()

                    return {
                        "feature": feature,
                        "total_decisions": row[0] or 0,
                        "promotions": row[1] or 0,
                        "rollbacks": row[2] or 0,
                        "first_decision": row[3],
                        "last_decision": row[4],
                        "promotion_rate": (row[1] or 0) / max(row[0] or 1, 1),
                    }

                else:
                    # Stats for all features
                    cursor = conn.execute(
                        """
                        SELECT 
                            feature,
                            COUNT(*) as total_decisions,
                            SUM(CASE WHEN decision IN ('promoted', 'enabled') THEN 1 ELSE 0 END) as promotions,
                            SUM(CASE WHEN decision IN ('rollback', 'disabled') THEN 1 ELSE 0 END) as rollbacks,
                            MAX(timestamp) as last_decision
                        FROM ab_decisions
                        GROUP BY feature
                        ORDER BY last_decision DESC
                    """
                    )

                    stats = []
                    for row in cursor.fetchall():
                        feature_stats = {
                            "feature": row[0],
                            "total_decisions": row[1],
                            "promotions": row[2],
                            "rollbacks": row[3],
                            "last_decision": row[4],
                            "promotion_rate": row[2] / max(row[1], 1),
                        }
                        stats.append(feature_stats)

                    return {"feature_stats": stats}

        except Exception as e:
            logger.error(f"Error getting feature stats: {e}")
            return {}

    def listen_for_ab_events(self):
        """Listen for A/B events from Redis pubsub."""
        logger.info("📡 Starting A/B events listener...")

        try:
            # Subscribe to A/B gate events
            pubsub = self.redis.pubsub()
            pubsub.subscribe("ab_gate:events")

            logger.info("✅ Subscribed to ab_gate:events channel")

            for message in pubsub.listen():
                try:
                    if message["type"] != "message":
                        continue

                    # Parse event data
                    event_data = json.loads(message["data"])

                    # Store the A/B decision
                    self.store_ab_decision(event_data)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in A/B event: {e}")
                except Exception as e:
                    logger.error(f"Error processing A/B event: {e}")

        except KeyboardInterrupt:
            logger.info("🛑 A/B events listener stopped by user")
        except Exception as e:
            logger.error(f"Error in A/B events listener: {e}")
            raise
        finally:
            try:
                pubsub.close()
            except:
                pass

    def cleanup_old_records(self, days_to_keep: int = 90):
        """Clean up old records to prevent database from growing too large."""
        try:
            cutoff_timestamp = time.time() - (days_to_keep * 24 * 3600)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM ab_decisions WHERE timestamp < ?", (cutoff_timestamp,)
                )

                deleted_count = cursor.rowcount
                conn.commit()

                if deleted_count > 0:
                    logger.info(f"🗑️ Cleaned up {deleted_count} old A/B records")

                return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0


def create_test_data(store: ABHistoryStore):
    """Create some test A/B decision data."""
    logger.info("🧪 Creating test A/B decision data...")

    import random

    features = ["EXEC_RL_LIVE", "BANDIT_WEIGHTS", "LLM_SENTIMENT", "TAIL_RISK_HEDGE"]
    decisions = ["promoted", "rollback", "enabled", "disabled"]
    evaluators = ["ab_eval_gate", "manual_override", "scheduled_check"]

    base_time = time.time() - (7 * 24 * 3600)  # Start a week ago

    for i in range(50):
        decision_data = {
            "timestamp": base_time + (i * 3600),  # One per hour
            "feature": random.choice(features),
            "decision": random.choice(decisions),
            "evaluator": random.choice(evaluators),
            "reason": f"Test decision #{i + 1}",
            "previous_state": random.randint(0, 1),
            "new_state": random.randint(0, 1),
            "metrics": {
                "sharpe_improvement": random.uniform(-0.1, 0.2),
                "slippage_improvement": random.uniform(-5, 10),
                "correlation": random.uniform(-0.1, 0.1),
            },
        }

        store.store_ab_decision(decision_data)

    logger.info("✅ Created test A/B decision data")


def main():
    """Main entry point for A/B history store."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B History Store")
    parser.add_argument(
        "--db-path", default="data/ab_history.db", help="Database file path"
    )
    parser.add_argument("--listen", action="store_true", help="Listen for A/B events")
    parser.add_argument(
        "--history", action="store_true", help="Show recent A/B history"
    )
    parser.add_argument("--stats", action="store_true", help="Show feature statistics")
    parser.add_argument("--feature", help="Filter by specific feature")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of results")
    parser.add_argument(
        "--cleanup", type=int, metavar="DAYS", help="Clean up records older than N days"
    )
    parser.add_argument(
        "--create-test-data", action="store_true", help="Create test data"
    )

    args = parser.parse_args()

    # Create store
    store = ABHistoryStore(args.db_path)

    if args.create_test_data:
        create_test_data(store)
        return

    if args.cleanup:
        deleted = store.cleanup_old_records(args.cleanup)
        print(f"Deleted {deleted} old records")
        return

    if args.history:
        # Show recent history
        history = store.get_ab_history(limit=args.limit, feature=args.feature)
        print(json.dumps(history, indent=2, default=str))
        return

    if args.stats:
        # Show statistics
        stats = store.get_feature_stats(args.feature)
        print(json.dumps(stats, indent=2, default=str))
        return

    if args.listen:
        # Listen for events
        store.listen_for_ab_events()
    else:
        print(
            "Use --listen to start event listener, --history to show history, or --stats for statistics"
        )


if __name__ == "__main__":
    main()
