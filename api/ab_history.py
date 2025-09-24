#!/usr/bin/env python3
"""
A/B History API Endpoint
FastAPI endpoint for serving A/B testing history data
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import time

from scripts.ab_history_store import ABHistoryStore

# Create FastAPI app
app = FastAPI(title="A/B History API", version="1.0.0")

# Initialize A/B history store
ab_store = ABHistoryStore()


class ABDecision(BaseModel):
    """A/B decision model."""

    id: int
    timestamp: float
    datetime: str
    feature: str
    decision: str
    evaluator: str
    reason: str
    previous_state: bool
    new_state: bool
    metrics: dict
    created_at: str


class FeatureStats(BaseModel):
    """Feature statistics model."""

    feature: str
    total_decisions: int
    promotions: int
    rollbacks: int
    last_decision: Optional[float]
    promotion_rate: float


class DailySummary(BaseModel):
    """Daily summary model."""

    date: str
    total_decisions: int
    promotions: int
    rollbacks: int
    features_affected: List[str]
    updated_at: str


@app.get("/api/ab-history", response_model=List[ABDecision])
async def get_ab_history(
    limit: int = Query(
        200, ge=1, le=1000, description="Maximum number of records to return"
    ),
    feature: Optional[str] = Query(None, description="Filter by specific feature"),
    since: Optional[float] = Query(
        None, description="Return decisions since this timestamp"
    ),
):
    """Get A/B testing decision history."""
    try:
        history = ab_store.get_ab_history(limit=limit, feature=feature, since=since)
        return history
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving A/B history: {str(e)}"
        )


@app.get("/api/ab-history/stats")
async def get_ab_stats(
    feature: Optional[str] = Query(None, description="Get stats for specific feature")
):
    """Get A/B testing statistics."""
    try:
        stats = ab_store.get_feature_stats(feature=feature)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving A/B stats: {str(e)}"
        )


@app.get("/api/ab-history/summaries", response_model=List[DailySummary])
async def get_daily_summaries(
    limit: int = Query(30, ge=1, le=90, description="Number of days to return")
):
    """Get daily A/B testing summaries."""
    try:
        summaries = ab_store.get_daily_summaries(limit=limit)
        return summaries
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving daily summaries: {str(e)}"
        )


@app.get("/api/ab-history/features")
async def get_features():
    """Get list of all features that have A/B decisions."""
    try:
        stats = ab_store.get_feature_stats()
        features = [fs["feature"] for fs in stats.get("feature_stats", [])]
        return {"features": features}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving features: {str(e)}"
        )


@app.get("/api/ab-history/recent")
async def get_recent_decisions(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back")
):
    """Get recent A/B decisions within specified hours."""
    try:
        since_timestamp = time.time() - (hours * 3600)
        history = ab_store.get_ab_history(limit=1000, since=since_timestamp)

        return {"hours": hours, "decisions": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving recent decisions: {str(e)}"
        )


@app.get("/api/ab-history/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        recent = ab_store.get_ab_history(limit=1)

        return {
            "status": "healthy",
            "service": "ab_history_api",
            "database": "connected",
            "records_available": len(recent) > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "A/B History API",
        "version": "1.0.0",
        "endpoints": {
            "history": "/api/ab-history",
            "stats": "/api/ab-history/stats",
            "summaries": "/api/ab-history/summaries",
            "features": "/api/ab-history/features",
            "recent": "/api/ab-history/recent",
            "health": "/api/ab-history/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
