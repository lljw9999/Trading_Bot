#!/usr/bin/env python3
"""
Explain Middleware Service

Generates plain-English explanations for trading decisions using GPT-4o.
Subscribes to order events and produces readable trade rationales.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import redis.asyncio as redis
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
EXPLANATIONS_GENERATED = Counter('explanations_generated_total', 'Total explanations generated', ['status'])
EXPLANATION_LATENCY = Histogram('explanation_latency_seconds', 'Time to generate explanation')
OPENAI_EXPLAIN_CALLS = Counter('openai_explain_calls_total', 'OpenAI API calls for explanations', ['status'])

@dataclass
class OrderEvent:
    """Order event structure."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: str
    order_type: str  # 'market', 'limit', etc.
    
    # Trading context
    edge_bps: Optional[float] = None
    confidence: Optional[float] = None
    portfolio_value: Optional[float] = None
    position_size_pct: Optional[float] = None
    
    # Alpha model contributions
    sentiment_score: Optional[float] = None
    technical_signal: Optional[str] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    big_bet_flag: Optional[bool] = None

@dataclass
class TradeExplanation:
    """Generated trade explanation."""
    order_id: str
    symbol: str
    explanation: str
    confidence_level: str
    key_factors: List[str]
    risk_assessment: str
    timestamp: str
    processing_time_ms: float

class ExplainService:
    """GPT-4o powered trade explanation service."""
    
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.redis_client = None
        
        # Explanation prompt template
        self.explanation_prompt = """
You are a professional trading analyst explaining a trading decision to a portfolio manager.

Order Details:
- Symbol: {symbol}
- Action: {side} {quantity:.4f} shares/units at ${price:.2f}
- Order Type: {order_type}
- Position Size: {position_size_pct:.1f}% of portfolio
- Timestamp: {timestamp}

Trading Signals:
- Expected Edge: {edge_bps:.1f} basis points
- Model Confidence: {confidence:.1%}
- Sentiment Score: {sentiment_score:.2f} (range: -1.0 to 1.0)
- Technical Signal: {technical_signal}
- Big Bet Flag: {big_bet_flag}

Risk Context:
- Portfolio Value: ${portfolio_value:,.0f}
- Risk Metrics: {risk_metrics}

Please provide a concise explanation (≤50 words) covering:
1. Primary reason for the trade
2. Key supporting factors
3. Risk level assessment

Format as a single paragraph suitable for a trading dashboard.
"""
    
    async def init_redis(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    
    async def generate_explanation(self, order_event: OrderEvent) -> TradeExplanation:
        """Generate explanation for a trade order."""
        start_time = datetime.now()
        
        try:
            # Prepare prompt with order context
            prompt = self.explanation_prompt.format(
                symbol=order_event.symbol,
                side=order_event.side.upper(),
                quantity=order_event.quantity,
                price=order_event.price,
                order_type=order_event.order_type,
                position_size_pct=(order_event.position_size_pct or 0) * 100,
                timestamp=order_event.timestamp,
                edge_bps=order_event.edge_bps or 0,
                confidence=order_event.confidence or 0,
                sentiment_score=order_event.sentiment_score or 0,
                technical_signal=order_event.technical_signal or "None",
                big_bet_flag="Yes" if order_event.big_bet_flag else "No",
                portfolio_value=order_event.portfolio_value or 0,
                risk_metrics=json.dumps(order_event.risk_metrics or {})
            )
            
            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3,  # Consistent but natural explanations
                timeout=15.0
            )
            
            # Extract explanation
            explanation_text = response.choices[0].message.content.strip()
            
            # Analyze key factors from the order
            key_factors = self._extract_key_factors(order_event)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(order_event.confidence)
            
            # Risk assessment
            risk_assessment = self._assess_risk_level(order_event)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            explanation = TradeExplanation(
                order_id=order_event.order_id,
                symbol=order_event.symbol,
                explanation=explanation_text,
                confidence_level=confidence_level,
                key_factors=key_factors,
                risk_assessment=risk_assessment,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
            
            OPENAI_EXPLAIN_CALLS.labels(status='success').inc()
            EXPLANATIONS_GENERATED.labels(status='success').inc()
            
            logger.info(f"Generated explanation for order {order_event.order_id}: {explanation_text[:50]}...")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            OPENAI_EXPLAIN_CALLS.labels(status='error').inc()
            EXPLANATIONS_GENERATED.labels(status='error').inc()
            
            # Fallback explanation
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TradeExplanation(
                order_id=order_event.order_id,
                symbol=order_event.symbol,
                explanation=f"Trade based on {order_event.edge_bps or 0:.1f}bp edge with {(order_event.confidence or 0)*100:.0f}% confidence. Analysis error: {str(e)[:30]}",
                confidence_level=self._determine_confidence_level(order_event.confidence),
                key_factors=["Analysis Error"],
                risk_assessment="Unknown",
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
    
    def _extract_key_factors(self, order_event: OrderEvent) -> List[str]:
        """Extract key factors driving the trade decision."""
        factors = []
        
        # Edge magnitude
        if order_event.edge_bps and abs(order_event.edge_bps) > 20:
            factors.append(f"Strong edge ({order_event.edge_bps:.0f}bp)")
        elif order_event.edge_bps and abs(order_event.edge_bps) > 5:
            factors.append(f"Moderate edge ({order_event.edge_bps:.0f}bp)")
        
        # Sentiment
        if order_event.sentiment_score:
            if abs(order_event.sentiment_score) > 0.7:
                sentiment_desc = "Positive" if order_event.sentiment_score > 0 else "Negative"
                factors.append(f"Strong {sentiment_desc.lower()} sentiment")
            elif abs(order_event.sentiment_score) > 0.3:
                sentiment_desc = "Positive" if order_event.sentiment_score > 0 else "Negative"
                factors.append(f"Moderate {sentiment_desc.lower()} sentiment")
        
        # Technical signals
        if order_event.technical_signal and order_event.technical_signal != "None":
            factors.append(f"Technical: {order_event.technical_signal}")
        
        # Big bet flag
        if order_event.big_bet_flag:
            factors.append("High-confidence big bet")
        
        # Position size
        if order_event.position_size_pct and order_event.position_size_pct > 0.15:  # >15%
            factors.append("Large position")
        
        return factors[:3]  # Limit to top 3 factors
    
    def _determine_confidence_level(self, confidence: Optional[float]) -> str:
        """Convert numeric confidence to descriptive level."""
        if not confidence:
            return "Low"
        
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_risk_level(self, order_event: OrderEvent) -> str:
        """Assess risk level of the trade."""
        risk_score = 0
        
        # Position size risk
        if order_event.position_size_pct:
            if order_event.position_size_pct > 0.20:  # >20%
                risk_score += 3
            elif order_event.position_size_pct > 0.10:  # >10%
                risk_score += 2
            elif order_event.position_size_pct > 0.05:  # >5%
                risk_score += 1
        
        # Confidence risk (lower confidence = higher risk)
        if order_event.confidence:
            if order_event.confidence < 0.4:
                risk_score += 2
            elif order_event.confidence < 0.6:
                risk_score += 1
        
        # Big bet flag adds risk
        if order_event.big_bet_flag:
            risk_score += 1
        
        # Market order adds execution risk
        if order_event.order_type == "market":
            risk_score += 1
        
        if risk_score >= 5:
            return "High Risk"
        elif risk_score >= 3:
            return "Moderate Risk"
        elif risk_score >= 1:
            return "Low Risk"
        else:
            return "Very Low Risk"
    
    async def store_explanation(self, explanation: TradeExplanation):
        """Store explanation in Redis."""
        try:
            # Store with expiry (7 days)
            key = f"explain:{explanation.order_id}"
            value = json.dumps(asdict(explanation))
            await self.redis_client.setex(key, 604800, value)  # 7 days
            
            # Also add to recent explanations list
            await self.redis_client.lpush("trades.explain.recent", value)
            await self.redis_client.ltrim("trades.explain.recent", 0, 99)  # Keep last 100
            
            logger.debug(f"Stored explanation for order {explanation.order_id}")
            
        except Exception as e:
            logger.error(f"Error storing explanation: {e}")

# FastAPI app
app = FastAPI(
    title="Trade Explanation Service",
    description="GPT-4o powered trade explanation generator",
    version="1.0.0"
)

explain_service = ExplainService()

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    await explain_service.init_redis()
    logger.info("Explain service started")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        await explain_service.redis_client.ping()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {e}")

@app.post("/explain")
async def explain_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate explanation for a single order."""
    try:
        # Convert to OrderEvent
        order_event = OrderEvent(**order_data)
        
        # Generate explanation
        with EXPLANATION_LATENCY.time():
            explanation = await explain_service.generate_explanation(order_event)
        
        # Store in Redis
        await explain_service.store_explanation(explanation)
        
        return asdict(explanation)
        
    except Exception as e:
        logger.error(f"Explain endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explanation/{order_id}")
async def get_explanation(order_id: str) -> Dict[str, Any]:
    """Retrieve explanation for an order."""
    try:
        key = f"explain:{order_id}"
        explanation_json = await explain_service.redis_client.get(key)
        
        if not explanation_json:
            raise HTTPException(status_code=404, detail="Explanation not found")
        
        return json.loads(explanation_json)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent")
async def get_recent_explanations(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent explanations."""
    try:
        recent_list = await explain_service.redis_client.lrange("trades.explain.recent", 0, limit-1)
        return [json.loads(exp) for exp in recent_list]
        
    except Exception as e:
        logger.error(f"Error getting recent explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Background worker for processing order events
async def order_event_worker():
    """Background worker that processes order events from Redis."""
    logger.info("Starting order event worker")
    
    while True:
        try:
            # Listen for order events
            result = await explain_service.redis_client.blpop("orders.filled", timeout=5)
            
            if result:
                _, order_json = result
                order_data = json.loads(order_json)
                
                # Convert to OrderEvent
                order_event = OrderEvent(**order_data)
                
                # Generate explanation
                explanation = await explain_service.generate_explanation(order_event)
                
                # Store explanation
                await explain_service.store_explanation(explanation)
                
                logger.info(f"Processed order {order_event.order_id} for {order_event.symbol}")
            
        except Exception as e:
            logger.error(f"Order event worker error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def start_worker():
    """Start the background order event worker."""
    asyncio.create_task(order_event_worker())

if __name__ == "__main__":
    uvicorn.run(
        "explain_middleware:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False
    ) 