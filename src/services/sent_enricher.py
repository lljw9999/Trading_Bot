#!/usr/bin/env python3
"""
GPT-4o Sentiment Enrichment Microservice

Async FastAPI service that:
1. Listens to raw news documents from Redis
2. Enriches them with GPT-4o sentiment analysis
3. Pushes enriched documents back to Redis

Enriched format: {sentiment_score: -1.0 to 1.0, rationale: "explanation"}
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
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
SENTIMENT_DOCS_TOTAL = Counter('soft_docs_total', 'Total sentiment documents processed', ['status'])
SENTIMENT_PROCESSING_TIME = Histogram('sentiment_processing_seconds', 'Time spent processing sentiment')
OPENAI_API_CALLS = Counter('openai_api_calls_total', 'Total OpenAI API calls', ['status'])

@dataclass
class EnrichedDocument:
    """Enriched news document with sentiment analysis."""
    timestamp: str
    symbol: str
    text: str
    source: str
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    rationale: str
    confidence: float  # 0.0 to 1.0
    processing_time: float
    url: Optional[str] = None
    author: Optional[str] = None

class DocumentRequest(BaseModel):
    """API request model for single document enrichment."""
    timestamp: str
    symbol: str
    text: str
    source: str
    url: Optional[str] = None
    author: Optional[str] = None

class SentimentEnricher:
    """Core sentiment enrichment logic using GPT-4o."""
    
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.redis_client = None
        
        # Sentiment analysis prompt template
        self.sentiment_prompt = """
Analyze the sentiment of this financial news text and provide a numerical sentiment score.

Text: "{text}"
Symbol: {symbol}

Provide your analysis in this exact JSON format:
{{
    "sentiment_score": <number between -1.0 and 1.0>,
    "rationale": "<brief explanation in 20 words or less>",
    "confidence": <number between 0.0 and 1.0>
}}

Guidelines:
- sentiment_score: -1.0 = very negative, 0.0 = neutral, 1.0 = very positive
- Focus on impact for the specific symbol
- confidence: how certain you are about this assessment
- rationale: concise explanation of key sentiment drivers

Return only the JSON, no other text.
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
    
    async def analyze_sentiment(self, text: str, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment using GPT-4o."""
        start_time = datetime.now()
        
        try:
            # Prepare prompt
            prompt = self.sentiment_prompt.format(text=text[:1000], symbol=symbol)
            
            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",  # Use gpt-4 or gpt-4-turbo
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent results
                timeout=30.0
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            try:
                # Extract JSON from response
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0].strip()
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0].strip()
                
                result = json.loads(content)
                
                # Validate response format
                required_fields = ['sentiment_score', 'rationale', 'confidence']
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in GPT response")
                
                # Clamp values to valid ranges
                result['sentiment_score'] = max(-1.0, min(1.0, float(result['sentiment_score'])))
                result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
                
                OPENAI_API_CALLS.labels(status='success').inc()
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'sentiment_score': result['sentiment_score'],
                    'rationale': result['rationale'][:100],  # Truncate if needed
                    'confidence': result['confidence'],
                    'processing_time': processing_time
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse GPT response: {content}, error: {e}")
                OPENAI_API_CALLS.labels(status='parse_error').inc()
                
                # Fallback to neutral sentiment
                return {
                    'sentiment_score': 0.0,
                    'rationale': 'Analysis failed - neutral default',
                    'confidence': 0.1,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            OPENAI_API_CALLS.labels(status='api_error').inc()
            
            # Fallback to neutral sentiment
            return {
                'sentiment_score': 0.0,
                'rationale': f'API error: {str(e)[:50]}',
                'confidence': 0.0,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def enrich_document(self, doc_data: Dict[str, Any]) -> EnrichedDocument:
        """Enrich a single document with sentiment analysis."""
        with SENTIMENT_PROCESSING_TIME.time():
            try:
                # Analyze sentiment
                sentiment_result = await self.analyze_sentiment(
                    doc_data['text'], 
                    doc_data['symbol']
                )
                
                # Create enriched document
                enriched = EnrichedDocument(
                    timestamp=doc_data['timestamp'],
                    symbol=doc_data['symbol'],
                    text=doc_data['text'],
                    source=doc_data['source'],
                    sentiment_score=sentiment_result['sentiment_score'],
                    rationale=sentiment_result['rationale'],
                    confidence=sentiment_result['confidence'],
                    processing_time=sentiment_result['processing_time'],
                    url=doc_data.get('url'),
                    author=doc_data.get('author')
                )
                
                SENTIMENT_DOCS_TOTAL.labels(status='success').inc()
                return enriched
                
            except Exception as e:
                logger.error(f"Document enrichment failed: {e}")
                SENTIMENT_DOCS_TOTAL.labels(status='error').inc()
                raise

# FastAPI app
app = FastAPI(
    title="Sentiment Enrichment Service",
    description="GPT-4o powered sentiment analysis for financial news",
    version="1.0.0"
)

enricher = SentimentEnricher()

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    await enricher.init_redis()
    logger.info("Sentiment enricher service started")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        await enricher.redis_client.ping()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {e}")

@app.post("/enrich")
async def enrich_document_endpoint(request: DocumentRequest) -> Dict[str, Any]:
    """Enrich a single document with sentiment analysis."""
    try:
        doc_data = request.dict()
        enriched = await enricher.enrich_document(doc_data)
        return asdict(enriched)
    except Exception as e:
        logger.error(f"Enrichment endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    try:
        redis_info = await enricher.redis_client.info()
        queue_length = await enricher.redis_client.llen("soft.raw.news")
        
        return {
            "redis_connected": True,
            "raw_queue_length": queue_length,
            "redis_memory_used": redis_info.get('used_memory_human', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Background worker for processing Redis queue
async def redis_worker():
    """Background worker that processes documents from Redis queue."""
    logger.info("Starting Redis worker")
    
    while True:
        try:
            # Block and pop from raw news queue
            result = await enricher.redis_client.blpop("soft.raw.news", timeout=5)
            
            if result:
                _, doc_json = result
                doc_data = json.loads(doc_json)
                
                # Enrich document
                enriched = await enricher.enrich_document(doc_data)
                
                # Push to enriched queue
                enriched_json = json.dumps(asdict(enriched))
                await enricher.redis_client.lpush("soft.enriched", enriched_json)
                
                logger.info(f"Enriched document for {enriched.symbol}: score={enriched.sentiment_score:.2f}")
            
        except Exception as e:
            logger.error(f"Redis worker error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def start_worker():
    """Start the background Redis worker."""
    asyncio.create_task(redis_worker())

if __name__ == "__main__":
    uvicorn.run(
        "sent_enricher:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False
    ) 