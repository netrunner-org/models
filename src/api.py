from fastapi import APIRouter, HTTPException, Depends, Request, Header
from typing import List, Dict, Optional, Any
import time
from collections import defaultdict
import threading
import logging

from src.models import (
    TextClassificationRequest, 
    TextClassificationResponse, 
    HealthResponse
)
from src.services import model_service
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.records = defaultdict(list)
        self.lock = threading.Lock()
        
    def is_rate_limited(self, client_id: str):
        with self.lock:
            now = time.time()
            
            # Clean old records
            self.records[client_id] = [t for t in self.records[client_id] if now - t < self.period]
            
            # Check limit
            if len(self.records[client_id]) >= self.calls:
                return True
                
            # Record request
            self.records[client_id].append(now)
            return False

rate_limiter = RateLimiter(
    calls=settings.RATE_LIMIT_CALLS,
    period=settings.RATE_LIMIT_PERIOD
)

# API key authentication
async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Rate limiting
async def check_rate_limit(request: Request):
    client_id = request.client.host
    logger.info(f"Checking Rate Limit: {client_id}")
    if rate_limiter.is_rate_limited(client_id):
        logger.info(f"Refusing Request: Rate Limited")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# Common dependencies
api_dependencies = [
    Depends(verify_api_key),
    Depends(check_rate_limit)
]

# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="ok",
        loaded_models=model_service.list_loaded_models()
    )


@router.post(
    "/models/text/classification", 
    response_model=TextClassificationResponse,
    dependencies=api_dependencies,
    tags=["models"]
)
async def get_probabilities(
    request: TextClassificationRequest,
):
    """Get class probabilities for the input text"""
    try:
        probabilities, labels = model_service.get_class_probabilities(
            model_id=request.model,
            text=request.text,
            temperature=request.temperature
        )
        return TextClassificationResponse(
            probabilities=probabilities,
            labels=labels,
            model_id=request.model
        )
    except Exception as e:
        logger.error(f"Error getting probabilities: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))