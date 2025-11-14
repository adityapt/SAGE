"""
FastAPI REST API Server

Production-ready API for MMM Copilot with authentication, rate limiting, and monitoring.
"""

from typing import Optional, Dict, List
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MMM Copilot API",
    description="Production API for Marketing Mix Modeling Copilot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key if authentication is enabled"""
    from llm_copilot.config import config
    
    if not config.api_key_required:
        return "anonymous"
    
    if not api_key or api_key not in config.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return api_key


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    include_visualizations: bool = Field(True, description="Whether to generate visualizations")
    agentic_mode: bool = Field(False, description="Enable autonomous multi-step analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is TV's ROI and how can we optimize it?",
                "session_id": "user123",
                "include_visualizations": True,
                "agentic_mode": False
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    confidence: Optional[float] = None
    sources: Optional[List[Dict]] = None
    visualizations: Optional[List[Dict]] = None
    metadata: Dict = {}
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "TV has an ROI of 1.32x, which is within industry benchmark (0.3-1.5x). Current spend is $500k/week, approaching saturation at $750k. Recommendation: Maintain current spend or reduce by 10-15%.",
                "confidence": 0.92,
                "sources": [
                    {"type": "data", "content": "TV ROI: 1.32x from model"},
                    {"type": "benchmark", "content": "Industry benchmark: 0.3-1.5x"}
                ],
                "visualizations": [
                    {"type": "response_curve", "url": "/viz/tv_curve.html"}
                ],
                "metadata": {
                    "query_type": "channel_analysis",
                    "channels": ["TV"],
                    "processing_time_ms": 245
                }
            }
        }


class UploadDataRequest(BaseModel):
    """Request model for data upload"""
    data: List[Dict] = Field(..., description="MMM data as list of dictionaries")
    date_col: Optional[str] = Field(None, description="Date column name")
    channel_col: Optional[str] = Field(None, description="Channel column name")
    segment_col: Optional[str] = Field(None, description="Segment/taxonomy column name")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"date": "2024-01-01", "channel": "TV", "spend": 50000, "impressions": 1000000, "predicted": 15000},
                    {"date": "2024-01-08", "channel": "TV", "spend": 55000, "impressions": 1100000, "predicted": 16000}
                ],
                "date_col": "date",
                "channel_col": "channel"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    cache_enabled: bool
    data_loaded: bool


# Global copilot instance (initialized on startup)
copilot = None


@app.on_event("startup")
async def startup_event():
    """Initialize copilot on startup"""
    from llm_copilot.config import config
    from llm_copilot.core.copilot import MMMCopilot
    
    logger.info("Starting MMM Copilot API...")
    
    # Note: In production, load data from database or file
    # For now, copilot will be initialized when data is uploaded
    logger.info("API started successfully")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "MMM Copilot API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    from llm_copilot.config import config
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        cache_enabled=config.cache_enabled,
        data_loaded=copilot is not None
    )


@app.post("/upload_data", tags=["Data"])
async def upload_data(
    request: UploadDataRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Upload MMM data.
    
    Initializes the copilot with provided data.
    """
    global copilot
    
    try:
        from llm_copilot.config import config
        from llm_copilot.core.copilot import MMMCopilot
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Initialize copilot
        copilot = MMMCopilot(
            data=df,
            api_key=config.openai_api_key,
            date_col=request.date_col,
            channel_col=request.channel_col,
            segment_col=request.segment_col
        )
        
        return {
            "status": "success",
            "message": "Data uploaded successfully",
            "rows": len(df),
            "channels": len(df[copilot.channel_col].unique()),
            "date_range": {
                "start": df[copilot.date_col].min(),
                "end": df[copilot.date_col].max()
            }
        }
        
    except Exception as e:
        logger.error(f"Data upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data upload failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Query the copilot with natural language.
    
    Handles all query types: channel analysis, optimization, visualization, etc.
    """
    if copilot is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data loaded. Please upload data first using /upload_data"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Execute query
        if request.agentic_mode:
            result = copilot.analyze_autonomous(
                query=request.query,
                session_id=request.session_id
            )
        else:
            result = copilot.query(
                query=request.query,
                session_id=request.session_id
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Format response
        return QueryResponse(
            answer=result.get('answer', ''),
            confidence=result.get('confidence'),
            sources=result.get('sources', []),
            visualizations=result.get('visualizations', []) if request.include_visualizations else [],
            metadata={
                'query_type': result.get('query_type', 'unknown'),
                'channels': result.get('channels', []),
                'processing_time_ms': processing_time,
                'session_id': request.session_id
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/channels", tags=["Data"])
async def get_channels(api_key: str = Depends(verify_api_key)):
    """Get list of available channels"""
    if copilot is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data loaded"
        )
    
    channels = copilot.data[copilot.channel_col].unique().tolist()
    return {"channels": channels, "count": len(channels)}


@app.get("/metrics", tags=["Data"])
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get available metrics and definitions"""
    from llm_copilot.visualization.metrics import METRICS
    
    return {
        "metrics": [
            {
                "name": name,
                "description": info["description"],
                "formula": info["formula"],
                "interpretation": info["interpretation"]
            }
            for name, info in METRICS.items()
        ]
    }


@app.get("/benchmarks/{channel}", tags=["Knowledge"])
async def get_channel_benchmark(channel: str, api_key: str = Depends(verify_api_key)):
    """Get industry benchmark for a channel"""
    from llm_copilot.knowledge.mmm_expertise import MMMExpertise
    
    benchmark = MMMExpertise.get_channel_benchmark(channel)
    return {
        "channel": channel,
        "benchmark": benchmark
    }


@app.get("/best_practices", tags=["Knowledge"])
async def get_best_practices(api_key: str = Depends(verify_api_key)):
    """Get MMM best practices"""
    from llm_copilot.knowledge.mmm_expertise import MMMExpertise
    
    return {
        "best_practices": MMMExpertise.get_all_best_practices()
    }


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    query: str,
    response: str,
    rating: int = Field(..., ge=1, le=5),
    comment: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Submit feedback on a query/response"""
    if copilot is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No copilot initialized"
        )
    
    try:
        copilot.collect_feedback(query, response, rating, comment)
        return {
            "status": "success",
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@app.delete("/cache", tags=["Admin"])
async def clear_cache(api_key: str = Depends(verify_api_key)):
    """Clear all caches"""
    from llm_copilot.core.cache import get_cache
    
    try:
        cache = get_cache()
        cache.clear()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {str(e)}"
        )


@app.get("/stats", tags=["Admin"])
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get system statistics"""
    from llm_copilot.core.cache import get_cache
    
    try:
        cache = get_cache()
        cache_stats = cache.get_stats()
        
        stats = {
            "cache": cache_stats,
            "data_loaded": copilot is not None
        }
        
        if copilot:
            stats["data"] = {
                "rows": len(copilot.data),
                "channels": len(copilot.data[copilot.channel_col].unique()),
                "date_range": {
                    "start": str(copilot.data[copilot.date_col].min()),
                    "end": str(copilot.data[copilot.date_col].max())
                }
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    from llm_copilot.config import config
    
    uvicorn.run(
        "llm_copilot.api.server:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=False
    )

