"""
FastAPI backend for sentiment analysis API.
Author: Mahtab (Project Lead)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import json
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import from Shiv's work (will be available after he completes)
try:
    from models.inference import load_predictor, SentimentPredictor
    MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Model inference module not available yet - waiting for Shiv to complete his work")
    MODELS_AVAILABLE = False

# API data models (these define what data the API expects)
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: str
    confidence: float
    probability_distribution: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    sentiment_summary: Dict[str, int]

# Initialize FastAPI app
app = FastAPI(
    title="CNN Sentiment Analysis API",
    description="API for social media sentiment analysis using CNN",
    version="1.0.0"
)

# Allow all origins (for development - change this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (will store our trained model)
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    global predictor
    if MODELS_AVAILABLE:
        try:
            predictor = load_predictor("sentiment_cnn")
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load model: {e}")
            print("This is normal if no model has been trained yet")
    else:
        print("⚠️ Model modules not available - waiting for Shiv to complete his work")

@app.get("/")
async def root():
    """Home page of the API."""
    return {
        "message": "CNN Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the API and model are working."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "models_available": MODELS_AVAILABLE
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: TextRequest):
    """Predict sentiment for a single text."""
    if not MODELS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Model modules not available yet - waiting for Shiv to complete his work"
        )
    
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded - please train a model first"
        )
    
    try:
        result = predictor.predict_single(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTextRequest):
    """Predict sentiment for multiple texts."""
    if not MODELS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Model modules not available yet - waiting for Shiv to complete his work"
        )
    
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded - please train a model first"
        )
    
    if len(request.texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts allowed per request")
    
    try:
        results = predictor.predict_batch(request.texts)
        predictions = [PredictionResponse(**result) for result in results]
        
        # Calculate sentiment summary
        sentiment_counts = {}
        for pred in predictions:
            sentiment = pred.predicted_sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            sentiment_summary=sentiment_counts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Test endpoint to check if API is working."""
    return {
        "message": "API is working!",
        "test_prediction": {
            "text": "This is a test",
            "predicted_sentiment": "neutral",
            "confidence": 0.85,
            "note": "This is a dummy response until the model is trained"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting sentiment analysis API server...")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🏠 API home page will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
