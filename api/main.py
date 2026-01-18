"""FastAPI endpoints for security event anomaly scoring."""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Initialize FastAPI app
app = FastAPI(
    title="Security Event Triage API",
    description="Anomaly detection and alert triage for security events using LightGBM, XGBoost, and CatBoost",
    version="1.0.0",
)

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"

# Global model instances
models = {}
feature_columns = []


# ==================== Pydantic Models ====================

class SecurityEvent(BaseModel):
    """Single security event for scoring."""
    src_port: Optional[int] = 0
    dst_port: Optional[int] = 0
    bytes_in: Optional[int] = 0
    bytes_out: Optional[int] = 0
    duration: Optional[float] = 0.0
    protocol: Optional[str] = None
    status: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "dst_port": 22,
                "bytes_in": 100,
                "bytes_out": 5000000,
                "duration": 0.001,
                "protocol": "ssh",
                "status": "failed",
            }
        }


class ScoreResponse(BaseModel):
    """Response with anomaly scores from all models."""
    anomaly_score: float = Field(..., ge=0, le=1, description="Combined anomaly score")
    classification: str = Field(..., description="ATTACK or NORMAL")
    severity: str = Field(..., description="critical, high, medium, low, info")
    model_scores: Dict[str, float] = Field(..., description="Individual model scores")
    features_used: Dict[str, float] = Field(..., description="Features fed to models")


class BatchScoreRequest(BaseModel):
    """Request to score multiple events."""
    events: List[SecurityEvent]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    feature_count: int
    version: str


class MetricsResponse(BaseModel):
    """Model metrics from training."""
    roc_auc: float
    pr_auc: float
    f1: float
    train_time: float


# ==================== Startup ====================

@app.on_event("startup")
async def startup():
    """Load models on startup."""
    global models, feature_columns
    
    # Load feature columns
    feature_cols_path = MODEL_DIR / "feature_columns.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_columns = json.load(f)
        print(f"Loaded {len(feature_columns)} feature columns")
    else:
        print("Warning: feature_columns.json not found")
    
    # Load LightGBM
    lgb_path = MODEL_DIR / "lightgbm.txt"
    if lgb_path.exists():
        import lightgbm as lgb
        models["lightgbm"] = lgb.Booster(model_file=str(lgb_path))
        print("Loaded LightGBM model")
    
    # Load XGBoost
    xgb_path = MODEL_DIR / "xgboost.json"
    if xgb_path.exists():
        import xgboost as xgb
        models["xgboost"] = xgb.Booster()
        models["xgboost"].load_model(str(xgb_path))
        print("Loaded XGBoost model")
    
    # Load CatBoost
    cat_path = MODEL_DIR / "catboost.cbm"
    if cat_path.exists():
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier()
        models["catboost"].load_model(str(cat_path))
        print("Loaded CatBoost model")
    
    # Load Isolation Forest (unsupervised)
    if_path = MODEL_DIR / "isolation"
    if if_path.exists():
        try:
            from src.models.isolation import IsolationForestScorer
            models["isolation_forest"] = IsolationForestScorer.load(if_path)
            print("Loaded Isolation Forest model")
        except Exception as e:
            print(f"Could not load Isolation Forest: {e}")
    
    print(f"Loaded {len(models)} models")


# ==================== Helper Functions ====================

def event_to_features(event: SecurityEvent) -> Dict[str, float]:
    """Convert SecurityEvent to feature dictionary."""
    features = {col: 0.0 for col in feature_columns}
    
    # Numeric features
    features["src_port"] = float(event.src_port or 0)
    features["dst_port"] = float(event.dst_port or 0)
    features["bytes_in"] = float(event.bytes_in or 0)
    features["bytes_out"] = float(event.bytes_out or 0)
    features["duration"] = float(event.duration or 0.0)
    
    # Protocol one-hot encoding
    if event.protocol:
        proto_col = f"proto_{event.protocol.lower()}"
        if proto_col in features:
            features[proto_col] = 1.0
    
    # Status
    if event.status == "failed":
        features["status_failed"] = 1.0
    
    return features


def get_severity(score: float) -> str:
    """Convert score to severity level."""
    if score >= 0.9:
        return "critical"
    elif score >= 0.7:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.3:
        return "low"
    else:
        return "info"


def predict_all(features_df: pd.DataFrame) -> Dict[str, float]:
    """Get predictions from all loaded models."""
    scores = {}
    
    if "lightgbm" in models:
        scores["lightgbm"] = float(models["lightgbm"].predict(features_df)[0])
    
    if "xgboost" in models:
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features_df)
        scores["xgboost"] = float(models["xgboost"].predict(dmatrix)[0])
    
    if "catboost" in models:
        scores["catboost"] = float(models["catboost"].predict_proba(features_df)[0, 1])
    
    if "isolation_forest" in models:
        try:
            if_scores = models["isolation_forest"].score(features_df)
            scores["isolation_forest"] = float(if_scores["combined_score"].iloc[0])
        except Exception:
            pass  # IF needs more data context
    
    # Placeholder for next line
        scores["catboost"] = float(models["catboost"].predict_proba(features_df)[0, 1])
    
    return scores


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "lightgbm": "lightgbm" in models,
            "xgboost": "xgboost" in models,
            "catboost": "catboost" in models,
            "isolation_forest": "isolation_forest" in models,
        },
        feature_count=len(feature_columns),
        version="1.0.0",
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def score_event(event: SecurityEvent):
    """Score a single security event.
    
    Returns anomaly scores from LightGBM, XGBoost, and CatBoost models.
    The combined score is the average of all models.
    """
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Convert event to features
    features = event_to_features(event)
    features_df = pd.DataFrame([features])[feature_columns]
    
    # Get predictions
    model_scores = predict_all(features_df)
    
    # Combine scores (average)
    combined_score = sum(model_scores.values()) / len(model_scores)
    
    return ScoreResponse(
        anomaly_score=round(combined_score, 4),
        classification="ATTACK" if combined_score >= 0.5 else "NORMAL",
        severity=get_severity(combined_score),
        model_scores={k: round(v, 4) for k, v in model_scores.items()},
        features_used={k: v for k, v in features.items() if v != 0},
    )


@app.post("/v1/score/batch", response_model=List[ScoreResponse])
async def score_batch(request: BatchScoreRequest):
    """Score multiple security events in a batch.
    
    More efficient than calling /v1/score for each event.
    """
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    results = []
    for event in request.events:
        features = event_to_features(event)
        features_df = pd.DataFrame([features])[feature_columns]
        model_scores = predict_all(features_df)
        combined_score = sum(model_scores.values()) / len(model_scores)
        
        results.append(ScoreResponse(
            anomaly_score=round(combined_score, 4),
            classification="ATTACK" if combined_score >= 0.5 else "NORMAL",
            severity=get_severity(combined_score),
            model_scores={k: round(v, 4) for k, v in model_scores.items()},
            features_used={k: v for k, v in features.items() if v != 0},
        ))
    
    return results


@app.get("/v1/models")
async def list_models():
    """List loaded models and their metadata."""
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return {
            "models": list(models.keys()),
            "feature_columns": feature_columns,
            "training_results": metadata.get("results", {}),
            "data_stats": metadata.get("data_stats", {}),
        }
    
    return {
        "models": list(models.keys()),
        "feature_columns": feature_columns,
    }


@app.get("/v1/metrics")
async def get_metrics():
    """Get model performance metrics from training."""
    metrics_path = MODEL_DIR.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    
    # Try model_metadata.json
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("results", {})
    
    return {"error": "No metrics available"}


@app.get("/docs/examples")
async def get_examples():
    """Get example requests for each endpoint."""
    return {
        "score_normal_event": {
            "endpoint": "POST /v1/score",
            "request": {
                "dst_port": 443,
                "bytes_in": 1500,
                "bytes_out": 500,
                "duration": 0.5,
                "protocol": "https",
            },
            "expected": "Low anomaly score (normal HTTPS traffic)"
        },
        "score_brute_force": {
            "endpoint": "POST /v1/score",
            "request": {
                "dst_port": 22,
                "bytes_in": 100,
                "bytes_out": 50,
                "duration": 0.01,
                "protocol": "ssh",
                "status": "failed",
            },
            "expected": "High anomaly score (failed SSH = brute force)"
        },
        "score_data_exfil": {
            "endpoint": "POST /v1/score",
            "request": {
                "dst_port": 443,
                "bytes_in": 100,
                "bytes_out": 5000000,
                "duration": 0.001,
                "protocol": "https",
            },
            "expected": "High anomaly score (massive outbound data)"
        },
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
