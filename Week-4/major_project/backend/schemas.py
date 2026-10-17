"""
schemas.py
==========
Pydantic data models for Document Understanding REST API requests and responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TopKItem(BaseModel):
    category: str = Field(..., description="Document category name")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    percentage: str = Field(..., description="Human-readable confidence percentage")


class PredictionResponse(BaseModel):
    predicted_category: str = Field(..., description="Top predicted document category")
    confidence: float = Field(..., description="Top predicted class probability")
    confidence_percentage: str = Field(..., description="Human-readable formatted percentage")
    top_k: List[TopKItem] = Field(..., description="Ranked top-k category predictions")
    inference_time_ms: float = Field(..., description="Model forward-pass latency in milliseconds")
    filename: str = Field(..., description="Original submitted filename")


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="API health status")
    model_loaded: bool = Field(..., description="Whether the deep learning model is active")
    model_name: str = Field(..., description="Identifier of the serving model")
    supported_classes: List[str] = Field(..., description="List of recognized document categories")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Short error descriptor")
    detail: Optional[str] = Field(default=None, description="Detailed explanatory message")
