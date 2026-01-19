"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    data: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    fraud_probability: float
    is_fraud: bool
