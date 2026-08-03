"""
Pydantic request/response schemas for the /analyze/* endpoints.
Separated from chat.py for clean modularity.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class LegalTextRequest(BaseModel):
    """Input for all analysis endpoints - raw legal text to be analyzed."""
    text: str = Field(..., min_length=10, description="Raw legal text to be analyzed.")


class LSIPrediction(BaseModel):
    """A single Legal Statute Identification prediction."""
    statute: str
    confidence: float
    matched: bool


class RRPrediction(BaseModel):
    """A single Rhetorical Role prediction for one sentence."""
    sentence: str
    rhetorical_role: str
    confidence: float


class CJPEPrediction(BaseModel):
    """Court Judgment Prediction and Explanation result."""
    outcome: str
    confidence: float


class LSIResponse(BaseModel):
    status: str = "success"
    predictions: List[LSIPrediction]


class RRResponse(BaseModel):
    status: str = "success"
    predictions: List[RRPrediction]


class CJPEResponse(BaseModel):
    status: str = "success"
    prediction: CJPEPrediction


class FullAnalysisResponse(BaseModel):
    """Combined response from all three InLegalBERT analysis models."""
    status: str = "success"
    lsi_predictions: Optional[List[LSIPrediction]] = None
    rr_predictions: Optional[List[RRPrediction]] = None
    cjpe_prediction: Optional[CJPEPrediction] = None
