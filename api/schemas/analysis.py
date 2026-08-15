
from pydantic import BaseModel, Field
from typing import List, Optional


class LegalTextRequest(BaseModel):
    
    text: str = Field(..., min_length=10, description="Raw legal text to be analyzed.")


class LSIPrediction(BaseModel):
    
    statute: str
    confidence: float
    matched: bool


class RRPrediction(BaseModel):
    
    sentence: str
    rhetorical_role: str
    confidence: float


class CJPEPrediction(BaseModel):
    
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
    
    status: str = "success"
    lsi_predictions: Optional[List[LSIPrediction]] = None
    rr_predictions: Optional[List[RRPrediction]] = None
    cjpe_prediction: Optional[CJPEPrediction] = None
