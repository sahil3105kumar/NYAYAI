"""
POST /analyze/lsi   — Legal Statute Identification
POST /analyze/rr    — Rhetorical Role classification
POST /analyze/cjpe  — Court Judgment Prediction
POST /analyze/full  — All three concurrently via asyncio.gather

All endpoints expect { "text": "..." } and use preloaded PyTorch models
from app.state.ml_models (loaded during the FastAPI lifespan startup).
"""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Request

from api.schemas.analysis import (
    LegalTextRequest,
    LSIResponse,
    LSIPrediction,
    RRResponse,
    RRPrediction,
    CJPEResponse,
    CJPEPrediction,
    FullAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


def _get_models(request: Request):
    """Retrieve preloaded ML models from app state."""
    models = getattr(request.app.state, "ml_models", None)
    if models is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded yet. Server may still be starting."
        )
    return models


@router.post("/lsi", response_model=LSIResponse)
async def analyze_lsi(req: LegalTextRequest, request: Request):
    """Identify applicable BNS/IPC legal statutes from text."""
    models = _get_models(request)
    try:
        from api.services.ml_service import NyayAI_Models

        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None,
            partial(NyayAI_Models.predict_lsi, models["lsi"], req.text)
        )
        return LSIResponse(predictions=[LSIPrediction(**p) for p in predictions])
    except Exception as e:
        logger.exception("LSI prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rr", response_model=RRResponse)
async def analyze_rr(req: LegalTextRequest, request: Request):
    """Classify rhetorical roles of each sentence in the text."""
    models = _get_models(request)
    try:
        from api.services.ml_service import NyayAI_Models

        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None,
            partial(NyayAI_Models.predict_rr, models["rr"], req.text)
        )
        return RRResponse(predictions=[RRPrediction(**p) for p in predictions])
    except Exception as e:
        logger.exception("RR prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cjpe", response_model=CJPEResponse)
async def analyze_cjpe(req: LegalTextRequest, request: Request):
    """Predict court judgment outcome (Accepted/Rejected)."""
    models = _get_models(request)
    try:
        from api.services.ml_service import NyayAI_Models

        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None,
            partial(NyayAI_Models.predict_cjpe, models["cjpe"], req.text)
        )
        return CJPEResponse(prediction=CJPEPrediction(**prediction))
    except Exception as e:
        logger.exception("CJPE prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full", response_model=FullAnalysisResponse)
async def analyze_full(req: LegalTextRequest, request: Request):
    """Run all three InLegalBERT analyses concurrently."""
    models = _get_models(request)
    try:
        from api.services.ml_service import NyayAI_Models

        loop = asyncio.get_event_loop()

        lsi_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_lsi, models["lsi"], req.text)
        )
        rr_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_rr, models["rr"], req.text)
        )
        cjpe_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_cjpe, models["cjpe"], req.text)
        )

        lsi_raw, rr_raw, cjpe_raw = await asyncio.gather(lsi_task, rr_task, cjpe_task)

        return FullAnalysisResponse(
            lsi_predictions=[LSIPrediction(**p) for p in lsi_raw],
            rr_predictions=[RRPrediction(**p) for p in rr_raw],
            cjpe_prediction=CJPEPrediction(**cjpe_raw),
        )
    except Exception as e:
        logger.exception("Full analysis error")
        raise HTTPException(status_code=500, detail=str(e))
