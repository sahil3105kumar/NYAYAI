

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


# def _get_models(request: Request):
#     """Retrieve preloaded ML models from app state."""
#     models = getattr(request.app.state, "ml_models", None)
#     if models is None:
#         raise HTTPException(
#             status_code=503,
#             detail="ML models not loaded yet. Server may still be starting."
#         )
#     return models


# @router.post("/lsi", response_model=LSIResponse)
# async def analyze_lsi(req: LegalTextRequest, request: Request):
#     """Identify applicable BNS/IPC legal statutes from text."""
#     models = _get_models(request)
#     try:
#         from api.services.ml_service import NyayAI_Models

#         loop = asyncio.get_event_loop()
#         predictions = await loop.run_in_executor(
#             None,
#             partial(NyayAI_Models.predict_lsi, models["lsi"], req.text)
#         )
#         return LSIResponse(predictions=[LSIPrediction(**p) for p in predictions])
#     except Exception as e:
#         logger.exception("LSI prediction error")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/rr", response_model=RRResponse)
# async def analyze_rr(req: LegalTextRequest, request: Request):
#     """Classify rhetorical roles of each sentence in the text."""
#     models = _get_models(request)
#     try:
#         from api.services.ml_service import NyayAI_Models

#         loop = asyncio.get_event_loop()
#         predictions = await loop.run_in_executor(
#             None,
#             partial(NyayAI_Models.predict_rr, models["rr"], req.text)
#         )
#         return RRResponse(predictions=[RRPrediction(**p) for p in predictions])
#     except Exception as e:
#         logger.exception("RR prediction error")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/cjpe", response_model=CJPEResponse)
# async def analyze_cjpe(req: LegalTextRequest, request: Request):
#     """Predict court judgment outcome (Accepted/Rejected)."""
#     models = _get_models(request)
#     try:
#         from api.services.ml_service import NyayAI_Models

#         loop = asyncio.get_event_loop()
#         prediction = await loop.run_in_executor(
#             None,
#             partial(NyayAI_Models.predict_cjpe, models["cjpe"], req.text)
#         )
#         return CJPEResponse(prediction=CJPEPrediction(**prediction))
#     except Exception as e:
#         logger.exception("CJPE prediction error")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/full", response_model=FullAnalysisResponse)
# async def analyze_full(req: LegalTextRequest, request: Request):
#     """Run all three InLegalBERT analyses concurrently."""
#     models = _get_models(request)
#     try:
#         from api.services.ml_service import NyayAI_Models

#         loop = asyncio.get_event_loop()

#         lsi_task = loop.run_in_executor(
#             None, partial(NyayAI_Models.predict_lsi, models["lsi"], req.text)
#         )
#         rr_task = loop.run_in_executor(
#             None, partial(NyayAI_Models.predict_rr, models["rr"], req.text)
#         )
#         cjpe_task = loop.run_in_executor(
#             None, partial(NyayAI_Models.predict_cjpe, models["cjpe"], req.text)
#         )

#         lsi_raw, rr_raw, cjpe_raw = await asyncio.gather(lsi_task, rr_task, cjpe_task)

#         return FullAnalysisResponse(
#             lsi_predictions=[LSIPrediction(**p) for p in lsi_raw],
#             rr_predictions=[RRPrediction(**p) for p in rr_raw],
#             cjpe_prediction=CJPEPrediction(**cjpe_raw),
#         )
#     except Exception as e:
#         logger.exception("Full analysis error")
#         raise HTTPException(status_code=500, detail=str(e))


def _load_model(name: str):
    """Blocking (weights from disk + .to(device)) — always run via executor."""
    from api.services.ml_service import NyayAI_Models
    try:
        return NyayAI_Models.get_model(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/lsi", response_model=LSIResponse)
async def analyze_lsi(req: LegalTextRequest):
    from api.services.ml_service import NyayAI_Models
    loop = asyncio.get_event_loop()
    try:
        bundle = await loop.run_in_executor(None, _load_model, "lsi")
        predictions = await loop.run_in_executor(
            None, partial(NyayAI_Models.predict_lsi, bundle, req.text)
        )
        return LSIResponse(predictions=[LSIPrediction(**p) for p in predictions])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LSI prediction error")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rr", response_model=RRResponse)
async def analyze_rr(req: LegalTextRequest):
    """Classify rhetorical roles of each sentence in the text."""
    from api.services.ml_service import NyayAI_Models

    loop = asyncio.get_event_loop()
    try:
        bundle = await loop.run_in_executor(None, _load_model, "rr")
        predictions = await loop.run_in_executor(
            None,
            partial(NyayAI_Models.predict_rr, bundle, req.text)
        )
        return RRResponse(predictions=[RRPrediction(**p) for p in predictions])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RR prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cjpe", response_model=CJPEResponse)
async def analyze_cjpe(req: LegalTextRequest):
    """Predict court judgment outcome (Accepted/Rejected)."""
    from api.services.ml_service import NyayAI_Models

    loop = asyncio.get_event_loop()
    try:
        bundle = await loop.run_in_executor(None, _load_model, "cjpe")
        prediction = await loop.run_in_executor(
            None,
            partial(NyayAI_Models.predict_cjpe, bundle, req.text)
        )
        return CJPEResponse(prediction=CJPEPrediction(**prediction))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CJPE prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full", response_model=FullAnalysisResponse)
async def analyze_full(req: LegalTextRequest):
    """Run all three InLegalBERT analyses concurrently."""
    from api.services.ml_service import NyayAI_Models

    loop = asyncio.get_event_loop()
    try:
        # per-model locks in get_model() make this safe even if this is
        # the very first request the process has seen for all three -
        # each name's load runs once, the others just wait on their own lock
        lsi_bundle, rr_bundle, cjpe_bundle = await asyncio.gather(
            loop.run_in_executor(None, _load_model, "lsi"),
            loop.run_in_executor(None, _load_model, "rr"),
            loop.run_in_executor(None, _load_model, "cjpe"),
        )

        lsi_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_lsi, lsi_bundle, req.text)
        )
        rr_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_rr, rr_bundle, req.text)
        )
        cjpe_task = loop.run_in_executor(
            None, partial(NyayAI_Models.predict_cjpe, cjpe_bundle, req.text)
        )

        lsi_raw, rr_raw, cjpe_raw = await asyncio.gather(lsi_task, rr_task, cjpe_task)

        return FullAnalysisResponse(
            lsi_predictions=[LSIPrediction(**p) for p in lsi_raw],
            rr_predictions=[RRPrediction(**p) for p in rr_raw],
            cjpe_prediction=CJPEPrediction(**cjpe_raw),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Full analysis error")
        raise HTTPException(status_code=500, detail=str(e))