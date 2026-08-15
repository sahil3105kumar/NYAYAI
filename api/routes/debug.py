

from pathlib import Path

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from config.settings import settings
from api.schemas.response import JobState
from workers.celery_app import app as celery_app
from workers.queues import PDF_PROCESSING_QUEUE

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/queue")
def queue_depth():
    """
    the filesystem broker has no service to ask for queue depth - a
    queued task is just a file under celery_broker_data_folder/out until
    a worker picks it up and moves it to .../processed. counting files in
    "out" is the closest equivalent to a real broker's queue-depth metric.
    """
    out_dir = Path(settings.celery_broker_data_folder) / "out"
    pending = list(out_dir.glob("*")) if out_dir.exists() else []
    return {"queue": PDF_PROCESSING_QUEUE, "pending_tasks": len(pending)}


@router.post("/jobs/{job_id}/force-status")
def force_job_status(job_id: str, status: JobState):
    """
    overwrites a job's status directly in the result backend, without
    actually running the pipeline - useful for exercising the frontend's
    polling states (queued/processing/failed/done) on demand instead of
    waiting on a real document each time.
    """
    result = AsyncResult(job_id, app=celery_app)
    try:
        result.backend.store_result(job_id, result=None, state=status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to force status: {exc}")

    return {"job_id": job_id, "status": status}