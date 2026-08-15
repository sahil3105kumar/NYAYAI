

import gc
import logging

import torch
from celery.exceptions import SoftTimeLimitExceeded

from workers.celery_app import app
from services.analysis import run_analysis

logger = logging.getLogger(__name__)


@app.task(name="workers.tasks.process_pdf", bind=True)
def process_pdf(self, job_id: str) -> dict:
    try:
        return run_analysis(job_id)
    except SoftTimeLimitExceeded:
        
        logger.error(
            "process_pdf soft time limit exceeded for job_id=%s "
            "(likely an oversized or pathologically slow PDF)",
            job_id,
        )
        raise
    finally:
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()