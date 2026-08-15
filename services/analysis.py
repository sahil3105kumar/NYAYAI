"""
orchestrates one document through the full pipeline:
  ocr.pipeline.extract -> pipeline.engine.analyze -> renderer -> save

this is the only file workers/tasks.py calls into - Celery knows nothing
about OCR, the model, or rendering, it just calls run_analysis(job_id).
"""

import gc

import torch

from ocr.pipeline import extract
from ocr.tokens import spans_to_markdown
from ocr.surya_extractor import unload_surya_models
from model.predict import unload_model
from pipeline.engine import analyze
from renderer.annotate_pdf import annotate
from renderer.report import build_report
from renderer.html_report import render_html


from services.storage import (
    upload_path,
    annotated_pdf_path,
    report_json_path,
    report_html_path,
    extracted_text_path,
    get_original_filename,
)

import json


def run_analysis(job_id: str) -> dict:
    """
    runs the full pipeline for an already-uploaded PDF (see
    services.storage.save_upload) and writes every output file.
    returns the report dict - this becomes the Celery task's result,
    in addition to being saved as report_json_path(job_id).
    """
    pdf_path = upload_path(job_id)
    # prefer the filename the user actually uploaded (e.g. "FIR_2024.pdf")
    # over the internal job_id-based storage name - falls back to the
    # storage name only if no metadata was saved for this job (e.g. an
    # upload from before services.storage tracked this, or a caller that
    # invoked save_upload without passing original_filename).
    source_filename = get_original_filename(job_id) or pdf_path.name

    # Mirror image of unload_surya_models() below: if a previous document
    # in this worker left the error-detection model resident, free it
    # before OCR starts so Surya isn't contending with it for VRAM on
    # this document. Without this, unload_surya_models() alone just
    # shifts the original startup-time conflict to recur on every
    # document instead of fixing it - see model/predict.py's unload_model().
    unload_model()

    spans = extract(pdf_path)

    # Save the plain-text/Markdown rendering of what OCR just extracted so
    # the Neo4j knowledge-graph ingestion endpoint (POST /api/v1/chat/ingest)
    # can reuse it later instead of re-opening and re-parsing the PDF from
    # scratch a second time. Cheap to do here - spans are already in memory
    # and already in reading order.
    extracted_text_path(job_id).parent.mkdir(parents=True, exist_ok=True)
    extracted_text_path(job_id).write_text(spans_to_markdown(spans), encoding="utf-8")

    # Surya's detection/recognition models (OCR stage, above) and
    # InLegalBERT (ML stage, below) can both have real weights resident on
    # the GPU at once on large/scanned documents. Surya caches its
    # predictors at module level for the worker's lifetime (see
    # ocr/surya_extractor.py) - gc.collect()/empty_cache() alone never
    # touch those, so unload_surya_models() is what actually releases
    # them, before the ML stage starts allocating. Costs a reload on the
    # next scanned document, which is the right trade on a 6GB card.
    unload_surya_models()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    errors = analyze(spans)

    total_pages = annotate(pdf_path, errors, annotated_pdf_path(job_id))

    report = build_report(errors, source_filename=source_filename, total_pages=total_pages)
    report_json_path(job_id).parent.mkdir(parents=True, exist_ok=True)
    report_json_path(job_id).write_text(json.dumps(report, indent=2))

    render_html(report, report_html_path(job_id))

    return report # returning the report dict allows the Celery task to return it as its result, which can be useful for logging, debugging, or further processing.