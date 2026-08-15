"""
job-id based file layout. one job_id, one uploaded PDF, one annotated PDF,
one JSON report, one HTML report - all named after the same job_id so
nothing has to track a mapping between them separately.

data/uploads/{job_id}.pdf
data/uploads/{job_id}.meta.json   (only if the client sent a filename)
data/outputs/{job_id}_annotated.pdf
data/outputs/{job_id}_report.json
data/outputs/{job_id}_report.html

no cleanup here - the roadmap already flags "no output cleanup" as a known
gap to fix before anything beyond local single-user use.
"""

import json
from pathlib import Path

from config.settings import settings


def upload_path(job_id: str) -> Path:
    return Path(settings.uploads_dir) / f"{job_id}.pdf"


def upload_meta_path(job_id: str) -> Path:
    return Path(settings.uploads_dir) / f"{job_id}.meta.json"


def annotated_pdf_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_annotated.pdf"


def report_json_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_report.json"


def report_html_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_report.html"


def extracted_text_path(job_id: str) -> Path:
    """
    Plain-Markdown text extracted by ocr.pipeline.extract() during the main
    analysis job (see services.analysis.run_analysis) - no bounding boxes,
    just the document's text in reading order. Written once per job so the
    Neo4j knowledge-graph ingestion endpoint (api/routes/chat.py) can reuse
    it instead of re-parsing the source PDF a second time.
    """
    return Path(settings.outputs_dir) / f"{job_id}_extracted.md"


def save_upload(job_id: str, file_bytes: bytes, original_filename: str | None = None) -> Path:
    path = upload_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)

    # the PDF itself is always stored under job_id.pdf regardless of what
    # the client named it (job_id is what every other path in this file
    # keys off) - the original name is saved separately, purely for
    # DISPLAY purposes later (the report should show "FIR_2024.pdf", not
    # the internal storage UUID). optional: older callers that don't pass
    # this still work exactly as before, just with no filename recorded.
    if original_filename:
        upload_meta_path(job_id).write_text(json.dumps({"original_filename": original_filename}))

    return path


def get_original_filename(job_id: str) -> str | None:
    """returns the filename the user actually uploaded (e.g. "FIR_2024.pdf"),
    not the internal job_id-based storage name. returns None if no metadata
    was ever saved for this job (e.g. an upload from before this existed,
    or the client sent no filename) - the caller decides its own fallback
    rather than this function silently fabricating one."""
    meta_path = upload_meta_path(job_id)
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        return data.get("original_filename")
    except Exception:
        return None