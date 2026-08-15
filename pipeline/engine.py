
from ocr.tokens import LineSpan
from model.schemas import ErrorSpan
from model.preprocess import build_chunks
from model.predict import predict
from model.postprocess import build_error_spans

from rules.registry import RULES

from pipeline.merger import merge_spans
from pipeline.deduplicate import deduplicate


def analyze(spans: list[LineSpan]) -> list[ErrorSpan]:
    ml_errors = _run_ml(spans)

    # run every registered rule checker — to add a new rule, edit rules/registry.py only
    rule_errors = []
    for rule in RULES:
        rule_errors.extend(rule(spans))

    merged = merge_spans(ml_errors, rule_errors)
    deduped = deduplicate(merged)

    return _sort_reading_order(deduped)


def _run_ml(spans: list[LineSpan]) -> list[ErrorSpan]:
    chunks = build_chunks(spans)
    label_id_sequences = predict(chunks)
    return build_error_spans(chunks, label_id_sequences, spans)


def _sort_reading_order(errors: list[ErrorSpan]) -> list[ErrorSpan]:
    """page by page, top-to-bottom, left-to-right - the order a human reading the document would hit them."""
    return sorted(errors, key=lambda e: (e.page_no, e.y0, e.x0))