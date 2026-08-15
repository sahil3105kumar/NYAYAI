

from model.schemas import ErrorSpan
from utils.bbox import iou

OVERLAP_THRESHOLD = 0.5  # IoU at or above this = "same error, detected twice"


def deduplicate(spans: list[ErrorSpan]) -> list[ErrorSpan]:
    groups: dict[tuple[int, str], list[ErrorSpan]] = {}
    for span in spans:
        key = (span.page_no, span.error_type)
        groups.setdefault(key, []).append(span)

    result = []
    for group in groups.values():
        result.extend(_dedupe_group(group))
    return result


def _dedupe_group(group: list[ErrorSpan]) -> list[ErrorSpan]:
    
    ordered = sorted(group, key=lambda s: s.confidence, reverse=True)
    kept: list[ErrorSpan] = []

    for span in ordered:
        if not any(iou(span.bbox, k.bbox) >= OVERLAP_THRESHOLD for k in kept):
            kept.append(span)

    return kept