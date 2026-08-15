

from model.schemas import ErrorSpan


def merge_spans(*span_lists: list[ErrorSpan]) -> list[ErrorSpan]:
    merged = []
    for spans in span_lists:
        merged.extend(spans)
    return merged