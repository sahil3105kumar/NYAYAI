

import pdfplumber
from pathlib import Path

from ocr.tokens import LineSpan
from ocr.native_extractor import NativeExtractor
from config.constants import MIN_CHARS_PER_PAGE, MIN_LINES_PER_PAGE, MAX_SCANNED_INDICATORS


def route(
    pdf_path: Path,
    min_chars_per_page: int = MIN_CHARS_PER_PAGE,
    min_lines_per_page: int = MIN_LINES_PER_PAGE,
    min_alphabetic_ratio: float = 0.6,
    max_scanned_indicators: int = MAX_SCANNED_INDICATORS
) -> tuple[list[LineSpan], list[int]]:
    
    native = NativeExtractor()
    
    
    all_spans, page_stats = native.extract(pdf_path)
    
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    
    spans_by_page: dict[int, list[LineSpan]] = {}
    
    for span in all_spans:
        spans_by_page.setdefault(span.page_no, []).append(span)
    
    native_spans = []
    scanned_pages = []
    
    for page_no in range(total_pages):
        
        stats = page_stats.get(page_no, {})
        
        
        is_native = (
            stats.get("char_count", 0) >= min_chars_per_page and
            stats.get("line_count", 0) >= min_lines_per_page and
            stats.get("alphabetic_ratio", 0) >= min_alphabetic_ratio and
            stats.get("scanned_indicators", max_scanned_indicators) <= max_scanned_indicators
        )
        
        if is_native:
            native_spans.extend(spans_by_page.get(page_no, []))
        else:
            scanned_pages.append(page_no)
    
    return native_spans, scanned_pages