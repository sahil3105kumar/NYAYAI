

import gc
import logging
import pypdfium2 as pdfium
import re
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from config.constants import LARGE_PDF_PAGE_THRESHOLD, LARGE_PDF_REDUCED_SCALE
from ocr.tokens import LineSpan

logger = logging.getLogger(__name__)



_CACHED_DETECTION_PREDICTOR: Optional[DetectionPredictor] = None
_CACHED_RECOGNITION_PREDICTOR: Optional[RecognitionPredictor] = None


def _get_shared_predictors() -> Tuple[DetectionPredictor, RecognitionPredictor]:
    global _CACHED_DETECTION_PREDICTOR, _CACHED_RECOGNITION_PREDICTOR

    if _CACHED_DETECTION_PREDICTOR is None:
        logger.info("loading surya detection + recognition models (first use this process)")
        _CACHED_DETECTION_PREDICTOR = DetectionPredictor()
        _CACHED_RECOGNITION_PREDICTOR = RecognitionPredictor()

    return _CACHED_DETECTION_PREDICTOR, _CACHED_RECOGNITION_PREDICTOR


def unload_surya_models() -> None:
    
    global _CACHED_DETECTION_PREDICTOR, _CACHED_RECOGNITION_PREDICTOR

    if _CACHED_DETECTION_PREDICTOR is None and _CACHED_RECOGNITION_PREDICTOR is None:
        return

    logger.info("unloading surya detection + recognition models")
    _CACHED_DETECTION_PREDICTOR = None
    _CACHED_RECOGNITION_PREDICTOR = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class SuryaExtractor:
    
    
    
    DEFAULT_SCALE = 2.0  # 144 DPI
    HIGH_ACCURACY_SCALE = 2.8  # ~200 DPI
    
    def __init__(
        self, 
        scale: Optional[float] = None,
        chunk_size: int = 2,
        min_confidence: float = 0.3,
        detect_layout: bool = True,
        page_count: Optional[int] = None,
    ):
        
        if scale is None:
            if page_count is not None and page_count > LARGE_PDF_PAGE_THRESHOLD:
                scale = LARGE_PDF_REDUCED_SCALE
            else:
                scale = self.DEFAULT_SCALE
        self.scale = scale
        self.chunk_size = chunk_size
        self.min_confidence = min_confidence
        self.detect_layout = detect_layout
        
        
        self.detection_predictor, self.recognition_predictor = _get_shared_predictors()
        
        
        self._image_cache: Dict[int, Image.Image] = {}
    
    def _render_chunk(
        self,
        doc: "pdfium.PdfDocument",
        page_numbers: List[int],
    ) -> List[Image.Image]:
        
        images = []
        for page_no in page_numbers:
            bitmap = doc[page_no].render(scale=self.scale) #type: ignore
            images.append(bitmap.to_pil())
        return images
    
    def _filter_noise_lines(self, lines: List[Any], page_no: int) -> List[Tuple[str, Tuple[float, float, float, float], Optional[float]]]:
        
        filtered = []
        
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            
            # Skip very short lines (likely noise)
            if len(text) < 2:
                continue
            
            # Skip lines that are just separators
            if all(c in "-_=~*" for c in text):
                continue
            
            # Skip page numbers
            if text.isdigit():
                continue
            if re.match(r'^Page \d+$', text, re.I):
                continue
            
            # Skip URLs
            if re.match(r'^www\.[a-z0-9]+\.[a-z]+$', text, re.I):
                continue
            if re.match(r'^https?://', text, re.I):
                continue
            
            # Get confidence
            confidence = None
            if hasattr(line, 'confidence'):
                confidence = line.confidence
            
            # Skip low confidence lines
            if confidence is not None and confidence < self.min_confidence:
                continue
            
            # Get bbox
            bbox = line.bbox
            
            # Skip invalid bboxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            
            filtered.append((text, bbox, confidence))
        
        return filtered
    
    def _detect_layout(self, lines: List[Tuple[str, Tuple[float, float, float, float], Optional[float]]]) -> List[Dict[str, Any]]:
        
        if not lines:
            return []
        
        # Sort by vertical position
        sorted_lines = sorted(lines, key=lambda x: x[1][1])
        
        result = []
        prev_y1 = None
        gaps = []
        
        # First pass: calculate gaps
        for i, (text, bbox, confidence) in enumerate(sorted_lines):
            x0, y0, x1, y1 = bbox
            if i == 0:
                gap = 0.0
            else:
                _, prev_bbox, _ = sorted_lines[i-1]
                gap = y0 - prev_bbox[3]
                gaps.append(gap)
            
            result.append({
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "gap": gap,
                "is_heading": False,
                "is_paragraph_start": False,
            })
        
        # Calculate adaptive threshold
        if gaps:
            median_gap = sorted(gaps)[len(gaps) // 2]
            threshold = median_gap * 1.8
        else:
            threshold = 10.0
        
        # Second pass: detect paragraph starts and headings
        for i, item in enumerate(result):
            # Paragraph start
            if i == 0:
                item["is_paragraph_start"] = True
            else:
                item["is_paragraph_start"] = item["gap"] > threshold
            
            # Heading detection
            text = item["text"]
            if len(text) < 80:
                # ALL CAPS
                if text.isupper() and len(text) > 5:
                    item["is_heading"] = True
                # Ends with colon
                elif text.endswith(':'):
                    item["is_heading"] = True
                # Short line with high alphabetic ratio
                elif len(text) < 60 and sum(c.isalpha() for c in text) / max(len(text), 1) > 0.8:
                    item["is_heading"] = True
        
        return result
    
    def extract(
        self, 
        pdf_path: Path, 
        page_numbers: List[int]
    ) -> List[LineSpan]:
        
        if not page_numbers:
            return []
        
        all_spans = []
        failed_pages: List[int] = []
        doc = pdfium.PdfDocument(pdf_path)
        
        try:
            
            pending: List[List[int]] = [
                page_numbers[i:i + self.chunk_size]
                for i in range(0, len(page_numbers), self.chunk_size)
            ]

            while pending:
                chunk_page_nos = pending.pop(0)
                chunk_images = self._render_chunk(doc, chunk_page_nos)
                
                # Run detection + recognition
                try:
                    results = self.recognition_predictor(
                        chunk_images,
                        [None] * len(chunk_images),
                        self.detection_predictor,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    chunk_images.clear()
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                    if len(chunk_page_nos) > 1:
                        mid = len(chunk_page_nos) // 2
                        logger.warning(
                            "CUDA OOM on pages %s (chunk of %d) - "
                            "splitting into %d + %d pages and retrying",
                            chunk_page_nos, len(chunk_page_nos),
                            mid, len(chunk_page_nos) - mid,
                        )
                        pending.insert(0, chunk_page_nos[mid:])
                        pending.insert(0, chunk_page_nos[:mid])
                    else:
                        
                        logger.error(
                            "CUDA OOM on page %d even at chunk size 1 - "
                            "this page's text will be MISSING from OCR "
                            "output: %s",
                            chunk_page_nos[0], e,
                        )
                        failed_pages.append(chunk_page_nos[0])
                    continue
                except Exception as e:
                    logger.warning(
                        "Error processing pages %s: %s", chunk_page_nos, e
                    )
                    chunk_images.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    failed_pages.extend(chunk_page_nos)
                    continue
                
                for page_no, page_result in zip(chunk_page_nos, results):
                    # Filter noise lines
                    filtered = self._filter_noise_lines(
                        page_result.text_lines, 
                        page_no
                    )
                    
                    if not filtered:
                        continue
                    
                    # Detect layout if enabled
                    if self.detect_layout:
                        layout_info = self._detect_layout(filtered)
                    else:
                        layout_info = [{"is_heading": False, "is_paragraph_start": False} 
                                      for _ in filtered]
                    
                    # Create LineSpans
                    for line_info, layout in zip(filtered, layout_info):
                        text, bbox, confidence = line_info
                        x0, y0, x1, y1 = bbox
                        
                        span = LineSpan(
                            text=text,
                            page_no=page_no,
                            source="surya",
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            confidence=confidence,
                            is_heading=layout.get("is_heading", False),
                            is_paragraph_start=layout.get("is_paragraph_start", False),
                            vertical_gap=layout.get("gap", 0.0),
                        )
                        
                        if span.is_valid():
                            all_spans.append(span)
                
                
                chunk_images.clear()
                gc.collect()
        finally:
            doc.close()

        if failed_pages:
            logger.error(
                "Surya OCR failed on %d/%d page(s) of this document "
                "(pages %s) - their text is missing from the result. "
                "Check GPU memory (nvidia-smi) if this happens often.",
                len(failed_pages), len(page_numbers), sorted(failed_pages),
            )
        
        
        self.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return all_spans
    
    def extract_with_stats(
        self, 
        pdf_path: Path, 
        page_numbers: List[int]
    ) -> Tuple[List[LineSpan], Dict[int, Dict[str, Any]]]:
        """
        Extract text and return page statistics.
        
        Returns:
            Tuple of (spans, page_stats)
        """
        spans = self.extract(pdf_path, page_numbers)
        
        # Group spans by page
        by_page: Dict[int, List[LineSpan]] = {}
        for span in spans:
            by_page.setdefault(span.page_no, []).append(span)
        
        # Calculate stats per page
        page_stats = {}
        for page_no, page_spans in by_page.items():
            page_stats[page_no] = {
                "line_count": len(page_spans),
                "char_count": sum(len(s.text) for s in page_spans),
                "avg_confidence": sum(s.confidence or 0 for s in page_spans) / max(len(page_spans), 1),
                "heading_count": sum(1 for s in page_spans if s.is_heading),
            }
        
        return spans, page_stats
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self._image_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()


# Convenience function for one-off extraction
def extract_scanned(
    pdf_path: Path,
    page_numbers: List[int],
    scale: float = SuryaExtractor.DEFAULT_SCALE,
    chunk_size: int = 2,
    min_confidence: float = 0.3,
    detect_layout: bool = True,
) -> List[LineSpan]:
    
    with SuryaExtractor(
        scale=scale,
        chunk_size=chunk_size,
        min_confidence=min_confidence,
        detect_layout=detect_layout,
    ) as extractor:
        return extractor.extract(pdf_path, page_numbers)