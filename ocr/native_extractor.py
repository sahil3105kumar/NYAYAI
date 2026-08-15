

import pdfplumber
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from itertools import groupby

from ocr.tokens import LineSpan


class NativeExtractor:
    
    
    
    VERTICAL_TOLERANCE = 2.0
    
    
    MIN_LINE_LENGTH = 2
    
    def extract(self, pdf_path: Path) -> Tuple[List[LineSpan], Dict[int, Dict[str, Any]]]:
       
        spans = []
        page_stats = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                # Extract words with additional options for better accuracy
                words = page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=['fontname', 'size']
                )
                
                if words:
                    page_spans = self._words_to_linespans(words, page_no)
                    spans.extend(page_spans)
                    
                    # Record page statistics
                    page_stats[page_no] = self._analyze_page_stats(words, page_spans)
                else:
                    page_stats[page_no] = {
                        "char_count": 0,
                        "line_count": 0,
                        "avg_line_length": 0,
                        "alphabetic_ratio": 0.0,
                        "word_count": 0,
                        "scanned_indicators": 3,  # Strongly indicates scanned
                    }
        
        return spans, page_stats
    
    def _words_to_linespans(
        self, 
        words: list[dict], 
        page_no: int
    ) -> list[LineSpan]:
        """
        Group words into lines using vertical tolerance.
        
        Args:
            words: List of word dicts from pdfplumber
            page_no: Page number
            
        Returns:
            List of LineSpan objects
        """
        if not words:
            return []
        
        
        words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
        
        spans = []
        
        
        lines = []
        current_line = []
        current_y = None
        
        for word in words_sorted:
            word_top = word["top"]
            
            if current_y is None:
                
                current_y = word_top
                current_line = [word]
            elif abs(word_top - current_y) <= self.VERTICAL_TOLERANCE:
                
                current_line.append(word)
            else:
                
                lines.append(current_line)
                current_y = word_top
                current_line = [word]
        
        if current_line:
            lines.append(current_line)
        
        
        for line_words in lines:
            
            line_words.sort(key=lambda w: w["x0"])
            
            
            text = " ".join(w["text"] for w in line_words)
            
            
            x0 = min(w["x0"] for w in line_words)
            x1 = max(w["x1"] for w in line_words)
            y0 = min(w["top"] for w in line_words)
            y1 = max(w["bottom"] for w in line_words)
            
            
            font_name = line_words[0].get("fontname")
            font_size = line_words[0].get("size")
            
            span = LineSpan(
                text=text,
                page_no=page_no,
                source="native",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                font_name=font_name,
                font_size=font_size,
            )
            
            if span.is_valid() and not span.is_noise():
                spans.append(span)
        
        return spans
    
    def _analyze_page_stats(
        self, 
        words: list[dict], 
        spans: list[LineSpan]
    ) -> Dict[str, Any]:
        """Analyze page statistics for router decision."""
        if not words:
            return {
                "char_count": 0,
                "line_count": 0,
                "avg_line_length": 0,
                "alphabetic_ratio": 0.0,
                "word_count": 0,
                "scanned_indicators": 3,
            }
        
        word_count = len(words)
        char_count = sum(len(w["text"]) for w in words)
        line_count = len(spans)
        
    
        all_text = " ".join(w["text"] for w in words)
        alpha_chars = sum(1 for c in all_text if c.isalpha())
        alphabetic_ratio = alpha_chars / max(len(all_text), 1)
        
        
        scanned_indicators = 0
        
        
        if word_count < 10:
            scanned_indicators += 1
        
        
        avg_word_len = char_count / max(word_count, 1)
        if avg_word_len < 3:
            scanned_indicators += 1
        
        
        if alphabetic_ratio < 0.5:
            scanned_indicators += 1
        
        return {
            "char_count": char_count,
            "line_count": line_count,
            "avg_line_length": char_count / max(line_count, 1),
            "alphabetic_ratio": alphabetic_ratio,
            "word_count": word_count,
            "scanned_indicators": scanned_indicators,
        }
    
    def has_text_layer(
        self, 
        pdf_path: Path, 
        min_chars_per_page: int = 20
    ) -> dict[int, bool]:
        """
        Legacy method: per-page check for text layer.
        
        Returns:
            Dict mapping page_no -> has_text_layer
        """
        spans, stats = self.extract(pdf_path)
        return {
            page_no: stats.get(page_no, {}).get("char_count", 0) >= min_chars_per_page
            for page_no in stats
        }