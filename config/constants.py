
MAX_TOKENS = 512

CHUNK_STRIDE = 128

INFERENCE_BATCH_SIZE = 8


ERROR_COLORS = {
    "spelling": "#FFD700",
    "grammar": "#FFA500",
    "citation": "#FF4444",
    "entity": "#00BFFF",
}

# Embeddings constants for corpus/embeddings.py
MODEL_NAME = "law-ai/InLegalBERT"
BATCH_SIZE = 16

# acts for parsing and chunking
ACTS = ["ipc", "bns", "bnss", "cpc","crpc", "constitution"]

# Rendering constants for anotate_pdf.py, report.py, html_report.py, colors.py
FILL_ALPHA = 0.35
STROKE_ALPHA = 0.9
STROKE_WIDTH = 1.2

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB - generous for a scanned multi-page FIR

# Router constants for ocr/router.py
MIN_CHARS_PER_PAGE = 200
MIN_LINES_PER_PAGE = 3
MAX_SCANNED_INDICATORS = 1


MAX_PAGES = 100


LARGE_PDF_PAGE_THRESHOLD = 15


LARGE_PDF_REDUCED_SCALE = 1.5
