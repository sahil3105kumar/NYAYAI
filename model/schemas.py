"""
shared schemas for the model/ package: the label scheme (BIO tags) and
the ErrorSpan dataclass that both model/postprocess.py and
rules/*.py return.
"""

from dataclasses import dataclass
from typing import Optional

# -------------------------------------------------------------------
# label scheme — BIO encoding, 7 labels
#
# B-ENT / I-ENT were removed: scripts/generate_data.py never generated
# ENT training examples, so those slots were never trained in the old
# 9-label head. The current checkpoint (model/checkpoint/) was trained
# from scratch on exactly these 7 classes with class-weighted loss.
# Entity-consistency detection is unaffected — it's handled entirely
# by rules/entity_checker.py (NER + fuzzy matching), independent of
# this model.
# -------------------------------------------------------------------

LABELS = ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM", "B-CITE", "I-CITE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# BIO prefix -> long-form error_type used by rules/*.py, renderer/colors.py,
# config/constants.py's ERROR_COLORS, and the frontend. "entity" stays here
# even though the model no longer emits it, because rules/entity_checker.py
# still produces ErrorSpans with error_type="entity".
ERROR_TYPES = {
    "SPELL": "spelling",
    "GRAM": "grammar",
    "CITE": "citation",
    "ENT": "entity",
}


@dataclass
class ErrorSpan:
    text: str
    error_type: str            # "spelling" | "grammar" | "citation" | "entity"
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0
    suggestion: Optional[str] = None
    source: str = "model"      # "model" | "citation_rule" | "entity_rule" | "cross_reference_rule" | "spelling_rule"
    explanation: Optional[str] = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def highlight_color(self) -> str:
        from config.constants import ERROR_COLORS
        return ERROR_COLORS.get(self.error_type, "#CCCCCC")