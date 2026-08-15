"""
span-level precision/recall/F1 via seqeval, wired into HF Trainer's
compute_metrics callback.

seqeval scores whole ENTITIES (e.g. a 3-token B-CITE/I-CITE/I-CITE span
counts as one correct/incorrect unit), not individual tokens - token-level
accuracy would look artificially high here since the vast majority of
tokens are "O" in any real document.

the part worth being careful about: predictions and labels both come in
padded to the batch's max length, with -100 at every position that isn't a
real word's first subword (see dataset.py). those positions have to be
stripped BEFORE handing anything to seqeval - seqeval has no concept of
"ignore this token," so a stray -100 would either crash it or, worse,
silently get decoded as whatever ID2LABEL.get(-100) returns.
"""

import numpy as np

from model.schemas import ID2LABEL

IGNORE_INDEX = -100


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
    """
    predictions: (batch, seq_len, num_labels) raw logits
    label_ids:   (batch, seq_len) with -100 at ignored positions

    returns (true_labels, pred_labels) as lists of string-label lists, one
    inner list per example, with every -100 position dropped from BOTH
    sides - dropping only from one side would misalign the remaining
    positions between predicted and gold.
    """
    pred_ids = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for example_preds, example_labels in zip(pred_ids, label_ids):
        true_seq = []
        pred_seq = []
        for pred_id, label_id in zip(example_preds, example_labels):
            if label_id == IGNORE_INDEX:
                continue
            true_seq.append(ID2LABEL[label_id])
            pred_seq.append(ID2LABEL[pred_id])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    return true_labels, pred_labels


def compute_metrics(eval_pred) -> dict:
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

    predictions, label_ids = eval_pred
    true_labels, pred_labels = align_predictions(predictions, label_ids)

    # ADDED: Generate classification report for macro metrics
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    macro = report["macro avg"]

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "macro_f1": macro["f1-score"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
    }

def full_report(true_labels: list[list[str]], pred_labels: list[list[str]]) -> str:
    """per-label precision/recall/F1 - used by evaluate.py, not during training."""
    from seqeval.metrics import classification_report

    return classification_report(true_labels, pred_labels, digits=3)