"""
fine-tunes InLegalBERT for token classification using the HuggingFace
Trainer API.

checkpoint loading/saving conventions here match model/predict.py exactly
on purpose: predict.py looks for a config.json in model/checkpoint/ and,
if found, loads BOTH the model and the tokenizer from that same directory.
so this file has to save the tokenizer into checkpoint_dir too, not just
the model weights - if it only saved model weights, predict.py would load
a tokenizer that doesn't necessarily match (e.g. after a future retrain
with a different base checkpoint).

hyperparameters below are reasonable BERT-fine-tuning defaults, not
empirically tuned for this specific task - this hasn't been run yet.
adjust batch size / gradient accumulation first if it doesn't fit in 6GB.
"""
import torch
from transformers import Trainer
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from config.settings import settings

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from model.schemas import LABELS, LABEL2ID, ID2LABEL
from train.dataset import load_dataset
from train.collator import build_collator
from train.metrics import compute_metrics

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_CHECKPOINT = settings.bert_checkpoint
OUTPUT_DIR = settings.checkpoint_dir

TRAIN_JSONL = settings.training_dir / "train.jsonl"
VAL_JSONL = settings.training_dir / "val.jsonl"

# starting point, not yet empirically tuned - this project's known 6GB VRAM
# constraint (RTX 4050) is why batch size is small with gradient
# accumulation making up the effective batch size, same discipline as
# model/predict.py's INFERENCE_BATCH_SIZE and surya's chunked page batching
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch size = 8 * 4 = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10                       # was 5
EARLY_STOPPING_PATIENCE = 3           # new

CLASS_WEIGHTS = {                     # from training_config.json — recompute if you regenerate data
    "O": 0.0969, "B-SPELL": 0.4195, "I-SPELL": 0.3053,
    "B-GRAM": 0.5402, "I-GRAM": 2.1885, "B-CITE": 2.0735, "I-CITE": 1.3759,
}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _git_commit() -> Optional[str]:
    """same technique as scripts/generate_data.py's own manifest - silently
    None if git isn't available or this isn't a checkout, never raises
    (a missing git binary shouldn't block a training run)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _sha256_of_file(path: str) -> Optional[str]:
    """checksums train.jsonl/val.jsonl as they ACTUALLY exist right now,
    rather than trusting generate_data.py's manifest.json to still be
    accurate - if the data was regenerated (or hand-edited) after that
    manifest was written but generate_data.py wasn't rerun again, this
    still correctly records what THIS training run actually consumed."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "device": torch.cuda.get_device_name(0),
                "vram_total_gb": round(props.total_memory / (1024 ** 3), 2),
            }
    except Exception:
        pass
    return {"device": "cpu", "vram_total_gb": None}


def _generate_training_manifest(
    output_dir: str,
    train_examples: int,
    val_examples: int,
    training_seconds: float,
    trainer: Trainer,
) -> None:
    """writes training_manifest.json alongside the saved checkpoint - same
    provenance discipline scripts/generate_data.py already applies to the
    training DATA (git commit, checksums, full config), applied here to
    the one artifact in this pipeline that was missing it: the checkpoint
    itself. without this, "which data and which code produced this exact
    model/checkpoint/" is only answerable from memory six months from now.
    """
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "base_checkpoint": BASE_CHECKPOINT,
        "training_data": {
            "train_jsonl": str(TRAIN_JSONL),
            "train_jsonl_sha256": _sha256_of_file(str(TRAIN_JSONL)),
            "train_examples": train_examples,
            "val_jsonl": str(VAL_JSONL),
            "val_jsonl_sha256": _sha256_of_file(str(VAL_JSONL)),
            "val_examples": val_examples,
        },
        "hyperparameters": {
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "fp16": True,
        },
        "hardware": _gpu_info(),
        "training_duration_seconds": round(training_seconds, 1),
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        # full per-epoch train/eval loss + F1 history, not just the final
        # number - lets you actually look at the loss curves later (see
        # overfitting concern) without needing to have watched the logs live
        "log_history": trainer.state.log_history,
    }

    manifest_path = Path(output_dir) / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"wrote training manifest to {manifest_path}")


def main():
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    train_dataset = load_dataset(TRAIN_JSONL, tokenizer)
    val_dataset = load_dataset(VAL_JSONL, tokenizer)
    logger.info(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_CHECKPOINT,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # base model has no classification head yet - expected
    )
    weight_tensor = torch.tensor([CLASS_WEIGHTS[l] for l in LABELS], dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True, 
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", # CHANGED from "f1"
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    # CHANGED from Trainer to WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=build_collator(tokenizer),
        compute_metrics=compute_metrics,
        # ADDED parameters below:
        class_weights=weight_tensor.to(model.device if hasattr(model, 'device') else 'cuda'),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # trainer.train() with load_best_model_at_end=True leaves the BEST
    # checkpoint (by eval f1) loaded in trainer.model, not just the last
    # epoch's - save_model() persists that best version
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(OUTPUT_DIR)  # predict.py loads the tokenizer from here too
    logger.info(f"saved best checkpoint (by eval f1) to {OUTPUT_DIR}")

    training_seconds = time.time() - start_time
    _generate_training_manifest(str(OUTPUT_DIR), len(train_dataset), len(val_dataset), training_seconds, trainer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()