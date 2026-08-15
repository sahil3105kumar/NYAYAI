import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import threading
import time
import logging
import gc
logger = logging.getLogger(__name__)

# Absolute base for model weights — avoids CWD-dependent resolution
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_SAVED_MODELS = _ROOT_DIR / "saved_models"

# Automatically download the sentence splitting rules if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID = "law-ai/InLegalBERT"

# 2. Reconstruct Label-Wise Attention Model Architecture
class LSILabelAttentionModel(nn.Module):
    def __init__(self, model_id, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id)
        self.label_attention = nn.Linear(768, num_labels)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        
        attn_weights = self.label_attention(sequence_output)  # [batch_size, seq_len, num_labels]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        label_repr = torch.bmm(attn_weights.transpose(1, 2), sequence_output)  # [batch_size, num_labels, 768]
        logits = self.classifier(label_repr).squeeze(-1)  # [batch_size, num_labels]
        return logits


class NyayAI_Models:
    @staticmethod
    def load_lsi_model():
        """
        Loads Tokenizer, Weights, and Label Mappings for LSI.
        """
        weights_path = str(_SAVED_MODELS / "lsi" / "model_weights.pt")
        mapping_path = str(_SAVED_MODELS / "lsi" / "lsi_label_mapping.json")

        if not os.path.exists(weights_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"LSI Model weights or mapping missing! Ensure '{weights_path}' and '{mapping_path}' exist."
            )

        # Load Label Mapping
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings['id2label'].items()}
        num_classes = len(id2label)

        # Initialize Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = LSILabelAttentionModel(MODEL_ID, num_labels=num_classes)

        # Load State Dict
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label
        }
    @staticmethod
    def load_rr_model():
        """Loads the Rhetorical Role classification model."""
        weights_path = str(_SAVED_MODELS / "rr" / "best_rr_model.pt")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # FIX: Change num_labels from 7 to 13 to match your trained checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=13)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        
        model.to(device)
        model.eval()
        
        # FIX: The standard 13-class mapping for Indian Legal Judgments
        id2label = {
            0: "PREAMBLE",
            1: "FACTS",
            2: "RULING BY LOWER COURT",
            3: "ISSUE",
            4: "ARGUMENT BY PETITIONER",
            5: "ARGUMENT BY RESPONDENT",
            6: "ANALYSIS",
            7: "STATUTE",
            8: "PRECEDENT RELIED",
            9: "PRECEDENT NOT RELIED",
            10: "RATIO OF THE DECISION",
            11: "RULING BY PRESENT COURT",
            12: "NONE"
        }
        
        return {"model": model, "tokenizer": tokenizer, "id2label": id2label}

    @staticmethod
    def load_cjpe_model():
        """Loads the Court Judgment Prediction model (Binary: 0=Rejected, 1=Accepted)."""
        weights_path = str(_SAVED_MODELS / "cjpe" / "best_cjpe_model.pt")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            
        model.to(device)
        model.eval()
        return {"model": model, "tokenizer": tokenizer}

    @staticmethod
    def predict_lsi(lsi_bundle, text: str, threshold: float = 0.30, top_k: int = 5):
        """
        Performs inference on input text using the loaded LSI bundle.
        """
        model = lsi_bundle["model"]
        tokenizer = lsi_bundle["tokenizer"]
        id2label = lsi_bundle["id2label"]

        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.sigmoid(logits)[0]

        top_probs, top_indices = torch.topk(probs, k=top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            confidence = float(prob.item())
            raw_label = id2label.get(idx.item(), f"Section {idx.item()}")
            
            # Clean up formatting
            clean_label = raw_label.replace("IPC Section Section", "IPC Section")

            results.append({
                "statute": clean_label,
                "confidence": round(confidence, 4),
                "matched": confidence >= threshold
            })

        return results
    
    @staticmethod
    def predict_rr(rr_bundle, text: str):
        """Splits text into sentences and classifies the rhetorical role of each."""
        model = rr_bundle["model"]
        tokenizer = rr_bundle["tokenizer"]
        id2label = rr_bundle["id2label"]

        # Naive sentence splitting (For production, consider installing 'nltk')
        sentences = [s.strip() + "." for s in text.split('.') if len(s.strip()) > 15]
        
        results = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = tokenizer(sentence, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                
                conf, idx = torch.max(probs, dim=0)
                results.append({
                    "sentence": sentence,
                    "rhetorical_role": id2label.get(idx.item(), "UNKNOWN"),
                    "confidence": round(float(conf.item()), 4)
                })
        return results

    @staticmethod
    def predict_cjpe(cjpe_bundle, text: str):
        """Predicts if the appeal will be accepted or rejected."""
        model = cjpe_bundle["model"]
        tokenizer = cjpe_bundle["tokenizer"]

        inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
            conf, idx = torch.max(probs, dim=0)
            outcome = "Accepted / Allowed" if idx.item() == 1 else "Rejected / Dismissed"
            
        return {
            "outcome": outcome,
            "confidence": round(float(conf.item()), 4)
        }
    


    _cache: dict = {}
    _locks = {"lsi": threading.Lock(), "rr": threading.Lock(), "cjpe": threading.Lock()}
    _last_used: dict = {}

    _LOADERS = {
        "lsi": "load_lsi_model",
        "rr": "load_rr_model",
        "cjpe": "load_cjpe_model",
    }

    @classmethod
    def get_model(cls, name: str) -> dict:
        """
        Thread-safe lazy loader. First caller for a given model name pays
        the load cost and populates the cache; every call after that (from
        any thread) just returns the cached bundle. Deliberately per-model
        locks, not one global lock — a request for "rr" shouldn't block
        behind an in-flight "lsi" load.
        """
        if name not in cls._LOADERS:
            raise ValueError(f"Unknown model '{name}'")

        cls._last_used[name] = time.monotonic()

        bundle = cls._cache.get(name)
        if bundle is not None:
            return bundle

        with cls._locks[name]:
            # re-check: another thread may have finished loading while we
            # were waiting on the lock
            bundle = cls._cache.get(name)
            if bundle is None:
                logger.info(f"Loading {name.upper()} model (first use this process)...")
                loader = getattr(cls, cls._LOADERS[name])
                bundle = loader()
                cls._cache[name] = bundle
                logger.info(f"{name.upper()} model ready")
            return bundle

    @classmethod
    def unload_all(cls):
        
        cls._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()