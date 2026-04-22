"""
DistilBERT classifier + high-level inference wrapper.
"""
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

LABEL_MAP = {"Left": 0, "Center": 1, "Right": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = 3

def build_model(model_name: str = "distilbert-base-uncased", num_labels: int = NUM_LABELS):
    return DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels,
        id2label = ID_TO_LABEL,
        label2id = LABEL_MAP,
    )

def load_saved_model(model_dir: str):
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model


class BiasClassifier:
    """High-level inference wrapper used by the Streamlit app."""

    def __init__(self, model_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_saved_model(model_dir).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 512) -> dict:
        from src.preprocessing import clean_text
        enc = self.tokenizer(
            clean_text(text),
            max_length = max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim = -1).squeeze().cpu()
        pred = probs.argmax().item()

        return {
            "label": ID_TO_LABEL[pred],
            "confidence": round(probs[pred].item(), 4),
            "scores": {ID_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(NUM_LABELS)},
        }