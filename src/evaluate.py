"""
Evaluate on test set. Saves metrics + confusion matrix.
"""
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from transformers import DistilBertTokenizerFast

from src.model import load_saved_model, ID_TO_LABEL
from src.dataset import NewsBiasDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "models/distilbert-bias"
TEST_CSV = "data/processed/test.csv"
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def predict_all(model, tokenizer, dataset, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(
                device), attention_mask=batch["attention_mask"].to(device))
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return all_preds, all_labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_saved_model(MODEL_DIR).to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    dataset = NewsBiasDataset(TEST_CSV, tokenizer)

    preds, labels = predict_all(model, tokenizer, dataset, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    names = [ID_TO_LABEL[i] for i in range(3)]

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print(classification_report(labels, preds, target_names=names))

    # Confusion matrix
    cm = confusion_matrix(labels, preds).astype(float)
    cm /= cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm * 100, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.title("Confusion matrix (%)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix → {ARTIFACTS / 'confusion_matrix.png'}")

    with open(ARTIFACTS / "test_metrics.json", "w") as f:
        json.dump({"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}, f)


if __name__ == "__main__":
    main()
