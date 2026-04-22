"""
Fine-tunes DistilBERT with MLflow experiment tracking.
"""
import os
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments

from src.model import build_model, NUM_LABELS
from src.dataset import NewsBiasDataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
MODEL_OUT = "models/distilbert-bias"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./artifacts/mlflow")

Path(MODEL_OUT).mkdir(parents=True, exist_ok=True)
Path("./artifacts/mlflow").mkdir(parents=True, exist_ok=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1_macro": f1_score(labels, pred, average="macro"),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("news-bias-detector")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = NewsBiasDataset(
        "data/processed/train.csv", tokenizer, MAX_LENGTH)
    val_dataset = NewsBiasDataset(
        "data/processed/val.csv", tokenizer, MAX_LENGTH)

    model = build_model(MODEL_NAME, NUM_LABELS)

    args = TrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=f"distilbert-e{NUM_EPOCHS}-lr{LEARNING_RATE}"):
        mlflow.log_params({
            "model": MODEL_NAME, "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        })
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": metrics["eval_accuracy"],
            "val_f1": metrics["eval_f1_macro"],
            "val_loss": metrics["eval_loss"],
        })
        trainer.save_model(MODEL_OUT)
        tokenizer.save_pretrained(MODEL_OUT)
        with open(f"{MODEL_OUT}/label_map.json", "w") as f:
            json.dump({"Left": 0, "Center": 1, "Right": 2}, f)

    logger.info("Training done !")


if __name__ == "__main__":
    main()
