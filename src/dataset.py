"""
PyTorch Dataset for the HuggingFace Trainer.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from src.preprocessing import clean_text

class NewsBiasDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: DistilBertTokenizerFast, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df["text"] = self.df["text"].fillna("").apply(clean_text)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label_id"], dtype = torch.long),
        }
