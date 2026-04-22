"""
Downloads and prepares the MBIC dataset.
Falls back to a synthetic demo dataset if the real one isn't available.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents = True, exist_ok = True)
PROCESSED_DIR.mkdir(parents = True, exist_ok = True)

LABEL_MAP = {"Left": 0, "Center": 1, "Right": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def create_demo_dataset(n_samples: int = 900) -> pd.DataFrame:
    print("Creating synthetic demo dataset...")
    templates = {
        "Left": [
            "The government must act immediately to protect workers from corporate greed. "
            "Progressive policies have shown time and again that investing in communities yields results.",
            "Climate change is an existential threat and we must embrace bold renewable energy mandates. "
            "The fossil fuel industry continues to prioritize profits over people.",
            "Universal healthcare is a human right, not a privilege. "
            "The current system leaves millions behind while insurance companies profit.",
        ],
        "Center": [
            "The legislation passed with bipartisan support after weeks of negotiation. "
            "Both parties expressed cautious optimism about the bill's potential impact.",
            "Economists remain divided on the long-term effects of the new trade policy. "
            "Supporters cite job creation while critics warn of unintended consequences.",
            "The committee reviewed evidence from multiple stakeholders before issuing its report. "
            "Officials said further analysis would be needed before reaching a conclusion.",
        ],
        "Right": [
            "Government overreach threatens the freedoms that made this nation great. "
            "Small businesses are being crushed by excessive regulation and taxation.",
            "The radical left's agenda is destroying traditional values and family structures. "
            "Americans must stand up for their constitutional rights against socialist policies.",
            "Border security is a matter of national sovereignty and public safety. "
            "Weak immigration policies have created a crisis that demands immediate action.",
        ],
    }

    rows = []
    np.random.seed(42)
    for label,texts in templates.items():
        for _ in range(n_samples // 3):
            base = np.random.choice(texts)
            noise = np.random.choice(["Furthermore,", "Additionally,", "Meanwhile,", "However,"])
            rows.append({"text": base + f" {noise} the debate continues.", "label": label})

    return pd.DataFrame(rows).sample(frac = 1, random_state = 42).reset_index(drop = True)

def prepare_splits(df: pd.DataFrame) -> dict:
    df["label_id"] = df["label"].map(LABEL_MAP)
    train_val, test = train_test_split(df, test_size = 0.15, random_state = 42, stratify = df["label_id"])
    train, val = train_test_split(train_val, test_size = 0.176, random_state = 42, stratify = train_val["label_id"])
    return {"train": train, "val": val, "test": test}

def main():
    df = create_demo_dataset(n_samples = 900)
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    splits = prepare_splits(df)
    for name, split_df in splits.items():
        out = PROCESSED_DIR / f"{name}.csv"
        split_df.to_csv(out, index = False)
        print(f"Saved {name}: {len(split_df)} rows → {out}")

    with open(PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent = 2)

    print("\n Data ready!")

if __name__ == "__main__":
    main()