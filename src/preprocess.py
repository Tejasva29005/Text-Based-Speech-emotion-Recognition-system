"""
Phase 1 & 2: Dataset loading and preprocessing
Dataset: GoEmotions (Google) — simplified 7-class version
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import os

# ── Label mapping ──────────────────────────────────────────────────────────────
# GoEmotions simplified has 28 labels; we map them to our 7 target emotions.
# Labels not in our target set are dropped during filtering.

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# GoEmotions simplified label names → our 7 classes
GO_EMOTIONS_MAP = {
    "anger":     "anger",
    "annoyance": "anger",
    "disgust":   "disgust",
    "fear":      "fear",
    "nervousness": "fear",
    "joy":       "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love":      "joy",
    "optimism":  "joy",
    "pride":     "joy",
    "admiration": "joy",
    "gratitude": "joy",
    "relief":    "joy",
    "caring":    "joy",
    "neutral":   "neutral",
    "sadness":   "sadness",
    "grief":     "sadness",
    "disappointment": "sadness",
    "remorse":   "sadness",
    "embarrassment": "sadness",
    "surprise":  "surprise",
    "realization": "surprise",
    "curiosity": "surprise",
    "confusion": "surprise",
    "desire":    None,   # dropped — ambiguous
    "approval":  None,   # dropped — not a clear emotion category
    "disapproval": None, # dropped
}


def load_and_map_dataset(save_path: str = "data/mapped") -> dict:
    print("Loading GoEmotions dataset...")
    raw = load_dataset("go_emotions", "simplified")

    label_names = raw["train"].features["labels"].feature.names

    def remap(example):
        """Map one example's labels list → single integer target or -1 to drop."""
        for lid in example["labels"]:
            go_name = label_names[lid]
            mapped = GO_EMOTIONS_MAP.get(go_name)
            if mapped is not None:
                return {"label": LABEL_TO_ID[mapped]}
        return {"label": -1}   # mark for removal

    SPLIT_LIMITS = {"train": 5000, "validation": 1000, "test": 1000}

    mapped = {}
    for split in ["train", "validation", "test"]:
        ds = raw[split].map(remap, remove_columns=["labels"])
        ds = ds.filter(lambda x: x["label"] != -1)
        ds = ds.select(range(min(SPLIT_LIMITS[split], len(ds))))
        mapped[split] = ds
        print(f"  {split}: {len(ds):,} samples after mapping")

    os.makedirs(save_path, exist_ok=True)
    for split, ds in mapped.items():
        ds.save_to_disk(os.path.join(save_path, split))

    # Save label metadata
    with open(os.path.join(save_path, "label_map.json"), "w") as f:
        json.dump({"id_to_label": ID_TO_LABEL, "label_to_id": LABEL_TO_ID}, f, indent=2)

    print(f"\nDataset saved to '{save_path}/'")
    return mapped


def tokenize_dataset(
    dataset_dict: dict,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    save_path: str = "data/tokenized",
) -> dict:
    """
    Tokenize text using the chosen model's tokenizer.
    Adds 'input_ids', 'attention_mask' columns.
    """
    print(f"\nTokenizing with '{model_name}' (max_length={max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = {}
    for split, ds in dataset_dict.items():
        tok_ds = ds.map(tokenize, batched=True, batch_size=512)
        tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        tokenized[split] = tok_ds
        print(f"  {split}: tokenized {len(tok_ds):,} samples")

    os.makedirs(save_path, exist_ok=True)
    for split, ds in tokenized.items():
        ds.save_to_disk(os.path.join(save_path, split))

    print(f"Tokenized dataset saved to '{save_path}/'")
    return tokenized


def show_class_distribution(dataset_dict: dict):
    """Print class distribution for each split."""
    for split, ds in dataset_dict.items():
        labels = ds["label"]
        counts = pd.Series(labels).value_counts().sort_index()
        print(f"\n{split.upper()} distribution:")
        for lid, count in counts.items():
            pct = count / len(labels) * 100
            print(f"  {ID_TO_LABEL[lid]:<12} {count:>6,}  ({pct:.1f}%)")


if __name__ == "__main__":
    mapped = load_and_map_dataset()
    show_class_distribution(mapped)
    tokenize_dataset(mapped)
    print("\nPreprocessing complete.")