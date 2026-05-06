import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

MODEL_PATH     = "models/emotion-classifier"
TOKENIZED_PATH = "data/tokenized"
LABEL_MAP_PATH = "data/mapped/label_map.json"
RESULTS_DIR    = "results"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def get_predictions(model_path: str, dataset_path: str, batch_size: int = 64):
    """Run inference on the test set, return (all_preds, all_labels)."""
    print(f"Loading model from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print(f"Loading test set from '{dataset_path}'...")
    test_ds = load_from_disk(dataset_path)

    all_preds, all_labels = [], []

    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i : i + batch_size]
        input_ids      = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels         = batch["label"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels if isinstance(labels, list) else labels.tolist())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(test_ds)):,} / {len(test_ds):,}")

    return np.array(all_preds), np.array(all_labels)


def print_report(preds, labels):
    """Print per-class precision, recall, F1."""
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=EMOTION_LABELS, digits=4))


def plot_confusion_matrix(preds, labels, save_path: str):
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cmap="Blues",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion matrix (normalised)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to '{save_path}'")


def plot_per_class_f1(preds, labels, save_path: str):
    """Bar chart of per-class F1 scores."""
    report = classification_report(
        labels, preds, target_names=EMOTION_LABELS, output_dict=True
    )
    f1_scores = [report[e]["f1-score"] for e in EMOTION_LABELS]
    colors = ["#4CAF50" if f >= 0.7 else "#FF9800" if f >= 0.5 else "#F44336"
              for f in f1_scores]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(EMOTION_LABELS, f1_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 scores")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="0.70 threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"F1 bar chart saved to '{save_path}'")


def quick_predict(texts: list[str], model_path: str = MODEL_PATH) -> list[dict]:
    clf = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )
    results = clf(texts, truncation=True, max_length=128)
    out = []
    for text, res in zip(texts, results):
        out.append({
            "text":       text,
            "emotion":    res["label"],
            "confidence": round(res["score"], 4),
        })
    return out

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    preds, labels = get_predictions(
        model_path=MODEL_PATH,
        dataset_path=os.path.join(TOKENIZED_PATH, "test"),
    )

    print_report(preds, labels)
    plot_confusion_matrix(preds, labels, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_per_class_f1(preds, labels, os.path.join(RESULTS_DIR, "f1_per_class.png"))

    samples = [
        "I can't believe how amazing this turned out!",
        "This is absolutely disgusting behaviour.",
        "I'm so scared I don't know what to do.",
        "Whatever, it doesn't matter.",
        "Oh wow, I did not see that coming at all.",
    ]
    print("\nQuick predictions on sample sentences:")
    for r in quick_predict(samples):
        print(f"  [{r['emotion']:<10} {r['confidence']:.2f}]  {r['text']}")
