import os
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

NUM_LABELS     = 7
MODEL_NAME     = "distilbert-base-uncased"
TOKENIZED_PATH = "data/tokenized"
OUTPUT_DIR     = "models/emotion-classifier"
LABEL_MAP_PATH = "data/mapped/label_map.json"

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training schedule
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=2e-5,

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    # Logging
    logging_steps=50,
    report_to="none",

    # Reproducibility
    seed=42,
    fp16=torch.cuda.is_available(),
)

accuracy_metric = hf_evaluate.load("accuracy")
f1_metric       = hf_evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}


def build_model(label_map: dict) -> AutoModelForSequenceClassification:
    """Load DistilBERT and attach a 7-class classification head."""
    id2label = {int(k): v for k, v in label_map["id_to_label"].items()}
    label2id = label_map["label_to_id"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")
    return model

def train():
    # Load label map
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    # Load tokenized datasets
    print("Loading tokenized datasets...")
    train_ds = load_from_disk(os.path.join(TOKENIZED_PATH, "train"))
    val_ds   = load_from_disk(os.path.join(TOKENIZED_PATH, "validation"))
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # Build model
    model = build_model(label_map)

    # Trainer
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()


    print(f"\nSaving best model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(OUTPUT_DIR)
    print("Done.")

    return trainer


if __name__ == "__main__":
    trainer = train()
    results = trainer.evaluate()
    print("\nFinal validation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")