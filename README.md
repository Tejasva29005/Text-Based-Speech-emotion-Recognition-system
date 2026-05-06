# Speech Emotion Recognition — NLP Project

Fine-tunes **DistilBERT** on the **GoEmotions** dataset to classify text into
7 emotions: anger · disgust · fear · joy · neutral · sadness · surprise.

---

## Project structure

```
emotion-nlp/
├── src/
│   ├── preprocess.py   # Phase 1 & 2 — data loading + tokenization
│   ├── train.py        # Phase 3 & 4 — model setup + training
│   └── evaluate.py     # Phase 5    — metrics + confusion matrix
├── app.py              # Phase 6    — Gradio demo
├── data/               # created at runtime
├── models/             # saved checkpoints
├── results/            # plots saved here
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess
Downloads GoEmotions, maps to 7 classes, tokenizes.
```bash
python src/preprocess.py
```

### 3. Train
Fine-tunes DistilBERT. Runs on CPU (slow) or GPU (recommended).
```bash
python src/train.py
```

### 4. Evaluate
Prints classification report, saves confusion matrix + F1 bar chart to `results/`.
```bash
python src/evaluate.py
```

### 5. Launch demo
```bash
python app.py
# Open http://localhost:7860
```

---

## Key design decisions

| Choice | Why |
|---|---|
| GoEmotions (simplified) | Largest public text-emotion dataset; clean 7-class variant |
| DistilBERT | 40% smaller than BERT, 60% faster, ~97% performance |
| Weighted F1 as primary metric | Dataset is class-imbalanced (joy/neutral >> fear/disgust) |
| EarlyStopping (patience=2) | Prevents overfitting; typical best epoch is 2–3 |
| max_length=128 | Reddit comments are short; 128 tokens covers 95%+ |

---

## Expected results (after fine-tuning)

| Metric | Typical value |
|---|---|
| Validation accuracy | ~65–70% |
| Weighted F1 | ~67–72% |
| Best per-class F1 | joy, neutral (~75–80%) |
| Hardest class | disgust, fear (~50–60%) |

---

## Tips for improvement

- **Class weighting** — pass `class_weight` to handle imbalance
- **Data augmentation** — back-translate minority classes
- **Larger model** — swap DistilBERT for `bert-base-uncased` or `roberta-base`
- **Hyperparameter search** — use `optuna` + `Trainer`'s `hyperparameter_search`
- **Multi-label** — GoEmotions is originally multi-label; explore that variant
