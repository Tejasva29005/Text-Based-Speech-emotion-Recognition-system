import torch
import gradio as gr
from transformers import pipeline

MODEL_PATH = "models/emotion-classifier"

EMOTION_EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😄",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "anger":    "#FF5252",
    "disgust":  "#8BC34A",
    "fear":     "#9C27B0",
    "joy":      "#FFD600",
    "neutral":  "#90A4AE",
    "sadness":  "#42A5F5",
    "surprise": "#FF7043",
}

# Load model once at startup
print("Loading model...")
clf = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    top_k=None,           # return scores for all classes
    device=0 if torch.cuda.is_available() else -1,
)
print("Model ready.")


def predict(text: str):
    """Return top emotion label + confidence bars for all 7 classes."""
    if not text.strip():
        return "Please enter some text.", {}

    results = clf(text, truncation=True, max_length=128)[0]
    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top       = results[0]
    top_label = top["label"]
    top_emoji = EMOTION_EMOJI.get(top_label, "")
    top_conf  = top["score"]

    summary = f"{top_emoji} **{top_label.capitalize()}** — {top_conf*100:.1f}% confidence"

    # Build dict for gr.Label (label → confidence)
    confidences = {
        f"{EMOTION_EMOJI.get(r['label'], '')} {r['label']}": r["score"]
        for r in results
    }

    return summary, confidences


EXAMPLES = [
    "I just got the job offer I've been waiting for!",
    "I can't stop crying, everything feels hopeless.",
    "There's something moving outside — I'm terrified.",
    "Yeah sure, whatever you say.",
    "Wait, you're getting married?! I had no idea!",
    "That is absolutely revolting, how could anyone do that.",
    "I am so done with this. This is infuriating.",
]

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:

    gr.Markdown("## 🎭 Speech Emotion Recognition")
    gr.Markdown(
        "Fine-tuned **DistilBERT** on GoEmotions · "
        "Detects: anger · disgust · fear · joy · neutral · sadness · surprise"
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text",
                placeholder="Type a sentence expressing an emotion...",
                lines=4,
            )
            with gr.Row():
                clear_btn  = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyse", variant="primary")

            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

        with gr.Column(scale=1):
            result_label  = gr.Markdown(label="Top prediction")
            result_scores = gr.Label(
                label="Confidence scores",
                num_top_classes=7,
            )

    submit_btn.click(fn=predict, inputs=text_input, outputs=[result_label, result_scores])
    text_input.submit(fn=predict, inputs=text_input, outputs=[result_label, result_scores])
    clear_btn.click(fn=lambda: ("", "", {}), outputs=[text_input, result_label, result_scores])

if __name__ == "__main__":
    demo.launch(share=False)
