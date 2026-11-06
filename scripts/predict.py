"""
predict.py — Single or Batch inference using the saved best_model.pt
Usage:
    python scripts/predict.py --text "This movie was great!"
    python scripts/predict.py --csv data/sample/sample_reviews.csv --output results/predictions.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "../models/best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"❌ Model not found at {SAVE_PATH}\n"
            "   Run the notebook first to train and save the model."
        )
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {SAVE_PATH}")
    return model, tokenizer


def predict_single(text, model, tokenizer):
    enc = tokenizer(
        text, max_length=MAX_LEN,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        )
    proba = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    pred  = int(np.argmax(proba))
    return {
        "label":      "Positive 😊" if pred == 1 else "Negative 😞",
        "confidence": float(proba[pred]),
        "prob_neg":   float(proba[0]),
        "prob_pos":   float(proba[1]),
    }


def predict_batch_csv(input_csv, output_csv, model, tokenizer,
                      text_col="review", batch_size=64):
    df = pd.read_csv(input_csv)
    texts = df[text_col].tolist()
    all_labels, all_conf = [], []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts, max_length=MAX_LEN,
            padding=True, truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            )
        proba = torch.softmax(out.logits, dim=1).cpu().numpy()
        preds = np.argmax(proba, axis=1)
        all_labels.extend(["positive" if p == 1 else "negative" for p in preds])
        all_conf.extend([float(proba[j][preds[j]]) for j in range(len(preds))])
        print(f"  Processed {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    df["predicted_label"] = all_labels
    df["confidence"]      = [f"{c:.4f}" for c in all_conf]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved → {output_csv}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB DistilBERT Inference")
    parser.add_argument("--text",   type=str, help="Single review text")
    parser.add_argument("--csv",    type=str, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="results/predictions.csv",
                        help="Output CSV path (for batch mode)")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.text:
        result = predict_single(args.text, model, tokenizer)
        print(f"\nReview    : {args.text}")
        print(f"Result    : {result['label']}  (confidence: {result['confidence']:.2%})")
        print(f"Probs     : Neg={result['prob_neg']:.3f}  Pos={result['prob_pos']:.3f}")

    elif args.csv:
        predict_batch_csv(args.csv, args.output, model, tokenizer)

    else:
        parser.print_help()
