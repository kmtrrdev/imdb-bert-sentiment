"""
app.py — FastAPI inference server for IMDB DistilBERT Sentiment
Run:      uvicorn app:app --reload
Frontend: http://localhost:8000
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD MODEL ON STARTUP ───────────────────────────────
print(f"🔄 Loading model on {device}...")

if not os.path.exists(SAVE_PATH):
    raise RuntimeError(
        f"❌ Model not found at: {SAVE_PATH}\n"
        "   Train the model first using the notebook, then place best_model.pt in models/"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model ready!")

# ─── FASTAPI APP ─────────────────────────────────────────
app = FastAPI(
    title="🎬 IMDB Sentiment API",
    description="Sentiment analysis using fine-tuned DistilBERT on IMDB 50K reviews",
    version="1.0.0",
)

# CORS — allow all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─── SCHEMAS ─────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely fantastic! Highly recommended."}
        }


class ReviewResponse(BaseModel):
    label:      str
    sentiment:  str
    confidence: float
    prob_neg:   float
    prob_pos:   float


class BatchRequest(BaseModel):
    reviews: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "This movie was fantastic!",
                    "Terrible waste of time.",
                    "It was okay, nothing special."
                ]
            }
        }


class BatchResponse(BaseModel):
    results: list[ReviewResponse]
    total:   int


# ─── HELPER ──────────────────────────────────────────────
def run_inference(texts: list[str]) -> list[dict]:
    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        )
    proba = torch.softmax(out.logits, dim=1).cpu().numpy()
    preds = np.argmax(proba, axis=1)

    results = []
    for i, pred in enumerate(preds):
        results.append({
            "label":      "positive" if pred == 1 else "negative",
            "sentiment":  "Positive 😊" if pred == 1 else "Negative 😞",
            "confidence": float(proba[i][pred]),
            "prob_neg":   float(proba[i][0]),
            "prob_pos":   float(proba[i][1]),
        })
    return results


# ─── ROUTES ──────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
def root():
    """Serve the CineScore frontend."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "🎬 IMDB Sentiment API", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": True, "device": str(device)}


@app.post("/predict", response_model=ReviewResponse, tags=["Inference"])
def predict(req: ReviewRequest):
    """Predict sentiment for a single movie review."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = run_inference([req.text])[0]
    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(req: BatchRequest):
    """Predict sentiment for multiple reviews at once (max 64)."""
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")
    if len(req.reviews) > 64:
        raise HTTPException(status_code=400, detail="Max 64 reviews per batch.")

    results = run_inference(req.reviews)
    return {"results": results, "total": len(results)}
