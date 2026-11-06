# IMDB Sentiment Analysis — DistilBERT

Fine-tunes `distilbert-base-uncased` for binary sentiment classification on the 50K IMDB movie reviews dataset, and serves the model through a FastAPI inference API with a small web frontend.

## Project structure

```
notebooks/IMDB_BERT_Sentiment.ipynb   # EDA, training, evaluation, single + batch inference
app.py                                # FastAPI server (prediction API, serves the frontend)
frontend/index.html                   # Web UI for trying the model
scripts/predict.py                    # CLI inference (single text or CSV)
data/sample/sample_reviews.csv        # Sample input for batch prediction
setup.sh                              # Environment setup (macOS / Linux)
```

## Training setup

| Setting | Value |
|---------|-------|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 256 |
| Batch size | 32 |
| Epochs | 3 |
| Optimizer | AdamW (lr 2e-5, weight decay 0.01) |
| Scheduler | Linear with 10% warmup, gradient clipping at 1.0 |
| Split | 80/20 stratified |
| Metrics | Accuracy, F1, classification report, confusion matrix |

The dataset is downloaded with `kagglehub` (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`), which needs a Kaggle API token in `~/.kaggle/kaggle.json`.

## Running

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Train — run the notebook; the best checkpoint is saved to models/best_model.pt
jupyter notebook notebooks/IMDB_BERT_Sentiment.ipynb

# 2. Serve
uvicorn app:app --reload
#   UI:   http://localhost:8000
#   Docs: http://localhost:8000/docs

# 3. CLI inference
python scripts/predict.py --text "This movie was absolutely fantastic!"
python scripts/predict.py --csv data/sample/sample_reviews.csv --output results/predictions.csv
```

The trained weights (~255 MB) exceed GitHub's file size limit and are not stored in this repository; run the notebook to reproduce them.
