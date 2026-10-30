"""Classical baseline: TF-IDF + logistic regression, one model per label.

    python -m src.baseline            # train on data/train.csv, save to models/

The point is not to beat the LLM. It is to answer "why use an LLM at all?"
with numbers: if a 50 ms, zero-cost classifier gets close on intent, the LLM
has to earn its place through the things a classifier can't do (summaries,
next actions, agent QA, no training data needed for new labels).
"""
from __future__ import annotations

import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .pii import redact

TARGETS = ["intent", "sentiment", "resolution_status"]


def make_model() -> Pipeline:
    # Word n-grams capture phrases; char n-grams help with Hinglish spelling variation.
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def prepare(texts: pd.Series) -> list[str]:
    return [redact(t)[0] for t in texts]


def train(train_csv=config.DATA_DIR / "train.csv") -> dict:
    df = pd.read_csv(train_csv)
    X = prepare(df["transcript"])
    models = {}
    for target in TARGETS:
        m = make_model()
        m.fit(X, df[target])
        models[target] = m
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(models, config.MODELS_DIR / "baseline.joblib")
    print(f"Trained baseline on {len(df)} rows -> {config.MODELS_DIR / 'baseline.joblib'}")
    return models


def load() -> dict:
    path = config.MODELS_DIR / "baseline.joblib"
    return joblib.load(path) if path.exists() else train()


def predict(models: dict, texts: pd.Series) -> tuple[pd.DataFrame, float]:
    """Return predictions and mean latency per conversation in ms."""
    X = prepare(texts)
    t0 = time.perf_counter()
    preds = pd.DataFrame({t: models[t].predict(X) for t in TARGETS})
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(X))
    return preds, ms


if __name__ == "__main__":
    train()
