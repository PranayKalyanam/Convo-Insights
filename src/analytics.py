"""Filtering and aggregate statistics over processed conversations."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from . import config, db

QA_COLS = [f"qa_{k}" for k in config.QA_CHECKS]
LIST_COLUMNS = ["conversation_id", "language", "intent", "sentiment", "resolution_status",
                "priority", "qa_score", "summary", "processed_at"]


@dataclass
class Filters:
    intent: list[str] | None = None
    sentiment: list[str] | None = None
    resolution_status: list[str] | None = None
    priority: list[str] | None = None
    language: list[str] | None = None
    q: str | None = None


def load_valid() -> pd.DataFrame:
    df = db.load_df()
    return df[df["valid"] == 1] if not df.empty else df


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["intent", "sentiment", "resolution_status", "priority", "language"]:
        values = getattr(f, col)
        if values:
            df = df[df[col].isin(values)]
    if f.q:
        mask = (df["transcript"].str.contains(f.q, case=False, na=False, regex=False)
                | df["summary"].str.contains(f.q, case=False, na=False, regex=False)
                | df["conversation_id"].str.contains(f.q, case=False, na=False, regex=False))
        df = df[mask]
    return df


def clean(value):
    """Make pandas / numpy values JSON-safe."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    return [{c: clean(r[c]) for c in columns} for _, r in df[columns].iterrows()]


def stats(df: pd.DataFrame) -> dict:
    """Everything the dashboard needs in one response."""
    if df.empty:
        return {"total": 0}
    n = len(df)
    by_intent = []
    for intent, g in df.groupby("intent"):
        by_intent.append({
            "intent": intent,
            "total": int(len(g)),
            "resolution": {s: int((g["resolution_status"] == s).sum()) for s in config.RESOLUTION_STATUSES},
            "sentiment": {s: int((g["sentiment"] == s).sum()) for s in config.SENTIMENTS},
        })
    by_intent.sort(key=lambda r: r["total"], reverse=True)
    queue = df[(df["priority"] == "high") & (df["resolution_status"] != "resolved")]
    return {
        "total": n,
        "unresolved_rate": round(float((df["resolution_status"] == "unresolved").mean()), 3),
        "negative_rate": round(float((df["sentiment"] == "negative").mean()), 3),
        "high_priority": int((df["priority"] == "high").sum()),
        "avg_qa_score": round(float(df["qa_score"].mean()), 1),
        "by_intent": by_intent,
        "qa_pass_rates": {c.removeprefix("qa_"): round(float(df[c].mean()), 3) for c in QA_COLS},
        "languages": {str(k): int(v) for k, v in df["language"].fillna("unknown").value_counts().items()},
        "follow_up_queue": records(
            queue.sort_values("processed_at", ascending=False).head(50),
            ["conversation_id", "intent", "sentiment", "summary", "next_action"]),
    }
