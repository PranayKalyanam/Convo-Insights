"""SQLite storage for processed conversations."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .extractor import ExtractionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id     TEXT PRIMARY KEY,
    language            TEXT,
    transcript          TEXT,          -- PII-redacted text that was sent to the LLM
    summary             TEXT,
    intent              TEXT,
    sentiment           TEXT,
    resolution_status   TEXT,
    priority            TEXT,
    next_action         TEXT,
    qa_greeted          INTEGER,
    qa_verified_details INTEGER,
    qa_showed_empathy   INTEGER,
    qa_offered_resolution INTEGER,
    qa_score            REAL,
    valid               INTEGER NOT NULL,
    attempts            INTEGER,
    latency_ms          REAL,
    provider            TEXT,
    model               TEXT,
    pii_counts          TEXT,
    raw_output          TEXT,
    error               TEXT,
    processed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_intent ON conversations(intent);
CREATE INDEX IF NOT EXISTS idx_status ON conversations(resolution_status);
"""


def connect(path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait for another writer (e.g. a background job) instead of failing
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def save(conn: sqlite3.Connection, conversation_id: str, result: ExtractionResult,
         provider: str, model: str, language: str | None = None) -> None:
    ins = result.insight
    qa = ins.agent_qa if ins else None
    row = {
        "conversation_id": conversation_id,
        "language": language,
        "transcript": result.redacted_transcript,
        "summary": ins.summary if ins else None,
        "intent": ins.intent if ins else None,
        "sentiment": ins.sentiment if ins else None,
        "resolution_status": ins.resolution_status if ins else None,
        "priority": ins.priority if ins else None,
        "next_action": ins.next_action if ins else None,
        "qa_greeted": int(qa.greeted) if qa else None,
        "qa_verified_details": int(qa.verified_details) if qa else None,
        "qa_showed_empathy": int(qa.showed_empathy) if qa else None,
        "qa_offered_resolution": int(qa.offered_resolution) if qa else None,
        "qa_score": qa.score if qa else None,
        "valid": int(result.valid),
        "attempts": result.attempts,
        "latency_ms": round(result.latency_ms, 1),
        "provider": provider,
        "model": model,
        "pii_counts": json.dumps(result.pii_counts),
        "raw_output": result.raw_output,
        "error": result.error,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cols = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT OR REPLACE INTO conversations ({cols}) VALUES ({marks})", row)
    conn.commit()


def existing_ids(conn: sqlite3.Connection, provider: str | None = None,
                 model: str | None = None) -> set[str]:
    """Ids already processed successfully, optionally only by this provider and model
    (so switching models re-processes rows instead of silently reusing old answers)."""
    sql, args = "SELECT conversation_id FROM conversations WHERE valid=1", []
    if provider is not None and model is not None:
        sql += " AND provider=? AND model=?"
        args = [provider, model]
    return {r[0] for r in conn.execute(sql, args)}


def load_df(path: Path | str = config.DB_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path, timeout=30)
    try:
        return pd.read_sql_query("SELECT * FROM conversations", conn)
    finally:
        conn.close()


def delete(conversation_id: str) -> bool:
    conn = connect()
    try:
        cur = conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
