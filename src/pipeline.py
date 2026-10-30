"""Batch-process transcripts into the SQLite database.

    python -m src.pipeline --input data/conversations.csv
    python -m src.pipeline --input data/test.csv --limit 20 --workers 4

The CSV needs `conversation_id` and `transcript` columns (`language` optional).
Rows already processed successfully by the SAME provider and model are skipped
unless --reprocess is given, so an interrupted run can simply be restarted, and
switching to a different model processes everything again. The same functions are used by the
API's batch-upload endpoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from . import config, db
from .extractor import extract
from .llm_client import BaseClient, LLMError, get_client

ProgressFn = Callable[[int, int, str, bool, str | None], None]  # done, total, id, valid, error


def prepare_frame(df: pd.DataFrame, reprocess: bool = False, limit: int | None = None,
                  provider: str | None = None, model: str | None = None) -> pd.DataFrame:
    """Validate columns, drop empty rows, and skip rows already processed successfully
    by the same provider and model (all valid rows if provider/model are not given)."""
    missing = {"conversation_id", "transcript"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
    df = df.copy()
    df["conversation_id"] = df["conversation_id"].astype(str)
    if "language" not in df.columns:
        df["language"] = None
    df = df.dropna(subset=["transcript"])
    df = df[df["transcript"].astype(str).str.strip().str.len() > 0]
    df = df.drop_duplicates("conversation_id", keep="last")
    if not reprocess:
        conn = db.connect()
        try:
            df = df[~df["conversation_id"].isin(db.existing_ids(conn, provider, model))]
        finally:
            conn.close()
    return df.head(limit) if limit else df


def process_records(rows: list[dict], client: BaseClient, workers: int = 4,
                    on_progress: ProgressFn | None = None) -> dict:
    """Extract and store each row. Network calls run in threads; writes stay here."""
    conn = db.connect()
    t0, valid = time.perf_counter(), 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(extract, str(r["transcript"]), client): r for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                result = fut.result()
                lang = r.get("language")
                db.save(conn, r["conversation_id"], result, client.provider, client.model,
                        lang if isinstance(lang, str) else None)
                valid += result.valid
                if on_progress:
                    on_progress(i, len(rows), r["conversation_id"], result.valid, result.error)
    finally:
        conn.close()
    elapsed = time.perf_counter() - t0
    return {"processed": len(rows), "valid": valid, "invalid": len(rows) - valid,
            "seconds": round(elapsed, 2)}


def run(input_csv: str, limit: int | None = None, workers: int = 4, reprocess: bool = False,
        provider: str | None = None, model: str | None = None,
        on_progress: ProgressFn | None = None) -> dict:
    """Process a CSV file. Raises ValueError / LLMError instead of exiting."""
    client = get_client(provider, model)
    df = prepare_frame(pd.read_csv(input_csv), reprocess, limit, client.provider, client.model)
    if df.empty:
        return {"processed": 0, "valid": 0, "invalid": 0, "seconds": 0.0}
    return process_records(df.to_dict("records"), client, workers, on_progress)


def _print_progress(i: int, total: int, cid: str, ok: bool, err: str | None) -> None:
    print(f"[{i:>4}/{total}] {'ok ' if ok else 'ERR'} {cid}" + ("" if ok else f"  -> {(err or '')[:120]}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.DATA_DIR / "conversations.csv"))
    ap.add_argument("--limit", type=int, default=None, help="process at most N rows")
    ap.add_argument("--workers", type=int, default=config.LLM_WORKERS,
                    help="parallel LLM calls (default LLM_WORKERS from .env; use 1-2 on free tiers)")
    ap.add_argument("--reprocess", action="store_true", help="redo rows already in the DB")
    ap.add_argument("--provider", default=None, help="override LLM_PROVIDER")
    ap.add_argument("--model", default=None, help="override LLM_MODEL")
    a = ap.parse_args()
    try:
        client = get_client(a.provider, a.model)
        df = prepare_frame(pd.read_csv(a.input), a.reprocess, a.limit, client.provider, client.model)
    except (ValueError, LLMError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")
    if df.empty:
        print(f"Nothing to process: every row is already stored for {client.provider}/{client.model} "
              "(use --reprocess to redo).")
        return
    print(f"Processing {len(df)} conversations with {client.provider}/{client.model} "
          f"using {a.workers} worker(s)...")
    s = process_records(df.to_dict("records"), client, a.workers, _print_progress)
    print(f"\nDone: {s['valid']}/{s['processed']} valid in {s['seconds']}s. Database: {config.DB_PATH}")


if __name__ == "__main__":
    main()
