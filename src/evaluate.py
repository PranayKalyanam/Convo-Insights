"""Evaluate the pipeline against labelled test data and compare with the baseline.

    python -m src.evaluate                       # uses data/test.csv
    python -m src.evaluate --test data/my_hand_labelled.csv

Any rows of the test file not yet in the database are processed first.
Outputs (in reports/):
  eval_report.md         human-readable summary (also shown in the dashboard)
  eval_metrics.json      the same numbers, machine-readable
  intent_confusion.csv   confusion matrix for intent (LLM)
  errors.csv             every row where the LLM got intent/sentiment/resolution wrong
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import baseline, config, db, pipeline
from .llm_client import LLMError

TARGETS = ["intent", "sentiment", "resolution_status", "priority"]
QA_COLS = [f"qa_{k}" for k in config.QA_CHECKS]


def _scores(gold: pd.Series, pred: pd.Series) -> dict:
    pred = pred.fillna("__invalid__")  # an invalid row counts as a wrong answer
    # Average F1 over the real labels only. Without `labels=`, "__invalid__" would be
    # treated as an extra class with F1 = 0 and unfairly drag the macro average down.
    labels = sorted(gold.dropna().unique())
    return {"accuracy": round(accuracy_score(gold, pred), 3),
            "macro_f1": round(f1_score(gold, pred, labels=labels, average="macro", zero_division=0), 3),
            "n": int(len(gold))}


def evaluate(test_csv: str, process: bool = True, on_progress=None, verbose: bool = True) -> dict:
    gold = pd.read_csv(test_csv)
    gold["conversation_id"] = gold["conversation_id"].astype(str)
    if process:
        pipeline.run(test_csv, on_progress=on_progress)

    preds = db.load_df()
    if preds.empty:
        raise RuntimeError("Database is empty. Run the pipeline first.")
    df = gold.merge(preds, on="conversation_id", how="left", suffixes=("_gold", "_pred"))
    missing = df["valid"].isna().sum()
    if missing:
        print(f"Warning: {missing} test rows have no prediction in the DB.")
    df["valid"] = df["valid"].fillna(0)
    valid = df[df["valid"] == 1]
    provider = ", ".join(sorted(valid["provider"].dropna().unique())) or "n/a"
    model = ", ".join(sorted(valid["model"].dropna().unique())) or "n/a"

    m: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "test_file": str(test_csv), "provider": provider, "model": model,
               "n_test": int(len(df))}

    # --- classification quality (LLM)
    m["llm"] = {t: _scores(df[f"{t}_gold"], df[f"{t}_pred"]) for t in TARGETS if f"{t}_gold" in df}
    if all(f"{c}_gold" in df for c in QA_COLS):
        m["agent_qa_accuracy"] = {
            c.removeprefix("qa_"): round(float((valid[f"{c}_gold"] == valid[f"{c}_pred"]).mean()), 3)
            for c in QA_COLS}

    # --- per language
    if "language_gold" in df:
        m["intent_by_language"] = {
            lang: _scores(g["intent_gold"], g["intent_pred"])
            for lang, g in df.groupby("language_gold")}

    # --- reliability and cost proxies
    lat = valid["latency_ms"]
    m["operational"] = {
        "invalid_output_rate": round(1 - df["valid"].mean(), 3),
        "needed_retry_rate": round(float((valid["attempts"] > 1).mean()), 3) if len(valid) else None,
        "latency_ms_mean": round(float(lat.mean()), 1) if len(lat) else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if len(lat) else None,
    }

    # --- baseline
    models = baseline.load()
    bpred, bms = baseline.predict(models, df["transcript_gold"])
    m["baseline"] = {t: _scores(df[f"{t}_gold"], bpred[t]) for t in baseline.TARGETS}
    m["baseline"]["latency_ms_mean"] = round(bms, 3)

    # --- artefacts for error analysis
    config.REPORTS_DIR.mkdir(exist_ok=True)
    pd.crosstab(df["intent_gold"], df["intent_pred"].fillna("__invalid__"),
                rownames=["gold"], colnames=["predicted"]).to_csv(config.REPORTS_DIR / "intent_confusion.csv")
    core = [t for t in ["intent", "sentiment", "resolution_status"] if f"{t}_gold" in df]
    wrong_mask = np.zeros(len(df), dtype=bool)
    for t in core:
        wrong_mask |= (df[f"{t}_gold"] != df[f"{t}_pred"]).to_numpy()
    wrong = df[wrong_mask]
    cols = ["conversation_id", "language_gold", "transcript_gold"] + \
           [f"{t}_{s}" for t in core for s in ("gold", "pred")] + ["error"]
    wrong[[c for c in cols if c in df]].to_csv(config.REPORTS_DIR / "errors.csv", index=False)
    m["n_rows_with_errors"] = int(len(wrong))

    (config.REPORTS_DIR / "eval_metrics.json").write_text(json.dumps(m, indent=2))
    (config.REPORTS_DIR / "eval_report.md").write_text(render_markdown(m))
    if verbose:
        print(render_markdown(m))
    return m


def render_markdown(m: dict) -> str:
    out = [f"# Evaluation report", "",
           f"Generated {m['generated_at']} on `{m['test_file']}` ({m['n_test']} conversations).",
           f"Extractor: **{m['provider']} / {m['model']}**.", ""]
    if "mock" in m["provider"]:
        out += ["> These numbers come from the offline keyword mock, not an LLM. "
                "Switch `LLM_PROVIDER` in `.env` and re-run for real results.", ""]
    out += ["## LLM pipeline vs TF-IDF baseline", "",
            "| Label | LLM accuracy | LLM macro-F1 | Baseline accuracy | Baseline macro-F1 |",
            "|---|---|---|---|---|"]
    for t, s in m["llm"].items():
        b = m["baseline"].get(t)
        out.append(f"| {t} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                   + (f"{b['accuracy']:.3f} | {b['macro_f1']:.3f} |" if b else "n/a | n/a |"))
    out += ["", "Invalid LLM outputs are counted as wrong answers. The baseline has no "
            "priority model because priority is derived from the other labels.", ""]
    if "agent_qa_accuracy" in m:
        out += ["## Agent QA checklist agreement", "", "| Check | Agreement with gold |", "|---|---|"]
        out += [f"| {k} | {v:.3f} |" for k, v in m["agent_qa_accuracy"].items()] + [""]
    if "intent_by_language" in m:
        out += ["## Intent by language", "", "| Language | n | Accuracy | Macro-F1 |", "|---|---|---|---|"]
        out += [f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |"
                for k, v in m["intent_by_language"].items()] + [""]
    o = m["operational"]
    out += ["## Reliability and speed", "",
            f"- Invalid output rate after retries: {o['invalid_output_rate']:.1%}",
            f"- Conversations that needed a retry: "
            + (f"{o['needed_retry_rate']:.1%}" if o["needed_retry_rate"] is not None else "n/a"),
            f"- LLM latency per conversation: mean {o['latency_ms_mean']} ms, p95 {o['latency_ms_p95']} ms",
            f"- Baseline latency per conversation: {m['baseline']['latency_ms_mean']} ms", "",
            f"{m['n_rows_with_errors']} rows had at least one wrong label; see `reports/errors.csv`.",
            "Summaries and next actions are not scored automatically: review a sample by hand "
            "and check that every claim is supported by the transcript.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=str(config.DATA_DIR / "test.csv"))
    ap.add_argument("--no-process", action="store_true", help="do not call the LLM for missing rows")
    a = ap.parse_args()
    try:
        evaluate(a.test, process=not a.no_process, on_progress=pipeline._print_progress)
    except (RuntimeError, ValueError, FileNotFoundError, LLMError) as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
