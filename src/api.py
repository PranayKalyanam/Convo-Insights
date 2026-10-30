"""REST API for the conversation-insights pipeline.

    uvicorn src.api:app --reload          # development
    python -m src.api                     # same, without auto-reload

Then open http://127.0.0.1:8000 for the web app, or http://127.0.0.1:8000/docs
for interactive API documentation (Swagger UI).
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analytics, config, db, evaluate, pipeline
from .extractor import ExtractionResult, extract
from .check_llm import TEST_TRANSCRIPT, explain
from .llm_client import BaseClient, LLMError, get_client
from .schema import ConversationInsight

FRONTEND_DIR = config.ROOT / "frontend"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_ITEMS = 50

app = FastAPI(
    title="Conversation Insights API",
    version="1.0.0",
    description="Turns customer-support transcripts into structured insights: summary, intent, "
                "sentiment, resolution, priority, next action and an agent-quality checklist.",
)
# The bundled frontend is served from this same origin, so CORS only matters if you
# build a separate frontend. Set CORS_ORIGINS="http://localhost:3000" etc. to allow one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)

_client: BaseClient | None = None
_client_lock = threading.Lock()


def client() -> BaseClient:
    """Create the LLM client once, on first use."""
    global _client
    with _client_lock:
        if _client is None:
            try:
                _client = get_client()
            except LLMError as e:
                raise HTTPException(503, f"LLM provider is not configured: {e}")
        return _client


# ======================================================================= models
class AnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=10, max_length=50_000,
                            examples=["Agent: Hello, RentEasy support.\nCustomer: My UPI payment "
                                      "failed but money was deducted."])
    save: bool = Field(False, description="Store the result in the database")
    conversation_id: str | None = Field(None, max_length=100, description="Defaults to a random id")
    language: str | None = Field(None, max_length=30)


class AnalyzeResponse(BaseModel):
    conversation_id: str | None
    saved: bool
    valid: bool
    insight: ConversationInsight | None
    qa_score: float | None
    masked_transcript: str
    pii_masked: dict[str, int]
    attempts: int
    latency_ms: float
    provider: str
    model: str
    error: str | None = None
    raw_output: str | None = None


class BatchRequest(BaseModel):
    items: list[AnalyzeRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class JobStatus(BaseModel):
    job_id: str
    kind: Literal["process", "evaluate"]
    status: Literal["queued", "running", "done", "failed"]
    total: int = 0
    done: int = 0
    valid: int = 0
    invalid: int = 0
    message: str | None = None
    result: dict | None = None
    started_at: float
    finished_at: float | None = None


def _to_response(res: ExtractionResult, cl: BaseClient, cid: str | None, saved: bool) -> AnalyzeResponse:
    return AnalyzeResponse(
        conversation_id=cid, saved=saved, valid=res.valid, insight=res.insight,
        qa_score=res.insight.agent_qa.score if res.insight else None,
        masked_transcript=res.redacted_transcript, pii_masked=res.pii_counts,
        attempts=res.attempts, latency_ms=round(res.latency_ms, 1),
        provider=cl.provider, model=cl.model, error=res.error,
        raw_output=None if res.valid else res.raw_output,
    )


def _analyze_one(req: AnalyzeRequest) -> AnalyzeResponse:
    cl = client()
    res = extract(req.transcript, cl)
    cid = req.conversation_id or (f"api_{uuid.uuid4().hex[:10]}" if req.save else None)
    if req.save:
        conn = db.connect()
        try:
            db.save(conn, cid, res, cl.provider, cl.model, req.language)
        finally:
            conn.close()
    return _to_response(res, cl, cid, req.save)


# ======================================================================= jobs
_jobs: dict[str, JobStatus] = {}
_jobs_lock = threading.Lock()
_job_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


def _new_job(kind: str, total: int = 0) -> JobStatus:
    job = JobStatus(job_id=uuid.uuid4().hex[:12], kind=kind, status="queued",
                    total=total, started_at=time.time())
    with _jobs_lock:
        _jobs[job.job_id] = job
    return job


def _progress(job: JobStatus):
    def cb(i, total, _cid, ok, _err):
        with _jobs_lock:
            job.done, job.total = i, total
            job.valid += int(ok)
            job.invalid += int(not ok)
    return cb


def _run_job(job: JobStatus, fn):
    job.status = "running"
    try:
        job.result = fn()
        job.status = "done"
    except Exception as e:  # report any failure to the client instead of losing it
        job.status, job.message = "failed", f"{type(e).__name__}: {e}"
    finally:
        job.finished_at = time.time()


# ======================================================================= routes: system
@app.get("/api/health", tags=["system"])
def health():
    df = db.load_df()
    try:
        client()
        setup_error = None
    except HTTPException as e:
        setup_error = e.detail
    return {
        "status": "ok",
        "llm_ready": setup_error is None,
        "llm_setup_error": setup_error,
        "provider": config.LLM_PROVIDER,
        "model": config.active_model(),
        "is_mock": config.LLM_PROVIDER == "mock",
        "conversations_stored": int(len(df)),
        "invalid_stored": int((df["valid"] == 0).sum()) if not df.empty else 0,
    }


@app.get("/api/labels", tags=["system"])
def labels():
    """Allowed values for every label, for building forms and filters."""
    return {
        "intents": config.INTENTS,
        "sentiments": config.SENTIMENTS,
        "resolution_statuses": config.RESOLUTION_STATUSES,
        "priorities": config.PRIORITIES,
        "qa_checks": config.QA_CHECKS,
    }


@app.get("/api/examples/random", tags=["system"])
def random_example():
    """A random transcript from the synthetic dataset, for trying the analyzer."""
    path = config.DATA_DIR / "conversations.csv"
    if not path.exists():
        raise HTTPException(404, "No dataset found. Run: python -m src.generate_data")
    row = pd.read_csv(path).sample(1).iloc[0]
    return {"conversation_id": row["conversation_id"], "language": row.get("language"),
            "transcript": row["transcript"]}


@app.post("/api/llm/test", tags=["system"])
def test_llm():
    """Send one short test conversation to the configured LLM and explain any failure.
    Same check as `python -m src.check_llm`."""
    provider, model = config.LLM_PROVIDER, config.active_model()
    try:
        cl = client()
    except HTTPException as e:
        return {"ok": False, "provider": provider, "model": model,
                "error": e.detail, "advice": explain(str(e.detail), provider, model)}
    res = extract(TEST_TRANSCRIPT, cl)
    out = {"ok": res.valid, "provider": cl.provider, "model": cl.model, "is_mock": cl.provider == "mock",
           "latency_ms": round(res.latency_ms, 1), "attempts": res.attempts}
    if res.valid:
        out["insight"] = res.insight.model_dump()
    else:
        out["error"] = res.error
        out["raw_output"] = res.raw_output
        out["advice"] = ("The model answered but not in the required JSON format; try a larger model."
                         if res.raw_output else explain(res.error or "", cl.provider, cl.model))
    return out


# ======================================================================= routes: analysis
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: AnalyzeRequest):
    """Analyze one transcript. Returns 200 even when the model output fails validation;
    check `valid` and `error`. Provider failures (bad key, network) return 502."""
    resp = _analyze_one(req)
    if not resp.valid and resp.error and " call failed:" in resp.error:
        raise HTTPException(502, resp.error)
    return resp


@app.post("/api/analyze/batch", response_model=list[AnalyzeResponse], tags=["analysis"])
def analyze_batch(req: BatchRequest):
    """Analyze up to 50 transcripts synchronously. For larger files use /api/jobs/upload."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(_analyze_one, req.items))


# ======================================================================= routes: jobs
@app.post("/api/jobs/upload", response_model=JobStatus, status_code=202, tags=["jobs"])
async def upload_csv(file: UploadFile = File(..., description="CSV with conversation_id, transcript"),
                     reprocess: bool = Form(False), limit: int | None = Form(None),
                     workers: int = Form(config.LLM_WORKERS)):
    """Upload a CSV and process it in the background. Poll /api/jobs/{job_id}."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 10 MB; split it or use the CLI pipeline.")
    cl = client()
    try:
        df = pd.read_csv(io.BytesIO(data))
        df = pipeline.prepare_frame(df, reprocess, limit, cl.provider, cl.model)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    job = _new_job("process", total=len(df))
    if df.empty:
        job.status, job.finished_at = "done", time.time()
        job.message = (f"Nothing to process: every row is already stored for {cl.provider}/{cl.model} "
                       "(tick 'Reprocess existing rows' to redo).")
        job.result = {"processed": 0, "valid": 0, "invalid": 0, "seconds": 0}
        return job
    rows = df.to_dict("records")
    workers = max(1, min(workers, 16))
    _job_pool.submit(_run_job, job, lambda: pipeline.process_records(rows, cl, workers, _progress(job)))
    return job


@app.post("/api/jobs/process-sample", response_model=JobStatus, status_code=202, tags=["jobs"])
def process_sample(limit: int | None = Query(None, ge=1), reprocess: bool = False):
    """Process the bundled synthetic dataset (data/conversations.csv) in the background."""
    path = config.DATA_DIR / "conversations.csv"
    if not path.exists():
        raise HTTPException(404, "No dataset found. Run: python -m src.generate_data")
    cl = client()
    df = pipeline.prepare_frame(pd.read_csv(path), reprocess, limit, cl.provider, cl.model)
    job = _new_job("process", total=len(df))
    if df.empty:
        job.status, job.finished_at = "done", time.time()
        job.message = (f"Nothing to process: the sample data is already processed by "
                       f"{cl.provider}/{cl.model} (tick 'Reprocess existing rows' to redo).")
        job.result = {"processed": 0, "valid": 0, "invalid": 0, "seconds": 0}
        return job
    rows = df.to_dict("records")
    _job_pool.submit(_run_job, job, lambda: pipeline.process_records(
        rows, cl, config.LLM_WORKERS, _progress(job)))
    return job


@app.post("/api/jobs/evaluate", response_model=JobStatus, status_code=202, tags=["jobs"])
def run_evaluation():
    """Run the evaluation on data/test.csv in the background (processes missing rows first)."""
    test = config.DATA_DIR / "test.csv"
    if not test.exists():
        raise HTTPException(404, "data/test.csv not found. Run: python -m src.generate_data")
    client()  # fail fast if the provider is misconfigured
    job = _new_job("evaluate")
    _job_pool.submit(_run_job, job, lambda: evaluate.evaluate(
        str(test), process=True, on_progress=_progress(job), verbose=False))
    return job


@app.get("/api/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id (jobs are kept in memory until the server restarts).")
    return job


# ======================================================================= routes: data
def _filters(intent, sentiment, resolution_status, priority, language, q) -> analytics.Filters:
    return analytics.Filters(intent, sentiment, resolution_status, priority, language, q)


@app.get("/api/conversations", tags=["data"])
def list_conversations(
    intent: list[str] | None = Query(None), sentiment: list[str] | None = Query(None),
    resolution_status: list[str] | None = Query(None), priority: list[str] | None = Query(None),
    language: list[str] | None = Query(None), q: str | None = None,
    limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0),
):
    """Paginated list of valid conversations. Repeat a filter to allow several values,
    e.g. ?intent=payment_issue&intent=deposit_refund"""
    df = analytics.apply_filters(analytics.load_valid(),
                                 _filters(intent, sentiment, resolution_status, priority, language, q))
    if not df.empty:
        df = df.sort_values("conversation_id")
    page = df.iloc[offset: offset + limit] if not df.empty else df
    return {"total": int(len(df)), "limit": limit, "offset": offset,
            "items": analytics.records(page, analytics.LIST_COLUMNS) if not page.empty else []}


@app.get("/api/conversations/{conversation_id}", tags=["data"])
def get_conversation(conversation_id: str):
    df = db.load_df()
    row = df[df["conversation_id"] == conversation_id] if not df.empty else df
    if row.empty:
        raise HTTPException(404, f"No conversation with id '{conversation_id}'")
    r = {k: analytics.clean(v) for k, v in row.iloc[0].to_dict().items()}
    r["pii_counts"] = json.loads(r["pii_counts"] or "{}")
    qa_values = {k: r.pop(f"qa_{k}", None) for k in config.QA_CHECKS}
    r["agent_qa"] = {k: (None if v is None else bool(v)) for k, v in qa_values.items()}
    r["valid"] = bool(r["valid"])
    return r


@app.delete("/api/conversations/{conversation_id}", status_code=204, tags=["data"])
def delete_conversation(conversation_id: str):
    if not db.delete(conversation_id):
        raise HTTPException(404, f"No conversation with id '{conversation_id}'")


@app.get("/api/stats", tags=["data"])
def get_stats(
    intent: list[str] | None = Query(None), sentiment: list[str] | None = Query(None),
    resolution_status: list[str] | None = Query(None), priority: list[str] | None = Query(None),
    language: list[str] | None = Query(None), q: str | None = None,
):
    """Aggregates for the dashboard, with the same filters as /api/conversations."""
    df = analytics.apply_filters(analytics.load_valid(),
                                 _filters(intent, sentiment, resolution_status, priority, language, q))
    return analytics.stats(df)


@app.get("/api/evaluation", tags=["data"])
def get_evaluation():
    """Latest evaluation metrics and intent confusion matrix."""
    path = config.REPORTS_DIR / "eval_metrics.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation yet. POST /api/jobs/evaluate or run python -m src.evaluate")
    out = json.loads(path.read_text())
    cm = config.REPORTS_DIR / "intent_confusion.csv"
    if cm.exists():
        m = pd.read_csv(cm, index_col=0)
        out["confusion"] = {"labels_gold": m.index.tolist(), "labels_pred": m.columns.tolist(),
                            "matrix": m.values.tolist()}
    return out


# ======================================================================= frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
