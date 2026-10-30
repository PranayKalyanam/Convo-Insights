"""API tests using FastAPI's in-process TestClient (mock provider, temp database)."""
import io
import time

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
SAMPLE = ("Agent: Hello, thank you for calling RentEasy.\n"
          "Customer: My UPI payment failed but money was deducted, call me on 9876543210. Ridiculous.\n"
          "Agent: Sorry for the inconvenience. The reversal will reach you within 5 to 7 working days.")


def wait(job_id: str, timeout: float = 20) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_health_and_labels():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["is_mock"]
    assert "payment_issue" in client.get("/api/labels").json()["intents"]


def test_frontend_is_served():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_analyze_masks_and_saves():
    r = client.post("/api/analyze", json={"transcript": SAMPLE, "save": True, "conversation_id": "t1"})
    assert r.status_code == 200
    d = r.json()
    assert d["valid"] and d["insight"]["intent"] == "payment_issue"
    assert "9876543210" not in d["masked_transcript"] and d["pii_masked"] == {"PHONE": 1}
    detail = client.get("/api/conversations/t1").json()
    assert detail["agent_qa"]["greeted"] is True


def test_analyze_rejects_short_input():
    assert client.post("/api/analyze", json={"transcript": "hi"}).status_code == 422


def test_batch():
    r = client.post("/api/analyze/batch", json={"items": [{"transcript": SAMPLE}, {"transcript": SAMPLE}]})
    assert r.status_code == 200 and len(r.json()) == 2


def test_upload_job_list_stats_delete():
    csv = "conversation_id,transcript,language\n" + "\n".join(
        f'u{i},"{SAMPLE}",english' for i in range(5))
    r = client.post("/api/jobs/upload", files={"file": ("c.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 202
    j = wait(r.json()["job_id"])
    assert j["status"] == "done" and j["valid"] == 5

    listing = client.get("/api/conversations", params={"intent": "payment_issue", "limit": 3}).json()
    assert listing["total"] >= 5 and len(listing["items"]) == 3
    assert client.get("/api/conversations", params={"q": "u3"}).json()["total"] == 1

    s = client.get("/api/stats").json()
    assert s["total"] >= 5 and s["by_intent"][0]["intent"] == "payment_issue"

    assert client.delete("/api/conversations/u0").status_code == 204
    assert client.get("/api/conversations/u0").status_code == 404


def test_upload_rejects_wrong_columns():
    r = client.post("/api/jobs/upload", files={"file": ("bad.csv", io.BytesIO(b"id,text\n1,hello\n"), "text/csv")})
    assert r.status_code == 400 and "transcript" in r.json()["detail"]


def test_unknown_job():
    assert client.get("/api/jobs/nope").status_code == 404


def test_llm_connection_check():
    r = client.post("/api/llm/test").json()
    assert r["ok"] and r["is_mock"] and r["insight"]["intent"] == "payment_issue"
