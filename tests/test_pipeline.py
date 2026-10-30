"""Run with:  pytest -q"""
import json

import pytest

from src.extractor import extract, parse_json
from src.llm_client import BaseClient, MockClient
from src.pii import redact

GOOD = {
    "summary": "Customer's UPI rent payment failed but money was deducted; refund promised.",
    "intent": "payment_issue",
    "sentiment": "Negative",               # wrong case on purpose: should be normalised
    "resolution_status": "partially resolved",  # space on purpose
    "priority": "high",
    "next_action": "Track the reversal with the payment gateway.",
    "agent_qa": {"greeted": True, "verified_details": False,
                 "showed_empathy": True, "offered_resolution": True},
}
TRANSCRIPT = "Agent: Hello.\nCustomer: UPI payment fail ho gaya par paisa kat gaya. Call me on 9876543210."


class ScriptedClient(BaseClient):
    """Returns pre-written responses in order, to test retry behaviour."""
    provider = "scripted"

    def __init__(self, responses):
        super().__init__("test")
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.responses.pop(0)


def test_parse_json_handles_fences_and_chatter():
    text = "Sure! Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps {not json}"
    assert parse_json(text)["intent"] == "payment_issue"


def test_parse_json_rejects_truncated():
    with pytest.raises(ValueError):
        parse_json('{"summary": "cut off')


def test_valid_output_is_normalised():
    res = extract(TRANSCRIPT, ScriptedClient([json.dumps(GOOD)]))
    assert res.valid and res.attempts == 1
    assert res.insight.sentiment == "negative"
    assert res.insight.resolution_status == "partially_resolved"
    assert res.insight.agent_qa.score == 75.0


def test_retry_then_success_feeds_error_back():
    bad = dict(GOOD, intent="billing")  # not an allowed label
    client = ScriptedClient([json.dumps(bad), json.dumps(GOOD)])
    res = extract(TRANSCRIPT, client)
    assert res.valid and res.attempts == 2
    assert "PREVIOUS ATTEMPT FAILED" in client.prompts[1]


def test_gives_up_after_max_retries():
    res = extract(TRANSCRIPT, ScriptedClient(["not json"] * 5))
    assert not res.valid and res.insight is None and res.error


def test_pii_is_masked_before_model_sees_it():
    client = ScriptedClient([json.dumps(GOOD)])
    extract(TRANSCRIPT, client)
    assert "9876543210" not in client.prompts[0] and "[PHONE]" in client.prompts[0]


def test_redact_common_indian_formats():
    text, counts = redact("+91 98765 43210, a.b@gmail.com, pay@okaxis, ABCDE1234F, 1234 5678 9012")
    assert counts == {"EMAIL": 1, "UPI": 1, "AADHAAR": 1, "PAN": 1, "PHONE": 1}
    assert "98765" not in text


def test_mock_client_speaks_the_contract():
    res = extract(TRANSCRIPT, MockClient("mock"))
    assert res.valid


def test_provider_presets_explain_missing_keys(monkeypatch):
    from src.llm_client import LLMError, get_client
    for provider, key in [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                          ("anthropic", "ANTHROPIC_API_KEY")]:
        monkeypatch.delenv(key, raising=False)
        with pytest.raises(LLMError, match=key):
            get_client(provider)
    assert get_client("ollama").base_url == "http://localhost:11434/v1"  # no key needed
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        get_client("nonsense")


def test_error_explanations():
    from src.check_llm import explain
    assert "rejected" in explain("Error code: 401 - invalid x-api-key", "groq", "m")
    assert "not available" in explain("Error code: 404 model_not_found", "groq", "m")
    assert "rate limit" in explain("Error code: 429 Rate limit reached", "groq", "m")
    assert "Ollama" in explain("Connection error.", "ollama", "m")
