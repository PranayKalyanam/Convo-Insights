"""Turn one transcript into a validated ConversationInsight.

Flow:  redact PII -> build prompt -> call LLM -> parse JSON -> validate
       -> on failure, retry with the validation error fed back to the model.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import config
from .llm_client import BaseClient, LLMError, get_client
from .pii import redact
from .schema import ConversationInsight, json_schema_for_prompt

SYSTEM_PROMPT = f"""You are a quality analyst for the customer-support team of an Indian \
rental and home-services platform (rentals, rental agreements, packers & movers, \
home services, paid plans, rent payments, property listings).

You read one support conversation and return a JSON object. Rules:
- Use ONLY information present in the transcript. Never invent amounts, dates or promises.
- Transcripts may be in English, Hindi or Hinglish (code-mixed). Write output in English.
- Personal data has been replaced with tags like [PHONE]; do not try to guess it.
- "sentiment" is the customer's sentiment by the END of the conversation.
- "resolution_status": resolved = the problem is fixed during the call; \
partially_resolved = some progress or a firm committed timeline but not finished; \
unresolved = nothing fixed, escalated with no commitment, or customer still waiting.
- "priority": high if the customer is negative AND the issue is not resolved, or there is \
money at risk / a legal or safety concern; low if resolved and customer is satisfied; \
otherwise medium.
- agent_qa fields are true only if the agent clearly did that thing.

Intent definitions:
{chr(10).join(f"- {k}: {v}" for k, v in config.INTENTS.items())}

Return ONLY the JSON object, no markdown fences, no commentary. Schema:
{json_schema_for_prompt()}"""


def build_user_prompt(transcript: str, previous_error: str | None = None) -> str:
    prompt = f"<transcript>\n{transcript}\n</transcript>"
    if previous_error:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED validation with this error:\n"
            f"{previous_error[:800]}\n"
            "Return a corrected JSON object that matches the schema exactly."
        )
    return prompt


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Handles the common failure modes: ```json fences, a sentence before the
    JSON, or trailing commentary after it.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("JSON object is not closed (response may be truncated)")


@dataclass
class ExtractionResult:
    insight: ConversationInsight | None
    valid: bool
    attempts: int
    latency_ms: float
    raw_output: str
    error: str | None
    redacted_transcript: str
    pii_counts: dict = field(default_factory=dict)


def extract(transcript: str, client: BaseClient | None = None) -> ExtractionResult:
    client = client or get_client()
    clean = re.sub(r"[ \t]+", " ", transcript).strip()
    clean, pii_counts = redact(clean)
    if len(clean) > config.MAX_TRANSCRIPT_CHARS:
        # Keep the opening (the problem) and the ending (the outcome).
        half = config.MAX_TRANSCRIPT_CHARS // 2
        clean = clean[:half] + "\n[...transcript truncated...]\n" + clean[-half:]

    error: str | None = None
    raw = ""
    t0 = time.perf_counter()
    attempts = 0
    for attempts in range(1, config.MAX_RETRIES + 2):
        try:
            raw = client.complete(SYSTEM_PROMPT, build_user_prompt(clean, error))
        except LLMError as e:
            error = str(e)
            break  # provider failure: retrying the same prompt rarely helps here
        try:
            insight = ConversationInsight.model_validate(parse_json(raw))
            return ExtractionResult(
                insight, True, attempts, (time.perf_counter() - t0) * 1000,
                raw, None, clean, pii_counts,
            )
        except (ValueError, ValidationError) as e:  # json.JSONDecodeError is a ValueError
            error = f"{type(e).__name__}: {e}"
    return ExtractionResult(
        None, False, attempts, (time.perf_counter() - t0) * 1000, raw, error, clean, pii_counts
    )
