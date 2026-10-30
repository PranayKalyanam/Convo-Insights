"""The structured output contract. The LLM must return JSON matching this.

If a response does not validate, the extractor retries and, failing that,
stores the row as invalid instead of letting bad data reach the dashboard.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import INTENTS, PRIORITIES, RESOLUTION_STATUSES, SENTIMENTS

Intent = Literal[tuple(INTENTS)]  # type: ignore[valid-type]
Sentiment = Literal[tuple(SENTIMENTS)]  # type: ignore[valid-type]
Resolution = Literal[tuple(RESOLUTION_STATUSES)]  # type: ignore[valid-type]
Priority = Literal[tuple(PRIORITIES)]  # type: ignore[valid-type]


class AgentQA(BaseModel):
    greeted: bool
    verified_details: bool
    showed_empathy: bool
    offered_resolution: bool

    @property
    def score(self) -> float:
        """Share of checklist items the agent met, 0-100."""
        checks = [self.greeted, self.verified_details, self.showed_empathy, self.offered_resolution]
        return round(100 * sum(checks) / len(checks), 1)


class ConversationInsight(BaseModel):
    summary: str = Field(min_length=15, max_length=600)
    intent: Intent
    sentiment: Sentiment
    resolution_status: Resolution
    priority: Priority
    next_action: str = Field(min_length=5, max_length=300)
    agent_qa: AgentQA

    # Small, forgiving normalisation: models sometimes return "Negative" or
    # "partially resolved". We fix case/spacing but never invent a label.
    @field_validator("intent", "sentiment", "resolution_status", "priority", mode="before")
    @classmethod
    def _normalise_label(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v


def json_schema_for_prompt() -> str:
    """A compact, human-readable description of the expected JSON."""
    return (
        "{\n"
        '  "summary": string, 1-3 sentences, only facts stated in the transcript,\n'
        f'  "intent": one of {list(INTENTS)},\n'
        f'  "sentiment": one of {SENTIMENTS}  (the CUSTOMER\'s sentiment at the end of the call),\n'
        f'  "resolution_status": one of {RESOLUTION_STATUSES},\n'
        f'  "priority": one of {PRIORITIES},\n'
        '  "next_action": string, one concrete follow-up for the support team,\n'
        '  "agent_qa": {"greeted": bool, "verified_details": bool, '
        '"showed_empathy": bool, "offered_resolution": bool}\n'
        "}"
    )
