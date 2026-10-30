"""Central configuration: label sets, file paths and LLM provider settings.

Everything that other modules need to agree on lives here, so that the
prompt, the validator, the baseline and the dashboard all use the same labels.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "insights.db"))

# ---------------------------------------------------------------- labels
# Intents are modelled on a rental / home-services platform like NoBroker.
INTENTS = {
    "rental_agreement": "Drafting, e-stamping, delivery or correction of a rental agreement",
    "owner_unresponsive": "Owner, tenant or broker not responding after contact was shared",
    "deposit_refund": "Security deposit or token amount refund disputes",
    "packers_movers": "Shifting / relocation service: delay, damage, missing items",
    "home_services": "Cleaning, painting, plumbing, electrician or repair bookings",
    "subscription_plan": "Paid plan benefits, upgrades, refunds of a subscription",
    "payment_issue": "Rent payment via card/UPI failed, deducted twice, or not credited",
    "property_listing": "Posting, editing, verifying or removing a property listing",
    "other": "Anything that does not fit the categories above",
}
SENTIMENTS = ["positive", "neutral", "negative"]
RESOLUTION_STATUSES = ["resolved", "partially_resolved", "unresolved"]
PRIORITIES = ["low", "medium", "high"]

# Agent quality checklist (each item is a yes/no judgement)
QA_CHECKS = {
    "greeted": "Agent greeted the customer and introduced themselves",
    "verified_details": "Agent verified the customer's identity, booking or account details",
    "showed_empathy": "Agent acknowledged the customer's frustration or inconvenience",
    "offered_resolution": "Agent offered a concrete fix, timeline or next step",
}

# ---------------------------------------------------------------- LLM
# LLM_PROVIDER: mock | groq | gemini | ollama | anthropic | openai
# (see src/llm_client.py and the README section "Using an LLM").
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> provider default
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # extra attempts on invalid JSON
# Automatic retries inside the provider SDK for rate limits (429) and network blips.
# The SDK waits as long as the provider asks (retry-after) between attempts.
LLM_SDK_RETRIES = int(os.getenv("LLM_SDK_RETRIES", "5"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "12000"))

# Used when LLM_MODEL is empty. Model names change over time; run
# `python -m src.check_llm` to see which ones your key can use.
DEFAULT_MODELS = {
    "mock": "keyword-heuristic-v1",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-flash-latest",
    "ollama": "llama3.1:8b",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}
# Parallel LLM calls for batch jobs. Free tiers have low rate limits: use 1-2.
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "4"))


def active_model() -> str:
    return LLM_MODEL or DEFAULT_MODELS.get(LLM_PROVIDER, "")
