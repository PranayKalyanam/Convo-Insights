"""One small interface over several LLM providers.

    client = get_client()
    text = client.complete(system_prompt, user_prompt)

Set LLM_PROVIDER in .env to one of:

mock       Offline keyword heuristic that returns the same JSON shape. Lets you
           run everything without an API key. It is NOT an LLM.
groq       Groq cloud (free tier, no card). Needs GROQ_API_KEY.
gemini     Google Gemini via Google AI Studio (free tier). Needs GEMINI_API_KEY.
ollama     A model running on your own computer with Ollama. No key, no internet.
anthropic  Claude via the official SDK (paid). Needs ANTHROPIC_API_KEY.
openai     OpenAI (paid), or any other OpenAI-compatible server if you also set
           OPENAI_BASE_URL. Needs OPENAI_API_KEY.

groq, gemini, ollama and openai all speak the OpenAI chat API, so they share
one client class; the presets below only fill in the address and key name.
"""
from __future__ import annotations

import json
import os
import random
import re

from . import config


class LLMError(RuntimeError):
    """Raised when the provider call itself fails (network, auth, quota)."""


# Address, key variable and default model for each OpenAI-compatible provider.
OPENAI_COMPATIBLE = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY",
               "key_page": "https://platform.openai.com/api-keys"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "key_page": "https://console.groq.com/keys"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "key_page": "https://aistudio.google.com/apikey"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "key_page": None},
}
KEY_PAGES = {"anthropic": "https://console.anthropic.com/settings/keys",
             **{k: v["key_page"] for k, v in OPENAI_COMPATIBLE.items()}}


class BaseClient:
    provider = "base"

    def __init__(self, model: str):
        self.model = model

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def list_models(self) -> list[str]:
        """Model names this provider/key can use (for diagnostics)."""
        return []


# --------------------------------------------------------------- Anthropic
class AnthropicClient(BaseClient):
    provider = "anthropic"

    def __init__(self, model: str):
        super().__init__(model)
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("The 'anthropic' package is missing: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is empty in .env "
                           f"(create a key at {KEY_PAGES['anthropic']})")
        self._client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_SDK_RETRIES)
        # Older SDK versions take `temperature`; newer ones dropped it. Only send it if accepted.
        import inspect
        params = inspect.signature(self._client.messages.create).parameters
        self._extra = {"temperature": config.LLM_TEMPERATURE} if "temperature" in params else {}

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=config.LLM_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                **self._extra,
            )
        except Exception as e:  # network, auth, rate limit, bad model name
            raise LLMError(f"Anthropic call failed: {e}") from e
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list(limit=100)]


# --------------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleClient(BaseClient):
    """OpenAI, Groq, Gemini, Ollama, or any server that speaks the OpenAI chat API."""

    def __init__(self, model: str, provider: str = "openai"):
        super().__init__(model)
        self.provider = provider
        try:
            import openai
        except ImportError as e:
            raise LLMError("The 'openai' package is missing: pip install openai") from e
        preset = OPENAI_COMPATIBLE[provider]
        base_url = os.getenv("OPENAI_BASE_URL") if provider == "openai" else preset["base_url"]
        base_url = base_url or None
        if preset["key_env"]:
            api_key = os.getenv(preset["key_env"])
            if not api_key and provider == "openai" and base_url:
                api_key = "not-needed"  # many self-hosted servers ignore the key
            if not api_key:
                raise LLMError(f"{preset['key_env']} is empty in .env "
                               f"(create a key at {preset['key_page']})")
        else:
            api_key = "ollama"  # Ollama ignores the key but the library requires one
        self.base_url = base_url
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                     timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_SDK_RETRIES)
        self._new_style_params = False  # set True for models that reject max_tokens/temperature

    def _create(self, messages):
        if self._new_style_params:
            # Newer OpenAI reasoning models (e.g. the gpt-5 family) only accept
            # max_completion_tokens and the default temperature.
            return self._client.chat.completions.create(
                model=self.model, messages=messages,
                max_completion_tokens=max(config.LLM_MAX_TOKENS, 4000))
        return self._client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE)

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            try:
                resp = self._create(messages)
            except Exception as e:
                msg = str(e)
                if not self._new_style_params and ("max_completion_tokens" in msg or
                                                   ("temperature" in msg and "unsupported" in msg.lower())):
                    self._new_style_params = True
                    resp = self._create(messages)
                else:
                    raise
        except Exception as e:
            raise LLMError(f"{self.provider} call failed: {e}") from e
        return resp.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())


# Kept for backwards compatibility with earlier versions of this project.
OpenAIClient = OpenAICompatibleClient


# --------------------------------------------------------------- Mock
_INTENT_KEYWORDS = {
    "rental_agreement": ["agreement", "e-stamp", "estamp", "stamp paper", "notar"],
    "owner_unresponsive": ["not responding", "not picking", "no response", "unresponsive",
                           "owner ka", "call nahi", "phone nahi", "not replying", "ghost"],
    "deposit_refund": ["deposit", "token amount", "token money", "advance"],
    "packers_movers": ["packers", "movers", "shifting", "truck", "relocation", "damaged",
                       "broken", "toot", "samaan"],
    "home_services": ["cleaning", "painting", "painter", "plumber", "electrician", "repair",
                      "technician", "deep clean", "leak"],
    "subscription_plan": ["plan", "subscription", "relationship manager", "upgrade",
                          "freedom", "contacts left"],
    "payment_issue": ["rent payment", "upi", "credit card", "deducted", "debited",
                      "transaction", "not credited", "paisa kat"],
    "property_listing": ["listing", "listed", "post my property", "photos", "verification",
                         "deactivate", "ad "],
}
_NEG = ["frustrat", "worst", "ridiculous", "angry", "useless", "pathetic", "disappointed",
        "unacceptable", "third time", "fed up", "bekaar", "pareshan", "complaint",
        "consumer forum", "not happy", "waste"]
_POS = ["thank you so much", "great", "perfect", "appreciate", "awesome", "helpful",
        "bahut accha", "shukriya", "wonderful"]
_UNRESOLVED = ["escalat", "still not", "nothing has", "no update", "not resolved",
               "abhi tak nahi", "will raise", "cannot", "can't do"]
_PARTIAL = ["partial", "remaining", "rest of", "baaki", "will update by", "within 48",
            "within 24", "follow up", "callback"]
_RESOLVED = ["has been processed", "done", "resolved", "rescheduled", "credited",
             "issue is fixed", "sent to your", "confirmed", "ho gaya"]

_ACTIONS = {
    "rental_agreement": "Verify agreement status and share the corrected / e-stamped copy.",
    "owner_unresponsive": "Re-contact the owner and offer the customer alternate matching listings.",
    "deposit_refund": "Check refund eligibility with finance and confirm a refund timeline.",
    "packers_movers": "Raise a damage/delay claim with the moving partner and confirm compensation.",
    "home_services": "Confirm technician assignment and send the customer a revised slot.",
    "subscription_plan": "Review plan usage and decide on extension or refund per policy.",
    "payment_issue": "Reconcile the transaction with the payment gateway and update the customer.",
    "property_listing": "Check listing moderation status and fix or reactivate the listing.",
    "other": "Review the transcript manually and route to the right team.",
}


class MockClient(BaseClient):
    """Deterministic keyword heuristic that speaks the same JSON contract."""

    provider = "mock"

    def __init__(self, model: str):
        super().__init__(model)
        # Set MOCK_FAIL_RATE=0.1 to make 10% of first attempts return broken JSON,
        # which exercises the retry / validation path.
        self.fail_rate = float(os.getenv("MOCK_FAIL_RATE", "0"))
        self._rng = random.Random(7)

    def complete(self, system: str, user: str) -> str:
        if "PREVIOUS ATTEMPT FAILED" not in user and self._rng.random() < self.fail_rate:
            return '{"summary": "truncated output...'
        m = re.search(r"<transcript>(.*?)</transcript>", user, re.S)
        transcript = (m.group(1) if m else user).strip()
        return json.dumps(self._analyse(transcript))

    @staticmethod
    def _count(text: str, words: list[str]) -> int:
        return sum(text.count(w) for w in words)

    def _analyse(self, transcript: str) -> dict:
        low = transcript.lower()
        customer = " ".join(l for l in low.splitlines() if l.startswith("customer"))
        agent_lines = [l for l in low.splitlines() if l.startswith("agent")]
        agent = " ".join(agent_lines)
        tail = " ".join(low.splitlines()[-4:])

        scores = {k: self._count(customer or low, v) for k, v in _INTENT_KEYWORDS.items()}
        intent = max(scores, key=scores.get) if max(scores.values()) > 0 else "other"

        neg, pos = self._count(customer, _NEG), self._count(customer, _POS)
        sentiment = "negative" if neg > pos else "positive" if pos > 0 else "neutral"

        if self._count(tail, _UNRESOLVED):
            resolution = "unresolved"
        elif self._count(tail, _PARTIAL):
            resolution = "partially_resolved"
        elif self._count(tail, _RESOLVED):
            resolution = "resolved"
        else:
            resolution = "partially_resolved"

        priority = ("high" if sentiment == "negative" and resolution != "resolved"
                    else "medium" if resolution != "resolved" or sentiment == "negative"
                    else "low")

        qa = {
            "greeted": bool(re.search(r"\b(hello|hi|namaste|good (morning|afternoon|evening)|"
                                      r"welcome|thank you for calling)\b", agent_lines[0]
                                      if agent_lines else "")),
            "verified_details": bool(re.search(r"(registered|confirm your|verify|booking id|"
                                               r"order id|request id|mobile number)", agent)),
            "showed_empathy": bool(re.search(r"(sorry|apolog|understand|inconvenience|"
                                             r"samajh sakta|maafi)", agent)),
            "offered_resolution": bool(re.search(r"(i have|i will|i'll|we will|we'll|"
                                                 r"within \d+|by tomorrow|raised|processed|"
                                                 r"rescheduled|assigned)", agent)),
        }
        label = intent.replace("_", " ")
        summary = (f"Customer contacted support about: {label}. "
                   f"The conversation ended {resolution.replace('_', ' ')} "
                   f"with the customer sounding {sentiment}.")
        return {
            "summary": summary,
            "intent": intent,
            "sentiment": sentiment,
            "resolution_status": resolution,
            "priority": priority,
            "next_action": _ACTIONS[intent] if resolution != "resolved"
            else "No action needed; send a CSAT survey.",
            "agent_qa": qa,
        }


# --------------------------------------------------------------- factory
PROVIDERS = ["mock", "groq", "gemini", "ollama", "anthropic", "openai"]


def get_client(provider: str | None = None, model: str | None = None) -> BaseClient:
    provider = (provider or config.LLM_PROVIDER).lower().strip()
    model = model or config.LLM_MODEL or config.DEFAULT_MODELS.get(provider, "")
    if provider == "mock":
        return MockClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider in OPENAI_COMPATIBLE:
        return OpenAICompatibleClient(model, provider)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}' in .env. Use one of: {', '.join(PROVIDERS)}.")
