"""Check that the LLM in .env is set up correctly, in plain language.

    python -m src.check_llm            # test the provider configured in .env
    python -m src.check_llm --models   # also list the model names your key can use

It sends one short test conversation, shows what came back, and if something
fails, explains the most likely cause and how to fix it.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config
from .extractor import extract
from .llm_client import KEY_PAGES, LLMError, get_client

TEST_TRANSCRIPT = (
    "Agent: Hello, thank you for calling RentEasy support, this is Priya.\n"
    "Customer: I paid my rent of Rs 18000 by UPI yesterday but the owner says it is not "
    "credited. Money is deducted from my account. This is the second time I am calling.\n"
    "Agent: I'm sorry for the trouble. Can you confirm your registered mobile number?\n"
    "Customer: 9876543210.\n"
    "Agent: Thank you. I have raised a ticket with the payment gateway; you will get an "
    "update within 24 hours.\n"
    "Customer: Okay, but please make sure it is fixed."
)


def explain(error: str, provider: str, model: str) -> str:
    """Turn a raw provider error into a likely cause and a fix."""
    e = error.lower()
    key_page = KEY_PAGES.get(provider)
    if "is empty in .env" in e or "unknown llm_provider" in e:
        return error
    if any(s in e for s in ["401", "invalid api key", "invalid_api_key", "authentication",
                            "unauthorized", "api key not valid", "permission_denied"]):
        return ("The API key was rejected. Copy it again from "
                f"{key_page or 'your provider dashboard'} into .env (no quotes, no spaces), "
                "save the file, and run this check again.")
    if any(s in e for s in ["404", "model_not_found", "does not exist", "not found",
                            "decommissioned", "no such model", "is not supported"]):
        hint = ("For Ollama, download it first: ollama pull " + model
                if provider == "ollama" else
                "Run `python -m src.check_llm --models` to see valid names, then set LLM_MODEL in .env.")
        return f"The model name '{model}' is not available for this key. {hint}"
    if any(s in e for s in ["429", "rate limit", "rate_limit", "quota", "resource_exhausted",
                            "too many requests"]):
        return ("You hit the provider's rate limit or quota. Wait a minute and retry; for batch "
                "jobs set LLM_WORKERS=1 in .env. If it says quota or billing, check your plan "
                "or credits on the provider's website.")
    if any(s in e for s in ["credit balance", "insufficient", "billing", "payment"]):
        return "Your account has no credit. Add billing on the provider's website or use a free provider (groq, gemini, ollama)."
    if any(s in e for s in ["connection", "connect", "timed out", "timeout", "name resolution",
                            "network", "refused"]):
        if provider == "ollama":
            return ("Could not reach Ollama at http://localhost:11434. Make sure the Ollama app is "
                    "installed and running (open it, or run `ollama serve` in another terminal).")
        return "Could not reach the provider. Check your internet connection, VPN or firewall, then retry."
    return "Unrecognised error. Read the message above; searching for it usually finds the fix."


def run_check(list_models: bool = False) -> bool:
    provider, model = config.LLM_PROVIDER, config.active_model()
    print(f"Provider : {provider}")
    print(f"Model    : {model}")

    if provider == "mock":
        print("\nYou are using the offline mock, which is NOT an LLM. To use a real model, set "
              "LLM_PROVIDER in .env\n(groq, gemini, ollama, anthropic or openai) and add its key. "
              "See the README section 'Using an LLM'.")

    try:
        client = get_client()
    except LLMError as e:
        print(f"\nSetup problem: {explain(str(e), provider, model)}")
        return False

    if list_models and provider != "mock":
        try:
            names = client.list_models()
            print(f"\nModels available to you ({len(names)}):")
            for n in names:
                print(f"  {n}")
        except Exception as e:
            print(f"\nCould not list models: {e}\n-> {explain(str(e), provider, model)}")

    print("\nSending a short test conversation...")
    t0 = time.perf_counter()
    res = extract(TEST_TRANSCRIPT, client)
    secs = time.perf_counter() - t0

    if res.valid:
        ins = res.insight
        print(f"OK: valid answer in {secs:.1f}s ({res.attempts} attempt(s)).\n")
        print(f"  summary     : {ins.summary}")
        print(f"  intent      : {ins.intent}   (expected: payment_issue)")
        print(f"  sentiment   : {ins.sentiment}")
        print(f"  resolution  : {ins.resolution_status}")
        print(f"  priority    : {ins.priority}")
        print(f"  next action : {ins.next_action}")
        print(f"  agent QA    : {ins.agent_qa.score:.0f}/100")
        print(f"  masked      : {res.pii_counts}  (the phone number never left your computer)")
        if provider == "mock":
            print("\nThe pipeline works, but with the offline mock. Set a real provider to use an LLM.")
        else:
            print("\nThe LLM is working. Restart the API server if it is running, so it picks up .env.")
        return True

    print(f"FAILED after {res.attempts} attempt(s) in {secs:.1f}s.\n")
    print(f"Error: {res.error}\n")
    if res.raw_output:
        print(f"What the model returned:\n{res.raw_output[:1000]}\n")
        print("-> The model answered but not in the required JSON format. Smaller models do "
              "this more often. Try a larger model (LLM_MODEL in .env) or raise MAX_RETRIES.")
    else:
        print(f"-> {explain(res.error or '', provider, model)}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="list the model names your key can use")
    a = ap.parse_args()
    sys.exit(0 if run_check(a.models) else 1)


if __name__ == "__main__":
    main()
