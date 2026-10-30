"""Mask personal data before a transcript is sent to an external LLM API.

Regex redaction is a first line of defence, not a guarantee: it will miss
names, addresses written in prose, and unusual formats. Production systems
usually add an NER model and keep the raw transcript out of third-party APIs
entirely unless a data-processing agreement covers it.
"""
import re

# Order matters: e-mail before UPI (both contain "@"), card before Aadhaar.
PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("UPI", re.compile(r"\b[\w.-]{2,}@[a-zA-Z]{2,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    # Indian mobile: optional +91 / 0, then 10 digits starting 6-9
    ("PHONE", re.compile(r"(?:\+91[ -]?|\b0)?\b[6-9]\d{4}[ -]?\d{5}\b")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, counts_per_type)."""
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts
