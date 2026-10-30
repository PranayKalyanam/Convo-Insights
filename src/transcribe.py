"""Optional: convert call recordings into a transcript CSV the pipeline can read.

    pip install faster-whisper
    python -m src.transcribe --audio-dir audio/ --out data/audio_conversations.csv
    python -m src.pipeline --input data/audio_conversations.csv

Uses faster-whisper (an efficient Whisper implementation that runs on CPU).
Limitations to mention in an interview:
  * Whisper does not separate speakers. Lines are labelled "Speaker" rather than
    Agent/Customer, so agent-QA judgements from audio are less reliable.
    Adding diarization (e.g. pyannote.audio) would fix this.
  * ASR errors (names, amounts, Hinglish words) flow into every downstream
    label. Compare intent accuracy on audio vs. text transcripts of the same
    calls to measure how much.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def transcribe_dir(audio_dir: str, model_size: str = "small", language: str | None = None) -> pd.DataFrame:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit("Install the optional dependency first: pip install faster-whisper") from e

    files = sorted(p for p in Path(audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"No audio files ({', '.join(sorted(AUDIO_EXT))}) found in {audio_dir}")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    rows = []
    for f in files:
        segments, info = model.transcribe(str(f), language=language, vad_filter=True)
        text = "\n".join(f"Speaker: {s.text.strip()}" for s in segments if s.text.strip())
        rows.append({"conversation_id": f.stem, "language": info.language, "transcript": text})
        print(f"{f.name}: {info.language} ({info.duration:.0f}s audio)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--out", default="data/audio_conversations.csv")
    ap.add_argument("--model-size", default="small", help="tiny, base, small, medium, large-v3")
    ap.add_argument("--language", default=None, help="force a language code, e.g. 'hi' or 'en'")
    a = ap.parse_args()
    df = transcribe_dir(a.audio_dir, a.model_size, a.language)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} transcripts to {a.out}")


if __name__ == "__main__":
    main()
