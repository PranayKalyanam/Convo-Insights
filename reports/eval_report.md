# Evaluation report

Generated 2026-10-10T18:00:32 on `C:\GitHub\convo-insights\data\test.csv` (100 conversations).
Extractor: **groq / openai/gpt-oss-120b**.

## LLM pipeline vs TF-IDF baseline

| Label | LLM accuracy | LLM macro-F1 | Baseline accuracy | Baseline macro-F1 |
|---|---|---|---|---|
| intent | 0.970 | 0.983 | 1.000 | 1.000 |
| sentiment | 0.930 | 0.948 | 0.910 | 0.897 |
| resolution_status | 0.820 | 0.839 | 0.950 | 0.957 |
| priority | 0.810 | 0.813 | n/a | n/a |

Invalid LLM outputs are counted as wrong answers. The baseline has no priority model because priority is derived from the other labels.

## Agent QA checklist agreement

| Check | Agreement with gold |
|---|---|
| greeted | 1.000 |
| verified_details | 1.000 |
| showed_empathy | 1.000 |
| offered_resolution | 0.948 |

## Intent by language

| Language | n | Accuracy | Macro-F1 |
|---|---|---|---|
| english | 69 | 0.957 | 0.977 |
| hinglish | 31 | 1.000 | 1.000 |

## Reliability and speed

- Invalid output rate after retries: 3.0%
- Conversations that needed a retry: 0.0%
- LLM latency per conversation: mean 11151.5 ms, p95 30956.3 ms
- Baseline latency per conversation: 1.109 ms

22 rows had at least one wrong label; see `reports/errors.csv`.
Summaries and next actions are not scored automatically: review a sample by hand and check that every claim is supported by the transcript.
