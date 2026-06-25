# Travel Assistant

A CLI travel assistant demonstrating LLM conversation quality with Pydantic and two external APIs. Uses DeepSeek as the LLM backend (pay-per-use cloud).

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env   # Mac/Linux
copy .env.example .env  # Windows Command Prompt

# 4. Open .env and set your DeepSeek API key:
#    DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
#    Get a key at: platform.deepseek.com → API keys → Create
```

---

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | _(required)_ | DeepSeek API key. Pay-per-use at [platform.deepseek.com](https://platform.deepseek.com). |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model. Comma-separated list for automatic rate-limit fallback. |
| `DEBUG` | `0` | Set to `1` to print full tracebacks on errors. |

> **Token warning:** every conversation turn makes two LLM calls (intent extraction + response generation). Token usage accumulates quickly during long conversations or `/eval` runs. Monitor your usage at platform.deepseek.com.

Attraction data comes from the **Overpass API** (OpenStreetMap) — no API key required.

---

## Run

```bash
python main.py
```

### Commands

| Command | Action |
|---|---|
| `/reset` | Reset conversation and trip context |
| `/state` | Show current structured trip context (JSON) |
| `/eval` | Run evaluator scenarios |
| `/save` | Save current conversation transcript to `transcripts/` |
| `/exit` | Quit |

---

## Tests

```bash
python -m pytest tests/ -v
```

The test suite is split into two tiers:

**Offline (no LLM key needed):** All tests in `tests/test_assistant.py` and most tests in `tests/test_suite.py` (T1–T16, T21–T23, T25, and all Layer 4 tests) run entirely offline using fake backends. These cover validators, context management, tool routing, evaluator deterministic checks, schema echo detection, hard override logic, and time/price pattern matching.

**Live LLM:** `TestDeepSeekBackendSmoke` (T28–T31) and `TestEvaluatorPenalizesVerbosity` (T24) require a valid `DEEPSEEK_API_KEY`. They are automatically skipped when the key is absent.

---

## Project structure

```
main.py            — CLI entry point
models.py          — All Pydantic data contracts
prompts.py         — All prompt strings and template functions
context.py         — DeepSeek backend, ContextManager, IntentExtractor, ResponseGenerator
tools.py           — Open-Meteo (weather) and Overpass/OpenStreetMap (attractions) tool calls
services/          — Overpass API async HTTP service (used by tools.py)
evaluator.py       — Evaluation runner (deterministic checks + LLM judge)
scenarios.py       — 22 fixed evaluation scenarios (10 categories)
tests/             — pytest test suite (unit-tier, mostly offline)
transcripts/       — Saved conversation transcripts
evaluator_output/  — Saved evaluator JSON results
PROMPT_ENGINEERING_NOTES.md  — Prompt design rationale
```

---

## What I did not build

- **Booking, flights, hotels, payments** — out of scope per assignment brief.
- **Web UI or REST API** — CLI only per assignment brief.
- **Database or session persistence** — context lives in-memory for the duration of the process.
- **Extra tools beyond weather + attractions** — no venue-detail lookup, no search.
- **Authentication or user accounts** — not required.
- **Deployment configuration** — not required.
