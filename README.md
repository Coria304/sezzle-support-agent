# Sezzle Support Agent

LLM-powered support agent that answers shopper questions, looks up order data, and escalates when required.

> **Note:** This repository includes post-submission improvements made after the original deadline. The original submission is tagged `submission-v1`; post-submission work is on the `post-submission-iteration` branch.

## How to run

Requires [Ollama](https://ollama.com) running locally with `llama3.2:3b` pulled.

```bash
# 1. Install Ollama and pull the model (one-time setup)
ollama pull llama3.2:3b

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run against the visible test cases (graded contract)
python run_cases.py CASES_-_golden_visible.jsonl answers.jsonl

# 5. Run the full eval harness (route + content checks, latency)
python eval.py CASES_-_golden_visible.jsonl --label my-run
```

No API key required. Ollama runs locally on `http://localhost:11434`.

To swap providers or models:
```bash
# Larger local model
AGENT_MODEL=llama3.2 python run_cases.py ...

# OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... AGENT_MODEL=gpt-4o-mini python run_cases.py ...

# Anthropic
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python run_cases.py ...
```

Each output line: `{"id": "v01", "route": "tool", "answer": "Your next payment..."}`

## Architecture

```
run_cases.py     CLI contract — reads JSONL in, writes JSONL out
eval.py          Eval harness — route + must_include/must_not_include checks, latency
agent.py         LLM loop, authorization, pre-filter, output validation, route extraction
llm.py           Provider abstraction (Ollama / OpenAI / Anthropic)
tools.py         get_user_orders() — reads DATA-orders.json
policies.py      Loads all 12 POLICIES/*.md into the system prompt
cases/           extra.jsonl — hostile + edge cases (post-submission, clearly labeled)
runs/            Eval outputs with per-case pass/fail and latency
```

One LLM call per question (default: `llama3.2:3b` via Ollama). The model receives all 12
policy documents in the system prompt plus one tool (`get_user_orders`).

**Pre-filter (`_FORCE_ESCALATE_RE`):** keyword regex on the question that, when matched,
disables the tool and injects escalation instructions into the system prompt. The LLM still
writes the answer — the filter restricts capability, not output. Covers fraud, hardship,
exact spending limits, dispute filing, and unshipped/unreceived orders.

**Authorization:** `get_user_orders` always fetches the authenticated `user_id`; any
`user_id` the model attempts to pass in tool arguments is ignored and logged.

**Output validation:** post-checks for spending-limit disclosure and unauthorized fee-waiver
promises; regenerates once on violation, escalates if still violated.

## Eval results (post-submission, llama3.2:3b)

| Run | Cases | Passed | Routes correct | Notes |
|-----|-------|--------|----------------|-------|
| run1-baseline | visible (10) | 3/10 | 5/10 | Full route + content checks via eval.py |
| hostile | extra (8) | see runs/ | — | Authorization, injection, paraphrase, false-positive |

The 7 visible failures split into two categories:
- **Route mis-classification (4 cases):** model calls tool on pure policy questions — model-size limit of llama3.2:3b
- **Content gaps (3 cases):** route correct but answer omits required policy details — small model truncates

The same architecture on a larger model (llama3.2 8B, Haiku, GPT-4o-mini) is expected to score 8–10/10; not measured due to no API key.

## What's honestly unfinished

- **Stronger model run:** no API key available; `llm.py` supports Anthropic/OpenAI but comparison run is labeled "not measured"
- **Multi-turn context:** each question is stateless; a real support session would carry conversation history
- **Cost optimization:** at 100k/day the current prompt is too expensive — see DR-3 in DECISIONS.md for the two-call pipeline that meets budget; p95 latency is assumption, not measured
- **Streaming:** `run_cases.py` buffers the full answer; production would stream

## Files

| File | Purpose |
|------|---------|
| `run_cases.py` | The graded contract |
| `eval.py` | Full eval harness (post-submission) |
| `agent.py` | Core LLM logic |
| `llm.py` | Provider abstraction (post-submission) |
| `tools.py` | Order data tool |
| `policies.py` | Policy loader |
| `DECISIONS.md` | 3 architecture decision records |
| `PROMPTS.md` | Build + system prompt log |
| `DATA-orders.json` | Frozen order data (today = 2026-07-01) |
| `POLICIES/` | 12 Sezzle policy documents |
| `CASES_-_golden_visible.jsonl` | 10 labeled test cases |
| `cases/extra.jsonl` | 8 hostile + edge cases (post-submission) |
| `runs/` | Eval run outputs |
