# Sezzle Support Agent

LLM-powered support agent that answers shopper questions, looks up order data, and escalates when required.

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

# 4. Run against the visible test cases
python run_cases.py CASES_-_golden_visible.jsonl answers.jsonl

# 5. Check the output
cat answers.jsonl
```

No API key required. Ollama runs locally on `http://localhost:11434`.

To swap models, set the environment variable before running:
```bash
AGENT_MODEL=llama3.2 python run_cases.py ...
```

Each output line: `{"id": "v01", "route": "tool", "answer": "Your next payment..."}`

## Architecture

```
run_cases.py          CLI contract — reads JSONL in, writes JSONL out
agent.py              LLM call, tool loop, route extraction
tools.py              get_user_orders() — reads DATA-orders.json
policies.py           Loads all 12 POLICIES/*.md into the system prompt
```

One LLM call per question (default: `llama3.2:3b` via Ollama, OpenAI-compatible API). The model receives all 12 policy documents in the system prompt plus one tool (`get_user_orders`). A deterministic pre-filter intercepts clear escalation cases (fraud, hardship, spending-limit requests) before the LLM loop — the brief explicitly allows "deterministic logic around the model." For the remaining questions, the model decides whether to call the tool and tags its own route classification (`policy | tool | both | escalate`).

## What's honestly unfinished

- **Eval harness:** There's no automated regex checker against `must_include` / `must_not_include`. After the first run, failures were diagnosed by reading the output manually.
- **Retry / error handling:** If the API call fails, the script crashes. Production would need retries with backoff.
- **Multi-turn context:** Each question is stateless. A real support session would carry conversation history.
- **Cost optimization:** At 100k/day the current prompt is too expensive. See DR-3 in DECISIONS.md for the two-call pipeline that meets budget.

## Files

| File | Purpose |
|------|---------|
| `run_cases.py` | The graded contract |
| `agent.py` | Core LLM logic |
| `tools.py` | Order data tool |
| `policies.py` | Policy loader |
| `DECISIONS.md` | 3 architecture decision records |
| `PROMPTS.md` | Build + system prompt log |
| `DATA-orders.json` | Frozen order data (today = 2026-07-01) |
| `POLICIES/` | 12 Sezzle policy documents |
| `CASES_-_golden_visible.jsonl` | 10 labeled test cases |
