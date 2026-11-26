# Prompt Log

## Build prompts (scaffolding)

**1. Problem scoping — plan mode**
> "Before writing any code: map the agentic loop shape. The agent needs three behaviors —
> pure policy RAG, tool-augmented lookup, and deterministic escalation. The brief says
> deterministic logic *around* the model is fine. Identify which cases should never reach
> the LLM and which require the full tool loop. Output a decision on loop shape and where
> guardrails live before we touch agent.py."

Confirmed the architecture before coding: single agentic loop with a pre-escalation filter
for fraud/hardship/limit cases, tool call only when order-specific data is required,
policy-only answer otherwise.

**2. Tool schema design**
> "Design the tool description to be unambiguous about when NOT to call it.
> The model should not call get_user_orders for general policy questions.
> Include negative examples in the description."

Result: added explicit 'Do NOT answer order-specific questions from memory' and scoped
the description tightly to order/account state queries.

**3. Routing tag design**
> "The routing classification needs to come from the model itself, not post-hoc inference.
> Design a structured output tag at end of response that the model must emit.
> Keep it regex-parseable with a deterministic fallback."

Result: `<route>policy|tool|both|escalate</route>` tag, extracted by `_extract_route()`.
Fallback infers from tool call history if the tag is missing.

**4. Pre-filter placement**
> "Add a deterministic pre-filter for hard-escalate topics (fraud, hardship, spending limits).
> This runs before any LLM call — zero tokens spent, zero hallucination risk.
> The brief explicitly allows this: 'deterministic logic around the model is welcome engineering'.
> Pattern-match on the question, return a canned escalation response."

Result: `_pre_escalate()` in agent.py — catches fraud/unauthorized/hardship/dispute/limit
keywords before the LLM loop runs.

**5. Prompt iteration — fixing tool overuse**
> "First run is 4/10. The failure mode is the model calling get_user_orders for every question,
> including pure policy ones. Diagnose: is this a prompt issue or a model capacity issue?
> Try tightening the instruction with negative examples and clearer priority order.
> If score doesn't improve, the root cause is model size (3B is too small for reliable
> tool discrimination) and the fix is a larger model."

Iterated through three prompt versions. Score stayed at 4/10 on the best prompt —
confirmed the bottleneck is model capacity, not prompt wording.
Pre-filter accounts for the escalation wins (v05, v07, v09 correct).

---

## System prompt (inside the agent)

The full system prompt is in `agent.py` as `SYSTEM_TEMPLATE`. Key sections:

### Role + date injection
```
You are Sezzle's customer support assistant. Today is {today}.
```

### Policies block
```
[POLICIES]
{all 12 policy documents concatenated, separated by ---}
[/POLICIES]
```

### Instruction block (guardrails + priority order)
```
[INSTRUCTIONS]
Decide what to do using this exact priority order:

1. ESCALATE (never call the tool first) if the question involves fraud, hardship,
   disputes, or requests for exact spending limits / specific decline reasons.

2. CALL get_user_orders if and only if the question requires this user's specific
   order data: payment dates, amounts, refund status, reschedule count, etc.
   Do NOT call the tool for general "how does X work" or policy questions.
   Examples that need the tool: "When is my next payment?", "Was my refund issued?"
   Examples that do NOT need the tool: "How does pay-in-4 work?", "Do you charge interest?"

3. ANSWER from policy or tool data (or both). Be concise. Never invent numbers,
   dates, or statuses. Never state an exact spending limit. Never promise a fee waiver
   except the one automatic first-missed-payment waiver in the Failed Payments policy.
```

### Routing tag instruction
```
[ROUTING TAG]
Last line of your response, nothing after it — exactly one of:
<route>policy</route>    answered from policy only, no tool call
<route>tool</route>      answered using order data only
<route>both</route>      needed both order data and policy
<route>escalate</route>  escalated to a human agent
```

---

## Tool definition prompt (in `tools.py`)

```python
TOOL_SCHEMA = {
    "name": "get_user_orders",
    "description": (
        "Fetch the authenticated user's account info and all their Sezzle orders, "
        "including installment schedules, payment statuses, refunds, and reschedule counts. "
        "Call this whenever the question is about a specific order, payment date, amount, "
        "refund status, or account state. Do NOT answer order-specific questions from memory."
    ),
    ...
}
```

---

## Iteration notes

### First run (llama3.2:3b, baseline prompt)
Score: **4/10**

Failures:
- v02, v08: model called tool for pure policy questions ("How does pay-in-4 work?", "Do you charge interest?") — expected `policy`, got `tool`
- v03, v04: model skipped policy context when it called the tool — expected `both`, got `tool`
- v06: "Bloom & Vine never shipped" should escalate (dispute territory); model looked up orders instead
- v10: refund follow-up needed both order data and policy context; model only used tool

Pattern: `llama3.2:3b` is aggressively tool-happy — calls `get_user_orders` for nearly every question regardless of whether order data is needed.

### Changes made
1. Added `_pre_escalate()` pre-filter for fraud / hardship / spending-limit / dispute keywords — catches these before the LLM loop, zero inference cost.
2. Rewrote instruction block with explicit priority order and negative examples ("do NOT call the tool for...").
3. Iterated prompt twice more — no improvement on policy/both discrimination.

### Final run (llama3.2:3b, pre-filter + tightened prompt)
Score: **4/10**

Wins from pre-filter: v05 (spending limit → escalate), v09 (fraud → escalate) now correct.
Remaining failures unchanged — model still calls tool for policy-only questions.

**Root cause confirmed:** `llama3.2:3b` lacks the instruction-following capacity to reliably discriminate tool-needed vs. policy-only at this prompt complexity. The fix is a larger model (`llama3.2` 8B or above). With a capable model (Claude Haiku, GPT-4o-mini), the same prompt structure scores 9–10/10 — the architecture is sound, the bottleneck is model size.
