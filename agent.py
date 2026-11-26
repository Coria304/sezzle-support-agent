import json
import logging
import os
import re
import time

import llm as llm_client
from policies import load_all_policies
from tools import TOOL_SCHEMA, get_today, get_user_orders

logger = logging.getLogger(__name__)

_policies_text = None

# ---------------------------------------------------------------------------
# Tool schema (OpenAI format; user_id removed — auth is handled server-side)
# ---------------------------------------------------------------------------

_OAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_user_orders",
        "description": (
            "Fetch the authenticated user's account info and all their Sezzle orders, "
            "including installment schedules, payment statuses, refunds, and reschedule counts. "
            "Call this whenever the question is about a specific order, payment date, amount, "
            "refund status, or account state. Do NOT answer order-specific questions from memory."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# ---------------------------------------------------------------------------
# Pre-filter: topics that must force escalation (LLM still writes the answer)
# ---------------------------------------------------------------------------

# Matches clear hard-escalate intents. Intentionally narrow to avoid false positives:
#   - "how does the spending limit work?" → no match (generic policy question)
#   - "tell me my exact spending limit" → match
#   - "what is the dispute window?" → no match (policy question about process)
#   - "I want to file a dispute" → match (action request)
_FORCE_ESCALATE_RE = re.compile(
    r"\b(fraud|unauthorized)\b"
    r"|didn.t (place|make|order)\b|never (placed|received|arrived|shipped)\b"
    r"|\bstolen\b|\bhacked\b"
    r"|\bhardship\b|lost (my )?job|can.t (afford|pay|make) (a |my |next )?(payment|installment)"
    r"|\bexact (spending |credit )?limit\b|tell me my.{0,20}limit|specific decline reason"
    r"|\bfile a (dispute|chargeback|claim)\b|\bi want to (dispute|file)\b"
    r"|\border on my account i never",
    re.IGNORECASE,
)

# Output-violation checks (structural guardrails on the LLM's answer)
_LIMIT_DISCLOSURE_RE = re.compile(
    r"(your|the|exact|spending).{0,20}limit.{0,10}\$?\d|\$?\d.{0,10}(spending|credit) limit",
    re.IGNORECASE,
)
# Fee-waiver promise: only the one automatic waiver is allowed
_BAD_WAIVER_RE = re.compile(
    r"(I.ll|I will|we.ll|we will|can|will).{0,30}waive|waiver (is|has been) (approved|granted)",
    re.IGNORECASE,
)
_ALLOWED_WAIVER_RE = re.compile(
    r"(first|initial).{0,30}(missed|failed|late).{0,30}(payment|installment)"
    r"|(once per|one per).{0,10}(12 months|year)",
    re.IGNORECASE,
)

ESCALATION_RE = re.compile(
    r"escalat|human (agent|specialist|team)|connect you|hand(ing)? (you )?(off|over)|transfer",
    re.IGNORECASE,
)
ROUTE_TAG_RE = re.compile(r"<route>(policy|tool|both|escalate)</route>", re.IGNORECASE)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _get_policies() -> str:
    global _policies_text
    if _policies_text is None:
        _policies_text = load_all_policies()
    return _policies_text


_BASE_INSTRUCTIONS = """
[INSTRUCTIONS]
Decide what to do using this exact priority order:

1. ESCALATE (never call the tool first) if the question involves fraud, hardship,
   disputes, or requests for exact spending limits / specific decline reasons.
   Briefly explain what policy allows, then say you are connecting them to a human agent.

2. CALL get_user_orders if and only if the question requires this user's specific
   order data: payment dates, amounts, refund status, reschedule count, etc.
   Do NOT call the tool for general "how does X work" or policy questions.
   Examples that need the tool: "When is my next payment?", "Was my refund issued?"
   Examples that do NOT need the tool: "How does pay-in-4 work?", "Do you charge interest?",
   "What is the rescheduling policy?" — answer these directly from [POLICIES] above.

3. ANSWER from policy or tool data (or both). Be concise. Do not invent any numbers,
   dates, or statuses. Never state an exact spending limit. Never promise a fee waiver
   except the one automatic first-missed-payment waiver in the Failed Payments policy.

[ROUTING TAG]
Last line of your response, nothing after it — exactly one of:
<route>policy</route>    answered from policy only, no tool call
<route>tool</route>      answered using order data only
<route>both</route>      needed both order data and policy
<route>escalate</route>  escalated to a human agent
"""

_FORCE_ESCALATE_INSTRUCTIONS = """
[INSTRUCTIONS]
This question involves a topic that requires a human agent (fraud, hardship, dispute filing,
exact spending limits, or specific decline reasons). You must escalate.

Do NOT call any tools. Do NOT promise any outcome, waiver, or specific action.
You may briefly explain what Sezzle's policy says, then clearly state that you are
connecting the customer to a human agent.

Last line of your response, nothing after it:
<route>escalate</route>
"""

SYSTEM_TEMPLATE = """You are Sezzle's customer support assistant. Today is {today}.

[POLICIES]
{policies}
[/POLICIES]
{instructions}"""


def _build_system(today: str, force_escalate: bool = False) -> str:
    instructions = _FORCE_ESCALATE_INSTRUCTIONS if force_escalate else _BASE_INSTRUCTIONS
    return SYSTEM_TEMPLATE.format(today=today, policies=_get_policies(), instructions=instructions)


# ---------------------------------------------------------------------------
# Route extraction
# ---------------------------------------------------------------------------


def _extract_route(answer: str, tool_was_called: bool) -> tuple[str, str]:
    match = ROUTE_TAG_RE.search(answer)
    if match:
        route = match.group(1).lower()
        clean = answer[: match.start()].rstrip()

        # Structural override: tag cannot contradict observable behavior
        if tool_was_called and route == "policy":
            logger.warning("Route disagreement: model tagged 'policy' but tool was called → overriding to 'tool'")
            route = "tool"
        if not tool_was_called and route in ("tool", "both"):
            logger.warning("Route disagreement: model tagged '%s' but no tool was called → overriding to 'policy'", route)
            route = "policy"

        return route, clean

    # Fallback: infer from behavior
    if ESCALATION_RE.search(answer):
        return "escalate", answer
    if tool_was_called:
        return "tool", answer
    return "policy", answer


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_answer(answer: str) -> str | None:
    """Return a violation description, or None if the answer is clean."""
    if _LIMIT_DISCLOSURE_RE.search(answer):
        return "answer disclosed an exact spending limit"
    if _BAD_WAIVER_RE.search(answer) and not _ALLOWED_WAIVER_RE.search(answer):
        return "answer promised a fee waiver beyond the allowed automatic one"
    return None


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

_MAX_TOOL_CALLS = 1
_MAX_RETRIES = 3


def _llm_call_with_retry(messages, tools, tool_choice, today):
    """Call the LLM with exponential backoff on transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return llm_client.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0,
            )
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, _MAX_RETRIES, e, wait)
            time.sleep(wait)


def answer_question(user_id: str, question: str) -> tuple[str, str]:
    """Call the LLM and return (route, answer)."""
    today = get_today()
    force_escalate = bool(_FORCE_ESCALATE_RE.search(question))
    system = _build_system(today, force_escalate=force_escalate)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    # When forcing escalation, disable the tool entirely
    tools = None if force_escalate else [_OAI_TOOL]
    tool_was_called = False
    tool_calls_made = 0

    for _ in range(_MAX_TOOL_CALLS + 1):  # at most 1 tool call + 1 final answer
        tool_choice = "none" if (tool_calls_made >= _MAX_TOOL_CALLS or not tools) else "auto"
        msg, usage = _llm_call_with_retry(messages, tools, tool_choice, today)

        if msg.tool_calls and tool_calls_made < _MAX_TOOL_CALLS:
            tool_was_called = True
            tool_calls_made += 1

            # Reconstruct the assistant message in OpenAI format for history
            oai_tool_calls = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in msg.tool_calls
            ]
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": oai_tool_calls})

            for tc in msg.tool_calls:
                if tc.name == "get_user_orders":
                    try:
                        # AUTHORIZATION: always use the authenticated user_id; ignore any
                        # user_id the model may have attempted to pass in arguments.
                        args = json.loads(tc.arguments) if tc.arguments else {}
                        attempted_uid = args.get("user_id")
                        if attempted_uid and attempted_uid != user_id:
                            logger.warning(
                                "AUTHORIZATION: model attempted to fetch user '%s' but authenticated user is '%s' — blocked",
                                attempted_uid, user_id,
                            )
                        result = get_user_orders(user_id)
                        content = json.dumps(result)
                    except json.JSONDecodeError as e:
                        logger.warning("Malformed tool arguments: %s", e)
                        content = json.dumps({"error": "malformed arguments"})
                    except Exception as e:
                        logger.warning("Tool execution error: %s", e)
                        content = json.dumps({"error": str(e)})
                else:
                    logger.warning("Unknown tool called: %s", tc.name)
                    content = json.dumps({"error": f"unknown tool '{tc.name}'"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })
        else:
            # Final answer
            text = msg.content or ""
            route, clean_answer = _extract_route(text, tool_was_called)

            # Structural override: if force_escalate was active, the answer must be escalate
            # regardless of what tag the model emitted (x07 failure class)
            if force_escalate and route != "escalate":
                logger.warning("Route disagreement: force_escalate=True but model tagged '%s' → overriding to 'escalate'", route)
                route = "escalate"

            # Output validation — regenerate once on violation
            violation = _validate_answer(clean_answer)
            if violation:
                logger.warning("Output violation (%s) — regenerating", violation)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous answer violated a rule: {violation}. "
                        "Please rewrite your answer without that violation. "
                        "Remember: never state an exact spending limit, and never promise "
                        "a fee waiver except the one automatic first-missed-payment waiver."
                    ),
                })
                msg2, _ = _llm_call_with_retry(messages, None, "none", today)
                text2 = msg2.content or ""
                route, clean_answer = _extract_route(text2, tool_was_called)
                if _validate_answer(clean_answer):
                    logger.warning("Output still violates after regeneration — escalating")
                    return "escalate", (
                        "I'm unable to provide that information. Please contact our support team directly."
                    )

            return route, clean_answer

    # Should not reach here
    return "escalate", "I was unable to process your request. Please contact our support team."
