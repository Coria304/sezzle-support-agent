import json
import os
import re

from openai import OpenAI

from policies import load_all_policies
from tools import TOOL_SCHEMA, get_today, get_user_orders

_client = None
_policies_text = None

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("AGENT_MODEL", "llama3.2:3b")

# Pre-filter: topics the LLM must never handle alone — escalate before any tool call.
# "Deterministic logic around the model is welcome engineering" (brief, design constraint).
_ESCALATE_PRE = re.compile(
    r"fraud|unauthorized|didn.t (place|make|order)|never placed|stolen|hacked"
    r"|hardship|lost (my )?job|can.t (afford|pay|make)"
    r"|exact spending limit|why (was|is) (my )?.*declined|specific decline reason"
    r"|dispute|chargeback|file a claim",
    re.IGNORECASE,
)

ESCALATION_RE = re.compile(
    r"escalat|human (agent|specialist|team)|connect you|hand(ing)? (you )?(off|over)|transfer",
    re.IGNORECASE,
)
ROUTE_TAG_RE = re.compile(r"<route>(policy|tool|both|escalate)</route>", re.IGNORECASE)

# OpenAI-format tool schema (Ollama uses this)
_OAI_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_SCHEMA["name"],
        "description": TOOL_SCHEMA["description"],
        "parameters": TOOL_SCHEMA["input_schema"],
    },
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _client


def _get_policies() -> str:
    global _policies_text
    if _policies_text is None:
        _policies_text = load_all_policies()
    return _policies_text


SYSTEM_TEMPLATE = """You are Sezzle's customer support assistant. Today is {today}.

[POLICIES]
{policies}
[/POLICIES]

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


def _build_system(today: str) -> str:
    return SYSTEM_TEMPLATE.format(today=today, policies=_get_policies())


def _extract_route(answer: str, tool_was_called: bool) -> tuple[str, str]:
    match = ROUTE_TAG_RE.search(answer)
    if match:
        route = match.group(1).lower()
        clean = answer[: match.start()].rstrip()
        return route, clean

    if ESCALATION_RE.search(answer):
        return "escalate", answer
    if tool_was_called:
        return "tool", answer
    return "policy", answer


def _pre_escalate(question: str) -> str | None:
    """Return a canned escalation answer if the question matches a hard-escalate topic,
    before any LLM or tool call. The LLM still writes the full response — this only
    determines whether we should skip tool calling and route directly."""
    if _ESCALATE_PRE.search(question):
        return (
            "I'm sorry to hear you're experiencing this. Based on Sezzle's policies, "
            "this type of issue requires direct assistance from our support team. "
            "I'm connecting you with a human agent who can investigate and take action "
            "on your behalf. Please do not share sensitive account credentials in chat."
        )
    return None


def answer_question(user_id: str, question: str) -> tuple[str, str]:
    """Call the LLM and return (route, answer)."""
    # Pre-filter: catch hard-escalate topics before any LLM or tool call
    pre = _pre_escalate(question)
    if pre is not None:
        return "escalate", pre

    client = _get_client()
    today = get_today()
    system = _build_system(today)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    tool_was_called = False

    # Agentic loop: may call get_user_orders once, then produce final answer
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            tools=[_OAI_TOOL],
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            tool_was_called = True
            messages.append(msg)

            for tc in msg.tool_calls:
                if tc.function.name == "get_user_orders":
                    args = json.loads(tc.function.arguments)
                    uid = args.get("user_id", user_id)
                    result = get_user_orders(uid)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
        else:
            text = msg.content or ""
            route, clean_answer = _extract_route(text, tool_was_called)
            return route, clean_answer
