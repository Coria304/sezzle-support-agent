"""
Provider-agnostic LLM client.

LLM_PROVIDER=ollama   (default) — local Ollama, OpenAI-compatible API
LLM_PROVIDER=openai   — OpenAI API, requires OPENAI_API_KEY
LLM_PROVIDER=anthropic — Anthropic API, requires ANTHROPIC_API_KEY

Returns a unified (message, usage) tuple from chat().
"""
import os
import time

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()


class Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class Message:
    def __init__(self, content: str | None, tool_calls: list):
        self.content = content
        self.tool_calls = tool_calls  # list of ToolCall objects (provider-normalised)


class ToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.name = name
        self.arguments = arguments  # raw JSON string


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0,
    max_tokens: int = 1024,
    model: str | None = None,
) -> tuple[Message, Usage]:
    if PROVIDER == "anthropic":
        return _chat_anthropic(messages, tools, tool_choice, temperature, max_tokens, model)
    return _chat_openai_compat(messages, tools, tool_choice, temperature, max_tokens, model)


# ---------------------------------------------------------------------------
# OpenAI-compatible (Ollama + OpenAI)
# ---------------------------------------------------------------------------

_oai_client = None


def _get_oai_client():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        if PROVIDER == "openai":
            _oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            _oai_client = OpenAI(base_url=base, api_key="ollama")
    return _oai_client


def _default_model_oai() -> str:
    if PROVIDER == "openai":
        return os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    return os.environ.get("AGENT_MODEL", "llama3.2:3b")


def _chat_openai_compat(messages, tools, tool_choice, temperature, max_tokens, model):
    client = _get_oai_client()
    kwargs = dict(
        model=model or _default_model_oai(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    tcs = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments))

    usage = Usage(
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
    )
    return Message(content=msg.content, tool_calls=tcs), usage


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

_ant_client = None


def _get_ant_client():
    global _ant_client
    if _ant_client is None:
        import anthropic
        _ant_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _ant_client


def _default_model_ant() -> str:
    return os.environ.get("AGENT_MODEL", "claude-haiku-4-5")


def _chat_anthropic(messages, tools, tool_choice, temperature, max_tokens, model):
    client = _get_ant_client()

    system = ""
    ant_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            ant_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}],
            })
        else:
            ant_messages.append(m)

    ant_tools = []
    if tools:
        for t in tools:
            f = t["function"]
            ant_tools.append({"name": f["name"], "description": f["description"], "input_schema": f["parameters"]})

    kwargs = dict(
        model=model or _default_model_ant(),
        max_tokens=max_tokens,
        temperature=temperature,
        messages=ant_messages,
    )
    if system:
        kwargs["system"] = system
    if ant_tools:
        kwargs["tools"] = ant_tools

    resp = client.messages.create(**kwargs)

    content_text = None
    tcs = []
    for block in resp.content:
        if hasattr(block, "text"):
            content_text = block.text
        elif block.type == "tool_use":
            import json
            tcs.append(ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input)))

    usage = Usage(
        prompt_tokens=resp.usage.input_tokens,
        completion_tokens=resp.usage.output_tokens,
    )
    return Message(content=content_text, tool_calls=tcs), usage
