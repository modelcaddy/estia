#!/usr/bin/env python3
"""A tool-calling loop: the model asks, your code runs a local function, the
model answers with the result.

What it shows:
  - declaring a tool with an OpenAI tool schema
  - reading `tool_calls` from the response (`finish_reason` is `tool_calls`)
  - running the function locally and giving the result back to the model
  - a bounded loop, so a model that keeps calling tools cannot spin forever

Scopes: generate

    export ESTIA_TOKEN=estia_...      # estia token new tools --scopes generate
    python tools.py                              # "What time is it in Tokyo?"
    python tools.py "Is it morning in Lisbon right now?"

The function runs on your machine, never on the engine. Only tools you
declare can be called, and your code decides whether to run each call.
"""

import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import openai
from openai import OpenAI

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
TOKEN = os.environ.get("ESTIA_TOKEN", "")
MODEL = os.environ.get("ESTIA_MODEL", "fast")
MAX_STEPS = 4


def current_time(timezone: str) -> dict:
    """The local function the model may call."""
    try:
        now = datetime.datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"unknown time zone {timezone!r}; use an IANA name such as Europe/Athens"}
    return {"timezone": timezone, "time": now.strftime("%H:%M"), "weekday": now.strftime("%A"), "date": now.date().isoformat()}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "The current local time, weekday and date in a time zone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA time zone, for example Asia/Tokyo"}},
                "required": ["timezone"],
            },
        },
    }
]
FUNCTIONS = {"current_time": current_time}


def explain(e: Exception) -> str:
    if isinstance(e, openai.APIConnectionError):
        return f"cannot reach the engine at {URL}. Is `estia serve` running? Set ESTIA_URL if it listens elsewhere."
    if isinstance(e, openai.APIStatusError):
        message = e.body.get("message") if isinstance(e.body, dict) else e.message
        text = f"the engine answered {e.status_code}: {message}"
        if e.request_id:
            text += f" (request id {e.request_id})"
        if e.status_code == 401:
            text += "\nCheck ESTIA_TOKEN, or mint a token: estia token new tools --scopes generate"
        return text
    return str(e)


def run(client: OpenAI, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        r = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0, max_tokens=300)
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        # Keep the model's call in the history, as OpenAI clients do.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = FUNCTIONS.get(name)
            result = fn(**args) if fn else {"error": f"no tool named {name!r}"}
            print(f"[tool] {name}({json.dumps(args)}) -> {json.dumps(result)}", file=sys.stderr)
            # The result goes back as a tool message that names the call it
            # answers. The model's chat template renders it after the
            # assistant turn above.
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": json.dumps(result)})
    return "(stopped: the model kept calling tools)"


def main() -> int:
    if not TOKEN:
        print(
            "ESTIA_TOKEN is not set. On the engine's machine run\n"
            "  estia token new tools --scopes generate\n"
            "and export the token it prints: export ESTIA_TOKEN=estia_...",
            file=sys.stderr,
        )
        return 2
    question = " ".join(sys.argv[1:]) or "What time is it in Tokyo?"
    client = OpenAI(base_url=f"{URL}/v1", api_key=TOKEN, max_retries=0)
    try:
        print(run(client, question))
    except openai.OpenAIError as e:
        print(f"error: {explain(e)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
