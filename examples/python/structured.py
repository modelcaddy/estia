#!/usr/bin/env python3
"""Pull structured data out of free text with a JSON Schema.

What it shows:
  - `response_format` with `json_schema`: the model is held to the schema
    (llama.cpp constrains decoding to it; on MLX the engine shows it to the
    model in the system prompt), and the engine checks the output against
    it, repairs common defects and retries once
  - `x_estia.repaired` and `x_estia.repairs`: what the engine had to fix
  - the typed error: output that still fails the schema is a 422 with type
    `invalid_request_error`, raised by the SDK as UnprocessableEntityError

Scopes: generate

    export ESTIA_TOKEN=estia_...      # estia token new extract --scopes generate
    python structured.py                          # extracts from a built-in note
    python structured.py "Lunch with Ana on Friday at 13:00 at Kima, bring the contract."
    python structured.py --impossible             # a schema no answer can meet: shows the 422
"""

import json
import os
import sys

import openai
from openai import OpenAI

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
TOKEN = os.environ.get("ESTIA_TOKEN", "")
MODEL = os.environ.get("ESTIA_MODEL", "fast")

SAMPLE = (
    "Hi, it's Maria Papadopoulou from Acme. Can we meet on 14 October at 10:30 "
    "at your office to go over the renewal? Please bring last year's invoice."
)

# The schema is the contract between your code and the model. Keep it small:
# required fields you will read, enums where the answer is a choice.
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string", "description": "as written in the text"},
        "time": {"type": "string"},
        "place": {"type": "string"},
        "people": {"type": "array", "items": {"type": "string"}},
        "todo": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string", "enum": ["meeting", "call", "deadline", "other"]},
    },
    "required": ["title", "date", "people", "kind"],
    "additionalProperties": False,
}

# Asks for an integer that is both >= 1000 and <= 999. No output can pass, so
# the engine retries once and then answers 422.
IMPOSSIBLE_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 1000, "maximum": 999}},
    "required": ["n"],
}


def explain(e: Exception) -> str:
    if isinstance(e, openai.APIConnectionError):
        return f"cannot reach the engine at {URL}. Is `estia serve` running? Set ESTIA_URL if it listens elsewhere."
    if isinstance(e, openai.APIStatusError):
        message = e.body.get("message") if isinstance(e.body, dict) else e.message
        text = f"the engine answered {e.status_code}: {message}"
        if e.request_id:
            text += f" (request id {e.request_id})"
        if e.status_code == 401:
            text += "\nCheck ESTIA_TOKEN, or mint a token: estia token new extract --scopes generate"
        return text
    return str(e)


def extract(client: OpenAI, text: str, schema: dict) -> dict:
    # Say what to do; the schema itself travels in response_format and the
    # engine makes sure the model sees it.
    system = "Extract the event described in the user's text."
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "event", "schema": schema}},
        temperature=0,
        max_tokens=400,
    )
    extra = (r.model_extra or {}).get("x_estia") or {}
    if extra.get("repaired"):
        print(f"(the engine repaired the output: {', '.join(extra.get('repairs') or [])})", file=sys.stderr)
    # The content is JSON that already validated against the schema.
    return json.loads(r.choices[0].message.content)


def main() -> int:
    if not TOKEN:
        print(
            "ESTIA_TOKEN is not set. On the engine's machine run\n"
            "  estia token new extract --scopes generate\n"
            "and export the token it prints: export ESTIA_TOKEN=estia_...",
            file=sys.stderr,
        )
        return 2
    args = sys.argv[1:]
    schema = EVENT_SCHEMA
    if args and args[0] == "--impossible":
        schema, args = IMPOSSIBLE_SCHEMA, args[1:]
    text = " ".join(args) or SAMPLE

    client = OpenAI(base_url=f"{URL}/v1", api_key=TOKEN, max_retries=0)
    try:
        event = extract(client, text, schema)
    except openai.UnprocessableEntityError as e:
        # 422: the model's answer did not fit the schema even after the
        # engine's repair step and one retry. Decide what your app does here:
        # ask again with a simpler schema, fall back to plain text, or tell
        # the user. Retrying the same request forever will not help.
        print(f"no valid answer for this schema: {explain(e)}", file=sys.stderr)
        return 3
    except openai.OpenAIError as e:
        print(f"error: {explain(e)}", file=sys.stderr)
        return 1

    print(json.dumps(event, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
