#!/usr/bin/env python3
"""A small personal assistant in the terminal, on top of Estia.

What it shows:
  - streaming replies through the OpenAI Python SDK
  - keeping the conversation history on the client
  - the prompt cache: every turn sends the same `user` value, so the engine
    reuses the conversation's KV cache and only prefills the new turn
  - reading Estia's extra facts (`x_estia`) from the last stream chunk
  - cancelling a reply: Ctrl-C closes the connection, and the engine stops
    generating

Scopes: generate

    export ESTIA_TOKEN=estia_...      # estia token new chat --scopes generate
    python chat.py                    # ESTIA_MODEL=text for the larger model

Type a message and press Enter. /reset starts a new conversation. /quit or
Ctrl-D exits.
"""

import os
import sys
import uuid

import openai
from openai import OpenAI

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
TOKEN = os.environ.get("ESTIA_TOKEN", "")
# A role, not a model id: the engine decides which model serves `fast`.
MODEL = os.environ.get("ESTIA_MODEL", "fast")

SYSTEM = "You are a concise personal assistant. Answer in a few short sentences."


def explain(e: Exception) -> str:
    """Turn an SDK error into one message a person can act on."""
    if isinstance(e, openai.APIConnectionError):
        return f"cannot reach the engine at {URL}. Is `estia serve` running? Set ESTIA_URL if it listens elsewhere."
    if isinstance(e, openai.APIStatusError):
        message = e.body.get("message") if isinstance(e.body, dict) else e.message
        text = f"the engine answered {e.status_code}: {message}"
        if e.request_id:
            text += f" (request id {e.request_id})"
        if e.status_code == 401:
            text += "\nCheck ESTIA_TOKEN, or mint a token: estia token new chat --scopes generate"
        return text
    return str(e)


def new_conversation():
    # A stable id per conversation. The engine uses it as the prompt-cache key,
    # scoped to this token, so no other client can share or probe the cache.
    return f"chat-{uuid.uuid4().hex[:12]}", [{"role": "system", "content": SYSTEM}]


def main() -> int:
    if not TOKEN:
        print(
            "ESTIA_TOKEN is not set. On the engine's machine run\n"
            "  estia token new chat --scopes generate\n"
            "and export the token it prints: export ESTIA_TOKEN=estia_...",
            file=sys.stderr,
        )
        return 2

    # max_retries=0: a local engine that refuses a request will refuse it again.
    client = OpenAI(base_url=f"{URL}/v1", api_key=TOKEN, max_retries=0)
    conversation_id, history = new_conversation()
    print(f"Estia at {URL}, model `{MODEL}`. /reset for a new conversation, /quit to exit.")

    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        if line == "/reset":
            conversation_id, history = new_conversation()
            print("(new conversation)")
            continue

        history.append({"role": "user", "content": line})
        reply = []
        stats = {}
        stream = None
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True,
                user=conversation_id,
                max_tokens=512,
            )
            print("assistant> ", end="", flush=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    reply.append(piece)
                    print(piece, end="", flush=True)
                # The last chunk carries usage and Estia's own facts.
                if chunk.usage:
                    stats["prompt"] = chunk.usage.prompt_tokens
                    details = chunk.usage.prompt_tokens_details
                    stats["cached"] = details.cached_tokens if details else 0
                extra = (chunk.model_extra or {}).get("x_estia")
                if extra:
                    stats["ms"] = extra.get("ms")
            # A failure after the stream started arrives as an error event,
            # which the SDK raises as openai.APIError (handled below).
            print()
        except KeyboardInterrupt:
            # Closing the stream closes the connection; the engine sees the
            # client go and cancels the generation in the runner.
            if stream is not None:
                stream.close()
            print("\n[cancelled]")
        except openai.OpenAIError as e:
            history.pop()
            print(f"\nerror: {explain(e)}", file=sys.stderr)
            if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.APIConnectionError)):
                return 1
            continue

        # Keep what was said, including a cancelled partial reply, so the
        # model sees the whole conversation. Appending (never rewriting
        # earlier turns) is what lets the next turn reuse the cache.
        history.append({"role": "assistant", "content": "".join(reply)})
        if stats:
            print(f"[{stats.get('prompt', '?')} prompt tokens, {stats.get('cached', 0)} from cache, {stats.get('ms', '?')} ms]")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
