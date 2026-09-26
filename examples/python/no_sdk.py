#!/usr/bin/env python3
"""The chat from chat.py with the standard library only: no SDK, no pip.

What it shows:
  - the raw HTTP calls: POST /v1/chat/completions with a bearer token
  - reading a server-sent event stream line by line (`data: {...}`, then
    `data: [DONE]`)
  - the error shape: {"error": {"message", "type", "code"}}
  - cancelling by closing the connection (Ctrl-C)

Scopes: generate

    export ESTIA_TOKEN=estia_...      # estia token new chat --scopes generate
    python no_sdk.py

Type a message and press Enter. /quit or Ctrl-D exits.
"""

import json
import os
import sys
import urllib.error
import urllib.request
import uuid

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
TOKEN = os.environ.get("ESTIA_TOKEN", "")
MODEL = os.environ.get("ESTIA_MODEL", "fast")


class EngineError(Exception):
    pass


def open_chat(messages: list, user: str):
    """Send the request; return the open streaming response."""
    body = {"model": MODEL, "messages": messages, "stream": True, "user": user, "max_tokens": 512}
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = raw or e.reason
        rid = e.headers.get("x-request-id")
        hint = "\nCheck ESTIA_TOKEN, or mint a token: estia token new chat --scopes generate" if e.code == 401 else ""
        raise EngineError(f"the engine answered {e.code}: {message}" + (f" (request id {rid})" if rid else "") + hint) from None
    except urllib.error.URLError as e:
        raise EngineError(f"cannot reach the engine at {URL} ({e.reason}). Is `estia serve` running?") from None


def read_stream(resp):
    """Yield text pieces, then one final dict with usage and x_estia."""
    # `with` closes the connection even when the caller stops early (Ctrl-C),
    # and a closed connection is how the engine learns to stop generating.
    with resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data: "):
                continue  # blank separators and keep-alive comments
            data = line[len("data: "):]
            if data == "[DONE]":
                return
            event = json.loads(data)
            if "error" in event:
                raise EngineError(f"the engine failed mid-stream: {event['error'].get('message')}")
            choice = event["choices"][0]
            piece = choice.get("delta", {}).get("content")
            if piece:
                yield piece
            if choice.get("finish_reason"):
                yield {"usage": event.get("usage", {}), "x_estia": event.get("x_estia", {})}


def main() -> int:
    if not TOKEN:
        print(
            "ESTIA_TOKEN is not set. On the engine's machine run\n"
            "  estia token new chat --scopes generate\n"
            "and export the token it prints: export ESTIA_TOKEN=estia_...",
            file=sys.stderr,
        )
        return 2

    user = f"chat-{uuid.uuid4().hex[:12]}"  # prompt-cache key for this conversation
    history = [{"role": "system", "content": "You are a concise personal assistant. Answer in a few short sentences."}]
    print(f"Estia at {URL}, model `{MODEL}`. /quit to exit.")
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
        history.append({"role": "user", "content": line})
        reply, final = [], {}
        try:
            gen = read_stream(open_chat(history, user))
        except EngineError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        try:
            print("assistant> ", end="", flush=True)
            for item in gen:
                if isinstance(item, str):
                    reply.append(item)
                    print(item, end="", flush=True)
                else:
                    final = item
            print()
        except KeyboardInterrupt:
            gen.close()  # runs the generator's `with`: the connection closes
            print("\n[cancelled]")
        except EngineError as e:
            history.pop()
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        history.append({"role": "assistant", "content": "".join(reply)})
        usage = final.get("usage") or {}
        if usage:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(f"[{usage.get('prompt_tokens')} prompt tokens, {cached} from cache, {final.get('x_estia', {}).get('ms')} ms]")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
