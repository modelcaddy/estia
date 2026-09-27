#!/usr/bin/env python3
"""Get a token by pairing, the way an app on another device does.

What it shows:
  - checking the engine with GET /engine/health (no token needed)
  - asking for a token with POST /engine/pair: a name and the scopes you need
  - polling GET /engine/pair/<id> until the operator approves or denies, and
    giving up before the engine drops the request (5 minutes after it was made)
  - saving the token to a file only you can read (mode 0600)

Scopes: none to run it. The token it receives has the scopes you ask for.
Standard library only.

    ESTIA_URL=http://192.168.1.20:27200 python pair.py --name "kitchen tablet"
    # on the engine's machine: estia pair approve <id>
    export ESTIA_TOKEN=$(cat estia-token)

It waits 290 s by default (`--wait`), just under the 5 minutes the engine
keeps a request, and then prints:

    no decision within 290 s. The engine drops a pairing request 5 minutes
    after it was made; run pair.py again to ask anew.

The operator sees the name and the scopes before approving. Ask for the least
your app needs: `generate` to chat, `embed` for embeddings, `models:read` to
list models. Never ask for `admin` unless your app manages the engine.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")

# The engine drops an undecided request 300 s after it was made. Stop polling
# a little before that: the last poll then still finds the request, and the
# user hears why we stopped rather than a bare 404.
EXPIRES_S = 300
DEFAULT_WAIT_S = 290
EXPIRY = "The engine drops a pairing request 5 minutes after it was made; run pair.py again to ask anew."


class EngineError(Exception):
    def __init__(self, status: int, message: str, request_id: str | None):
        self.status = status
        self.message = message
        self.request_id = request_id
        super().__init__(f"{status}: {message}")


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{URL}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = raw or e.reason
        raise EngineError(e.code, message, e.headers.get("x-request-id")) from None


def save_token(path: str, token: str) -> None:
    # Create the file 0600 from the start, so the token is never readable by
    # others, not even for a moment. fchmod covers a file that already existed.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (token + "\n").encode())
    finally:
        os.close(fd)


def main() -> int:
    p = argparse.ArgumentParser(description="Ask an Estia engine for a token and wait for approval.")
    p.add_argument("--name", default="my-app", help="how the operator sees this device (letters, digits, spaces, . _ - ' ( ))")
    p.add_argument("--scopes", default="generate,embed,models:read", help="comma-separated scopes to ask for")
    p.add_argument("--out", default="estia-token", help="file to save the token in (created with mode 0600)")
    p.add_argument(
        "--wait",
        type=int,
        default=DEFAULT_WAIT_S,
        help=f"seconds to wait for approval (default {DEFAULT_WAIT_S}; the engine drops a request after {EXPIRES_S})",
    )
    args = p.parse_args()

    try:
        health = call("GET", "/engine/health")
    except (urllib.error.URLError, OSError) as e:
        print(f"cannot reach the engine at {URL}: {getattr(e, 'reason', e)}. Set ESTIA_URL to its address.", file=sys.stderr)
        return 1
    except EngineError as e:
        print(f"the engine answered {e.status}: {e.message}", file=sys.stderr)
        return 1
    print(f"engine {health.get('version')} (api v{health.get('api_version')}) at {URL}")

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    try:
        pairing = call("POST", "/engine/pair", {"name": args.name, "scopes": scopes})
    except EngineError as e:
        if e.status == 429:
            print(f"too many pairing requests are waiting: {e.message}\nWait for one to be decided or to expire (5 minutes).", file=sys.stderr)
        else:
            print(f"pairing refused ({e.status}): {e.message}", file=sys.stderr)
        return 1

    pid = pairing["id"]
    print(f"pairing id: {pid}  (name {args.name!r}, scopes {','.join(scopes)})")
    print("Approve it on the engine's machine:")
    print(f"  estia pair approve {pid}")
    print(f"or from an admin client: POST {URL}/engine/pairings/{pid}/approve")
    print("waiting", end="", flush=True)

    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            state = call("GET", f"/engine/pair/{pid}")
        except EngineError as e:
            if e.status == 404:
                # Expired: the wait was longer than the engine keeps requests.
                print(f"\nthe request has expired ({e.message}). {EXPIRY}", file=sys.stderr)
                return 1
            # 500 means the engine could not read or write its pairing file.
            # The token is only handed out once its collection is saved, so
            # polling again is safe.
            print(f"\n(poll failed: {e.status} {e.message}; retrying)", file=sys.stderr)
            continue
        except (urllib.error.URLError, OSError):
            print("x", end="", flush=True)
            continue
        status = state.get("status")
        if status == "pending":
            print(".", end="", flush=True)
            continue
        if status == "denied":
            print("\nthe operator denied the request", file=sys.stderr)
            return 1
        if status == "approved":
            token = state.get("token")
            if not token:
                # The token is returned exactly once. If this poll did not get
                # it, another poll did: ask the operator to deny that pairing
                # (which revokes the token) and pair again.
                print("\napproved, but the token was already collected by another poll", file=sys.stderr)
                return 1
            save_token(args.out, token)
            print(f"\napproved. Token saved to {args.out} (mode 0600).")
            print(f"  export ESTIA_TOKEN=$(cat {args.out})")
            return 0
    print(f"\nno decision within {args.wait} s. {EXPIRY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
