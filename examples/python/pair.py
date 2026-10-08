#!/usr/bin/env python3
"""Get a token by pairing, the way an app on another device does.

What it shows:
  - checking the engine with GET /engine/health (no token needed)
  - over https: recording the certificate the engine presents, and proving
    with two 6-digit codes, one each way, that it is the engine's own
    (estia_tls.py)
  - asking for a token with POST /engine/pair: a name, the scopes you need, and
    a commitment to a random nonce; revealing the nonce once the engine has
    answered with its own, and computing the codes
  - polling GET /engine/pair/<id> until the operator approves or denies, and
    giving up before the engine drops the request (5 minutes after it was made)
  - asking for the confirmation code the operator's `estia pair approve`
    printed, and keeping the token only when it matches
  - saving the token to a file only you can read (mode 0600), and printing the
    certificate fingerprint to pin from now on (ESTIA_PIN)

Scopes: none to run it. The token it receives has the scopes you ask for.
Standard library only.

    ESTIA_URL=https://192.168.1.20:27200 python pair.py --name "kitchen tablet"
    # it prints a code; on the engine's machine: estia pair approve <id> --code <code>
    # that prints a confirmation code: type it here when asked
    export ESTIA_TOKEN=$(cat estia-token) ESTIA_PIN=<the fingerprint it prints>

Why the codes: nothing vouches for an engine's own certificate, so the first
connection could be to something in between that presents its own. Both
codes are computed from the certificate this script saw and from two nonces,
one of which it committed to before seeing the other; the engine computes them
from its real certificate, so they agree only when the certificate is the
engine's. The engine checks the first (the operator types it); this script
checks the second (you type it), so neither side can be fooled by an answer
that did not come from the other. If the engine refuses the code, or the
confirmation code does not match here, stop: do not use the token.

`http://127.0.0.1:27200` (an engine on this machine) pairs without codes.

It waits 290 s by default (`--wait`), just under the 5 minutes the engine
keeps a request, and then prints:

    no decision within 290 s. The engine drops a pairing request 5 minutes
    after it was made; run pair.py again to ask anew.

The operator sees the name and the scopes before approving. Ask for the least
your app needs: `generate` to chat, `embed` for embeddings, `models:read` to
list models. Never ask for `admin` unless your app manages the engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import estia_tls

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
HTTPS = URL.lower().startswith("https://")
# Over https: pinned to ESTIA_PIN when it is set, else the first certificate
# seen is recorded and required from then on (the codes below prove it).
if HTTPS:
    CTX = estia_tls.PinnedContext(os.environ["ESTIA_PIN"]) if os.environ.get("ESTIA_PIN") else estia_tls.PinnedContext.learning()
else:
    CTX = None

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
        with urllib.request.urlopen(req, timeout=10, context=CTX) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = raw or e.reason
        raise EngineError(e.code, message, e.headers.get("x-request-id")) from None


def check_stated(answer: dict) -> None:
    """An answer naming another certificate than the one this connection saw
    means something in between rewrote it. Not the security check (the code
    is), but a clear early stop."""
    stated = (answer.get("tls") or {}).get("fingerprint")
    if CTX is not None and stated and CTX.pin and estia_tls.normalize(stated) != CTX.pin:
        raise estia_tls.CertificateMismatch(
            f"the engine says its certificate is {estia_tls.short(stated)}, but this connection presented {estia_tls.short(CTX.pin)}"
        )


def confirm(expected: str) -> bool:
    """Ask for the confirmation code `estia pair approve` printed on the
    engine's machine; three tries, for typos. Read from stdin, never shown."""
    for _ in range(3):
        print("\napproved. Type the confirmation code `estia pair approve` printed on the engine's machine: ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            print("\nno confirmation code: the token is not kept until it is typed", file=sys.stderr)
            return False
        typed = "".join(c for c in line if c.isdigit())
        if typed == expected:
            return True
        print("that is not the confirmation code this device expects.", file=sys.stderr)
    print(
        "pairing not confirmed: if you typed the code `estia pair approve` printed, the approval did not come from that "
        "engine: something between this device and the engine answered in its place. Do not use the token; deny the "
        "pairing on the engine and pair again.",
        file=sys.stderr,
    )
    return False


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

    if not HTTPS and not estia_tls.is_this_machine(URL):
        print(
            f"{URL} is plain http to another machine: the token would cross the network in clear.\n"
            "Use https:// (an engine serving the LAN serves https): ESTIA_URL=https://<engine>:27200",
            file=sys.stderr,
        )
        return 1
    try:
        health = call("GET", "/engine/health")
    except (urllib.error.URLError, OSError) as e:
        print(f"cannot reach the engine at {URL}: {getattr(e, 'reason', e)}. Set ESTIA_URL to its address.", file=sys.stderr)
        return 1
    except EngineError as e:
        print(f"the engine answered {e.status}: {e.message}", file=sys.stderr)
        return 1
    print(f"engine {health.get('version')} (api v{health.get('api_version')}) at {URL}")
    if CTX is not None:
        print(f"certificate {estia_tls.short(CTX.pin)} (not trusted yet: the code below checks it)")

    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    nonce = os.urandom(32)
    body = {"name": args.name, "scopes": scopes}
    if HTTPS:
        # Commit to the nonce before seeing the engine's.
        body["commitment"] = estia_tls.pairing_commitment(nonce)
    try:
        pairing = call("POST", "/engine/pair", body)
        check_stated(pairing)
        pid = pairing["id"]
        code = expected_confirmation = None
        if HTTPS:
            try:
                engine_nonce = bytes.fromhex(pairing.get("engine_nonce") or "")
            except ValueError:
                engine_nonce = b""
            if len(engine_nonce) != 32:
                print("the engine answered without a nonce; is it older than 0.5?", file=sys.stderr)
                return 1
            call("POST", f"/engine/pair/{pid}/reveal", {"nonce": nonce.hex()})
            code = estia_tls.pairing_code(CTX.pin, nonce, engine_nonce)
            expected_confirmation = estia_tls.pairing_confirmation(CTX.pin, nonce, engine_nonce)
    except EngineError as e:
        if e.status == 429:
            print(f"too many pairing requests are waiting: {e.message}\nWait for one to be decided or to expire (5 minutes).", file=sys.stderr)
        else:
            print(f"pairing refused ({e.status}): {e.message}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"pairing failed: {getattr(e, 'reason', e)}", file=sys.stderr)
        return 1

    print(f"pairing id: {pid}  (name {args.name!r}, scopes {','.join(scopes)})")
    if code:
        print(f"\n    code: {code[:3]} {code[3:]}\n")
        print("Approve it on the engine's machine with this code:")
        print(f"  estia pair approve {pid} --code {code}")
        print("It prints a confirmation code: type that here when asked.")
        print("If the engine says the code does not match, stop here: something answered in its place.")
    else:
        print("Approve it on the engine's machine:")
        print(f"  estia pair approve {pid}")
    print(f"or from an admin client: POST {URL}/engine/pairings/{pid}/approve", flush=True)
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
        except (urllib.error.URLError, OSError) as e:
            if isinstance(getattr(e, "reason", e), estia_tls.CertificateMismatch):
                print(f"\n{getattr(e, 'reason', e)}", file=sys.stderr)
                return 1
            print("x", end="", flush=True)
            continue
        try:
            check_stated(state)
        except estia_tls.CertificateMismatch as e:
            print(f"\n{e}", file=sys.stderr)
            return 1
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
            # Over https an "approved" answer alone proves nothing: whatever
            # answered the request could answer the polls too.
            if expected_confirmation and not confirm(expected_confirmation):
                return 1
            save_token(args.out, token)
            print(f"\napproved. Token saved to {args.out} (mode 0600).")
            print(f"  export ESTIA_TOKEN=$(cat {args.out})")
            if CTX is not None:
                # The certificate the codes proved: pin it for every later call.
                print(f"  export ESTIA_PIN={CTX.pin}")
            return 0
    print(f"\nno decision within {args.wait} s. {EXPIRY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
