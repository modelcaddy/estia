#!/usr/bin/env python3
"""Get a token by pairing, the way an app on another device does.

What it shows:
  - checking the engine with GET /engine/health (no token needed)
  - over https: recording the certificate the engine presents, and proving
    with one 6-digit code, shown on both ends and compared by a person, that
    it is the engine's own (estia_tls.py)
  - asking for a token with POST /engine/pair: a name, the scopes you need, and
    a commitment to a random nonce; revealing the nonce once the engine has
    answered with its own, and computing the code
  - polling GET /engine/pair/<id> until the operator approves or denies, and
    giving up before the engine drops the request (5 minutes after it was made)
  - asking whether the engine shows the same code, and keeping the token only
    on a yes (typed at a terminal; a script gives the six digits the engine
    shows instead, which must equal this script's code)
  - saving the token to a file only you can read (mode 0600), and printing the
    certificate fingerprint to pin from now on (ESTIA_PIN)

Scopes: none to run it. The token it receives has the scopes you ask for.
Standard library only.

    ESTIA_URL=https://192.168.1.20:27200 python pair.py --name "kitchen tablet"
    # it prints a code; on the engine's machine, `estia pair approve <id>` shows
    # the engine's code and asks whether this device shows the same
    # once approved, this script asks whether the engine shows its code: y
    export ESTIA_TOKEN=$(cat estia-token) ESTIA_PIN=<the fingerprint it prints>

With stdin not a terminal (a script drives it), "y" is not an answer: a
script could write it without anyone looking. Write the six digits the engine
shows for the request instead (`estia pair list` on its machine); this script
keeps the token only if they equal its own code.

Why the code: nothing vouches for an engine's own certificate, so the first
connection could be to something in between that presents its own. The code
is computed from the certificate this script saw and from two nonces, one of
which it committed to before seeing the other; the engine computes its own
from its real certificate, so the two agree only when the certificate is the
engine's. Neither end sends its code to the other: you compare the two
screens, and say so on both. The operator approves only when this device
shows the engine's code; this script keeps the token only when you say the
engine shows its code, so neither side is fooled by an answer that did not
come from the other. If the codes differ, stop: deny the request on the
engine, and do not use the token.

`http://127.0.0.1:27200` (an engine on this machine) pairs without a code.

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
import re
import sys
import time
import urllib.error
import urllib.request

import estia_tls

URL = os.environ.get("ESTIA_URL", "http://127.0.0.1:27200").rstrip("/")
HTTPS = URL.lower().startswith("https://")
# Over https: pinned to ESTIA_PIN when it is set, else the first certificate
# seen is recorded and required from then on (the code below proves it).
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
# What an engine's pairing ids are made of: 16 lowercase letters, none that
# passes for a digit. This script shows the id next to its code, and whatever
# answers the request chooses it: an id such as "771204" would put a code on
# this screen that is not this script's.
PAIRING_ID = re.compile(r"[acdefhjkmnprtuvwxy]{16}")


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


def confirm(code: str, pid: str) -> bool:
    """Ask whether the engine shows this script's code for the request (on its
    machine: `estia pair approve`, `estia pair list`, an admin app). At a
    terminal a yes counts, and so do the six digits the engine shows. Without
    one only those digits do: a "y" piped in proves nobody compared anything.
    Anything else, or no answer, keeps the token back."""
    at_terminal = sys.stdin.isatty()
    if at_terminal:
        print(f"\napproved. Does the engine show {code[:3]} {code[3:]} for this request? [y/N] ", end="", flush=True)
    else:
        print(f"\napproved. The six digits the engine shows for request {pid}: ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    digits = answer.replace(" ", "").replace("-", "")
    if len(digits) == 6 and digits.isascii() and digits.isdigit():
        if digits == code:
            return True
        print(
            f"\npairing not confirmed; the token is not kept. The engine shows {digits[:3]} {digits[3:]}, this device "
            f"{code[:3]} {code[3:]}: something between them answered in the engine's place. Deny the request there "
            f"(estia pair deny {pid}) and pair again.",
            file=sys.stderr,
        )
        return False
    if answer in ("y", "yes") and at_terminal:
        return True
    if answer in ("y", "yes"):
        print(
            "\npairing not confirmed; the token is not kept. stdin is not a terminal, so \"y\" is not an answer: a script "
            f"could write it without anyone comparing the codes. Write the six digits the engine shows for request {pid} "
            "(estia pair list on its machine) instead.",
            file=sys.stderr,
        )
        return False
    print(
        "\npairing not confirmed; the token is not kept. If the engine shows another code, or none, something between "
        f"this device and the engine answered in its place: deny the request there (estia pair deny {pid}) and pair again.",
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
    p.add_argument(
        "--name",
        default="my-app",
        help="how the operator sees this device: letters, digits, single spaces and . _ - ' ( ), "
        "with at most 4 digits in a row (only a letter such as a-z ends a row)",
    )
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
        pid = str(pairing.get("id") or "")
        code = None
        if not pid:
            print("the engine answered without a pairing id", file=sys.stderr)
            return 1
        if HTTPS and not PAIRING_ID.fullmatch(pid):
            print(
                f"the answer's pairing id {pid[:32]!r} is not one an engine makes (16 letters, none that pass for a digit): "
                "whatever answered may be putting a code of its own on this screen. Stop, and pair again from a network you trust.",
                file=sys.stderr,
            )
            return 1
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
        print("Approve it on the engine's machine, which shows its own code for this request:")
        print(f"  estia pair approve {pid}")
        print(f"Approve only if the engine shows {code[:3]} {code[3:]} too. If it shows another code, stop here:")
        print("something between this device and the engine answered in its place.")
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
            # answered the request could answer the polls too. The person
            # compares this script's code with the engine's screen.
            if code and not confirm(code, pid):
                return 1
            save_token(args.out, token)
            print(f"\napproved. Token saved to {args.out} (mode 0600).")
            print(f"  export ESTIA_TOKEN=$(cat {args.out})")
            if CTX is not None:
                # The certificate the code proved: pin it for every later call.
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
