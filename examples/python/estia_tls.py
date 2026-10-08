#!/usr/bin/env python3
"""Pin an Estia engine's certificate from Python, with the standard library.

An engine on the LAN (`estia serve --lan`) serves HTTPS with its own
certificate. Nothing vouches for that certificate, so a client pins it: it
trusts exactly the certificate whose SHA-256 fingerprint it was given. Pairing
(pair.py) gives you the fingerprint along with the token; `estia tls
fingerprint` on the engine's machine prints it too.

    import estia_tls
    ctx = estia_tls.context_for(URL)        # https: pinned to ESTIA_PIN; http to this machine: None
    urllib.request.urlopen(request, context=ctx)

    # The OpenAI SDK, pinned the same way (None, the SDK's default, for http):
    client = OpenAI(base_url=f"{URL}/v1", api_key=TOKEN,
                    http_client=estia_tls.openai_http_client(URL))

How it checks: the context does not verify the certificate's chain, names or
dates (a self-signed certificate fails all three), and after every handshake
compares the SHA-256 of the certificate the server presented with the pin,
closing the connection on a mismatch. OpenSSL still verifies the handshake
signature, so only a server holding the pinned certificate's private key gets
through. The check runs in the handshake itself (sslsocket_class and
sslobject_class), so it covers sockets, deferred handshakes and asyncio
transports alike.

Run it to check the pairing-code functions against the known-answer vector in
docs/api.md:

    python estia_tls.py

Python 3.9 or newer.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import ssl
import sys
import urllib.parse


class CertificateMismatch(ssl.SSLCertVerificationError):
    """The engine presented a certificate other than the pinned one."""

    def __str__(self) -> str:
        # ssl.SSLError prints its arguments as a tuple, `('the engine …',)`,
        # when it has no errno; the message is what people read.
        return str(self.args[0]) if self.args else "the engine presented another certificate"


def normalize(fingerprint: str) -> str:
    """64 lowercase hex digits from a fingerprint in any common spelling:
    `estia tls fingerprint`, `openssl x509 -fingerprint -sha256` (colons,
    upper case, a `SHA256 Fingerprint=` prefix) or a browser's spaced form."""
    s = fingerprint.strip()
    for prefix in ("sha256 fingerprint=", "sha256:", "sha-256:", "sha256//", "sha256 "):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    s = "".join(c for c in s if c not in ": -\t").lower()
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"not a SHA-256 certificate fingerprint: {fingerprint[:24]!r} (expected 64 hex digits)")
    return s


def fingerprint(der: bytes) -> str:
    """The fingerprint of a certificate (its DER bytes), as the engine prints it."""
    return hashlib.sha256(der).hexdigest()


def short(fp: str) -> str:
    """The first 64 bits in four groups, `3f2a 9c41 7b0e d512`, for comparing by eye."""
    h = normalize(fp)[:16]
    return " ".join(h[i:i + 4] for i in range(0, 16, 4))


def is_this_machine(url: str) -> bool:
    """Does `url` name this machine (localhost, a loopback address)? Only
    those may be reached over plain http with a token. A URL with a user name
    or password in it never counts."""
    parts = urllib.parse.urlsplit(url)
    if parts.username is not None or parts.password is not None:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


class _PinnedSocket(ssl.SSLSocket):
    def do_handshake(self, *args, **kwargs):
        super().do_handshake(*args, **kwargs)
        try:
            self.context.check_peer(self.getpeercert(binary_form=True))
        except CertificateMismatch:
            self.close()
            raise


class _PinnedObject(ssl.SSLObject):
    def do_handshake(self):
        super().do_handshake()
        self.context.check_peer(self.getpeercert(binary_form=True))


# What PinnedContext.learning() passes instead of a pin.
_LEARN = object()


class PinnedContext(ssl.SSLContext):
    """A client context that trusts one certificate, by fingerprint.

    `PinnedContext(pin)` needs the pin: an empty or missing one is a
    ValueError, never a context that trusts whatever answers first. Pairing
    alone, which has nothing to pin yet, uses `PinnedContext.learning()`.
    """

    sslsocket_class = _PinnedSocket
    sslobject_class = _PinnedObject

    def __new__(cls, pin):
        return super().__new__(cls, ssl.PROTOCOL_TLS_CLIENT)

    def __init__(self, pin):
        # The chain, names and dates are not what is checked: the pin is.
        self.check_hostname = False
        self.verify_mode = ssl.CERT_NONE
        self.set_alpn_protocols(["http/1.1"])
        if pin is _LEARN:
            self.pin = None
        elif not pin or not str(pin).strip():
            raise ValueError(
                "PinnedContext needs the engine's certificate fingerprint (ESTIA_PIN): pair.py prints it when it "
                "pairs, and `estia tls fingerprint` prints it on the engine's machine"
            )
        else:
            self.pin = normalize(pin)

    @classmethod
    def learning(cls) -> PinnedContext:
        """For pairing only: records the first certificate it sees (in `.pin`)
        and requires that one from then on, so one context carries a whole
        pairing and a certificate swapped in the middle is refused. Nothing
        has vouched for that certificate until the pairing's codes match: see
        pair.py."""
        return cls(_LEARN)

    def check_peer(self, der: bytes | None) -> None:
        if not der:
            raise CertificateMismatch("the engine presented no certificate")
        seen = fingerprint(der)
        if self.pin is None:
            self.pin = seen
        elif seen != self.pin:
            raise CertificateMismatch(
                f"the engine presented certificate {short(seen)}, but this app pinned {short(self.pin)}: "
                "if its owner ran `estia tls rotate`, pair again; otherwise something on the network is impersonating it"
            )


def context_for(url: str, pin: str | None = None) -> PinnedContext | None:
    """The context for `url`: pinned to `pin` (default: ESTIA_PIN) for https,
    None for plain http to this machine. An https URL without a pin is a
    ValueError rather than an unchecked connection, and so is plain http to
    another machine: the token would cross the network in clear."""
    if not url.lower().startswith("https://"):
        if not is_this_machine(url):
            raise ValueError(
                f"{url} is plain http to another machine: the token, prompts and outputs would cross the network in "
                "clear. An engine serving the LAN serves https: use https:// and pin its certificate (ESTIA_PIN)."
            )
        return None
    pin = pin or os.environ.get("ESTIA_PIN", "").strip()
    if not pin:
        raise ValueError(
            "ESTIA_PIN is not set: an https engine needs its certificate fingerprint. pair.py prints it when it pairs; "
            "`estia tls fingerprint` on the engine's machine prints it too."
        )
    return PinnedContext(pin)


def context_or_exit(url: str) -> PinnedContext | None:
    """context_for(url) for a script: a missing pin prints what to do and exits 2."""
    try:
        return context_for(url)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)


def openai_http_client(url: str):
    """The `http_client` for the OpenAI SDK: pinned like context_for(url) for
    https, None (the SDK's own default) for http to this machine. Exits 2 with
    a message when the pin is missing (context_or_exit). Uses the SDK's
    DefaultHttpxClient where it has one, and plain httpx.Client on SDKs from
    before it."""
    ctx = context_or_exit(url)
    if ctx is None:
        return None
    import openai

    make = getattr(openai, "DefaultHttpxClient", None)
    if make is None:
        import httpx  # the OpenAI SDK depends on it

        make = httpx.Client
    return make(verify=ctx)


def mismatch_in(error: BaseException | None) -> CertificateMismatch | None:
    """The CertificateMismatch behind an exception, if there is one: SDKs wrap
    it (openai.APIConnectionError, around httpx.ConnectError, around it)."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, CertificateMismatch):
            return error
        seen.add(id(error))
        reason = getattr(error, "reason", None)  # urllib.error.URLError carries it here
        error = reason if isinstance(reason, BaseException) else (error.__cause__ or error.__context__)
    return None


def pairing_commitment(device_nonce: bytes) -> str:
    """What the device sends with its pairing request: SHA-256 of a label and its nonce, in hex."""
    return hashlib.sha256(b"estia pairing commitment v1\x00" + device_nonce).hexdigest()


def _six_digits(label: bytes, fp: str, device_nonce: bytes, engine_nonce: bytes) -> str:
    d = hashlib.sha256(label + b"\x00" + bytes.fromhex(normalize(fp)) + device_nonce + engine_nonce).digest()
    return f"{int.from_bytes(d[:8], 'big') % 1_000_000:06d}"


def pairing_code(fp: str, device_nonce: bytes, engine_nonce: bytes) -> str:
    """The 6-digit code the device shows: computed from the certificate it
    saw; the engine computes it from its own. The operator types the device's
    into `estia pair approve <id> --code <code>`."""
    return _six_digits(b"estia pairing code v1", fp, device_nonce, engine_nonce)


def pairing_confirmation(fp: str, device_nonce: bytes, engine_nonce: bytes) -> str:
    """The 6-digit code `estia pair approve` prints once it accepted the
    pairing code. The user types it into the device, which compares it with
    this, computed from the certificate it saw, and pins that certificate only
    when they match: the proof that the approval came from the engine. Never
    show it on the device."""
    return _six_digits(b"estia pairing confirm v1", fp, device_nonce, engine_nonce)


if __name__ == "__main__":
    fp = bytes(range(0x00, 0x20)).hex()
    device, engine = bytes(range(0x20, 0x40)), bytes(range(0x40, 0x60))
    assert pairing_commitment(device) == "71f9884a2523f984d667e49fff85b931cdaff466dc51fe953ca46436319974dd"
    assert pairing_code(fp, device, engine) == "865695"
    assert pairing_confirmation(fp, device, engine) == "452618"
    assert short(fp) == "0001 0203 0405 0607"
    assert normalize("SHA256 Fingerprint=" + ":".join(fp[i:i + 2].upper() for i in range(0, 64, 2))) == fp
    # A pinned context needs a pin; only learning() starts without one.
    for missing in (None, "", "  "):
        try:
            PinnedContext(missing)
        except ValueError:
            pass
        else:
            raise AssertionError(f"PinnedContext({missing!r}) must refuse to trust the first certificate it sees")
    assert PinnedContext.learning().pin is None
    assert PinnedContext(fp).pin == fp
    # Plain http only to this machine.
    assert context_for("http://127.0.0.1:27200") is None and context_for("http://[::1]:1") is None
    for url in ("http://192.168.1.20:27200", "http://studio.local:27200", "http://user@127.0.0.1:1"):
        try:
            context_for(url)
        except ValueError:
            pass
        else:
            raise AssertionError(f"context_for({url!r}) must refuse plain http to another machine")
    assert context_for("https://192.168.1.20:27200", pin=fp).pin == fp
    print("ok: the pairing codes and commitment match the known-answer vector; contexts refuse to run without a pin")
