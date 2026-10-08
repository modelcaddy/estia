# Security policy

## Reporting a vulnerability

Report it privately through GitHub. On
[github.com/modelcaddy/estia](https://github.com/modelcaddy/estia), open the
**Security** tab and choose **Report a vulnerability**, or go straight to
<https://github.com/modelcaddy/estia/security/advisories/new>. Only the
maintainers can see the report. Do not open a public issue, pull request or
discussion for a security problem.

Include the Estia version (`estia --version`), your OS, and the steps to
reproduce. We will confirm receipt, work on a fix, and agree a disclosure date
with you. We credit reporters who want to be credited.

## Supported versions

Estia is before 1.0. Only the latest release gets security fixes. Upgrade to
it before you report, if you can.

## Known limitations

These are known and documented. They are not vulnerabilities, but reports that
show a practical impact beyond what is described here are welcome.

- **TLS on the LAN, pinned at pairing.** On a LAN bind (`estia serve --lan`,
  or `estia service install` without `--local`) the engine serves HTTPS with
  its own self-signed certificate (ECDSA P-256), made on the first LAN start
  and kept in `tls/` in the data directory. The key file is owner-only (0600
  in a 0700 directory); whoever can read it can impersonate the engine.
  Tokens, prompts and output from other devices cross the network encrypted.
  Plain HTTP stays, on the same port, for peers on the engine's own machine
  only (loopback); plain HTTP from another device is refused with a 400
  (`https_required`) before any handler runs, but whatever that request
  carried (a token, a prompt) has already crossed the network in clear; the
  engine logs the name of any valid token that arrives that way (a `warn`),
  but does not revoke it. After upgrading from 0.4, pair each device again
  and revoke the token it had. The Rust client and the Python examples refuse
  to send a token over plain HTTP to another machine at all.

  Nothing vouches for the certificate, so each device pins its SHA-256
  fingerprint, and what makes that safe is how the device learns it:

  - **Pairing** (`estia pair request`, `begin_pairing` in the Rust client,
    `examples/python/pair.py`) proves, in both directions, which certificate
    each end used. The device commits to a random nonce before it sees the
    engine's nonce, and both sides compute two 6-digit codes from the
    certificate fingerprint each of them used and the two nonces. You type
    the pairing code the device shows into `estia pair approve <id> --code
    <code>`; the engine approves only if it computed the same code from its
    own certificate, so something in between gets no token for the engine.
    `estia pair approve` then prints a confirmation code, which you type into
    the device; the device pins the certificate only if it computed the same
    code from the certificate it saw, so it never trusts an "approved"
    answered by something in between, with a token of its own making.
    Something between the device and the engine that presents its own
    certificate gets one guess in a million at each code per pairing and
    cannot prepare one in advance. The engine's certificate is public, so a
    code taken from the fingerprint alone could be matched by a certificate
    ground offline; the nonces are what prevent that.
  - **Requests with no code.** A browser or curl cannot compute a code, so
    it pairs without a commitment, and so does a device whose commitment
    something in between removed on the way. Such a request (`check: none`
    in `estia pair list`) is approved only when you say so explicitly:
    `estia pair approve <id> --no-code`, or `{"no_code": true}` on the HTTP
    approve route. Every other approval path refuses it. Never approve one
    without a code when the device in front of you shows a code: deny it.
    Even then that device would not pin the certificate, because the engine
    has no confirmation code to give for such a request; but whatever made
    the request would hold a token.
  - **Trust on first use, where there is no pairing exchange.** A browser
    (`/client`) cannot pin or show a code: it pairs without one (approved with
    `--no-code`), and you compare the SHA-256 in the browser's certificate
    viewer with `estia tls fingerprint` before accepting the warning. An app
    given a token from `estia token new` should ask for the fingerprint too
    (the command prints it), or show the 64-bit short form of what it sees
    and pin it only after you confirm it matches.
  - **Expiry.** The certificate is valid for 820 days from its creation:
    Apple's TLS stack refuses a server certificate valid for longer than 825
    days, even one the user trusts. Pinned devices ignore its dates. Browsers
    and clients that trust it as an authority (Node, curl `--cacert`) see it
    expire; `estia tls rotate` then makes a new one, and paired devices must
    pair again. Pins name the whole certificate, so it is never re-issued on
    its own.
  - **Rotation.** `estia tls rotate` makes a new certificate; every paired
    device must pair again, and their old tokens stay valid until you revoke
    them. A pinned device refuses any other certificate before sending its
    token, and says so.

  Loopback stays plain: a process on the engine's machine can watch its
  loopback traffic only with privileges that would let it read the data
  directory anyway.
- **Pairing trusts the operator's judgement.** Any device on the LAN can send a
  pairing request, with a name and scopes of its choosing. It gets a token only
  after you approve it with `estia pair approve <id> --code <code>` or an app
  that embeds Estia and holds an admin token (`--no-code`, or `no_code` over
  HTTP, for a request that has no code). Names are limited to printable letters,
  digits and a little punctuation so they cannot rewrite your terminal, and
  `estia pair approve` refuses a request for `admin` without
  `--allow-admin`. The HTTP approve route and the `/client` Admin tab grant
  exactly what was asked. Denying an approved pairing revokes its token.
- **Host names.** The server refuses a request whose `Host` is not an IP
  address, `localhost`, a `.local` name, the machine's hostname or a name you
  allowed (`--allow-host`, `ESTIA_ALLOWED_HOSTS`), and a cross-origin browser
  write. This is the defence against DNS rebinding. `--allow-host '*'` turns
  it off; do that only behind a reverse proxy that checks `Host` itself, and
  make any proxy forward the original `Host`.
- **Availability on the LAN.** There are caps on pending pairing requests (24,
  4 per address), request size (`max_tokens` 8192, 256 embedding inputs),
  header time (10 s, which also bounds the first byte and the TLS handshake of
  a connection) and connections (32 per address), but no request rate limit.
  A device on the LAN can still keep the engine busy with valid work or with
  TLS handshakes, or hold connections open with slowly sent request bodies.
- **Tokens only, no accounts.** Access is controlled by bearer tokens with
  scopes (`generate`, `embed`, `models:read`, `models:write`, `admin`).
  `tokens.json` stores SHA-256 hashes, not the tokens. Whoever can write to
  the data directory can mint tokens and approve pairings.
- **`--no-auth` is for testing.** It accepts requests without a token, and the
  server refuses it on any address but loopback. Every local process can then
  use the engine. Web pages cannot, thanks to the Host and Origin checks.
- **Downloaded code and weights.** `estia setup` downloads code and weights
  for the backend.
  - MLX: a Python build checked against a SHA-256 pinned in the source, then
    `mlx-vlm` (`>=0.6.13,<0.7`), `mlx-lm` (a floor only) and
    `mlx-embeddings` (`==0.1.0`) from PyPI with pip, with no hash check.
  - llama.cpp: upstream `llama-server` archives from GitHub, checked against
    SHA-256s pinned in the source. They are not signed or notarized on
    macOS. `ESTIA_LLAMA_SERVER` runs whatever binary you name.
  - Weights: MLX large (LFS) files are checked against the SHA-256 that
    Hugging Face publishes, small files by size. GGUF files are pinned to a
    commit and checked against their SHA-256.

  Estia trusts PyPI, GitHub releases and Hugging Face.
