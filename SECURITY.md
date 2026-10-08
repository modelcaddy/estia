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
    each end used, the way Bluetooth's numeric comparison does. The device
    commits to a random nonce before it sees the engine's nonce, and both
    sides compute one 6-digit code from the certificate fingerprint each of
    them used and the two nonces. Each shows its own, and neither ever sends
    its code to the other. `estia pair approve <id>` shows the engine's code
    and asks whether the device shows the same; the engine approves only on
    your yes, with that code (a script passes `--code`, the code the device
    shows, and the engine refuses one that differs from its own), so
    something in between gets no token for the engine. The device then asks
    whether the engine shows its code, and pins the certificate only on your
    yes, so it never trusts an "approved" answered by something in between,
    with a token of its own making. Something between the device and the
    engine that presents its own certificate makes the two codes differ,
    except by a one-in-a-million chance per pairing that it cannot improve
    by preparing in advance. The engine's certificate is public, so a code
    taken from the fingerprint alone could be matched by a certificate
    ground offline; the nonces are what prevent that.

    **What the code leaves to you.** The check is a person comparing two
    screens. An operator who approves, or a user who confirms on the device,
    without looking can pair a device with something in between: an
    approval gives it a token for the engine, a confirmation on the device
    sends it that device's prompts. Compare every digit, and answer no when
    unsure; a pairing is cheap to start again. Compare the code where it is
    labelled as the code (the `code:` line, the CODE column, the Code column
    of the Admin tab), never digits found elsewhere on the screen.

    Some of what shares those screens is chosen by whoever sent the request,
    and the rules below remove the direct ways to make it look like a code,
    not every way:

    - A device name may not hold a run of more than 4 digits, and only a
      letter with case (a-z, α-ω, а-я, …) ends a run: `phone 482 913` is
      refused, and so are digits split by a letter that prints as a blank, a
      dash or a dot (`phone 482ㅡ913`, the Hangul fillers). Letters that look
      like digits are still letters: a name such as `4B2 9I3`, or one in
      Cyrillic or Cherokee look-alikes, passes the rule and can still
      imitate a code for a careless glance.
    - The FROM column and the approve prompt show the address the request
      came from. On a LAN a sender can choose its own IPv6 address, digits
      included (`fe80::482:913`).
    - Pairing ids are 16 letters, none that passes for a digit, and a device
      (the Rust client, `pair.py`) refuses an id an engine would not make, so
      whatever answers a request cannot put a code on the device's screen as
      its id.
    - No approval path takes a request that has a code without that code:
      the HTTP route, the Admin tab and `--code` all need it. That ties an
      approval to the request whose code was shown and stops a codeless one
      (an old client, a commitment stripped on the way). It does not prove
      that anyone compared: an admin app or a script that sends back the code
      the engine listed, without asking anyone, approves without looking.
      Send the code only on a person's yes; a script that approves passes
      the code the device shows, never the one `estia pair list` shows.
    - On the device, `estia pair request`, `pair.py` and `remote_client.rs`
      take a yes only from a terminal: piped in, `y` is refused, and a
      script must give the six digits the engine shows. An app that calls
      `PendingPairing::confirm_match` on its own say-so, or wraps `estia
      pair request` and writes the device's own code back without asking
      anyone, removes the device's half of the check.
  - **Requests with no code.** A browser or curl cannot compute a code, so
    it pairs without a commitment, and so does a device whose commitment
    something in between removed on the way. Such a request (`check: none`
    in `estia pair list`) is approved only when you say so explicitly:
    `estia pair approve <id> --no-code`, or `{"no_code": true}` on the HTTP
    approve route. Every other approval path refuses it. Never approve one
    without a code when the device in front of you shows a code: deny it.
    The engine shows no code for such a request, so a device that shows one
    and asks whether the engine shows it too should get a no; but whatever
    made the request would hold a token.
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
  after you approve it with `estia pair approve <id>` (once the device shows
  the same code as the engine) or an app that embeds Estia and holds an admin
  token (`--no-code`, or `no_code` over HTTP, for a request that has no
  code). Names are limited to printable letters, digits (no more than 4 in a
  row) and a little punctuation so they cannot rewrite your terminal or pass
  for a code, and
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
