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

- **No TLS on the LAN.** `estia serve --lan` serves plain HTTP. Bearer tokens,
  prompts and model output cross the network unencrypted. Anyone who can
  observe that traffic can read it and reuse a token. Use LAN mode only on a
  network you trust, such as your home network, not shared or public Wi-Fi.
  Note that `estia service install` runs `estia serve --lan` unless you pass
  `--local`.
- **Pairing trusts the operator's judgement.** Any device on the LAN can send a
  pairing request, with a name and scopes of its choosing. It gets a token only
  after you approve it with `estia pair approve <id>` or a host app. Names are
  limited to printable letters, digits and a little punctuation so they cannot
  rewrite your terminal, and `estia pair approve` refuses a request for
  `admin` without `--allow-admin`. The HTTP approve route and the `/client`
  Admin tab grant exactly what was asked. Denying an approved pairing revokes
  its token.
- **Host names.** The server refuses a request whose `Host` is not an IP
  address, `localhost`, a `.local` name, the machine's hostname or a name you
  allowed (`--allow-host`, `ESTIA_ALLOWED_HOSTS`), and a cross-origin browser
  write. This is the defence against DNS rebinding. `--allow-host '*'` turns
  it off; do that only behind a reverse proxy that checks `Host` itself, and
  make any proxy forward the original `Host`.
- **Availability on the LAN.** There are caps on pending pairing requests (24,
  4 per address), request size (`max_tokens` 8192, 256 embedding inputs),
  header time (10 s) and connections (32 per address), but no request rate
  limit. A device on the LAN can still keep the engine busy with valid work,
  or hold connections open with slowly sent request bodies.
- **Tokens only, no accounts.** Access is controlled by bearer tokens with
  scopes (`generate`, `embed`, `models:read`, `models:write`, `admin`).
  `tokens.json` stores SHA-256 hashes, not the tokens. Whoever can write to
  the data directory can mint tokens and approve pairings.
- **`--no-auth` is for testing.** It accepts requests without a token, and the
  server refuses it on any address but loopback. Every local process can then
  use the engine. Web pages cannot, thanks to the Host and Origin checks.
- **Downloaded code and weights.** The MLX backend runs on a Python runtime
  that `estia setup` downloads. The Python build is checked against a SHA-256
  pinned in the source. The MLX packages are then installed from PyPI with
  pip, with minimum versions and no hash check. Model weights come from
  Hugging Face; large (LFS) files are checked against the SHA-256 that
  Hugging Face publishes. Estia trusts PyPI and Hugging Face.
