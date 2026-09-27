# Contributing to Estia

Thanks for helping. This file covers how to build, test and send a change.

## Build

You need Rust 1.89 or newer: the `rust-version` in `Cargo.toml`, which CI
builds with. Raise it only when a change needs a newer Rust, and say so in the
pull request.

```bash
cargo build
cargo run -p estia -- --help
```

The workspace builds on macOS and Linux (on Linux, also install a C compiler,
`pkg-config` and the OpenSSL headers). `cargo run -p estia -- setup` installs
the backend's runtime and the default models: MLX on an Apple Silicon Mac,
llama.cpp everywhere else, or llama.cpp on a Mac with `--backend llama`.

Pass `--data-dir <dir>` (or set `ESTIA_DATA_DIR`) to keep a development
engine's models, tokens and config apart from an installed one.
[docs/running-and-testing.md](docs/running-and-testing.md#running-a-development-engine-beside-an-installed-one)
shows how, and how to check it with `scripts/smoke-test.sh`.

## Test

```bash
cargo test --workspace
```

The tests drive a fake runner written in Python, so they need `python3` on
`PATH`. Without it, those tests print `skip:` and pass. They do not need MLX
or a model.

Some tests are marked `#[ignore]` because they use the network: the model
download tests reach huggingface.co, and the Bonjour tests advertise on the
LAN. Run them with `cargo test -p <crate> -- --ignored` when you change that
code.

## Format and lint

CI runs these, and a pull request must pass them:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo deny check
cargo +1.89.0 check --workspace --all-targets --all-features --locked   # the minimum Rust version
```

`rustfmt.toml` sets the line width. `deny.toml` sets the policy for the Rust
dependency graph: permissive licences only (no GPL, AGPL, LGPL or other
copyleft) and crates.io sources only. It does not cover the Python packages
the MLX runtime installs at run time; the README's
[Licence](README.md#licence) section lists those. Install the tool with
`cargo install cargo-deny --locked`.

Release tarballs include `THIRD_PARTY_LICENSES`, the licence texts of the
crates compiled into `estia`, built by
[cargo-about](https://github.com/EmbarkStudios/cargo-about) from `about.toml`.
Its `accepted` list mirrors the `allow` list in `deny.toml`, so change the two
together. CI also builds the file, and fails when a dependency ships no
licence file or a checksum in `about.toml` no longer matches; the fix is a
`clarify` entry in `about.toml`. To run the same check locally (install the
tool with `cargo install cargo-about --locked --features cli`):

```bash
.github/scripts/third-party-licenses.sh target/THIRD_PARTY_LICENSES
```

## Changes to the runner protocol

`proto` defines the wire format between the engine and its runners. If you
change it, change both resident runners in the same pull request:
`runners/mlx-python/estia-runner.py` and the llama.cpp adapter in `llama/`.
Bump `PROTOCOL_VERSION` only for an incompatible change.

The adapter's integration tests (`llama/tests/adapter.rs`) run against a real
`llama-server` when `ESTIA_LLAMA_SERVER`, `ESTIA_LLAMA_TEST_MODEL` and
`ESTIA_LLAMA_TEST_EMBED_MODEL` are set, and skip otherwise. The `llama` job in
`.github/workflows/ci.yml` shows where to get the pinned build and the two
small test models.

## Checking a change against a real model

CI does not run a model through `estia serve`. When you change the server,
the engine or a runner, run a development engine with a real model and the
smoke test against it (`scripts/smoke-test.sh`; see
[docs/running-and-testing.md](docs/running-and-testing.md#the-smoke-test)),
and say in the pull request which backend and model it ran with, and the
output of `estia version`.

## Versions and releases

[docs/versioning.md](docs/versioning.md) says what counts as a breaking
change, when the API and protocol versions change, and how a release is made.
Add a line under `## Unreleased` in `CHANGELOG.md` for a change users or
client authors would notice.

## Commits and pull requests

- Keep a pull request to one change, and explain why in its description.
- Prefer `type(scope): summary` for commit subjects, for example
  `docs(server): describe the token scopes`. Types: `feat`, `fix`, `docs`,
  `refactor`, `test`, `chore`. Scopes: `engine`, `server`, `cli`, `proto`,
  `llama`, `runners`, `client`, `ci`.
- Add or update a test when you change behaviour.

For a large change, open an issue first so we can agree on the approach.

## Licence

Estia is licensed under Apache-2.0. By submitting a contribution you agree
that it is licensed under the same terms, as section 5 of the licence says.
There is no CLA and no sign-off requirement.

Do not report security problems in a public issue. See [SECURITY.md](SECURITY.md).
