# Contributing to Estia

Thanks for helping. This file covers how to build, test and send a change.

## Build

You need a stable Rust toolchain.

```bash
cargo build
cargo run -p estia -- --help
```

The workspace builds on macOS and Linux. Running a real model needs an Apple
Silicon Mac, because MLX is the only backend today. There, `cargo run -p estia
-- setup` installs the Python MLX runtime and the default models.

Pass `--data-dir <dir>` (or set `ESTIA_DATA_DIR`) to keep a development
engine's models, tokens and config apart from an installed one.

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
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check
```

`rustfmt.toml` sets the line width. `deny.toml` sets the dependency policy:
permissive licences only (no GPL, AGPL, LGPL or other copyleft) and crates.io
sources only. Install the tool with `cargo install cargo-deny --locked`.

## Changes to the runner protocol

`proto` defines the wire format between the engine and its runners. If you
change it, change `runners/mlx-python/estia-runner.py` in the same pull
request. Bump `PROTOCOL_VERSION` only for an incompatible change.

## Commits and pull requests

- Keep a pull request to one change, and explain why in its description.
- Write commit subjects as `type(scope): summary`, for example
  `docs(server): describe the token scopes`. Types: `feat`, `fix`, `docs`,
  `refactor`, `test`, `chore`. Scopes: `engine`, `server`, `cli`, `proto`,
  `runners`, `client`, `ci`.
- Add or update a test when you change behaviour.

For a large change, open an issue first so we can agree on the approach.

## Licence

Estia is licensed under Apache-2.0. By submitting a contribution you agree
that it is licensed under the same terms, as section 5 of the licence says.
There is no CLA and no sign-off requirement.

Do not report security problems in a public issue. See [SECURITY.md](SECURITY.md).
