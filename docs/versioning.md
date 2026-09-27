# Versioning

How Estia numbers its releases, which changes count as breaking, how to tell
which build is running, and how a release is made.

## Which build is running

`estia --version` prints one line: the version, the commit it was built from
and the UTC day it was built.

```console
$ estia --version
estia 0.4.0 (8642bfc4e+dirty, 2026-09-26)
```

`estia version` prints the rest of what identifies a build. It reads no data
directory or config, so it works before `estia setup` and on a machine with no
models.

```console
$ estia version
version  : 0.4.0
commit   : 8642bfc4e+dirty
date     : 2026-09-26
target   : aarch64-apple-darwin (debug build)
rustc    : 1.95.0 (59807616e 2026-04-14)
api      : v1 (HTTP /engine/*)
protocol : v2 (runners)
backends : mlx-python (default on this machine), llama-cpp
features : python-mlx, llama-runtime
llama.cpp: b11146 (pinned for `estia runtime install --backend llama`)
runner   : mlx-python 2.3.0 (compiled in)
```

- `commit`: the first 9 hex digits of the git commit. `+dirty` means files
  that are compiled in (the crates, `runners/`, `Cargo.toml`, `Cargo.lock`)
  differed from that commit when it was built. The release tarball, built
  from a clean checkout of the tag, never has it.
  `unknown` means the build had no way to find the commit (see
  [Build information](#build-information)).
- `target`: the Rust target triple, and `debug` or `release`. Timings from a
  debug build are not comparable with a release build's.
- `backends`: both backends are compiled into every build; this says which one
  the machine defaults to and whether MLX can run here (Apple Silicon only).
  The backend a data directory actually uses is in `estia status`.
- `features`: the `estia-engine` features the binary is built with.
  `python-mlx` and `llama-runtime` are the runtime installers behind
  `estia runtime install`.
- `runner`: the version of the MLX runner script compiled into the binary.
  The runner on disk beside a release binary is the same file.

`estia version --json` prints the same as one JSON object, for scripts and bug
reports:

```console
$ estia version --json
{
  "api_version": 1,
  "backends": [
    {
      "default": true,
      "id": "mlx-python",
      "supported": true
    },
    {
      "default": false,
      "id": "llama-cpp",
      "supported": true
    }
  ],
  "build": {
    "commit": "8642bfc4e+dirty",
    "date": "2026-09-26",
    "profile": "debug",
    "rustc": "1.95.0 (59807616e 2026-04-14)",
    "target": "aarch64-apple-darwin"
  },
  "features": [
    "python-mlx",
    "llama-runtime"
  ],
  "llama_cpp_build": "b11146",
  "mlx_runner_version": "2.3.0",
  "protocol_version": 2,
  "version": "0.4.0"
}
```

One field, with `jq`:

```console
$ estia version --json | jq -r .build.commit
8642bfc4e+dirty
```

A running engine reports its own build. `/engine/health` needs no token:

```console
$ curl -s http://127.0.0.1:27200/engine/health | jq '{version, build}'
{
  "version": "0.4.0",
  "build": {
    "commit": "8642bfc4e+dirty",
    "date": "2026-09-26"
  }
}
```

`version`, `build.commit` and `build.date` mean the same in both outputs, so a
script can check that a service runs the binary it expects. The engine's
startup log line carries the version and commit too (see
[logging.md](logging.md)):

```text
INFO estia_server: estia serving version=0.4.0 commit=8642bfc4e+dirty api_version=1 protocol_version=2 ...
```

Other places show only the version: `estia dashboard`, the `health` line of
`estia remote-check`, and `estia discover`, which reads the `engine_version`
that the Bonjour advertisement carries.

`estia service install` copies the binary into the data directory, so the
service keeps running the build that was installed, not the one you rebuilt
since. Ask the service itself (`/engine/health`), or run the copy:
`"<data dir>/engine/estia" --version`.

When you report a bug, include the output of `estia version`, or of
`/engine/health` for a remote engine.

## The numbers

Estia carries several version numbers. They change independently.

| Number | Where | Now | Changes when |
|---|---|---|---|
| Estia version | `version` in the root `Cargo.toml`, shared by every crate | 0.4.0 | every release |
| API version | `estia_server::API_VERSION`, `api_version` in `/engine/health` and the Bonjour record | 1 | an incompatible change to `/engine/*` |
| Protocol version | `estia_proto::PROTOCOL_VERSION`, `protocol_version` in `/engine/health` | 2 | an incompatible change between the engine and its runners |
| MLX runner version | `RUNNER_VERSION` in `runners/mlx-python/estia-runner.py`, sent in the runner's `hello` | 2.3.0 | a change to the runner's behaviour |
| llama.cpp adapter version | the `estia-llama` crate version, sent in its `hello` | 0.4.0 | with Estia |
| llama.cpp build | `LLAMA_BUILD` in `engine/src/runtime/llama_pins.rs` | b11146 | when the pin is bumped |

All crates in the workspace (`estia-proto`, `estia-engine`, `estia-llama`,
`estia-server`, `estia`) have the same version and are released together. They
depend on each other with `version = "0.4.0"`, which Cargo reads as 0.4.0 or a
later 0.4.x. An application that embeds them should use the same version of
each.

## Semantic versioning before 1.0

Estia follows [Semantic Versioning](https://semver.org) as Cargo applies it to
0.x versions:

- **Patch release** (0.4.0 to 0.4.1): fixes, and additions that nothing
  written against 0.4.0 can notice.
- **Minor release** (0.4 to 0.5): may include breaking changes, each listed in
  the CHANGELOG. Large additions, such as a new backend, and a raised minimum
  Rust version also go in a minor release.
- **From 1.0**, breaking changes need a major release.

A change is breaking when it can stop something written against the previous
release from working. For Estia that covers:

- **The HTTP API.** Routes under `/v1/*` and `/engine/*`, their request and
  response fields, status codes and error `type` values, the scope each route
  needs, the server-sent event shapes, `x_estia`, and the `X-Request-Id`
  header. Removing or renaming any of these, changing a field's type or
  meaning, making an optional request field required, or requiring a stronger
  scope is breaking. Adding a route, an optional request field or a response
  field is not; clients must ignore fields they do not know.
- **The runner protocol.** The JSON lines between the engine and its runners
  ([protocol.md](protocol.md)). A change that an existing runner or engine
  cannot handle is breaking and bumps the protocol version. A new optional
  field, or a new capability announced in `hello`, is not.
- **Files in the data directory.** `config.json` (roles, backend),
  `tokens.json` (token names, hashes, scopes), `pairings.json`, and the layout
  of `models/` and `runtime/`. A new release must read what an older release
  wrote, without the user editing anything. A change that leaves installed
  models or runtimes unusable is breaking.
- **The CLI.** Commands, flags, their defaults, exit codes, the `ESTIA_*`
  environment variables, the default port 27200, and output meant for
  scripts: `--json` output such as `estia version --json`. Output meant for
  people (tables, `key : value` lines, messages) may change in any release;
  scripts should use `--json` or the HTTP API. Log messages and fields
  ([logging.md](logging.md)) may change too, and a renamed field is listed in
  the CHANGELOG.
- **Embedding fingerprints.** A fingerprint (`<artifact id>@<backend>`) names
  the vectors a model produces, and clients store it with their index. A
  change that alters the vectors behind an existing fingerprint (other
  weights, task prefixes, pooling, normalisation, truncation) must come with a
  new artifact id and so a new fingerprint. Changing the fingerprint of a
  model that already has one is breaking, because stored indexes stop matching
  `expect_fingerprint`.
- **Discovery.** The Bonjour service type `_estia._tcp` and the keys of its
  record (`api_version`, `engine_version`, `protocol_version`).
- **The Rust API** of the library crates: their public items, under
  [Cargo's SemVer rules](https://doc.rust-lang.org/cargo/reference/semver.html).
- **The model registry.** Models may be added in any release. Removing one,
  or changing which model a role resolves to by default, goes in a minor
  release.

## API version and protocol version

The API version and the protocol version are separate from the Estia version
and change far less often. A client that talks to several engines should check
them rather than the Estia version.

**`api_version`** (now 1) is the version of the `/engine/*` contract.
`/engine/health` reports it without a token, and the Bonjour record carries
it, so a client can check an engine before it pairs. It goes up by one when an
`/engine/*` change is breaking in the sense above. Additions do not change it.
Raising it always means a minor (or, after 1.0, major) Estia release. The
`/v1/*` routes follow OpenAI's shapes and have no number of their own; a
breaking change to them, or to Estia's additions such as `x_estia`, also needs
a minor release.

**`protocol_version`** (now 2) is the version of the runner protocol
(`estia_proto::PROTOCOL_VERSION`). It goes up only for a change that an
existing runner or engine cannot handle. The engine still works with version 1
runners: a runner that does not answer `hello` is treated as version 1, with no
capabilities. New optional fields and capabilities keep the number. When it
changes, both runners change in the same pull request: the MLX runner and the
llama.cpp adapter (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

The MLX runner has its own `RUNNER_VERSION`, bumped on any change to its
behaviour that a client could care about, so a runner's `hello` and the
`runner handshake` log line say which script answered. The llama.cpp adapter
reports the Estia version.

## Minimum supported Rust version

The minimum is `rust-version` in the root `Cargo.toml`, now 1.89 (for
`std::fs::File::lock`). The `msrv` job in `.github/workflows/ci.yml` builds
the workspace with exactly that version. It is raised only when a change needs
a newer Rust, only in a minor release, and the CHANGELOG says so. To check it
locally:

```bash
rustup toolchain install 1.89.0 --profile minimal
cargo +1.89.0 check --workspace --all-targets --all-features --locked
```

Separately, CI and the release build use one pinned Rust release,
`RUST_TOOLCHAIN` in `.github/workflows/ci.yml` and `release.yml` (now 1.95.0),
so a new stable Rust cannot fail CI with lints that nobody has seen locally.
Bump it on purpose: install the new version, run `cargo clippy --workspace
--all-targets --locked -- -D warnings` with it, fix what it reports, and change
both files in the same commit.

## Bumping the llama.cpp pin

Estia installs one llama.cpp build per release, named in
`engine/src/runtime/llama_pins.rs`. To move to another build:

1. List the new release's archives with their sizes and SHA-256 digests:

   ```bash
   gh api repos/ggml-org/llama.cpp/releases/tags/<build> \
     --jq '.assets[] | [.name, .size, .digest] | @tsv'
   ```

2. In `llama_pins.rs`, change `LLAMA_BUILD`, `LLAMA_RELEASE_BASE`, and the
   `file`, `bytes` and `sha256` of every `LlamaAsset`, including the `cudart`
   archives. Keep the source comment at the top in step. `cargo test -p
   estia-engine pins_are_well_formed` checks the shape.
3. Open a pull request. The `llama` job in CI reads the pin from
   `llama_pins.rs`, downloads the Linux x64 CPU and macOS arm64 Metal archives,
   checks their SHA-256 against the file, and runs the adapter's tests against
   them. A hash that does not match, or a `llama-server` change the tests
   catch, fails there.
4. Run a real model through the new build on the machines you can
   (`estia --backend llama runtime install`, then
   `echo hello | estia --backend llama chat --model fast`), and note the
   change, with the old and new build, in the CHANGELOG. Update the build the
   README names, in its text and in example output.

A new pin does not by itself make a release breaking. Installed runtimes of the
old build keep working until `estia runtime install` is run again.

## Release checklist

1. **Version.** Change `version` under `[workspace.package]` in the root
   `Cargo.toml`, and the `version = "…"` of the six path dependencies between
   the crates (in the `engine`, `llama`, `server` and `cli` manifests). Run
   `cargo build` so `Cargo.lock` follows, and commit it: releases build with
   `--locked`. After a minor bump, a requirement left at the old version fails
   to resolve; after a patch bump it still resolves, so check them all:

   ```bash
   grep -n '^version = \|path = "\.\./[a-z]*", version' Cargo.toml */Cargo.toml
   ```

2. **Changelog.** Move the entries under `## Unreleased` in `CHANGELOG.md`
   into a new `## X.Y.Z — YYYY-MM-DD` section, and leave `## Unreleased` empty.
   Give breaking changes their own heading.
3. **Other docs.** `git grep -n` the previous version: the README's status
   section, example output in `docs/`, and the version `ROADMAP.md` was last
   reviewed against. Examples of output that include the old version may stay
   if they are still accurate in every other way.
4. **Benchmarks.** If the release changes speed, add rows to `BENCH.md` from a
   release build (`cargo build --release`), and say which build it was
   (`estia version`).
5. **Checks.** The same as CI, from a clean tree:

   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --locked -- -D warnings
   cargo test --workspace --locked
   cargo package --workspace --locked
   cargo deny check
   cargo +1.89.0 check --workspace --all-targets --all-features --locked
   ```

   Then the live adapter tests against the pinned `llama-server` and the two
   small models the `llama` CI job uses (it shows where to get them). They
   skip themselves when a variable is missing, so check that nothing printed
   `skipping`:

   ```bash
   ESTIA_LLAMA_SERVER=/path/to/llama-server \
   ESTIA_LLAMA_TEST_MODEL=/path/to/tinygemma3-Q8_0.gguf \
   ESTIA_LLAMA_TEST_EMBED_MODEL=/path/to/all-MiniLM-L6-v2-Q8_0.gguf \
   cargo test -p estia-llama --locked -- --show-output
   ```

   And one real model on each backend you can run, with a release build:
   `estia setup`, `estia serve`, then `scripts/smoke-test.sh` against it
   (12 checks; [running-and-testing.md](running-and-testing.md#the-smoke-test)).
   Every check should pass.
6. **Tag.** On the release commit, with a clean tree:

   ```bash
   git tag -a vX.Y.Z -m "Estia X.Y.Z"
   git push origin vX.Y.Z
   ```

   A tag with a hyphen (`v0.5.0-rc.1`) makes a pre-release; the crate version
   must then be `0.5.0-rc.1` as well.
7. **Check the release** as described below. The workflow leaves the release
   notes empty; add a link to the CHANGELOG section.

### What the release workflow produces

Pushing a `v*` tag runs `.github/workflows/release.yml` on a pinned macOS
Apple Silicon runner. It:

1. stops unless the tag, minus the `v`, equals the version of the `estia`
   crate;
2. builds `cargo build --release --locked -p estia --target
   aarch64-apple-darwin`, for macOS 11 and later;
3. packs `estia-X.Y.Z-aarch64-apple-darwin.tar.gz`: the `estia` binary,
   `runners/mlx-python/*.py`, `LICENSE`, `NOTICE`, `README.md` and
   `THIRD_PARTY_LICENSES` (the licences of the Rust crates in the binary),
   and writes its SHA-256 to `estia-X.Y.Z-aarch64-apple-darwin.tar.gz.sha256`;
4. checks the tarball it just made: the hash, the files, `estia --version`
   and `estia version`, that `estia version --json` names the tag's commit
   (the first 9 hex digits, without `+dirty`), and that `estia status`, run
   from an unrelated directory, finds the runner inside the tarball;
5. creates a GitHub release for the tag with the two files, marked as a
   pre-release when the tag has a hyphen.

There is no Linux or Intel macOS archive yet. The binary is not signed with a
Developer ID or notarized: macOS blocks a copy downloaded with a browser until
its quarantine attribute is removed, while a copy downloaded with `curl` is
not quarantined.

### Checking a release download

```bash
curl -fLO https://github.com/modelcaddy/estia/releases/download/v0.4.0/estia-0.4.0-aarch64-apple-darwin.tar.gz
curl -fLO https://github.com/modelcaddy/estia/releases/download/v0.4.0/estia-0.4.0-aarch64-apple-darwin.tar.gz.sha256
shasum -a 256 -c estia-0.4.0-aarch64-apple-darwin.tar.gz.sha256
tar -xzf estia-0.4.0-aarch64-apple-darwin.tar.gz
./estia-0.4.0-aarch64-apple-darwin/estia --version
```

`shasum` prints `estia-0.4.0-aarch64-apple-darwin.tar.gz: OK` (on Linux,
`sha256sum -c` does the same). `--version` must name the tag's commit, with no
`+dirty`:

```bash
git rev-parse 'v0.4.0^{commit}' | cut -c1-9
```

The two `curl` lines have not been run: no release existed when this page was
written. The other commands were run on a tarball with the same layout, built
locally.

## Build information

A build script (`cli/build.rs`; `server/build.rs` links to it) embeds the
commit, the date, the target, the build profile and the compiler version. It
never fails the build. The commit comes from the first of:

1. `ESTIA_BUILD_COMMIT` in the environment of the build, for a build from a
   source archive without `.git`. Letters, digits and `.+-_`, at most 64
   characters; anything else is ignored.
2. git, when the crate is inside a git checkout of the Estia workspace itself.
   A crate copied into another repository, or unpacked by `cargo package`, is
   not, and does not take that repository's commit.
3. `.cargo_vcs_info.json`, which `cargo package` writes into every published
   crate, so a crate built from crates.io knows its commit. `+dirty` there
   means it was packaged from a dirty tree.
4. Otherwise `unknown`.

The date is the UTC day the build script ran. With `SOURCE_DATE_EPOCH` set (a
Unix time, as reproducible builds use), it is that day instead:

```console
$ ESTIA_BUILD_COMMIT=src-0.4.0 SOURCE_DATE_EPOCH=1790467200 cargo build -p estia
$ target/debug/estia --version
estia 0.4.0 (src-0.4.0, 2026-09-27)
```

The script runs again when the commit moves (commit, checkout, reset), when a
compiled-in file changes, or when either variable changes. A rebuild that
changes nothing reuses the earlier date.

## Why the first tagged version is 0.4.0

Estia's history holds four milestones before its first tag, and each would
have been a release of its own:

| Version | Date | Milestone |
|---|---|---|
| 0.1.0 | 2026-09-08 | The engine and the CLI: runner sessions, the model registry and downloader, roles, structured output, MLX through a Python runner |
| 0.2.0 | 2026-09-26 | The LAN daemon: `estia serve` with the OpenAI-compatible and `/engine/*` APIs, pairing, Bonjour, the login service, the remote client, runner protocol v2 |
| 0.3.0 | 2026-09-26 | Hardening, logs and request ids, examples and the builders' guide, the documentation, CI and the release workflow |
| 0.4.0 | 2026-09-27 | The llama.cpp backend, and the build information above |

Before 0.4.0, every build reported 0.0.1 or 0.1.0 whatever it contained. The
CHANGELOG sections for 0.1.0 to 0.3.0 were written afterwards from the git
history; those versions were never tagged or published.

## What was run for this page

Every output above comes from a debug build of the commit it names, with
uncommitted changes (hence `+dirty`), on an Apple Silicon Mac. The health and
log examples come from an engine on a spare port (not 27200) with its own data
directory.
The following were not run, and why:

- `git tag` and `git push`: making a release is the maintainer's decision.
- The two `curl` downloads: there was no release yet.
- The live adapter tests and `estia --backend llama chat`: they need the
  pinned `llama-server` and GGUF models, which were not downloaded for this
  page.
- `"<data dir>/engine/estia" --version`: the service on the machine was in use
  by other tests.
- `cargo package --workspace --locked` ran as `cargo package --workspace
  --allow-dirty --offline --locked`, because the tree had uncommitted changes.

The release workflow's package and check steps were run locally, on a new git
repository holding a copy of this tree, with a debug build instead of a
release build and a placeholder `THIRD_PARTY_LICENSES` (cargo-about was not
installed). The clean build reported its commit without `+dirty` and the
check passed; the same check with a `+dirty` binary stopped with exit code 1.
GitHub has not run the workflow: no tag has been pushed.
