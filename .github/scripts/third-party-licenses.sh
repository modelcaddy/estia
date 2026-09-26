#!/usr/bin/env bash
# Write THIRD_PARTY_LICENSES for the `estia` binary with cargo-about (config:
# about.toml, template: about.hbs), and fail when the result is incomplete:
#
# - a `clarify` entry in about.toml whose checksum no longer matches, or a
#   licence file that could not be fetched: cargo-about only warns about these;
# - a crate that ships no licence file and has no `clarify` entry: cargo-about
#   then prints the bare SPDX template, whose "<year>" placeholder stands where
#   the crate's copyright line should be.
#
# Run from the repository root. Used by the CI and release workflows.
# Usage: .github/scripts/third-party-licenses.sh <output file>
set -euo pipefail

out="${1:?usage: $0 <output file>}"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if ! cargo about generate --locked --fail -c about.toml -m cli/Cargo.toml about.hbs -o "$out" 2>"$log"; then
  cat "$log" >&2
  exit 1
fi
cat "$log" >&2
if grep -qE "WARN|ERROR" "$log"; then
  echo "cargo-about reported a problem (above); fix about.toml" >&2
  exit 1
fi
if grep -n "<year>" "$out" >&2; then
  echo "a crate ships no licence file, so its copyright line is missing from $out;" >&2
  echo "add a clarify entry for it to about.toml" >&2
  exit 1
fi
echo "wrote $out" >&2
