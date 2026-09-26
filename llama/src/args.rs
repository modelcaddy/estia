//! The adapter's command line, shared by the `estia-llama` binary and the
//! CLI's hidden `estia runner llama`:
//!
//! ```text
//! --server <llama-server> --run-dir <dir> [--ctx <n>] [-- <llama-server args>…]
//! ```

use crate::AdapterOptions;
use anyhow::{anyhow, bail, Context};
use std::ffi::OsString;
use std::path::PathBuf;

/// Help text for the arguments [`parse_args`] accepts.
pub const USAGE: &str = "\
usage: estia-llama --server <llama-server> --run-dir <dir> [--ctx <n>] [-- <llama-server args>...]

Speaks Estia's runner protocol v2 on stdin/stdout and serves it with one
upstream llama-server process (docs/protocol.md).

  --server <path>   the llama-server executable to start
  --run-dir <dir>   private directory for the socket, API-key file and pid
                    record (created 0700 if missing); keep the path short
  --ctx <n>         context length for generation models (default 8192)
  -- <args>...      extra llama-server arguments, e.g. `-- -ngl 0`";

/// llama-server flags the adapter owns or never allows: where it listens, how
/// it authenticates, and features that fetch from the network or give the
/// model file and shell access (docs/design/llama-backend.md).
const REFUSED_EXTRA: &[&str] = &[
    "--host",
    "--port",
    "--path",
    "--api-key",
    "--api-key-file",
    "--ui",
    "--webui",
    "--slots",
    "--tools",
    "--agent",
    "--mcp-servers-config",
    "--mcp-servers-json",
    "--media-path",
    "--models-dir",
    "--models-preset",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "-mu",
    "--model-url",
    "--ssl-key-file",
    "--ssl-cert-file",
];

/// Parse the adapter's arguments: everything after the program name (and
/// after `runner llama` for the CLI).
pub fn parse_args<I, T>(args: I) -> anyhow::Result<AdapterOptions>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut server_bin: Option<PathBuf> = None;
    let mut run_dir: Option<PathBuf> = None;
    let mut context_length: Option<u32> = None;
    let mut extra_args = Vec::new();

    let mut it = args.into_iter().map(Into::into);
    while let Some(arg) = it.next() {
        let text = arg.to_str().ok_or_else(|| anyhow!("argument is not valid UTF-8: {}", arg.to_string_lossy()))?;
        if text == "--" {
            for rest in it.by_ref() {
                let rest = rest.into_string().map_err(|a| anyhow!("llama-server argument is not valid UTF-8: {}", a.to_string_lossy()))?;
                extra_args.push(rest);
            }
            break;
        }
        let (flag, inline) = match text.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(OsString::from(v))),
            _ => (text, None),
        };
        let flag = flag.to_string();
        let mut value = || -> anyhow::Result<OsString> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => it.next().ok_or_else(|| anyhow!("{flag} needs a value")),
            }
        };
        match flag.as_str() {
            "--server" => set_once(&mut server_bin, "--server", PathBuf::from(value()?))?,
            "--run-dir" => set_once(&mut run_dir, "--run-dir", PathBuf::from(value()?))?,
            "--ctx" => {
                let v = value()?;
                let n: u32 = v
                    .to_str()
                    .and_then(|s| s.parse().ok())
                    .filter(|n| *n > 0)
                    .with_context(|| format!("--ctx wants a positive whole number, not `{}`", v.to_string_lossy()))?;
                set_once(&mut context_length, "--ctx", n)?;
            }
            other => bail!("unknown argument `{other}`"),
        }
    }

    for a in &extra_args {
        let name = a.split_once('=').map(|(n, _)| n).unwrap_or(a);
        if REFUSED_EXTRA.contains(&name) {
            bail!("`{name}` cannot be passed to llama-server: the adapter sets it, or never allows it");
        }
    }

    let server_bin = server_bin.ok_or_else(|| anyhow!("--server <path to llama-server> is required"))?;
    let run_dir = run_dir.ok_or_else(|| anyhow!("--run-dir <dir> is required"))?;
    if server_bin.as_os_str().is_empty() || run_dir.as_os_str().is_empty() {
        bail!("--server and --run-dir must not be empty");
    }
    Ok(AdapterOptions { server_bin, run_dir, context_length, extra_args })
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, v: T) -> anyhow::Result<()> {
    if slot.is_some() {
        bail!("{flag} given twice");
    }
    *slot = Some(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> anyhow::Result<AdapterOptions> {
        parse_args(args.iter().copied())
    }

    #[test]
    fn the_contract_parses() {
        let o = parse(&["--server", "/opt/llama/llama-server", "--run-dir", "/d/run"]).unwrap();
        assert_eq!(o.server_bin, PathBuf::from("/opt/llama/llama-server"));
        assert_eq!(o.run_dir, PathBuf::from("/d/run"));
        assert_eq!(o.context_length, None);
        assert!(o.extra_args.is_empty());

        let o = parse(&["--run-dir=/r", "--ctx", "4096", "--server=/s", "--", "-ngl", "0", "--ctx", "x"]).unwrap();
        assert_eq!(o.server_bin, PathBuf::from("/s"));
        assert_eq!(o.run_dir, PathBuf::from("/r"));
        assert_eq!(o.context_length, Some(4096));
        // After `--` everything is llama-server's, even our own flag names.
        assert_eq!(o.extra_args, vec!["-ngl", "0", "--ctx", "x"]);
    }

    #[test]
    fn owned_strings_and_os_strings_work_too() {
        let v: Vec<String> = vec!["--server".into(), "/s".into(), "--run-dir".into(), "/r".into()];
        assert!(parse_args(v).is_ok());
        let v: Vec<OsString> = vec!["--server".into(), "/s".into(), "--run-dir".into(), "/r".into()];
        assert!(parse_args(v).is_ok());
    }

    #[test]
    fn mistakes_are_errors() {
        let err = |a: &[&str]| parse(a).unwrap_err().to_string();
        assert!(err(&["--run-dir", "/r"]).contains("--server"));
        assert!(err(&["--server", "/s"]).contains("--run-dir"));
        assert!(err(&["--server"]).contains("needs a value"));
        assert!(err(&["--server", "/s", "--run-dir", "/r", "--ctx", "0"]).contains("--ctx"));
        assert!(err(&["--server", "/s", "--run-dir", "/r", "--ctx", "-5"]).contains("--ctx"));
        assert!(err(&["--server", "/s", "--run-dir", "/r", "--bogus"]).contains("unknown argument"));
        assert!(err(&["--server", "/s", "--server", "/t", "--run-dir", "/r"]).contains("twice"));
        assert!(err(&["--server", "/s", "--run-dir", "/r", "positional"]).contains("unknown argument"));
    }

    #[test]
    fn owned_llama_server_flags_are_refused() {
        for bad in ["--host", "--port=9", "--api-key", "--tools", "--agent", "-hf", "--ui"] {
            let e = parse(&["--server", "/s", "--run-dir", "/r", "--", bad, "x"]).unwrap_err().to_string();
            assert!(e.contains("cannot be passed"), "{bad}: {e}");
        }
        assert!(parse(&["--server", "/s", "--run-dir", "/r", "--", "-ngl", "99", "--flash-attn", "on"]).is_ok());
    }
}
