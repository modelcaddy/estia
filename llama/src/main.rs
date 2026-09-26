//! `estia-llama`: the llama.cpp runner adapter as its own binary. The `estia`
//! CLI runs the same code as `estia runner llama`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    // Only our own flags: everything after `--` belongs to llama-server.
    let ours = args.iter().take_while(|a| *a != "--");
    for a in ours {
        if a == "-h" || a == "--help" {
            println!("{}", estia_llama::USAGE);
            return ExitCode::SUCCESS;
        }
        if a == "-V" || a == "--version" {
            println!("{} {}", estia_llama::RUNNER, env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
    }
    let opts = match estia_llama::parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("estia-llama: {e:#}\n\n{}", estia_llama::USAGE);
            return ExitCode::from(2);
        }
    };
    match estia_llama::run_stdio(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("estia-llama: {e:#}");
            ExitCode::FAILURE
        }
    }
}
