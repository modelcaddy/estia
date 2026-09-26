//! Which inference backend runs a model.
//!
//! A backend is a runner process kind plus the artifact format it loads:
//! `mlx-python` runs MLX safetensors through `runners/mlx-python/estia-runner.py`;
//! `llama-cpp` runs GGUF files through upstream `llama-server`, driven by the
//! `estia-llama` adapter. The id is part of every embedding fingerprint
//! (`<model>@<backend>`), because the same weights give different vectors on
//! different backends.

use crate::models::Format;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// MLX through the Python runner. Apple Silicon only.
    MlxPython,
    /// llama.cpp through `llama-server`. macOS, Linux, Windows; CPU and GPUs.
    LlamaCpp,
}

impl Backend {
    pub const ALL: [Backend; 2] = [Backend::MlxPython, Backend::LlamaCpp];

    /// The id used in fingerprints, `x_estia.backend` and config files.
    pub fn id(self) -> &'static str {
        match self {
            Backend::MlxPython => crate::models::embed::BACKEND_MLX_PYTHON,
            Backend::LlamaCpp => BACKEND_LLAMA_CPP,
        }
    }

    /// The artifact format this backend loads.
    pub fn format(self) -> Format {
        match self {
            Backend::MlxPython => Format::Mlx,
            Backend::LlamaCpp => Format::Gguf,
        }
    }

    /// Whether this backend can run on the machine at all (not whether it is
    /// installed). MLX needs Apple Silicon.
    pub fn supported_here(self) -> bool {
        match self {
            Backend::MlxPython => cfg!(all(target_os = "macos", target_arch = "aarch64")),
            Backend::LlamaCpp => true,
        }
    }

    /// The default for this machine: MLX on Apple Silicon, llama.cpp
    /// everywhere else.
    pub fn platform_default() -> Backend {
        if Backend::MlxPython.supported_here() {
            Backend::MlxPython
        } else {
            Backend::LlamaCpp
        }
    }
}

pub const BACKEND_LLAMA_CPP: &str = "llama-cpp";

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mlx-python" | "mlx" => Ok(Backend::MlxPython),
            "llama-cpp" | "llama" | "llama.cpp" | "llamacpp" => Ok(Backend::LlamaCpp),
            other => Err(format!("unknown backend `{other}` (mlx-python | llama-cpp)")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_aliases_parse() {
        for b in Backend::ALL {
            assert_eq!(b.id().parse::<Backend>().unwrap(), b);
        }
        assert_eq!("llama".parse::<Backend>().unwrap(), Backend::LlamaCpp);
        assert_eq!("MLX".parse::<Backend>().unwrap(), Backend::MlxPython);
        assert!("onnx".parse::<Backend>().is_err());
        assert_eq!(Backend::LlamaCpp.format(), Format::Gguf);
        assert_eq!(serde_json::to_string(&Backend::LlamaCpp).unwrap(), "\"llama-cpp\"");
    }
}
