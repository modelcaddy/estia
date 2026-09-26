//! Backend runtimes the engine installs for itself.
//!
//! - [`python`]: the interpreter + packages the Python MLX runner needs
//!   (`python-mlx` feature).
//! - [`llama`]: upstream llama.cpp's prebuilt `llama-server` for the
//!   `llama-cpp` backend, pinned in [`llama_pins`] (`llama-runtime` feature).
//!
//! Without the features, status, paths and removal still work; only the
//! code that downloads executables is left out.

pub mod llama;
pub mod llama_pins;
pub mod python;

pub use llama::{inspect_server, AcceleratorFacts, LlamaProbe, LlamaRuntime, LlamaRuntimeStatus, LlamaRuntimeSummary, LlamaServerInfo};
pub use python::{phase, PythonRuntime, RuntimeState, RuntimeStatus, RuntimeSummary, SetupProgress, RUNTIME_APPROX_BYTES};
