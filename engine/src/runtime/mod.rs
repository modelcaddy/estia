//! Backend runtimes the engine installs for itself.
//!
//! Today: [`python`], the interpreter + packages the Python MLX runner needs.
//! A compiled backend (the Swift MLX runner, llama.cpp) ships as a binary and
//! needs nothing here.

pub mod llama_pins;
pub mod python;

pub use python::{phase, PythonRuntime, RuntimeState, RuntimeStatus, RuntimeSummary, SetupProgress, RUNTIME_APPROX_BYTES};
