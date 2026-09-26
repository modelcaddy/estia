//! The llama.cpp build Estia installs, pinned per release of Estia.
//!
//! Upstream publishes about a hundred builds a week; Estia pins one named
//! release (`b11146` is upstream's v0.5.0, 2026-09-23) and checks every
//! archive against the SHA-256 GitHub reports for it. Bumping the pin is a
//! reviewed change that must pass the llama CI job.
//!
//! Source: `gh api repos/ggml-org/llama.cpp/releases/tags/b11146`.

/// Upstream build tag.
pub const LLAMA_BUILD: &str = "b11146";

/// Where release archives are downloaded from.
pub const LLAMA_RELEASE_BASE: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b11146";

/// One prebuilt archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaAsset {
    /// `macos`, `linux` or `windows`.
    pub os: &'static str,
    /// `aarch64` or `x86_64` (Rust's `std::env::consts::ARCH` spelling).
    pub arch: &'static str,
    /// `metal` (macOS arm64 builds include CPU), `cpu`, `vulkan`, `cuda-12`,
    /// `cuda-13`, `rocm`.
    pub accel: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// Companion archive with the CUDA runtime libraries, when the variant
    /// needs one and the machine may not have them.
    pub cudart: Option<(&'static str, u64, &'static str)>,
}

pub const LLAMA_ASSETS: &[LlamaAsset] = &[
    LlamaAsset {
        os: "macos",
        arch: "aarch64",
        accel: "metal",
        file: "llama-b11146-bin-macos-arm64.tar.gz",
        bytes: 11_189_714,
        sha256: "1ad3f9eff80edb9dbef4259ad564d1720612ef7eea48fa4afed0e54f5f3d5711",
        cudart: None,
    },
    LlamaAsset {
        os: "macos",
        arch: "x86_64",
        accel: "cpu",
        file: "llama-b11146-bin-macos-x64.tar.gz",
        bytes: 11_237_237,
        sha256: "305f0e3a17d2c01eb205cd0a62128357f1ec3b55329cb084d94e5ec0115d7a3b",
        cudart: None,
    },
    LlamaAsset {
        os: "linux",
        arch: "x86_64",
        accel: "cpu",
        file: "llama-b11146-bin-ubuntu-x64.tar.gz",
        bytes: 16_998_357,
        sha256: "c150306eb16b5ab696f76a8bdf810c35fd98a24e82158742e6fa28f420ff8410",
        cudart: None,
    },
    LlamaAsset {
        os: "linux",
        arch: "aarch64",
        accel: "cpu",
        file: "llama-b11146-bin-ubuntu-arm64.tar.gz",
        bytes: 13_598_346,
        sha256: "4aeda6fe68831547e49b7fa87607383ca5352b3d72ca5f70d52ed265f58c131f",
        cudart: None,
    },
    LlamaAsset {
        os: "linux",
        arch: "x86_64",
        accel: "vulkan",
        file: "llama-b11146-bin-ubuntu-vulkan-x64.tar.gz",
        bytes: 30_598_492,
        sha256: "d3ce40fce7403cc93bcf5718fc46c6efb61ed9709f8e5d9f10c86bf0e30e8fb3",
        cudart: None,
    },
    LlamaAsset {
        os: "linux",
        arch: "aarch64",
        accel: "vulkan",
        file: "llama-b11146-bin-ubuntu-vulkan-arm64.tar.gz",
        bytes: 24_410_274,
        sha256: "5dcebe3ecbcb43a1ed85e3284453f9edf54dcca833e1cb1f54b4022b753c1da5",
        cudart: None,
    },
    LlamaAsset {
        os: "linux",
        arch: "x86_64",
        accel: "cuda-12",
        file: "llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz",
        bytes: 168_920_581,
        sha256: "c2ab9e19838513ff69d1af8d999ad717dd3c7ee4714ac04c7ed5ab9077c50e4e",
        cudart: Some((
            "cudart-llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz",
            594_373_356,
            "1466daea60aad1144819e151b2bae19d54556cf1da6c129c4f55a5ded2637c25",
        )),
    },
    LlamaAsset {
        os: "linux",
        arch: "x86_64",
        accel: "cuda-13",
        file: "llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz",
        bytes: 149_265_156,
        sha256: "1603d9c00a4b6eac8298c5c7868cdb080a3ac31948ab1e457441d71ce274dd7e",
        cudart: Some((
            "cudart-llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz",
            440_231_388,
            "7c2af505f8b26ecd3707ab7723fa985fee1df233b7c1d60e5e17724b536d15bb",
        )),
    },
    LlamaAsset {
        os: "linux",
        arch: "x86_64",
        accel: "rocm",
        file: "llama-b11146-bin-ubuntu-rocm-10.0-x64.tar.gz",
        bytes: 234_721_151,
        sha256: "50e79dc559a11af3ea59391d416e9a704a715ac6be94352dbba710782c5dd7d1",
        cudart: None,
    },
    LlamaAsset {
        os: "windows",
        arch: "x86_64",
        accel: "cpu",
        file: "llama-b11146-bin-win-cpu-x64.zip",
        bytes: 18_560_055,
        sha256: "14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1",
        cudart: None,
    },
    LlamaAsset {
        os: "windows",
        arch: "aarch64",
        accel: "cpu",
        file: "llama-b11146-bin-win-cpu-arm64.zip",
        bytes: 12_034_624,
        sha256: "1727d241f3bf6d27360e984e851cf013928fd655bf89f8628e70da027f377b7d",
        cudart: None,
    },
    LlamaAsset {
        os: "windows",
        arch: "x86_64",
        accel: "vulkan",
        file: "llama-b11146-bin-win-vulkan-x64.zip",
        bytes: 32_127_004,
        sha256: "55a378aa095b466979d85075234f66d7655c7a7483222af0c006c0e55b4d7bd6",
        cudart: None,
    },
    LlamaAsset {
        os: "windows",
        arch: "x86_64",
        accel: "cuda-12",
        file: "llama-b11146-bin-win-cuda-12.4-x64.zip",
        bytes: 253_869_799,
        sha256: "3c806a6ceccc3dae1c743ceb1a1fb2cce5b76f40bfbd4c6b7b8afb6ef45a5807",
        cudart: Some((
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            391_443_627,
            "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        )),
    },
    LlamaAsset {
        os: "windows",
        arch: "x86_64",
        accel: "cuda-13",
        file: "llama-b11146-bin-win-cuda-13.4-x64.zip",
        bytes: 149_758_833,
        sha256: "b1866c0ce76bc7bfb0c24b33e9a37e9669f1be18539b12c74ce361f81c41f047",
        cudart: Some((
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
            423_535_356,
            "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        )),
    },
];

/// The assets for this OS and CPU. Order carries no preference: the
/// installer's accelerator probe decides.
pub fn assets_for<'a>(os: &'a str, arch: &'a str) -> impl Iterator<Item = &'static LlamaAsset> + 'a {
    LLAMA_ASSETS.iter().filter(move |a| a.os == os && a.arch == arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_are_well_formed() {
        for a in LLAMA_ASSETS {
            assert!(a.file.contains(LLAMA_BUILD) || a.file.starts_with("cudart-"), "{}", a.file);
            assert_eq!(a.sha256.len(), 64, "{}", a.file);
            assert!(a.sha256.bytes().all(|b| b.is_ascii_hexdigit()), "{}", a.file);
            if let Some((f, _, h)) = a.cudart {
                assert!(f.starts_with("cudart-") && h.len() == 64);
            }
        }
        assert_eq!(assets_for("macos", "aarch64").count(), 1);
        assert!(assets_for("linux", "x86_64").any(|a| a.accel == "cpu"));
    }
}
