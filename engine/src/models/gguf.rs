//! A small GGUF metadata reader: the header and the key/value pairs, never
//! the tensors.
//!
//! Enough to describe a file someone wants to import: its architecture, the
//! context length and embedding width it declares, its name, its pooling,
//! and whether it carries a chat template. Arrays (the tokenizer's vocabulary
//! is one) are skipped, not loaded.
//!
//! Layout (GGUF v2 and v3, little-endian; v1 used 32-bit counts):
//!
//! ```text
//! "GGUF" | version u32 | tensor_count u64 | kv_count u64 | kv_count × (key: string, type: u32, value)
//! string = len u64 + UTF-8 bytes; array = elem type u32 + count u64 + elements
//! ```

use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Longest string value kept. Chat templates are about 20 KB; anything much
/// larger is not metadata we use, and a corrupt length must not allocate
/// gigabytes.
const MAX_KEPT_STRING: u64 = 1024 * 1024;
/// Longest string the reader will skip over. Beyond this the file is not a
/// GGUF we can trust.
const MAX_STRING: u64 = 256 * 1024 * 1024;
const MAX_KV: u64 = 1 << 20;
const MAX_NESTING: u32 = 4;

/// One metadata value. Arrays are summarised, long strings kept by length only.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// A string longer than the reader keeps.
    LongStr(u64),
    Array {
        elem_type: u32,
        len: u64,
    },
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U64(v) => Some(v),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// The metadata of one GGUF file.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv: BTreeMap<String, GgufValue>,
}

impl GgufMetadata {
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.get(key)
    }

    /// `general.architecture` (`gemma3`, `bert`, `gemma-embedding`, …).
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// `general.name`, when the converter set one.
    pub fn name(&self) -> Option<&str> {
        self.get("general.name").and_then(GgufValue::as_str)
    }

    fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}")).and_then(GgufValue::as_u64)
    }

    /// `<arch>.context_length`: the context the model was trained for.
    pub fn context_length(&self) -> Option<u64> {
        self.arch_u64("context_length")
    }

    /// `<arch>.embedding_length`: the hidden width, which is the vector width
    /// of an embedding model without a projection head.
    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_u64("embedding_length")
    }

    /// `<arch>.pooling_type` (llama.cpp's enum), present on embedding models.
    pub fn pooling_type(&self) -> Option<u64> {
        self.arch_u64("pooling_type")
    }

    /// Whether `tokenizer.chat_template` is present.
    pub fn has_chat_template(&self) -> bool {
        self.get("tokenizer.chat_template").is_some()
    }

    /// The chat template, when it is short enough to have been kept.
    pub fn chat_template(&self) -> Option<&str> {
        self.get("tokenizer.chat_template").and_then(GgufValue::as_str)
    }

    /// A best guess at whether this is an embedding model: it declares a
    /// pooling type, or its architecture is an encoder llama.cpp embeds with,
    /// and it has no chat template.
    pub fn looks_like_embedding(&self) -> bool {
        const ENCODERS: &[&str] = &[
            "bert",
            "nomic-bert",
            "nomic-bert-moe",
            "modern-bert",
            "jina-bert-v2",
            "jina-bert-v3",
            "gemma-embedding",
            "t5encoder",
            "neo-bert",
        ];
        if self.has_chat_template() {
            return false;
        }
        self.pooling_type().is_some_and(|p| p > 0) || self.architecture().is_some_and(|a| ENCODERS.contains(&a))
    }
}

/// Read the metadata of the GGUF file at `path`.
pub fn read_metadata(path: &Path) -> Result<GgufMetadata> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    read_from(BufReader::with_capacity(256 * 1024, file)).with_context(|| format!("read GGUF metadata from {}", path.display()))
}

/// Read metadata from any seekable reader positioned at the start of a GGUF.
pub fn read_from<R: Read + Seek>(mut r: R) -> Result<GgufMetadata> {
    let mut magic = [0_u8; 4];
    r.read_exact(&mut magic).context("file is shorter than a GGUF header")?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file (magic {:02x?})", magic);
    }
    let version = read_u32(&mut r)?;
    if version == 0 || version > 0xffff {
        // A big-endian GGUF reads as a byte-swapped version here.
        bail!("unsupported GGUF version {version} (big-endian files are not supported)");
    }
    if version > 3 {
        bail!("unsupported GGUF version {version}");
    }
    let (tensor_count, kv_count) =
        if version == 1 { (read_u32(&mut r)? as u64, read_u32(&mut r)? as u64) } else { (read_u64(&mut r)?, read_u64(&mut r)?) };
    if kv_count > MAX_KV {
        bail!("implausible metadata count {kv_count}");
    }
    let mut kv = BTreeMap::new();
    for i in 0..kv_count {
        let key = read_string(&mut r, version, MAX_KEPT_STRING)?.map_err(|_| anyhow!("metadata key {i} is too long"))?;
        let ty = read_u32(&mut r)?;
        let value = read_value(&mut r, version, ty, 0).with_context(|| format!("value of `{key}`"))?;
        kv.insert(key, value);
    }
    // Skipping seeks, and a seek past the end succeeds: check the metadata
    // actually fit in the file.
    let pos = r.stream_position()?;
    if pos > r.seek(SeekFrom::End(0))? {
        bail!("file ends inside its metadata");
    }
    Ok(GgufMetadata { version, tensor_count, kv })
}

fn read_value<R: Read + Seek>(r: &mut R, version: u32, ty: u32, depth: u32) -> Result<GgufValue> {
    Ok(match ty {
        0 => GgufValue::U64(read_n::<1, _>(r)?[0] as u64),
        1 => GgufValue::I64(read_n::<1, _>(r)?[0] as i8 as i64),
        2 => GgufValue::U64(u16::from_le_bytes(read_n(r)?) as u64),
        3 => GgufValue::I64(i16::from_le_bytes(read_n(r)?) as i64),
        4 => GgufValue::U64(read_u32(r)? as u64),
        5 => GgufValue::I64(i32::from_le_bytes(read_n(r)?) as i64),
        6 => GgufValue::F64(f32::from_le_bytes(read_n(r)?) as f64),
        7 => GgufValue::Bool(read_n::<1, _>(r)?[0] != 0),
        8 => match read_string(r, version, MAX_KEPT_STRING)? {
            Ok(s) => GgufValue::Str(s),
            Err(len) => GgufValue::LongStr(len),
        },
        9 => {
            if depth >= MAX_NESTING {
                bail!("arrays nested too deeply");
            }
            let elem_type = read_u32(r)?;
            let len = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };
            skip_array(r, version, elem_type, len, depth + 1)?;
            GgufValue::Array { elem_type, len }
        }
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(i64::from_le_bytes(read_n(r)?)),
        12 => GgufValue::F64(f64::from_le_bytes(read_n(r)?)),
        other => bail!("unknown GGUF value type {other}"),
    })
}

/// Bytes per element of a fixed-size type; `None` for strings and arrays.
fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn skip_array<R: Read + Seek>(r: &mut R, version: u32, elem_type: u32, len: u64, depth: u32) -> Result<()> {
    if let Some(size) = fixed_size(elem_type) {
        let bytes = size.checked_mul(len).ok_or_else(|| anyhow!("array length overflows"))?;
        i64::try_from(bytes).map_err(|_| anyhow!("array too large"))?;
        return skip(r, bytes);
    }
    for _ in 0..len {
        match elem_type {
            8 => {
                let n = read_len(r, version)?;
                if n > MAX_STRING {
                    bail!("implausible string length {n}");
                }
                skip(r, n)?;
            }
            9 => {
                let inner = read_u32(r)?;
                let inner_len = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };
                if depth >= MAX_NESTING {
                    bail!("arrays nested too deeply");
                }
                skip_array(r, version, inner, inner_len, depth + 1)?;
            }
            other => bail!("unknown GGUF array element type {other}"),
        }
    }
    Ok(())
}

/// Skip `n` bytes. Short skips are read and dropped: `BufReader::seek`
/// throws its buffer away, and a vocabulary is hundreds of thousands of short
/// strings, so seeking over each one refills the buffer every time (seconds
/// for a Gemma vocabulary instead of milliseconds).
fn skip<R: Read + Seek>(r: &mut R, n: u64) -> Result<()> {
    const READ_UNDER: u64 = 64 * 1024;
    if n <= READ_UNDER {
        let copied = std::io::copy(&mut r.by_ref().take(n), &mut std::io::sink())?;
        if copied != n {
            bail!("unexpected end of GGUF metadata");
        }
    } else {
        r.seek(SeekFrom::Current(n as i64))?;
    }
    Ok(())
}

fn read_n<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut b = [0_u8; N];
    r.read_exact(&mut b).context("unexpected end of GGUF metadata")?;
    Ok(b)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_n(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_n(r)?))
}

fn read_len<R: Read>(r: &mut R, version: u32) -> Result<u64> {
    if version == 1 {
        Ok(read_u32(r)? as u64)
    } else {
        read_u64(r)
    }
}

/// A string, or `Err(length)` when it is longer than `keep` (it is skipped).
fn read_string<R: Read + Seek>(r: &mut R, version: u32, keep: u64) -> Result<std::result::Result<String, u64>> {
    let n = read_len(r, version)?;
    if n > MAX_STRING {
        bail!("implausible string length {n}");
    }
    if n > keep {
        skip(r, n)?;
        return Ok(Err(n));
    }
    let mut buf = vec![0_u8; n as usize];
    r.read_exact(&mut buf).context("unexpected end of GGUF metadata")?;
    Ok(Ok(String::from_utf8_lossy(&buf).into_owned()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF v3 header with the given metadata and no tensors: what the
    /// import tests use as a stand-in model file.
    pub(crate) fn synthetic_gguf(arch: &str, ctx: u32, width: u32, chat_template: Option<&str>, pooling: Option<u32>) -> Vec<u8> {
        let mut kv: Vec<(String, Vec<u8>)> = Vec::new();
        let string = |s: &str| {
            let mut v = 8_u32.to_le_bytes().to_vec();
            put_str(&mut v, s);
            v
        };
        let u32v = |n: u32| {
            let mut v = 4_u32.to_le_bytes().to_vec();
            v.extend(n.to_le_bytes());
            v
        };
        kv.push(("general.architecture".into(), string(arch)));
        kv.push(("general.name".into(), string("Synthetic Test Model")));
        kv.push((format!("{arch}.context_length"), u32v(ctx)));
        kv.push((format!("{arch}.embedding_length"), u32v(width)));
        if let Some(p) = pooling {
            kv.push((format!("{arch}.pooling_type"), u32v(p)));
        }
        // An array of strings, like a vocabulary, and one of floats.
        let mut tokens = 9_u32.to_le_bytes().to_vec();
        tokens.extend(8_u32.to_le_bytes());
        tokens.extend(3_u64.to_le_bytes());
        for t in ["<bos>", "hello", "world"] {
            put_str(&mut tokens, t);
        }
        kv.push(("tokenizer.ggml.tokens".into(), tokens));
        let mut scores = 9_u32.to_le_bytes().to_vec();
        scores.extend(6_u32.to_le_bytes());
        scores.extend(3_u64.to_le_bytes());
        for f in [0.0_f32, -1.0, -2.0] {
            scores.extend(f.to_le_bytes());
        }
        kv.push(("tokenizer.ggml.scores".into(), scores));
        if let Some(t) = chat_template {
            kv.push(("tokenizer.chat_template".into(), string(t)));
        }
        let mut out = b"GGUF".to_vec();
        out.extend(3_u32.to_le_bytes());
        out.extend(0_u64.to_le_bytes());
        out.extend((kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            put_str(&mut out, &k);
            out.extend(v);
        }
        out
    }

    #[test]
    fn reads_a_synthetic_header() {
        let bytes = synthetic_gguf("gemma3", 8192, 640, Some("{{ messages }}"), None);
        let m = read_from(Cursor::new(bytes)).unwrap();
        assert_eq!(m.version, 3);
        assert_eq!(m.architecture(), Some("gemma3"));
        assert_eq!(m.name(), Some("Synthetic Test Model"));
        assert_eq!(m.context_length(), Some(8192));
        assert_eq!(m.embedding_length(), Some(640));
        assert!(m.has_chat_template());
        assert_eq!(m.chat_template(), Some("{{ messages }}"));
        assert!(!m.looks_like_embedding());
        assert_eq!(m.get("tokenizer.ggml.tokens"), Some(&GgufValue::Array { elem_type: 8, len: 3 }));
        assert_eq!(m.get("tokenizer.ggml.scores"), Some(&GgufValue::Array { elem_type: 6, len: 3 }));

        let e = read_from(Cursor::new(synthetic_gguf("bert", 512, 384, None, Some(1)))).unwrap();
        assert!(e.looks_like_embedding());
        assert_eq!(e.pooling_type(), Some(1));
    }

    #[test]
    fn refuses_what_is_not_a_gguf() {
        assert!(read_from(Cursor::new(b"GGML\x03\0\0\0".to_vec())).is_err());
        assert!(read_from(Cursor::new(b"GG".to_vec())).is_err());
        // Truncated in the middle of the metadata.
        let mut bytes = synthetic_gguf("gemma3", 8192, 640, None, None);
        bytes.truncate(bytes.len() - 5);
        assert!(read_from(Cursor::new(bytes)).is_err());
        // A string length no file could hold.
        let mut bad = b"GGUF".to_vec();
        bad.extend(3_u32.to_le_bytes());
        bad.extend(0_u64.to_le_bytes());
        bad.extend(1_u64.to_le_bytes());
        bad.extend(u64::MAX.to_le_bytes());
        assert!(read_from(Cursor::new(bad)).is_err());
    }

    /// The two local test models the llama CI job uses. Gated on the same
    /// environment variables as the adapter's tests; skipped when unset.
    #[test]
    fn reads_the_local_test_models() {
        if let Ok(p) = std::env::var("ESTIA_LLAMA_TEST_MODEL") {
            let m = read_metadata(Path::new(&p)).unwrap();
            assert_eq!(m.architecture(), Some("gemma3"), "{p}");
            assert!(m.has_chat_template(), "tinygemma3 has a chat template");
            assert!(m.context_length().unwrap() > 0);
            assert!(m.embedding_length().unwrap() > 0);
            assert!(!m.looks_like_embedding());
            eprintln!("{p}: name {:?} ctx {:?} width {:?}", m.name(), m.context_length(), m.embedding_length());
        }
        if let Ok(p) = std::env::var("ESTIA_LLAMA_TEST_EMBED_MODEL") {
            let m = read_metadata(Path::new(&p)).unwrap();
            assert_eq!(m.architecture(), Some("bert"), "{p}");
            assert_eq!(m.embedding_length(), Some(384));
            assert!(!m.has_chat_template());
            assert!(m.looks_like_embedding());
            eprintln!("{p}: name {:?} ctx {:?} pooling {:?}", m.name(), m.context_length(), m.pooling_type());
        }
    }
}
