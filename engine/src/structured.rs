//! Structured output: the caller gets valid JSON — schema-conforming when a
//! schema was given — or a typed error. Never a string to clean.
//!
//! Small local models produce two recurring JSON defects: invalid backslash
//! escapes (code quoted inside strings: Swift `\(url)`, regex `\d`, Windows
//! `C:\Users`) and unescaped interior quotes (`"port "18018""`). Either one
//! fails a strict parse of the whole output. The repair ladder here runs
//! cheapest step first and only ever turns unparseable text into parseable
//! text: raw → strip fences and preamble → repair escapes → repair quotes. Every step is recorded so a caller can see that the model did not
//! produce the output clean.
//!
//! Where a backend can constrain decoding to a schema (`llama` via a grammar,
//! `mlx` later via a logits processor) the runner does that and this module
//! only validates. Where it cannot, this module is the enforcement: parse,
//! validate, and hand the caller enough to retry once with the error
//! appended to the prompt ([`Structured::retry_hint`]).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What shape of output the caller wants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Free text. Nothing is parsed.
    Text,
    /// Any valid JSON value.
    Json,
    /// JSON that validates against `schema` (JSON Schema, draft 2020-12 or
    /// earlier drafts the validator accepts).
    JsonSchema { schema: Value },
}

impl OutputFormat {
    pub fn wants_json(&self) -> bool {
        !matches!(self, OutputFormat::Text)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StructuredError {
    /// Not JSON even after every repair. Carries the head of the raw text so
    /// the failure is debuggable from a log.
    #[error("output is not JSON ({reason}); raw head: {raw_head}")]
    NotJson { reason: String, raw_head: String },
    /// Parsed, but does not match the schema. `problems` are the validator's
    /// messages, one per violation.
    #[error("output does not match the schema: {}", problems.join("; "))]
    SchemaMismatch { problems: Vec<String>, value: Value },
    /// The schema itself could not be compiled.
    #[error("invalid schema: {0}")]
    BadSchema(String),
}

/// A successful structured parse.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Structured {
    pub value: Value,
    /// The model did not produce this clean: fences or preamble were
    /// stripped, or escapes or quotes had to be repaired.
    pub repaired: bool,
    /// Which repair steps were needed, in order, for the curious.
    pub repairs: Vec<&'static str>,
}

impl Structured {
    /// Text to append to a prompt for one retry after a
    /// [`StructuredError::SchemaMismatch`].
    pub fn retry_hint(err: &StructuredError) -> String {
        match err {
            StructuredError::SchemaMismatch { problems, .. } => format!(
                "\n\nYour previous answer did not match the required JSON schema: {}. \
                 Reply with only the corrected JSON object.",
                problems.join("; ")
            ),
            StructuredError::NotJson { reason, .. } => format!(
                "\n\nYour previous answer was not valid JSON ({reason}). \
                 Reply with only a JSON object, no prose and no code fences."
            ),
            StructuredError::BadSchema(_) => String::new(),
        }
    }
}

/// Strip code fences and advance to the first `{` or `[` so models that add
/// preamble or markdown wrappers don't break JSON parsing.
pub fn clean_json(raw: &str) -> String {
    let s = raw.trim();
    let s = if s.starts_with("```") {
        s.lines().skip(1).take_while(|l| !l.starts_with("```")).collect::<Vec<_>>().join("\n")
    } else {
        s.to_string()
    };
    let obj = s.find('{');
    let arr = s.find('[');
    match (obj, arr) {
        (Some(o), Some(a)) => s[o.min(a)..].to_string(),
        (Some(o), None) => s[o..].to_string(),
        (None, Some(a)) => s[a..].to_string(),
        (None, None) => s,
    }
}

/// Escape every backslash that does NOT start a valid JSON escape sequence
/// (`\" \\ \/ \b \f \n \r \t \uXXXX`). Code quoted inside the model's JSON
/// strings routinely carries invalid ones — `\(` from Swift interpolation,
/// `\d` from regexes, `\U` from Windows paths — and a single occurrence
/// fails the whole parse ("invalid escape at line 1 column 411").
pub fn repair_json_escapes(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            let next = bytes.get(i + 1).copied();
            let valid = match next {
                Some(b'"') | Some(b'\\') | Some(b'/') | Some(b'b') | Some(b'f') | Some(b'n') | Some(b'r') | Some(b't') => true,
                Some(b'u') => bytes.len() > i + 5 && bytes[i + 2..i + 6].iter().all(|b| b.is_ascii_hexdigit()),
                _ => false,
            };
            if valid {
                out.push('\\');
                if let Some(n) = next {
                    out.push(n as char);
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }
            out.push_str("\\\\");
            i += 1;
            continue;
        }
        // Copy the full UTF-8 char, not just one byte.
        let ch_len = match bytes[i] {
            b if b < 0x80 => 1,
            b if b < 0xE0 => 2,
            b if b < 0xF0 => 3,
            _ => 4,
        };
        out.push_str(&s[i..(i + ch_len).min(s.len())]);
        i += ch_len;
    }
    out
}

/// Escape UNESCAPED interior double-quotes inside JSON string values. A `"` is
/// a legitimate string terminator only when the next non-whitespace character
/// is one of `,` `}` `]` `:` (or end of input); any other `"` while inside a
/// string is content the model forgot to escape — escape it. Leaves
/// already-escaped quotes (`\"`) and structural quotes untouched. Only reached
/// after a real parse failure, so it can only turn unparseable JSON into
/// parseable JSON.
pub fn repair_json_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !in_string {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
            i += 1;
            continue;
        }
        if escaped {
            out.push(c);
            escaped = false;
            i += 1;
            continue;
        }
        if c == '\\' {
            out.push(c);
            escaped = true;
            i += 1;
            continue;
        }
        if c == '"' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let terminator = j >= chars.len() || matches!(chars[j], ',' | '}' | ']' | ':');
            if terminator {
                in_string = false;
                out.push('"');
            } else {
                out.push('\\');
                out.push('"');
            }
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn raw_head(text: &str) -> String {
    text.chars().take(240).collect()
}

/// Parse model output as JSON through the repair ladder.
pub fn parse_lenient(text: &str) -> Result<Structured, StructuredError> {
    let mut repairs = Vec::new();
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Ok(Structured { value, repaired: false, repairs });
    }
    let cleaned = clean_json(text);
    if cleaned != text.trim() {
        repairs.push("strip_fences_or_preamble");
    }
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        return Ok(Structured { value, repaired: !repairs.is_empty(), repairs });
    }
    let escaped = repair_json_escapes(&cleaned);
    if let Ok(value) = serde_json::from_str::<Value>(&escaped) {
        repairs.push("repair_escapes");
        return Ok(Structured { value, repaired: true, repairs });
    }
    let quoted = repair_json_quotes(&escaped);
    match serde_json::from_str::<Value>(&quoted) {
        Ok(value) => {
            repairs.push("repair_escapes");
            repairs.push("repair_quotes");
            Ok(Structured { value, repaired: true, repairs })
        }
        Err(e) => Err(StructuredError::NotJson { reason: e.to_string(), raw_head: raw_head(text) }),
    }
}

/// Validate `value` against `schema`.
pub fn validate(schema: &Value, value: &Value) -> Result<(), StructuredError> {
    let validator = jsonschema::validator_for(schema).map_err(|e| StructuredError::BadSchema(e.to_string()))?;
    let problems: Vec<String> = validator
        .iter_errors(value)
        .map(|e| {
            let path = e.instance_path.to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{path}: {e}")
            }
        })
        .collect();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(StructuredError::SchemaMismatch { problems, value: value.clone() })
    }
}

/// Parse and, when a schema was asked for, validate. `Text` always succeeds
/// with a JSON string value so callers can treat every format alike.
pub fn enforce(text: &str, format: &OutputFormat) -> Result<Structured, StructuredError> {
    match format {
        OutputFormat::Text => Ok(Structured { value: Value::String(text.to_string()), repaired: false, repairs: Vec::new() }),
        OutputFormat::Json => parse_lenient(text),
        OutputFormat::JsonSchema { schema } => {
            let parsed = parse_lenient(text)?;
            validate(schema, &parsed.value)?;
            Ok(parsed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn clean_json_strips_fences_and_preamble() {
        assert_eq!(clean_json("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(clean_json("Sure! Here you go: {\"a\":1}"), "{\"a\":1}");
        assert_eq!(clean_json("list: [1,2]"), "[1,2]");
        assert_eq!(clean_json("no json here"), "no json here");
    }

    #[test]
    fn swift_interpolation_survives() {
        // The exact live failure shape: Swift `\(...)` inside a description.
        let raw = r#"{"title": "Troubleshoot Image Loading", "tasks": [{"title": "Handle errors", "description": "Modify `loadImageFromFileURL` to log \(url.path) on failure"}]}"#;
        let s = parse_lenient(raw).expect("repaired parse");
        assert!(s.repaired);
        assert_eq!(s.repairs, vec!["repair_escapes"]);
        assert!(s.value["tasks"][0]["description"].as_str().unwrap().contains("(url.path)"));
    }

    #[test]
    fn valid_escapes_untouched() {
        let s = r#"{"a": "line\nbreak \"quoted\" é back\\slash"}"#;
        assert_eq!(repair_json_escapes(s), s);
        let parsed = parse_lenient(s).unwrap();
        assert!(!parsed.repaired);
    }

    #[test]
    fn regex_and_windows_paths_repaired() {
        let s = r#"{"a": "match \d+ in C:\Users\geo"}"#;
        let v: Value = serde_json::from_str(&repair_json_escapes(s)).unwrap();
        assert_eq!(v["a"].as_str().unwrap(), r"match \d+ in C:\Users\geo");
    }

    #[test]
    fn unescaped_interior_quotes_repaired() {
        // The live failure ("expected `,` or `}`"): the model wrote an
        // unescaped quote inside a description string.
        let raw = r#"{"title": "Diagnose Connection Refused", "tasks": [{"title": "Check port", "description": "Verify service listening on port "18018" via netstat"}], "context_summary": "Connection refused on port 18018; check the listener."}"#;
        let s = parse_lenient(raw).expect("quote-repaired parse");
        assert_eq!(s.repairs, vec!["repair_escapes", "repair_quotes"]);
        assert!(s.value["tasks"][0]["description"].as_str().unwrap().contains("18018"));
    }

    #[test]
    fn quote_repair_preserves_valid_json() {
        let raw = r#"{"title": "X", "decisions": [{"title": "Use \"foo\" mode", "reasoning": "clearer"}], "context_summary": "A said \"go\" to B."}"#;
        let s = parse_lenient(raw).expect("parse");
        assert!(!s.repaired);
        assert!(s.value["context_summary"].as_str().unwrap().contains("\"go\""));
    }

    #[test]
    fn fenced_json_is_a_repair_and_garbage_is_typed() {
        let s = parse_lenient("```json\n{\"title\": \"T\"}\n```").unwrap();
        assert_eq!(s.repairs, vec!["strip_fences_or_preamble"]);
        let err = parse_lenient("Sure! Here's what I found:").unwrap_err();
        assert!(matches!(err, StructuredError::NotJson { ref raw_head, .. } if raw_head.starts_with("Sure!")));
        let err = parse_lenient(r#"{"title": "Cut off mid-"#).unwrap_err();
        assert!(matches!(err, StructuredError::NotJson { .. }));
    }

    #[test]
    fn schema_validation_reports_each_problem() {
        let schema = json!({
            "type": "object",
            "required": ["title", "tasks"],
            "properties": {
                "title": {"type": "string"},
                "tasks": {"type": "array", "items": {"type": "object", "required": ["title"]}}
            }
        });
        let format = OutputFormat::JsonSchema { schema: schema.clone() };
        let ok = enforce(r#"{"title": "T", "tasks": [{"title": "a"}]}"#, &format).unwrap();
        assert_eq!(ok.value["title"], "T");

        let err = enforce(r#"{"title": 3, "tasks": [{}]}"#, &format).unwrap_err();
        match &err {
            StructuredError::SchemaMismatch { problems, value } => {
                assert_eq!(problems.len(), 2, "{problems:?}");
                assert_eq!(value["title"], 3);
            }
            other => panic!("{other:?}"),
        }
        let hint = Structured::retry_hint(&err);
        assert!(hint.contains("did not match the required JSON schema"));

        let bad = enforce("{}", &OutputFormat::JsonSchema { schema: json!({"type": "nonsense"}) }).unwrap_err();
        assert!(matches!(bad, StructuredError::BadSchema(_)));
    }

    #[test]
    fn text_and_json_formats() {
        let t = enforce("hello", &OutputFormat::Text).unwrap();
        assert_eq!(t.value, Value::String("hello".into()));
        let j = enforce("[1, 2]", &OutputFormat::Json).unwrap();
        assert_eq!(j.value, json!([1, 2]));
        let f: OutputFormat = serde_json::from_str(r#"{"type":"json_schema","schema":{"type":"object"}}"#).unwrap();
        assert!(f.wants_json());
        assert!(!OutputFormat::Text.wants_json());
    }
}
