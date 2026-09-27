//! Structured output: the caller gets valid JSON — schema-conforming when a
//! schema was given — or a typed error. Never a string to clean.
//!
//! Small local models produce two recurring JSON defects: invalid backslash
//! escapes (code quoted inside strings: Swift `\(url)`, regex `\d`, Windows
//! `C:\Users`) and unescaped interior quotes (`"port "18018""`). Either one
//! fails a strict parse of the whole output. The repair ladder here runs
//! cheapest step first and only ever turns unparseable text into parseable
//! text: raw → strip fences and preamble → repair escapes → repair quotes.
//! Past the raw parse, each step takes the first complete object or array
//! and drops any text after it (a sign-off, a stray `}`). Every step is
//! recorded so a caller can see that the model did not produce the output
//! clean.
//!
//! Where a backend can constrain decoding to a schema (`llama` via a grammar,
//! `mlx` later via a logits processor) the runner does that and this module
//! only validates. Where it cannot, the model is shown the schema in its
//! prompt ([`with_prompt_hint`], [`prompt_with_hint`]) and this module is the
//! enforcement: parse, validate, and hand the caller enough to retry once
//! with the error appended to the prompt ([`Structured::retry_hint`]).

use crate::proto::Message;
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
    /// The model did not produce this clean: fences, preamble or trailing
    /// text were stripped, or escapes or quotes had to be repaired.
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

/// What a model is told when its runner cannot constrain decoding: the shape
/// wanted, and the schema when there is one. `None` for plain text. The
/// output is still parsed and validated afterwards.
pub fn prompt_hint(format: &OutputFormat) -> Option<String> {
    match format {
        OutputFormat::Text => None,
        OutputFormat::Json => Some("Answer with one JSON object and nothing else: no prose, no code fences.".to_string()),
        OutputFormat::JsonSchema { schema } => Some(format!(
            "Answer with one JSON value that matches this JSON Schema, and nothing else: no prose, no code fences.\nJSON Schema: {schema}"
        )),
    }
}

/// `messages` with [`prompt_hint`] in the system prompt: appended to a
/// leading system message, or added in front as one. Unchanged for plain
/// text.
pub fn with_prompt_hint(messages: &[Message], format: &OutputFormat) -> Vec<Message> {
    let mut out = messages.to_vec();
    let Some(hint) = prompt_hint(format) else {
        return out;
    };
    match out.first_mut() {
        Some(first) if first.role == "system" => {
            if !first.content.trim().is_empty() {
                first.content.push_str("\n\n");
            }
            first.content.push_str(&hint);
        }
        _ => out.insert(0, Message::new("system", hint)),
    }
    out
}

/// A raw prompt with [`prompt_hint`] after it. Unchanged for plain text.
pub fn prompt_with_hint(prompt: &str, format: &OutputFormat) -> String {
    match prompt_hint(format) {
        Some(hint) => format!("{prompt}\n\n{hint}"),
        None => prompt.to_string(),
    }
}

/// What [`unwrap_json`] took off around the JSON.
struct Unwrapped<'a> {
    /// From the first `{` or `[` on (or the whole text when there is none).
    text: &'a str,
    /// Code fences, or text before the JSON, were removed.
    front: bool,
    /// Non-blank text after a closing code fence was dropped.
    after_fence: bool,
}

fn first_bracket(s: &str) -> Option<usize> {
    s.find(['{', '['])
}

/// A fenced block's body when a code fence opens before the JSON does, then
/// advance to the first `{` or `[`. A fence counts only when the rest of its
/// line is a bare language tag (`json`, or nothing): anything else on that
/// line means the backticks are not an opener, and the text is read from its
/// first bracket instead.
fn unwrap_json(raw: &str) -> Unwrapped<'_> {
    let s = raw.trim();
    let mut fenced = false;
    let mut after_fence = false;
    let mut body = s;
    if let Some(f) = s.find("```") {
        let opener = &s[f + 3..];
        let (tag, inner) = opener.split_once('\n').unwrap_or((opener, ""));
        let is_tag = tag.trim().chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '+' | '.'));
        if is_tag && first_bracket(s).is_none_or(|b| f < b) {
            fenced = true;
            // Valid JSON holds no raw newline inside a string, so a line that
            // starts with a fence closes the block.
            let close = if inner.starts_with("```") { Some(0) } else { inner.find("\n```").map(|i| i + 1) };
            body = match close {
                Some(c) => {
                    after_fence = !inner[c + 3..].trim().is_empty();
                    &inner[..c]
                }
                None => inner,
            };
            body = body.trim();
        }
    }
    let start = first_bracket(body).unwrap_or(0);
    Unwrapped { text: &body[start..], front: fenced || start > 0, after_fence }
}

/// Strip code fences and advance to the first `{` or `[` so models that add
/// preamble or markdown wrappers don't break JSON parsing. A fence is honoured
/// when it opens before the first bracket, after a preamble or not; text after
/// its closing fence is dropped. Text after the JSON itself is left in place (see
/// [`first_json_value`]).
pub fn clean_json(raw: &str) -> String {
    unwrap_json(raw).text.to_string()
}

/// The first complete JSON value in `text` (leading whitespace allowed), and
/// whether non-blank text follows it. Only an object or an array may be
/// followed by text: both end unambiguously, so a model that closed its
/// object and then added a sign-off (`Hope this helps!`) or a stray `}` still
/// answered. A scalar must be the whole text. The error is the strict parse's.
pub fn first_json_value(text: &str) -> Result<(Value, bool), serde_json::Error> {
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    if let Some(Ok(value)) = values.next() {
        let trailing = !text[values.byte_offset()..].trim().is_empty();
        if !trailing || value.is_object() || value.is_array() {
            return Ok((value, trailing));
        }
    }
    serde_json::from_str::<Value>(text).map(|value| (value, false))
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

/// Parse model output as JSON through the repair ladder. Each step after the
/// strict parse takes the first complete object or array and drops what
/// follows it (`strip_trailing_text`); a value cut off before it closes is
/// still an error.
pub fn parse_lenient(text: &str) -> Result<Structured, StructuredError> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Ok(Structured { value, repaired: false, repairs: Vec::new() });
    }
    let unwrapped = unwrap_json(text);
    let done = |(value, trailing): (Value, bool), fixes: &[&'static str]| {
        let mut repairs = Vec::new();
        if unwrapped.front {
            repairs.push("strip_fences_or_preamble");
        }
        if unwrapped.after_fence || trailing {
            repairs.push("strip_trailing_text");
        }
        repairs.extend_from_slice(fixes);
        Structured { value, repaired: !repairs.is_empty(), repairs }
    };
    if let Ok(parsed) = first_json_value(unwrapped.text) {
        return Ok(done(parsed, &[]));
    }
    let escaped = repair_json_escapes(unwrapped.text);
    if let Ok(parsed) = first_json_value(&escaped) {
        return Ok(done(parsed, &["repair_escapes"]));
    }
    let quoted = repair_json_quotes(&escaped);
    match first_json_value(&quoted) {
        Ok(parsed) => Ok(done(parsed, &["repair_escapes", "repair_quotes"])),
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
    fn prompt_hint_shows_the_schema_in_the_system_prompt() {
        let schema = json!({"type": "object", "required": ["n"]});
        let format = OutputFormat::JsonSchema { schema: schema.clone() };
        let hint = prompt_hint(&format).unwrap();
        assert!(hint.contains(&schema.to_string()), "{hint}");
        assert!(prompt_hint(&OutputFormat::Json).unwrap().contains("JSON object"));
        assert_eq!(prompt_hint(&OutputFormat::Text), None);

        // No system message: one is added in front.
        let user = vec![Message::new("user", "hi")];
        let out = with_prompt_hint(&user, &format);
        assert_eq!((out.len(), out[0].role.as_str(), out[0].content.as_str()), (2, "system", hint.as_str()));
        assert_eq!(out[1], user[0]);
        // A leading system message keeps its text and gets the hint after it.
        let with_system = vec![Message::new("system", "Be terse."), Message::new("user", "hi")];
        let out = with_prompt_hint(&with_system, &format);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, format!("Be terse.\n\n{hint}"));
        // Plain text: untouched.
        assert_eq!(with_prompt_hint(&with_system, &OutputFormat::Text), with_system);
        assert_eq!(prompt_with_hint("p", &OutputFormat::Text), "p");
        assert_eq!(prompt_with_hint("p", &format), format!("p\n\n{hint}"));
    }

    #[test]
    fn clean_json_strips_fences_and_preamble() {
        assert_eq!(clean_json("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(clean_json("Sure! Here you go: {\"a\":1}"), "{\"a\":1}");
        assert_eq!(clean_json("list: [1,2]"), "[1,2]");
        assert_eq!(clean_json("no json here"), "no json here");
        // A fence after a preamble is honoured, and prose after it dropped.
        assert_eq!(clean_json("Here:\n```json\n{\"a\":1}\n```\nHope this helps!"), "{\"a\":1}");
        assert_eq!(clean_json("Here you go ```\n[2]\n```"), "[2]");
        // A fence that opens after the JSON (inside a string) is content.
        assert_eq!(clean_json("x {\"md\": \"```js\"}"), "{\"md\": \"```js\"}");
        // An unclosed fence: the rest is the block.
        assert_eq!(clean_json("```json\n{\"a\":1}"), "{\"a\":1}");
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
    fn text_after_a_complete_object_is_dropped_and_recorded() {
        // api-F1: a stray closing brace after a complete object.
        let s = parse_lenient(r#"{"a":1}}"#).unwrap();
        assert_eq!((s.value.clone(), s.repaired, s.repairs.clone()), (json!({"a": 1}), true, vec!["strip_trailing_text"]));
        // A sign-off after the object.
        let s = parse_lenient("{\"a\":1}\nHope this helps!").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!({"a": 1}), vec!["strip_trailing_text"]));
        // Fenced JSON followed by prose, with and without a preamble.
        let s = parse_lenient("```json\n{\"a\":1}\n```\nHope this helps!").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!({"a": 1}), vec!["strip_fences_or_preamble", "strip_trailing_text"]));
        let s = parse_lenient("Here it is:\n```json\n{\"a\":1}\n```\nLet me know if you need more.").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!({"a": 1}), vec!["strip_fences_or_preamble", "strip_trailing_text"]));
        // Backticks that do not open a block (a one-line fence, or prose
        // after them) leave the text to be read from its first bracket.
        let s = parse_lenient("```{\"a\":1}```").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!({"a": 1}), vec!["strip_fences_or_preamble", "strip_trailing_text"]));
        let s = parse_lenient("Use ```json blocks? Anyway: {\"a\":1}").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!({"a": 1}), vec!["strip_fences_or_preamble"]));
        // Arrays too; and the repairs still combine with the other steps.
        let s = parse_lenient("[1, 2] and that is all").unwrap();
        assert_eq!((s.value.clone(), s.repairs.clone()), (json!([1, 2]), vec!["strip_trailing_text"]));
        let s = parse_lenient("Sure: {\"a\": \"C:\\Users\"} done").unwrap();
        assert_eq!(s.value, json!({"a": "C:\\Users"}));
        assert_eq!(s.repairs, vec!["strip_fences_or_preamble", "strip_trailing_text", "repair_escapes"]);
        // A second object after the first: the first one wins (the
        // acceptance run's `ops-f1-echo-two` output).
        assert_eq!(parse_lenient("{\"a\":1} {\"b\":2}").unwrap().value, json!({"a": 1}));
        assert_eq!(parse_lenient("{\"a\":1}\n{\"b\":2}").unwrap().value, json!({"a": 1}));
        // Trailing whitespace alone is not a repair.
        assert!(!parse_lenient("{\"a\":1}\n\n  ").unwrap().repaired);
    }

    #[test]
    fn a_truncated_or_scalar_answer_still_fails() {
        // Cut off before the object closes: no complete value to take.
        for raw in [r#"{"a": 1, "b": "cut"#, "{\"a\":1", "{\"a\": {\"b\": 2}", "```json\n{\"a\":1\n```\nsorry", "Sure: [1, 2"] {
            let err = parse_lenient(raw).unwrap_err();
            assert!(matches!(err, StructuredError::NotJson { .. }), "{raw:?}: {err:?}");
        }
        // A scalar must be the whole answer: "42 is the answer" is prose.
        assert!(matches!(parse_lenient("42 is the answer"), Err(StructuredError::NotJson { .. })));
        assert!(matches!(parse_lenient("true, I think"), Err(StructuredError::NotJson { .. })));
        assert_eq!(parse_lenient(" 42 ").unwrap().value, json!(42));
    }

    #[test]
    fn trailing_text_is_dropped_before_schema_validation() {
        let schema = json!({
            "type": "object", "additionalProperties": false,
            "properties": {"title": {"type": "string", "maxLength": 8}}, "required": ["title"]
        });
        let format = OutputFormat::JsonSchema { schema };
        let ok = enforce("{\"title\": \"Standup\"}\nHope this helps!", &format).unwrap();
        assert_eq!((ok.value["title"].as_str(), ok.repairs), (Some("Standup"), vec!["strip_trailing_text"]));
        // Validation still runs on what was kept.
        let err = enforce("{\"title\": \"Quarterly budget review\"}}", &format).unwrap_err();
        assert!(matches!(err, StructuredError::SchemaMismatch { .. }), "{err:?}");
        let err = enforce("{\"title\": \"Standup\", \"extra\": 1} thanks", &format).unwrap_err();
        assert!(matches!(err, StructuredError::SchemaMismatch { .. }), "{err:?}");
    }

    #[test]
    fn first_json_value_reports_what_follows() {
        assert_eq!(first_json_value(" {\"a\":1} ").unwrap(), (json!({"a": 1}), false));
        assert_eq!(first_json_value("{\"a\":1}}").unwrap(), (json!({"a": 1}), true));
        assert_eq!(first_json_value("\"s\"").unwrap(), (json!("s"), false));
        assert!(first_json_value("\"s\" and more").is_err());
        assert!(first_json_value("").is_err());
        assert!(first_json_value("{\"a\":").is_err());
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
