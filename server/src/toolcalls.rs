//! Turn a model's tool-call text into OpenAI `tool_calls`.
//!
//! Gemma 4 answers a tools request in its own syntax:
//! `<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>` — bare keys,
//! `<|"|>` as the string quote, several calls allowed back to back. The manual
//! template fallback asks instead for `{"tool_call": {"name": …, "arguments": {…}}}`.
//! Both are recognised; anything else is plain content.
//!
//! A runner that parses calls itself (the llama.cpp adapter declares
//! `parses_tool_calls`) returns them in `meta.tool_calls`; its text is then
//! plain content and is not parsed again ([`from_output`]).

use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Arguments as a JSON object, serialized (OpenAI sends a string).
    pub arguments: String,
}

impl ToolCall {
    /// This call in OpenAI's shape, with a fresh `call_<random>` id. Build a
    /// response's calls once and reuse them: clients match a `tool` message
    /// to its call by this id, so it must not change within a response and
    /// must not repeat across responses (a conversation holds many).
    pub fn to_openai(&self) -> Value {
        json!({
            "id": call_id(),
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments}
        })
    }
}

/// A new tool-call id: `call_` and 24 random hex digits.
pub fn call_id() -> String {
    crate::unique_id("call_")
}

/// Gemma's `{key:<|"|>value<|"|>,n:3,list:[…]}` → JSON text.
fn gemma_args_to_json(raw: &str) -> String {
    // 1. string quotes
    let mut s = raw.replace("<|\"|>", "\"");
    // 2. quote bare keys: an identifier followed by ':' that is not inside a string
    let mut out = String::with_capacity(s.len() + 16);
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut in_str = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            in_str = !in_str;
            out.push(c);
            i += 1;
            continue;
        }
        if !in_str && (c.is_alphabetic() || c == '_') {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == ':' {
                out.push('"');
                out.push_str(&word);
                out.push('"');
            } else {
                out.push_str(&word);
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    s = out;
    s
}

/// Parse every tool call in `text`. Empty when the text is ordinary content.
pub fn parse(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    // Gemma native syntax.
    while let Some(start) = rest.find("<|tool_call>") {
        let after = &rest[start + "<|tool_call>".len()..];
        let end = after.find("<tool_call|>").unwrap_or(after.len());
        let body = after[..end].trim();
        if let Some(call) = body.strip_prefix("call:") {
            let brace = call.find('{').unwrap_or(call.len());
            let name = call[..brace].trim().to_string();
            let args_raw = if brace < call.len() { &call[brace..] } else { "{}" };
            let args_json = gemma_args_to_json(args_raw);
            let arguments = match serde_json::from_str::<Value>(&args_json) {
                Ok(v) => v.to_string(),
                Err(_) => json!({"raw": args_raw}).to_string(),
            };
            if !name.is_empty() {
                calls.push(ToolCall { name, arguments });
            }
        }
        rest = &after[end.min(after.len())..];
    }
    if !calls.is_empty() {
        return calls;
    }
    // Manual-fallback JSON: {"tool_call": {...}} or {"name": …, "arguments": …},
    // possibly followed by a sign-off the model added after closing it.
    let cleaned = estia_engine::structured::clean_json(text);
    if let Ok((v, _trailing)) = estia_engine::structured::first_json_value(&cleaned) {
        let candidates: Vec<&Value> = match &v {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for c in candidates {
            let obj = c.get("tool_call").unwrap_or(c);
            let name = obj.get("name").and_then(Value::as_str);
            if let Some(name) = name {
                let args = obj.get("arguments").cloned().unwrap_or_else(|| json!({}));
                let arguments = match args {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                calls.push(ToolCall { name: name.to_string(), arguments });
            }
        }
    }
    calls
}

/// The tool calls of a finished generation, OpenAI-shaped. `runner_parses`
/// is the runner's `parses_tool_calls` capability: its `meta.tool_calls` are
/// used as they came (with an `id`, `type` and string `arguments` filled in
/// where missing), and the text is not parsed. Otherwise the text is parsed.
/// Every call gets its id here, once: `call_<random>` unless the runner
/// named it.
pub fn from_output(runner_parses: bool, text: &str, meta_calls: Option<&[Value]>) -> Vec<Value> {
    if runner_parses {
        return meta_calls.unwrap_or_default().iter().filter_map(normalize).collect();
    }
    parse(text).iter().map(ToolCall::to_openai).collect()
}

/// One runner-parsed call in OpenAI's shape; `None` when it has no name.
fn normalize(c: &Value) -> Option<Value> {
    let f = c.get("function")?;
    let name = f.get("name").and_then(Value::as_str).filter(|n| !n.is_empty())?;
    let arguments = match f.get("arguments") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
        None | Some(Value::Null) | Some(Value::String(_)) => "{}".to_string(),
        Some(other) => other.to_string(),
    };
    let id = c.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(call_id);
    Some(json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}))
}

/// Calls as a streamed delta carries them: each with its `index`, which
/// OpenAI's clients use to put the pieces of a call together.
pub fn indexed(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = c.clone();
            if let Some(obj) = c.as_object_mut() {
                obj.insert("index".into(), json!(i));
            }
            c
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_parsed_calls_are_used_and_the_text_is_not_parsed() {
        let meta = vec![
            json!({"id": "c9", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Athens\"}"}}),
            json!({"function": {"name": "now", "arguments": {"tz": "UTC"}}}),
            json!({"function": {"name": "", "arguments": "{}"}}),
            json!({"function": {"name": "ping"}}),
        ];
        let gemma = r#"<|tool_call>call:wrong{}<tool_call|>"#;
        let calls = from_output(true, gemma, Some(&meta));
        assert_eq!(calls.len(), 3, "the nameless call is dropped: {calls:?}");
        assert_eq!(
            calls[0],
            json!({"id": "c9", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Athens\"}"}})
        );
        let id1 = calls[1]["id"].as_str().unwrap();
        let id2 = calls[2]["id"].as_str().unwrap();
        assert!(id1.starts_with("call_") && id1.len() == "call_".len() + 24, "{id1}");
        assert_ne!(id1, id2, "calls in one response have distinct ids");
        assert_eq!(calls[1]["function"]["arguments"], "{\"tz\":\"UTC\"}");
        assert_eq!(calls[2]["function"]["arguments"], "{}");
        assert!(from_output(true, gemma, None).is_empty(), "a parsing runner's text is content");
        // A runner that does not parse: the text is.
        let calls = from_output(false, gemma, Some(&meta));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "wrong");
        let streamed = indexed(&calls);
        assert_eq!(streamed[0]["index"], 0);
        assert_eq!(streamed[0]["function"]["name"], "wrong");
    }

    #[test]
    fn gemma_native_syntax() {
        let calls = parse(r#"<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"city": "Athens"}));
        let calls = parse(r#"<|tool_call>call:a{n:3,flag:true,list:[<|"|>x<|"|>,2]}<tool_call|><|tool_call>call:b{}<tool_call|>"#);
        assert_eq!(calls.len(), 2);
        assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"n": 3, "flag": true, "list": ["x", 2]}));
        assert_eq!(calls[1].arguments, "{}");
        let oai = calls[0].to_openai();
        assert_eq!(oai["function"]["name"], "a");
        assert_eq!(oai["type"], "function");
    }

    /// Ids never repeat: not between the calls of one response, not between
    /// responses (a conversation sends many `tool` results back, matched by id).
    #[test]
    fn call_ids_are_unique() {
        let text = r#"<|tool_call>call:a{}<tool_call|><|tool_call>call:b{}<tool_call|>"#;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            for c in from_output(false, text, None) {
                let id = c["id"].as_str().unwrap().to_string();
                assert!(id.starts_with("call_"), "{id}");
                assert!(seen.insert(id.clone()), "{id} repeated");
            }
        }
        // A parsing runner's own ids are kept; a missing one is filled in fresh.
        let meta = vec![json!({"id": "srv-1", "function": {"name": "x"}}), json!({"function": {"name": "y"}})];
        let calls = from_output(true, "", Some(&meta));
        assert_eq!(calls[0]["id"], "srv-1");
        assert!(calls[1]["id"].as_str().unwrap().starts_with("call_"));
    }

    /// A fallback call the model closed and then signed off after still
    /// counts, as structured output does (api-F1); prose is still prose.
    #[test]
    fn a_json_call_followed_by_text_is_a_call() {
        let calls =
            parse("{\"tool_call\": {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Athens\"}}}\nLet me know if you need more.");
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"city": "Athens"}));
        assert!(parse("The weather in Athens is sunny.").is_empty());
        assert!(parse("{\"city\": \"Athens\"} is what I found.").is_empty(), "JSON without a name is not a call");
    }

    #[test]
    fn json_fallback_and_plain_content() {
        let calls = parse(r#"{"tool_call": {"name": "get_weather", "arguments": {"city": "Athens"}}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let calls = parse("Sure, the weather in Athens is sunny.");
        assert!(calls.is_empty());
        let calls = parse(r#"{"title": "not a tool call"}"#);
        assert!(calls.is_empty());
    }
}
