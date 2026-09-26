//! Turn a model's tool-call text into OpenAI `tool_calls`.
//!
//! Gemma 4 answers a tools request in its own syntax:
//! `<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>` — bare keys,
//! `<|"|>` as the string quote, several calls allowed back to back. The manual
//! template fallback asks instead for `{"tool_call": {"name": …, "arguments": {…}}}`.
//! Both are recognised; anything else is plain content.

use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Arguments as a JSON object, serialized (OpenAI sends a string).
    pub arguments: String,
}

impl ToolCall {
    pub fn to_openai(&self, index: usize) -> Value {
        json!({
            "id": format!("call_{index}"),
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments}
        })
    }
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
    // Manual-fallback JSON: {"tool_call": {...}} or {"name": …, "arguments": …}.
    let cleaned = estia_engine::structured::clean_json(text);
    if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let oai = calls[0].to_openai(0);
        assert_eq!(oai["function"]["name"], "a");
        assert_eq!(oai["type"], "function");
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
