//! Roles: stable names a client asks for, bound by the host to model families.
//!
//! A client says `text`, `fast`, `vision`, `embed` or `code` — or any name the
//! host added — and the host decides which family serves it. Binding is
//! checked against the family's capabilities, so an embedding model can never
//! be bound to `code`, and an unset role falls back along a fixed chain rather
//! than failing: `code → text`, `vision → text`, `fast → text`.
//!
//! Roles carry sampling defaults and a pin flag. They never carry a prompt.

use crate::models::registry::{self, Capability, Format};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const TEXT: &str = "text";
pub const FAST: &str = "fast";
pub const VISION: &str = "vision";
pub const EMBED: &str = "embed";
pub const CODE: &str = "code";

/// What a role needs from the family bound to it.
pub fn required_capability(role: &str) -> Capability {
    match role {
        EMBED => Capability::Embed,
        VISION => Capability::Vision,
        _ => Capability::Text,
    }
}

/// Where an unset role looks next. `None` means the role is terminal.
pub fn fallback(role: &str) -> Option<&'static str> {
    match role {
        CODE | VISION | FAST => Some(TEXT),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleBinding {
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Keep this role's model resident regardless of idle time.
    #[serde(default)]
    pub pin: bool,
}

impl RoleBinding {
    pub fn family(family: impl Into<String>) -> Self {
        Self { family: family.into(), temperature: None, max_tokens: None, pin: false }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RoleError {
    #[error("unknown family `{0}`")]
    UnknownFamily(String),
    #[error("family `{family}` lacks the `{needed:?}` capability role `{role}` needs")]
    Incapable { role: String, family: String, needed: Capability },
    #[error("role `{0}` is not bound and has no fallback")]
    Unbound(String),
}

/// The host's role table.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Roles {
    #[serde(flatten)]
    bindings: BTreeMap<String, RoleBinding>,
}

/// What a role resolved to.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    /// The role that was asked for.
    pub asked: String,
    /// The role whose binding answered (the same, or a fallback).
    pub served_by: String,
    pub binding: RoleBinding,
}

impl Roles {
    /// The built-in defaults for an Apple silicon host today.
    pub fn defaults() -> Self {
        let mut r = Roles::default();
        r.bindings.insert(TEXT.into(), RoleBinding::family("gemma4-e4b"));
        r.bindings.insert(FAST.into(), RoleBinding::family("gemma4-e2b"));
        r.bindings.insert(VISION.into(), RoleBinding::family("gemma4-e4b"));
        r
    }

    pub fn get(&self, role: &str) -> Option<&RoleBinding> {
        self.bindings.get(role)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &RoleBinding)> {
        self.bindings.iter()
    }

    /// Bind a role, refusing a family that lacks what the role needs or that
    /// the registry does not know. Any role name is accepted; the built-in
    /// names carry capability requirements, others need `text`.
    pub fn bind(&mut self, role: &str, binding: RoleBinding) -> Result<(), RoleError> {
        let needed = required_capability(role);
        if registry::family_has(&binding.family, needed) {
            self.bindings.insert(role.to_string(), binding);
            Ok(())
        } else if registry::family_known(&binding.family) {
            Err(RoleError::Incapable { role: role.to_string(), family: binding.family, needed })
        } else {
            Err(RoleError::UnknownFamily(binding.family))
        }
    }

    pub fn unbind(&mut self, role: &str) -> Option<RoleBinding> {
        self.bindings.remove(role)
    }

    /// Resolve a role through its fallback chain.
    pub fn resolve(&self, role: &str) -> Result<Resolution, RoleError> {
        let mut current = role;
        loop {
            if let Some(binding) = self.bindings.get(current) {
                return Ok(Resolution { asked: role.to_string(), served_by: current.to_string(), binding: binding.clone() });
            }
            match fallback(current) {
                Some(next) => current = next,
                None => return Err(RoleError::Unbound(role.to_string())),
            }
        }
    }

    /// Resolve a role to the artifact a host running `format` would load.
    pub fn resolve_artifact(&self, role: &str, format: Format) -> Result<(Resolution, &'static registry::Artifact), RoleError> {
        let res = self.resolve(role)?;
        let artifact = registry::find_family_default(&res.binding.family, format)
            .ok_or_else(|| RoleError::UnknownFamily(res.binding.family.clone()))?;
        Ok((res, artifact))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_resolve_and_code_falls_back_to_text() {
        let r = Roles::defaults();
        let t = r.resolve(TEXT).unwrap();
        assert_eq!(t.binding.family, "gemma4-e4b");
        let c = r.resolve(CODE).unwrap();
        assert_eq!(c.asked, CODE);
        assert_eq!(c.served_by, TEXT);
        assert_eq!(c.binding.family, "gemma4-e4b");
        let (_, artifact) = r.resolve_artifact(FAST, Format::Mlx).unwrap();
        assert_eq!(artifact.id, "gemma4-e2b-it-4bit-mlx");
        assert_eq!(r.resolve(EMBED), Err(RoleError::Unbound(EMBED.into())));
        assert_eq!(r.resolve("writer"), Err(RoleError::Unbound("writer".into())));
    }

    #[test]
    fn binding_checks_capabilities_and_registry() {
        let mut r = Roles::defaults();
        r.bind("writer", RoleBinding::family("gemma4-12b-qat")).unwrap();
        assert_eq!(r.resolve("writer").unwrap().binding.family, "gemma4-12b-qat");
        assert_eq!(r.bind(CODE, RoleBinding::family("nope")), Err(RoleError::UnknownFamily("nope".into())));
        // No text-capable embed family exists in the registry today, and the
        // embed role needs `Embed`, which no generation family has.
        assert!(matches!(r.bind(EMBED, RoleBinding::family("gemma4-e4b")), Err(RoleError::Incapable { .. })));
        assert!(r.unbind("writer").is_some());
        assert!(r.unbind("writer").is_none());
    }

    #[test]
    fn roles_round_trip_through_json() {
        let mut r = Roles::defaults();
        r.bind(CODE, RoleBinding { family: "gemma4-e2b".into(), temperature: Some(0.2), max_tokens: None, pin: true }).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"code\":{\"family\":\"gemma4-e2b\",\"temperature\":0.2,\"pin\":true}"), "{json}");
        let back: Roles = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }
}
