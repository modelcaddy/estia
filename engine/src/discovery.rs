//! Engines on the LAN, found over Bonjour/mDNS (`_estia._tcp`). Blocking.
//!
//! `estia discover` and host apps (ModelCaddy's "engine on another machine")
//! share this. Behind the `discovery` feature, which brings `mdns-sd`.

use std::time::Instant;

/// The service type `estia serve --lan` advertises.
pub const MDNS_SERVICE_TYPE: &str = "_estia._tcp.local.";

/// This machine's non-loopback, non-link-local IPv4 addresses (via `ifconfig`).
pub fn lan_ipv4_addresses() -> Vec<std::net::Ipv4Addr> {
    let mut out = Vec::new();
    if let Ok(o) = std::process::Command::new("ifconfig").output() {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            if let Some(rest) = line.trim().strip_prefix("inet ") {
                if let Some(ip) = rest.split_whitespace().next().and_then(|s| s.parse::<std::net::Ipv4Addr>().ok()) {
                    if !ip.is_loopback() && !ip.is_link_local() {
                        out.push(ip);
                    }
                }
            }
        }
    }
    out
}

/// One engine found on the LAN.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Discovered {
    pub name: String,
    pub host: String,
    pub addresses: Vec<String>,
    pub port: u16,
    pub api_version: Option<String>,
    pub engine_version: Option<String>,
}

/// Order the addresses by what a client can actually dial: routable IPv4
/// first, then routable IPv6, then link-local IPv6, loopback last.
///
/// An engine on *this* machine is a special case worth handling rather than
/// printing: macOS resolves the machine's own `.local` name to 127.0.0.1, so a
/// local engine is discovered at an address that is correct here and useless in
/// the "point your other device at this" line the CLI prints from it. When
/// nothing routable came back, this machine's own LAN addresses are added.
fn usable_addresses(addrs: Vec<String>) -> Vec<String> {
    fn rank(a: &str) -> u8 {
        match a.parse::<std::net::IpAddr>() {
            Ok(ip) if ip.is_loopback() => 3,
            Ok(std::net::IpAddr::V4(_)) => 0,
            Ok(std::net::IpAddr::V6(v6)) if (v6.segments()[0] & 0xffc0) == 0xfe80 => 2,
            Ok(std::net::IpAddr::V6(_)) => 1,
            Err(_) => 3,
        }
    }
    let mut out = addrs;
    let routable_v4 = out.iter().any(|a| rank(a) == 0);
    if !routable_v4 && out.iter().any(|a| rank(a) == 3) {
        for ip in lan_ipv4_addresses() {
            out.push(ip.to_string());
        }
    }
    out.sort_by_key(|a| rank(a));
    let mut seen = std::collections::HashSet::new();
    out.retain(|a| seen.insert(a.clone()));
    out
}

/// Browse for `_estia._tcp` for `window`. Blocking.
pub fn discover(window: std::time::Duration) -> anyhow::Result<Vec<Discovered>> {
    let mdns = mdns_sd::ServiceDaemon::new()?;
    let rx = mdns.browse(crate::discovery::MDNS_SERVICE_TYPE)?;
    let deadline = Instant::now() + window;
    let mut found: Vec<Discovered> = Vec::new();
    #[allow(unused_mut)]
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(info)) => {
                let name = info.get_fullname().split('.').next().unwrap_or("").to_string();
                let mut addresses: Vec<String> = info.get_addresses().iter().map(|a| a.to_string()).collect();
                let host = info.get_hostname().trim_end_matches('.').to_string();
                // No IPv4 in the announcement: ask the resolver for `host.local`,
                // which on macOS and Linux goes through mDNS and returns IPv4.
                if !addresses.iter().any(|a| a.parse::<std::net::Ipv4Addr>().is_ok()) {
                    use std::net::ToSocketAddrs;
                    addresses.extend(
                        (host.as_str(), info.get_port())
                            .to_socket_addrs()
                            .map(|it| it.filter(|a| a.is_ipv4()).map(|a| a.ip().to_string()).collect::<Vec<_>>())
                            .unwrap_or_default(),
                    );
                }
                let addresses = usable_addresses(addresses);
                let d = Discovered {
                    name,
                    host,
                    addresses,
                    port: info.get_port(),
                    api_version: info.get_property_val_str("api_version").map(str::to_string),
                    engine_version: info.get_property_val_str("engine_version").map(str::to_string),
                };
                if !found.iter().any(|f| f.name == d.name && f.port == d.port) {
                    found.push(d);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let _ = mdns.shutdown();
    if found.is_empty() {
        // The mdns-sd browser misses instances the system resolver sees (a
        // daemon under launchd, in practice). Ask mDNSResponder itself.
        #[cfg(target_os = "macos")]
        {
            found = discover_via_dns_sd(window.min(std::time::Duration::from_secs(4)));
        }
    }
    Ok(found)
}

/// macOS fallback: `dns-sd -B` to list instances, `dns-sd -L` per instance for
/// host, port and TXT, then the OS resolver for the host's IPv4.
#[cfg(target_os = "macos")]
pub fn discover_via_dns_sd(window: std::time::Duration) -> Vec<Discovered> {
    use std::io::Read;
    use std::net::ToSocketAddrs;
    fn run_for(args: &[&str], window: std::time::Duration) -> String {
        // `dns-sd` browses forever and block-buffers whenever its stdout is not
        // a terminal: killed at the end of the window it flushes nothing, so
        // reading it straight into a pipe or a file returns an empty string no
        // matter which signal it is sent. Run it under `script`, which gives it
        // a pty and forwards what it prints, line by line, to `script`'s own
        // stdout — which we point at a file and read once the window closes.
        //
        // Two details that both look like the same empty-output bug:
        // the typescript `script` writes (its first argument) is itself
        // buffered and only flushed on a clean exit, hence `/dev/null` there
        // and the redirect for the real capture; and `script` ends the session
        // when its stdin reaches EOF, so stdin is a pipe we hold open rather
        // than `/dev/null`, which would kill `dns-sd` immediately.
        let out_path =
            std::env::temp_dir().join(format!("estia-dns-sd-{}-{}.txt", std::process::id(), args.join("_").replace(['/', '.', ' '], "-")));
        let Ok(out_file) = std::fs::File::create(&out_path) else {
            return String::new();
        };
        let mut cmd = std::process::Command::new("script");
        cmd.arg("-q").arg("/dev/null").arg("dns-sd").args(args);
        cmd.stdout(std::process::Stdio::from(out_file)).stderr(std::process::Stdio::null()).stdin(std::process::Stdio::piped());
        let Ok(mut child) = cmd.spawn() else {
            let _ = std::fs::remove_file(&out_path);
            return String::new();
        };
        let stdin = child.stdin.take();
        std::thread::sleep(window);
        let _ = child.kill();
        let _ = child.wait();
        drop(stdin);
        let mut out = String::new();
        if let Ok(mut f) = std::fs::File::open(&out_path) {
            let mut buf = Vec::new();
            let _ = f.read_to_end(&mut buf);
            out = String::from_utf8_lossy(&buf).replace('\r', "");
        }
        let _ = std::fs::remove_file(&out_path);
        out
    }
    let ty = MDNS_SERVICE_TYPE.trim_end_matches(".local.");
    let browse = run_for(&["-B", ty, "local."], window);
    let mut instances: Vec<String> = Vec::new();
    for line in browse.lines() {
        // "Timestamp  A/R Flags if Domain  Service Type  Instance Name"
        if line.contains(" Add ") && line.contains(ty) {
            if let Some(idx) = line.find(ty) {
                let name = line[idx + ty.len()..].trim().trim_start_matches('.').trim().to_string();
                if !name.is_empty() && !instances.contains(&name) {
                    instances.push(name);
                }
            }
        }
    }
    let mut found = Vec::new();
    for inst in instances {
        let lookup = run_for(&["-L", &inst, ty, "local."], std::time::Duration::from_secs(2));
        let mut host = String::new();
        let mut port: u16 = 0;
        let mut api_version = None;
        let mut engine_version = None;
        for line in lookup.lines() {
            if let Some(idx) = line.find("can be reached at ") {
                let rest = line[idx + "can be reached at ".len()..].trim();
                let hp = rest.split_whitespace().next().unwrap_or("");
                if let Some((h, p)) = hp.rsplit_once(':') {
                    host = h.trim_end_matches('.').to_string();
                    port = p.parse().unwrap_or(0);
                }
            }
            for kv in line.split_whitespace() {
                if let Some(v) = kv.strip_prefix("api_version=") {
                    api_version = Some(v.to_string());
                }
                if let Some(v) = kv.strip_prefix("engine_version=") {
                    engine_version = Some(v.to_string());
                }
            }
        }
        if host.is_empty() || port == 0 {
            continue;
        }
        let addresses: Vec<String> = (host.as_str(), port)
            .to_socket_addrs()
            .map(|it| it.filter(|a| a.is_ipv4()).map(|a| a.ip().to_string()).collect())
            .unwrap_or_default();
        let addresses = usable_addresses(addresses);
        found.push(Discovered { name: inst, host, addresses, port, api_version, engine_version });
    }
    found
}
