//! Engines on the LAN, found over Bonjour/mDNS (`_estia._tcp`). Blocking.
//!
//! `estia discover` and host apps (ModelCaddy's "engine on another machine")
//! share this. Behind the `discovery` feature, which brings `mdns-sd`.
//!
//! An engine serving HTTPS says so in its TXT record (`tls=1`) along with its
//! certificate's fingerprint (`fp=<64 hex>`). That claim is not trusted:
//! anyone on the LAN can advertise anything. Its use is to find a known
//! engine again at a new address (it moved to another DHCP lease) and connect
//! there with the pin the app already holds, which is then checked for real,
//! or to say "this engine's certificate changed since you paired" before
//! connecting at all.

use crate::tls::{Fingerprint, UnverifiedFingerprint};
use std::time::Instant;

/// The service type `estia serve --lan` advertises.
pub const MDNS_SERVICE_TYPE: &str = "_estia._tcp.local.";

/// This machine's non-loopback, non-link-local IPv4 addresses: from
/// `ifconfig`, or from `ip -4 -o addr show` where there is no `ifconfig` (a
/// Linux without net-tools, such as a stock Ubuntu).
pub fn lan_ipv4_addresses() -> Vec<std::net::Ipv4Addr> {
    let tools: [(&str, &[&str]); 2] = [("ifconfig", &[]), ("ip", &["-4", "-o", "addr", "show"])];
    for (tool, args) in tools {
        if let Ok(o) = std::process::Command::new(tool).args(args).output() {
            if o.status.success() {
                return inet_addresses(&String::from_utf8_lossy(&o.stdout));
            }
        }
    }
    Vec::new()
}

/// The IPv4 address after each `inet` word, as `ifconfig` prints it (BSD
/// `inet 192.168.1.20 netmask …`; net-tools `inet 192.168.1.20` or the older
/// `inet addr:192.168.1.20`) and as `ip -o` prints it (`inet
/// 192.168.1.20/24`), without loopback, link-local or repeats.
fn inet_addresses(text: &str) -> Vec<std::net::Ipv4Addr> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        if words.by_ref().any(|w| w == "inet") {
            let Some(word) = words.next() else { continue };
            let word = word.strip_prefix("addr:").unwrap_or(word);
            let Ok(ip) = word.split('/').next().unwrap_or(word).parse::<std::net::Ipv4Addr>() else { continue };
            if !ip.is_loopback() && !ip.is_link_local() && !out.contains(&ip) {
                out.push(ip);
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
    /// The advertisement says the engine serves HTTPS (`tls=1`).
    pub tls: bool,
    /// The certificate fingerprint the advertisement claims (`fp`). A hint:
    /// compare it with a stored pin, never pin it.
    pub fingerprint: Option<UnverifiedFingerprint>,
}

impl Discovered {
    /// Where to reach it: the first usable address (they are ordered by what a
    /// client can dial), `https` when the engine serves TLS. `None` when the
    /// record carried no address at all.
    pub fn url(&self) -> Option<String> {
        let host = self.addresses.first().or(Some(&self.host)).filter(|h| !h.is_empty())?;
        let scheme = if self.tls { "https" } else { "http" };
        let host = if host.contains(':') && !host.starts_with('[') { format!("[{}]", host.replace('%', "%25")) } else { host.clone() };
        Some(format!("{scheme}://{host}:{}", self.port))
    }
}

/// The TXT keys an engine advertises, as `discover` reads them.
#[derive(Debug, Default, Clone, PartialEq)]
struct Txt {
    api_version: Option<String>,
    engine_version: Option<String>,
    tls: bool,
    fingerprint: Option<UnverifiedFingerprint>,
}

impl Txt {
    fn set(&mut self, key: &str, value: &str) {
        match key {
            "api_version" => self.api_version = Some(value.to_string()),
            "engine_version" => self.engine_version = Some(value.to_string()),
            "tls" => self.tls = value == "1",
            "fp" => self.fingerprint = Fingerprint::parse(value).ok().map(UnverifiedFingerprint::new),
            _ => {}
        }
    }
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
                let mut txt = Txt::default();
                for key in ["api_version", "engine_version", "tls", "fp"] {
                    if let Some(v) = info.get_property_val_str(key) {
                        txt.set(key, v);
                    }
                }
                let d = Discovered {
                    name,
                    host,
                    addresses,
                    port: info.get_port(),
                    api_version: txt.api_version,
                    engine_version: txt.engine_version,
                    tls: txt.tls,
                    fingerprint: txt.fingerprint,
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
        let Some((host, port, txt)) = parse_lookup(&lookup) else { continue };
        let addresses: Vec<String> = (host.as_str(), port)
            .to_socket_addrs()
            .map(|it| it.filter(|a| a.is_ipv4()).map(|a| a.ip().to_string()).collect())
            .unwrap_or_default();
        let addresses = usable_addresses(addresses);
        found.push(Discovered {
            name: inst,
            host,
            addresses,
            port,
            api_version: txt.api_version,
            engine_version: txt.engine_version,
            tls: txt.tls,
            fingerprint: txt.fingerprint,
        });
    }
    found
}

/// What `dns-sd -L` printed for one instance: the host and port from the
/// "can be reached at" line, and the TXT keys from the `key=value` words
/// (on that line or the next). `None` when no host and port were found.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_lookup(text: &str) -> Option<(String, u16, Txt)> {
    let mut host = String::new();
    let mut port: u16 = 0;
    let mut txt = Txt::default();
    for line in text.lines() {
        if let Some(idx) = line.find("can be reached at ") {
            let rest = line[idx + "can be reached at ".len()..].trim();
            let hp = rest.split_whitespace().next().unwrap_or("");
            if let Some((h, p)) = hp.rsplit_once(':') {
                host = h.trim_end_matches('.').to_string();
                port = p.parse().unwrap_or(0);
            }
        }
        for kv in line.split_whitespace() {
            if let Some((k, v)) = kv.split_once('=') {
                txt.set(k, v);
            }
        }
    }
    (!host.is_empty() && port != 0).then_some((host, port, txt))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The address lines of macOS `ifconfig`, both net-tools formats, and
    /// `ip -4 -o addr show` (all a Linux without net-tools has).
    #[test]
    fn lan_addresses_from_every_tool() {
        let mac =
            "lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384\n\tinet 127.0.0.1 netmask 0xff000000\n\tinet6 ::1 prefixlen 128\n\
                   en0: flags=8863<UP> mtu 1500\n\tinet6 fe80::1c%en0 prefixlen 64 secured scopeid 0xb\n\
                   \tinet 192.168.1.20 netmask 0xffffff00 broadcast 192.168.1.255\n\tinet 169.254.3.4 netmask 0xffff0000\n";
        assert_eq!(inet_addresses(mac), ["192.168.1.20".parse::<std::net::Ipv4Addr>().unwrap()]);
        let net_tools = "eth0: flags=4163<UP>  mtu 1500\n        inet 10.0.0.7  netmask 255.255.255.0  broadcast 10.0.0.255\n\
                         eth1      Link encap:Ethernet\n          inet addr:10.0.1.7  Bcast:10.0.1.255  Mask:255.255.255.0\n";
        assert_eq!(inet_addresses(net_tools), ["10.0.0.7".parse::<std::net::Ipv4Addr>().unwrap(), "10.0.1.7".parse().unwrap()]);
        let ip = "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever\n\
                  2: eth0    inet 192.168.1.30/24 brd 192.168.1.255 scope global dynamic eth0\\       valid_lft 86000sec\n\
                  3: tailscale0    inet 100.101.102.103/32 scope global tailscale0\n\
                  4: eth1    inet 192.168.1.30/24 scope global secondary eth1\n";
        assert_eq!(inet_addresses(ip), ["192.168.1.30".parse::<std::net::Ipv4Addr>().unwrap(), "100.101.102.103".parse().unwrap()]);
        assert!(inet_addresses("").is_empty());
    }

    /// A `dns-sd -L` capture from macOS (under `script`, carriage returns
    /// already stripped), of an engine serving HTTPS.
    const LOOKUP: &str = "Lookup studio._estia._tcp.local.
DATE: ---Wed 08 Oct 2026---
12:01:02.345  ...STARTING...
12:01:02.346  studio._estia._tcp.local. can be reached at Studio.local.:27200 (interface 14)
 api_version=1 engine_version=0.5.0 protocol_version=2 tls=1 fp=3f2a9c417b0ed5120001020304050607080900aabbccddeeff00112233445566
";

    #[test]
    fn dns_sd_lookup_lines_give_host_port_and_txt() {
        let (host, port, txt) = parse_lookup(LOOKUP).expect("parsed");
        assert_eq!((host.as_str(), port), ("Studio.local", 27200));
        assert_eq!(txt.api_version.as_deref(), Some("1"));
        assert_eq!(txt.engine_version.as_deref(), Some("0.5.0"));
        assert!(txt.tls);
        let fp = txt.fingerprint.expect("fp");
        assert_eq!(fp.short(), "3f2a 9c41 7b0e d512");
        // An older engine: no tls, no fp.
        let (_, _, old) = parse_lookup(&LOOKUP.replace(" tls=1", "").replace(" fp=", " x=")).unwrap();
        assert!(!old.tls && old.fingerprint.is_none());
        // A malformed fp is dropped, not trusted.
        let (_, _, bad) = parse_lookup(&LOOKUP.replace("fp=3f2a", "fp=zz2a")).unwrap();
        assert!(bad.fingerprint.is_none());
        assert!(parse_lookup("nothing here").is_none());
    }

    #[test]
    fn url_follows_the_tls_key() {
        let mut d = Discovered {
            name: "studio".into(),
            host: "studio.local".into(),
            addresses: vec!["192.168.1.20".into(), "fe80::1".into()],
            port: 27200,
            api_version: Some("1".into()),
            engine_version: None,
            tls: true,
            fingerprint: None,
        };
        assert_eq!(d.url().as_deref(), Some("https://192.168.1.20:27200"));
        d.tls = false;
        assert_eq!(d.url().as_deref(), Some("http://192.168.1.20:27200"));
        d.addresses = vec!["fe80::1%en0".into()];
        assert_eq!(d.url().as_deref(), Some("http://[fe80::1%25en0]:27200"));
        d.addresses.clear();
        assert_eq!(d.url().as_deref(), Some("http://studio.local:27200"));
        d.host.clear();
        assert_eq!(d.url(), None);
    }
}
