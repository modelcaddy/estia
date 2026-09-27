//! The physical memory a process is using, by pid: what a host shows per
//! runner (`/engine/stats`) so a user can see which model holds the memory.
//!
//! - macOS: the process's physical footprint (`ri_phys_footprint` from
//!   `proc_pid_rusage`), the figure Activity Monitor's Memory column shows.
//!   It counts memory the process dirtied and still holds, compressed or
//!   swapped pages included, and the GPU (Metal) buffers it allocated, so an
//!   MLX runner's weights and KV cache are in it. Readable for processes of
//!   the same user; another user's process reads as `None`.
//! - Linux: the resident set size (`VmRSS` in `/proc/<pid>/status`). Not the
//!   same measure: it counts shared pages (mapped libraries, a model file
//!   mapped read-only) and leaves out swapped pages and GPU memory.
//! - Elsewhere: `None`.
//!
//! [`tree_footprint`] adds up a process and everything under it: an MLX
//! runner holds its weights itself, while the llama.cpp adapter's weights
//! live in the `llama-server` it starts.

/// Bytes of physical memory process `pid` is using (see the module notes for
/// what that counts on each platform), or `None` when the process does not
/// exist, may not be inspected, or the platform has no reading.
pub fn phys_footprint(pid: u32) -> Option<u64> {
    imp::phys_footprint(pid)
}

/// [`phys_footprint`] of `pid` plus that of every process descended from it
/// (a descendant that cannot be read counts as nothing). `None` when `pid`
/// itself cannot be read. Walks at most 256 processes.
pub fn tree_footprint(pid: u32) -> Option<u64> {
    let mut total = phys_footprint(pid)?;
    let mut seen = vec![pid];
    let mut stack = children_of(pid);
    while let Some(p) = stack.pop() {
        if seen.contains(&p) || seen.len() >= 256 {
            continue;
        }
        seen.push(p);
        total += phys_footprint(p).unwrap_or(0);
        stack.extend(children_of(p));
    }
    Some(total)
}

/// The direct children of `pid`; empty when there are none, when `pid` does
/// not exist, or on a platform without a way to list them.
pub fn children_of(pid: u32) -> Vec<u32> {
    imp::children_of(pid)
}

#[cfg(target_os = "macos")]
mod imp {
    pub fn phys_footprint(pid: u32) -> Option<u64> {
        let pid = libc::c_int::try_from(pid).ok()?;
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
        // SAFETY: `info` is a writable `rusage_info_v2`, the struct the
        // RUSAGE_INFO_V2 flavour fills. The C signature takes it as a
        // `rusage_info_t *` (`rusage_info_t` is `void *`), hence the cast.
        let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast::<libc::rusage_info_t>()) };
        if rc != 0 {
            return None;
        }
        // SAFETY: the call succeeded, so the kernel filled the struct (and it
        // was zeroed before, so every field is initialised either way).
        Some(unsafe { info.assume_init() }.ri_phys_footprint)
    }

    pub fn children_of(pid: u32) -> Vec<u32> {
        let Ok(ppid) = libc::pid_t::try_from(pid) else { return Vec::new() };
        let mut buf = vec![0 as libc::pid_t; 1024];
        let bytes = libc::c_int::try_from(buf.len() * std::mem::size_of::<libc::pid_t>()).unwrap_or(libc::c_int::MAX);
        // SAFETY: `buf` is writable for `bytes` bytes, which is what we pass.
        let n = unsafe { libc::proc_listchildpids(ppid, buf.as_mut_ptr().cast(), bytes) };
        if n <= 0 {
            return Vec::new();
        }
        // The return value is a count of pids on some macOS releases and of
        // bytes on others; the buffer was zeroed, and a pid is never 0.
        buf.truncate((n as usize).min(buf.len()));
        buf.into_iter().filter(|p| *p > 0).map(|p| p as u32).collect()
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn phys_footprint(pid: u32) -> Option<u64> {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        super::vm_rss_bytes(&status)
    }

    /// From every thread's `children` list: a child is listed under the
    /// thread that forked it.
    pub fn children_of(pid: u32) -> Vec<u32> {
        let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else { return Vec::new() };
        let mut out = Vec::new();
        for t in tasks.flatten() {
            if let Ok(list) = std::fs::read_to_string(t.path().join("children")) {
                out.extend(list.split_whitespace().filter_map(|p| p.parse::<u32>().ok()));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn phys_footprint(_pid: u32) -> Option<u64> {
        None
    }

    pub fn children_of(_pid: u32) -> Vec<u32> {
        Vec::new()
    }
}

/// `VmRSS` from the text of a Linux `/proc/<pid>/status`, in bytes. The
/// kernel reports it in kB (KiB). `None` when the line is missing (a kernel
/// thread, or a zombie).
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
fn vm_rss_bytes(status: &str) -> Option<u64> {
    let line = status.lines().find_map(|l| l.strip_prefix("VmRSS:"))?;
    let mut parts = line.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") | None => n.checked_mul(1024),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn the_current_process_has_a_footprint_in_bytes() {
        let n = phys_footprint(std::process::id()).expect("own footprint");
        // A test binary holds at least a few MiB; a count in KiB or pages
        // would come out far smaller, one in bytes far below a TiB.
        assert!(n > MIB && n < 1024 * 1024 * MIB, "{n}");
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn touching_memory_raises_the_footprint() {
        let pid = std::process::id();
        let before = phys_footprint(pid).unwrap();
        // Written, not zeroed: a zeroed allocation can stay untouched pages.
        let block = std::hint::black_box(vec![0x5au8; (64 * MIB) as usize]);
        let during = phys_footprint(pid).unwrap();
        assert!(during >= before + 32 * MIB, "before {before}, with 64 MiB touched {during}");
        drop(block);
    }

    #[test]
    fn a_missing_process_has_none() {
        // Above every platform's pid ceiling (macOS 99999, Linux 2^22).
        assert_eq!(phys_footprint(i32::MAX as u32), None);
        assert_eq!(phys_footprint(u32::MAX), None);
        assert_eq!(tree_footprint(i32::MAX as u32), None);
        assert!(children_of(i32::MAX as u32).is_empty());
    }

    /// A process tree: `sh` starts `sleep` and waits. The tree's footprint
    /// counts the grandchild too, and the children are listed.
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn a_tree_counts_its_descendants() {
        let mut sh = std::process::Command::new("sh").args(["-c", "sleep 5 & wait"]).spawn().unwrap();
        let pid = sh.id();
        assert!(children_of(std::process::id()).contains(&pid));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let kids = loop {
            let kids = children_of(pid);
            if !kids.is_empty() || std::time::Instant::now() > deadline {
                break kids;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        assert_eq!(kids.len(), 1, "sh's one child, sleep: {kids:?}");
        let own = phys_footprint(pid).unwrap();
        let sleep = phys_footprint(kids[0]).unwrap();
        let tree = tree_footprint(pid).unwrap();
        // Both read at slightly different moments; the tree is at least the
        // shell plus most of the sleep.
        assert!(tree >= own + sleep / 2, "tree {tree}, sh {own}, sleep {sleep}");
        let _ = std::process::Command::new("kill").arg(kids[0].to_string()).status();
        let _ = sh.kill();
        let _ = sh.wait();
        assert!(!children_of(std::process::id()).contains(&pid));
    }

    #[test]
    fn vm_rss_is_read_in_kib() {
        let status = "Name:\tcat\nVmPeak:\t    8000 kB\nVmRSS:\t    1904 kB\nRssAnon:\t     88 kB\n";
        assert_eq!(vm_rss_bytes(status), Some(1904 * 1024));
        assert_eq!(vm_rss_bytes("Name:\tkthreadd\nState:\tS (sleeping)\n"), None);
        assert_eq!(vm_rss_bytes("VmRSS:\t  12 MB\n"), None);
        assert_eq!(vm_rss_bytes("VmRSS:\n"), None);
    }
}
