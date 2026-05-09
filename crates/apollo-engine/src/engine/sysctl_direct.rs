//! Direct sysctl access via `libc::sysctlbyname` — no subprocess.
//!
//! Replaces ALL `Command::new("/usr/sbin/sysctl")` calls across the codebase.
//! Each call takes <1µs vs 5-10ms for spawning a subprocess.

use std::ffi::CString;

/// Read a sysctl value as a trimmed String.
pub fn read_str(key: &str) -> Option<String> {
    let ckey = CString::new(key).ok()?;
    let mut size: libc::size_t = 0;

    // First call: get required buffer size.
    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    if size == 0 {
        return None;
    }

    let mut buf = vec![0u8; size];
    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    // Truncate to actual returned size.
    buf.truncate(size);
    let raw = buf.clone(); // Preserve original bytes for binary fallback.

    // Try UTF-8 string first (string-valued sysctls like kern.ostype).
    // Trim null terminators.
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
    if let Ok(s) = String::from_utf8(buf) {
        let trimmed = s.trim().to_string();
        if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            return Some(trimmed);
        }
    }

    // Binary integer: common sizes are 4 (i32) and 8 (i64).
    // Convert to decimal string to match `sysctl -n` output.
    if raw.len() >= 4 && size == 4 {
        let val = i32::from_ne_bytes(raw[..4].try_into().ok()?);
        return Some(val.to_string());
    }
    if raw.len() >= 8 && size == 8 {
        let val = i64::from_ne_bytes(raw[..8].try_into().ok()?);
        return Some(val.to_string());
    }

    None
}

/// Read a sysctl value as u64 (native size).
pub fn read_u64(key: &str) -> Option<u64> {
    let ckey = CString::new(key).ok()?;
    let mut val: u64 = 0;
    let mut size = std::mem::size_of::<u64>();

    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            &mut val as *mut _ as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    Some(val)
}

/// Read a sysctl value as i32.
pub fn read_i32(key: &str) -> Option<i32> {
    let ckey = CString::new(key).ok()?;
    let mut val: i32 = 0;
    let mut size = std::mem::size_of::<i32>();

    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            &mut val as *mut _ as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    Some(val)
}

/// Read a sysctl value as u32.
pub fn read_u32_val(key: &str) -> Option<u32> {
    read_i32(key).map(|v| v as u32)
}

/// Write an i32 value to a sysctl key. Returns true on success.
pub fn write_i32(key: &str, value: i32) -> bool {
    let ckey = match CString::new(key) {
        Ok(k) => k,
        Err(_) => return false,
    };

    unsafe {
        libc::sysctlbyname(
            ckey.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &value as *const i32 as *mut _,
            std::mem::size_of::<i32>(),
        ) == 0
    }
}

/// Write a string value to a sysctl key, parsing as i32 if possible.
/// This mirrors `sysctl -w key=value` behavior.
pub fn write_str_value(key: &str, value: &str) -> bool {
    // Try parsing as integer first (most common case).
    if let Ok(int_val) = value.parse::<i32>() {
        return write_i32(key, int_val);
    }
    // Try as i64.
    if let Ok(int_val) = value.parse::<i64>() {
        let ckey = match CString::new(key) {
            Ok(k) => k,
            Err(_) => return false,
        };
        return unsafe {
            libc::sysctlbyname(
                ckey.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &int_val as *const i64 as *mut _,
                std::mem::size_of::<i64>(),
            ) == 0
        };
    }
    // Raw string write.
    let ckey = match CString::new(key) {
        Ok(k) => k,
        Err(_) => return false,
    };
    unsafe {
        libc::sysctlbyname(
            ckey.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            value.as_ptr() as *mut _,
            value.len(),
        ) == 0
    }
}

/// Read raw bytes from a sysctl key. Used for struct-valued sysctls
/// like `net.inet.tcp.stats`.
pub fn read_raw(key: &str, buf: &mut [u8]) -> Option<usize> {
    let ckey = CString::new(key).ok()?;
    let mut size = buf.len();

    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    Some(size)
}

/// Check if a sysctl key exists (readable without error).
pub fn exists(key: &str) -> bool {
    let ckey = match CString::new(key) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let mut size: libc::size_t = 0;
    unsafe {
        libc::sysctlbyname(
            ckey.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) == 0
    }
}

/// Formatted swap usage via `vm.swapusage` sysctl.
/// Returns (total_bytes, used_bytes) parsed from the xsw_usage struct.
pub fn read_swap_usage() -> Option<(u64, u64)> {
    // struct xsw_usage { u64 total, u64 avail, u64 used, u32 pagesize, i32 encrypted }
    #[repr(C)]
    #[allow(dead_code)]
    struct XswUsage {
        xsu_total: u64,
        xsu_avail: u64,
        xsu_used: u64,
        xsu_pagesize: u32,
        xsu_encrypted: i32,
    }

    let ckey = CString::new("vm.swapusage").ok()?;
    // First query the actual size to ensure compatibility.
    let mut size: libc::size_t = 0;
    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    // The struct must be at least 24 bytes (3 × u64) for us to read total/used.
    if size < 24 {
        return None;
    }

    let mut buf = vec![0u8; size];
    unsafe {
        if libc::sysctlbyname(
            ckey.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    // Read the first three u64 fields (total, avail, used) at known offsets.
    let total = u64::from_ne_bytes(buf[0..8].try_into().ok()?);
    let used = u64::from_ne_bytes(buf[16..24].try_into().ok()?);

    Some((total, used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_kern_ostype() {
        let val = read_str("kern.ostype");
        assert_eq!(val.as_deref(), Some("Darwin"));
    }

    #[test]
    fn read_hw_ncpu() {
        let val = read_u64("hw.ncpu");
        assert!(val.is_some());
        assert!(val.unwrap() >= 1);
    }

    #[test]
    fn read_i32_pagesize() {
        let val = read_i32("hw.pagesize");
        assert!(val.is_some());
        let ps = val.unwrap();
        assert!(ps == 4096 || ps == 16384, "unexpected page size: {}", ps);
    }

    #[test]
    fn exists_returns_true_for_known_key() {
        assert!(exists("kern.ostype"));
    }

    #[test]
    fn exists_returns_false_for_bogus_key() {
        assert!(!exists("kern.this_key_does_not_exist_12345"));
    }

    #[test]
    fn read_nonexistent_returns_none() {
        assert!(read_str("kern.bogus_nonexistent_key_xyz").is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn swap_usage_readable() {
        let result = read_swap_usage();
        // Should succeed on any macOS system.
        assert!(result.is_some());
        let (total, used) = result.unwrap();
        assert!(used <= total || total == 0);
    }

    #[test]
    fn write_str_value_parse() {
        // Verify parsing logic without actually writing (would need root).
        assert!("42".parse::<i32>().is_ok());
        assert!("not_a_number".parse::<i32>().is_err());
    }
}
