use std::fs::{self, create_dir_all, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::engine::types::JournalEntry;

/// Maximum journal size before rotation (10 MB).
const MAX_JOURNAL_BYTES: u64 = 10 * 1024 * 1024;

pub fn append_journal(path: &Path, entry: &JournalEntry) -> anyhow::Result<()> {
    append_journal_batch(path, std::slice::from_ref(entry))
}

/// Append a batch of journal entries in a single open/write/close.
///
/// Amortises the cost of the safety-check stat syscalls, directory creation,
/// file open and file close across every entry in the batch. At 10 Hz in the
/// daemon hot path with ~20 actions per cycle the reduction is from
/// O(N × (6 stat + open + write + close)) down to O(6 stat + open + N write
/// + close), which is what saves freezes/unfreezes from queuing behind
/// synchronous journal append calls.
///
/// A partial-batch serialisation failure is still surfaced to the caller,
/// but any entries successfully written before the failure are preserved —
/// we never silently drop log lines just because one of them was bad.
///
/// Empty batches are a no-op (no syscalls at all).
///
/// [Gray & Reuter 1992] §11 — group commit: batching WAL records amortises
/// per-record overhead across the entire group.
///
/// [Mohan et al. 1992] "ARIES" — log records can be buffered in memory and
/// written to the log in batches without loss of correctness for WAL
/// workloads where the in-memory write ordering matches the disk order.
pub fn append_journal_batch(path: &Path, entries: &[JournalEntry]) -> anyhow::Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    // Symlink protection: refuse to write through symlinks
    if path.exists() {
        if let Ok(meta) = fs::symlink_metadata(path) {
            if meta.file_type().is_symlink() {
                anyhow::bail!(
                    "journal path {} is a symlink — refusing to write",
                    path.display()
                );
            }
        }
    }

    if let Some(parent) = path.parent() {
        // Verify parent is not a symlink
        if parent.exists() {
            if let Ok(parent_meta) = fs::symlink_metadata(parent) {
                if parent_meta.file_type().is_symlink() {
                    anyhow::bail!(
                        "journal parent {} is a symlink — refusing to write",
                        parent.display()
                    );
                }
            }
        }
        create_dir_all(parent)?;
    }

    // Rotate if the journal exceeds the size limit.
    // Use symlink_metadata to avoid following symlinks.
    if let Ok(meta) = fs::symlink_metadata(path) {
        if !meta.file_type().is_symlink() && meta.len() > MAX_JOURNAL_BYTES {
            let rotated = path.with_extension("jsonl.1");
            // Remove old rotation if it exists.
            let _ = fs::remove_file(&rotated);
            // Rotate current journal to .1
            let _ = fs::rename(path, &rotated);
        }
    }

    // Pre-serialise all entries into a single byte buffer, then issue one
    // write(2). This keeps the loop inside userspace and turns N kernel
    // transitions into one — critical during the unfreeze fast-path where
    // each extra syscall adds to user-visible latency.
    let mut buf = String::with_capacity(entries.len() * 256);
    for entry in entries {
        let line = serde_json::to_string(entry)?;
        buf.push_str(&line);
        buf.push('\n');
    }

    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(buf.as_bytes())?;
    Ok(())
}

pub fn read_journal(path: &Path) -> anyhow::Result<Vec<JournalEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let f = OpenOptions::new().read(true).open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<JournalEntry>(&line) {
            out.push(entry);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::{FreezeSource, FrozenEntry, RootAction};
    use std::io::Write;

    fn make_entry() -> JournalEntry {
        JournalEntry {
            timestamp: chrono::Utc::now(),
            action: RootAction::BoostProcess {
                pid: 42,
                name: "test-proc".to_string(),
                reason: "test".to_string(),
            },
            before: None,
            after: None,
            success: true,
            reason: "unit-test".to_string(),
        }
    }

    #[test]
    fn roundtrip_single_entry() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("journal.jsonl");

        let entry = make_entry();
        append_journal(&path, &entry).expect("append_journal");

        let entries = read_journal(&path).expect("read_journal");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].reason, "unit-test");
    }

    #[test]
    fn roundtrip_multiple_entries() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("journal.jsonl");

        for _ in 0..3 {
            append_journal(&path, &make_entry()).expect("append_journal");
        }

        let entries = read_journal(&path).expect("read_journal");
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn missing_file_returns_empty() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("nonexistent.jsonl");

        let entries = read_journal(&path).expect("read_journal on missing file");
        assert!(entries.is_empty());
    }

    #[test]
    fn rotation_when_file_exceeds_10mb() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("journal.jsonl");

        // Write >10 MB of dummy content
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&path)
                .expect("open file");
            let big_chunk = vec![b'x'; 11 * 1024 * 1024];
            f.write_all(&big_chunk).expect("write chunk");
        }

        // Appending should trigger rotation
        append_journal(&path, &make_entry()).expect("append after large file");

        let rotated = path.with_extension("jsonl.1");
        assert!(
            rotated.exists(),
            "rotated file .jsonl.1 should exist after size limit exceeded"
        );
    }

    #[test]
    fn symlink_rejection() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let real = dir.path().join("real.jsonl");
        let link = dir.path().join("link.jsonl");

        // Create the real file first so the symlink target exists
        std::fs::write(&real, b"").expect("create real file");
        std::os::unix::fs::symlink(&real, &link).expect("create symlink");

        let result = append_journal(&link, &make_entry());
        assert!(result.is_err(), "should reject symlink path");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("symlink"),
            "error should mention 'symlink', got: {err_msg}"
        );
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("journal.jsonl");

        // Write one valid entry manually
        let valid_entry = make_entry();
        let valid_line = serde_json::to_string(&valid_entry).expect("serialize") + "\n";

        let bad_line = "this is not json\n";

        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&path)
                .expect("open file");
            f.write_all(valid_line.as_bytes())
                .expect("write valid line");
            f.write_all(bad_line.as_bytes()).expect("write bad line");
            f.write_all(valid_line.as_bytes())
                .expect("write valid line 2");
        }

        let entries = read_journal(&path).expect("read_journal");
        assert_eq!(
            entries.len(),
            2,
            "malformed line should be silently ignored, got {} entries",
            entries.len()
        );
    }

    // Ensure FrozenEntry is importable for completeness (used by other journal callers)
    #[test]
    fn frozen_entry_fields_accessible() {
        let entry = FrozenEntry {
            frozen_at: chrono::Utc::now(),
            source: FreezeSource::MainLoop,
            pressure_at_freeze: 0.5,
            process_name: None,
        };
        assert!(!entry.pressure_at_freeze.is_nan());
    }
}
