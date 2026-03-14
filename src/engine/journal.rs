use std::fs::{self, create_dir_all, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::engine::types::JournalEntry;

/// Maximum journal size before rotation (10 MB).
const MAX_JOURNAL_BYTES: u64 = 10 * 1024 * 1024;

pub fn append_journal(path: &Path, entry: &JournalEntry) -> anyhow::Result<()> {
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

    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(entry)?;
    writeln!(f, "{}", line)?;
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
