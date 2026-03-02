use std::fs::{create_dir_all, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::engine::types::JournalEntry;

pub fn append_journal(path: &Path, entry: &JournalEntry) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
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
