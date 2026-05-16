#!/usr/bin/env python3
"""notebook-memory-pull.py — Detect MEMORY.md entries that duplicate NotebookLM sources.

Why this exists
---------------
MEMORY.md is the curated, always-loaded Tier 0 memory. It has a 200-line
limit (lines beyond are truncated by the harness). The NotebookLM
notebook is the deeper, on-demand corpus (~80 sources, hundreds of
commits). They drift apart over time: an entry in MEMORY.md may already
exist verbatim as a notebook source, in which case MEMORY.md is paying
context budget for redundant information.

This script does NOT delete anything. It prints candidates for the
human to review. Per NotebookLM verdict (2026-05-16): trust the metric
when it disagrees with NotebookLM — same principle here, the human is
the gatekeeper.

Usage
-----
    ./scripts/notebook-memory-pull.py
    ./scripts/notebook-memory-pull.py --threshold 0.6   # looser match

Outputs a TSV-ish list:
    OVERLAP_PCT  MEMORY_ENTRY              SUGGESTED_NOTEBOOK_SOURCE
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MEMORY_PATH = Path.home() / ".claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/MEMORY.md"
NOTEBOOK_ID_DEFAULT = "8344b94c-a014-4803-abea-076a55753cfd"

ENTRY_LINE_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)\s*[—-]\s*(.*)$")


def parse_memory_entries(path: Path) -> list[tuple[str, str, str]]:
    """Returns [(title, file, hook), ...] for each index entry in MEMORY.md."""
    out: list[tuple[str, str, str]] = []
    if not path.exists():
        print(f"[err] MEMORY.md not found at {path}", file=sys.stderr)
        return out
    for line in path.read_text().splitlines():
        m = ENTRY_LINE_RE.match(line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return out


def fetch_notebook_titles(notebook_id: str) -> list[str]:
    """Calls `nlm source list` and extracts titles. Returns [] on failure."""
    try:
        proc = subprocess.run(
            ["nlm", "source", "list", notebook_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[err] nlm unavailable: {e}", file=sys.stderr)
        return []
    if proc.returncode != 0:
        print(f"[err] nlm source list failed: {proc.stderr[:200]}", file=sys.stderr)
        return []
    # nlm output is tabular; pull anything that looks like a title.
    # Be lenient — we just need a corpus of strings to compare against.
    titles: list[str] = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("-") or s.startswith("="):
            continue
        # heuristic: a title likely has alpha chars + space.
        if " " in s and any(c.isalpha() for c in s):
            titles.append(s)
    return titles


def normalize(s: str) -> set[str]:
    """Lowercase token set, drop stopwords."""
    stop = {"the", "and", "a", "an", "of", "for", "in", "on", "to", "with"}
    toks = re.findall(r"[a-z0-9]{3,}", s.lower())
    return {t for t in toks if t not in stop}


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="overlap ratio above which an entry is flagged (default 0.45)")
    ap.add_argument("--notebook-id", default=NOTEBOOK_ID_DEFAULT)
    args = ap.parse_args()

    entries = parse_memory_entries(MEMORY_PATH)
    if not entries:
        print("[info] no MEMORY.md entries parsed", file=sys.stderr)
        return 1

    titles = fetch_notebook_titles(args.notebook_id)
    if not titles:
        print("[info] no notebook titles fetched (will print everything as orphan)", file=sys.stderr)

    print("OVERLAP\tMEMORY_TITLE\tSUGGESTED_NOTEBOOK_SOURCE")
    flagged = 0
    for title, _file, hook in entries:
        mem_tokens = normalize(title + " " + hook)
        best_score = 0.0
        best_match = ""
        for t in titles:
            s = overlap(mem_tokens, normalize(t))
            if s > best_score:
                best_score, best_match = s, t
        if best_score >= args.threshold:
            flagged += 1
            print(f"{best_score:.2f}\t{title[:50]}\t{best_match[:70]}")

    print(f"\n[summary] {flagged}/{len(entries)} entries overlap ≥ {args.threshold:.2f}",
          file=sys.stderr)
    print("[next-step] review flagged rows; if confident, move detail to notebook",
          file=sys.stderr)
    print("[next-step] and prune MEMORY.md entry to a one-liner pointer.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
