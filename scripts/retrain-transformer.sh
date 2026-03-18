#!/bin/bash
# retrain-transformer.sh — Automatic Transformer retraining for Apollo.
#
# Called by launchd every 6 hours.  The Python script handles ALL smart
# decisions (data maturity, quality gates, warm-start, etc.).  This shell
# script only verifies prerequisites (Python, PyTorch, data directory).
#
# Quality gates handled by train_transformer.py --auto:
#   - Immature data (< 7 days) → skip, keep collecting
#   - Low feature diversity → skip, need varied workloads
#   - Low hour coverage → skip, need more of the daily cycle
#   - System under memory pressure → skip, try next interval
#   - Less than 20% new data since last training → skip
#   - Less than 12 hours since last training → skip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_transformer.py"

if [ "$(id -u)" -eq 0 ]; then
    DATA_DIR="/var/lib/apollo/telemetry"
    DEPLOY_DIR="/var/lib/apollo"
    LOG_FILE="/var/lib/apollo/retrain.log"
else
    DATA_DIR="/tmp/apollo-telemetry"
    DEPLOY_DIR="/tmp"
    LOG_FILE="/tmp/apollo-retrain.log"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── Prerequisite: Python ────────────────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    log "python3 not found, skipping"
    exit 0
fi

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    log "PyTorch not installed (pip install torch), skipping"
    exit 0
fi

# ── Prerequisite: data directory ────────────────────────────────────────

if [ ! -d "$DATA_DIR" ]; then
    log "No telemetry directory ($DATA_DIR), skipping"
    exit 0
fi

# ── Lock (prevent concurrent runs) ─────────────────────────────────────

LOCK_FILE="${DEPLOY_DIR}/.retrain.lock"
if [ -f "$LOCK_FILE" ]; then
    if [ "$(find "$LOCK_FILE" -mmin +60 2>/dev/null)" ]; then
        rm -f "$LOCK_FILE"
    else
        log "Already running (lock exists), skipping"
        exit 0
    fi
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ── Run training (all smart decisions are in the Python script) ─────────

log "Invoking train_transformer.py --auto"

"$PYTHON" "$TRAIN_SCRIPT" \
    --auto \
    --data-dir "$DATA_DIR" \
    --deploy-dir "$DEPLOY_DIR" \
    2>&1 | tee -a "$LOG_FILE"

log "Done (exit code: ${PIPESTATUS[0]})"
