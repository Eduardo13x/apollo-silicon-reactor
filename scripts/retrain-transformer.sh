#!/bin/bash
# retrain-transformer.sh — Automatic Transformer retraining for Apollo.
#
# Called by launchd (com.eduardocortez.apollo-retrain.plist) nightly.
# Checks prerequisites, system idle state, and minimum data before training.
#
# Flow:
#   1. Verify Python + PyTorch available
#   2. Check system memory pressure (skip if under pressure)
#   3. Check minimum telemetry data accumulated
#   4. Train with warm-start (reuses previous weights)
#   5. Deploy ONNX model → daemon hot-reloads automatically
#
# All output goes to /var/lib/apollo/retrain.log (or /tmp/ for non-root).

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────

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

# ── Gate 1: Python + torch available ────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    log "ERROR: python3 not found, skipping retrain"
    exit 0
fi

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    log "ERROR: PyTorch not installed (pip install torch), skipping"
    exit 0
fi

# ── Gate 2: data directory exists ───────────────────────────────────────

if [ ! -d "$DATA_DIR" ]; then
    log "No telemetry directory yet ($DATA_DIR), skipping"
    exit 0
fi

N_FILES=$(find "$DATA_DIR" -name "*.bin" -type f 2>/dev/null | wc -l | tr -d ' ')
log "Found $N_FILES telemetry files in $DATA_DIR"

if [ "$N_FILES" -lt 200 ]; then
    log "Not enough data ($N_FILES < 200 files), skipping"
    exit 0
fi

# ── Gate 3: system memory pressure ──────────────────────────────────────

PRESSURE=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo "1")
if [ "$PRESSURE" -gt 1 ]; then
    log "System under memory pressure (level=$PRESSURE), skipping"
    exit 0
fi

# ── Gate 4: not already running ─────────────────────────────────────────

LOCK_FILE="${DEPLOY_DIR}/.retrain.lock"
if [ -f "$LOCK_FILE" ]; then
    # Check if the lock is stale (> 1 hour old).
    if [ "$(find "$LOCK_FILE" -mmin +60 2>/dev/null)" ]; then
        log "Removing stale lock file"
        rm -f "$LOCK_FILE"
    else
        log "Training already in progress (lock exists), skipping"
        exit 0
    fi
fi

# ── Train ───────────────────────────────────────────────────────────────

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log "Starting automatic training (warm-start)"

"$PYTHON" "$TRAIN_SCRIPT" \
    --auto \
    --data-dir "$DATA_DIR" \
    --deploy-dir "$DEPLOY_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -eq 0 ]; then
    log "Training complete, model deployed to $DEPLOY_DIR"
    log "Daemon will hot-reload on next hourly check"
else
    log "Training exited with code $EXIT_CODE"
fi
