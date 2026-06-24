#!/bin/bash
# apollo-liveness-watchdog — external liveness watchdog for apollo-optimizerd.
#
# ROOT-CAUSE FIX for the 2026-06-24 stall: the main cognitive cycle froze
# ~16.5 min (process alive, cycles frozen, metrics file stale) under extreme
# memory pressure. launchd's KeepAlive only restarts on CRASH — a hung-but-alive
# process slips through. This external probe (separate address space → immune to
# whatever froze the daemon: swap-eviction, lock, blocking syscall) detects the
# stall via the staleness of runtime_metrics.json and force-restarts the daemon.
#
# It is MECHANISM-AGNOSTIC: it recovers from any stall class, which is the right
# property since the precise cause (swap vs lock) was never confirmed.
#
# SAFETY — must NOT become a restart-storm aggressor during a sustained crisis
# (the exact scenario it exists for). Hard budget: at most MAX_RESTARTS in
# WINDOW_SEC; beyond that it stops restarting and only raises an alert flag, so a
# machine that needs a human is not hammered with restarts that worsen pressure.
#
# Invoked by launchd com.eduardocortez.apollo-liveness-watchdog every 60s.
# Reasoned recovery only — never edits daemon state.

set -u

METRICS="/var/lib/apollo/runtime_metrics.json"
LABEL="com.eduardocortez.systemoptimizerd"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
LOG="/var/lib/apollo/watchdog.log"
STATE_DIR="/var/run"
CONSEC_FILE="${STATE_DIR}/apollo-watchdog-consec"      # consecutive stale checks
RESTARTS_FILE="${STATE_DIR}/apollo-watchdog-restarts"  # "epoch epoch ..." recent restarts
ALERT_FLAG="${STATE_DIR}/apollo-watchdog-alert"        # presence = budget exhausted, needs human

STALE_SEC=180          # metrics file older than this = candidate stall (healthy writes ~15s)
CONSEC_NEEDED=2        # require 2 consecutive stale checks (~stall persisted >180s across 2 runs)
MAX_RESTARTS=2         # at most this many restarts ...
WINDOW_SEC=1800        # ... per 30 min, then alert-only

now=$(date +%s)

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG" 2>/dev/null; }

# ── Liveness signal: age of the metrics file the main loop writes (~every 15s).
if [ ! -f "$METRICS" ]; then
    log "metrics file missing ($METRICS) — daemon may be mid-restart; skipping"
    exit 0
fi
mtime=$(stat -f %m "$METRICS" 2>/dev/null || echo "$now")
age=$(( now - mtime ))

if [ "$age" -le "$STALE_SEC" ]; then
    # Healthy — reset the consecutive-stale counter.
    echo 0 > "$CONSEC_FILE" 2>/dev/null
    exit 0
fi

# Stale this check — bump the consecutive counter.
consec=$(cat "$CONSEC_FILE" 2>/dev/null || echo 0)
consec=$(( consec + 1 ))
echo "$consec" > "$CONSEC_FILE" 2>/dev/null
log "metrics stale: age=${age}s consec=${consec}/${CONSEC_NEEDED}"

[ "$consec" -lt "$CONSEC_NEEDED" ] && exit 0   # not confirmed yet, wait one more cycle

# ── Confirmed stall. Only act if the daemon is actually loaded+running (don't
# fight launchd if it is already restarting a crashed process).
if ! sudo launchctl print "system/${LABEL}" >/dev/null 2>&1; then
    log "daemon not loaded — leaving to launchd KeepAlive, not intervening"
    echo 0 > "$CONSEC_FILE" 2>/dev/null
    exit 0
fi

# ── Restart budget: prune restarts older than WINDOW_SEC, count what remains.
recent=""
count=0
if [ -f "$RESTARTS_FILE" ]; then
    for ts in $(cat "$RESTARTS_FILE" 2>/dev/null); do
        if [ $(( now - ts )) -lt "$WINDOW_SEC" ]; then
            recent="$recent $ts"; count=$(( count + 1 ))
        fi
    done
fi

if [ "$count" -ge "$MAX_RESTARTS" ]; then
    # Budget exhausted — restarting is not helping. Stop hammering; raise alert.
    log "BUDGET EXHAUSTED: ${count} restarts in last ${WINDOW_SEC}s — NOT restarting (would worsen a sustained crisis). Raising alert flag for human."
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apollo-optimizerd stalled (metrics age ${age}s) AND watchdog restart budget exhausted (${count} in ${WINDOW_SEC}s). Needs human: check memory pressure / swap. The main cycle is not advancing." > "$ALERT_FLAG" 2>/dev/null
    exit 1
fi

# ── Within budget: force-restart via clean bootout + bootstrap (handles the
# I/O-error-5 tombstone gotcha; bootout may say "No such process" — fine).
log "STALL CONFIRMED (age=${age}s) — restarting daemon (restart ${count}+1/${MAX_RESTARTS} in window)"
sudo launchctl bootout "system/${LABEL}" 2>>"$LOG"
sleep 2
if sudo launchctl bootstrap system "$PLIST" 2>>"$LOG"; then
    log "bootstrap OK — daemon restarted"
else
    log "bootstrap FAILED — will retry next cycle"
fi

echo "$recent $now" > "$RESTARTS_FILE" 2>/dev/null
echo 0 > "$CONSEC_FILE" 2>/dev/null
exit 0
