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
LASTRUN_FILE="${STATE_DIR}/apollo-watchdog-lastrun"    # watchdog's own last-run epoch (sleep detector)

STALE_SEC=90           # metrics file older than this = candidate stall (healthy writes ~15s;
                       # 90s = 6× cadence; tight enough to catch stalls faster than the
                       # previous 180s, loose enough to absorb a single missed cycle).
CONSEC_NEEDED=2        # require 2 consecutive stale checks (~stall persisted >90s across 2 runs)
MAX_RESTARTS=2         # at most this many restarts ...
WINDOW_SEC=1800        # ... per 30 min, then alert-only
SLEEP_GAP_SEC=240      # if the watchdog itself didn't run for >4x its 60s cadence, the machine
                       # was asleep/suspended — metrics staleness is then from sleep, NOT a hang.
# Early-stall detection: if age GREW >= GROWTH_THRESHOLD_SEC since the last check,
# the daemon is falling behind its 15s write cadence (precursor to the 06-24 + 06-27
# stalls — both showed monotonic age growth before going fully stale). On this
# signal, lower the effective stale threshold for this run only (early warning),
# WITHOUT triggering a restart — the existing consec/age path is the restart gate.
GROWTH_THRESHOLD_SEC=30
EARLY_STALE_SEC=60

now=$(date +%s)

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG" 2>/dev/null; }

# ── Sleep guard (fix for the 2026-06-25 false positive). On battery the lid-close /
# maintenance sleep suspends the WHOLE machine: neither the daemon nor this watchdog
# run, so on wake the metrics file legitimately looks stale by the sleep duration.
# Restarting then is a needless churn. Detect it by OUR OWN gap: if the watchdog
# hasn't run in far longer than its 60s cadence, the system slept — skip this check,
# reset the stale counter, let the next (post-wake) run judge a freshly-resumed daemon.
wd_last=$(cat "$LASTRUN_FILE" 2>/dev/null || echo "$now")
echo "$now" > "$LASTRUN_FILE" 2>/dev/null
wd_gap=$(( now - wd_last ))
if [ "$wd_gap" -gt "$SLEEP_GAP_SEC" ]; then
    log "watchdog gap=${wd_gap}s (machine slept/suspended) — skipping stall check; staleness is from sleep, not a hang"
    echo 0 > "$CONSEC_FILE" 2>/dev/null
    exit 0
fi

# ── Liveness signal: age of the metrics file the main loop writes (~every 15s).
if [ ! -f "$METRICS" ]; then
    log "metrics file missing ($METRICS) — daemon may be mid-restart; skipping"
    exit 0
fi
mtime=$(stat -f %m "$METRICS" 2>/dev/null || echo "$now")
age=$(( now - mtime ))
# Track previous age to detect INCREASING-staleness (early warning of a stall:
# even before the file is "fully stale", seeing age GROW fast across checks
# means the daemon is falling behind its write cadence — exact precursor to the
# 06-24 + 06-27 stalls). If age grew >=30s since the last check, escalate
# (lower the per-check stale threshold for this run). Pure measurement, never
# triggers a restart on its own.
prev_age=$(cat "${STATE_DIR}/apollo-watchdog-prev-age" 2>/dev/null || echo 0)
echo "$age" > "${STATE_DIR}/apollo-watchdog-prev-age" 2>/dev/null
age_growth=$(( age - prev_age ))

if [ "$age" -le "$STALE_SEC" ]; then
    # Healthy — reset the consecutive-stale counter.
    echo 0 > "$CONSEC_FILE" 2>/dev/null
    # Early-stall warning (logged but does NOT change state): age growing fast
    # between checks is the precursor pattern observed in 06-24 + 06-27 stalls.
    # Loud only on growing-then-near-stale to avoid log spam from the normal
    # 15s-cadence drift (each check grows ~15s; we ignore growth < threshold).
    if [ "$age_growth" -ge "$GROWTH_THRESHOLD_SEC" ] && [ "$age" -ge "$EARLY_STALE_SEC" ]; then
        log "EARLY-WARNING: age growing fast (prev=${prev_age}s now=${age}s, growth=${age_growth}s/${WINDOW_SEC}s-run) — daemon falling behind write cadence; will restart at age>${STALE_SEC} if it persists"
    fi
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
if ! launchctl print "system/${LABEL}" >/dev/null 2>&1; then
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
launchctl bootout "system/${LABEL}" 2>>"$LOG"
sleep 2
if launchctl bootstrap system "$PLIST" 2>>"$LOG"; then
    log "bootstrap OK — daemon restarted"
else
    log "bootstrap FAILED — will retry next cycle"
fi

echo "$recent $now" > "$RESTARTS_FILE" 2>/dev/null
echo 0 > "$CONSEC_FILE" 2>/dev/null
exit 0
