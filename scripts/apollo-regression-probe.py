#!/usr/bin/env python3
"""apollo-regression-probe — PRODUCTION behavioral regression probe for
apollo-silicon-reactor (macOS optimization daemon, M1 8GB).

WHAT IT IS
==========
A cheap, deterministic, READ-ONLY probe that reads live production state
(runtime_metrics.json + journal.jsonl + learned_state.json + llm_state.json +
live sysctls) and detects the runtime/journal/sysctl SIGNATURES of the
regressions fixed THIS SESSION. It NEVER edits state — detection only; the fix
is reasoned on-demand (supervision doctrine: the human is the gatekeeper).

It is COMPLEMENTARY to scripts/apollo-learned-state-audit.py, which already
covers the learned_state-pathology side (boost-loop-risk via weights,
debias-saturated, humble-latched, restore-quality, weight-futile, ais-degraded,
daemon-failures, llm-stale, purge-strangled). THIS probe focuses on the
runtime+journal+sysctl BEHAVIORAL signatures and, crucially, a CYCLE-OVER-CYCLE
TREND LOG so p95 / thrashing / refault / ais can be watched evolving over time.

SCARS GUARDED (each detector guards one regression fixed this session)
=====================================================================
- purge-strangled        f770478 / 44266d8 / c041d88
    suppression strangling memory relief. thrashing>=50k AND purge skip-ratio
    >0.9 AND skipped>1000. (2026-06-15: purge suppressed 127,959x vs 147 fired,
    thrashing hit 69k, AIS read 93 — invisible to AIS.)
- cycle-inflation        22eb524
    NARS belief store grew to 22,113 (cap 3000) -> learned_state.json 5.7MB ->
    p95 climbs -> daemon degrades ~100k cycles. beliefs>3000 OR ls>3MB OR
    p95 climbing vs trend baseline.
- refault-storm          667f7d1 / dcba17e
    refault_peak_per_sec storms (150k-476k pages/s = 2.5-7.4 GB/s) -> microstutter.
    Tracks storm frequency/magnitude over the trend log.
- meet-network-tamper    da7df34 / 5802c89
    any of the 4 TCP/IPC sysctls != macOS default (131072,131072,3,128). Apollo
    now hard-blocks writing these; non-default = Apollo touched the network and
    breaks Meet/WebRTC audio. ZERO tolerance.
- meet-scheduler/boost   4b0753e / 4ae0f27
    a single process name dominates BoostProcess in the recent journal (node was
    boosted 54x), OR boost churn on one pid.
- llm-stale              d6c659b / 4b6da93 / 3df9235
    teacher enabled but last SUCCESSFUL call >2d, OR consecutive_failures>=10,
    OR last_error indicates metal-oom/parse failure.
- complete-mediation     03472d7 / 6e0e1ce / a98b33a
    a hard-protected/Apple/dev-runtime process is the TARGET of
    FreezeProcess/ThrottleProcess/SetMemorystatus in the journal.
- performance-baseline   (general)
    ais_score<80, failures>0, last_error set.

NOTE ON PRODUCTION SHAPE (verified against live /var/lib/apollo state)
=====================================================================
- runtime_metrics.json has NO `cycle_count` field (it is absent / None) —
  handled gracefully.
- journal SetMemorystatus entries carry NO `name` field (only pid/priority/
  reason/decision_reason). The target process name is embedded in `reason`
  (e.g. "zombie-hunter MemoryHoarder: Brave Browser (287MB) ..."), so the
  complete-mediation detector extracts the name from `reason` for that variant.
- BoostProcess / ThrottleProcess / FreezeProcess DO carry `name`.

HOW TO RUN
==========
  One-shot (human):        sudo ./scripts/apollo-regression-probe.py
  Quiet (cron):            sudo ./scripts/apollo-regression-probe.py --quiet
  Watch the trend table:   sudo ./scripts/apollo-regression-probe.py --trend 20

  Cron (root LaunchDaemon, like the audit cron — every ~30 min for trend
  resolution):  StartInterval 1800, RunAtLoad, exec this script with --quiet.

EXIT CODES:  0 = clean, 1 = findings present, 2 = at least one HIGH.

This probe is READ-ONLY. It writes ONLY its own log, trend log, and flag file;
it never touches daemon state.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
STATE_DIR = "/var/lib/apollo"
METRICS = STATE_DIR + "/runtime_metrics.json"
JOURNAL = STATE_DIR + "/journal.jsonl"
LEARNED_STATE = STATE_DIR + "/learned_state.json"
LLM = STATE_DIR + "/llm_state.json"

# Trend log = the cycle-over-cycle metric history (one JSON line per probe run).
TREND = STATE_DIR + "/regression-metrics.jsonl"
LOG = STATE_DIR + "/regression.log"
FLAG = "/var/run/apollo-regression-alert"  # presence = HIGH finding pending

# How many journal lines to look back over for behavioral signatures.
JOURNAL_TAIL_LINES = 500
# How many trend rows to read back for cycle-over-cycle comparison.
TREND_TAIL_ROWS = 30

# macOS TCP/IPC sysctl defaults. Apollo hard-blocks writing these post-da7df34.
SYSCTL_KEYS = [
    "net.inet.tcp.sendspace",
    "net.inet.tcp.recvspace",
    "net.inet.tcp.delayed_ack",
    "kern.ipc.somaxconn",
]
SYSCTL_DEFAULTS = {
    "net.inet.tcp.sendspace": 131072,
    "net.inet.tcp.recvspace": 131072,
    "net.inet.tcp.delayed_ack": 3,
    "kern.ipc.somaxconn": 128,
}

# Hard-protected / Apple / dev-runtime process names that must NEVER be the
# target of FreezeProcess / ThrottleProcess / SetMemorystatus. Brave/Chromium
# are freeze-exempt (freeze breaks their IPC contract — see safety.rs), but a
# THROTTLE or jetsam demote on them is still a mediation breach.
PROTECTED_NAMES = [
    "kernel_task", "launchd", "WindowServer", "Finder",
    "coreaudiod", "CVMServer", "powerd", "sharingd", "configd",
    "language_server", "Brave", "Chromium", "node",
]


# ── Loaders (never crash) ─────────────────────────────────────────────────────
def load(path):
    """Load a JSON file. Returns {} on missing/unreadable, {'__error__': ..} on
    a parse error. Never raises."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {"__error__": str(e)}


def read_journal_tail(path, n):
    """Read the last n parseable JSON lines of a .jsonl file, efficiently
    (seek from the end so we don't slurp a multi-MB journal). Returns a list of
    dicts in chronological order. Never raises."""
    try:
        size = os.path.getsize(path)
    except Exception:
        return []
    if size == 0:
        return []
    # Read a tail chunk sized generously for n lines; grow if we came up short.
    chunk = min(size, max(65536, n * 1024))
    try:
        with open(path, "rb") as f:
            while True:
                f.seek(size - chunk)
                data = f.read()
                # Drop a possibly-partial first line unless we're at file start.
                blob = data if chunk >= size else data.split(b"\n", 1)[-1]
                lines = blob.split(b"\n")
                if len(lines) > n or chunk >= size:
                    break
                chunk = min(size, chunk * 2)
    except Exception:
        return []
    out = []
    for raw in lines[-n:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            continue  # skip malformed
    return out


def fresh_entries(journal_tail, minutes):
    """Filter journal entries to those within the last `minutes`, relative to
    wall-clock UTC now. Journal-based detectors MUST use this — the daemon
    restarts on every deploy, so the raw tail spans PRE-FIX behavior. Counting
    a fixed regression's residue (e.g. node boosted 89x BEFORE the foreground
    gate landed) is a false positive. Lesson: fresh-only timestamp filter
    (supervision doctrine). Entries with an unparseable/absent timestamp are
    DROPPED (conservative: only count what we can date as recent)."""
    if not journal_tail:
        return []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)
    out = []
    for e in journal_tail:
        ts = e.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= cutoff:
            out.append(e)
    return out


# Journal-based detectors count over this rolling window (matches the 30-min
# cron cadence): a live loop produces many actions inside it; stale pre-fix
# residue ages out.
FRESH_WINDOW_MIN = 30


def teacher_config_disabled():
    """True iff the [llm] config explicitly sets enabled=false. Stdlib line
    parse (no tomllib on python 3.9). Conservative: returns False on any read
    failure so a genuinely-dead teacher is never masked."""
    try:
        with open("/etc/apollo-optimizer/config.toml") as f:
            in_llm = False
            for line in f:
                s = line.strip()
                if s.startswith("#"):
                    continue
                if s.startswith("[") and s.endswith("]"):
                    in_llm = s == "[llm]"
                    continue
                if in_llm and s.replace(" ", "").startswith("enabled="):
                    return "false" in s.lower()
    except Exception:
        pass
    return False


def read_sysctls():
    """Read the 4 critical sysctls via subprocess. Returns a dict
    {key: int-value}; a key maps to None on read/parse failure. Never raises."""
    out = {}
    for key in SYSCTL_KEYS:
        val = None
        try:
            r = subprocess.run(
                ["sysctl", "-n", key],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=2,
            )
            if r.returncode == 0:
                txt = r.stdout.decode("utf-8", "replace").strip()
                try:
                    val = int(txt)
                except (ValueError, TypeError):
                    val = None
        except Exception:
            val = None
        out[key] = val
    return out


def learned_state_stats(path):
    """Return (size_mb, beliefs_count). size_mb from the file on disk; beliefs
    by walking outcome_tracker.drift_detector.beliefs (a dict). Either may be
    None on failure. Avoids parsing the whole file twice by reusing load()."""
    size_mb = None
    try:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    except Exception:
        size_mb = None
    beliefs = None
    ls = load(path)
    if isinstance(ls, dict) and "__error__" not in ls:
        try:
            b = ls.get("outcome_tracker", {}).get("drift_detector", {}).get("beliefs", {})
            if isinstance(b, dict):
                beliefs = len(b)
        except Exception:
            beliefs = None
    return size_mb, beliefs


def read_trend_tail(path, k):
    """Read the last k rows from the trend JSONL (each a metrics dict).
    Returns a list of dicts in chronological order. Never raises."""
    rows = read_journal_tail(path, k)
    return [r for r in rows if isinstance(r, dict)]


# ── Journal helpers ───────────────────────────────────────────────────────────
def _action_kind_and_payload(entry):
    """Return (kind, payload_dict) for a journal entry, or (None, None)."""
    if not isinstance(entry, dict):
        return None, None
    action = entry.get("action")
    if not isinstance(action, dict) or not action:
        return None, None
    # Each entry is {action: {<Kind>: {..}}}; take the first (only) key.
    for kind, payload in action.items():
        if isinstance(payload, dict):
            return kind, payload
        return kind, {}
    return None, None


def _name_from_payload(kind, payload):
    """Resolve the target process name for an action payload. BoostProcess /
    ThrottleProcess / FreezeProcess carry `name`. SetMemorystatus does NOT — its
    target name is embedded in `reason` ('... MemoryHoarder: <name> (NNNmb) ...'),
    so we parse it out. Returns a name string or None."""
    name = payload.get("name")
    if name:
        return name
    if kind == "SetMemorystatus":
        reason = payload.get("reason", "")
        if isinstance(reason, str) and "MemoryHoarder:" in reason:
            tail = reason.split("MemoryHoarder:", 1)[1].strip()
            # Strip a trailing " (NNNmb)" size annotation if present.
            cut = tail.find(" (")
            if cut > 0:
                tail = tail[:cut]
            return tail.strip() or None
    return None


# ── Detectors (common signature) ──────────────────────────────────────────────
# Every detector: detect_x(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend)
#   rt           : runtime_metrics.json dict (may be {} or {'__error__':..})
#   journal_tail : list of recent journal entry dicts
#   llm          : llm_state.json dict
#   ls_size_mb   : float | None  (learned_state.json size in MB)
#   ls_beliefs   : int   | None  (drift_detector.beliefs count)
#   sysctls      : dict {key: int|None}
#   trend        : list of prior metric dicts (chronological), for trend logic
# Returns: list of (severity, code, detail) tuples. Never raises.

def detect_purge_strangled(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """f770478 / 44266d8 / c041d88 — suppression strangling memory relief.

    Signature requires ALL THREE (prevents false positives on transient storms
    or idle-box purge suppression alone):
      1. thrashing_score >= 50000 (crisis level; 2026-06-15 reached 69k)
      2. skip_ratio = skipped/(skipped+total) > 0.9 (relief strangled; 99.9%)
      3. skipped > 1000 (sustained cumulative, not startup noise)
    Two of three -> MED (drift, investigate before it escalates).
    """
    findings = []
    if not isinstance(rt, dict) or "__error__" in rt:
        return findings

    thrash = rt.get("thrashing_score", 0.0)
    if not isinstance(thrash, (int, float)):
        thrash = 0.0
    skipped = rt.get("maintenance_purge_skipped_idle_total", 0)
    if not isinstance(skipped, int):
        skipped = 0
    purged = rt.get("maintenance_purge_total", 0)
    if not isinstance(purged, int):
        purged = 0
    total = skipped + purged
    skip_ratio = skipped / total if total > 0 else 0.0

    thrash_crisis = thrash >= 50_000.0
    thrash_elevated = thrash >= 30_000.0  # approaching crisis
    ratio_strangled = skip_ratio > 0.9
    count_sustained = skipped > 1_000

    if thrash_crisis and ratio_strangled and count_sustained:
        findings.append((
            "HIGH", "purge-strangled",
            "thrashing_score={:.0f} (crisis) AND purge suppressed {}/{} "
            "({:.1%} skip) AND sustained ({} skips) — relief strangulation; "
            "check feedback_suppression_survival_first gate and "
            "is_high_bw_workload_active false-negative (2026-06-15 shape)".format(
                thrash, skipped, total, skip_ratio, skipped)))
    elif thrash_elevated and (ratio_strangled or count_sustained):
        # Thrashing is MANDATORY for the partial: high skip + LOW thrashing is
        # benign idle (nothing to purge), not strangulation. Diagnosed
        # 2026-06-20 by the loop — skip 99.9% at thrashing=172 was healthy and
        # over-fired here. Now the partial requires thrashing genuinely climbing
        # (>=30k) WHILE relief is suppressed — the real pre-regression drift.
        sig = ["thrashing={:.0f}".format(thrash)]
        if ratio_strangled:
            sig.append("skip_ratio={:.1%}".format(skip_ratio))
        if count_sustained:
            sig.append("sustained_skips={}".format(skipped))
        findings.append((
            "MED", "purge-strangled-partial",
            "thrashing climbing ({:.0f}) WHILE relief suppressed: {} — monitor; "
            "may precede the full strangulation regression".format(
                thrash, ", ".join(sig))))
    return findings


def detect_cycle_inflation(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """22eb524 — belief store bloat -> learned_state.json size climb -> p95 degrade.

    HIGH: beliefs > 3000 (cap broken), OR ls_size_mb > 5.0 (regression magnitude),
          OR p95_cycle_ms > 100 (severe overhead; regression was 135-140ms).
    MED:  ls_size_mb > 3.0 (2x healthy ~1.4MB), OR p95 climbing > baseline+20ms
          vs the trend-log median (sustained, not a single spike).
    """
    findings = []
    if not isinstance(rt, dict):
        rt = {}
    if not isinstance(ls_beliefs, int) or ls_beliefs < 0:
        ls_beliefs = None
    if ls_size_mb is not None and not isinstance(ls_size_mb, (int, float)):
        ls_size_mb = None
    if not isinstance(trend, list):
        trend = []

    MAX_BELIEFS = 3000
    if ls_beliefs is not None and ls_beliefs > MAX_BELIEFS:
        findings.append((
            "HIGH", "belief-cap-broken",
            "beliefs_count={} exceeds cap {} — unbounded growth (22eb524). "
            "Is enforce_capacity() wired into observe_salient, or was the cap "
            "raised? Regression reached 22,113 (7.4x).".format(ls_beliefs, MAX_BELIEFS)))

    if ls_size_mb is not None:
        if ls_size_mb > 5.0:
            findings.append((
                "HIGH", "ls-size-critical",
                "learned_state.json={:.1f} MB (>5 MB) — regression magnitude "
                "reached (22eb524 was 5.7 MB); each persist re-serializes "
                "multi-MB JSON, p95 climbs. Check beliefs_count.".format(ls_size_mb)))
        elif ls_size_mb > 3.0:
            findings.append((
                "MED", "ls-size-elevated",
                "learned_state.json={:.1f} MB (>3 MB, healthy ~1.4 MB) — "
                "approaching regression magnitude; monitor beliefs_count and "
                "p95 trend.".format(ls_size_mb)))

    p95 = rt.get("p95_cycle_ms")
    if isinstance(p95, (int, float)) and p95 > 0:
        # The 22eb524 regression is BLOAT-driven (beliefs/ls_size); p95 is only
        # the symptom. A high p95 WITHOUT bloat is transient scheduler load (the
        # daemon competing for CPU under a heavy workload), NOT the regression —
        # so only escalate to HIGH when a bloat signal co-occurs. (Diagnosed
        # 2026-06-20: a lone p95=153ms fired HIGH while beliefs=3000/ls=1.4MB
        # were healthy and the box was at load 5.9 — a false positive.)
        bloat = (isinstance(ls_beliefs, int) and ls_beliefs > MAX_BELIEFS) or (
            isinstance(ls_size_mb, (int, float)) and ls_size_mb > 3.0
        )
        if p95 > 100.0 and bloat:
            findings.append((
                "HIGH", "p95-crisis-bloat",
                "p95_cycle_ms={:.1f} ms (>100) WITH state bloat — the 22eb524 "
                "inflation shape (was 135-140ms). Fix the bloat, not p95.".format(p95)))
        else:
            # No bloat: p95 alone is at most MED. Prefer a sustained trend climb;
            # fall back to a very-high single reading flagged as likely-transient.
            samples = []
            for row in trend[-10:]:
                ep = row.get("p95_cycle_ms")
                if isinstance(ep, (int, float)) and ep > 0:
                    samples.append(ep)
            flagged = False
            if len(samples) >= 3:
                samples.sort()
                baseline = samples[len(samples) // 2]
                if p95 > baseline + 20.0:
                    flagged = True
                    findings.append((
                        "MED", "p95-climbing",
                        "p95_cycle_ms={:.1f} ms vs trend baseline {:.1f} ms "
                        "(+{:.1f}) — sustained climb with no state bloat; watch "
                        "for it persisting (vs transient load).".format(
                            p95, baseline, p95 - baseline)))
            if not flagged and p95 > 150.0:
                findings.append((
                    "MED", "p95-high-no-bloat",
                    "p95_cycle_ms={:.1f} ms (>150) but no state bloat — likely "
                    "transient CPU contention, NOT 22eb524. Confirm the load is "
                    "not the daemon itself before acting.".format(p95)))
    return findings


def detect_refault_storm(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """667f7d1 / dcba17e — refault microstutter storms.

    Healthy: refault_delta ~2-8k pages/s, peak <100k. Storms: 150k-476k
    (2.5-7.4 GB/s).
    HIGH: delta >= 100k AND peak >= 150k (storm active, page-in/decompress stall).
    MED:  delta sustained >= 100k across >=3 consecutive trend rows (peak may be
          recovering), OR peak >= 150k while delta recovering but still >40k.
    LOW:  single high-delta spike (<3 consecutive) — typical large app switch.
    """
    findings = []
    if not isinstance(rt, dict) or "__error__" in rt:
        return findings

    def _num(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return 0.0

    delta = _num(rt.get("refault_delta_per_sec", 0.0))
    peak = _num(rt.get("refault_peak_per_sec", 0.0))

    CRISIS_DELTA = 100_000.0
    CRISIS_PEAK = 150_000.0
    SUSTAINED = 3

    # Consecutive high-delta count = tail of trend rows at/above CRISIS_DELTA,
    # plus the current sample if it is also high.
    consecutive = 0
    if isinstance(trend, list):
        for row in reversed(trend):
            if isinstance(row, dict) and _num(row.get("refault_delta_per_sec", 0)) >= CRISIS_DELTA:
                consecutive += 1
            else:
                break
    if delta >= CRISIS_DELTA:
        consecutive += 1
    else:
        consecutive = 0

    if delta >= CRISIS_DELTA and peak >= CRISIS_PEAK:
        findings.append((
            "HIGH", "refault-storm-active",
            "refault microstutter ACTIVE: delta={:.0f} pages/s (>crisis {:.0f}), "
            "peak={:.0f} (>session crisis {:.0f}) — major page-in/decompress "
            "stall under workload.".format(delta, CRISIS_DELTA, peak, CRISIS_PEAK)))
    elif delta >= CRISIS_DELTA and consecutive >= SUSTAINED:
        findings.append((
            "MED", "refault-storm-sustained",
            "refault delta sustained {} probes: current={:.0f} pages/s (>crisis), "
            "peak={:.0f} — ongoing paging under load.".format(consecutive, delta, peak)))
    elif peak >= CRISIS_PEAK and delta < CRISIS_DELTA and delta > 40_000.0:
        findings.append((
            "MED", "refault-storm-recovery",
            "refault peak={:.0f} pages/s hit crisis; current delta={:.0f} "
            "recovering but still elevated (>40k) — may re-trigger if workload "
            "repeats.".format(peak, delta)))
    elif delta >= CRISIS_DELTA and consecutive <= 2:
        findings.append((
            "LOW", "refault-spike-transient",
            "refault spike: delta={:.0f} pages/s for {} probe(s) — typical of a "
            "large app context-switch; escalates to MED if sustained.".format(
                delta, consecutive)))
    return findings


def detect_meet_network_tamper(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """da7df34 / 5802c89 — any of the 4 TCP/IPC sysctls != macOS default.

    ZERO tolerance: post-fix Apollo hard-blocks writing these at the emit_sysctl
    chokepoint, so any non-default is a regression (or manual tamper / pre-fix
    daemon). Non-default TCP buffers/delayed_ack/somaxconn degrade WebRTC (Meet)
    audio measurably ("mic cortado de la verga", 2026-06-18). Fires HIGH.
    """
    findings = []
    if not isinstance(sysctls, dict) or not sysctls:
        return findings
    nondefault = []
    for key, expected in SYSCTL_DEFAULTS.items():
        actual = sysctls.get(key)
        if actual is None:
            continue  # unreadable -> skip this key, don't false-alarm
        if not isinstance(actual, int):
            try:
                actual = int(actual)
            except (ValueError, TypeError):
                continue
        if actual != expected:
            nondefault.append("{}={} (default {}, delta {:+d})".format(
                key, actual, expected, actual - expected))
    if nondefault:
        findings.append((
            "HIGH", "meet-network-tamper",
            "non-default network sysctls: {} — post-da7df34 Apollo must NEVER "
            "write these (hard-block at emit_sysctl). Check: (a) regression broke "
            "the block, (b) another tool changed them, (c) daemon ran pre-fix. "
            "WebRTC/Meet audio degrades.".format("; ".join(nondefault))))
    return findings


def detect_boost_loop(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """4b0753e / 4ae0f27 — a single process name dominates BoostProcess.

    Healthy: 2-5 boosts per process per window. Regression: node boosted 54x.
    HIGH: any name boosted > 20 in the window, OR same pid boosted > 15 (churn).
    MED:  name boosted 10-20 (elevated), OR same pid 9-15 (elevated churn).
    Also flags boost of a protected/dev-runtime process (foreground-only policy
    violation) at MED.
    """
    findings = []
    # Fresh-window only — the daemon restarts on deploy, so the raw tail spans
    # pre-fix behavior (node was boosted 89x BEFORE the foreground gate landed).
    journal_tail = fresh_entries(journal_tail, FRESH_WINDOW_MIN)
    if not journal_tail:
        return findings
    name_counts = {}
    pid_counts = {}
    pid_to_name = {}
    try:
        for entry in journal_tail:
            kind, payload = _action_kind_and_payload(entry)
            if kind != "BoostProcess":
                continue
            name = payload.get("name")
            pid = payload.get("pid")
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1
            if pid is not None:
                pid_counts[pid] = pid_counts.get(pid, 0) + 1
                if name:
                    pid_to_name.setdefault(pid, name)
    except Exception as e:
        return [("LOW", "journal-parse-error",
                 "could not parse boost entries: {}".format(str(e)[:80]))]

    if name_counts:
        top_name = max(name_counts, key=lambda k: name_counts[k])
        top_n = name_counts[top_name]
        total = sum(name_counts.values())
        pct = (top_n / total * 100.0) if total else 0.0
        if top_n > 20:
            findings.append((
                "HIGH", "boost-loop",
                "{}: {} boosts ({:.0f}% of {} recent boosts) — loop shape "
                "(node was 54x pre-fix); check RL weight over-promise / "
                "interactive misclassification, and that boost requires "
                "foreground|visible (4ae0f27).".format(top_name, top_n, pct, total)))
        elif top_n > 10:
            findings.append((
                "MED", "boost-elevation",
                "{}: {} boosts ({:.0f}% of recent) — elevated from healthy "
                "baseline (2-5); monitor for loop growth.".format(top_name, top_n, pct)))

    if pid_counts:
        top_pid = max(pid_counts, key=lambda k: pid_counts[k])
        top_pn = pid_counts[top_pid]
        if top_pn > 15:
            findings.append((
                "HIGH", "boost-churn",
                "PID {} ({}): {} boosts — repeated churn, fail-boost-fail "
                "cycle.".format(top_pid, pid_to_name.get(top_pid, "?"), top_pn)))
        elif top_pn > 8:
            findings.append((
                "MED", "boost-churn-elevated",
                "PID {} ({}): {} boosts — elevated churn, monitor for loop "
                "instability.".format(top_pid, pid_to_name.get(top_pid, "?"), top_pn)))

    # Boost of a protected/dev-runtime process: foreground-only policy slipped.
    for name in name_counts:
        if name in PROTECTED_NAMES and name_counts[name] > 5:
            findings.append((
                "MED", "boost-protected-process",
                "{}: {} boosts — protected/dev-runtime process should not need "
                "an RL boost (policy mismatch or non-foreground boost; "
                "4ae0f27).".format(name, name_counts[name])))
    return findings


def detect_llm_stale(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """d6c659b / 4b6da93 / 3df9235 — teacher silently dead.

    Trusts ground truth only: last_call_at (set SOLELY on success) and
    consecutive_failures. Ignores calls_today/triggers (they advance on failed
    attempts and masked the 2026-06-14 outage for 11 days).
    HIGH: enabled AND (last_call_at missing OR > 2 days old).
    MED:  enabled AND (consecutive_failures >= 10, OR last_error has a metal-oom
          / parse / json structural marker).
    LOW:  enabled AND consecutive_failures in [5,10).
    """
    findings = []
    if not isinstance(llm, dict) or "__error__" in llm:
        return findings
    if not llm.get("enabled"):
        return findings
    # The teacher can be turned off via config without the persisted
    # llm_state.enabled flipping (2026-06-20: disabled because it was a net
    # wash-with-overhead). A deliberately-off teacher is NOT "silently dead" —
    # respect the config so this detector doesn't false-flag it.
    if teacher_config_disabled():
        return findings

    cf = llm.get("consecutive_failures", 0)
    if not isinstance(cf, int):
        cf = 0
    last_call = llm.get("last_call_at")
    last_err = llm.get("last_error")

    age_days = None
    if last_call:
        try:
            lc = datetime.fromisoformat(str(last_call).replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - lc).total_seconds() / 86400.0
        except Exception:
            age_days = None

    if age_days is None or age_days > 2.0:
        shown = "never" if age_days is None else "{:.1f}d ago".format(age_days)
        findings.append((
            "HIGH", "llm-stale",
            "teacher enabled but last SUCCESSFUL call {} (consecutive_failures="
            "{}, last_error={}) — ignore calls_today/triggers, they advance on "
            "failed attempts.".format(shown, cf, last_err)))
    else:
        if cf >= 10:
            findings.append((
                "MED", "llm-failing",
                "teacher calling but failing: consecutive_failures={}, "
                "last_error={} — check metal-oom gate (4b6da93) / thinking "
                "truncation (3df9235).".format(cf, last_err)))
        elif cf >= 5:
            findings.append((
                "LOW", "llm-elevated-failures",
                "teacher at {} consecutive failures — elevated, monitor.".format(cf)))
        le = last_err.lower() if isinstance(last_err, str) else ""
        if le and ("metal-oom" in le or "parse" in le or "json" in le):
            findings.append((
                "MED", "llm-structural-fail",
                "last_error has a structural marker: {} — not transient; check "
                "metal-oom swap ceiling (4b6da93) / disable_thinking "
                "(3df9235).".format(last_err)))
    return findings


def detect_complete_mediation(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """03472d7 / 6e0e1ce / a98b33a — a protected process is the TARGET of a
    Freeze/Throttle/SetMemorystatus action.

    Every path that nominates freeze/throttle/jetsam MUST pass through safety.rs
    (complete mediation, Saltzer & Kaashoek 2009). If a protected/Apple/dev-
    runtime name appears as a target here, a path bypassed it.

    Severity splits on `success` (verified in production: most jetsam demotes on
    protected procs land `success:false` with `skip:memorystatus-send-failed`):
      HIGH — action SUCCEEDED on a protected process (the breach LANDED; harm done).
      MED  — action was NOMINATED but failed/skipped (success:false). The decision
             path still chose a protected target — a mediation gap — but nothing
             landed; the upstream guard (or the kernel) caught it.
    FreezeProcess on Brave/Chromium is freeze-exempt by design (their IPC
    contract), so freeze of those does NOT fire — but a throttle or jetsam demote
    of them still does.

    NOTE: SetMemorystatus carries no `name` in production; the target name is
    parsed out of `reason` ('... MemoryHoarder: <name> (NNNmb) ...').
    """
    findings = []
    # Fresh-window only — don't re-flag a fixed mediation breach from pre-deploy
    # journal residue (e.g. the LSP/stale-app freezes before 03472d7/a98b33a).
    journal_tail = fresh_entries(journal_tail, FRESH_WINDOW_MIN)
    if not journal_tail:
        return findings
    # Chromium family is exempt from ALL kinds here: jetsam BACKGROUND demote
    # (SetMemorystatus) and PRIO_DARWIN_BG throttle are the SANCTIONED
    # Chromium-cooperative paths (the documented alternative to freeze), and
    # freeze itself is permanently disabled. None of these is a mediation breach
    # — flagging them is a false positive (diagnosed 2026-06-20: Brave×17 jetsam
    # nominations are the normal cooperative path, not a safety bypass).
    chromium_family = {"Brave", "Chromium"}
    # (kind, landed) -> {name: count}
    landed = {}     # success == True
    nominated = {}  # success != True
    try:
        for entry in journal_tail:
            kind, payload = _action_kind_and_payload(entry)
            if kind not in ("FreezeProcess", "ThrottleProcess", "SetMemorystatus"):
                continue
            name = _name_from_payload(kind, payload)
            if not name:
                continue
            matched = None
            nl = name.lower()
            for pn in PROTECTED_NAMES:
                pl = pn.lower()
                # substring (covers "Brave Browser Helper" → "brave") OR
                # macOS 15-char truncation (policy stores full, runtime truncates).
                if pl in nl or (len(pl) >= 8 and pl.startswith(nl) and len(nl) >= 8):
                    matched = pn
                    break
            if not matched:
                continue
            if matched in chromium_family:
                continue  # all Chromium actions are cooperative/disabled, not a breach
            bucket = landed if entry.get("success") is True else nominated
            bucket.setdefault(kind, {}).setdefault(matched, 0)
            bucket[kind][matched] += 1
    except Exception:
        return findings

    for kind, names in landed.items():
        parts = ["{}x{}".format(n, c) for n, c in sorted(names.items())]
        findings.append((
            "HIGH", "mediation-break-{}".format(kind.lower()),
            "{} LANDED on protected process(es): {} — complete-mediation breach; "
            "the action succeeded against a safety.rs-protected target "
            "(03472d7/6e0e1ce/a98b33a).".format(kind, ", ".join(parts))))
    for kind, names in nominated.items():
        parts = ["{}x{}".format(n, c) for n, c in sorted(names.items())]
        findings.append((
            "MED", "mediation-nominate-{}".format(kind.lower()),
            "{} NOMINATED but failed/skipped on protected process(es): {} — "
            "decision path chose a protected target (mediation gap); nothing "
            "landed, but the nomination should be blocked upstream at safety.rs "
            "(03472d7/6e0e1ce/a98b33a).".format(kind, ", ".join(parts))))
    return findings


def detect_performance_baseline(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend):
    """General baseline: ais_score<80, failures>0, last_error set.

    HIGH: failures>0, OR last_error is a non-empty string, OR 0<ais<80.
    MED:  80<=ais<87 (degraded band), OR ais dropped >5 pts vs a healthy trend
          baseline (avg>=90 over >=2 prior rows).
    """
    findings = []
    if not isinstance(rt, dict) or "__error__" in rt:
        return findings

    failures = rt.get("failures", 0)
    last_err = rt.get("last_error")
    ais = rt.get("ais_score")

    if isinstance(failures, (int, float)) and failures > 0:
        findings.append((
            "HIGH", "daemon-failures",
            "failures={}, last_error={} — daemon crashed/errored.".format(failures, last_err)))
    elif isinstance(last_err, str) and last_err.strip():
        findings.append((
            "HIGH", "daemon-error",
            "last_error={} — daemon recorded an error.".format(last_err)))

    if isinstance(ais, (int, float)):
        if 0 < ais < 80:
            findings.append((
                "HIGH", "ais-degraded",
                "ais_score={:.1f} (<80, healthy 90-96) — system under stress.".format(ais)))
        elif 80 <= ais < 87:
            findings.append((
                "MED", "ais-elevated",
                "ais_score={:.1f} (80-87 degraded band) — approaching "
                "degradation.".format(ais)))
        elif ais >= 80 and isinstance(trend, list) and len(trend) >= 2:
            prior = [r.get("ais_score") for r in trend[-5:]
                     if isinstance(r, dict) and isinstance(r.get("ais_score"), (int, float))]
            if len(prior) >= 2:
                avg = sum(prior) / len(prior)
                if avg >= 90 and ais < avg - 5:
                    findings.append((
                        "MED", "ais-decline",
                        "ais_score={:.1f} declined from trend avg {:.1f} "
                        "(>5pt drop) — learned_state bloat or cycle inflation?".format(
                            ais, avg)))
    return findings


DETECTORS = [
    detect_purge_strangled,
    detect_cycle_inflation,
    detect_refault_storm,
    detect_meet_network_tamper,
    detect_boost_loop,
    detect_llm_stale,
    detect_complete_mediation,
    detect_performance_baseline,
]


# ── Metrics (cycle-over-cycle trend row) ──────────────────────────────────────
def build_metrics(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls):
    """Build the metrics dict appended to the trend JSONL each run."""
    def num(v, default=None):
        return v if isinstance(v, (int, float)) else default

    # purge skip ratio
    skipped = rt.get("maintenance_purge_skipped_idle_total", 0)
    purged = rt.get("maintenance_purge_total", 0)
    skipped = skipped if isinstance(skipped, int) else 0
    purged = purged if isinstance(purged, int) else 0
    tot = skipped + purged
    skip_ratio = (skipped / tot) if tot > 0 else 0.0

    # max boost for a single name + protected action counts (over journal_tail).
    # protected_action_count = LANDED only (matches the HIGH detector);
    # protected_nominated_count = attempted-but-failed (the MED gap).
    name_counts = {}
    protected_actions = 0
    protected_nominated = 0
    freeze_exempt = {"Brave", "Chromium"}
    for entry in journal_tail:
        kind, payload = _action_kind_and_payload(entry)
        if kind == "BoostProcess":
            nm = payload.get("name")
            if nm:
                name_counts[nm] = name_counts.get(nm, 0) + 1
        elif kind in ("FreezeProcess", "ThrottleProcess", "SetMemorystatus"):
            nm = _name_from_payload(kind, payload)
            if nm:
                nl = nm.lower()
                for pn in PROTECTED_NAMES:
                    if pn.lower() in nl:
                        if kind == "FreezeProcess" and pn in freeze_exempt:
                            break
                        if entry.get("success") is True:
                            protected_actions += 1
                        else:
                            protected_nominated += 1
                        break
    max_boost = max(name_counts.values()) if name_counts else 0

    nondefault = 0
    for key, expected in SYSCTL_DEFAULTS.items():
        v = sysctls.get(key)
        if isinstance(v, int) and v != expected:
            nondefault += 1

    cf = llm.get("consecutive_failures", 0) if isinstance(llm, dict) else 0
    if not isinstance(cf, int):
        cf = 0

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ais_score": num(rt.get("ais_score")),
        "p95_cycle_ms": num(rt.get("p95_cycle_ms")),
        "thrashing_score": num(rt.get("thrashing_score")),
        "memory_pressure": num(rt.get("memory_pressure")),
        "refault_delta_per_sec": num(rt.get("refault_delta_per_sec")),
        "refault_peak_per_sec": num(rt.get("refault_peak_per_sec")),
        "beliefs_count": ls_beliefs,
        "ls_size_mb": round(ls_size_mb, 3) if isinstance(ls_size_mb, (int, float)) else None,
        "purge_skip_ratio": round(skip_ratio, 4),
        "max_boost_single_name": max_boost,
        "protected_action_count": protected_actions,
        "protected_nominated_count": protected_nominated,
        "llm_consecutive_failures": cf,
        "tcp_sysctls_nondefault_count": nondefault,
        "failures": num(rt.get("failures"), 0),
        "cycle_count": num(rt.get("cycle_count")),
    }


# ── Trend table (--trend N) ───────────────────────────────────────────────────
def print_trend_table(n):
    rows = read_trend_tail(TREND, n)
    if not rows:
        print("(no trend rows in {})".format(TREND))
        return
    cols = [
        ("ts", 25), ("ais_score", 8), ("p95_cycle_ms", 8),
        ("thrashing_score", 12), ("refault_peak_per_sec", 14),
        ("beliefs_count", 9), ("ls_size_mb", 9),
        ("max_boost_single_name", 8), ("purge_skip_ratio", 9),
    ]
    header = " ".join(name[:width].rjust(width) for name, width in cols)
    print(header)
    print("-" * len(header))
    for r in rows[-n:]:
        cells = []
        for name, width in cols:
            v = r.get(name)
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = "{:.2f}".format(v) if abs(v) < 1000 else "{:.0f}".format(v)
            else:
                s = str(v)
            cells.append(s[:width].rjust(width))
        print(" ".join(cells))


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv):
    quiet = "--quiet" in argv
    as_json = "--json" in argv
    if "--trend" in argv:
        i = argv.index("--trend")
        try:
            n = int(argv[i + 1])
        except (IndexError, ValueError):
            n = TREND_TAIL_ROWS
        print_trend_table(n)
        return 0

    rt = load(METRICS)
    llm = load(LLM)
    journal_tail = read_journal_tail(JOURNAL, JOURNAL_TAIL_LINES)
    sysctls = read_sysctls()
    ls_size_mb, ls_beliefs = learned_state_stats(LEARNED_STATE)
    trend = read_trend_tail(TREND, TREND_TAIL_ROWS)

    # Run every detector (one bad detector can't crash the probe).
    findings = []
    for det in DETECTORS:
        try:
            findings.extend(det(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls, trend))
        except Exception as e:
            findings.append(("LOW", "detector-error",
                             "{} raised {}".format(det.__name__, str(e)[:120])))

    # Build + append the metrics row (the cycle-over-cycle history).
    metrics = build_metrics(rt, journal_tail, llm, ls_size_mb, ls_beliefs, sysctls)
    # Record this run's finding codes so the NEXT run can detect persistence:
    # a MED present across 2 consecutive runs is actionable (the loop escalates
    # it), whereas a one-shot MED (e.g. pre-deploy residue) self-clears.
    metrics["finding_codes"] = sorted({c for (_s, c, _d) in findings})
    try:
        with open(TREND, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception:
        pass  # read-only env / no write perm — never crash

    # Verdict + report.
    ts = datetime.now(timezone.utc).isoformat()
    high = [f for f in findings if f[0] == "HIGH"]
    if rt.get("__error__"):
        findings.insert(0, ("HIGH", "metrics-unreadable", rt["__error__"]))
        high = [f for f in findings if f[0] == "HIGH"]

    # Persisted-actionable: a MED present in BOTH this run and the previous
    # trend row (2 consecutive runs) — no longer transient, the loop escalates
    # it. `trend[-1]` is the prior run; this run's row (appended above) is not
    # in `trend`, which was read before the append.
    prev_codes = set(trend[-1].get("finding_codes", [])) if trend else set()
    persistent_med = sorted(
        {c for (s, c, _d) in findings if s == "MED"} & prev_codes
    )

    # --json: structured output for the fix-loop (Layer 2) to consume.
    if as_json:
        print(json.dumps({
            "timestamp": ts,
            "findings": [{"severity": s, "code": c, "detail": d} for s, c, d in findings],
            "persistent_med": persistent_med,
            "metrics": metrics,
        }, indent=1))
        return 2 if high else (1 if findings else 0)
    if findings:
        verdict = "{} finding(s)".format(len(findings)) + (
            " — {} HIGH".format(len(high)) if high else "")
    else:
        verdict = "clean"

    line = "[{}] {}".format(ts, verdict)
    # severity order for readability: HIGH, MED, LOW
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    for sev, code, detail in sorted(findings, key=lambda f: order.get(f[0], 9)):
        line += "\n    {:4} {}: {}".format(sev, code, detail)

    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

    # Flag wakes the Layer-2 Monitor → fix-loop. Raise on HIGH (immediate) OR
    # on a MED that persisted 2 runs (event-driven escalation ~30min vs the
    # hourly heartbeat). Clear only when neither is present. The flag is the
    # single cross-process signal; the tier line tells the woken agent which.
    try:
        if high or persistent_med:
            tier = "HIGH" if high else "PERSISTENT-MED"
            with open(FLAG, "w") as f:
                f.write("[{}] {}\n{}".format(tier, ts, line))
        elif os.path.exists(FLAG):
            os.remove(FLAG)
    except Exception:
        pass

    if not quiet:
        print(line)
    elif high:
        print("[{}] {} HIGH finding(s) — see {}".format(ts, len(high), LOG))

    if high:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
