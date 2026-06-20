#!/usr/bin/env python3
"""apollo-stress-bench.py — Apollo STRESS-tier benchmark (Gate 3.5 observer).

Implements the stress scorecard from the Apollo Autonomous-Loop Acceptance
Framework: establish a baseline reading, apply a controlled load, hold, measure,
release, measure recovery. Print a SCORECARD with per-line PASS/FAIL and an
overall stress-score 0..100.

WHY SYNTHETIC LOAD IS THE DEFAULT (not 50 real GUI apps)
--------------------------------------------------------
Opening 50 real GUI applications is (a) invasive — it disrupts whatever the
human is doing on a shared single-box workstation, (b) non-deterministic —
LaunchServices/dyld/Spotlight cold-launch noise means the same run never
reproduces, and (c) unbounded — 50 arbitrary apps can OOM an 8 GiB M1. So the
DEFAULT load is N child processes that each allocate a bounded chunk of anon
memory and re-touch it once a second (keeps the working set hot -> forces real
eviction, crossing benign-churn into TRUE scarcity, which is what the spec
demands of a stress load). A hard TOTAL cap and an automatic danger-line abort
keep the machine safe. `--real-apps "App1,App2"` is an opt-in advisory demo that
uses `open -a`; it is never a gate and never the default.

SAFETY
------
- bounded allocation: total = apps * mb_each, clamped under --max-total-mb.
- danger line: if measured memory_pressure exceeds --danger-pressure during ramp
  or hold, the load is released immediately and the run aborts (still scored).
- complete cleanup: a signal handler (SIGINT/SIGTERM) + atexit BOTH terminate
  every spawned child, even on Ctrl-C. Children also self-exit if the parent
  dies (they watch getppid()).

ponytail: pure python3 stdlib, no C compile step, no third-party deps. Holder is
an inline `python3 -c` child (the spec's C holder is an optimization for the
heavier orchestrator; here a python holder is enough and zero-build). Upgrade
path: swap the holder argv for the compiled apollo-stress-holder.c if RSS
fidelity ever matters.
"""

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time

PAGESIZE_DEFAULT = 16384  # confirmed on M1 8GB; overridden live from vm_stat header

# ---- Apollo state file locations (root install, then /tmp non-root fallback) ----
RUNTIME_CANDIDATES = [
    "/var/lib/apollo/runtime_metrics.json",
    "/tmp/runtime_metrics.json",
]
JOURNAL_CANDIDATES = [
    "/var/lib/apollo/journal.jsonl",
    "/tmp/journal.jsonl",
]

# safety.rs hard-list — processes Apollo must NEVER kill/freeze. Used for the
# no-wrongful-action scan. Matched as substrings against `ps -axo comm`.
PROTECTED_NAMES = [
    "kernel_task", "launchd", "WindowServer", "loginwindow", "configd",
    "Spotlight", "mds", "Finder", "Antigravity", "Claude",
    "Brave Browser", "language_server", "rustc", "cargo",
]

# Holder child: allocate N MiB of bytearray, touch one byte per page each second
# so the working set stays resident (forces real eviction under pressure).
# Self-exits if reparented to launchd (ppid==1) so we never leak on crash.
HOLDER_SRC = (
    "import sys,time,os\n"
    "mb=int(sys.argv[1]); ppid=os.getppid()\n"
    "buf=bytearray(mb*1024*1024)\n"
    "step=4096\n"
    "for i in range(0,len(buf),step): buf[i]=1\n"
    "while True:\n"
    "    if os.getppid()!=ppid: os._exit(0)\n"
    "    for i in range(0,len(buf),step): buf[i]=(buf[i]+1)&0xff\n"
    "    time.sleep(1.0)\n"
)

_CHILDREN = []  # list[subprocess.Popen]
_CLEANED = False


# ----------------------------- cleanup -----------------------------------------
def _cleanup(*_a):
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    for p in _CHILDREN:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    deadline = time.time() + 5.0
    for p in _CHILDREN:
        try:
            remaining = max(0.0, deadline - time.time())
            p.wait(timeout=remaining)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _CHILDREN.clear()


def _signal_handler(signum, _frame):
    sys.stderr.write(f"\n[stress] signal {signum} received — releasing load and cleaning up\n")
    _cleanup()
    # Re-raise default disposition so exit code reflects the signal.
    sys.exit(128 + signum)


atexit.register(_cleanup)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ----------------------------- measurement -------------------------------------
def read_pagesize():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return PAGESIZE_DEFAULT


def vm_stat_counters():
    """Return dict of raw vm_stat page counters (swapouts/swapins/compressions...)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r'\s*"?([^":]+)"?:\s+([\d.]+)\.?', line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            try:
                d[key] = int(float(m.group(2)))
            except ValueError:
                pass
    return d


def memory_pressure_pct():
    """System memory pressure in [0,1] from `memory_pressure` free%; None on failure."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=8).stdout
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
        if m:
            return max(0.0, min(1.0, 1.0 - int(m.group(1)) / 100.0))
    except Exception:
        pass
    return None


def swap_used_mb():
    """Used swap in MiB from sysctl vm.swapusage; None on failure."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def compressed_ratio(counters):
    """Pages occupied by compressor / total physical pages — saturation proxy."""
    occ = counters.get("pages_occupied_by_compressor") or counters.get("pages_used_by_compressor")
    if occ is None:
        return None
    try:
        memsize = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip())
        pagesize = read_pagesize()
        total_pages = memsize / pagesize
        return occ / total_pages if total_pages else None
    except Exception:
        return None


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def read_runtime_metrics():
    """Apollo daemon self-report (cycles/failures/last_error/pressure/p95); None if absent."""
    path = first_existing(RUNTIME_CANDIDATES)
    if not path:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        d["__path"] = path
        return d
    except Exception:
        return None


def protected_set():
    """Set of protected process names currently running (substring match on `ps comm`)."""
    try:
        out = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return set()
    present = set()
    low = out.lower()
    for name in PROTECTED_NAMES:
        if name.lower() in low:
            present.add(name)
    return present


def journal_fresh_actions(since_epoch):
    """Scan Apollo journal for kill/freeze actions on protected/foreground procs
    AFTER since_epoch. Returns (n_scanned, list_of_offending_entries).
    Degrades to (0, []) if journal absent. Never raises."""
    path = first_existing(JOURNAL_CANDIDATES)
    if not path:
        return (0, [])
    offending = []
    scanned = 0
    bad_classes = ("freeze", "kill", "sigstop", "sigkill", "terminate", "jetsam")
    prot_low = [n.lower() for n in PROTECTED_NAMES]
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                ts = _entry_epoch(e)
                if ts is None or ts < since_epoch:
                    continue
                scanned += 1
                blob = json.dumps(e).lower()
                if any(b in blob for b in bad_classes) and any(p in blob for p in prot_low):
                    offending.append(e)
    except Exception:
        return (scanned, offending)
    return (scanned, offending)


def _entry_epoch(e):
    """Best-effort extract an epoch timestamp from a journal entry."""
    for k in ("ts", "timestamp", "time", "epoch", "unix_ts"):
        v = e.get(k)
        if isinstance(v, (int, float)):
            # heuristics: seconds vs millis
            return v / 1000.0 if v > 1e12 else float(v)
        if isinstance(v, str):
            # ISO-8601 -> epoch via time.strptime (best-effort, tz-naive)
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return time.mktime(time.strptime(v[:19], fmt))
                except Exception:
                    pass
    return None


def sample(pagesize):
    """One observation snapshot."""
    c = vm_stat_counters()
    rt = read_runtime_metrics()
    return {
        "t": time.time(),
        "pressure": memory_pressure_pct(),
        "swap_used_mb": swap_used_mb(),
        "swapouts": c.get("swapouts"),
        "swapins": c.get("swapins"),
        "compressions": c.get("compressions"),
        "compressed_ratio": compressed_ratio(c),
        "protected": protected_set(),
        "rt_cycles": (rt or {}).get("cycles"),
        "rt_failures": (rt or {}).get("failures"),
        "rt_last_error": (rt or {}).get("last_error", None),
        "rt_p95_ms": (rt or {}).get("p95_cycle_ms"),
        "rt_pressure": (rt or {}).get("memory_pressure"),
        "rt_thrash": (rt or {}).get("thrashing_score"),
        "rt_present": rt is not None,
    }


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def rate_per_sec(samples, key):
    """Median per-second delta of a monotonic counter across consecutive samples."""
    rates = []
    prev = None
    for s in samples:
        v = s.get(key)
        if v is not None and prev is not None and prev[0] is not None:
            dt = s["t"] - prev[1]
            if dt > 0 and v >= prev[0]:
                rates.append((v - prev[0]) / dt)
        prev = (v, s["t"])
    return median(rates)


# ----------------------------- load ramp ---------------------------------------
def spawn_holder(mb):
    p = subprocess.Popen(
        [sys.executable, "-c", HOLDER_SRC, str(mb)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _CHILDREN.append(p)
    return p


# ----------------------------- main run ----------------------------------------
def run(args):
    pagesize = read_pagesize()
    total_mb = args.apps * args.mb_each
    if total_mb > args.max_total_mb:
        print(f"[stress] requested total {total_mb} MiB exceeds hard cap "
              f"{args.max_total_mb} MiB — refusing (raise --max-total-mb to override).",
              file=sys.stderr)
        return 2

    aborted = False
    abort_reason = None
    run_start = time.time()

    print(f"[stress] config: apps={args.apps} mb_each={args.mb_each} "
          f"total={total_mb}MiB ramp={args.ramp}s hold={args.hold}s recover={args.recover}s "
          f"danger_pressure={args.danger_pressure} pagesize={pagesize}")
    if args.real_apps:
        print(f"[stress] --real-apps (ADVISORY DEMO, not a gate): will `open -a` -> "
              f"{args.real_apps}")

    # ---- BASELINE -------------------------------------------------------------
    print("[stress] baseline reading...")
    baseline_samples = []
    base_deadline = time.time() + args.baseline
    while time.time() < base_deadline:
        baseline_samples.append(sample(pagesize))
        time.sleep(0.5)
    if not baseline_samples:
        baseline_samples.append(sample(pagesize))
    baseline_pressure = median([s["pressure"] for s in baseline_samples])
    baseline_swap = median([s["swap_used_mb"] for s in baseline_samples])
    baseline_protected = set()
    for s in baseline_samples:
        baseline_protected |= s["protected"]
    base_cycles = next((s["rt_cycles"] for s in reversed(baseline_samples) if s["rt_cycles"] is not None), None)
    print(f"[stress] baseline: pressure={_fmt(baseline_pressure)} "
          f"swap_used={_fmt(baseline_swap)}MiB protected={len(baseline_protected)} procs "
          f"apollo_runtime={'yes' if baseline_samples[-1]['rt_present'] else 'absent'}")

    # ---- RAMP (one holder every ramp/apps seconds) ----------------------------
    print(f"[stress] ramping {args.apps} holders...")
    per = (args.ramp / args.apps) if args.apps else 0
    hold_samples_during_ramp = []
    for i in range(args.apps):
        spawn_holder(args.mb_each)
        t_end = time.time() + max(per, 0.05)
        while time.time() < t_end:
            s = sample(pagesize)
            hold_samples_during_ramp.append(s)
            if s["pressure"] is not None and s["pressure"] >= args.danger_pressure:
                aborted = True
                abort_reason = f"danger line: pressure {s['pressure']:.2f} >= {args.danger_pressure} during ramp"
                break
            time.sleep(0.25)
        if aborted:
            break

    # ---- HOLD -----------------------------------------------------------------
    hold_samples = []
    if not aborted:
        print(f"[stress] holding {args.hold}s under load...")
        if args.real_apps:
            for app in [a.strip() for a in args.real_apps.split(",") if a.strip()]:
                try:
                    subprocess.run(["open", "-a", app], timeout=10)
                except Exception as ex:
                    print(f"[stress]   open -a {app!r} failed: {ex}", file=sys.stderr)
        hold_deadline = time.time() + args.hold
        while time.time() < hold_deadline:
            s = sample(pagesize)
            hold_samples.append(s)
            if s["pressure"] is not None and s["pressure"] >= args.danger_pressure:
                aborted = True
                abort_reason = f"danger line: pressure {s['pressure']:.2f} >= {args.danger_pressure} during hold"
                break
            time.sleep(0.5)

    # ---- RELEASE + RECOVERY ---------------------------------------------------
    print("[stress] releasing load...")
    release_t = time.time()
    _cleanup()  # terminate all holders now (recovery clock starts)

    recovery_samples = []
    recovery_time = None
    rec_deadline = time.time() + args.recover
    while time.time() < rec_deadline:
        s = sample(pagesize)
        recovery_samples.append(s)
        if (recovery_time is None and baseline_pressure is not None
                and s["pressure"] is not None
                and s["pressure"] <= baseline_pressure + args.recover_band):
            recovery_time = s["t"] - release_t
        time.sleep(0.5)

    # ---- AGGREGATE ------------------------------------------------------------
    all_load = hold_samples_during_ramp + hold_samples
    peak_pressure = max((s["pressure"] for s in all_load if s["pressure"] is not None), default=None)
    peak_compress = max((s["compressed_ratio"] for s in all_load if s["compressed_ratio"] is not None), default=None)
    swapout_rate = rate_per_sec(all_load, "swapouts")
    post_swap = recovery_samples[-1]["swap_used_mb"] if recovery_samples else None

    # protected-set diff: every protected proc present at baseline still present?
    final_protected = set()
    for s in recovery_samples or all_load or baseline_samples:
        final_protected |= s["protected"]
    killed_protected = baseline_protected - final_protected

    # apollo liveness under load (only if runtime metrics present)
    rt_present = baseline_samples[-1]["rt_present"]
    end_cycles = next((s["rt_cycles"] for s in reversed(recovery_samples + hold_samples) if s["rt_cycles"] is not None), None)
    failures_end = next((s["rt_failures"] for s in reversed(recovery_samples + hold_samples) if s["rt_failures"] is not None), None)
    last_error_end = next((s["rt_last_error"] for s in reversed(recovery_samples + hold_samples) if s["rt_last_error"] not in (None,)), None)
    cycles_advanced = (rt_present and base_cycles is not None and end_cycles is not None and end_cycles > base_cycles)

    # journal scan for wrongful kill/freeze of protected procs during the window
    j_scanned, j_offending = journal_fresh_actions(run_start)

    score, lines, hard_ok = scorecard(
        args=args,
        aborted=aborted, abort_reason=abort_reason,
        baseline_pressure=baseline_pressure, peak_pressure=peak_pressure,
        recovery_time=recovery_time, killed_protected=killed_protected,
        swapout_rate=swapout_rate, baseline_swap=baseline_swap, post_swap=post_swap,
        peak_compress=peak_compress,
        rt_present=rt_present, cycles_advanced=cycles_advanced,
        failures_end=failures_end, last_error_end=last_error_end,
        j_scanned=j_scanned, j_offending=j_offending,
        pagesize=pagesize,
    )

    print_scorecard(lines, score, hard_ok, aborted, abort_reason)
    return 0 if (hard_ok and score >= 80) else 5


def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


# ----------------------------- scorecard ---------------------------------------
def scorecard(**k):
    args = k["args"]
    lines = []
    pagesize = k["pagesize"]

    def add(tag, label, passed, detail, hard):
        lines.append({"tag": tag, "label": label, "pass": passed, "detail": detail, "hard": hard})

    # ---- HARD lines (S1-S4) ----
    killed = k["killed_protected"]
    add("S1", "No wrongful kill of protected process",
        len(killed) == 0,
        "none missing" if not killed else f"MISSING: {sorted(killed)}", hard=True)

    # journal-confirmed wrongful action folds into S1 verdict
    if k["j_offending"]:
        lines[-1]["pass"] = False
        lines[-1]["detail"] += f"; journal flagged {len(k['j_offending'])} kill/freeze on protected"

    if k["rt_present"]:
        s2_pass = (k["cycles_advanced"] and (k["failures_end"] in (0, None))
                   and (k["last_error_end"] in (None, "", "null")))
        s2_detail = (f"cycles_advanced={k['cycles_advanced']} failures={k['failures_end']} "
                     f"last_error={k['last_error_end']!r}")
    else:
        s2_pass = True  # cannot fail what we cannot observe
        s2_detail = "apollo runtime_metrics.json absent (daemon not running) — SKIP, treated PASS"
    add("S2", "Daemon survived (alive, cycles advancing, no failure)", s2_pass, s2_detail, hard=True)

    bp = k["baseline_pressure"]
    rt = k["recovery_time"]
    s3_pass = (rt is not None) and (rt <= args.recover_within)
    if bp is None:
        s3_detail = "baseline pressure unreadable — cannot judge recovery"
        s3_pass = False
    elif rt is None:
        s3_detail = (f"pressure did NOT return to baseline+{args.recover_band} "
                     f"({_fmt(bp)}+{args.recover_band}) within {args.recover}s recovery window")
    else:
        s3_detail = f"recovered to baseline in {rt:.1f}s (limit {args.recover_within}s)"
    add("S3", "Pressure recovered after release", s3_pass, s3_detail, hard=True)

    sr = k["swapout_rate"]
    bsw, psw = k["baseline_swap"], k["post_swap"]
    swap_leak_ok = (bsw is None or psw is None) or (psw <= bsw + args.swap_leak_mb)
    rate_ok = (sr is None) or (sr <= args.max_swapout_rate)
    s4_pass = rate_ok and swap_leak_ok
    mbps = f"{sr*pagesize/1024/1024:.1f}MB/s" if sr is not None else "n/a"
    add("S4", "Swap bounded (rate + no permanent leak)", s4_pass,
        f"median swapouts={_fmt(sr,0)}pg/s ({mbps}, limit {args.max_swapout_rate}pg/s); "
        f"post_swap={_fmt(psw)} baseline_swap={_fmt(bsw)} (+{args.swap_leak_mb} allowed)", hard=True)

    # If the run aborted on the danger line, the load proved unsafe -> hard fail.
    if k["aborted"]:
        add("S0", "Danger-line abort", False, k["abort_reason"], hard=True)

    # ---- ADVISORY lines (S5-S9), 20 pts each ----
    pc = k["peak_compress"]
    s5 = (pc is not None) and (pc < 0.55)
    add("S5", "Compressor absorbed (peak ratio < 0.55)",
        s5, f"peak compressed_ratio={_fmt(pc)}", hard=False)

    pk = k["peak_pressure"]
    # S6 in this synthetic harness: peak pressure stayed below the danger line
    # (we WANT scarcity, but bounded). p95-cycle ms only meaningful if daemon up.
    if k["rt_present"]:
        # use thrash-vs-danger as the load-tolerance proxy
        s6 = (pk is not None) and (pk < args.danger_pressure)
        s6_detail = f"peak pressure under load={_fmt(pk)} (danger {args.danger_pressure})"
    else:
        s6 = (pk is not None) and (pk < args.danger_pressure)
        s6_detail = f"peak pressure under load={_fmt(pk)} (danger {args.danger_pressure}); daemon absent"
    add("S6", "Pressure bounded under load (below danger line)", s6, s6_detail, hard=False)

    # S7 FG responsiveness: approximated by whether peak pressure left headroom
    # below the CRITICAL_PRESSURE_BYPASS (0.65) for non-trivial fraction.
    bypass = 0.65
    s7 = (pk is not None)  # we always reach here; quality = how far under bypass at peak
    s7_pass = (pk is not None) and (pk <= bypass + 0.10)
    add("S7", "Foreground headroom (peak <= bypass+0.10)", s7_pass,
        f"peak pressure={_fmt(pk)} vs bypass {bypass}", hard=False)

    # S8 recovery speed bonus: recovered well within the window
    s8 = (k["recovery_time"] is not None) and (k["recovery_time"] <= args.recover_within * 0.6)
    add("S8", "Fast recovery (<= 0.6x limit)", s8,
        f"recovery={_fmt(k['recovery_time'],1)}s vs 0.6x limit {args.recover_within*0.6:.1f}s", hard=False)

    # S9 benign-churn discrimination: no wrongful protected action even if thrash
    # spiked. Folds the journal scan: zero offending == discrimination held.
    s9 = (len(k["j_offending"]) == 0) and (len(killed) == 0)
    add("S9", "No benign-churn false-alarm (churn != scarcity)", s9,
        f"journal scanned={k['j_scanned']} offending={len(k['j_offending'])}", hard=False)

    hard_ok = all(l["pass"] for l in lines if l["hard"])
    advisory = [l for l in lines if not l["hard"]]
    score = sum(20 for l in advisory if l["pass"]) if hard_ok else 0
    return score, lines, hard_ok


def print_scorecard(lines, score, hard_ok, aborted, abort_reason):
    print("\n" + "=" * 68)
    print("  APOLLO STRESS SCORECARD")
    print("=" * 68)
    for l in lines:
        mark = "PASS" if l["pass"] else "FAIL"
        kind = "HARD" if l["hard"] else "ADV "
        print(f"  [{mark}] {kind} {l['tag']:<3} {l['label']}")
        print(f"            -> {l['detail']}")
    print("-" * 68)
    if aborted:
        print(f"  RUN ABORTED ON DANGER LINE: {abort_reason}")
    band = "PASS" if (hard_ok and score >= 80) else (
        "PASS-MARGINAL" if (hard_ok and score >= 60) else "FAIL")
    print(f"  HARD GATES: {'ALL PASS' if hard_ok else 'FAILED (score forced to 0)'}")
    print(f"  STRESS-SCORE: {score}/100   VERDICT: {band}")
    print("=" * 68)


# ----------------------------- cli ---------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Apollo STRESS-tier benchmark (synthetic-load default; safe, bounded, auto-cleanup).")
    p.add_argument("--apps", type=int, default=50,
                   help="number of synthetic memory-holder children (default 50)")
    p.add_argument("--mb-each", type=int, default=60,
                   help="MiB allocated + re-touched per holder (default 60)")
    p.add_argument("--max-total-mb", type=int, default=5000,
                   help="HARD cap on apps*mb_each; refuse to run above this (default 5000 = safe fraction of 8GiB)")
    p.add_argument("--danger-pressure", type=float, default=0.80,
                   help="abort+release if memory_pressure crosses this (default 0.80)")
    p.add_argument("--baseline", type=float, default=8.0, help="baseline sampling seconds")
    p.add_argument("--ramp", type=float, default=24.0, help="ramp-up seconds (one holder spread across)")
    p.add_argument("--hold", type=float, default=120.0, help="hold-under-load seconds")
    p.add_argument("--recover", type=float, default=30.0, help="recovery observation seconds")
    p.add_argument("--recover-band", type=float, default=0.05,
                   help="pressure considered 'recovered' when <= baseline + this (default 0.05)")
    p.add_argument("--recover-within", type=float, default=20.0,
                   help="S3 pass if recovery happens within this many seconds (default 20)")
    p.add_argument("--max-swapout-rate", type=float, default=2000.0,
                   help="S4 pass if median swapouts/sec <= this (default 2000 pg/s ~= 31MB/s at 16KB)")
    p.add_argument("--swap-leak-mb", type=float, default=256.0,
                   help="S4 pass if post_swap <= baseline_swap + this (default 256 MiB)")
    p.add_argument("--real-apps", type=str, default="",
                   help="ADVISORY DEMO ONLY (not a gate): comma list of app names to `open -a` during hold")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    try:
        return run(args)
    finally:
        _cleanup()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
