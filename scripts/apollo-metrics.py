#!/usr/bin/env python3
"""Apollo v0.5.0 metrics collector — run as root for full data.

Usage:
    sudo python3 scripts/apollo-metrics.py          # one-shot
    sudo python3 scripts/apollo-metrics.py --watch   # continuous (5s interval)
"""

import json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

STATE_DIR = Path("/var/lib/apollo")
METRICS = STATE_DIR / "runtime_metrics.json"
OVERFLOW = STATE_DIR / "overflow_history.json"
MARKOV = STATE_DIR / "markov_transitions.json"
HOLTWINTERS = STATE_DIR / "holt_winters.json"
GOVERNOR = STATE_DIR / "governor_state.json"
FROZEN = STATE_DIR / "frozen_state.json"

def load_json(path):
    try:
        return json.loads(path.read_text())
    except:
        return None

def fmt_bytes(b):
    if b > 1e9: return f"{b/1e9:.1f}GB"
    if b > 1e6: return f"{b/1e6:.0f}MB"
    return f"{b/1e3:.0f}KB"

def get_pressure():
    """Read kern.memorystatus_level via sysctl."""
    try:
        r = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                          capture_output=True, text=True, timeout=2)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except:
        return None

def get_vm_stats():
    """Parse vm_stat for compressor and page-in data."""
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
        stats = {}
        for line in r.stdout.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                try:
                    stats[key.strip()] = int(val)
                except:
                    pass
        page_size = 16384  # ARM64
        return {
            "compressor_pages": stats.get("Pages stored in compressor", 0),
            "compressed_pages": stats.get("Pages occupied by compressor", 0),
            "pageins": stats.get("Pageins", 0),
            "pageouts": stats.get("Pageouts", 0),
            "swapins": stats.get("Swapins", 0),
            "swapouts": stats.get("Swapouts", 0),
            "compressor_bytes": stats.get("Pages stored in compressor", 0) * page_size,
        }
    except:
        return {}

def show_metrics():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  Apollo v0.5.0 Metrics — {now}")
    print(f"{'='*70}")

    # System pressure
    pressure = get_pressure()
    if pressure is not None:
        pct = 100 - pressure  # memorystatus_level is inverted (100=no pressure)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\n  Memory Pressure: {pct}%  [{bar}]")

    # VM stats
    vm = get_vm_stats()
    if vm:
        print(f"  Compressor: {fmt_bytes(vm.get('compressor_bytes', 0))} "
              f"({vm.get('compressor_pages', 0)} pages)")
        print(f"  Page-ins: {vm.get('pageins', 0):,}  Page-outs: {vm.get('pageouts', 0):,}")

    # Runtime metrics
    m = load_json(METRICS)
    if m:
        print(f"\n  ── Apollo Runtime ──")
        print(f"  Profile: {m.get('current_profile', '?')}")
        print(f"  Workload: {m.get('current_workload', '?')}")
        print(f"  Foreground: {m.get('foreground_app', '?')}")
        print(f"  Cycles: {m.get('cycle_count', 0):,}")
        print(f"  Throttles: {m.get('throttles_applied', 0):,}")
        print(f"  Freezes: {m.get('freezes_applied', 0):,}")
        print(f"  Unfreezes: {m.get('unfreezes_applied', 0):,}")
        print(f"  Boosts: {m.get('boosts_applied', 0):,}")
        skipped = m.get("top_skipped_processes", [])
        if skipped:
            print(f"  Low-value skipped: {', '.join(skipped[:5])}")

    # Overflow history
    oh = load_json(OVERFLOW)
    if oh:
        print(f"\n  ── Overflow Guard ──")
        print(f"  Total overflows (lifetime): {oh.get('total_overflows', 0)}")
        print(f"  Threshold offset: {oh.get('threshold_offset', 0):.3f}")
        events = oh.get("events", [])
        if events:
            last = events[-1]
            print(f"  Last overflow: pressure={last.get('memory_pressure', 0):.2f} "
                  f"cause={last.get('cause', '?')}")

    # Markov chain
    mk = load_json(MARKOV)
    if mk:
        print(f"\n  ── Markov Chain ──")
        print(f"  Total transitions: {mk.get('total_transitions', 0):,}")
        transitions = mk.get("transitions", {})
        print(f"  Tracked apps: {len(transitions)}")
        # Show top 3 most-used source apps
        top_sources = sorted(
            transitions.items(),
            key=lambda x: sum(t["count"] for t in x[1].values()),
            reverse=True
        )[:3]
        for src, targets in top_sources:
            total = sum(t["count"] for t in targets.values())
            top_target = max(targets.items(), key=lambda x: x[1]["count"])
            prob = top_target[1]["count"] / total * 100
            print(f"    {src} → {top_target[0]} ({prob:.0f}%, {total} obs)")

    # Holt-Winters
    hw = load_json(HOLTWINTERS)
    if hw:
        print(f"\n  ── Holt-Winters Seasonal ──")
        print(f"  Level: {hw.get('level', 0):.3f}  "
              f"Trend: {hw.get('trend', 0):+.4f}/h  "
              f"Observations: {hw.get('observations', 0)}")
        seasonal = hw.get("seasonal", [1.0] * 24)
        hour_now = datetime.now().hour
        # Show seasonal factors for nearby hours
        hours_display = [(hour_now + i) % 24 for i in range(-2, 4)]
        factors = " ".join(
            f"{'→' if h == hour_now else ' '}{h:02d}:{seasonal[h]:.2f}"
            for h in hours_display
        )
        print(f"  Seasonal: {factors}")

    # Frozen state
    fs = load_json(FROZEN)
    if fs:
        frozen = fs.get("frozen", [])
        if frozen:
            print(f"\n  ── Frozen Processes ({len(frozen)}) ──")
            for f in frozen[:5]:
                print(f"    PID {f.get('pid', '?')} since {f.get('since', '?')}")
        else:
            print(f"\n  ── No frozen processes ──")

    # Governor
    gov = load_json(GOVERNOR)
    if gov:
        print(f"\n  ── Governor ──")
        print(f"  Profile: {gov.get('profile', '?')}")
        print(f"  Reason: {gov.get('transition_reason', '?')}")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("⚠  Run as root for full metrics: sudo python3 scripts/apollo-metrics.py")

    if "--watch" in sys.argv:
        try:
            while True:
                os.system("clear")
                show_metrics()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_metrics()
