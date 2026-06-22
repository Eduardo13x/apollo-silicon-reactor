#!/usr/bin/env python3
"""apollo-learned-state-audit — cheap, deterministic pathology detector for
Apollo's persisted brain (learned_state.json) + live metrics.

Runs from a launchd cron (root) every few hours. Reads the JSON state, checks
a catalog of KNOWN pathologies (each one a real incident from the 2026-06
sessions), and writes a verdict to a log. On any HIGH finding it sets a flag
file so a human / Claude knows to look. It NEVER edits state — detection only;
the fix is reasoned on-demand (supervision doctrine: the human is gatekeeper).

Exit code: 0 = clean, 1 = findings (severity in the report).
"""
import json
import sys
import os
from datetime import datetime, timezone

STATE = "/var/lib/apollo/learned_state.json"
METRICS = "/var/lib/apollo/runtime_metrics.json"
POLICY = "/var/lib/apollo/learned_policy.json"
LLM = "/var/lib/apollo/llm_state.json"
LOG = "/var/lib/apollo/audit.log"
FLAG = "/var/run/apollo-audit-alert"  # presence = HIGH finding pending

# Teacher is considered dead if enabled but no SUCCESSFUL call in this long.
LLM_STALE_DAYS = 2.0


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def teacher_config_disabled():
    """True iff the [llm] config explicitly sets enabled=false. Mirrors the
    regression probe's check (de1a2cd): the llm_state.json `enabled` flag can be
    stale-true after the teacher is deliberately disabled in config, so the
    llm-stale detector must consult the config (the source of truth the daemon
    reads per-tick), not just the runtime state. Stdlib line parse (no tomllib on
    py3.9). Conservative: returns False on any read failure so a genuinely-dead
    teacher is never masked."""
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


def audit(state, metrics, policy, llm):
    findings = []  # (severity, code, detail)

    # The learned policy lists. A futile weight is only DANGEROUS when the
    # process sits in `interactive` — that is the exact Brave-0607 shape:
    # low effectiveness + interactive classification → RL boosts it → loop.
    # Futile-and-protected / futile-and-noise are harmless (parked / throttled).
    interactive = set(policy.get("interactive_patterns", []))
    protected = set(policy.get("protected_patterns", []))

    # ── P1: boost-loop shape (the Brave-0607 incident) ─────────────────────
    ot = state.get("outcome_tracker", {})
    weights = ot.get("weights", {})
    for name, w in weights.items():
        if isinstance(w, dict):
            t, e = w.get("throttle_count", 0), w.get("effective_count", 0)
        elif isinstance(w, list) and len(w) >= 2:
            t, e = w[0], w[1]
        else:
            continue
        if t < 40:
            continue
        eff = (e + 1) / (t + 2)
        if eff < 0.08 and name in interactive:
            # The dangerous case: futile AND boostable.
            findings.append((
                "HIGH", "boost-loop-risk",
                f"{name}: {t} actions at {eff:.1%} eff but classified INTERACTIVE "
                f"— Brave-0607 boost-loop shape (futile + boostable)"))
        elif eff < 0.08 and name not in protected:
            # Mild: futile, not protected, not interactive — wasted budget,
            # no loop. A teach pass could park it; not urgent.
            findings.append((
                "LOW", "weight-futile",
                f"{name}: {t} throttles at {eff:.1%} eff, unprotected — "
                f"wasting budget (move to protected on next teach)"))

    # ── P2: debias clamp saturation (calibration loop-closure aftermath) ───
    # If a subsystem's actual/predicted ratio sits at the 0.25 floor, the
    # predictor over-promises >4x and the online debias can't fully correct —
    # a STRUCTURAL fix (tighten clamp / retrain) is needed, not a weight nudge.
    mc = state.get("meta_cognition", {})
    for sid, acc in mc.get("subsystems", []):
        pred = acc.get("predicted_ema", 0.0)
        act = acc.get("actual_ema", 0.0)
        obs = acc.get("observations", 0)
        if obs >= 500 and pred > 0.05:
            ratio = act / pred
            if ratio <= 0.26:  # at/below the 0.25 clamp floor
                findings.append((
                    "MED", "debias-saturated",
                    f"{sid}: predicts {pred:.3f} delivers {act:.3f} "
                    f"(ratio {ratio:.2f} at clamp floor, {obs} obs) — "
                    f"structural over-promise the online debias can't fix"))

    # ── P3: humble_mode latched (hallucinated-calibration shape) ───────────
    if mc.get("humble_mode") is True:
        ce = mc.get("calibration_error", 0.0)
        findings.append((
            "MED", "humble-latched",
            f"humble_mode ON, calibration_error={ce:.3f} — global damper "
            f"active; check if a subsystem gap is real or a measurement bug"))

    # ── P4: restore quality low (RestoreQualityMonitor would reset zones) ──
    lrq = state.get("last_restore_quality")
    if isinstance(lrq, (int, float)) and 0 < lrq < 0.40:
        findings.append((
            "LOW", "restore-quality-low",
            f"last_restore_quality={lrq:.2f} — restored state looks stale; "
            f"zones may reset to defaults (loss of learning)"))

    # ── P5: live metrics sanity (the cheap cross-check) ───────────────────
    if "__error__" not in metrics:
        fails = metrics.get("failures", 0)
        if fails and fails > 0:
            findings.append(("HIGH", "daemon-failures",
                             f"failures={fails}, last_error={metrics.get('last_error')}"))
        ais = metrics.get("ais_score", 0)
        if 0 < ais < 80:
            findings.append(("HIGH", "ais-degraded", f"ais_score={ais:.1f} (<80)"))

        # ── P7: purge strangled (2026-06-15 regression blind spot) ──────────
        # That incident — Apollo suppressing the maintenance purge 127,959× vs
        # 147 fired, thrashing climbing to 69k with no relief — was invisible to
        # BOTH ais (read 93) and this auditor. Catch the exact shape: crisis-
        # level thrashing WHILE purge is being suppressed (skips dominate). Each
        # signal alone is benign (a brief storm spikes thrashing; idle systems
        # skip purge), so require BOTH. Counters are cumulative-since-start, so
        # this fires on a sustained pattern, not a transient spike.
        thrash = metrics.get("thrashing_score", 0)
        skipped = metrics.get("maintenance_purge_skipped_idle_total", 0)
        purged = metrics.get("maintenance_purge_total", 0)
        skip_ratio = skipped / (skipped + purged) if (skipped + purged) > 0 else 0.0
        if thrash >= 50000 and skip_ratio > 0.9 and skipped > 1000:
            findings.append((
                "HIGH", "purge-strangled",
                f"thrashing={thrash:.0f} (crisis) but purge suppressed "
                f"{skipped}/{skipped + purged} ({skip_ratio:.0%} skip) — relief "
                f"is being strangled; check is_high_bw_workload_active survival "
                f"escape (see feedback_suppression_survival_first)"))

    # ── P0: file unreadable / corrupt ─────────────────────────────────────
    if "__error__" in state:
        findings.append(("HIGH", "state-unreadable", state["__error__"]))

    # ── P6: teacher LLM silently dead ─────────────────────────────────────
    # The 2026-06-14 incident: the teacher had not made a SUCCESSFUL call for
    # 11 days (metal-oom gate permanently tripped, then thinking truncated the
    # JSON), yet `calls_today` / `trigger_events` kept advancing on every
    # attempt — so it LOOKED alive. Trust only ground truth: `last_call_at`
    # (set sole­ly on success) and `consecutive_failures`. Never `calls_today`.
    if "__error__" not in llm and llm.get("enabled") and not teacher_config_disabled():
        cf = llm.get("consecutive_failures", 0)
        last_call = llm.get("last_call_at")
        age_days = None
        if last_call:
            try:
                lc = datetime.fromisoformat(last_call.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - lc).total_seconds() / 86400.0
            except Exception:
                age_days = None
        if age_days is None or age_days > LLM_STALE_DAYS:
            shown = "never" if age_days is None else f"{age_days:.1f}d ago"
            findings.append((
                "HIGH", "llm-stale",
                f"teacher enabled but last SUCCESSFUL call {shown} "
                f"(consecutive_failures={cf}, last_error={llm.get('last_error')}) "
                f"— ignore calls_today/triggers, they advance on failed attempts"))
        elif cf >= 10:
            findings.append((
                "MED", "llm-failing",
                f"teacher calling but failing: consecutive_failures={cf}, "
                f"last_error={llm.get('last_error')}"))

    return findings


def main():
    state = load(STATE)
    metrics = load(METRICS)
    policy = load(POLICY)
    llm = load(LLM)
    findings = audit(state, metrics, policy, llm)

    ts = datetime.now(timezone.utc).isoformat()
    high = [f for f in findings if f[0] == "HIGH"]
    if findings:
        verdict = f"{len(findings)} finding(s)" + (f" — {len(high)} HIGH" if high else "")
    else:
        verdict = "clean"

    line = f"[{ts}] {verdict}"
    for sev, code, detail in findings:
        line += f"\n    {sev:4} {code}: {detail}"

    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        print(line)  # fallback if log not writable

    # HIGH findings raise the flag; clean runs clear it.
    try:
        if high:
            with open(FLAG, "w") as f:
                f.write(line)
        elif os.path.exists(FLAG):
            os.remove(FLAG)
    except Exception:
        pass

    print(line)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
