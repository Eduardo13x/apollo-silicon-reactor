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
LOG = "/var/lib/apollo/audit.log"
FLAG = "/var/run/apollo-audit-alert"  # presence = HIGH finding pending


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def audit(state, metrics, policy):
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

    # ── P0: file unreadable / corrupt ─────────────────────────────────────
    if "__error__" in state:
        findings.append(("HIGH", "state-unreadable", state["__error__"]))

    return findings


def main():
    state = load(STATE)
    metrics = load(METRICS)
    policy = load(POLICY)
    findings = audit(state, metrics, policy)

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
