export const meta = {
  name: 'apollo-regression-fix-loop',
  description: 'Closed improvement loop for Apollo: run the regression probe, diagnose each finding to its ROOT cause in the code, classify SAFE vs RISKY, and propose a concrete patch per finding. Does NOT auto-apply — surfaces a ranked remediation plan. Honors the project doctrine that gating/suppression/calibration fixes are never blind-applied.',
  phases: [
    { title: 'Probe', detail: 'run apollo-regression-probe.py --json' },
    { title: 'Diagnose', detail: 'per finding: confirm in prod + root-cause in code + classify + propose patch' },
    { title: 'Plan', detail: 'ranked remediation plan: safe quick-wins vs human-gated risky' },
  ],
}

const REPO = '/Users/eduardocortez/proyectos/system-optimizer'
const PROBE = '/usr/local/libexec/apollo-regression-probe.py'

const DOCTRINE = `
You are diagnosing apollo-silicon-reactor (macOS optimization daemon, Rust, ${REPO}).
This project has an EXPENSIVE history of regressions from blind fixes. Classify every proposed fix:

SAFE (auto-appliable after the deploy-gate keeps/reverts on AIS):
- telemetry plumbing (LSE counter -> sync_from_lockfree -> RuntimeMetrics field)
- doc/comment corrections (lying TODOs)
- ADDITIVE protection guards (adding is_protected_name / is_protected_pid / foreground|visible checks BEFORE a freeze/throttle/boost/demote — can only REDUCE action, never strangle)
- operational resets (e.g. resetting a tampered sysctl back to the macOS default)
- enforcing a cap at the MUTATION point (not changing the cap value)

RISKY (NEVER blind-apply — propose + surface for human design + >=500-obs validation):
- changing any GATING / SUPPRESSION threshold or window (purge/freeze/demote gates, high_bw gate, survival floors)
- changing CALIBRATION (debias clamp, drift baseline, avg_delta thresholds, RL bands)
- loosening a "starving gate" whose consumer you have NOT verified
- anything where over-firing/under-firing changes daemon behavior under load

THE SCARS each probe code maps to (root-cause hints):
- purge-strangled* -> f770478/44266d8: suppression must yield to survival (transient gate + pressure escape). RISKY to retune.
- belief-cap-broken / ls-size* / p95-crisis-bloat -> 22eb524: cap enforced at mutation. The cap-at-mutation fix is SAFE; raising the cap value is RISKY.
- refault-storm -> 667f7d1/dcba17e: mostly inherent 8GB; suppression already conservative. Usually NOT a code fix.
- meet-network-tamper -> 5802c89: the 4 TCP/IPC sysctls must equal macOS defaults; if not, RESET them (operational, SAFE) and check the emit_sysctl hard-block didn't regress.
- boost-loop / boost-churn -> 4ae0f27: boost must require foreground|visible. If recurring post-fix, find the bypass path (ADDITIVE guard = SAFE).
- mediation-break-* (LANDED on protected) -> 03472d7/6e0e1ce/a98b33a: a path bypassed safety.rs. Adding is_protected_name to that path is SAFE (additive).
- mediation-nominate-* (failed) -> minor gap; the decision path chose a protected target but nothing landed. Low urgency.
- llm-stale -> d6c659b/4b6da93/3df9235: teacher dead. Check metal-oom gate, MLX server alive, config. Operational.
- ais-degraded / daemon-failures -> read last_error + journal; could be anything.

Production state to confirm against (read-only): /var/lib/apollo/{runtime_metrics.json,journal.jsonl,learned_state.json,llm_state.json}; live sysctls; ${REPO} source.
`

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'code', 'detail'],
        properties: {
          severity: { type: 'string' }, code: { type: 'string' }, detail: { type: 'string' },
        },
      },
    },
  },
}

const DIAG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['code', 'still_present', 'root_cause', 'file', 'classification', 'proposed_patch', 'risk', 'confidence'],
  properties: {
    code: { type: 'string' },
    still_present: { type: 'boolean', description: 'true if the condition is confirmed LIVE right now (not pre-fix journal residue / transient load)' },
    root_cause: { type: 'string', description: 'the actual root cause in the code or operationally — not the symptom' },
    file: { type: 'string', description: 'file:line of the root, or "operational" / "transient-load" / "inherent-hardware"' },
    classification: { type: 'string', enum: ['SAFE', 'RISKY', 'NO-ACTION'] },
    proposed_patch: { type: 'string', description: 'concrete patch: what to change + the guard/value + a test. For NO-ACTION explain why (transient/inherent/already-correct).' },
    risk: { type: 'string', description: 'what could regress if applied; for RISKY name the validation needed (>=500 obs etc.)' },
    confidence: { type: 'number' },
  },
}

phase('Probe')
const probe = await agent(
  `Run the Apollo regression probe and return its findings. Execute exactly:\n  sudo python3 ${PROBE} --json\nParse the JSON it prints (it has {timestamp, findings:[{severity,code,detail}], metrics}). Return the findings array. If it prints nothing or errors, return an empty findings array. Do not invent findings.`,
  { label: 'run-probe', phase: 'Probe', schema: FINDINGS_SCHEMA }
)

const findings = (probe && probe.findings) || []
if (!findings.length) {
  log('Probe is clean — no regressions to diagnose.')
  return { clean: true, findings: [], diagnoses: [], plan: 'No findings. System healthy.' }
}
log(`Probe surfaced ${findings.length} finding(s). Diagnosing each to root cause...`)

phase('Diagnose')
const diagnoses = (await parallel(findings.map((f) => () =>
  agent(`${DOCTRINE}\n\n=== DIAGNOSE THIS PROBE FINDING ===\n${JSON.stringify(f)}\n\nSteps: (1) Confirm it is LIVE right now — read the current /var/lib/apollo state / live sysctls; rule out pre-fix journal residue (check timestamps vs the last daemon restart), transient CPU load (a high p95/thrashing under heavy build/workflow load is NOT a regression), and inherent 8GB hardware limits. Set still_present accordingly. (2) If live, find the ROOT cause in the code (file:line), not the symptom. (3) Classify SAFE / RISKY / NO-ACTION per the doctrine. (4) Propose a concrete patch (or explain NO-ACTION). Be conservative: when unsure, classify RISKY and demand validation.`,
    { label: `diagnose:${f.code}`, phase: 'Diagnose', agentType: 'Explore', schema: DIAG_SCHEMA })
    .catch(() => null)))).filter(Boolean)

const live = diagnoses.filter((d) => d.still_present)
const safe = live.filter((d) => d.classification === 'SAFE')
const risky = live.filter((d) => d.classification === 'RISKY')
log(`${live.length} live, ${diagnoses.length - live.length} stale/transient. ${safe.length} SAFE, ${risky.length} RISKY.`)

phase('Plan')
const plan = await agent(
  `${DOCTRINE}\n\nProduce a tight remediation plan from these diagnoses. Sections:\n` +
  `1. STALE/TRANSIENT/INHERENT (still_present=false) — list briefly so the maintainer knows the probe fired on residue/load, not a real regression. These are also signals the PROBE may need a detector refinement (note which).\n` +
  `2. SAFE quick-wins — ordered; for each: root cause (file:line), the exact patch, the test, and the verification commands. Verification is TWO-STAGE: (i) ./scripts/apollo-deploy-gate.sh deploys + keeps/reverts on its own AIS>=87 baseline; (ii) THEN sudo bash scripts/apollo-accept-gate.sh (the brutal acceptance gate: hard SLOs H1-H5 + no-regression-vs-rolling-baseline R1/R3 + composite S) as an ADVISORY check — if it REJECTs (exit nonzero), the change is SURFACED with the failing criteria and a revert recommended (human pulls the trigger; auto-revert is OFF). A SAFE change is only confirmed-kept when BOTH the deploy-gate keeps AND apollo-accept-gate.sh ACCEPTs.\n` +
  `3. RISKY — surfaced for human design ONLY. For each: root cause, why it is risky, and the validation required (>=500 obs, watch which probe metric / cron detector). DO NOT present these as ready-to-apply.\n` +
  `4. Single highest-value next action and why.\n\nDIAGNOSES:\n${JSON.stringify(diagnoses, null, 1)}`,
  { label: 'remediation-plan', phase: 'Plan' }
)

return { findings, diagnoses, safe_count: safe.length, risky_count: risky.length, plan }
