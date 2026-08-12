# Task 5 Finalization Report

## Scope

This finalization was checked against base `89da703` and the frozen Task 5
contract. No Task 6 work, deployment work, Cargo invocation, or production/test
edit was performed by the finalizer.

## Exact Changed Files

The implementation/evidence diff contains these 12 Task 5 paths:

- `crates/apollo-engine/src/engine/decision_ledger.rs`
- `crates/apollo-engine/src/engine/exploration_scheduler.rs`
- `crates/apollo-engine/src/engine/learned_state.rs`
- `crates/apollo-engine/src/engine/mod.rs`
- `crates/apollo-engine/src/engine/telemetry_medallion.rs`
- `crates/apollo-engine/tests/exploration_scheduler_contract.rs`
- `crates/apollo-engine/tests/level3_arousal_decay_stress.rs`
- `src/bin/apollo-optimizerd/daemon_cycle_tail.rs`
- `src/bin/apollo-optimizerd/daemon_dispatch_tick.rs`
- `src/bin/apollo-optimizerd/daemon_markov_tick.rs`
- `src/bin/apollo-optimizerd/learning_tick.rs`
- `src/bin/apollo-optimizerd/main.rs`

This report is the one additional finalization artifact, so the commit contains
13 paths relative to `89da703`.

## Evidence

Focused evidence reported by the original implementer:

- Scheduler contract: 21/21.
- Daemon scheduler: 3/3.
- Boost fallback: 1/1.

Fresh controller evidence supplied for this finalization:

- `/Users/edcrtz/.cargo/bin/cargo test -p apollo-engine --lib`: 2663 passed,
  0 failed, 1 ignored.
- `/Users/edcrtz/.cargo/bin/cargo test --bin apollo-optimizerd`: 251 passed,
  0 failed.

The raw output from the focused runs and any RED run was not retained. No RED
run is being claimed or reconstructed here.

## Scope And Safety Checks

The working-tree changes before this report matched the 12 implementation and
evidence paths above. The reviewed behavior is limited to the pure bounded
exploration scheduler, the three allowed families (`Boost`, `InteractionQos`,
and `MarkovPrewarm`), normal ledger correlation/metadata, existing medallion
holdout admission, existing QoS/Markov release owners, and one persisted
serde-defaulted scheduler field. No parallel decision allocator, synthetic
outcome path, scheduler-side kernel authority, or second persistence writer was
introduced.

`git diff --check` was clean for the implementation diff. The final staged
diff was checked again before commit.

## Persistence Caps

- Candidates: at most 4 per family and 12 per cycle.
- Active reservation: 1.
- Persisted cooldown keys: 256.
- Terminal correlation deduplication: 128.
- Serialized scheduler payload: 64 KiB.
- Global mutable-probe interval: 900 seconds.
- Same action/context cooldown: 86,400 seconds.
- Global commit window: at most 96 committed probes in the retained 24-hour
  window.
- Ledger diagnostic/detail text remains bounded to 160 bytes by the existing
  ledger boundary.

Live reservations, leases, process/member identity, monotonic state, and
medallion endpoints are not persisted. Invalid, future, oversized, duplicate,
over-cap, imported, or origin-mismatched state resets to a cold scheduler.

## Complexity

Selection examines at most 12 candidates and uses a bounded minimum-key scan;
it does not queue candidates or perform a full sort. Cooldown pruning scans at
most 256 entries, with average O(1) hash-map lookup for the selected key.
Terminal deduplication is also bounded at 128 entries. The fixed worst-case
work is therefore O(12 + 256), with all persisted collections capped.

## Known Warnings And Remaining Uncertainty

No warning lines were retained in the supplied controller or implementer
evidence, so no pre-existing warning can be identified or newly attributed
from that evidence. Cargo and formatting commands were not rerun during this
mechanical finalization, per the explicit instruction not to run Cargo.

Remaining uncertainty is limited to the absence of retained raw focused-test
output and the fact that the controller-supplied broad test results were not
rerun here. No blocker was found in the requested diff and whitespace checks.
