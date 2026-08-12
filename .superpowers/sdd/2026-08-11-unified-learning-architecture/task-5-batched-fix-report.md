# Task 5 Batched Safety-Fix Report

Date: 2026-08-11
Scope: mechanical finalization of the approved one-batch response to
`task-5-batched-review.md`

## Review findings and defenses

### Critical 1: stale or fail-open daemon safety gates

The daemon now builds a fresh exploration environment through
`capture_exploration_environment` in `src/bin/apollo-optimizerd/main.rs`.
The snapshot reads the stop flag, kill-switch path, CoreAudio state, media
probe state, hazard probability, metrics, thermal state, circuit-breaker
state, speculation policy, build phase, and a current wall/monotonic time
point. The old ten-second Markov gate cache and the hard-coded circuit and
kill-switch values were removed.

All three production seams receive an environment closure: Markov, the
InteractionQos lease path, and Boost counterfactual dispatch. Each seam takes
a request snapshot and performs a second fresh snapshot through
`ExplorationScheduler::recheck` immediately before its first effect. The
target-specific recheck also refreshes target identity, protected-process and
ownership fields. A failed late check cancels the reservation and leaves the
ordinary safe path available.

### Critical 2: wall-only key cooldowns and restart loss of monotonic time

`PersistedCooldown` and `PersistedCommit` now carry wall seconds, monotonic
seconds, and a stable boot identifier. Admission requires both wall and
same-boot monotonic elapsed time for the 900-second global interval and the
86,400-second key cooldown. Wall rollback and forward jumps therefore cannot
shorten a same-boot deadline. `system_boot_session` uses macOS boot identity
and continuous Mach time, so daemon restart on the same OS boot does not reset
the monotonic barrier. Across a reboot, wall time remains the documented trust
boundary.

The focused contract tests cover exact 899/900 and 86,399/86,400 boundaries,
wall rollback, forward jumps, and same-boot restore.

### Critical 3: PID reuse during exploratory member mutation

Markov coalition members retain the captured `ProcessIdentity` and call
`reverify_member_identity` immediately before each member's cache, Jetsam,
Mach-tier, task-QoS, or related mutation. A mismatch emits a blocked member
event, marks the exploration stale, and prevents credit; any partial lease is
released through the existing owner.

InteractionQos similarly captures each member identity, rechecks before I/O
promotion, and rechecks again immediately before task-QoS mutation. Release
also verifies identity before rollback and records a blocked rollback when a
PID has been recycled. Existing effect ownership and release paths remain in
charge of actual mutations.

### Important 1: selection was not cycle-global or preference ordered

The daemon now begins one scheduler cycle before Markov work, runs the Markov
seam first, admits InteractionQos before dispatch, and evaluates Boost
omission last. This makes the frozen preference order operational across the
three seams without adding a candidate queue. Each seam submits the matched
natural observation candidate when available alongside its mutable candidate;
the scheduler selects deterministically, tracks the cycle-wide candidate
count, enforces the 4-per-family and 12-total limits, and the global interval
prevents a later seam from consuming a second mutable slot.

The scheduler selection tuple is stable and independent of PID, hash order,
randomness, or wall-clock subsecond detail. Natural observation returns before
budget, cooldown, or reservation mutation.

### Important 2: terminal cancellation metadata was dropped

`DecisionLedger::ingest_cycle_events` now retains terminal exploration metadata
and merges it into the correlated pending envelope after validating the
correlation, family, and canonical key. A terminal committed bit is preserved
and a terminal cancellation diagnostic replaces the earlier clean metadata.
The normal ledger still allocates the single DecisionId and settles the single
episode. The focused ledger test proves a release failure reaches the settled
episode and makes it ineligible for authority.

### Important 3: hostile scheduler payloads were bounded too late

Scheduler collections now use bounded serde visitors for cooldowns, commits,
terminal deduplication, and the reserved string. Over-cap sequences and
oversized strings fail during typed deserialization, before an unrestricted
`serde_json::Value` tree is materialized. `LearnedState::load` now probes the
version and parses directly into the typed `LearnedState`, allowing those
field-level bounds to fire before scheduler state is constructed. The existing
64 KiB serialized-state validation remains in place, and the learned-state
test proves over-cap and oversized primary files fall back to `.previous`.

### Minor 1: malformed requests did not return the documented first blocker

`select` and `recheck` now run `first_safety_gate_blocker` before capacity,
origin, family, or key-specific policy checks. The scheduler test combines
multiple faults and verifies the lifecycle blocker wins, while the matrix
covers the remaining documented order and exact pressure/hazard boundaries.

### Cannot Verify 1: tests and formatting were not run in the review

The controller subsequently supplied fresh verification evidence for this
batch:

- `cargo test -p apollo-engine --lib`: 2664 passed, 0 failed, 1 ignored.
- `cargo test --bin apollo-optimizerd`: 254 passed, 0 failed.
- `cargo fmt --all -- --check`: exit 0.
- `git diff --check`: exit 0.
- Focused original-fix evidence: engine 28 passed; daemon 3 passed.

No Cargo command was run during this mechanical finalization, per instruction.

### Cannot Verify 2: imported-M1 provenance in a production restore path

The scheduler restore API now distinguishes `Local`, `Unknown`, and
`ImportedM1` sources. Any non-local source, unknown origin, installation
change, or hardware mismatch cold-starts the scheduler and clears mutable
state. The focused restore test covers matching restart, changed hardware,
unknown provenance, and imported M1 state. The production daemon currently
uses the local restore context because its learned-state path is local; this
batch does not invent an external import path. The API therefore fails closed
if such provenance is supplied, but end-to-end imported-state provenance is
not exercised by the daemon.

## Verification and baseline limitation

The approved diff contains nine Task 5 source/test files. The known full
engine-test limitation is unrelated to this batch: in baseline commit
`89da703`, `git show 89da703:tests/level2_integration.rs` shows an unedited
`execute_actions` call at line 63 with 12 arguments, while
`git show 89da703:crates/apollo-engine/src/engine/execute_actions.rs` already
shows the 13-parameter signature, including `async_commands`. Therefore
`cargo test -p apollo-engine` without `--lib` reaches a pre-existing compile
gap in that integration test. The test was not edited.

## Deployment and uncertainty

There was no deployment, daemon start, Task 6 start, workspace release pass,
Clippy pass, E2E run, or performance measurement. The evidence is focused
unit/daemon verification plus the supplied affected engine and daemon passes.
The remaining uncertainty is limited to live macOS effect timing and
environment transitions, full-workspace integration beyond the documented
baseline gap, and end-to-end imported-M1 provenance. No production authority,
Gold admission, or Task 6 metrics surface was changed by this batch.
