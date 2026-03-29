# Apollo AutoResearch — Program

> Following Karpathy's AutoResearch (2025) principles exactly.
> "The human writes research directions, the machine executes."

## The Contract

You are an autonomous research agent improving Apollo's **decision quality** —
how well the AdaptiveGovernor decides what to do with each process for both
memory optimization AND performance protection.

### The One File

You may ONLY modify: **`src/engine/adaptive_governor.rs`**

This is your `train.py`. Everything else is off-limits.

### The Fixed Benchmark

`tests/prepare.rs` contains 20 scenario tests (s01–s20) that simulate real
system states and check whether AdaptiveGovernor makes the correct decision.
**You must NEVER modify this file.**

Scenarios cover:
- Performance protection (foreground, essential, compilers, GUI apps)
- Memory pressure response (stale, telemetry, zombies)
- Workload awareness (build mode, video, idle)
- Resource efficiency (don't over-throttle when pressure is low)
- App helper safety (Chrome helpers crash on SIGSTOP)
- Utility/waste scoring sanity

### The Metric

```
score = scenarios_passed * 50     [DOMINANT: each correct decision = 50 pts]
      + all_tests_passed          [regression gate]
      - clippy_warnings * 5
      - binary_bloat * 0.01
```

**Higher scenario score = better decisions = better optimizer.**

### The Loop

```
LOOP FOREVER:
  1. Read adaptive_governor.rs and recent results
  2. Modify adaptive_governor.rs with ONE experimental idea
  3. git commit
  4. Run: bash ./autoresearch/evaluate.sh
  5. Parse: SCORE, SCENARIOS, PASS
  6. If SCORE improved → KEEP (advance branch)
  7. If SCORE same or worse → DISCARD (git reset --hard HEAD~1)
  8. Log result to autoresearch/results.tsv
  9. Go to 1
```

### Rules

1. **Only modify `src/engine/adaptive_governor.rs`** — never touch tests, evaluate.sh, or prepare.rs.
2. **One experiment per iteration** — a single focused change.
3. **~5 min per experiment** — target 12 experiments/hour.
4. **PASS=1 mandatory** — if PASS=0, revert immediately.
5. **SCORE must improve** — equal score only kept if code is simpler.
6. **Log everything** in `autoresearch/results.tsv`.
7. **Never stop** — run until interrupted.
8. **Simplicity bias** — removing code for equal results is a WIN.

### Results Log Format (TSV)

```
commit	score	scenarios	tests	status	description
```

- `keep` = score improved, committed
- `discard` = score same or worse, reverted
- `crash` = build/test failed, reverted

## Research Directions

These are the improvement axes for adaptive_governor.rs. Each should produce
measurable scenario score gains:

### 1. Smarter Utility Scoring
- Factor in RSS size (large background = higher freeze priority)
- Factor in compression ratio (high compression = cheap to freeze)
- Weight process age (older stale = more confident freeze)
- Consider pageins rate (high pageins = actively used despite low CPU)

### 2. Better Workload Detection
- Detect build mode from process names (rustc, cargo, clang, make)
- Protect compilers from throttle/freeze during builds
- Detect video/audio workloads from helper process patterns
- Adaptive thresholds per workload type

### 3. Improved Tier Classification
- Better heuristics for distinguishing interactive vs background
- Network activity as a stronger signal for utility
- Mach port count as a proxy for IPC activity
- Process group awareness (helpers inherit parent tier)

### 4. Decision Edge Cases
- Gray zone tie-breaking improvements
- Hysteresis to prevent rapid throttle/unthrottle oscillation
- AppHelper protection refinements (detect audio/video better)
- Ephemeral process detection improvements

### 5. Waste Detection Improvements
- Better waste scoring formula (CPU × time × !GUI)
- RSS-weighted waste (large background processes = higher waste)
- Cumulative waste tracking (process consistently wasteful → stronger signal)

## Anti-Patterns (DO NOT)

- Do NOT modify any file except `src/engine/adaptive_governor.rs`
- Do NOT modify tests or evaluation harness
- Do NOT add dependencies
- Do NOT add features unrelated to decision quality
- Do NOT make multiple unrelated changes in one experiment
