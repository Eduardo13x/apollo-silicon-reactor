# Apollo AutoResearch — Program

> Inspired by Karpathy's AutoResearch (2026).
> "The human's job shifts from writing code to writing research directions."

## The Contract

You are an autonomous research agent improving Apollo, a macOS system optimizer.

### Rules

1. **Modify only `src/`** — engine modules, daemon, CLI. Never touch `autoresearch/evaluate.sh`.
2. **One experiment per iteration** — a single focused change, not a sprawling refactor.
3. **Run `./autoresearch/evaluate.sh`** after every change. The output is your ground truth.
4. **PASS=1 is mandatory** — if PASS=0, revert immediately (`git checkout -- .`). No exceptions.
5. **SCORE must improve** — if SCORE ≤ previous best, revert. Equal score is only kept if the change simplifies code (fewer lines for same score).
6. **Log everything** in `autoresearch/results.tsv` — kept, discarded, AND crashed experiments.
7. **Commit kept experiments** with a descriptive message. Discarded experiments leave no git trace.
8. **Never stop** — if you run out of ideas, re-read the code, re-read results.tsv for patterns, try combining near-misses, try the opposite of what failed.
9. **Simplicity bias** — "A 0.001 improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome."

### The Metric

```
score = tests_passed
      - clippy_warnings * 5
      - max(0, binary_bloat_kb) * 0.01
      + new_tests * 0.5
```

**Lower clippy = better. More tests = better. Smaller binary = better. All tests must pass.**

### Time Budget

Each experiment should complete evaluation in < 3 minutes (build + test + clippy). If an experiment requires architectural changes that take longer to validate, break it into smaller steps.

## Research Directions

### Tier 1: High-Value (likely to improve score)

1. **Dead code removal** — Find functions/structs never called from any binary. Remove them. Score improves via smaller binary + potential clippy reduction. Candidates:
   - `optimizer.rs:optimize()` (confirmed dead)
   - `TelemetryLogger` (confirmed disabled)
   - Any `pub` function with 0 callers outside its own module

2. **Test coverage gaps** — Modules with 0 tests. Each new passing test adds +1 to score. Priority:
   - `src/engine/wait_graph.rs` — has functions but no unit tests
   - `src/engine/gpu_manager.rs` — only has Default impl test
   - `src/engine/page_reclaim.rs` — check if tested
   - `src/engine/process_tree.rs` — check if tested
   - `src/engine/coalition.rs` — check if tested

3. **Clippy fixes** — Each warning eliminated is +5 to score. Run `cargo clippy --all-targets` and fix everything it flags.

### Tier 2: Medium-Value (structural improvements)

4. **Redundant computation elimination** — Profile the hot path in `apollo-optimizerd.rs` for repeated work:
   - Multiple `lock_recover()` calls on the same mutex in the same scope
   - Repeated string formatting of the same process names
   - `collector.system().processes()` iterated multiple times

5. **Const promotion** — Find `let` bindings of literal values that should be `const`. Compiler can optimize better.

6. **Allocation reduction** — Find `Vec::new()` or `String::new()` in the hot loop that could be pre-allocated or reused across cycles.

### Tier 3: Exploratory (may or may not improve score)

7. **Algorithm improvements** — Better heuristics in existing decision code:
   - `behavioral_protection_score()` — can the formula be tighter?
   - `overflow_guard` thresholds — are they well-calibrated?
   - `rl_threshold` learning rate schedule — does EMA α converge?

8. **Error handling audit** — Find `.unwrap()` calls that should be `.unwrap_or()` or `?`. Not for score, but for daemon stability.

9. **Module consolidation** — If two small modules (<50 lines each) serve related purposes, merging them reduces cognitive overhead and may eliminate unused `pub` exports.

### Tier 4: Maintenance

10. **Dependency audit** — Are all Cargo.toml dependencies actually used? Unused deps increase compile time and binary size.

11. **Feature flag cleanup** — `tract-onnx` is optional and disabled. Are there other dead features?

## Anti-Patterns (DO NOT)

- Do NOT add features the user didn't ask for
- Do NOT refactor working code just because it's "ugly"
- Do NOT add comments, docstrings, or type annotations to code you didn't functionally change
- Do NOT add error handling for impossible scenarios
- Do NOT create new modules — extend existing ones
- Do NOT add dependencies
- Do NOT modify `autoresearch/evaluate.sh`
- Do NOT modify tests to make them pass (fix the code, not the test)
- Do NOT make multiple unrelated changes in one experiment

## How to Read results.tsv

```
experiment  branch  score_before  score_after  delta  status  description
```

- `kept` = score improved, committed
- `discarded` = score same or worse, reverted
- `crash` = build/test failed, reverted
- Look for patterns: which directions yield improvements? Which are dead ends?
