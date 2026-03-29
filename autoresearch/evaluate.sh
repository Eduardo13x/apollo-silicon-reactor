#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo AutoResearch — Evaluation Harness (Karpathy-style)
# ══════════════════════════════════════════════════════════════════════════════
#
# THIS FILE IS READ-ONLY.  The agent must NEVER modify it.
#
# Measures DECISION QUALITY, not code health.
# The ground truth is prepare.rs: 20 scenario tests that check whether
# AdaptiveGovernor makes correct memory + performance decisions.
#
# Metric: decision_score = scenarios_passed / total_scenarios
#         Higher is better. Range [0.0, 1.0].
#
# Also tracks:
#   - All tests still pass (no regressions)
#   - Clippy clean (code quality gate)
#   - Binary size (bloat penalty)
#
# Output: SCORE=<float> SCENARIOS=<passed>/<total> TESTS=<int> CLIPPY=<int> SIZE_KB=<int> PASS=<0|1>
#
# PASS=1 requires: non-scenario tests pass AND clippy clean AND compiles.
# Scenario failures are the TARGET to improve, not a gate.
#
# Score formula:
#   score = scenarios_passed * 50           [each correct decision = 50 points]
#         + all_tests_passed                [base: regression gate]
#         - clippy_warnings * 5
#         - max(0, (binary_kb - baseline_kb)) * 0.01
#
# This makes scenarios_passed the DOMINANT factor (~1000 pts for 20 scenarios)
# while still gating on regressions and code quality.
#
# Baseline (frozen):
BASELINE_TESTS=1370
BASELINE_SIZE_KB=4055
TOTAL_SCENARIOS=20
# ══════════════════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.."

T0=$(python3 -c 'import time; print(time.time())')

# ── Step 1: Build ─────────────────────────────────────────────────────────────
if ! cargo build --release 2>/dev/null; then
    echo "SCORE=0 SCENARIOS=0/$TOTAL_SCENARIOS TESTS=0 CLIPPY=999 SIZE_KB=0 TIME_S=0 PASS=0"
    exit 0
fi

# ── Step 2: Scenario tests (prepare.rs) ──────────────────────────────────────
# Run ONLY the scenario tests from prepare.rs
SCENARIO_OUTPUT=$(cargo test --test prepare 2>&1 || true)

# Count passed scenarios (each test is named s01_, s02_, etc.)
SCENARIOS_PASSED=$(echo "$SCENARIO_OUTPUT" | grep -cE 'test scenarios::s[0-9]+.*ok$' || echo 0)
SCENARIOS_PASSED=${SCENARIOS_PASSED:-0}
SCENARIOS_FAILED=$(echo "$SCENARIO_OUTPUT" | grep -cE 'test scenarios::s[0-9]+.*FAILED' || echo 0)
SCENARIOS_FAILED=${SCENARIOS_FAILED:-0}

# ── Step 3: Non-scenario tests (regression gate) ────────────────────────────
# Exclude prepare.rs scenarios — those are the optimization target, not a gate.
TEST_OUTPUT=$(cargo test --lib --bins 2>&1 || true)

TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oE 'cargo test: ([0-9]+) passed' | grep -oE '[0-9]+' | head -1)
if [ -z "$TESTS_PASSED" ]; then
    TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | paste -sd+ - | bc 2>/dev/null || echo 0)
fi
TESTS_PASSED=${TESTS_PASSED:-0}

TESTS_FAILED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
TESTS_FAILED=${TESTS_FAILED:-0}

# ── Step 4: Clippy ────────────────────────────────────────────────────────────
CLIPPY_OUTPUT=$(cargo clippy --all-targets 2>&1 || true)

CLIPPY_WARNINGS=$(echo "$CLIPPY_OUTPUT" | grep -oE 'cargo clippy:.* ([0-9]+) warnings' | grep -oE '[0-9]+ warnings' | grep -oE '[0-9]+' | head -1)
if [ -z "$CLIPPY_WARNINGS" ]; then
    CLIPPY_WARNINGS=$(echo "$CLIPPY_OUTPUT" | grep -c 'warning\[' || true)
fi
CLIPPY_WARNINGS=${CLIPPY_WARNINGS:-0}

# ── Step 5: Binary size ──────────────────────────────────────────────────────
BINARY="target/release/apollo-optimizerd"
if [ -f "$BINARY" ]; then
    SIZE_BYTES=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY" 2>/dev/null || echo 0)
    SIZE_KB=$((SIZE_BYTES / 1024))
else
    SIZE_KB=0
fi

# ── Step 6: Compute score ─────────────────────────────────────────────────────
T1=$(python3 -c 'import time; print(time.time())')
TIME_S=$(python3 -c "print(round($T1 - $T0, 1))")

# Gate: PASS requires all tests pass, clippy clean, and no scenario regressions.
if [ "$TESTS_FAILED" -eq 0 ] && [ "$CLIPPY_WARNINGS" -eq 0 ] && [ "$TESTS_PASSED" -gt 0 ]; then
    PASS=1
else
    PASS=0
fi

SCORE=$(python3 -c "
scenarios = $SCENARIOS_PASSED
tests = $TESTS_PASSED
clippy = $CLIPPY_WARNINGS
size_kb = $SIZE_KB
baseline_size = $BASELINE_SIZE_KB

scenario_score = scenarios * 50
bloat_penalty = max(0, (size_kb - baseline_size)) * 0.01
score = scenario_score + tests - clippy * 5 - bloat_penalty

print(round(score, 2))
")

echo "SCORE=$SCORE SCENARIOS=$SCENARIOS_PASSED/$TOTAL_SCENARIOS TESTS=$TESTS_PASSED CLIPPY=$CLIPPY_WARNINGS SIZE_KB=$SIZE_KB TIME_S=$TIME_S PASS=$PASS"
