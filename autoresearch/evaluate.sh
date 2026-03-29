#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo AutoResearch — Immutable Evaluation Harness
# ══════════════════════════════════════════════════════════════════════════════
#
# THIS FILE IS READ-ONLY.  The agent must NEVER modify it.
# It is the equivalent of Karpathy's prepare.py: the fixed yardstick against
# which every experiment is measured.
#
# Outputs a single line to stdout:
#   SCORE=<float> TESTS=<int> CLIPPY=<int> SIZE_KB=<int> TIME_S=<float> PASS=<0|1>
#
# The score formula:
#   score = tests_passed
#         - clippy_warnings * 5
#         - max(0, (binary_kb - baseline_kb)) * 0.01
#         + (tests_passed - baseline_tests) * 0.5   [bonus for new tests]
#
# PASS=1 means: all tests passed AND clippy clean AND compiles.
# PASS=0 means: something is broken — experiment MUST be reverted.
#
# Baseline (frozen at system creation):
BASELINE_TESTS=1236
BASELINE_SIZE_KB=4056   # 4153808 bytes ≈ 4056 KB
# ══════════════════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.."

T0=$(python3 -c 'import time; print(time.time())')

# ── Step 1: Build ─────────────────────────────────────────────────────────────
if ! cargo build --release 2>/dev/null; then
    echo "SCORE=0 TESTS=0 CLIPPY=999 SIZE_KB=0 TIME_S=0 PASS=0"
    exit 0
fi

# ── Step 2: Tests ─────────────────────────────────────────────────────────────
# RTK may rewrite output.  Handle both formats:
#   RTK:    "cargo test: 1236 passed, 7 ignored (17 suites, 5.19s)"
#   Native: "test result: ok. 489 passed; 0 failed; ..."  (per suite)
TEST_OUTPUT=$(cargo test 2>&1 || true)

# Try RTK summary first (single line with total)
TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oE 'cargo test: ([0-9]+) passed' | grep -oE '[0-9]+' | head -1)
if [ -z "$TESTS_PASSED" ]; then
    # Fall back to native: sum all "N passed" lines
    TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | paste -sd+ - | bc 2>/dev/null || echo 0)
fi
TESTS_PASSED=${TESTS_PASSED:-0}

TESTS_FAILED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
TESTS_FAILED=${TESTS_FAILED:-0}

# ── Step 3: Clippy ────────────────────────────────────────────────────────────
# RTK format: "cargo clippy: 2 errors, 132 warnings"
# Native: individual "warning[...]" lines
CLIPPY_OUTPUT=$(cargo clippy --all-targets 2>&1 || true)

# Try RTK summary first
CLIPPY_WARNINGS=$(echo "$CLIPPY_OUTPUT" | grep -oE 'cargo clippy:.* ([0-9]+) warnings' | grep -oE '[0-9]+ warnings' | grep -oE '[0-9]+' | head -1)
if [ -z "$CLIPPY_WARNINGS" ]; then
    # Native: count warning lines
    CLIPPY_WARNINGS=$(echo "$CLIPPY_OUTPUT" | grep -c 'warning\[' || true)
fi
CLIPPY_WARNINGS=${CLIPPY_WARNINGS:-0}

# ── Step 4: Binary size ──────────────────────────────────────────────────────
BINARY="target/release/apollo-optimizerd"
if [ -f "$BINARY" ]; then
    SIZE_BYTES=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY" 2>/dev/null || echo 0)
    SIZE_KB=$((SIZE_BYTES / 1024))
else
    SIZE_KB=0
fi

# ── Step 5: Compute score ─────────────────────────────────────────────────────
T1=$(python3 -c 'import time; print(time.time())')
TIME_S=$(python3 -c "print(round($T1 - $T0, 1))")

# Gate: PASS requires all tests pass AND zero clippy warnings.
if [ "$TESTS_FAILED" -eq 0 ] && [ "$CLIPPY_WARNINGS" -eq 0 ] && [ "$TESTS_PASSED" -gt 0 ]; then
    PASS=1
else
    PASS=0
fi

# Score computation.
SCORE=$(python3 -c "
tests = $TESTS_PASSED
clippy = $CLIPPY_WARNINGS
size_kb = $SIZE_KB
baseline_tests = $BASELINE_TESTS
baseline_size = $BASELINE_SIZE_KB

bloat_penalty = max(0, (size_kb - baseline_size)) * 0.01
new_test_bonus = max(0, (tests - baseline_tests)) * 0.5
score = tests - clippy * 5 - bloat_penalty + new_test_bonus

print(round(score, 2))
")

echo "SCORE=$SCORE TESTS=$TESTS_PASSED CLIPPY=$CLIPPY_WARNINGS SIZE_KB=$SIZE_KB TIME_S=$TIME_S PASS=$PASS"
