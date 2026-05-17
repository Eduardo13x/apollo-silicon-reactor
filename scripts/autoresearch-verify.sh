#!/bin/bash
# autoresearch composite dev metric — outputs a single number (higher = better).
#
# Components:
#   tests_passed              × 1   (capped at 3000)
#   (50 - clippy_warnings) × 2      (negative penalty if warnings > 50)
#   wired_phases           × 100    (production callers exist)
#   ais_live               × 5      (production AIS from running daemon, 0-100)
#   build_ok               × 50     (binary compile success)
#
# Reads:
#   - cargo test output (apollo-engine lib)
#   - cargo clippy --all-targets
#   - source grep for `LSE_COUNTERS.inc_*`, `LSE_COUNTERS.add_*` PROD callers
#   - /var/lib/apollo/runtime_metrics.json (live daemon)
#
# Outputs ONE number to stdout (the composite score).
# Logs intermediate values to stderr for human review.

set -uo pipefail
cd "$(dirname "$0")/.."

# Component 1: build (50 if ok, 0 if not)
BUILD_OK=0
if cargo build --release --bin apollo-optimizerd >/dev/null 2>&1; then
    BUILD_OK=50
fi

# Component 2: tests passed (apollo-engine lib)
TESTS_PASSED=0
if [ "$BUILD_OK" -gt 0 ]; then
    TESTS_PASSED=$(cargo test -p apollo-engine --lib 2>&1 | \
        grep -oE "[0-9]+ passed" | head -1 | awk '{print $1}')
    TESTS_PASSED=${TESTS_PASSED:-0}
    [ "$TESTS_PASSED" -gt 3000 ] && TESTS_PASSED=3000
fi

# Component 3: clippy warnings (lower = better)
CLIPPY_WARNINGS=$(cargo clippy --all-targets 2>&1 | \
    grep -c "^warning:" || echo 0)
CLIPPY_SCORE=$(( (50 - CLIPPY_WARNINGS) * 2 ))

# Component 4: wired phases — count distinct LSE_COUNTERS.{inc,add}_*
# calls in PROD code (src/ or crates/*/src/ excluding tests/).
# A wired phase = its dedicated counter has ≥1 production caller.
WIRED_PHASES=0
declare -a PHASE_PATTERNS=(
    "add_skill_aware_modulations"                # 3.1
    "add_arousal_decay_accelerations"            # 3.2
    "add_companion_cross_group_inferences"       # 3.3
    "add_adaptive_drift_threshold_raises"        # 4.1
    "inc_causal_external_thermal_blame"          # 4.2
    "inc_policy_rollback_evaluation"             # 4.3
    "add_user_presence_suppressions"             # 5.1
    "inc_battery_aware_penalty_emission"         # 5.2
    "inc_journal_rationale_attached"             # 5.3
)
for pattern in "${PHASE_PATTERNS[@]}"; do
    count=$(grep -rE "LSE_COUNTERS\.${pattern}|\.${pattern}\(" \
        --include="*.rs" src/ crates/apollo-engine/src/ 2>/dev/null | \
        grep -v "/tests/\|::tests\|#\[cfg(test)\]\|fn ${pattern}\|//\|/\*" | \
        wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        WIRED_PHASES=$((WIRED_PHASES + 1))
    fi
done
WIRED_SCORE=$((WIRED_PHASES * 100))

# Component 5: live AIS from prod daemon (0-100)
# Reads from /tmp/apollo-ais-mirror.json (operator-maintained snapshot) to
# avoid sudo in non-interactive autoresearch contexts. Set up via:
#   sudo cp /var/lib/apollo/runtime_metrics.json /tmp/apollo-ais-mirror.json
# (re-run periodically or after deploy)
AIS_LIVE=0
if [ -f /tmp/apollo-ais-mirror.json ]; then
    AIS_LIVE=$(python3 -c "import json; print(int(json.load(open('/tmp/apollo-ais-mirror.json')).get('ais_score',0)))" 2>/dev/null)
    AIS_LIVE=${AIS_LIVE:-0}
fi
AIS_SCORE=$((AIS_LIVE * 5))

# Composite
COMPOSITE=$((TESTS_PASSED + CLIPPY_SCORE + WIRED_SCORE + AIS_SCORE + BUILD_OK))

# Log breakdown to stderr
{
    echo "[autoresearch-verify]"
    echo "  build_ok     = $BUILD_OK"
    echo "  tests_passed = $TESTS_PASSED"
    echo "  clippy_warn  = $CLIPPY_WARNINGS (score $CLIPPY_SCORE)"
    echo "  wired_phases = $WIRED_PHASES / 9 (score $WIRED_SCORE)"
    echo "  ais_live     = $AIS_LIVE (score $AIS_SCORE)"
    echo "  COMPOSITE    = $COMPOSITE"
} >&2

# stdout = single number for autoresearch metric parser
echo "$COMPOSITE"
