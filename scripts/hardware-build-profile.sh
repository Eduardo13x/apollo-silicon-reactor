# Source from Apollo build scripts. Enables the measured parallel process
# refresh only on heterogeneous Apple Silicon with enough cores to amortize it.
APOLLO_CARGO_FEATURE_ARGS=()
APOLLO_BUILD_PROFILE="sequential"

if [ "$(uname -s 2>/dev/null || true)" = "Darwin" ]; then
    APOLLO_HW_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
    APOLLO_P_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 0)
    APOLLO_E_CORES=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo 0)
    if [ "$APOLLO_HW_CORES" -ge 10 ] && [ "$APOLLO_P_CORES" -ge 4 ] && [ "$APOLLO_E_CORES" -ge 4 ]; then
        APOLLO_CARGO_FEATURE_ARGS=(--features adaptive-multicore)
        APOLLO_BUILD_PROFILE="adaptive-multicore"
    fi
fi
