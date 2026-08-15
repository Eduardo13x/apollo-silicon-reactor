#!/bin/sh
set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
. "$REPO_ROOT/scripts/lib/apollo-deploy-health.sh"

TESTS=0
FAILED=0

check() {
    description=$1
    expected=$2
    shift 2
    TESTS=$((TESTS + 1))
    actual=$(apollo_deploy_health_verdict "$@")
    if [ "$actual" != "$expected" ]; then
        echo "not ok - $description: expected '$expected', got '$actual'" >&2
        FAILED=$((FAILED + 1))
    else
        echo "ok - $description"
    fi
}

# Arguments after the expected verdict:
# pre_ais post_ais cycles failures last_error ais_mature
check "fresh AIS regression does not reject a healthy daemon" \
    pass-warming 83.91 80.11 834 0 None 0
check "zero AIS is tolerated while the daemon is still warming" \
    pass-warming 83.91 0.0 834 0 None 0
check "mature AIS regression remains a health failure" \
    fail-ais 83.91 80.11 3000 0 None 1
check "mature AIS within the regression budget passes" \
    pass 83.91 81.00 3000 0 None 1
check "daemon failures dominate AIS warmup" \
    fail-failures 83.91 83.00 834 1 None 0
check "last error dominates AIS warmup" \
    fail-last-error 83.91 83.00 834 0 timeout 0
check "no cycle progress is unhealthy" \
    fail-cycles 83.91 83.00 0 0 None 0
check "insufficient cycle warmup remains non-deployable" \
    fail-warmup 83.91 83.00 799 0 None 0
check "recovery baseline below absolute floor may remain close" \
    pass 74.9 74.5 3000 0 None 1
check "non-finite mature AIS is rejected" \
    fail-ais-invalid 83.91 NaN 3000 0 None 1

if [ "$FAILED" -ne 0 ]; then
    echo "$FAILED of $TESTS deploy health tests failed" >&2
    exit 1
fi
echo "ok - $TESTS deploy health policy tests"
