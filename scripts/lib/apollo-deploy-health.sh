#!/bin/sh

# Pure post-deploy health policy. The caller owns sampling and rollback; this
# function only classifies one coherent snapshot.
apollo_deploy_health_verdict() {
    pre_ais=$1
    post_ais=$2
    cycles=$3
    failures=$4
    last_error=$5
    ais_mature=$6

    case "$cycles" in ''|*[!0-9]*) echo fail-cycles; return ;; esac
    case "$failures" in ''|*[!0-9]*) echo fail-failures; return ;; esac
    [ "$cycles" -gt 0 ] || { echo fail-cycles; return; }
    [ "$cycles" -ge "${APOLLO_DEPLOY_MIN_CYCLES:-800}" ] || {
        echo fail-warmup
        return
    }
    [ "$failures" -eq 0 ] || { echo fail-failures; return; }
    case "$last_error" in None|null|'') ;; *) echo fail-last-error; return ;; esac

    [ "$ais_mature" = 1 ] || { echo pass-warming; return; }

    /usr/bin/python3 - \
        "$pre_ais" "$post_ais" \
        "${APOLLO_AIS_ABSOLUTE_FLOOR:-75.0}" \
        "${APOLLO_AIS_MAX_REGRESSION:-3.0}" \
        "${APOLLO_AIS_BELOW_FLOOR_TOLERANCE:-0.5}" \
        "${APOLLO_AIS_COMPARISON_EPSILON:-0.01}" <<'PY'
import math
import sys

try:
    pre, post, absolute, regression, recovery, epsilon = map(float, sys.argv[1:])
except (TypeError, ValueError):
    print("fail-ais-invalid")
    raise SystemExit

if not all(math.isfinite(value) for value in (pre, post, absolute, regression, recovery, epsilon)):
    print("fail-ais-invalid")
elif pre <= 0:
    print("pass" if post + epsilon >= absolute else "fail-ais")
elif pre < absolute:
    floor = max(0.0, pre - recovery)
    print("pass" if post + epsilon >= floor else "fail-ais")
else:
    floor = max(absolute, pre - regression)
    print("pass" if post + epsilon >= floor else "fail-ais")
PY
}
