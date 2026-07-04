#!/usr/bin/env bash
# Apollo supervision loop: repeat safe verification without deploy/restart.
#
# This is the conservative counterpart to scripts/pipeline.sh and
# scripts/watch-deploy.sh. It never kills the daemon, never copies binaries,
# never calls launchctl, and never updates the acceptance baseline.
#
# Default loop:
#   1. Read live runtime_metrics.json.
#   2. Run focused thermal tests.
#   3. Run acceptance gate in --dry-run mode.
#   4. Scan today's journal tail for fresh failures.
#
# Usage:
#   ./scripts/apollo-supervision-loop.sh
#   ./scripts/apollo-supervision-loop.sh --iterations 6 --sleep 30
#   ./scripts/apollo-supervision-loop.sh --full-tests
#   ./scripts/apollo-supervision-loop.sh --no-tests
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ITERATIONS=3
SLEEP_SECS=15
RUN_TESTS=1
FULL_TESTS=0
RUN_DEPLOY_DRY_RUN=0
REPORT="${APOLLO_LOOP_REPORT:-/tmp/apollo-supervision-loop-$(date -u +%Y%m%dT%H%M%SZ).log}"

usage() {
  sed -n '1,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --iterations)
      ITERATIONS="${2:?--iterations needs a number}"
      shift 2
      ;;
    --sleep)
      SLEEP_SECS="${2:?--sleep needs seconds}"
      shift 2
      ;;
    --full-tests)
      FULL_TESTS=1
      shift
      ;;
    --no-tests)
      RUN_TESTS=0
      shift
      ;;
    --deploy-dry-run)
      RUN_DEPLOY_DRY_RUN=1
      shift
      ;;
    --report)
      REPORT="${2:?--report needs a path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ITERATIONS" in
  ''|*[!0-9]*) echo "--iterations must be a positive integer" >&2; exit 2 ;;
esac
case "$SLEEP_SECS" in
  ''|*[!0-9]*) echo "--sleep must be a non-negative integer" >&2; exit 2 ;;
esac
[ "$ITERATIONS" -gt 0 ] || { echo "--iterations must be > 0" >&2; exit 2; }

mkdir -p "$(dirname "$REPORT")"
exec > >(tee -a "$REPORT") 2>&1

FAILURES=0

banner() {
  printf '\n== %s ==\n' "$*"
}

run_step() {
  local name="$1"
  shift
  banner "$name"
  if "$@"; then
    echo "[pass] $name"
  else
    local rc=$?
    echo "[fail] $name rc=$rc"
    FAILURES=$((FAILURES + 1))
  fi
}

runtime_snapshot() {
  if [ -r /var/lib/apollo/runtime_metrics.json ]; then
    python3 - /var/lib/apollo/runtime_metrics.json <<'PY'
import json, sys
path = sys.argv[1]
keys = [
    "cycles", "failures", "last_error", "ais_score", "ais_grade",
    "p95_cycle_ms", "stage_reason_avg_ms", "stage_reason_max_ms",
    "memory_pressure", "thrashing_score", "warm_band_fires",
    "warm_boost_sum_x1000", "reactor_health",
]
with open(path) as f:
    m = json.load(f)
print(json.dumps({k: m.get(k) for k in keys}, indent=2, sort_keys=True))
PY
    return $?
  fi

  sudo -n python3 - /var/lib/apollo/runtime_metrics.json <<'PY'
import json, sys
path = sys.argv[1]
keys = [
    "cycles", "failures", "last_error", "ais_score", "ais_grade",
    "p95_cycle_ms", "stage_reason_avg_ms", "stage_reason_max_ms",
    "memory_pressure", "thrashing_score", "warm_band_fires",
    "warm_boost_sum_x1000", "reactor_health",
]
with open(path) as f:
    m = json.load(f)
print(json.dumps({k: m.get(k) for k in keys}, indent=2, sort_keys=True))
PY
}

journal_failures_today() {
  local today
  local journal_tail
  local rc
  today="$(date -u +%Y-%m-%d)"
  journal_tail="$(mktemp -t apollo_loop_journal.XXXXXX)"
  sudo -n tail -n 1000 /var/lib/apollo/journal.jsonl > "$journal_tail" 2>/dev/null || true
  python3 - "$today" "$journal_tail" <<'PY'
import json, sys
today = sys.argv[1]
path = sys.argv[2]
scanned = 0
blocked = []
bad = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = str(rec.get("timestamp", ""))
        if not ts.startswith(today):
            continue
        scanned += 1
        reason = str(rec.get("reason", ""))
        reason_l = reason.lower()
        if rec.get("success") is False:
            if reason_l.startswith("skip:"):
                blocked.append((ts, reason[:160]))
            else:
                bad.append((ts, reason[:160]))
        elif any(term in reason_l for term in ("panic", "crash", "error")):
            bad.append((ts, reason[:160]))
print(f"today={today} scanned={scanned} blocked={len(blocked)} hard_bad={len(bad)}")
for ts, reason in blocked[-10:]:
    print(f"- blocked {ts} {reason}")
for ts, reason in bad[-20:]:
    print(f"- hard_bad {ts} {reason}")
sys.exit(1 if bad else 0)
PY
  rc=$?
  rm -f "$journal_tail"
  return "$rc"
}

focused_tests() {
  cargo test -p apollo-engine thermal_bailout::tests -- --quiet
}

full_tests() {
  cargo test -p apollo-engine --lib -- --quiet &&
    cargo test --bin apollo-optimizerd -- --quiet
}

accept_dry_run() {
  ./scripts/apollo-accept-gate.sh --dry-run
}

deploy_gate_dry_run() {
  ./scripts/apollo-deploy-gate.sh --dry-run
}

echo "Apollo supervision loop"
echo "repo=$REPO_ROOT"
echo "report=$REPORT"
echo "iterations=$ITERATIONS sleep=${SLEEP_SECS}s tests=$RUN_TESTS full_tests=$FULL_TESTS deploy_dry_run=$RUN_DEPLOY_DRY_RUN"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for i in $(seq 1 "$ITERATIONS"); do
  banner "iteration $i/$ITERATIONS"
  run_step "runtime snapshot" runtime_snapshot

  if [ "$RUN_TESTS" -eq 1 ]; then
    if [ "$FULL_TESTS" -eq 1 ]; then
      run_step "full tests" full_tests
    else
      run_step "focused thermal tests" focused_tests
    fi
  fi

  run_step "acceptance gate dry-run" accept_dry_run

  if [ "$RUN_DEPLOY_DRY_RUN" -eq 1 ]; then
    run_step "deploy gate dry-run" deploy_gate_dry_run
  fi

  run_step "journal failures today" journal_failures_today

  if [ "$i" -lt "$ITERATIONS" ] && [ "$SLEEP_SECS" -gt 0 ]; then
    echo "sleeping ${SLEEP_SECS}s..."
    sleep "$SLEEP_SECS"
  fi
done

echo
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "failures=$FAILURES"
exit "$FAILURES"
