#!/usr/bin/env bash
# apollo-deploy-gate.sh — Disciplined daemon deploy with adversarial-test guard.
#
# Per NotebookLM verdict (2026-05-16): a "pre-deploy gate" without
# mechanical verification falls into the same tautology trap as the
# F1-F7 shadow-mode (commit 1198c73). This script enforces three gates
# before allowing the launchctl bootstrap:
#
#   GATE 1 — TEST EVIDENCE: the HEAD commit (or staged diff) must
#            add/modify at least one #[test] item. The "Disobedience
#            Rule" from CLAUDE.md says: write the failing test FIRST.
#            We cannot mechanically prove it failed, but we can refuse
#            to deploy if there is literally no test diff at all.
#
#   GATE 2 — PRE-SNAPSHOT: capture runtime_metrics.json + cycle count
#            before swapping the binary. Used for post-deploy diff.
#
#   GATE 3 — POST-SNAPSHOT (90s after restart): AIS must stay ≥ 87,
#            failures must stay 0, last_error must be null. Otherwise
#            the script alerts loudly. Rollback is suggested but not
#            executed — the human decides (CLAUDE.md supervision rule).
#
# Usage:
#   ./scripts/apollo-deploy-gate.sh                      # full guarded deploy
#   ./scripts/apollo-deploy-gate.sh --skip-test-check    # explicit override
#                                                       # (logged loudly)
#   ./scripts/apollo-deploy-gate.sh --dry-run            # gates only, no deploy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY_SRC="$REPO_ROOT/target/release/apollo-optimizerd"
BINARY_DST="/usr/local/libexec/apollo-optimizerd"
PLIST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"
LABEL="com.eduardocortez.systemoptimizerd"

AIS_FLOOR=87.0
SKIP_TEST_CHECK=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --skip-test-check) SKIP_TEST_CHECK=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

# ── Gate 1: test evidence ────────────────────────────────────────────
if [ "$SKIP_TEST_CHECK" = "1" ]; then
  yellow "[gate-1] --skip-test-check set — bypassing test-diff requirement."
  yellow "[gate-1] this is logged; if you regress, it's on you."
else
  cd "$REPO_ROOT"
  # Look at staged + last-commit diff. Any line starting with '+' that
  # contains a #[test] attribute or an fn test_* signature counts.
  TEST_DIFF=$(git diff --cached --unified=0 HEAD~1..HEAD 2>/dev/null \
              | grep -E '^\+.*(#\[test\]|fn test_|#\[tokio::test\])' \
              || true)
  if [ -z "$TEST_DIFF" ]; then
    # Fall back to scanning the last commit alone (in case nothing is staged).
    TEST_DIFF=$(git show --unified=0 HEAD -- '*.rs' 2>/dev/null \
                | grep -E '^\+.*(#\[test\]|fn test_|#\[tokio::test\])' \
                || true)
  fi
  if [ -z "$TEST_DIFF" ]; then
    # Merge commits have no diff under `git show HEAD`. Walk the last 3
    # parent commits to surface tests added by the merged branches.
    TEST_DIFF=$(git log -3 --unified=0 -p --no-merges -- '*.rs' 2>/dev/null \
                | grep -E '^\+.*(#\[test\]|fn test_|#\[tokio::test\])' \
                | head -5 || true)
  fi
  if [ -z "$TEST_DIFF" ]; then
    red "[gate-1] FAILED: no #[test] added/modified in HEAD or staged diff."
    red "[gate-1] The Disobedience Rule (CLAUDE.md) requires a failing test"
    red "[gate-1] before the fix. Re-run with --skip-test-check to override,"
    red "[gate-1] but understand: F1-F7 shipped 7 commits without one, and"
    red "[gate-1] NotebookLM called it 'shadow-mode theater'."
    exit 1
  fi
  green "[gate-1] ok — test diff present:"
  echo "$TEST_DIFF" | head -3
fi

# ── Gate 2: pre-snapshot ─────────────────────────────────────────────
PRE_SNAP="/tmp/apollo_pre_snap_$$.json"
sudo cat /var/lib/apollo/runtime_metrics.json > "$PRE_SNAP" 2>/dev/null || echo '{}' > "$PRE_SNAP"
PRE_AIS=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('ais_score', 0))")
PRE_CYCLES=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('cycles', 0))")
PRE_FAILS=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('failures', 0))")
green "[gate-2] pre-snap: AIS=$PRE_AIS cycles=$PRE_CYCLES failures=$PRE_FAILS"

if [ "$DRY_RUN" = "1" ]; then
  yellow "[dry-run] not deploying. Gates 1+2 ok."
  exit 0
fi

# ── Deploy: copy binary, bootout, bootstrap ──────────────────────────
if [ ! -x "$BINARY_SRC" ]; then
  red "[deploy] binary not found at $BINARY_SRC — run 'cargo build --release' first."
  exit 3
fi
echo "[deploy] sudo cp $BINARY_SRC $BINARY_DST"
sudo cp "$BINARY_SRC" "$BINARY_DST"
echo "[deploy] sudo launchctl bootout system/$LABEL"
sudo launchctl bootout "system/$LABEL" 2>&1 || true
sleep 2
echo "[deploy] sudo launchctl bootstrap system $PLIST"
sudo launchctl bootstrap system "$PLIST"

# ── Gate 3: post-snapshot 90s window ─────────────────────────────────
yellow "[gate-3] sleeping 90s for daemon to stabilize before health check..."
sleep 90
POST_SNAP="/tmp/apollo_post_snap_$$.json"
sudo cat /var/lib/apollo/runtime_metrics.json > "$POST_SNAP" 2>/dev/null || echo '{}' > "$POST_SNAP"
# B.6 fix v2 (2026-06-10): ais_score serializes as 0.0 default from cycle 1
# — presence is meaningless; only a COMPUTED score (>0) is judgeable. AIS
# needs ~3-5 min of warmup cycles. Poll up to 300s more for ais_score > 0.
# A genuinely sick daemon still fails: failures/last_error are checked
# independently, and a daemon that never computes AIS in 6.5 min total
# fails the floor check with 0.0 as before.
# Gate calibration v3 (2026-06-10): two consecutive false-FAILs showed the
# pattern — ais_resource is unstable below ~400 cycles (gate saw 84-85 at
# cycle 150; same daemon read 92+ S at cycle 450 both times). Judge only
# once BOTH ais_score > 0 AND cycles >= 400, waiting up to 480s extra.
AIS_READY=$(python3 -c "import json; m=json.load(open('$POST_SNAP')); print(1 if m.get('ais_score', 0) > 0 and m.get('cycles', 0) >= 400 else 0)")
WAITED=0
while [ "$AIS_READY" = "0" ] && [ "$WAITED" -lt 480 ]; do
  yellow "[gate-3] AIS warming (need score>0 and cycles>=400) — waiting 30s (waited ${WAITED}s)..."
  sleep 30
  WAITED=$((WAITED + 30))
  sudo cat /var/lib/apollo/runtime_metrics.json > "$POST_SNAP" 2>/dev/null || echo '{}' > "$POST_SNAP"
  AIS_READY=$(python3 -c "import json; m=json.load(open('$POST_SNAP')); print(1 if m.get('ais_score', 0) > 0 and m.get('cycles', 0) >= 400 else 0)")
done
POST_AIS=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('ais_score', 0))")
POST_FAILS=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('failures', 0))")
POST_ERR=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('last_error', None))")
POST_CYCLES=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('cycles', 0))")

echo "[gate-3] post-snap: AIS=$POST_AIS cycles=$POST_CYCLES failures=$POST_FAILS last_error=$POST_ERR"

# Verdict.
HEALTHY=1
python3 -c "import sys; sys.exit(0 if float('$POST_AIS') >= $AIS_FLOOR else 1)" || HEALTHY=0
[ "$POST_FAILS" = "0" ] || HEALTHY=0
[ "$POST_ERR" = "None" ] || HEALTHY=0
[ "$POST_CYCLES" = "0" ] && HEALTHY=0  # daemon never started a cycle

if [ "$HEALTHY" = "1" ]; then
  green "[gate-3] PASS — AIS≥$AIS_FLOOR, failures=0, error=None, cycles progressing"
  green "[deploy] success."
  exit 0
fi

red "[gate-3] FAILED post-deploy sanity:"
red "         AIS=$POST_AIS (floor $AIS_FLOOR)"
red "         failures=$POST_FAILS (must be 0)"
red "         last_error=$POST_ERR (must be None)"
red "         cycles=$POST_CYCLES (must be > 0)"
yellow "[suggest] rollback options (review BEFORE running):"
yellow "  1. git revert HEAD && cargo build --release && rerun this script"
yellow "  2. sudo cp <prior-binary> $BINARY_DST && sudo launchctl bootout/bootstrap"
yellow "[suggest] capture the diff for post-mortem before rollback:"
yellow "  cp $PRE_SNAP /tmp/incident_pre.json"
yellow "  cp $POST_SNAP /tmp/incident_post.json"
exit 4
