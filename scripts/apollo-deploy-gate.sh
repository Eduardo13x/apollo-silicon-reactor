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
#   GATE 3 — POST-SNAPSHOT (90s after restart): honest AIS must stay within
#            3 points of its pre-deploy value and above the safety floor,
#            failures must stay 0, last_error must be null. Otherwise
#            the script pauses and prints the exact scoped rollback command.
#            Rollback runs only when --auto-revert was explicitly requested.
#
# Usage:
#   ./scripts/apollo-deploy-gate.sh                      # full guarded deploy
#   ./scripts/apollo-deploy-gate.sh --skip-test-check    # explicit override
#                                                       # (logged loudly)
#   ./scripts/apollo-deploy-gate.sh --dry-run            # gates only, no deploy
#   ./scripts/apollo-deploy-gate.sh --auto-revert         # deploy; rollback on failure
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY_SRC="$REPO_ROOT/target/release/apollo-optimizerd"
BINARY_CTL_SRC="$REPO_ROOT/target/release/apollo-optimizerctl"
INSTALLED_CTL="/usr/local/bin/apollo-optimizerctl"
DEPLOYER="/usr/local/sbin/apollo-deploy"

AIS_ABSOLUTE_FLOOR=75.0
AIS_MAX_REGRESSION=3.0
AIS_BELOW_FLOOR_TOLERANCE=0.5
SKIP_TEST_CHECK=0
DRY_RUN=0
AUTO_REVERT=0
EXPECTED_HEAD=""
EXPECTED_DAEMON_SHA=""
EXPECTED_CTL_SHA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-test-check) SKIP_TEST_CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --auto-revert) AUTO_REVERT=1; shift ;;
    --expected-head) EXPECTED_HEAD="${2:?--expected-head needs a SHA}"; shift 2 ;;
    --expected-daemon-sha) EXPECTED_DAEMON_SHA="${2:?--expected-daemon-sha needs a SHA-256}"; shift 2 ;;
    --expected-ctl-sha) EXPECTED_CTL_SHA="${2:?--expected-ctl-sha needs a SHA-256}"; shift 2 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

sha256_file() {
  /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

valid_hex_length() {
  local value="$1" length="$2"
  [ "${#value}" -eq "$length" ] || return 1
  case "$value" in *[!0-9a-fA-F]*) return 1 ;; esac
  return 0
}

extract_backup_path() {
  printf '%s\n' "$1" | awk '
    /^APOLLO_BACKUP_PATH=\/var\/lib\/apollo\/backups\/deploy-/ {
      sub(/^APOLLO_BACKUP_PATH=/, ""); path=$0
    }
    /^backup_path=\/var\/lib\/apollo\/backups\/deploy-/ {
      sub(/^backup_path=/, ""); path=$0
    }
    /^Apollo deployed; backup: \/var\/lib\/apollo\/backups\/deploy-/ {
      sub(/^Apollo deployed; backup: /, ""); path=$0
    }
    /^deploy failed; rollback: .* rollback \/var\/lib\/apollo\/backups\/deploy-/ {
      path=$NF
    }
    END { if (path != "") print path }
  '
}

backup_path_valid() {
  case "$1" in
    /var/lib/apollo/backups/deploy-*) ;;
    *) return 1 ;;
  esac
  local leaf="${1#/var/lib/apollo/backups/deploy-}"
  [ -n "$leaf" ] || return 1
  case "$leaf" in
    *..*|*/*|*[!A-Za-z0-9_-]*) return 1 ;;
  esac
  return 0
}

emit_rollback_record() {
  printf 'APOLLO_BACKUP_PATH=%s\n' "$BACKUP_PATH"
  printf 'APOLLO_ROLLBACK_COMMAND=sudo -n %s rollback %s\n' "$DEPLOYER" "$BACKUP_PATH"
}

handle_failure_rollback() {
  local reason="$1"
  if [ -z "${BACKUP_PATH:-}" ] || ! backup_path_valid "$BACKUP_PATH"; then
    red "[rollback] unavailable: no valid scoped backup path was captured."
    printf 'APOLLO_ROLLBACK_STATUS=unavailable\n'
    return 0
  fi

  emit_rollback_record
  if [ "$AUTO_REVERT" != "1" ]; then
    yellow "[rollback] paused after $reason; --auto-revert was not requested."
    yellow "[rollback] exact command: sudo -n $DEPLOYER rollback $BACKUP_PATH"
    printf 'APOLLO_ROLLBACK_STATUS=not-requested\n'
    return 0
  fi

  yellow "[rollback] $reason; executing the explicitly approved scoped rollback."
  local rollback_output rollback_status
  set +e
  rollback_output=$(sudo -n "$DEPLOYER" rollback "$BACKUP_PATH" 2>&1)
  rollback_status=$?
  set -e
  printf '%s\n' "$rollback_output"
  if [ "$rollback_status" -eq 0 ]; then
    green "[rollback] completed and verified by $DEPLOYER."
    printf 'APOLLO_ROLLBACK_STATUS=succeeded\n'
    return 0
  fi

  red "[rollback] FAILED with status $rollback_status."
  printf 'APOLLO_ROLLBACK_STATUS=failed\n'
  return 1
}

capture_runtime_metrics() {
  local output_path="$1"
  if sudo -n "$INSTALLED_CTL" status \
      | python3 -c 'import json,sys; json.dump(json.load(sys.stdin).get("metrics", {}), sys.stdout)' \
      > "$output_path" 2>/dev/null; then
    return 0
  fi
  echo '{}' > "$output_path"
}

# ── Gate 1: test evidence ────────────────────────────────────────────
if [ "$SKIP_TEST_CHECK" = "1" ]; then
  yellow "[gate-1] --skip-test-check set — bypassing test-diff requirement."
  yellow "[gate-1] this is logged; if you regress, it's on you."
else
  cd "$REPO_ROOT"
  # Look at staged + last-commit diff. Any line starting with '+' that
  # contains a #[test] attribute or an fn test_* signature counts.
  TEST_DIFF=$(git diff --unified=0 HEAD -- '*.rs' 2>/dev/null \
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

# A diff is evidence of intent, not evidence that the code works. Run the
# actual full workspace gate before touching the installed daemon.
"$REPO_ROOT/scripts/pipeline.sh" --skip-deploy

# --risky pins the verified source and release artifacts across the pipeline.
# Direct legacy invocations may omit this tuple, but a partial tuple fails.
EXPECTED_COUNT=0
[ -n "$EXPECTED_HEAD" ] && EXPECTED_COUNT=$((EXPECTED_COUNT + 1))
[ -n "$EXPECTED_DAEMON_SHA" ] && EXPECTED_COUNT=$((EXPECTED_COUNT + 1))
[ -n "$EXPECTED_CTL_SHA" ] && EXPECTED_COUNT=$((EXPECTED_COUNT + 1))
if [ "$EXPECTED_COUNT" -ne 0 ] && [ "$EXPECTED_COUNT" -ne 3 ]; then
  red "[gate-1] incomplete immutable-candidate tuple."
  exit 2
fi
if [ "$EXPECTED_COUNT" -eq 3 ]; then
  valid_hex_length "$EXPECTED_HEAD" 40 || valid_hex_length "$EXPECTED_HEAD" 64 \
    || { red "[gate-1] invalid expected HEAD SHA."; exit 2; }
  valid_hex_length "$EXPECTED_DAEMON_SHA" 64 \
    || { red "[gate-1] invalid daemon SHA-256."; exit 2; }
  valid_hex_length "$EXPECTED_CTL_SHA" 64 \
    || { red "[gate-1] invalid ctl SHA-256."; exit 2; }
  CURRENT_HEAD=$(git -C "$REPO_ROOT" rev-parse --verify HEAD^{commit} 2>/dev/null) \
    || { red "[gate-1] cannot resolve HEAD."; exit 1; }
  [ "$CURRENT_HEAD" = "$(printf '%s' "$EXPECTED_HEAD" | tr '[:upper:]' '[:lower:]')" ] \
    || { red "[gate-1] HEAD changed after verification."; exit 1; }
  [ -z "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] \
    || { red "[gate-1] worktree changed after verification."; exit 1; }
  [ -x "$BINARY_SRC" ] && [ "$(sha256_file "$BINARY_SRC")" = "$EXPECTED_DAEMON_SHA" ] \
    || { red "[gate-1] daemon candidate changed after verification."; exit 1; }
  [ -x "$BINARY_CTL_SRC" ] && [ "$(sha256_file "$BINARY_CTL_SRC")" = "$EXPECTED_CTL_SHA" ] \
    || { red "[gate-1] ctl candidate changed after verification."; exit 1; }
  green "[gate-1] immutable HEAD and candidate hashes match the verification record."
fi

# ── Gate 2: pre-snapshot ─────────────────────────────────────────────
PRE_SNAP="/tmp/apollo_pre_snap_$$.json"
capture_runtime_metrics "$PRE_SNAP"
PRE_AIS=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('ais_score', 0))")
PRE_CYCLES=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('cycles', 0))")
PRE_FAILS=$(python3 -c "import json; print(json.load(open('$PRE_SNAP')).get('failures', 0))")
green "[gate-2] pre-snap: AIS=$PRE_AIS cycles=$PRE_CYCLES failures=$PRE_FAILS"

if [ "$DRY_RUN" = "1" ]; then
  yellow "[dry-run] not deploying. Gates 1+2 ok."
  exit 0
fi

# ── Deploy through the scoped, audited root helper ───────────────────
if [ ! -x "$BINARY_SRC" ] || [ ! -x "$BINARY_CTL_SRC" ]; then
  red "[deploy] release binaries missing after pipeline."
  exit 3
fi
if [ ! -x "$DEPLOYER" ]; then
  red "[deploy] scoped deployer missing: $DEPLOYER"
  exit 3
fi
cp -f "$BINARY_SRC" /private/tmp/apollo-optimizerd-candidate
cp -f "$BINARY_CTL_SRC" /private/tmp/apollo-optimizerctl-candidate
chmod 755 /private/tmp/apollo-optimizerd-candidate /private/tmp/apollo-optimizerctl-candidate
codesign --force --sign - /private/tmp/apollo-optimizerd-candidate
codesign --force --sign - /private/tmp/apollo-optimizerctl-candidate
set +e
DEPLOY_OUTPUT=$(sudo -n "$DEPLOYER" deploy 2>&1)
DEPLOY_STATUS=$?
set -e
printf '%s\n' "$DEPLOY_OUTPUT"
BACKUP_PATH=$(extract_backup_path "$DEPLOY_OUTPUT")

if [ -n "$BACKUP_PATH" ] && ! backup_path_valid "$BACKUP_PATH"; then
  red "[deploy] deployer returned an invalid backup path."
  printf 'APOLLO_ROLLBACK_STATUS=unavailable\n'
  exit 3
fi

if [ "$DEPLOY_STATUS" -ne 0 ]; then
  red "[deploy] scoped deployer failed with status $DEPLOY_STATUS."
  if ! handle_failure_rollback "deployer failure"; then
    exit 5
  fi
  exit "$DEPLOY_STATUS"
fi

if [ -z "$BACKUP_PATH" ]; then
  red "[deploy] deployer succeeded without a machine-readable backup path."
  printf 'APOLLO_ROLLBACK_STATUS=unavailable\n'
  exit 3
fi
emit_rollback_record
printf 'APOLLO_ROLLBACK_STATUS=armed\n'

# ── Gate 3: post-snapshot 90s window ─────────────────────────────────
yellow "[gate-3] sleeping 90s for daemon to stabilize before health check..."
sleep 90
POST_SNAP="/tmp/apollo_post_snap_$$.json"
capture_runtime_metrics "$POST_SNAP"
# B.6 fix v2 (2026-06-10): ais_score serializes as 0.0 default from cycle 1
# — presence is meaningless; only a COMPUTED score (>0) is judgeable. AIS
# needs ~3-5 min of warmup cycles. Poll up to 300s more for ais_score > 0.
# A genuinely sick daemon still fails: failures/last_error are checked
# independently, and a daemon that never computes AIS in 6.5 min total
# fails the floor check with 0.0 as before.
# Gate calibration v4 (2026-06-10): v3's cycles>=400 still false-FAILed —
# the windowed AIS components (signal, adaptability) need ~800 cycles to
# fill, not 400 (gate saw 84-85 at cycle 425; same daemon read 92+ S at
# cycle 825). Judge only once score>0 AND cycles>=800, up to 720s extra.
AIS_READY=$(python3 -c "import json; m=json.load(open('$POST_SNAP')); print(1 if m.get('ais_score', 0) > 0 and m.get('cycles', 0) >= 800 else 0)")
WAITED=0
while [ "$AIS_READY" = "0" ] && [ "$WAITED" -lt 720 ]; do
  yellow "[gate-3] AIS warming (need score>0 and cycles>=800) — waiting 30s (waited ${WAITED}s)..."
  sleep 30
  WAITED=$((WAITED + 30))
  capture_runtime_metrics "$POST_SNAP"
  AIS_READY=$(python3 -c "import json; m=json.load(open('$POST_SNAP')); print(1 if m.get('ais_score', 0) > 0 and m.get('cycles', 0) >= 800 else 0)")
done
POST_AIS=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('ais_score', 0))")
POST_FAILS=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('failures', 0))")
POST_ERR=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('last_error', None))")
POST_CYCLES=$(python3 -c "import json; print(json.load(open('$POST_SNAP')).get('cycles', 0))")
# A baseline already just below the absolute floor must be allowed to improve.
# Otherwise a healthy 74.9 -> 74.9 deploy can never pass. In that recovery
# state, enforce tight non-regression; once baseline reaches 75, restore the
# strict absolute floor plus the normal max-regression rule.
AIS_FLOOR=$(python3 -c "
pre=float('$PRE_AIS')
absolute=$AIS_ABSOLUTE_FLOOR
if pre <= 0:
    print(absolute)
elif pre < absolute:
    print(max(0.0, pre - $AIS_BELOW_FLOOR_TOLERANCE))
else:
    print(max(absolute, pre - $AIS_MAX_REGRESSION))
")

echo "[gate-3] post-snap: AIS=$POST_AIS cycles=$POST_CYCLES failures=$POST_FAILS last_error=$POST_ERR"

# Verdict.
HEALTHY=1
python3 -c "import sys; sys.exit(0 if float('$POST_AIS') >= $AIS_FLOOR else 1)" || HEALTHY=0
[ "$POST_FAILS" = "0" ] || HEALTHY=0
[ "$POST_ERR" = "None" ] || HEALTHY=0
[ "$POST_CYCLES" = "0" ] && HEALTHY=0  # daemon never started a cycle

if [ "$HEALTHY" = "1" ]; then
  green "[gate-3] PASS — AIS≥$AIS_FLOOR (absolute floor $AIS_ABSOLUTE_FLOOR), failures=0, error=None, cycles progressing"
  green "[deploy] success."
  exit 0
fi

red "[gate-3] FAILED post-deploy sanity:"
red "         AIS=$POST_AIS (floor $AIS_FLOOR)"
red "         failures=$POST_FAILS (must be 0)"
red "         last_error=$POST_ERR (must be None)"
red "         cycles=$POST_CYCLES (must be > 0)"
yellow "[suggest] rollback options (review BEFORE running):"
yellow "  1. git revert HEAD && rerun this script"
yellow "  2. restore the backup created by $DEPLOYER"
yellow "[suggest] capture the diff for post-mortem before rollback:"
yellow "  cp $PRE_SNAP /tmp/incident_pre.json"
yellow "  cp $POST_SNAP /tmp/incident_post.json"
if ! handle_failure_rollback "post-deploy health failure"; then
  exit 5
fi
exit 4
