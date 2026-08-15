#!/bin/sh
# Shell contract tests for the acceptance/deploy trust boundary.
set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
ACCEPT_GATE="$REPO_ROOT/scripts/apollo-accept-gate.sh"
DEPLOY_GATE="$REPO_ROOT/scripts/apollo-deploy-gate.sh"
TESTS_RUN=0
TESTS_FAILED=0

fail() {
    echo "# $*" >&2
    return 1
}

assert_contains() {
    needle=$1
    file=$2
    context=$3
    /usr/bin/grep -F -- "$needle" "$file" >/dev/null 2>&1 \
        || fail "$context: missing '$needle' in $file"
}

assert_not_contains() {
    needle=$1
    file=$2
    context=$3
    if /usr/bin/grep -F -- "$needle" "$file" >/dev/null 2>&1; then
        fail "$context: unexpected '$needle' in $file"
    fi
}

run_test() {
    TESTS_RUN=$((TESTS_RUN + 1))
    name=$1
    shift
    if "$@"; then
        echo "ok $TESTS_RUN - $name"
    else
        echo "not ok $TESTS_RUN - $name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

test_acceptance_binds_six_artifact_hashes() {
    assert_contains 'context_agent_sha256' "$ACCEPT_GATE" \
        "evidence context-agent hash"
    assert_contains 'web_bridge_sha256' "$ACCEPT_GATE" \
        "evidence web bridge hash"
    assert_contains 'coreml_model_sha256' "$ACCEPT_GATE" \
        "evidence Core ML hash"
    assert_contains 'context_plist_sha256' "$ACCEPT_GATE" \
        "evidence context plist hash"
    assert_contains '--expected-agent-sha' "$ACCEPT_GATE" \
        "deploy context-agent hash"
    assert_contains '--expected-web-bridge-sha' "$ACCEPT_GATE" \
        "deploy web bridge hash"
    assert_contains '--expected-model-sha' "$ACCEPT_GATE" \
        "deploy Core ML hash"
    assert_contains '--expected-plist-sha' "$ACCEPT_GATE" \
        "deploy context plist hash"
    assert_contains 'all six candidate hashes' "$DEPLOY_GATE" \
        "immutable six-artifact gate"
}

test_deployment_does_not_require_passwordless_sudo() {
    assert_not_contains 'sudo -n' "$ACCEPT_GATE" \
        "acceptance gate passwordless sudo"
    assert_not_contains 'sudo -n' "$DEPLOY_GATE" \
        "deploy gate passwordless sudo"
    assert_not_contains 'sudo -n' "$REPO_ROOT/scripts/pipeline.sh" \
        "pipeline passwordless sudo"
    assert_not_contains 'sudo -n' "$REPO_ROOT/scripts/deploy.sh" \
        "quick deploy passwordless sudo"
    assert_contains 'normal `sudo`' "$ACCEPT_GATE" \
        "acceptance trust-boundary documentation"
}

run_test "acceptance binds all six deployment artifacts" \
    test_acceptance_binds_six_artifact_hashes
run_test "deployment uses normal sudo at the trust boundary" \
    test_deployment_does_not_require_passwordless_sudo

echo "1..$TESTS_RUN"
[ "$TESTS_FAILED" -eq 0 ] || exit 1
