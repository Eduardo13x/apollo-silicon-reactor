#!/bin/sh
# Isolated behavioral tests for scripts/apollo-deploy. No root privileges required.
set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
DEPLOY_SCRIPT="$REPO_ROOT/scripts/apollo-deploy"
TESTS_RUN=0
TESTS_FAILED=0

fail() {
    echo "# $*" >&2
    return 1
}

assert_eq() {
    expected=$1
    actual=$2
    context=$3
    [ "$actual" = "$expected" ] ||
        fail "$context: expected '$expected', got '$actual'"
}

assert_contains() {
    needle=$1
    file=$2
    context=$3
    /usr/bin/grep -F "$needle" "$file" >/dev/null 2>&1 ||
        fail "$context: missing '$needle' in $file"
}

assert_file_content() {
    file=$1
    expected=$2
    context=$3
    [ -f "$file" ] || fail "$context: missing $file"
    actual=$(/bin/cat "$file")
    assert_eq "$expected" "$actual" "$context"
}

sha256() {
    /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

manifest_value() {
    /usr/bin/awk -F= -v key="$2" \
        '$1 == key { print substr($0, length(key) + 2); exit }' "$1"
}

cleanup_fixture() {
    case "${root:-}" in
        /private/tmp/apollo-deploy-test.*) /bin/rm -rf "$root" ;;
    esac
}

new_fixture() {
    root=$(/usr/bin/mktemp -d /private/tmp/apollo-deploy-test.XXXXXX)
    trap cleanup_fixture EXIT HUP INT TERM

    /bin/mkdir -p \
        "$root/private/tmp" \
        "$root/private/var/lib/apollo" \
        "$root/usr/local/libexec" \
        "$root/usr/local/bin" \
        "$root/usr/local/share/apollo/models" \
        "$root/Library/LaunchAgents"
    # Mirror the real macOS /var -> /private/var alias. Rollback must accept
    # the helper's textual /var path even though physical canonicalization
    # resolves through /private/var.
    /bin/ln -s private/var "$root/var"

    printf '%s\n' daemon-old > "$root/usr/local/libexec/apollo-optimizerd"
    printf '%s\n' ctl-old > "$root/usr/local/bin/apollo-optimizerctl"
    printf '%s\n' agent-old > "$root/usr/local/libexec/apollo-context-agent"
    printf '%s\n' bridge-old > "$root/usr/local/libexec/apollo-web-bridge"
    printf '%s\n' model-old > "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel"
    printf '%s\n' plist-old > "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist"
    printf '%s\n' daemon-new > "$root/private/tmp/apollo-optimizerd-candidate"
    printf '%s\n' ctl-new > "$root/private/tmp/apollo-optimizerctl-candidate"
    printf '%s\n' agent-new > "$root/private/tmp/apollo-context-agent-candidate"
    printf '%s\n' bridge-new > "$root/private/tmp/apollo-web-bridge-candidate"
    printf '%s\n' model-new > "$root/private/tmp/apollo-temporal-v1.mlmodel-candidate"
    printf '%s\n' plist-new > "$root/private/tmp/com.eduardocortez.apollo-context-agent.plist-candidate"
    printf '%s\n' '{"version":2,"marker":"state-current"}' \
        > "$root/var/lib/apollo/learned_state.json"
    printf '%s\n' '{"version":2,"marker":"state-previous"}' \
        > "$root/var/lib/apollo/learned_state.json.previous"
    : > "$root/private/tmp/apollo-optimizerd-candidate.signature-ok"
    : > "$root/private/tmp/apollo-optimizerctl-candidate.signature-ok"
    : > "$root/private/tmp/apollo-context-agent-candidate.signature-ok"
    : > "$root/private/tmp/apollo-web-bridge-candidate.signature-ok"

    /bin/chmod 0755 \
        "$root/usr/local/libexec/apollo-optimizerd" \
        "$root/usr/local/bin/apollo-optimizerctl" \
        "$root/usr/local/libexec/apollo-context-agent" \
        "$root/usr/local/libexec/apollo-web-bridge" \
        "$root/private/tmp/apollo-optimizerd-candidate" \
        "$root/private/tmp/apollo-optimizerctl-candidate" \
        "$root/private/tmp/apollo-context-agent-candidate"
    /bin/chmod 0755 "$root/private/tmp/apollo-web-bridge-candidate"
}

run_helper() {
    APOLLO_DEPLOY_TEST_ROOT="$root" /bin/sh "$DEPLOY_SCRIPT" "$@"
}

run_deploy_helper() {
    run_helper deploy \
        "$(sha256 "$root/private/tmp/apollo-optimizerd-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-optimizerctl-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-context-agent-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-web-bridge-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-temporal-v1.mlmodel-candidate")" \
        "$(sha256 "$root/private/tmp/com.eduardocortez.apollo-context-agent.plist-candidate")"
}

deploy_fixture() {
    if ! run_deploy_helper > "$root/deploy.stdout" 2> "$root/deploy.stderr"; then
        /bin/cat "$root/deploy.stderr" >&2
        return 1
    fi
}

only_backup() {
    set -- "$root"/var/lib/apollo/backups/deploy-*
    [ "$#" -eq 1 ] && [ -d "$1" ] ||
        fail "expected exactly one deploy backup"
    backup=$1
}

test_deploy_backs_up_state_pair_and_manifest() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup

    assert_file_content "$backup/apollo-optimizerd" daemon-old \
        "daemon backup"
    assert_file_content "$backup/apollo-optimizerctl" ctl-old \
        "ctl backup"
    assert_file_content "$backup/apollo-context-agent" agent-old \
        "context agent backup"
    assert_file_content "$backup/apollo-web-bridge" bridge-old \
        "web bridge backup"
    assert_file_content "$backup/apollo-temporal-v1.mlmodel" model-old \
        "Core ML model backup"
    assert_file_content "$backup/com.eduardocortez.apollo-context-agent.plist" plist-old \
        "context plist backup"
    assert_file_content "$backup/learned_state.json" \
        '{"version":2,"marker":"state-current"}' \
        "current state backup"
    assert_file_content "$backup/learned_state.json.previous" \
        '{"version":2,"marker":"state-previous"}' \
        "previous state backup"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "deployed daemon"
    assert_file_content "$root/usr/local/bin/apollo-optimizerctl" ctl-new \
        "deployed ctl"
    assert_file_content "$root/usr/local/libexec/apollo-context-agent" agent-new \
        "deployed context agent"
    assert_file_content "$root/usr/local/libexec/apollo-web-bridge" bridge-new \
        "deployed web bridge"
    assert_file_content "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel" model-new \
        "deployed Core ML model"
    assert_file_content "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist" plist-new \
        "deployed context plist"

    manifest="$backup/manifest"
    [ -f "$manifest" ] || fail "manifest was not created"
    assert_eq 600 "$(/usr/bin/stat -f %Lp "$manifest")" "manifest mode"
    assert_eq apollo-deploy-v2 "$(manifest_value "$manifest" format)" \
        "manifest format"
    assert_eq system/com.eduardocortez.systemoptimizerd \
        "$(manifest_value "$manifest" label)" "manifest service label"
    assert_eq "$(sha256 "$backup/apollo-optimizerd")" \
        "$(manifest_value "$manifest" daemon_installed_sha256)" \
        "manifest daemon hash"
    assert_eq "$(sha256 "$backup/apollo-optimizerctl")" \
        "$(manifest_value "$manifest" ctl_installed_sha256)" \
        "manifest ctl hash"
    assert_eq "$(sha256 "$backup/apollo-context-agent")" \
        "$(manifest_value "$manifest" agent_installed_sha256)" \
        "manifest context-agent hash"
    assert_eq "$(sha256 "$backup/apollo-web-bridge")" \
        "$(manifest_value "$manifest" web_bridge_installed_sha256)" \
        "manifest web bridge hash"
    assert_eq "$(sha256 "$backup/apollo-temporal-v1.mlmodel")" \
        "$(manifest_value "$manifest" model_installed_sha256)" \
        "manifest Core ML hash"
    assert_eq "$(sha256 "$backup/com.eduardocortez.apollo-context-agent.plist")" \
        "$(manifest_value "$manifest" agent_plist_installed_sha256)" \
        "manifest context plist hash"
    assert_eq "$(sha256 "$backup/learned_state.json")" \
        "$(manifest_value "$manifest" learned_state_sha256)" \
        "manifest current state hash"
    assert_eq "$(sha256 "$backup/learned_state.json.previous")" \
        "$(manifest_value "$manifest" learned_state_previous_sha256)" \
        "manifest previous state hash"
)

test_rollback_restores_state_pair_and_binaries() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup

    printf '%s\n' '{"version":2,"marker":"state-mutated"}' \
        > "$root/var/lib/apollo/learned_state.json"
    printf '%s\n' '{"version":2,"marker":"previous-mutated"}' \
        > "$root/var/lib/apollo/learned_state.json.previous"

    run_helper rollback "$backup" > "$root/rollback.stdout" 2> "$root/rollback.stderr"

    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-old \
        "rolled back daemon"
    assert_file_content "$root/usr/local/bin/apollo-optimizerctl" ctl-old \
        "rolled back ctl"
    assert_file_content "$root/usr/local/libexec/apollo-context-agent" agent-old \
        "rolled back context agent"
    assert_file_content "$root/usr/local/libexec/apollo-web-bridge" bridge-old \
        "rolled back web bridge"
    assert_file_content "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel" model-old \
        "rolled back Core ML model"
    assert_file_content "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist" plist-old \
        "rolled back context plist"
    assert_file_content "$root/var/lib/apollo/learned_state.json" \
        '{"version":2,"marker":"state-current"}' \
        "rolled back current state"
    assert_file_content "$root/var/lib/apollo/learned_state.json.previous" \
        '{"version":2,"marker":"state-previous"}' \
        "rolled back previous state"
)

test_first_fabric_deploy_rolls_back_to_absence() (
    set -eu
    new_fixture
    /bin/rm -f \
        "$root/usr/local/libexec/apollo-context-agent" \
        "$root/usr/local/libexec/apollo-web-bridge" \
        "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel" \
        "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist"
    deploy_fixture
    only_backup
    assert_eq absent "$(manifest_value "$backup/manifest" agent_installed_sha256)" \
        "absent agent manifest"
    assert_eq absent "$(manifest_value "$backup/manifest" web_bridge_installed_sha256)" \
        "absent web bridge manifest"

    run_helper rollback "$backup" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    [ ! -e "$root/usr/local/libexec/apollo-context-agent" ] \
        || fail "first-deploy rollback retained context agent"
    [ ! -e "$root/usr/local/libexec/apollo-web-bridge" ] \
        || fail "first-deploy rollback retained web bridge"
    [ ! -e "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel" ] \
        || fail "first-deploy rollback retained model"
    [ ! -e "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist" ] \
        || fail "first-deploy rollback retained plist"
)

test_rollback_rejects_corrupt_hash_before_changes() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup
    printf '%s\n' corrupted >> "$backup/apollo-optimizerd"

    set +e
    run_helper rollback "$backup" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "corrupt backup exit status"
    assert_contains "daemon backup hash mismatch" "$root/rollback.stderr" \
        "corrupt backup diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "daemon after rejected corrupt rollback"
    assert_file_content "$root/var/lib/apollo/learned_state.json" \
        '{"version":2,"marker":"state-current"}' \
        "state after rejected corrupt rollback"
)

test_rollback_rejects_parent_traversal() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup

    outside="$root/var/lib/apollo/outside-backup"
    /bin/cp -R "$backup" "$outside"
    /bin/mkdir "$root/var/lib/apollo/backups/deploy-hop"
    traversal="$root/var/lib/apollo/backups/deploy-hop/../../outside-backup"

    set +e
    run_helper rollback "$traversal" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "parent traversal exit status"
    assert_contains "invalid backup directory" "$root/rollback.stderr" \
        "parent traversal diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "daemon after rejected parent traversal"
)

test_rollback_rejects_backup_directory_symlink() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup
    link="$root/var/lib/apollo/backups/deploy-symlink"
    /bin/ln -s "$backup" "$link"

    set +e
    run_helper rollback "$link" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "backup symlink exit status"
    assert_contains "invalid backup directory" "$root/rollback.stderr" \
        "backup symlink diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "daemon after rejected backup symlink"
)

test_rollback_rejects_intermediate_directory_symlink() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup

    outside="$root/var/lib/apollo/outside-parent"
    /bin/mkdir "$outside"
    /bin/cp -R "$backup" "$outside/archive"
    link="$root/var/lib/apollo/backups/deploy-link"
    /bin/ln -s "$outside" "$link"
    through_link="$link/archive"

    set +e
    run_helper rollback "$through_link" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "intermediate symlink exit status"
    assert_contains "invalid backup directory" "$root/rollback.stderr" \
        "intermediate symlink diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "daemon after rejected intermediate symlink"
)

test_deploy_rejects_future_learned_state() (
    set -eu
    new_fixture
    printf '%s\n' '{"version":3,"marker":"future"}' \
        > "$root/var/lib/apollo/learned_state.json"

    set +e
    run_deploy_helper > "$root/deploy.stdout" 2> "$root/deploy.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "future state exit status"
    assert_contains "future learned-state schema: 3" "$root/deploy.stderr" \
        "future state diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-old \
        "daemon after rejected future state"
)

test_deploy_rejects_a_staged_model_hash_mismatch_before_changes() (
    set -eu
    new_fixture
    bad_hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

    set +e
    run_helper deploy \
        "$(sha256 "$root/private/tmp/apollo-optimizerd-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-optimizerctl-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-context-agent-candidate")" \
        "$(sha256 "$root/private/tmp/apollo-web-bridge-candidate")" \
        "$bad_hash" \
        "$(sha256 "$root/private/tmp/com.eduardocortez.apollo-context-agent.plist-candidate")" \
        > "$root/deploy.stdout" 2> "$root/deploy.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "model hash mismatch exit status"
    assert_contains "model candidate hash mismatch" "$root/deploy.stderr" \
        "model hash mismatch diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-old \
        "daemon after rejected model hash"
)

test_deploy_waits_for_transient_runtime_readiness() (
    set -eu
    new_fixture
    printf '%s\n' 2 > "$root/runtime-failures-remaining"

    deploy_fixture

    assert_file_content "$root/runtime-failures-remaining" 0 \
        "transient runtime readiness attempts"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "deployed daemon after transient readiness"
)

test_mid_copy_failure_restores_complete_previous_release() (
    set -eu
    new_fixture
    printf '%s\n' 3 > "$root/fail-copy-after"

    set +e
    run_deploy_helper > "$root/deploy.stdout" 2> "$root/deploy.stderr"
    status=$?
    set -e

    assert_eq 70 "$status" "mid-copy failure exit status"
    only_backup
    assert_contains "APOLLO_BACKUP_PATH=$backup" "$root/deploy.stderr" \
        "mid-copy failure backup path"
    assert_contains "previous release restored" "$root/deploy.stderr" \
        "mid-copy restoration diagnostic"
    [ "$(/bin/cat "$root/service-bootstrap-attempts")" -ge 1 ] || \
        fail "service restart was not attempted after mid-copy failure"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-old \
        "daemon after mid-copy rollback"
    assert_file_content "$root/usr/local/bin/apollo-optimizerctl" ctl-old \
        "ctl after mid-copy rollback"
    assert_file_content "$root/usr/local/libexec/apollo-context-agent" agent-old \
        "context agent after mid-copy rollback"
    assert_file_content "$root/usr/local/libexec/apollo-web-bridge" bridge-old \
        "web bridge after mid-copy rollback"
    assert_file_content "$root/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel" model-old \
        "model after mid-copy rollback"
    assert_file_content "$root/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist" plist-old \
        "context plist after mid-copy rollback"
    assert_file_content "$root/var/lib/apollo/learned_state.json" \
        '{"version":2,"marker":"state-current"}' \
        "state after mid-copy rollback"
    assert_file_content "$root/var/lib/apollo/learned_state.json.previous" \
        '{"version":2,"marker":"state-previous"}' \
        "previous state after mid-copy rollback"
)

test_rollback_rejects_malformed_manifest() (
    set -eu
    new_fixture
    deploy_fixture
    only_backup
    printf '%s\n' 'broken' > "$backup/manifest"

    set +e
    run_helper rollback "$backup" > "$root/rollback.stdout" 2> "$root/rollback.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "malformed manifest exit status"
    assert_contains "invalid backup directory" "$root/rollback.stderr" \
        "malformed manifest diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-new \
        "daemon after rejected malformed manifest"
)

run_test() {
    TESTS_RUN=$((TESTS_RUN + 1))
    name=$1
    shift
    "$@"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "ok $TESTS_RUN - $name"
    else
        echo "not ok $TESTS_RUN - $name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

run_test "deploy backs up current+previous state and manifest" \
    test_deploy_backs_up_state_pair_and_manifest
run_test "rollback restores current+previous state and binaries" \
    test_rollback_restores_state_pair_and_binaries
run_test "first fabric deploy rollback restores absent optional artifacts" \
    test_first_fabric_deploy_rolls_back_to_absence
run_test "rollback rejects a corrupt backup hash before changes" \
    test_rollback_rejects_corrupt_hash_before_changes
run_test "rollback rejects parent traversal" \
    test_rollback_rejects_parent_traversal
run_test "rollback rejects a backup directory symlink" \
    test_rollback_rejects_backup_directory_symlink
run_test "rollback rejects an intermediate directory symlink" \
    test_rollback_rejects_intermediate_directory_symlink
run_test "deploy rejects a future learned-state schema" \
    test_deploy_rejects_future_learned_state
run_test "deploy rejects a staged model hash mismatch before changes" \
    test_deploy_rejects_a_staged_model_hash_mismatch_before_changes
run_test "deploy waits for transient runtime readiness" \
    test_deploy_waits_for_transient_runtime_readiness
run_test "mid-copy failure restores the complete previous release" \
    test_mid_copy_failure_restores_complete_previous_release
run_test "rollback rejects a malformed manifest" \
    test_rollback_rejects_malformed_manifest

echo "1..$TESTS_RUN"
[ "$TESTS_FAILED" -eq 0 ] || exit 1
