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
        "$root/usr/local/libexec" \
        "$root/usr/local/bin" \
        "$root/var/lib/apollo"

    printf '%s\n' daemon-old > "$root/usr/local/libexec/apollo-optimizerd"
    printf '%s\n' ctl-old > "$root/usr/local/bin/apollo-optimizerctl"
    printf '%s\n' daemon-new > "$root/private/tmp/apollo-optimizerd-candidate"
    printf '%s\n' ctl-new > "$root/private/tmp/apollo-optimizerctl-candidate"
    printf '%s\n' '{"version":2,"marker":"state-current"}' \
        > "$root/var/lib/apollo/learned_state.json"
    printf '%s\n' '{"version":2,"marker":"state-previous"}' \
        > "$root/var/lib/apollo/learned_state.json.previous"
    : > "$root/private/tmp/apollo-optimizerd-candidate.signature-ok"
    : > "$root/private/tmp/apollo-optimizerctl-candidate.signature-ok"

    /bin/chmod 0755 \
        "$root/usr/local/libexec/apollo-optimizerd" \
        "$root/usr/local/bin/apollo-optimizerctl" \
        "$root/private/tmp/apollo-optimizerd-candidate" \
        "$root/private/tmp/apollo-optimizerctl-candidate"
}

run_helper() {
    APOLLO_DEPLOY_TEST_ROOT="$root" /bin/sh "$DEPLOY_SCRIPT" "$@"
}

deploy_fixture() {
    if ! run_helper deploy > "$root/deploy.stdout" 2> "$root/deploy.stderr"; then
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

    manifest="$backup/manifest"
    [ -f "$manifest" ] || fail "manifest was not created"
    assert_eq 600 "$(/usr/bin/stat -f %Lp "$manifest")" "manifest mode"
    assert_eq apollo-deploy-v1 "$(manifest_value "$manifest" format)" \
        "manifest format"
    assert_eq system/com.eduardocortez.systemoptimizerd \
        "$(manifest_value "$manifest" label)" "manifest service label"
    assert_eq "$(sha256 "$backup/apollo-optimizerd")" \
        "$(manifest_value "$manifest" daemon_installed_sha256)" \
        "manifest daemon hash"
    assert_eq "$(sha256 "$backup/apollo-optimizerctl")" \
        "$(manifest_value "$manifest" ctl_installed_sha256)" \
        "manifest ctl hash"
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
    assert_file_content "$root/var/lib/apollo/learned_state.json" \
        '{"version":2,"marker":"state-current"}' \
        "rolled back current state"
    assert_file_content "$root/var/lib/apollo/learned_state.json.previous" \
        '{"version":2,"marker":"state-previous"}' \
        "rolled back previous state"
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
    run_helper deploy > "$root/deploy.stdout" 2> "$root/deploy.stderr"
    status=$?
    set -e

    assert_eq 65 "$status" "future state exit status"
    assert_contains "future learned-state schema: 3" "$root/deploy.stderr" \
        "future state diagnostic"
    assert_file_content "$root/usr/local/libexec/apollo-optimizerd" daemon-old \
        "daemon after rejected future state"
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
run_test "rollback rejects a malformed manifest" \
    test_rollback_rejects_malformed_manifest

echo "1..$TESTS_RUN"
[ "$TESTS_FAILED" -eq 0 ] || exit 1
