#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cat > "$tmp_dir/cargo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$APOLLO_TEST_CARGO_LOG"
case "${1:-}" in
  test) printf 'test result: ok. 1 passed; 0 failed; 0 ignored\n' ;;
  clippy) printf 'warning: mock warning for score parser\n' ;;
esac
exit 0
EOF
chmod +x "$tmp_dir/cargo"

export APOLLO_TEST_CARGO_LOG="$tmp_dir/cargo.log"
export APOLLO_UNAME_OVERRIDE=Darwin
export APOLLO_HW_CORES_OVERRIDE=10
export APOLLO_P_CORES_OVERRIDE=4
export APOLLO_E_CORES_OVERRIDE=6
PATH="$tmp_dir:$PATH" "$repo_root/scripts/autoresearch-verify.sh" >/dev/null

grep -q '^build .*--features adaptive-multicore' "$APOLLO_TEST_CARGO_LOG"
grep -q '^test .*--features adaptive-multicore' "$APOLLO_TEST_CARGO_LOG"
grep -q '^clippy .*--features adaptive-multicore' "$APOLLO_TEST_CARGO_LOG"

echo "autoresearch profile propagation: ok"
