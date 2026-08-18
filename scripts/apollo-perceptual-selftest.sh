#!/bin/zsh
# End-to-end self-test for the Perceptual Interaction Layer.
#
# Verifies every hop that can be checked without a human touching the browser,
# then defers exactly one manual step. Prints PASS / PARTIAL / FAIL and names
# the hop responsible.
set -u
EXT_DIR="${APOLLO_EXT_DIR:-extensions/apollo-webflow-chromium}"
CTL="${APOLLO_CTL:-apollo-optimizerctl}"
fail=0; partial=0
# node usually arrives through nvm, which a non-interactive shell never sources.
# Resolve it explicitly rather than reporting a healthy file as unparseable.
NODE="$(command -v node 2>/dev/null)"
if [ -z "$NODE" ]; then
  for candidate in "$HOME"/.nvm/versions/node/*/bin/node /opt/homebrew/bin/node /usr/local/bin/node; do
    [ -x "$candidate" ] && NODE="$candidate" && break
  done
fi
say() { printf '  [%s] %-22s %s\n' "$1" "$2" "$3"; }

echo "Apollo perceptual self-test"
echo

# ── 1. package integrity ───────────────────────────────────────────────────
required=(manifest.json background.js content.js protocol.js native-host.json)
missing=()
for f in $required; do [ -f "$EXT_DIR/$f" ] || missing+=("$f"); done
if [ ${#missing[@]} -eq 0 ]; then
  say ok "package files" "all ${#required[@]} present"
else
  say FAIL "package files" "missing: ${missing[*]}"; fail=1
fi

MANIFEST_V=$(python3 -c "import json;print(json.load(open('$EXT_DIR/manifest.json'))['version'])" 2>/dev/null)
if [ -n "$NODE" ]; then
  PROTO_V=$("$NODE" -e "process.stdout.write(String(require('$PWD/$EXT_DIR/protocol.js').EXTENSION_VERSION))" 2>/dev/null)
  if [ -n "$MANIFEST_V" ] && [ "$MANIFEST_V" = "$PROTO_V" ]; then
    say ok "version agreement" "manifest=$MANIFEST_V protocol=$PROTO_V"
  else
    say FAIL "version agreement" "manifest=$MANIFEST_V protocol=$PROTO_V"; fail=1
  fi
  syntax_ok=1
  for f in background.js content.js protocol.js; do
    "$NODE" --check "$EXT_DIR/$f" 2>/dev/null || { say FAIL "syntax $f" "does not parse"; fail=1; syntax_ok=0; }
  done
  [ $syntax_ok -eq 1 ] && say ok "syntax" "background/content/protocol parse"
else
  say PARTIAL "node toolchain" "node not found; syntax and version unchecked"; partial=1
fi

if grep -q "document_start" "$EXT_DIR/background.js"; then
  say ok "injection timing" "content script registered at document_start"
else
  say FAIL "injection timing" "document_start not requested"; fail=1
fi

if grep -q '"nativeMessaging"' "$EXT_DIR/manifest.json"; then
  say ok "permissions" "nativeMessaging declared"
else
  say FAIL "permissions" "nativeMessaging missing"; fail=1
fi

# ── 2. native host ─────────────────────────────────────────────────────────
HOST_PATH="$HOME/Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts/com.eduardocortez.apollo_webflow.json"
if [ -f "$HOST_PATH" ]; then
  say ok "native host" "manifest installed for Brave"
else
  say PARTIAL "native host" "not installed at Brave path"; partial=1
fi

# The native host manifest names the exact path Brave will execute; check that
# one rather than PATH, which Brave never consults.
BRIDGE_PATH=$(python3 -c "import json,os;p=os.path.expanduser('$HOST_PATH');print(json.load(open(p))['path'])" 2>/dev/null)
if [ -n "$BRIDGE_PATH" ] && [ -x "$BRIDGE_PATH" ]; then
  say ok "bridge binary" "$BRIDGE_PATH"
else
  say FAIL "bridge binary" "native host points at ${BRIDGE_PATH:-?} which is not executable"; fail=1
fi

# The four binaries must come from one build: WebFlowEvent is
# deny_unknown_fields, so a stale component rejects every newer payload and the
# daemon never learns the event existed.
STALE=()
for b in /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-web-bridge \
         /usr/local/libexec/apollo-context-agent /usr/local/bin/apollo-optimizerctl; do
  [ -x "$b" ] || { STALE+=("$b:missing"); continue; }
done
NEWEST=$(ls -t /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-web-bridge \
  /usr/local/libexec/apollo-context-agent /usr/local/bin/apollo-optimizerctl 2>/dev/null | head -1)
OLDEST=$(ls -t /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-web-bridge \
  /usr/local/libexec/apollo-context-agent /usr/local/bin/apollo-optimizerctl 2>/dev/null | tail -1)
SKEW=$(python3 -c "
import os,sys
try: print(int(abs(os.path.getmtime('$NEWEST')-os.path.getmtime('$OLDEST'))))
except Exception: print(-1)" 2>/dev/null)
if [ ${#STALE[@]} -ne 0 ]; then
  say FAIL "binary set" "missing: ${STALE[*]}"; fail=1
elif [ "$SKEW" -ge 0 ] && [ "$SKEW" -le 600 ]; then
  say ok "binary set" "all four within ${SKEW}s of each other"
else
  say FAIL "binary set" "components differ by ${SKEW}s — oldest: $(basename $OLDEST)"; fail=1
fi

# ── 3. daemon circuit ──────────────────────────────────────────────────────
if $CTL metrics >/dev/null 2>&1; then
  say ok "daemon socket" "reachable"
else
  say FAIL "daemon socket" "unreachable"; fail=1
fi

DOCTOR=$($CTL perceptual-doctor 2>/dev/null)
VERDICT=$(printf '%s' "$DOCTOR" | awk '/verdict:/{print $2}')
case "$VERDICT" in
  READY_FOR_0B)        say ok      "circuit" "$VERDICT" ;;
  OBSERVATION_PARTIAL) say PARTIAL "circuit" "$VERDICT — browser may be idle"; partial=1 ;;
  STALE_EXTENSION)     say PARTIAL "circuit" "$VERDICT — reload the extension"; partial=1 ;;
  NO_DATA)             say PARTIAL "circuit" "$VERDICT — extension not loaded"; partial=1 ;;
  "")                  say FAIL    "circuit" "perceptual-doctor unavailable"; fail=1 ;;
  *)                   say FAIL    "circuit" "$VERDICT"; fail=1 ;;
esac

echo
if [ $fail -ne 0 ]; then
  echo "  RESULT: FAIL"
  exit 2
elif [ $partial -ne 0 ]; then
  echo "  RESULT: PARTIAL"
  printf '\n  Manual step (Brave blocks programmatic extension reload):\n'
  printf '    1. Open brave://extensions, enable Developer mode\n'
  printf '    2. Load unpacked → %s\n' "$(cd "$EXT_DIR" && pwd)"
  printf '    3. Click the Apollo WebFlow toolbar icon to grant host access\n'
  printf '    4. Interact with any page, then re-run this self-test\n'
  exit 1
else
  echo "  RESULT: PASS"
  exit 0
fi
