#!/bin/bash
# RE Lab — launch the reverse engineering VM
# Usage: ./scripts/re-lab.sh [start|stop|ssh|status|snapshot|setup]
#
# Prerequisites:
#   brew install cirruslabs/cli/tart
#   tart clone ghcr.io/cirruslabs/macos-sequoia-vanilla:latest re-lab
#   tart set re-lab --memory 4096 --cpu 2

VM_NAME="re-lab"

check_tart() {
  if ! command -v tart &>/dev/null; then
    echo "ERROR: tart not found. Install with:"
    echo "  brew install cirruslabs/cli/tart"
    exit 1
  fi
}

case "$1" in
  start)
    check_tart
    echo "Starting RE lab VM (headless)..."
    tart run "$VM_NAME" --no-graphics &
    echo "Waiting for VM to get an IP..."
    for i in $(seq 1 30); do
      IP=$(tart ip "$VM_NAME" 2>/dev/null)
      if [ -n "$IP" ]; then
        echo "VM is up at $IP"
        echo "SSH: ssh admin@$IP"
        echo "Or:  ./scripts/re-lab.sh ssh"
        exit 0
      fi
      sleep 2
    done
    echo "VM started but no IP yet. Try: tart ip $VM_NAME"
    ;;

  stop)
    check_tart
    echo "Stopping RE lab VM..."
    tart stop "$VM_NAME"
    ;;

  ssh)
    check_tart
    IP=$(tart ip "$VM_NAME" 2>/dev/null)
    if [ -z "$IP" ]; then
      echo "VM not running or no IP. Start with: $0 start"
      exit 1
    fi
    echo "Connecting to $IP..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@"$IP"
    ;;

  status)
    check_tart
    echo "=== Tart VMs ==="
    tart list
    echo ""
    IP=$(tart ip "$VM_NAME" 2>/dev/null)
    if [ -n "$IP" ]; then
      echo "re-lab is RUNNING at $IP"
    else
      echo "re-lab is NOT running (or not yet booted)"
    fi
    ;;

  snapshot)
    check_tart
    SNAP_NAME="${2:-re-lab-snap-$(date +%Y%m%d-%H%M)}"
    echo "Creating snapshot: $SNAP_NAME"
    tart clone "$VM_NAME" "$SNAP_NAME"
    echo "Snapshot saved as: $SNAP_NAME"
    ;;

  setup)
    check_tart
    IP=$(tart ip "$VM_NAME" 2>/dev/null)
    if [ -z "$IP" ]; then
      echo "VM not running. Start with: $0 start"
      exit 1
    fi
    echo "Installing RE tools in VM at $IP..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@"$IP" bash <<'REMOTE'
      set -e
      echo "--- Installing Xcode CLI tools ---"
      xcode-select --install 2>/dev/null || echo "(already installed or requires manual interaction)"

      echo "--- Installing Homebrew ---"
      if ! command -v brew &>/dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"
      fi

      echo "--- Installing RE tools ---"
      brew install rizin radare2 binutils

      echo "--- Optional: Hopper (GUI) ---"
      echo "Run manually if desired: brew install --cask hopper-disassembler"

      echo "--- Done. Verify with: lldb --version, r2 -v, rz-bin -v ---"
REMOTE
    ;;

  *)
    echo "RE Lab VM Helper"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start      Boot VM in headless mode, wait for IP"
    echo "  stop       Shut down VM"
    echo "  ssh        SSH into running VM"
    echo "  status     Show VM state and IP"
    echo "  snapshot   Clone current VM state (optional name arg)"
    echo "  setup      Install RE tools inside VM (VM must be running)"
    echo ""
    echo "First-time setup:"
    echo "  brew install cirruslabs/cli/tart"
    echo "  tart clone ghcr.io/cirruslabs/macos-sequoia-vanilla:latest re-lab"
    echo "  tart set re-lab --memory 4096 --cpu 2"
    echo "  $0 start"
    echo "  # Boot into Recovery Mode, run: csrutil disable"
    echo "  $0 stop && $0 start  # reboot with SIP off"
    echo "  $0 setup"
    ;;
esac
