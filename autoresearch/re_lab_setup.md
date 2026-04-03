# RE Lab Setup — macOS VM for Reverse Engineering

## Environment Summary

- **Host**: MacBook Air M1, macOS (Apple Silicon / arm64)
- **Host disk**: 228 GB total, ~6 GB free — **NOT enough to pull a Tart image**
- **Virtualization**: Tart (Apple Hypervisor.framework, no Apple signature required)

---

## Current Status

| Component | Status |
|-----------|--------|
| Tart | NOT installed |
| Homebrew | Installed (5.1.3) |
| Disk space | INSUFFICIENT (6 GB free, need ≥50 GB) |
| VM `re-lab` | Does not exist yet |

---

## Blocker: Disk Space

The host has only **6 GB free** on a 228 GB drive (98% full). A Tart macOS Sequoia image
requires approximately 25–35 GB to pull and store. **Free at least 50 GB before proceeding.**

Suggested cleanup actions:
```bash
# Check large directories
du -sh ~/Downloads/* | sort -hr | head -20
du -sh ~/Library/Caches/* | sort -hr | head -20

# Tart stores VMs here (empty now):
du -sh ~/.tart/ 2>/dev/null

# Xcode / simulators (often multi-GB)
du -sh ~/Library/Developer/CoreSimulator/Caches/ 2>/dev/null

# Docker images if present
docker system df 2>/dev/null
```

---

## Setup Steps (once disk space is freed)

### 1. Install Tart

```bash
brew install cirruslabs/cli/tart
tart --version
```

### 2. Pull the vanilla Sequoia image

```bash
# ~25-35 GB download + storage
tart clone ghcr.io/cirruslabs/macos-sequoia-vanilla:latest re-lab
```

### 3. Configure VM resources (4 GB RAM, 2 CPUs — appropriate for 8 GB host)

```bash
tart set re-lab --memory 4096 --cpu 2
```

### 4. Disable SIP inside the VM

SIP cannot be disabled via a simple flag from the host on macOS guests — it must be done
from Recovery Mode inside the VM.

```bash
# Start VM with GUI (first boot only, to access Recovery Mode)
tart run re-lab

# Inside VM:
# 1. Shut down the VM
# 2. Hold the power button equivalent at next boot to enter Recovery Mode
#    (for Tart VMs: restart the VM and immediately hold Cmd+R, or use the
#     Startup Security Utility in the VM's Apple menu if accessible)
# 3. Open Terminal in Recovery Mode
csrutil disable
reboot

# Verify SIP is off:
csrutil status
# Expected: System Integrity Protection status: disabled.
```

Alternatively, some Tart builds support boot-args injection:
```bash
# This may work to set CSR flags at boot (not guaranteed on all Tart versions):
tart run re-lab -- nvram boot-args="csr-active-config=0xff000000"
```

### 5. Install RE tools inside the VM

Use the helper script:
```bash
./scripts/re-lab.sh start
./scripts/re-lab.sh setup    # installs rizin, radare2, binutils via Homebrew
```

Or manually via SSH:
```bash
./scripts/re-lab.sh ssh
# Then inside VM:
xcode-select --install          # lldb, otool, nm, objdump, dwarfdump
brew install rizin               # Cutter/rizin (Ghidra alternative, open source)
brew install radare2             # r2 CLI analysis suite
brew install binutils            # GNU binutils (cross-tool complementary)
brew install --cask hopper-disassembler  # GUI disassembler (license required for full features)
```

---

## RE Tool Stack Inside VM

| Tool | Purpose | How to get |
|------|---------|------------|
| `lldb` | Debugger — can attach to ANY process (SIP off) | Xcode CLI tools |
| `otool` | Mach-O inspection, dylib deps, disassembly | Xcode CLI tools |
| `nm` | Symbol table lister | Xcode CLI tools |
| `dwarfdump` | DWARF debug info | Xcode CLI tools |
| `objdump` | General object file disassembly | Xcode CLI tools or binutils |
| `dtruss` | dtrace-based syscall tracing (like strace) | macOS built-in (works with SIP off) |
| `dtrace` | Full dynamic tracing framework | macOS built-in |
| `rizin` / `rz-bin` | Binary analysis, scripting, Cutter GUI | `brew install rizin` |
| `radare2` | CLI binary analysis, scripting | `brew install radare2` |
| Hopper | GUI disassembler with pseudocode | `brew install --cask hopper-disassembler` |

---

## What You Can Do in the VM That You Cannot Do on the Host

### Requires SIP disabled:

| Capability | Host (SIP on) | VM (SIP off) |
|-----------|--------------|-------------|
| `lldb` attach to system processes (launchd, notifyd, etc.) | BLOCKED | YES |
| `dtruss` on system daemons | BLOCKED | YES |
| Patch binaries in `/usr/lib`, `/usr/libexec` | BLOCKED | YES |
| Write to SIP-protected paths (`/System`, `/usr`, `/sbin`) | BLOCKED | YES |
| `task_for_pid()` on arbitrary processes without entitlements | BLOCKED | YES |
| Load unsigned kernel extensions | BLOCKED | YES |
| Run unsigned binaries freely (no Gatekeeper quarantine) | Requires approval | YES |
| Modify NVRAM boot-args | BLOCKED | YES |
| Use DTrace probes on system processes | Limited | YES |

### Always works (no SIP needed):
- Analyze your own binaries with lldb / r2 / rizin
- Static analysis of any binary you have read access to
- Instrument processes you own

---

## Helper Script

`scripts/re-lab.sh` manages the VM lifecycle:

```bash
./scripts/re-lab.sh start      # Boot headless, wait for IP
./scripts/re-lab.sh ssh        # SSH into running VM
./scripts/re-lab.sh stop       # Shut down
./scripts/re-lab.sh status     # Show IP and state
./scripts/re-lab.sh snapshot [name]  # Clone current state for safe experiments
./scripts/re-lab.sh setup      # Install RE tools inside VM
```

---

## Recommended Workflow

```
Host: identify binary/process to analyze
  └─> ./scripts/re-lab.sh start
  └─> scp target-binary admin@$(tart ip re-lab):~/targets/
  └─> ./scripts/re-lab.sh ssh
       └─> VM: lldb / r2 / dtruss experiment
       └─> VM: notes / output saved to ~/experiments/
  └─> ./scripts/re-lab.sh snapshot experiment-name  (save state)
  └─> (optional) tart delete bad-experiment         (destroy and restore)
Host: retrieve results via scp
```

### Safe experimentation pattern

```bash
# Before risky experiment: snapshot
./scripts/re-lab.sh snapshot pre-kernel-patch

# Run experiment
./scripts/re-lab.sh ssh
# ... do destructive things ...

# If VM is broken: restore from snapshot
tart delete re-lab
tart clone pre-kernel-patch re-lab
```

---

## Notes

- Tart uses Apple's `Hypervisor.framework` — no kernel extensions, no Apple
  silicon entitlements required, works on any M-series Mac.
- The VM is isolated: anything you break stays in the VM.
- VMs are stored in `~/.tart/vms/` as sparse disk images.
- Default SSH credentials for cirruslabs vanilla images: user `admin`, password `admin`.
  Change this after first boot for any long-lived VM.
- For lldb use inside VM, `sudo lldb` is usually needed to attach to other processes even
  with SIP off. The `admin` user should have sudo rights in the vanilla image.
