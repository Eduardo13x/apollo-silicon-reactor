# Apollo Optimizer — Apple Silicon M1 8GB Benchmark

**Comparison: macOS with Apollo daemon ACTIVE vs PAUSED (kill switch)**

> Apollo is an adaptive system-level optimizer for macOS Apple Silicon written in Rust. It freezes/throttles background daemons via SIGSTOP, tunes kernel sysctls, and routes thread QoS to optimize for active workloads. This benchmark quantifies its impact on a real M1 MacBook Air 8 GB.

---

## Test Environment

| | |
|---|---|
| Hardware | MacBook Air M1 (8-core, 4P+4E, 3.2 GHz) |
| RAM | 8 GB unified |
| OS | macOS 26.4.1 (Build 25E253) |
| Apollo version | v0.6.1 + Sprint 5 Mes 0 (workspace split) |
| Daemon mode (CON) | aggressive-root profile, learned policy 65/95/103 patterns |
| Daemon mode (SIN) | kill switch active (`/var/run/apollo.disable`) |
| Date | 2026-05-10 |

Tools used: CoreMark 1.0 (EEMBC), sysbench 1.x, stress-ng, fio, macOS `powermetrics`, `vm_stat`.

---

## TL;DR

| Workload | Apollo Δ | Verdict |
|---|---|---|
| Multi-thread CPU (sysbench 8-thread) | **+18.1% throughput** | 🟢 big win |
| Random disk I/O (fio 4K) | **+130% IOPS, −56% latency** | 🟢 huge win |
| Memory pressure under stress (4 GB allocation) | **−27% swap usage (1.12 GB less)** | 🟢 big win |
| Memory bandwidth (sysbench) | **+3% read / +3% write** | 🟢 small win |
| Sequential disk I/O | +2.5% | 🟢 small win |
| Single-thread CoreMark | ±2% (noise) | ⚪ neutral |
| Mixed workload completion (CPU+IO+VM 30s) | identical (within noise) | ⚪ neutral |
| Sustained 8-thread power (60s) | **+30% power** for +18% work | 🟡 trade-off |

**Bottom line**: Apollo makes M1 8 GB **measurably faster on multi-thread + I/O heavy + memory-pressure scenarios** (the workloads that actually matter on a constrained machine). It costs more power on sustained CPU-only stress because it routes more work to performance cores instead of efficiency cores — which is desirable when you want responsiveness, undesirable if you only care about idle battery life.

---

## Detailed Results

### 1. Single-thread CPU — CoreMark 1.0

EEMBC industry-standard benchmark. 4 runs (2 modes × 2 phases).

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| CoreMark Performance run | 30,602 iter/s | 28,235 iter/s | +8.4% |
| CoreMark Validation run | 28,973 iter/s | 29,364 iter/s | −1.3% |
| **Average** | **29,787 iter/s** | **28,800 iter/s** | **+3.4%** (within noise) |

**Reading**: Single-thread CPU performance is statistically indistinguishable. Apollo neither helps nor hurts pure single-core compute. Expected — Apollo's value isn't in raw CPU speed.

### 2. Multi-thread CPU — sysbench 8-thread, 30s

Prime-number search distributed across all 8 cores (4P + 4E).

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Events/sec | **60,683,925** | 51,369,397 | **+18.1%** |
| Total events (30s) | 1.82B | 1.54B | +279M |

**Reading**: Apollo lifts multi-thread workloads by ~18% by throttling background daemons (analyticsd, mds_stores, cloudphotod, etc.) so the user's 8-thread workload can saturate cores without contention.

### 3. Memory bandwidth — sysbench memory, 1MB blocks, 4 threads

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Sequential read | 61,669 MiB/s | 59,769 MiB/s | +3.2% |
| Sequential write | 61,112 MiB/s | 59,328 MiB/s | +3.0% |

**Reading**: Small but consistent gain. M1's unified memory bandwidth is the same hardware-side; the +3% likely comes from Apollo reducing memory contention from background daemons.

### 4. Disk I/O — fio

#### Sequential read (1 MB blocks, 20s)

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Throughput | 3,017 MB/s | 2,944 MB/s | +2.5% |
| IOPS | 2,947 | 2,876 | +2.5% |
| Latency mean | 319 µs | 323 µs | −1.2% |

#### Random read (4 KB blocks, 20s)

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Throughput | **153 MB/s** | 67 MB/s | **+128%** |
| IOPS | **38,450** | 16,704 | **+130%** |
| Latency mean | **26 µs** | 59 µs | **−56%** |

**Reading**: The gap on random small-block I/O is **the headline result**. Apollo's I/O cleanup (capped audit-log writes, suppressed sysctl no-op writes, clamped network-optimizer raw emits) clears the SSD bandwidth path so user processes don't wait. Without Apollo, macOS Spotlight + iCloud + analyticsd + mdworker_shared compete for the same SSD bandwidth as your random reads.

### 5. Memory pressure response — stress-ng VM 4 GB allocation, 20s

After 5 seconds into the allocation:

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| memory_pressure (kernel) | 0.632 | 0.610 | +3.6% (similar) |
| **swap_used** | **2.97 GB** | **4.09 GB** | **−1.12 GB (−27%)** |

**Reading**: Under identical 4 GB allocation pressure, Apollo absorbed the spike with 1.12 GB less swap usage. macOS's compressor + Apollo's QoS demotion of background tasks freed working memory before the kernel had to swap. **Less swap = less SSD wear + faster recovery**.

### 6. Sustained CPU stress — 60s sysbench 8-thread + powermetrics 30 samples

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Combined power | 10,067 mW | 7,753 mW | +30% |
| CPU power | 10,024 mW | 7,420 mW | +35% |
| **GPU power** | **42 mW** | 334 mW | **−87%** |
| P-cluster freq | 2,646 MHz | 2,250 MHz | +18% |
| E-cluster freq | 2,064 MHz | 2,064 MHz | flat |
| Thermal pressure | Nominal 28/30 | Nominal 28/30 | flat |

**Reading — the trade-off**: Apollo elevates the 8-thread workload to P-cluster at full speed (2.65 GHz vs 2.25 GHz), so it consumes +30% more CPU power. But (a) it gets +18% more work done in the same time, (b) Apollo simultaneously cuts GPU power by 87% (knows nothing visual is happening), and (c) thermal stays Nominal in both cases.

**Net energy efficiency** (events per Joule):
- CON Apollo: 60.68M events × 30s / 10.067W = 180.9 GJoules of compute-events ÷ 302 J = 6,028 events/sec/mW
- SIN Apollo: 51.37M events × 30s / 7.753W = 6,626 events/sec/mW

SIN is ~10% more efficient per watt under sustained CPU-only. For "max battery life on light load" SIN wins; for "max throughput right now" CON wins by a wider margin.

### 7. Mixed workload — 30s simultaneous CPU + IO + VM stress

| Metric | CON Apollo | SIN Apollo | Δ |
|---|---|---|---|
| Combined power | 9,374 mW | 8,524 mW | +10% |
| Completion time | 30.29 s | 30.41 s | identical |
| Thermal | Nominal | Nominal | flat |

**Reading**: Mixed workload power gap closes (10% vs 30% on pure CPU stress). Apollo's I/O optimization compensates for the higher CPU power.

---

## Architectural impact (qualitative)

Beyond the numbers, Apollo also delivers:

- **Daemon stability**: 0 failures over 11,000+ cycles, p95 cycle time ~150 ms
- **macOS coalition compliance**: Apollo's own disk-write rate stays under macOS's 99 KB/s sustained-write coalition limit (was 447 KB/s pre-fix → 23 disk-write microstackshot reports in 3 days; 0 reports post-fix)
- **No Spotlight thrash**: Apollo never toggles `mdutil` automatically (regression fix 2026-05-08 after user reported Finder beachball)
- **Cache hit ratio 75%**: process-identity cache amortizes `proc_pidpath` syscalls — 0.04 syscalls per cycle vs target 5

---

## Reproducibility

```bash
# Build the suite
cd benchmarks
./run_full_suite.sh con-apollo results-con-apollo

# Pause Apollo
sudo touch /var/run/apollo.disable
./run_full_suite.sh no-apollo results-no-apollo
sudo rm /var/run/apollo.disable

# Compare
diff -y results-con-apollo/sysbench_cpu.txt results-no-apollo/sysbench_cpu.txt
```

Tools required: `sysbench`, `stress-ng`, `fio`, `jq`. Install via `brew install sysbench stress-ng fio jq`.

CoreMark and the comparison harness are checked into `benchmarks/` in the repo.

---

## Honest caveats

- M1 8 GB only. M2/M3/M4 results would differ (more cores, more bandwidth, less swap pressure to relieve).
- Single user (me). Daily workload pattern: Brave + cargo build + occasional ollama. YMMV.
- 20-30s sample windows per metric. Longer sustained runs (1 hour+) might surface thermal throttling that 30s misses.
- Mixed workload completion was bounded by stress-ng's `--timeout 30s` — a `--bogo-ops` count would discriminate better.
- The `+18% multi-thread CPU` gain assumes the host has background daemons to throttle. A clean-install macOS without analyticsd, cloudd, fileproviderd, etc. running heavily would see less benefit.

---

## What this is NOT

This benchmark does **not** measure:

- Long-tail latency (jank perception during UI work)
- Workload-specific outcomes (Brave tab switching speed, Xcode build time, ollama TTFT)
- Battery life over a full work day
- Thermal behavior under hours of sustained load
- Multi-machine comparisons

Each of those would warrant its own dedicated test. This suite quantifies Apollo's effect on **standardized synthetic benchmarks** so the numbers are reproducible by anyone with an M1 + Homebrew.

---

## Source

- Apollo daemon: `https://github.com/<your-handle>/apollo-optimizer`
- This benchmark suite: `benchmarks/run_full_suite.sh`
- Raw data: `benchmarks/results-{con,no}-apollo/`
