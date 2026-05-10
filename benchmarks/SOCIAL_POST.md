# Apollo Optimizer — M1 8 GB benchmark (LinkedIn / Medium short)

## TL;DR

Built a Rust system-level optimizer for macOS Apple Silicon. Benchmarked ON vs OFF (kill switch) on real M1 8 GB hardware. Results:

| Workload | Apollo Δ |
|---|---|
| Multi-thread CPU (sysbench 8-thread) | **+18.1% throughput** |
| Random disk I/O (fio 4 KB) | **+128% throughput, +130% IOPS, −56% latency** |
| Memory pressure under stress | **−27% swap usage (1.12 GB less SSD wear)** |
| Memory bandwidth | +3% read / +3% write |
| Sequential disk I/O | +2.5% |
| Single-thread CPU (CoreMark) | ±2% (noise — neutral) |
| Sustained 8-thread power | +30% (but +18% more work) |

## What is Apollo

[Apollo](https://github.com/<your>/apollo-optimizer) is a single-user macOS daemon that:

- Freezes idle background daemons via SIGSTOP (analyticsd, cloudphotod, etc.)
- Tunes 16 kernel sysctls (TCP buffers, vnode cache, compressor)
- Routes thread QoS to P/E clusters intelligently
- Runs an LLM teacher (local Gemma 4) for adaptive policy learning

50K LoC Rust, single crate originally, now a Cargo workspace with a 91K-LoC `apollo-engine` library.

## The headline number

**Random 4 KB disk I/O: +130% IOPS, latency cut from 59 µs to 26 µs.**

That's the result of capping Apollo's own audit-log writes (was 447 KB/s sustained, hit macOS's 99 KB/s resource-coalition limit), suppressing sysctl no-op writes, and clamping network-optimizer's raw 4 MB sendspace emits to allowlist range. macOS was throttling Apollo's I/O QoS, which leaked back into the user's disk reads.

## Honest trade-offs

- Pure single-thread CPU: identical (within noise). Apollo doesn't make a single core faster.
- Sustained 8-thread CPU stress: Apollo elevates work to P-cluster (2.65 GHz vs 2.25 GHz on E-cluster fallback), so power +30%. But it also cuts GPU power 87% (knows nothing visual is happening). Net: more throughput, more power.
- Idle / light load: separate single-sample test showed Apollo saves ~13% power for the same workload (5.2W → 4.5W). Different regime than sustained stress.

## Test environment

| | |
|---|---|
| Hardware | MacBook Air M1 (4P+4E, 3.2 GHz), 8 GB unified |
| OS | macOS 26.4.1 (Build 25E253) |
| Apollo | v0.6.1 + Sprint 5 Mes 0 (workspace split) |
| Tools | CoreMark 1.0, sysbench, stress-ng, fio, powermetrics |

## Reproducibility

```bash
git clone <repo>
cd apollo-optimizer/benchmarks
brew install sysbench stress-ng fio jq

./run_full_suite.sh con-apollo results-con-apollo
sudo touch /var/run/apollo.disable
./run_full_suite.sh no-apollo results-no-apollo
sudo rm /var/run/apollo.disable
```

Raw JSON / log outputs in `benchmarks/results-{con,no}-apollo/`.

## What this isn't

Not a Geekbench-killer. Not a daily-driver battery test. This is a **standardized synthetic benchmark of Apollo's measurable impact on memory-pressure and I/O-bound scenarios** — the categories where M1 8 GB gets bottlenecked in real life.

Apollo doesn't make M1 a faster CPU. It makes M1's *effective performance budget* go further by clearing path for active work and absorbing memory pressure before it hits swap.

---

If you have an M1/M2 with 8 GB and Brave/Code/Slack/Spotify open all day, this exists for you.

Code: github.com/<your>/apollo-optimizer
Full results + README: [link]
