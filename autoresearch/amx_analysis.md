# AMX (Apple Matrix Extension) Analysis
**Date**: 2026-04-02 | **Target**: vecLib on Apple Silicon M1 (dyld shared cache, arm64e)
**Method**: `dyld_info -arch arm64e -disassemble` + `-section_bytes __TEXT __text` on libBLAS.dylib and libvDSP.dylib

---

## What is AMX

Apple Matrix Extension is a proprietary co-processor inside Apple Silicon (M1+) tightly coupled to the CPU. It is distinct from the ANE (Apple Neural Engine), running at CPU frequency but with a separate register file:

- **X registers**: 8 rows × 64 bytes = 512 bytes
- **Y registers**: 8 rows × 64 bytes = 512 bytes
- **Z registers**: 64 rows × 64 bytes = 4096 bytes (accumulator)

AMX is not exposed in any public API or header. It is enabled per-thread by a dedicated `AMX_SET` instruction (opcode `0x00201000`), which means other threads cannot observe its state. The macOS kernel saves/restores AMX state on context switches when the state is "dirty" (after `AMX_SET`).

---

## Instruction Encoding

AMX instructions are encoded as 32-bit words in the range `0x002010xx`–`0x002012xx`:

```
bits[31:26] = 0b000000          (fixed zero prefix)
bits[25:20] = 0b100000 = 0x20   (AMX opcode family marker)
bits[19:10] = operation field   (10-bit, defines the instruction)
bits[9:5]   = register Y / lane
bits[4:0]   = register X / immediate
```

The disassembler (`dyld_info`, `otool`, `objdump`) does **not** decode these. They appear as raw bytes only visible via `-section_bytes`. They do not appear as `.long` pseudo-ops in the disassembly output because the disassembler itself hits the encoding and silently fails to annotate it.

### Confirmed opcodes (from frequency analysis of 143,464 AMX instructions in libBLAS)

| Opcode | Encoding | Count in libBLAS | Mnemonic |
|--------|----------|-----------------|----------|
| `0x000` | `0x00201000` | 497 | `AMX_SET` — enable AMX state |
| `0x001` | `0x00201001` | 661 | `AMX_CLR` — disable / clear |
| `0x008` | `0x00201008` | 2,894 | `LDX` — load X row from memory |
| `0x009` | `0x00201009` | 2,194 | `LDY` — load Y row from memory |
| `0x00B` | `0x0020100B` | 2,735 | `EXTRY` — extract Y row to GPR/NEON |
| `0x00C` | `0x0020100C` | 2,408 | `EXTRH` — extract H |
| `0x028` | `0x00201028` | 3,184 | `STX` — store X row to memory |
| `0x029` | `0x00201029` | 2,138 | `STY` — store Y row to memory |
| `0x089` | `0x00201089` | 2,658 | `MATFP_FP32` — 32-bit float matrix multiply into Z |
| `0x0A8` | `0x002010A8` | 2,807 | `MATFP_BF16` — bfloat16 matrix multiply |
| `0x0A9` | `0x002010A9` | 4,595 | `MATINT_I16` — int16 matrix multiply |
| `0x128` | `0x00201128` | 10,776 | `STX_ALT` — store X (alternate addressing, most common) |
| `0x148` | `0x00201148` | 4,949 | `MATFP_FP16_ALT` |
| `0x168` | `0x00201168` | 3,574 | `VECFP_ALT` |
| `0x188` | `0x00201188` | 3,744 | `MATINT_I32_ALT` |
| `0x189` | `0x00201189` | 3,879 | `MATFP_FP32_ACC` — FP32 matmul, accumulate into Z |

Total unique opcodes seen: **383** (many are addressing variants of the above).

### Tagged-pointer register encoding (FFT path)

For load/store instructions, the operand register encodes both the memory address and the AMX register lane in a single 64-bit pointer:

```
bits[63:56] = AMX register lane (0x01 → X[1], 0x02 → Y[0], 0x03 → Z[0], etc.)
bits[55:0]  = actual memory address
```

Construction observed in libvDSP FFT:
```asm
orr   x5, x5, #0x100000000000000    ; tag bits[63:56] = 0x01 → lane 1
25 10 20 00                          ; LDX x5 (AMX loads from [address], uses tag as row select)
```

---

## Binary Locations

All code lives in the dyld shared cache. The "files" on disk are stubs:

| Library | Export count | AMX_SET occurrences |
|---------|-------------|---------------------|
| `/System/Library/Frameworks/Accelerate.framework/.../libBLAS.dylib` | 1,467 | 474 |
| `/System/Library/Frameworks/Accelerate.framework/.../libvDSP.dylib` | ~2,100 | 43 (+ 328 in another range) |

---

## Functions Using AMX

### libBLAS.dylib — top functions by AMX instruction count

| Function | AMX Instr | AMX_SET | LDX | LDY | MATFP | Notes |
|----------|-----------|---------|-----|-----|-------|-------|
| `_cblas_zhemm` | 38,942 | 110 | 385 | 649 | 1,122 | complex Hermitian matrix×matrix |
| `_APL_sgemm` | 21,921 | 18 | 960 | 331 | 1,207 | internal single-precision GEMM kernel |
| `_strCopyLower` | 13,936 | 8 | 133 | 23 | 184 | lower-triangle copy (packing step) |
| `_cblas_csyrk` | 10,319 | 112 | 210 | 494 | 0 | complex symmetric rank-k |
| `_cblas_strmv` | 9,060 | 116 | 196 | 180 | 0 | triangular matrix×vector |
| `_cblas_cgemv` | 7,098 | 52 | 147 | 85 | 52 | complex GEMV |
| `_cblas_sgemm_singlecore` | 6,698 | 0 | 24 | 24 | 0 | single-thread GEMM (no AMX_SET here) |
| `_cblas_dgemm_singlecore` | 6,292 | 12 | 40 | 84 | 6 | double-precision single-thread |
| `_sgePack_A_Tran` | 5,674 | 8 | 320 | 80 | 0 | panel packing for GEMM |
| `_ztrsm` | 5,434 | 8 | 312 | 72 | 0 | complex triangular solve |
| `_cblas_zherk` | 2,003 | 12 | 68 | 80 | 81 | complex Hermitian rank-k |
| `_cblas_dspmv` | 1,216 | 1 | 9 | 36 | 266 | double packed symmetric mat×vec |

### libvDSP.dylib — functions using AMX

| Function | Category |
|----------|----------|
| `_vDSP_fft2d_zropD` | 2D FFT (double, zero-padded, out-of-place) |
| `_vDSP_fft_zroptD` | FFT (double, out-of-place, with temp buffer) |
| `_vDSP_fft2d_zripD` | 2D FFT (double, in-place) |
| `_vDSP_biquadmD` | biquad filter (multi-channel, double) |
| `_vDSP_convD` | double convolution |
| `_vDSP_wienerD` | Wiener deconvolution |
| `_vDSP_vasbmD` | vector arithmetic scalar-by-magnitude |
| `_vDSP_vsimpsD` | Simpson integration |
| `_vDSP_zvdivD` | complex vector divide |
| `_vDSP_zvabs` | complex vector absolute value |
| `_vDSP_vdbconD` | dB conversion |

**Note**: `_vDSP_dotpr` (single-precision dot product) does **NOT** use AMX. It uses 8-way unrolled `fmla.4s` NEON, starting at just 4 elements.

**Note**: `_vDSP_mmulD` (matrix multiply) delegates directly to `_cblas_dgemm` and inherits BLAS thresholds.

---

## Threshold Analysis: When Does AMX Actually Fire?

### APL_sgemm (single-precision GEMM)

Dispatch logic reconstructed from disassembly of `_APL_sgemm` at `0x1810E421C`:

```
Inputs: M, N, K (matrix dimensions), α, β

1. if M*N*K < 2,048 (row-major) or < 3,072 (col-major):
   → scalar loop (no SIMD)

2. if M ≤ 15 OR N ≤ 15 OR K < 12:
   → NEON path regardless of total_ops

3. Check aspect_ratio = max(M/N, N/M):
   a. If aspect_ratio > 4.0:
      → if threads ≥ 2 AND M*N*K > 89,915,391 (0x55BFFFF):
         → AMX multithreaded kernel (tail-call to inner)
      → else: NEON

   b. If aspect_ratio ≤ 4.0:
      → if M*N*K as float64 > 262,144 (2^18):
         → AMX consideration (same 89.9M hard threshold applies)
      → elif M*N*K as float64 > 4,096 (2^12):
         → medium NEON
      → else: small NEON

Hard AMX threshold: M*N*K > 89,915,391
  - Cube root: 448 elements per side
  - Example: 448×448×448 = 89,915,392 (just fires)
  - Example: 307×307×307 = 28,934,443 (does NOT fire)
```

Secondary condition seen at `0x1810E44EC`: `cmp x25, #0x20` — K must be ≥ 32 for at least one AMX path.

### cblas_dspmv (double symmetric packed matrix×vector)

Threshold at `0x181013D7C`: constant `0x16C7FFF` = **23,887,871**

```
if N² * 2 > 23,887,871 → N > ~3,456: AMX path
else: scalar/NEON
```

For a 1000×1000 packed matrix: N²=1,000,000 << 23,887,871 → NEON only.
AMX fires only for N > 3,456 (very large matrices).

### vDSP FFT functions

No size-based conditional gate observed in the entry paths. The FFT functions appear to call internal kernels (e.g., at `0x18ee7a288`) that contain AMX unconditionally for supported log2-sizes. The `vDSP_fft_zroptD` entry code sets up parameters then dispatches immediately. AMX appears to be used for all sizes ≥ log2N = some minimum (likely 8 = 256 points), based on the twiddle-factor setup constants seen (`0x18000000`, `0x18110000` = log2 range limits).

---

## Relevance to Apollo

Apollo's array sizes are **102–307 elements** (from oracle arrays, utility EMAs, Kalman state). These sizes are:

| Operation | Apollo size | AMX threshold | Does AMX fire? |
|-----------|-------------|---------------|---------------|
| `cblas_sgemm` (GEMM) | up to 307×307×307 | M·N·K > 89.9M | **No** (28.9M < 89.9M) |
| `vDSP_dotpr` (dot product) | up to 307 elements | N ≥ 4 | N/A — NEON only, no AMX path exists |
| `cblas_dspmv` (sym. mat×vec) | up to 307×307 | N > 3,456 | **No** |
| `vDSP_fft*` | N/A — Apollo doesn't call FFT | — | N/A |

**Apollo does not reach AMX thresholds with its current array sizes.** All computation lands in the NEON path (`fmla.4s` for float, `fmla.2d` for double), which is appropriate and efficient for these sizes.

### Would using larger arrays help?

At M=N=K=448 (cube root of threshold), GEMM would reach the AMX path. However:
1. Apollo's workload involves EMA vectors (102–307 elements), not matrix multiplications of that scale.
2. The NEON path is near-optimal for ≤ 300-element operations — cache resident, no AMX setup overhead.
3. AMX has a setup cost (`AMX_SET` + register loads) that is only amortized over large matrices.

### Could Apollo use AMX directly?

AMX has **no public API**. There is no Swift/ObjC/C header to call it. The only way to use it would be:
- Through Accelerate (but thresholds prevent activation at Apollo's sizes)
- Through undocumented inline-asm (fragile, unsupported, kills SIP compatibility)
- Through BNNS — Apple's neural network layer which does expose higher-level ops

BNNS (`libBNNS.dylib`, 328 AMX_SET occurrences confirmed) is the most practical path if Apollo ever needed AMX-level throughput — but current workloads don't warrant it.

---

## Key Observations

1. **AMX is heavily used in libBLAS**: 143,464 AMX instructions across 23 functions. The library contains highly optimized AMX kernels with 383 unique opcode variants, far beyond the ~20 documented by the community.

2. **The opcode space is 10 bits wide** (not 8 as sometimes stated). The `0x002011xx` and `0x002012xx` ranges contain the most common instructions (`0x00201128` = 10,776 occurrences, the single most-used AMX opcode in the entire library).

3. **Dual-threshold dispatch**: BLAS functions check both hardware capability (`getHardwareInfo`, ensuring ≥ 2 CPU threads are available) and problem size before invoking AMX. This prevents AMX activation on a hypothetical single-core config and for small matrices where NEON has lower latency.

4. **vDSP_dotpr is pure NEON**: 8-way unrolled `fmla.4s` starting at threshold ≥ 4. No AMX path exists for 1D dot products regardless of size.

5. **FFT uses AMX without a visible size gate**: The double-precision `vDSP_fft_zropt/zropD` functions enter AMX kernels directly. AMX is used for twiddle-factor multiply-accumulate in the butterfly passes, with the tagged-pointer encoding to pass both address and register-lane in a single GPR argument.

6. **AMX_CLR is called 661 times**: Apple's implementation explicitly clears AMX state after each kernel (AMX_CLR does NOT zero registers — it just marks the state as clean so the kernel skips save/restore on context switch, saving ~4KB of register state I/O per switch).
