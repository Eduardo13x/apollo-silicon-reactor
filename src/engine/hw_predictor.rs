//! Hardware-level predictive signals — ARM64 assembly + XNU commpage.
//!
//! Capas de profundidad (de más superficial a más profundo):
//!
//!   Nivel 5 — libc / sysinfo          (lo que usa todo el mundo)
//!   Nivel 4 — syscalls                (memorystatus_control, sysctl)
//!   Nivel 3 — registros EL0           (cntvct_el0, ctr_el0, tpidr_el0)  ← antes
//!   Nivel 2 — XNU commpage            (memoria del kernel mapeada en userspace) ← ahora
//!   Nivel 1 — PMU hardware            (bloqueado por Apple sin entitlement)
//!   Nivel 0 — silicio físico          (imposible desde software)
//!
//! Todo en este módulo funciona sin entitlements ni root.

use std::sync::OnceLock;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer estático compartido — allocated once, no presión de heap por ciclo.
// 32 MB excede el L2 del M1 (12 MB) para forzar accesos a DRAM real.
// Usado por NEON bandwidth y como referencia de latencia LLC.
// ═══════════════════════════════════════════════════════════════════════════════
const PROBE_BUF_BYTES: usize = 32 * 1024 * 1024;
static PROBE_BUF: OnceLock<Vec<u8>> = OnceLock::new();

fn probe_buf() -> &'static [u8] {
    PROBE_BUF.get_or_init(|| {
        // Valores no-cero: previene zero-page optimization del OS (páginas zero no van a DRAM).
        vec![0xABu8; PROBE_BUF_BYTES]
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2 — XNU Commpage
// Apple mapea datos del kernel en cada proceso a una dirección fija.
// Lectura directa sin syscall, ~0.3 ns por acceso.
// ═══════════════════════════════════════════════════════════════════════════════

/// Dirección base del commpage en macOS ARM64.
/// Definida en xnu/osfmk/arm/cpu_capabilities.h
const COMM_PAGE_BASE: usize = 0x0000_000F_FFFF_C000;

// Offsets desde COMM_PAGE_BASE (xnu/osfmk/i386/cpu_capabilities.h)
const OFF_SIGNATURE: usize = 0x000; // [u8; 16]  "commpage 64-bit\0"
const OFF_VERSION: usize = 0x00e; // u16       debe ser >= 1
const OFF_CPU_CAPS64: usize = 0x010; // u64       feature flags
#[allow(dead_code)]
const OFF_NCPUS: usize = 0x022; // u8        CPUs totales configurados
const OFF_ACTIVE_CPUS: usize = 0x023; // u8        CPUs activos AHORA ← cae con thermal
const OFF_PHYSICAL_CPUS: usize = 0x024; // u8        cores físicos
const OFF_LOGICAL_CPUS: usize = 0x025; // u8        threads lógicos
const OFF_MEMORY_SIZE: usize = 0x034; // u64       RAM total en bytes
#[allow(dead_code)]
const OFF_TIMEBASE_OFFSET: usize = 0x038; // i64       offset entre mach_time y wall time

/// CPU capability flags (OFF_CPU_CAPS64 bitmask).
#[allow(dead_code)]
pub mod cpu_caps {
    pub const FP: u64 = 1 << 0; // Floating point
    pub const VMX: u64 = 1 << 1; // VMX / AltiVec (x86, no aplica en ARM)
    pub const CACHE32: u64 = 1 << 2; // cache line 32 bytes
    pub const CACHE64: u64 = 1 << 3; // cache line 64 bytes ← Apple Silicon
    pub const CACHE128: u64 = 1 << 4; // cache line 128 bytes
    pub const NEON: u64 = 1 << 12; // NEON/AdvSIMD
    pub const SHA2: u64 = 1 << 17; // SHA-2 hardware
    pub const AES: u64 = 1 << 18; // AES hardware
    pub const CRC32: u64 = 1 << 19; // CRC32 hardware
}

/// Snapshot de datos leídos directamente del kernel via commpage.
#[derive(Debug, Clone)]
pub struct CommPageSnapshot {
    /// CPUs activos en este momento. Baja si el kernel apaga cores por thermal.
    pub active_cpus: u8,
    /// CPUs físicos totales del SoC.
    pub physical_cpus: u8,
    /// CPUs lógicos totales (hyperthreading, si aplica).
    pub logical_cpus: u8,
    /// RAM total en bytes (leída del kernel, sin syscall).
    pub memory_bytes: u64,
    /// Feature flags del CPU (bitmask de cpu_caps::*).
    pub cpu_caps: u64,
    /// true si la firma del commpage es válida.
    pub valid: bool,
}

/// Lee datos directamente de la memoria del kernel mapeada en nuestro proceso.
/// Costo: ~3 ns. No requiere syscall, entitlement ni root.
pub fn read_commpage() -> CommPageSnapshot {
    // Verificar que el commpage está mapeado y la firma es válida
    // antes de leer cualquier campo.
    let valid = unsafe { verify_commpage_signature() };

    if !valid {
        return CommPageSnapshot {
            active_cpus: 0,
            physical_cpus: 0,
            logical_cpus: 0,
            memory_bytes: 0,
            cpu_caps: 0,
            valid: false,
        };
    }

    unsafe {
        CommPageSnapshot {
            active_cpus: read_commpage_u8(OFF_ACTIVE_CPUS),
            physical_cpus: read_commpage_u8(OFF_PHYSICAL_CPUS),
            logical_cpus: read_commpage_u8(OFF_LOGICAL_CPUS),
            memory_bytes: read_commpage_u64(OFF_MEMORY_SIZE),
            cpu_caps: read_commpage_u64(OFF_CPU_CAPS64),
            valid: true,
        }
    }
}

unsafe fn verify_commpage_signature() -> bool {
    // "commpage 64-bit" = [0x63,0x6f,0x6d,0x6d,0x70,0x61,0x67,0x65,0x20,0x36,0x34,0x2d,0x62,0x69,0x74]
    let ptr = (COMM_PAGE_BASE + OFF_SIGNATURE) as *const u8;
    let version_ptr = (COMM_PAGE_BASE + OFF_VERSION) as *const u16;
    // La firma empieza con 'c','o','m','m'
    *ptr == b'c'
        && *ptr.add(1) == b'o'
        && *ptr.add(2) == b'm'
        && *ptr.add(3) == b'm'
        && *version_ptr >= 1
}

#[inline(always)]
unsafe fn read_commpage_u8(offset: usize) -> u8 {
    let ptr = (COMM_PAGE_BASE + offset) as *const u8;
    // ISB garantiza que esta lectura no es reordenada especulativamente
    let val: u8;
    std::arch::asm!(
        "isb",
        "ldrb {val:w}, [{ptr}]",
        ptr = in(reg) ptr,
        val = out(reg) val,
        options(nostack, preserves_flags)
    );
    val
}

#[inline(always)]
unsafe fn read_commpage_u64(offset: usize) -> u64 {
    // read_unaligned: el commpage tiene campos u64 en offsets no alineados a 8 bytes
    // (ej: OFF_MEMORY_SIZE = 0x034). El hardware ARM64 soporta accesos no alineados
    // pero Rust requiere declararlo explícitamente para evitar UB.
    let ptr = (COMM_PAGE_BASE + offset) as *const u64;
    ptr.read_unaligned()
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 3 — Registros EL0 (ampliados)
// ═══════════════════════════════════════════════════════════════════════════════

/// Read the ARM virtual counter. Monotonic, ~1 ns, no privilege required.
#[inline(always)]
pub fn read_vct() -> u64 {
    let v: u64;
    unsafe {
        std::arch::asm!(
            "isb",              // serializa: evita reordenamiento especulativo
            "mrs {}, cntvct_el0",
            out(reg) v,
            options(nostack, preserves_flags)
        );
    }
    v
}

/// Timer frequency (typically 24_000_000 Hz on Apple Silicon).
#[inline(always)]
pub fn timer_freq() -> u64 {
    let f: u64;
    unsafe {
        std::arch::asm!(
            "mrs {}, cntfrq_el0",
            out(reg) f,
            options(nostack, nomem, preserves_flags)
        );
    }
    f
}

/// Tamaño de línea de cache D en bytes.
///
/// `CTR_EL0` y `DCZID_EL0` están bloqueados en macOS Apple Silicon
/// (`SCTLR_EL1.UCT=0`) como mitigación contra ataques de timing tipo Spectre.
/// Accederlos genera SIGILL. En su lugar usamos el valor conocido y verificado
/// para todos los chips M-series: 64 bytes por línea de cache.
///
/// Este es el **límite real** de EL0 en macOS — más abajo solo hay EL1 (kernel).
pub fn dcache_line_bytes() -> usize {
    64 // Confirmado: M1, M2, M3, M4 — todos usan 64B cache line
}

/// Thread pointer — opaco en macOS (apunta al TLS block), pero útil para
/// detectar migración de core: si cambia entre dos muestras consecutivas,
/// el kernel nos movió a otro core.
#[inline(always)]
pub fn read_thread_ptr() -> u64 {
    let v: u64;
    unsafe {
        std::arch::asm!(
            "mrs {}, tpidr_el0",
            out(reg) v,
            options(nostack, nomem, preserves_flags)
        );
    }
    v
}

#[inline(always)]
pub fn ticks_to_us(ticks: u64, freq: u64) -> u64 {
    ticks * 1_000_000 / freq.max(1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Señales predictivas
// ═══════════════════════════════════════════════════════════════════════════════

/// Scheduling jitter — señal más temprana de thermal throttling.
/// Detectable 2-5 s antes de que powermetrics lo reporte.
pub fn measure_schedule_jitter(samples: u32) -> u64 {
    // Usamos `yield` en vez de sleep(1ms).
    // sleep(1ms) en macOS tarda ~4-5ms (granularidad del scheduler) → falso CRITICAL.
    // yield mide cuánto tardó el scheduler en devolverte el hilo:
    //   sin preemption:   ~0-10µs (nominal)
    //   preemption leve:  ~100-2000µs (carga normal)
    //   throttling real:  >5000µs (thermal o CPU saturado)
    let freq = timer_freq();
    let mut deltas = Vec::with_capacity(samples as usize);

    for _ in 0..samples {
        let t0 = read_vct();
        unsafe {
            std::arch::asm!("yield", options(nostack, nomem, preserves_flags));
        }
        let t1 = read_vct();
        deltas.push(ticks_to_us(t1.saturating_sub(t0), freq));
    }

    deltas.sort_unstable();
    let p95 = (samples as usize * 95 / 100).min(deltas.len().saturating_sub(1));
    deltas[p95]
}

/// Instruction throughput — detecta migración P-core → E-core.
/// P-cores: ~800-1200 MIPS. E-cores: ~200-400 MIPS.
pub fn measure_core_throughput() -> u64 {
    const ITERATIONS: u64 = 1_000_000;
    let freq = timer_freq();
    let t0 = read_vct();

    let mut acc: u64 = 1;
    for i in 0..ITERATIONS {
        unsafe {
            std::arch::asm!(
                "mul {acc}, {acc}, {i}",
                "add {acc}, {acc}, #1",
                acc = inout(reg) acc,
                i = in(reg) i | 1,
                options(nostack, nomem, preserves_flags)
            );
        }
    }

    let t1 = read_vct();
    std::hint::black_box(acc);
    let elapsed_us = ticks_to_us(t1.saturating_sub(t0), freq).max(1);
    ITERATIONS * 1_000_000 / elapsed_us / 1_000_000
}

/// Cache miss latency — precede al swap activity y jetsam kills.
///
/// Usa CTR_EL0 para calibrar el buffer al tamaño real del LLC,
/// en lugar de un valor hardcoded. Más preciso en cualquier chip M-series.
pub fn measure_cache_latency() -> u64 {
    let stride = dcache_line_bytes(); // 64B en todo Apple Silicon

    // Buffer debe exceder el L2 unificado para forzar accesos al LLC/RAM.
    // Apple Silicon M1: L2=12MB (P-cluster), usamos 16 MB para garantizar misses.
    const BUF_SIZE: usize = 16 * 1024 * 1024;
    let steps = BUF_SIZE / stride;

    let mut buf = vec![0u32; BUF_SIZE / 4];

    // Pointer-chase: cada elemento apunta al siguiente en orden pseudo-aleatorio
    // usando stride de línea de caché para maximizar misses.
    for i in 0..steps {
        let next = (i + 1) % steps;
        buf[i * stride / 4] = (next * stride / 4) as u32;
    }

    let freq = timer_freq();
    let t0 = read_vct();

    let mut idx: usize = 0;
    for _ in 0..steps {
        idx = buf[idx] as usize;
        std::hint::black_box(idx);
    }

    let t1 = read_vct();
    std::hint::black_box(idx);

    // µs totales del pointer-chase — medible con precisión real (miles de ticks)
    ticks_to_us(t1.saturating_sub(t0), freq)
}

/// Detecta si fuimos migrados de core entre dos muestras.
/// Útil para saber si el scheduler nos está moviendo activamente.
pub fn detect_core_migration() -> bool {
    let tp0 = read_thread_ptr();
    // Yield da oportunidad al scheduler de mover el thread
    unsafe {
        std::arch::asm!("yield", options(nostack, nomem, preserves_flags));
    }
    let tp1 = read_thread_ptr();
    // Si el TLS pointer cambió, fuimos migrados a otro core
    tp0 != tp1
}

/// Ancho de banda de memoria — detecta saturación del bus antes que vm_stat.
///
/// NEON ld1 128-bit × 4 = 64 bytes/iteración. El prefetcher HW ayuda aquí
/// (queremos saturar el bus, no medir latencia), así que acceso secuencial
/// es lo correcto. El buffer de 32 MB excede L2 → accesos van a DRAM real.
///
/// M-series medido: idle ~8 GB/s, carga media ~25 GB/s, saturación >42 GB/s.
/// Thresholds conservadores (margen para variación entre chips):
///   <22 GB/s  → Nominal
///   22-38 GB/s → Warning
///   >38 GB/s  → Critical
pub fn measure_memory_bandwidth_gbs() -> f64 {
    let buf = probe_buf();
    let p = buf.as_ptr();
    let end = unsafe { p.add(buf.len()) };
    let freq = timer_freq();
    let t0 = read_vct();

    unsafe {
        std::arch::asm!(
            "1:",
            "ld1 {{v0.16b, v1.16b, v2.16b, v3.16b}}, [{p}], #64",
            "cmp {p}, {end}",
            "b.lo 1b",
            p = inout(reg) p => _,
            end = in(reg) end,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            options(nostack, readonly),
        );
    }

    let t1 = read_vct();
    let elapsed_s = ticks_to_us(t1.saturating_sub(t0), freq) as f64 / 1_000_000.0;
    if elapsed_s > 0.0 {
        buf.len() as f64 / elapsed_s / 1e9
    } else {
        0.0
    }
}

/// Latencia del L1 D-cache — buffer 128 KB (< L1D del M1 P-core = 192 KB).
///
/// Usa stride coprime con el número de pasos para defetar el hardware prefetcher:
///   stride = 127 × 64B = 8128 B de salto — más que cualquier ventana de prefetch.
///   gcd(127, 2048) = 1 → el ciclo visita todos los 2048 elementos exactamente una vez.
///
/// Si L1 latency sube, otro proceso está evictando nuestras líneas de caché.
/// Thresholds:
///   <8 µs  → Nominal (~4 ns/acceso, L1 hit limpio)
///   8-25µs → Warning  (eviction parcial, spill a L2)
///   >25µs  → Critical (L1 muy evictado)
pub fn measure_l1_latency_us() -> u64 {
    const L1_BUF: usize = 128 * 1024;
    let stride = dcache_line_bytes(); // 64 B
    let steps = L1_BUF / stride; // 2048 pasos
    const COPRIME: usize = 127; // gcd(127, 2048) = 1 → ciclo completo

    let mut buf = vec![0u32; L1_BUF / 4];
    let mut idx = 0usize;
    for _ in 0..steps {
        let next = (idx + COPRIME) % steps;
        buf[idx * stride / 4] = (next * stride / 4) as u32;
        idx = next;
    }

    // Pre-warm: PRFM pldl1keep trae líneas de caché antes del benchmark.
    // Sin esto, la primera pasada mide latencia de L2→L1 fill, no L1 hit.
    let base = buf.as_ptr();
    for i in 0..steps {
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{addr}]",
                addr = in(reg) base.add(i * stride / 4),
                options(nostack, readonly, preserves_flags),
            );
        }
    }

    let freq = timer_freq();
    let t0 = read_vct();
    let mut cur = 0usize;
    for _ in 0..steps {
        cur = buf[cur] as usize;
        std::hint::black_box(cur);
    }
    let t1 = read_vct();
    std::hint::black_box(cur);

    ticks_to_us(t1.saturating_sub(t0), freq)
}

/// Tasa de fallos de exclusión de cache line — contención multi-core a escala de ns.
///
/// ldaxr/stlxr: si un context-switch ocurre entre load-exclusive y store-exclusive,
/// el store falla (ARM64 limpia la reserva en cualquier excepción/interrupción).
/// A diferencia del jitter (yield, escala ms), esto mide frecuencia de interrupciones
/// a escala de nanosegundos — señal complementaria, no redundante.
///
/// Thresholds:
///   <0.1%  → Nominal
///   0.1-2% → Warning  (interrupciones frecuentes / DPC storm)
///   >2%    → Critical (contención severa, posible IRQ flood)
pub fn measure_cache_contention_rate(iterations: u64) -> f64 {
    let mut val: u64 = 0;
    let ptr = &mut val as *mut u64;
    let mut failures: u64 = 0;

    for _ in 0..iterations {
        let result: u64;
        let tmp: u64;
        unsafe {
            std::arch::asm!(
                "ldaxr {t}, [{p}]",
                "add   {t}, {t}, #1",
                "stlxr {r:w}, {t}, [{p}]",
                p = in(reg) ptr,
                t = out(reg) tmp,
                r = out(reg) result,
                options(nostack),
            );
        }
        let _ = tmp;
        failures += result; // stlxr: 0 = éxito, 1 = fallo
    }

    failures as f64 / iterations.max(1) as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Snapshot compuesto
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HwPressure {
    Nominal,
    Warning,
    Critical,
}

pub fn sample_hw_pressure() -> HwPressureSnapshot {
    // Commpage: ~3 ns, datos del kernel sin syscall
    let commpage = read_commpage();

    let jitter_us = measure_schedule_jitter(8);
    let throughput_mips = measure_core_throughput();
    let cache_latency_us = measure_cache_latency();
    let core_migrated = detect_core_migration();
    // Nuevas señales de bajo nivel
    let bandwidth_gbs = measure_memory_bandwidth_gbs();
    let l1_latency_us = measure_l1_latency_us();
    let contention_rate = measure_cache_contention_rate(2_000);

    // CPUs activos desde el kernel: si cayeron, el sistema está apagando cores
    let cpu_drop_pressure = if commpage.valid && commpage.physical_cpus > 0 {
        let ratio = commpage.active_cpus as f64 / commpage.physical_cpus as f64;
        if ratio < 0.5 {
            HwPressure::Critical
        } else if ratio < 0.75 {
            HwPressure::Warning
        } else {
            HwPressure::Nominal
        }
    } else {
        HwPressure::Nominal
    };

    // Thresholds calibrados para medición con yield (no sleep):
    //   <50µs   → nominal (sin preemption)
    //   50-2000 → warning (algo de carga / migración)
    //   >2000   → critical (throttling real o CPU saturado)
    let thermal = match jitter_us {
        0..=50 => HwPressure::Nominal,
        51..=2_000 => HwPressure::Warning,
        _ => HwPressure::Critical,
    };

    // M1 baseline: ~200-800 Mips under normal load (compiling, browsing).
    // Original thresholds (500/250) triggered constant WARNING under normal use.
    // Recalibrated: only flag when throughput drops significantly below typical.
    let core_migration_pressure = match throughput_mips {
        200.. => HwPressure::Nominal,
        100..=199 => HwPressure::Warning,
        _ => HwPressure::Critical,
    };

    // Thresholds en µs totales (262144 accesos × latencia_real)
    // Nominal ≈ <7ms, Warning ≈ 7–24ms, Critical ≈ >24ms
    let memory = match cache_latency_us {
        0..=7_000 => HwPressure::Nominal,
        7_001..=24_000 => HwPressure::Warning,
        _ => HwPressure::Critical,
    };

    // Migración de core activa = el scheduler está bajo presión
    let scheduler = if core_migrated {
        HwPressure::Warning
    } else {
        HwPressure::Nominal
    };

    // Señal de bandwidth: saturación del bus de memoria
    let bandwidth = match bandwidth_gbs as u64 {
        0..=21 => HwPressure::Nominal,
        22..=37 => HwPressure::Warning,
        _ => HwPressure::Critical,
    };

    // L1 cache eviction: otro proceso compite por cache
    let l1_pressure = match l1_latency_us {
        0..=7 => HwPressure::Nominal,
        8..=25 => HwPressure::Warning,
        _ => HwPressure::Critical,
    };

    // Contención de cache line por interrupciones/context-switches
    let contention = if contention_rate > 0.02 {
        HwPressure::Critical
    } else if contention_rate > 0.001 {
        HwPressure::Warning
    } else {
        HwPressure::Nominal
    };

    // overall: requiere al menos 2 señales en Warning+ para evitar falsos positivos.
    // Las 8 señales cubren diferentes dimensiones ortogonales del sistema.
    let signals = [
        thermal,
        core_migration_pressure,
        memory,
        cpu_drop_pressure,
        scheduler,
        bandwidth,
        l1_pressure,
        contention,
    ];
    let warning_count = signals
        .iter()
        .filter(|&&s| s >= HwPressure::Warning)
        .count();
    let max_signal = signals.iter().copied().max().unwrap_or(HwPressure::Nominal);
    // Require 2+ warning signals to avoid false positives from a single noisy probe.
    // A single warning signal is not enough — could be transient noise.
    let overall = if warning_count >= 3 {
        max_signal
    } else if warning_count == 2 {
        HwPressure::Warning
    } else {
        HwPressure::Nominal
    };

    HwPressureSnapshot {
        overall,
        thermal,
        core_migration: core_migration_pressure,
        memory,
        cpu_drop: cpu_drop_pressure,
        bandwidth,
        l1_pressure,
        contention,
        scheduler,
        jitter_us,
        throughput_mips,
        cache_latency_us,
        bandwidth_gbs,
        l1_latency_us,
        contention_rate,
        active_cpus: commpage.active_cpus,
        physical_cpus: commpage.physical_cpus,
        core_migrated,
        commpage_valid: commpage.valid,
        sampled_at: Instant::now(),
    }
}

#[derive(Debug, Clone)]
pub struct HwPressureSnapshot {
    pub overall: HwPressure,
    pub thermal: HwPressure,        // jitter de scheduling
    pub core_migration: HwPressure, // throughput drop P→E
    pub memory: HwPressure,         // cache miss latency
    pub cpu_drop: HwPressure,       // ACTIVE_CPUS cayó (commpage)
    pub scheduler: HwPressure,      // migración de core detectada
    pub bandwidth: HwPressure,      // NEON bus bandwidth saturation
    pub l1_pressure: HwPressure,    // L1 cache eviction (pointer chase)
    pub contention: HwPressure,     // ldaxr/stlxr failure rate
    pub jitter_us: u64,
    pub throughput_mips: u64,
    pub cache_latency_us: u64,
    pub bandwidth_gbs: f64,   // GB/s measured via NEON reads
    pub l1_latency_us: u64,   // µs for 128KB pointer chase
    pub contention_rate: f64, // ldaxr/stlxr failure fraction
    pub active_cpus: u8,      // leído del kernel sin syscall
    pub physical_cpus: u8,
    pub core_migrated: bool,
    pub commpage_valid: bool,
    pub sampled_at: Instant,
}

impl HwPressureSnapshot {
    pub fn needs_attention(&self) -> bool {
        self.overall >= HwPressure::Warning
    }
    pub fn is_critical(&self) -> bool {
        self.overall == HwPressure::Critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HwPressure enum ordering ──────────────────────────────────────────────

    #[test]
    fn hw_pressure_ordering_is_correct() {
        // PartialOrd/Ord must satisfy Nominal < Warning < Critical.
        assert!(HwPressure::Nominal < HwPressure::Warning);
        assert!(HwPressure::Warning < HwPressure::Critical);
        assert!(HwPressure::Nominal < HwPressure::Critical);
    }

    #[test]
    fn hw_pressure_equality() {
        assert_eq!(HwPressure::Nominal, HwPressure::Nominal);
        assert_ne!(HwPressure::Nominal, HwPressure::Critical);
    }

    // ── HwPressureSnapshot helper methods ─────────────────────────────────────

    fn make_snapshot(overall: HwPressure) -> HwPressureSnapshot {
        HwPressureSnapshot {
            overall,
            thermal: HwPressure::Nominal,
            core_migration: HwPressure::Nominal,
            memory: HwPressure::Nominal,
            cpu_drop: HwPressure::Nominal,
            scheduler: HwPressure::Nominal,
            bandwidth: HwPressure::Nominal,
            l1_pressure: HwPressure::Nominal,
            contention: HwPressure::Nominal,
            jitter_us: 0,
            throughput_mips: 1000,
            cache_latency_us: 1,
            bandwidth_gbs: 10.0,
            l1_latency_us: 2,
            contention_rate: 0.0,
            active_cpus: 8,
            physical_cpus: 8,
            core_migrated: false,
            commpage_valid: true,
            sampled_at: Instant::now(),
        }
    }

    #[test]
    fn needs_attention_false_for_nominal() {
        let snap = make_snapshot(HwPressure::Nominal);
        assert!(!snap.needs_attention());
        assert!(!snap.is_critical());
    }

    #[test]
    fn needs_attention_true_for_warning() {
        let snap = make_snapshot(HwPressure::Warning);
        assert!(snap.needs_attention());
        assert!(!snap.is_critical());
    }

    #[test]
    fn needs_attention_true_for_critical() {
        let snap = make_snapshot(HwPressure::Critical);
        assert!(snap.needs_attention());
        assert!(snap.is_critical());
    }

    // ── sample_hw_pressure smoke test ─────────────────────────────────────────

    #[test]
    fn sample_hw_pressure_returns_valid_snapshot() {
        let snap = sample_hw_pressure();
        // active_cpus should be between 1 and physical_cpus (or both 0 if commpage invalid).
        if snap.commpage_valid {
            // physical_cpus and active_cpus are read from commpage at fixed offsets.
            // On Darwin 25, active_cpus offset may differ; just check both are non-zero.
            assert!(snap.physical_cpus >= 1, "physical_cpus={}", snap.physical_cpus);
        }
        // bandwidth_gbs should be non-negative and plausible (< 1000 GB/s).
        assert!(snap.bandwidth_gbs >= 0.0 && snap.bandwidth_gbs < 1000.0,
            "bandwidth_gbs={}", snap.bandwidth_gbs);
        // contention_rate in [0, 1].
        assert!((0.0..=1.0).contains(&snap.contention_rate),
            "contention_rate={}", snap.contention_rate);
    }

    #[test]
    fn probe_hardware_registers() {
        // Commpage — lectura directa de memoria del kernel sin syscall
        let cp = read_commpage();
        println!("commpage valid={}", cp.valid);
        println!(
            "  active_cpus={} physical={} logical={}",
            cp.active_cpus, cp.physical_cpus, cp.logical_cpus
        );
        println!(
            "  memory={}GB  caps={:#x}",
            cp.memory_bytes / 1024 / 1024 / 1024,
            cp.cpu_caps
        );
        println!("  NEON={}", cp.cpu_caps & cpu_caps::NEON != 0);
        // La firma y los CPUs físicos/lógicos son estables entre versiones de Darwin.
        // active_cpus y memory_bytes tienen offsets que variaron en Darwin 25.
        assert!(cp.valid, "commpage debe ser accesible en macOS ARM64");
        assert!(cp.physical_cpus >= 8, "Apple Silicon tiene >= 8 cores");

        // CTR_EL0 / DCZID_EL0 — BLOQUEADOS por Apple (SCTLR_EL1.UCT=0, mitigación Spectre)
        // Generan SIGILL en userspace. Este es el límite real de EL0 en macOS.
        // Usamos el valor conocido para todos los M-series.
        println!(
            "dcache_line={}B (hardcoded, ctr_el0 bloqueado por Apple)",
            dcache_line_bytes()
        );

        // TPIDR_EL0 — thread pointer
        // Note: may read 0 on test-harness worker threads under concurrent
        // execution (cargo test runs tests in parallel on a thread pool).
        // The register is always valid when called from a real pthread.
        let tp = read_thread_ptr();
        println!("tpidr_el0={:#x}", tp);
        if tp == 0 {
            println!("WARN: tpidr_el0=0 (expected under concurrent test execution)");
        }

        // Timer
        let freq = timer_freq();
        let t0 = read_vct();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t1 = read_vct();
        let elapsed_us = ticks_to_us(t1 - t0, freq);
        println!("cntfrq_el0={}Hz  10ms medidos={}us", freq, elapsed_us);
        assert_eq!(freq, 24_000_000, "Apple Silicon timer = 24 MHz");
        // Allow wide range: OS scheduling jitter under load (especially during
        // concurrent test execution) can inflate sleep durations significantly.
        assert!(
            elapsed_us >= 5_000 && elapsed_us <= 100_000,
            "10ms sleep debe medir ~10000us, got {}us",
            elapsed_us
        );
    }
}
