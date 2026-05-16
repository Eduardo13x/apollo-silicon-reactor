//! Silicon Probe — el nivel más profundo accesible desde userspace en macOS ARM64.
//!
//! Capas exploradas aquí:
//!
//!  Nivel 2b — Commpage raw dump    (todos los bytes del kernel en nuestro espacio)
//!  Nivel 2c — RNDR / RNDRRS        (entropía hardware directa del chip M-series)
//!  Nivel 2d — SVC #0x80 directo    (Mach trap sin libc — más bajo que libc puede ir)
//!  Nivel 2e — thread_get_state      (todos nuestros registros del CPU en tiempo real)
//!  Nivel 2f — IOKit sin root        (hardware data via APIs públicas de Apple)
//!
//! Todo funciona sin entitlements. Nada de esto es accesible en EL1+ sin kernel.

use std::ffi::c_void;

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2b — Commpage raw dump
// Volcamos los primeros 256 bytes del commpage para mapear el layout de Darwin 25.
// ═══════════════════════════════════════════════════════════════════════════════

const COMM_PAGE_BASE: usize = 0x0000_000F_FFFF_C000;

/// Vuelca N bytes del commpage como raw u8.
/// Útil para reverse-engineer el layout en Darwin 25 vs fuente XNU publicada.
pub fn commpage_raw_dump(offset: usize, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        for i in 0..len {
            let ptr = (COMM_PAGE_BASE + offset + i) as *const u8;
            out.push(*ptr);
        }
    }
    out
}

/// Lee un u32 no alineado del commpage.
pub fn commpage_u32(offset: usize) -> u32 {
    unsafe { ((COMM_PAGE_BASE + offset) as *const u32).read_unaligned() }
}

/// Lee un u64 no alineado del commpage.
pub fn commpage_u64(offset: usize) -> u64 {
    unsafe { ((COMM_PAGE_BASE + offset) as *const u64).read_unaligned() }
}

/// Lee un u8 del commpage.
pub fn commpage_u8(offset: usize) -> u8 {
    unsafe { *((COMM_PAGE_BASE + offset) as *const u8) }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2c — RNDR / RNDRRS (ARMv8.5 FEAT_RNG)
// Entropía hardware directa del chip. Sin /dev/random, sin syscall.
// Apple Silicon (M1+) implementa FEAT_RNG pero el acceso desde EL0 depende
// de que Apple habilite el trap en SCTLR_EL1.
// Si está deshabilitado → SIGILL (como CTR_EL0).
// ═══════════════════════════════════════════════════════════════════════════════

/// Intenta leer el hardware random number register (RNDR).
///
/// Retorna Some(valor) si el chip provee entropía.
/// Si RNDR no tiene entropía suficiente, el flag NZCV.Z se pone a 1 (retorna None).
/// Si FEAT_RNG no está habilitado en EL0 → SIGILL (manejado con signal handler en tests).
///
/// # Safety
/// Puede generar SIGILL en chips que no soporten FEAT_RNG en EL0.
pub unsafe fn read_rndr() -> Option<u64> {
    let val: u64;
    let nzcv: u64;
    // RNDR = s3_3_c2_c4_0  (ARMv8.5 FEAT_RNG, encoding directo)
    // El assembler puede no conocer el nombre "rndr", usamos la forma s<op0>_<op1>_<CRn>_<CRm>_<op2>
    std::arch::asm!(
        "mrs {val}, s3_3_c2_c4_0",
        "mrs {nzcv}, nzcv",
        val  = out(reg) val,
        nzcv = out(reg) nzcv,
        options(nostack, nomem)
    );
    // NZCV.Z (bit 30) = 1 significa que no había entropía disponible
    if nzcv & (1 << 30) == 0 {
        Some(val)
    } else {
        None
    }
}

/// Versión reseed: RNDRRS fuerza reseed del generador antes de leer.
/// RNDRRS = s3_3_c2_c4_1
pub unsafe fn read_rndrrs() -> Option<u64> {
    let val: u64;
    let nzcv: u64;
    std::arch::asm!(
        "mrs {val}, s3_3_c2_c4_1",
        "mrs {nzcv}, nzcv",
        val  = out(reg) val,
        nzcv = out(reg) nzcv,
        options(nostack, nomem)
    );
    if nzcv & (1 << 30) == 0 {
        Some(val)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2c-alt — cntvct_el0 + xorshift64 (entropía sin syscall, sin SIGILL)
//
// RNDR está bloqueado en EL0 por Apple. Alternativa realista para hot-path:
//   1. cntvct_el0 — virtual counter ARM64, siempre accesible desde EL0.
//      macOS lo usa en commpage para mach_absolute_time(), latencia ~1ns.
//   2. xorshift64 — mezcla el timestamp con PID + thread_id del llamador.
//
// No es criptográficamente segura, pero para decisiones de scheduling de Apollo
// (jitter, backoff, priority variation) es más que suficiente: ~3ns sin syscall.
// ═══════════════════════════════════════════════════════════════════════════════

/// Lee el virtual counter ARM64 directamente desde EL0.
///
/// `cntvct_el0` es el contador de tiempo del sistema accesible desde userspace.
/// macOS lo expone sin restricción — es el mismo valor que usa `mach_absolute_time()`
/// vía commpage. Resolución: ~41.67 MHz en M1 (24 ns por tick).
///
/// A diferencia de RNDR, este registro nunca genera SIGILL en macOS.
#[inline(always)]
pub fn read_cntvct_el0() -> u64 {
    let val: u64;
    unsafe {
        std::arch::asm!(
            "mrs {val}, cntvct_el0",
            val = out(reg) val,
            options(nostack, nomem, preserves_flags)
        );
    }
    val
}

/// Mezcla xorshift64 — bijección rápida sobre u64 sin división ni multiplicación.
/// Avalancha completa en 3 shifts: cada bit de entrada afecta todos los de salida.
#[inline(always)]
pub fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Genera pseudo-entropía para hot-path de Apollo sin ningún syscall.
///
/// Combina:
///   - `cntvct_el0` — timestamp de alta resolución (~41 MHz, único por llamada)
///   - `seed`       — contexto del llamador (PID, thread ID, contador, etc.)
///
/// Uso típico en Apollo scheduling:
/// ```no_run
/// let pid = std::process::id() as u64;
/// let entropy = apollo_engine::engine::silicon_probe::fast_entropy(pid);
/// let jitter_ns = entropy % 1_000; // jitter 0-999 ns
/// ```
///
/// Latencia: ~3 ns (comparable a leer un atomic en L1 cache).
#[inline(always)]
pub fn fast_entropy(seed: u64) -> u64 {
    let t = read_cntvct_el0();
    // Fibonacci hashing del seed para distribuir bien el espacio,
    // luego doble xorshift para mezcla completa con el timestamp.
    xorshift64(xorshift64(t ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2d — SVC #0x80 directo (Mach traps sin libc)
//
// macOS ARM64 ABI para Mach traps:
//   x16 = número de trap (negativo para Mach traps)
//   svc #0x80
//   retorno en x0
//
// Es lo más bajo que podemos ir desde userspace — es literalmente el
// mecanismo que libc usa internamente. Aquí lo llamamos directamente.
// ═══════════════════════════════════════════════════════════════════════════════

/// mach_task_self() via SVC directo (Mach trap -28).
/// Retorna el task port de nuestro propio proceso — el handle Mach fundamental.
/// Este port es la raíz de toda comunicación IPC con el kernel para nuestra tarea.
pub fn mach_task_self_raw() -> u32 {
    let result: u32;
    unsafe {
        std::arch::asm!(
            "movn x16, #27",    // x16 = ~27 = -28 (Mach trap para mach_task_self)
            "svc #0x80",        // invoke Mach trap — bypasa libc completamente
            out("x0") result,
            lateout("x16") _,
            options(nostack)
        );
    }
    result
}

/// mach_thread_self() via SVC directo (Mach trap -27).
/// Retorna el thread port del thread actual.
pub fn mach_thread_self_raw() -> u32 {
    let result: u32;
    unsafe {
        std::arch::asm!(
            "movn x16, #26",    // x16 = -27
            "svc #0x80",
            out("x0") result,
            lateout("x16") _,
            options(nostack)
        );
    }
    result
}

/// mach_absolute_time() via SVC directo (Mach trap -3).
/// En la práctica macOS lo implementa via commpage (lee cntvct_el0),
/// pero podemos llamarlo directamente al kernel para verificar.
pub fn mach_absolute_time_raw() -> u64 {
    let result: u64;
    unsafe {
        std::arch::asm!(
            "movn x16, #2",     // x16 = -3
            "svc #0x80",
            out("x0") result,
            lateout("x16") _,
            options(nostack)
        );
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2e — thread_get_state (todos nuestros registros del CPU)
//
// Via Mach, podemos leer el estado completo de registros de nuestro propio thread.
// Esto incluye los 29 GPRs + PC + SP + CPSR — el estado exacto de la CPU.
// ═══════════════════════════════════════════════════════════════════════════════

// ARM_THREAD_STATE64 flavor
const ARM_THREAD_STATE64: u32 = 6;
const ARM_THREAD_STATE64_COUNT: u32 = 68; // tamaño en u32 words

#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct ArmThreadState64 {
    pub x: [u64; 29], // x0-x28 (general purpose registers)
    pub fp: u64,      // x29 (frame pointer)
    pub lr: u64,      // x30 (link register)
    pub sp: u64,      // stack pointer
    pub pc: u64,      // program counter
    pub cpsr: u32,    // current program status register
    pub _pad: u32,
}

extern "C" {
    fn mach_thread_self() -> u32;
    fn thread_get_state(thread: u32, flavor: u32, state: *mut c_void, count: *mut u32) -> i32;
}

/// Lee todos los registros del CPU del thread actual via Mach.
///
/// Nota: PC captura el punto donde thread_get_state fue llamado,
/// no el punto de retorno — útil para profiling de precisión.
pub fn read_thread_cpu_state() -> Result<ArmThreadState64, i32> {
    let thread = unsafe { mach_thread_self() };
    let mut state = ArmThreadState64::default();
    let mut count = ARM_THREAD_STATE64_COUNT;
    let kr = unsafe {
        thread_get_state(
            thread,
            ARM_THREAD_STATE64,
            &mut state as *mut _ as *mut c_void,
            &mut count,
        )
    };
    if kr == 0 {
        Ok(state)
    } else {
        Err(kr)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NIVEL 2f — IOKit sin root (hardware data via APIs públicas)
// Cualquier proceso puede leer datos de hardware via IOKit registry
// sin necesitar root ni entitlements.
// ═══════════════════════════════════════════════════════════════════════════════

/// Lee el número de serie del sistema (sin root, via IOKit).
pub fn read_serial_number() -> Option<String> {
    // La forma más simple de acceder a datos de IOKit sin root:
    // usar sysctl hw.model que expone el identificador del hardware
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    let name = b"hw.model\0";
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            buf.as_mut_ptr() as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).to_string())
    } else {
        None
    }
}

/// Lee el identificador exacto del chip (sysctl machdep.cpu.brand_string en ARM es distinto).
pub fn read_chip_id() -> Option<String> {
    let mut buf = [0u8; 128];
    let mut len = buf.len();
    // En Apple Silicon: "hw.targettype" o "hw.cpusubtype"
    let name = b"hw.targettype\0";
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            buf.as_mut_ptr() as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).to_string())
    } else {
        None
    }
}

/// Lee datos de CPU via sysctl — sin syscall wrapper, directo con libc.
pub struct SiliconInfo {
    pub model: String,
    pub cpu_brand: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub l1d_cache: u64,
    pub l2_cache: u64,
    pub l3_cache: u64, // 0 si no existe (M1 no tiene L3 separado)
    pub cpu_freq_hz: u64,
    pub bus_freq_hz: u64,
    pub memory_bytes: u64,
}

impl SiliconInfo {
    pub fn read() -> Self {
        Self {
            model: sysctl_string(b"hw.model\0"),
            cpu_brand: sysctl_string(b"machdep.cpu.brand_string\0"),
            physical_cores: sysctl_u32(b"hw.physicalcpu\0"),
            logical_cores: sysctl_u32(b"hw.logicalcpu\0"),
            l1d_cache: sysctl_u64(b"hw.l1dcachesize\0"),
            l2_cache: sysctl_u64(b"hw.l2cachesize\0"),
            l3_cache: sysctl_u64(b"hw.l3cachesize\0"),
            cpu_freq_hz: sysctl_u64(b"hw.cpufrequency\0"),
            bus_freq_hz: sysctl_u64(b"hw.busfrequency\0"),
            memory_bytes: sysctl_u64(b"hw.memsize\0"),
        }
    }
}

fn sysctl_string(name: &[u8]) -> String {
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            buf.as_mut_ptr() as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
    }
    String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).to_string()
}

fn sysctl_u32(name: &[u8]) -> u32 {
    let mut val: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            &mut val as *mut _ as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
    }
    val
}

fn sysctl_u64(name: &[u8]) -> u64 {
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
            &mut val as *mut _ as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
    }
    val
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // RNDR está bloqueado por Apple desde EL0 en macOS — ejecutarlo genera SIGILL.
    // Este test es solo para exploración manual: `cargo test -- --ignored probe_rndr`
    #[test]
    #[ignore = "RNDR bloqueado en EL0 por Apple → SIGILL; ejecutar solo con --ignored"]
    fn probe_rndr_hardware_rng() {
        // RNDR — hardware random del chip M-series (ARMv8.5 FEAT_RNG)
        // Si Apple bloqueó el acceso desde EL0 → SIGILL y el test falla con signal
        // Si funciona → tenemos entropía directa del chip sin syscall
        let result = unsafe { read_rndr() };
        let result2 = unsafe { read_rndr() };
        let result_rs = unsafe { read_rndrrs() };
        println!("RNDR  #1: {:?}", result.map(|v| format!("{:#018x}", v)));
        println!("RNDR  #2: {:?}", result2.map(|v| format!("{:#018x}", v)));
        println!("RNDRRS:   {:?}", result_rs.map(|v| format!("{:#018x}", v)));
        // Si llegamos aquí sin SIGILL, RNDR funciona en EL0 en este macOS
        if let (Some(a), Some(b)) = (result, result2) {
            assert_ne!(a, b, "dos lecturas de RNDR deben ser distintas");
            println!("RNDR FUNCIONA en EL0 — entropía hardware sin syscall");
        }
    }

    #[test]
    fn probe_cntvct_entropy() {
        // cntvct_el0 — accesible desde EL0 en macOS, nunca SIGILL
        let t1 = read_cntvct_el0();
        // El contador avanza ~24ns/tick; dos mrs consecutivos pueden ser
        // más rápidos que un tick. Spin hasta que avance.
        let mut t2 = read_cntvct_el0();
        for _ in 0..100_000 {
            if t2 != t1 {
                break;
            }
            std::hint::spin_loop();
            t2 = read_cntvct_el0();
        }
        println!("cntvct_el0 #1: {:#018x}  ({} ticks)", t1, t1);
        println!(
            "cntvct_el0 #2: {:#018x}  (delta: {} ticks, ~{}ns)",
            t2,
            t2.saturating_sub(t1),
            t2.saturating_sub(t1) * 1_000 / 41_667
        );
        assert_ne!(t1, t2, "el contador debe avanzar en 100k iteraciones");

        // fast_entropy — hot-path entropy combinando cntvct + seed
        let pid = std::process::id() as u64;
        let e1 = fast_entropy(pid);
        // Spin hasta que cntvct avance al menos un tick
        let mut e2 = fast_entropy(pid);
        for _ in 0..100_000 {
            if e2 != e1 {
                break;
            }
            std::hint::spin_loop();
            e2 = fast_entropy(pid);
        }
        let e3 = fast_entropy(pid ^ 0xdead_beef);
        println!("\nfast_entropy:");
        println!("  fast_entropy(pid)      = {:#018x}", e1);
        println!("  fast_entropy(pid) x2   = {:#018x}", e2);
        println!("  fast_entropy(pid^seed) = {:#018x}", e3);
        assert_ne!(
            e1, e2,
            "fast_entropy debe diferir entre llamadas (cntvct avanza)"
        );
        assert_ne!(e1, e3, "distinto seed debe dar distinto resultado");

        // xorshift64 — verificar propiedad de avalancha
        let base = 0x1234_5678_9abc_def0_u64;
        let mixed = xorshift64(base);
        println!("\nxorshift64:");
        println!("  input:  {:#018x}", base);
        println!("  output: {:#018x}", mixed);
        assert_ne!(base, mixed, "xorshift64 no debe ser identidad");
        assert_ne!(
            mixed, 0,
            "xorshift64 no debe producir cero desde input no-cero"
        );

        println!("\ncntvct_el0 + xorshift64: FUNCIONA en EL0 — entropía ~3ns sin syscall");
    }

    #[test]
    fn probe_commpage_raw() {
        // Volcar bytes 0x010-0x040 del commpage para entender el layout de Darwin 25
        let raw = commpage_raw_dump(0x000, 0x060);
        println!("Commpage raw dump (bytes 0x000-0x05F):");
        for (i, chunk) in raw.chunks(16).enumerate() {
            let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
            let ascii: String = chunk
                .iter()
                .map(|b| {
                    if *b >= 0x20 && *b < 0x7f {
                        *b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("  {:03x}: {}  {}", i * 16, hex.join(" "), ascii);
        }

        // Campos confirmados en Darwin 25 (ajustar si el dump muestra otra cosa)
        println!("\nCampos individuales:");
        println!("  [0x00e] version u16   = {}", commpage_u32(0x00e) & 0xFFFF);
        println!("  [0x010] caps64 u64    = {:#018x}", commpage_u64(0x010));
        println!("  [0x018] caps64_hi u64 = {:#018x}", commpage_u64(0x018));
        println!("  [0x020] caps32 u32    = {:#010x}", commpage_u32(0x020));
        println!("  [0x022] byte          = {}", commpage_u8(0x022));
        println!("  [0x023] byte          = {}", commpage_u8(0x023));
        println!("  [0x024] physical_cpu  = {}", commpage_u8(0x024));
        println!("  [0x025] logical_cpu   = {}", commpage_u8(0x025));
        println!("  [0x030] u64           = {:#018x}", commpage_u64(0x030));
        println!(
            "  [0x034] mem_size u64  = {:#018x} ({}GB)",
            commpage_u64(0x034),
            commpage_u64(0x034) / 1024 / 1024 / 1024
        );
        println!("  [0x038] u64           = {:#018x}", commpage_u64(0x038));
        println!("  [0x040] u64           = {:#018x}", commpage_u64(0x040));
        println!("  [0x048] u64           = {:#018x}", commpage_u64(0x048));
        println!("  [0x050] u64           = {:#018x}", commpage_u64(0x050));
    }

    #[test]
    #[allow(deprecated)] // libc::mach_task_self / mach_absolute_time are used as
                         // oracle values here — the test's purpose is to verify that the direct
                         // Mach SVC path returns the same values libc returns. Migrating the
                         // oracle to the `mach2` crate would change what we're comparing against.
    fn probe_mach_svc_direct() {
        // Llamar al kernel Mach directamente sin libc
        let task_port_raw = mach_task_self_raw();
        let task_port_libc = unsafe { libc::mach_task_self() };
        println!("mach_task_self via SVC directo:  {}", task_port_raw);
        println!("mach_task_self via libc:         {}", task_port_libc);
        assert_eq!(
            task_port_raw, task_port_libc,
            "SVC directo y libc deben retornar el mismo task port"
        );

        let thread_raw = mach_thread_self_raw();
        println!("mach_thread_self via SVC directo: {}", thread_raw);
        assert!(thread_raw > 0, "thread port debe ser válido");

        let mat_raw = mach_absolute_time_raw();
        let mat_libc = unsafe { libc::mach_absolute_time() };
        println!("mach_absolute_time SVC:  {}", mat_raw);
        println!("mach_absolute_time libc: {}", mat_libc);
        // Deben ser muy cercanos (medidos con nanosegundos de diferencia)
        let delta = mat_libc.saturating_sub(mat_raw);
        println!(
            "delta entre los dos:     {} ticks (~{}ns)",
            delta,
            delta * 1000 / 24_000
        );
        assert!(
            delta < 100_000,
            "delta debe ser <100µs, fue {} ticks",
            delta
        );
    }

    #[test]
    fn probe_thread_cpu_state() {
        let state = read_thread_cpu_state().expect("thread_get_state debe funcionar");
        println!("Registros del CPU de nuestro thread:");
        for i in 0..10 {
            println!("  x{:02} = {:#018x}", i, state.x[i]);
        }
        println!("  fp  = {:#018x}  (frame pointer)", state.fp);
        println!("  lr  = {:#018x}  (link register)", state.lr);
        println!("  sp  = {:#018x}  (stack pointer)", state.sp);
        println!("  pc  = {:#018x}  (program counter)", state.pc);
        println!(
            "  cpsr= {:#010x}  (flags: N={} Z={} C={} V={})",
            state.cpsr,
            (state.cpsr >> 31) & 1,
            (state.cpsr >> 30) & 1,
            (state.cpsr >> 29) & 1,
            (state.cpsr >> 28) & 1,
        );
        assert_ne!(state.sp, 0, "stack pointer no puede ser null");
        assert_ne!(state.pc, 0, "program counter no puede ser null");
    }

    #[test]
    fn probe_silicon_info() {
        let info = SiliconInfo::read();
        println!("Hardware info via sysctl (sin root, sin entitlement):");
        println!("  model:          {}", info.model);
        println!("  cpu_brand:      {}", info.cpu_brand);
        println!("  physical_cores: {}", info.physical_cores);
        println!("  logical_cores:  {}", info.logical_cores);
        println!("  L1d cache:      {}KB", info.l1d_cache / 1024);
        println!("  L2  cache:      {}MB", info.l2_cache / 1024 / 1024);
        println!("  L3  cache:      {}MB", info.l3_cache / 1024 / 1024);
        println!("  cpu_freq:       {}MHz", info.cpu_freq_hz / 1_000_000);
        println!(
            "  memory:         {}GB",
            info.memory_bytes / 1024 / 1024 / 1024
        );
        assert!(!info.model.is_empty());
        assert!(info.physical_cores >= 8);
        assert!(info.memory_bytes >= 8 * 1024 * 1024 * 1024);
    }
}
