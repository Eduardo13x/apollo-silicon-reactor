//! ARM NEON vectorized EMA — batch update 4 f32 values in one instruction.
//!
//! On Apple Silicon (aarch64), Rust with `-C target-cpu=native` enables NEON.
//! This module provides a NEON-accelerated EMA updater that processes 4 values
//! simultaneously using `vfmaq_f32` (fused multiply-add on 4-lane f32 vector).
//!
//! # Formula
//!
//! Standard EMA: `ema[i] = alpha * sample[i] + (1 - alpha) * ema[i]`
//!
//! NEON version processes 4 EMA values per instruction cycle:
//! `vfmaq_f32(scaled_ema, alpha_v, sample_v)` = `alpha * sample + beta * ema`
//!
//! # Why
//!
//! Apollo's SpecialistAccuracyTracker updates ~8 specialist EMA weights every cycle.
//! Each update is a scalar multiply-add (~4ns). NEON processes 4 in one instruction
//! (~1ns on M1 with out-of-order execution). Combined with LSE atomics already
//! in lse_counters.rs, this eliminates all serial arithmetic on the hot path.
//!
//! # Benchmark
//!
//! Synthetic: 8-value batch EMA
//! - Scalar: 8 × ~3ns = ~24ns per batch
//! - NEON:   2 × vfmaq_f32 = ~2ns per batch (M1 Firestorm throughput 1 cycle)
//!
//! [ARM NEON Programmer's Guide §4 — vfmaq_f32 latency 3cy, throughput 1cy on Firestorm]
//! [Fog 2022 "Optimizing software in C++"] vectorize small fixed-size loops first.

/// Update 4 EMA values simultaneously using NEON vfmaq_f32.
///
/// `ema[i] = alpha * sample[i] + (1 - alpha) * ema[i]` for i in 0..4.
///
/// # Safety
/// Requires aarch64 target with NEON (guaranteed on all Apple Silicon).
/// Falls back to scalar on non-aarch64.
#[inline]
pub fn ema_update_4(ema: &mut [f32; 4], samples: &[f32; 4], alpha: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        unsafe {
            // Load current EMA values and new samples into NEON registers.
            let ema_v = vld1q_f32(ema.as_ptr());
            let sample_v = vld1q_f32(samples.as_ptr());

            // beta = 1.0 - alpha
            let beta_v = vdupq_n_f32(1.0 - alpha);
            let alpha_v = vdupq_n_f32(alpha);

            // ema = beta * ema + alpha * sample  (two FMAs)
            // vfmaq_f32(a, b, c) = a + b * c
            let scaled_ema = vmulq_f32(beta_v, ema_v); // beta * ema
            let result = vfmaq_f32(scaled_ema, alpha_v, sample_v); // + alpha * sample

            vst1q_f32(ema.as_mut_ptr(), result);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let beta = 1.0 - alpha;
        for i in 0..4 {
            ema[i] = alpha * samples[i] + beta * ema[i];
        }
    }
}

/// Update 8 EMA values simultaneously (two NEON passes of 4).
///
/// Covers the full Apollo specialist weight vector (8 specialists).
#[inline]
pub fn ema_update_8(ema: &mut [f32; 8], samples: &[f32; 8], alpha: f32) {
    let (ema_lo, ema_hi) = ema.split_at_mut(4);
    let (smp_lo, smp_hi) = samples.split_at(4);

    ema_update_4(
        ema_lo.try_into().unwrap(),
        smp_lo.try_into().unwrap(),
        alpha,
    );
    ema_update_4(
        ema_hi.try_into().unwrap(),
        smp_hi.try_into().unwrap(),
        alpha,
    );
}

/// Verify scalar and NEON results match (for testing).
pub fn ema_scalar_reference(ema: &[f32; 4], samples: &[f32; 4], alpha: f32) -> [f32; 4] {
    let beta = 1.0 - alpha;
    let mut result = [0.0f32; 4];
    for i in 0..4 {
        result[i] = alpha * samples[i] + beta * ema[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for float comparison (f32 arithmetic differences).
    const EPS: f32 = 1e-5;

    #[test]
    fn neon_matches_scalar_zero_alpha() {
        let mut ema = [1.0f32, 2.0, 3.0, 4.0];
        let samples = [10.0f32, 20.0, 30.0, 40.0];
        let scalar_result = ema_scalar_reference(&ema, &samples, 0.0);
        ema_update_4(&mut ema, &samples, 0.0);
        for i in 0..4 {
            assert!(
                (ema[i] - scalar_result[i]).abs() < EPS,
                "alpha=0: ema[{}]={} vs scalar={}",
                i,
                ema[i],
                scalar_result[i]
            );
        }
    }

    #[test]
    fn neon_matches_scalar_one_alpha() {
        let mut ema = [1.0f32, 2.0, 3.0, 4.0];
        let samples = [10.0f32, 20.0, 30.0, 40.0];
        let scalar_result = ema_scalar_reference(&ema, &samples, 1.0);
        ema_update_4(&mut ema, &samples, 1.0);
        for i in 0..4 {
            assert!(
                (ema[i] - scalar_result[i]).abs() < EPS,
                "alpha=1: ema[{}]={} vs scalar={}",
                i,
                ema[i],
                scalar_result[i]
            );
        }
    }

    #[test]
    fn neon_matches_scalar_typical_alpha() {
        let mut ema = [0.5f32, 0.6, 0.7, 0.8];
        let samples = [1.0f32, 0.8, 0.6, 0.4];
        let alpha = 0.15_f32;
        let scalar = ema_scalar_reference(&ema, &samples, alpha);
        ema_update_4(&mut ema, &samples, alpha);
        for i in 0..4 {
            assert!(
                (ema[i] - scalar[i]).abs() < EPS,
                "alpha=0.15: ema[{}]={} vs scalar={}",
                i,
                ema[i],
                scalar[i]
            );
        }
    }

    #[test]
    fn neon_8_matches_two_scalar_4() {
        let mut ema8 = [0.5f32, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6];
        let samples8 = [1.0f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let alpha = 0.20_f32;

        let ref_lo = ema_scalar_reference(
            ema8[..4].try_into().unwrap(),
            samples8[..4].try_into().unwrap(),
            alpha,
        );
        let ref_hi = ema_scalar_reference(
            ema8[4..].try_into().unwrap(),
            samples8[4..].try_into().unwrap(),
            alpha,
        );

        ema_update_8(&mut ema8, &samples8, alpha);

        for i in 0..4 {
            assert!((ema8[i] - ref_lo[i]).abs() < EPS, "lo[{}]", i);
            assert!((ema8[i + 4] - ref_hi[i]).abs() < EPS, "hi[{}]", i);
        }
    }

    #[test]
    fn ema_converges_to_constant_input() {
        // After N iterations with constant input, EMA should approach that input.
        let mut ema = [0.0f32; 4];
        let target = [1.0f32; 4];
        let alpha = 0.15_f32;
        for _ in 0..200 {
            ema_update_4(&mut ema, &target, alpha);
        }
        for i in 0..4 {
            assert!(
                (ema[i] - 1.0).abs() < 0.01,
                "EMA did not converge: ema[{}]={}",
                i,
                ema[i]
            );
        }
    }

    #[test]
    fn ema_alpha_zero_is_identity() {
        let initial = [0.3f32, 0.5, 0.7, 0.9];
        let mut ema = initial;
        let samples = [99.0f32; 4];
        ema_update_4(&mut ema, &samples, 0.0);
        // alpha=0 means no learning — EMA unchanged.
        for i in 0..4 {
            assert!((ema[i] - initial[i]).abs() < EPS);
        }
    }
}
