//! Evolved Anomaly Detector
//!
//! Lightweight online anomaly detection combining three paradigms:
//!
//! 1. **Modern Hopfield Memory** (Ramsauer et al. 2020) — associative energy-based
//!    pattern matching.  Stores prototypes of "normal" system states; anomaly =
//!    high energy (poor match to any stored pattern).
//!
//! 2. **Sparse Autoencoder Population** (Anthropic SAE / Bricken et al. 2023) —
//!    8 online autoencoders with TopK activation.  Each has a Darwinian feature
//!    mask that evolves via quality-diversity selection (MAP-Elites, Mouret 2015).
//!
//! 3. **Free Energy Fusion** (Friston 2006/2010) — scores combined through
//!    variational free energy: F = complexity + inaccuracy.  Anomaly = surprise
//!    under the generative model.
//!
//! The population evolves via tournament selection with speciation:
//! detectors specialise for different pressure regimes (idle/moderate/heavy)
//! and feature focuses (cpu-heavy/memory-heavy/balanced), preventing premature
//! convergence to one mediocre generalist (Stanley & Miikkulainen 2002, NEAT).
//!
//! Total memory: ~50KB.  Per-cycle cost: ~3µs typical, ~8µs on evolution steps.
//! Zero external dependencies.
//!
//! ## References
//!
//! - Ramsauer et al. 2020, "Hopfield Networks is All You Need" (exponential capacity)
//! - Bricken et al. 2023, "Sparse Autoencoders Find Interpretable Features" (TopK SAE)
//! - Templeton et al. 2024, "Scaling Monosemanticity" (TopK > L1)
//! - Friston 2006, "A Free Energy Principle for the Brain"
//! - Mouret & Clune 2015, "Quality-Diversity through MAP-Elites"
//! - Gardner 1988, "Phase Space of Interactions" (storage capacity bound)
//! - Stanley & Miikkulainen 2002, "NEAT" (speciation in neuroevolution)
//! - Cejnek & Bukovsky 2024, "Scalable Online Anomaly Detection" (per-sample SGD)

use crate::engine::telemetry_logger::N_FEATURES;

// ── NEON-accelerated f32 operations ─────────────────────────────────────────
// N_FEATURES=16 = exactly 4 × float32x4_t lanes. Zero remainder.

/// NEON dot product for f32 slices of length 16 (encoder/decoder inner loops).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_dot16_f32(a: &[f32], b: &[f32; N_FEATURES]) -> f32 {
    debug_assert!(a.len() >= N_FEATURES);
    use std::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        // 16 elements / 4 lanes = 4 iterations, fully unrolled.
        let a0 = vld1q_f32(a.as_ptr());
        let b0 = vld1q_f32(b.as_ptr());
        acc0 = vfmaq_f32(acc0, a0, b0);
        let a1 = vld1q_f32(a.as_ptr().add(4));
        let b1 = vld1q_f32(b.as_ptr().add(4));
        acc1 = vfmaq_f32(acc1, a1, b1);
        let a2 = vld1q_f32(a.as_ptr().add(8));
        let b2 = vld1q_f32(b.as_ptr().add(8));
        acc2 = vfmaq_f32(acc2, a2, b2);
        let a3 = vld1q_f32(a.as_ptr().add(12));
        let b3 = vld1q_f32(b.as_ptr().add(12));
        acc3 = vfmaq_f32(acc3, a3, b3);
        // Horizontal reduction: acc0+acc1+acc2+acc3
        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let sum = vaddq_f32(sum01, sum23);
        vaddvq_f32(sum)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn neon_dot16_f32(a: &[f32], b: &[f32; N_FEATURES]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N_FEATURES {
        s += a[i] * b[i];
    }
    s
}

/// NEON dot product for f32 slices of length 24 (decoder inner loop: HIDDEN=24).
/// 24 / 4 = 6 NEON iterations, zero remainder.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_dot24_f32(a: &[f32], b: &[f32; HIDDEN]) -> f32 {
    debug_assert!(a.len() >= HIDDEN);
    use std::arch::aarch64::*;
    unsafe {
        let mut acc0 = vmulq_f32(vld1q_f32(a.as_ptr()), vld1q_f32(b.as_ptr()));
        let mut acc1 = vmulq_f32(vld1q_f32(a.as_ptr().add(4)), vld1q_f32(b.as_ptr().add(4)));
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(8)),
            vld1q_f32(b.as_ptr().add(8)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(12)),
            vld1q_f32(b.as_ptr().add(12)),
        );
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(16)),
            vld1q_f32(b.as_ptr().add(16)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(20)),
            vld1q_f32(b.as_ptr().add(20)),
        );
        vaddvq_f32(vaddq_f32(acc0, acc1))
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn neon_dot24_f32(a: &[f32], b: &[f32; HIDDEN]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..HIDDEN {
        s += a[i] * b[i];
    }
    s
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Hopfield memory: number of stored "normal" prototypes.
/// Gardner 1988: capacity ≈ 0.14·N for N-dimensional patterns with low error.
/// 64 prototypes in 16D is well within the exponential capacity of modern
/// continuous Hopfield networks (Ramsauer et al. 2020).
const HOPFIELD_SLOTS: usize = 64;

/// Inverse temperature for Hopfield energy.  Higher β → sharper discrimination.
/// β=4.0 gives good separation without numerical overflow for 16D normalised vectors.
const HOPFIELD_BETA: f32 = 4.0;

/// SAE population size.  8 individuals over 6 MAP-Elites niches ensures at
/// least one specialist per pressure regime with room for competition.
const POP_SIZE: usize = 8;

/// Hidden layer size for each sparse autoencoder.
/// 24 hidden units for 16 features gives compression ratio 1.5:1 — enough
/// to force information bottleneck while preserving reconstruction fidelity.
const HIDDEN: usize = 24;

/// TopK activation: keep only top 4 of 24 hidden units active.
/// Sparsity ratio 4/24 ≈ 17% matches Anthropic's empirical sweet spot for
/// interpretable, disentangled feature learning (Templeton et al. 2024).
const TOP_K: usize = 4;

/// Evolution step every 120 samples (~1 minute at 2Hz).
/// This is also the Gardner capacity crossover: 120 samples for ~400 effective
/// parameters per SAE means α ≈ 0.3, safely below α_c ≈ 2.0.
const EVOLVE_INTERVAL: usize = 120;

/// Anomaly threshold: mean + THRESHOLD_SIGMA × std of recent free energy.
const THRESHOLD_SIGMA: f32 = 3.0;

/// RMSProp decay factor (ρ = 0.999 for slow-moving online training).
const RMSPROP_DECAY: f32 = 0.999;

/// Base learning rate for online SGD (modulated by RMSProp per-parameter).
const BASE_LR: f32 = 0.001;

// ── PRNG (xorshift64) ────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform f32 in [0, 1).
    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Gaussian-ish via Box-Muller (cheap, good enough for mutation noise).
    fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    /// Random index in [0, n).
    fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── Welford online stats ─────────────────────────────────────────────────────

/// Vanilla Welford online stats — exact mean/variance over all samples.
/// Used for feature normalisation where the distribution is stationary.
struct WelfordF32 {
    mean: f32,
    m2: f32,
    count: u64,
}

impl WelfordF32 {
    fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
        }
    }
    fn update(&mut self, x: f32) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    fn variance(&self) -> f32 {
        if self.count < 2 {
            1.0
        } else {
            self.m2 / (self.count - 1) as f32
        }
    }
    fn std(&self) -> f32 {
        self.variance().sqrt().max(1e-8)
    }
}

/// EMA-Welford stats — exponentially-weighted mean/variance with finite memory.
/// Unlike vanilla Welford (which accumulates forever), this uses an effective
/// window of ~1/α samples.  With α=0.002 the window is ~500 samples (~4 min
/// at 2Hz), so the threshold adapts to gradual workload shifts without
/// forgetting too fast.  Avoids the classic "Welford drift" where after days
/// of uptime the std becomes meaninglessly tight or loose.
/// Ref: Finch 2009, "Incremental calculation of weighted mean and variance".
struct EmaWelford {
    mean: f32,
    var: f32,
    count: u64,
    alpha: f32,
}

impl EmaWelford {
    fn new(alpha: f32) -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            alpha,
        }
    }
    fn update(&mut self, x: f32) {
        self.count += 1;
        if self.count == 1 {
            self.mean = x;
            self.var = 1.0;
            return;
        }
        let delta = x - self.mean;
        self.mean += self.alpha * delta;
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta);
    }
    fn std(&self) -> f32 {
        if self.count < 2 {
            1.0
        } else {
            self.var.sqrt().max(1e-8)
        }
    }
}

// ── Normaliser (Welford per-feature) ─────────────────────────────────────────

struct OnlineNormaliser {
    stats: [WelfordF32; N_FEATURES],
}

impl OnlineNormaliser {
    fn new() -> Self {
        Self {
            stats: std::array::from_fn(|_| WelfordF32::new()),
        }
    }
    fn update_and_normalise(&mut self, raw: &[f32; N_FEATURES]) -> [f32; N_FEATURES] {
        let mut out = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            self.stats[i].update(raw[i]);
            out[i] = if self.stats[i].count < 5 {
                0.0 // not enough data yet
            } else {
                (raw[i] - self.stats[i].mean) / self.stats[i].std()
            };
        }
        out
    }
}

// ── Hopfield Associative Memory (Ramsauer et al. 2020) ───────────────────────

struct HopfieldMemory {
    /// Ring buffer of stored normal-state prototypes.
    prototypes: [[f32; N_FEATURES]; HOPFIELD_SLOTS],
    write_head: usize,
    filled: usize,
}

impl HopfieldMemory {
    fn new() -> Self {
        Self {
            prototypes: [[0.0; N_FEATURES]; HOPFIELD_SLOTS],
            write_head: 0,
            filled: 0,
        }
    }

    /// Store a normalised observation as a "normal" prototype.
    fn store(&mut self, x: &[f32; N_FEATURES]) {
        self.prototypes[self.write_head] = *x;
        self.write_head = (self.write_head + 1) % HOPFIELD_SLOTS;
        if self.filled < HOPFIELD_SLOTS {
            self.filled += 1;
        }
    }

    /// Hopfield energy of query x.  Lower = better match to stored patterns.
    /// E(x) = -log(Σᵢ exp(β · x·ξᵢ))   (Ramsauer et al. 2020, Eq. 4)
    /// Returns normalised score in [0, 1]: 0 = perfect match, 1 = total mismatch.
    fn energy(&self, x: &[f32; N_FEATURES]) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        // log-sum-exp trick for numerical stability.
        let mut max_dot = f32::NEG_INFINITY;
        for i in 0..self.filled {
            let d = dot(x, &self.prototypes[i]);
            let scaled = HOPFIELD_BETA * d;
            if scaled > max_dot {
                max_dot = scaled;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..self.filled {
            let d = dot(x, &self.prototypes[i]);
            sum_exp += (HOPFIELD_BETA * d - max_dot).exp();
        }
        let log_sum = max_dot + sum_exp.ln();
        // Normalise: high similarity → log_sum is large → low energy.
        // Map to [0,1] via sigmoid-like transform.
        let raw_energy = -log_sum;
        sigmoid(raw_energy / (N_FEATURES as f32))
    }
}

// ── Sparse Autoencoder Individual ────────────────────────────────────────────

/// One member of the evolving SAE population.
/// Architecture: N_FEATURES → HIDDEN (ReLU, TopK) → N_FEATURES
/// Parameters: encoder W_e[HIDDEN×N_FEATURES] + b_e[HIDDEN]
///           + decoder W_d[N_FEATURES×HIDDEN] + b_d[N_FEATURES]
struct SaeGenome {
    /// Feature mask: bit i = 1 means feature i is used.
    feature_mask: u16,
    /// Learning rate (evolved).
    lr: f32,

    // Weights (row-major).
    w_enc: [[f32; N_FEATURES]; HIDDEN], // encoder: HIDDEN rows × N_FEATURES cols
    b_enc: [f32; HIDDEN],
    w_dec: [[f32; HIDDEN]; N_FEATURES], // decoder: N_FEATURES rows × HIDDEN cols
    b_dec: [f32; N_FEATURES],

    // RMSProp accumulators (per-parameter EMA of squared gradients).
    rms_w_enc: [[f32; N_FEATURES]; HIDDEN],
    rms_b_enc: [f32; HIDDEN],
    rms_w_dec: [[f32; HIDDEN]; N_FEATURES],
    rms_b_dec: [f32; N_FEATURES],

    /// Cumulative fitness (inverse reconstruction MSE over recent window).
    fitness: f32,
    fitness_count: u32,
}

impl SaeGenome {
    fn random(rng: &mut Rng) -> Self {
        // Xavier initialisation: scale = sqrt(2 / (fan_in + fan_out)).
        let enc_scale = (2.0 / (N_FEATURES + HIDDEN) as f32).sqrt();
        let dec_scale = (2.0 / (HIDDEN + N_FEATURES) as f32).sqrt();

        let mut w_enc = [[0.0f32; N_FEATURES]; HIDDEN];
        let mut w_dec = [[0.0f32; HIDDEN]; N_FEATURES];
        for h in 0..HIDDEN {
            for f in 0..N_FEATURES {
                w_enc[h][f] = rng.gaussian() * enc_scale;
            }
        }
        for f in 0..N_FEATURES {
            for h in 0..HIDDEN {
                w_dec[f][h] = rng.gaussian() * dec_scale;
            }
        }

        Self {
            feature_mask: rng.next_u64() as u16 | 0x000F, // at least 4 features on
            lr: BASE_LR * (0.5 + rng.uniform()),
            w_enc,
            b_enc: [0.0; HIDDEN],
            w_dec,
            b_dec: [0.0; N_FEATURES],
            rms_w_enc: [[1.0; N_FEATURES]; HIDDEN],
            rms_b_enc: [1.0; HIDDEN],
            rms_w_dec: [[1.0; HIDDEN]; N_FEATURES],
            rms_b_dec: [1.0; N_FEATURES],
            fitness: 0.0,
            fitness_count: 0,
        }
    }

    /// Apply feature mask: zero out unused features.
    fn mask_input(&self, x: &[f32; N_FEATURES]) -> [f32; N_FEATURES] {
        let mut out = *x;
        for i in 0..N_FEATURES {
            if self.feature_mask & (1 << i) == 0 {
                out[i] = 0.0;
            }
        }
        out
    }

    /// Forward pass: encode → TopK ReLU → decode.  Returns (reconstruction, hidden_activations).
    fn forward(&self, x: &[f32; N_FEATURES]) -> ([f32; N_FEATURES], [f32; HIDDEN]) {
        let masked = self.mask_input(x);

        // Encode: h = ReLU(W_e · x + b_e) — NEON-accelerated inner product.
        let mut h = [0.0f32; HIDDEN];
        for j in 0..HIDDEN {
            h[j] = (self.b_enc[j] + neon_dot16_f32(&self.w_enc[j], &masked)).max(0.0);
        }

        // TopK: keep only the top K activations, zero the rest.
        // Partial sort via selection — O(HIDDEN·K) ≈ 96 comparisons.
        let mut topk_indices = [0usize; TOP_K];
        let mut topk_vals = [f32::NEG_INFINITY; TOP_K];
        for j in 0..HIDDEN {
            // Find minimum in current topk.
            let mut min_idx = 0;
            for k in 1..TOP_K {
                if topk_vals[k] < topk_vals[min_idx] {
                    min_idx = k;
                }
            }
            if h[j] > topk_vals[min_idx] {
                topk_vals[min_idx] = h[j];
                topk_indices[min_idx] = j;
            }
        }
        let mut h_sparse = [0.0f32; HIDDEN];
        for k in 0..TOP_K {
            if topk_vals[k] > 0.0 {
                h_sparse[topk_indices[k]] = h[topk_indices[k]];
            }
        }

        // Decode: x̂ = W_d · h_sparse + b_d — NEON-accelerated inner product.
        let mut recon = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            recon[i] = self.b_dec[i] + neon_dot24_f32(&self.w_dec[i], &h_sparse);
        }

        (recon, h_sparse)
    }

    /// Reconstruction MSE (only over masked features).
    fn mse(&self, x: &[f32; N_FEATURES]) -> f32 {
        let masked = self.mask_input(x);
        let (recon, _) = self.forward(x);
        let active = self.feature_mask.count_ones().max(1) as f32;
        let mut sum = 0.0f32;
        for i in 0..N_FEATURES {
            if self.feature_mask & (1 << i) != 0 {
                let d = masked[i] - recon[i];
                sum += d * d;
            }
        }
        sum / active
    }

    /// Online SGD update with RMSProp adaptive learning rate (Cejnek & Bukovsky 2024).
    /// Single-sample gradient: ∂MSE/∂W via chain rule through TopK sparse activations.
    fn train_step(&mut self, x: &[f32; N_FEATURES]) {
        let masked = self.mask_input(x);
        let (recon, h_sparse) = self.forward(x);

        // Gradient of MSE w.r.t. reconstruction: dL/dx̂ = 2(x̂ - x) / N_active
        let active = self.feature_mask.count_ones().max(1) as f32;
        let mut d_recon = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            if self.feature_mask & (1 << i) != 0 {
                d_recon[i] = 2.0 * (recon[i] - masked[i]) / active;
            }
        }

        // Update decoder: W_d -= lr * d_recon ⊗ h_sparse^T
        for i in 0..N_FEATURES {
            for j in 0..HIDDEN {
                let g = d_recon[i] * h_sparse[j];
                self.rms_w_dec[i][j] =
                    RMSPROP_DECAY * self.rms_w_dec[i][j] + (1.0 - RMSPROP_DECAY) * g * g;
                let step = self.lr * g / (self.rms_w_dec[i][j].sqrt() + 1e-8);
                self.w_dec[i][j] -= step;
            }
            let g = d_recon[i];
            self.rms_b_dec[i] = RMSPROP_DECAY * self.rms_b_dec[i] + (1.0 - RMSPROP_DECAY) * g * g;
            self.b_dec[i] -= self.lr * g / (self.rms_b_dec[i].sqrt() + 1e-8);
        }

        // Backprop through decoder → hidden gradient (only non-zero for TopK active units).
        let mut d_h = [0.0f32; HIDDEN];
        for j in 0..HIDDEN {
            if h_sparse[j] > 0.0 {
                for i in 0..N_FEATURES {
                    d_h[j] += d_recon[i] * self.w_dec[i][j];
                }
            }
        }

        // Update encoder: W_e -= lr * d_h ⊗ x^T  (only active hidden units)
        for j in 0..HIDDEN {
            if d_h[j] == 0.0 {
                continue;
            }
            for i in 0..N_FEATURES {
                let g = d_h[j] * masked[i];
                self.rms_w_enc[j][i] =
                    RMSPROP_DECAY * self.rms_w_enc[j][i] + (1.0 - RMSPROP_DECAY) * g * g;
                let step = self.lr * g / (self.rms_w_enc[j][i].sqrt() + 1e-8);
                self.w_enc[j][i] -= step;
            }
            let g = d_h[j];
            self.rms_b_enc[j] = RMSPROP_DECAY * self.rms_b_enc[j] + (1.0 - RMSPROP_DECAY) * g * g;
            self.b_enc[j] -= self.lr * g / (self.rms_b_enc[j].sqrt() + 1e-8);
        }
    }

    /// Update fitness with new reconstruction error (exponential moving average).
    fn update_fitness(&mut self, mse: f32) {
        self.fitness_count += 1;
        // Fitness = inverse MSE (lower error = higher fitness).
        let inv_mse = 1.0 / (mse + 1e-6);
        if self.fitness_count == 1 {
            self.fitness = inv_mse;
        } else {
            self.fitness = 0.95 * self.fitness + 0.05 * inv_mse;
        }
    }
}

// ── MAP-Elites Niche (Mouret & Clune 2015) ──────────────────────────────────

/// 3 pressure zones × 2 feature focuses = 6 niches.
/// Each SAE competes within its niche only (speciation).
fn niche_of(pressure_zone: u8, feature_focus: u8) -> usize {
    (pressure_zone as usize % 3) * 2 + (feature_focus as usize % 2)
}

fn classify_pressure_zone(pressure: f32) -> u8 {
    if pressure < 0.30 {
        0
    }
    // idle
    else if pressure < 0.65 {
        1
    }
    // moderate
    else {
        2
    } // heavy
}

fn classify_feature_focus(mask: u16) -> u8 {
    // Balanced split: compare RATIO of active bits in each group, not raw count.
    // Group A (memory/pressure): bits 0-4 (pressure, velocity, predicted, swap, integral) = 5 bits
    // Group B (detectors+system): bits 5-15 (cusum, entropy, oom, monopoly, urgency, cpu, comp, dom, lat, proc, therm) = 11 bits
    // Normalise by group size so the niche isn't always "memory-heavy".
    let mem_frac = (mask & 0x001F).count_ones() as f32 / 5.0;
    let sys_frac = (mask >> 5).count_ones() as f32 / 11.0;
    if mem_frac >= sys_frac {
        0
    } else {
        1
    }
}

// ── Darwin-Boltzmann Anomaly Detector ────────────────────────────────────────

pub struct EvolvedAnomalyDetector {
    normaliser: OnlineNormaliser,
    hopfield: HopfieldMemory,
    population: Vec<SaeGenome>,
    rng: Rng,
    sample_count: usize,

    /// Fusion weight: α for Hopfield vs (1-α) for SAE ensemble.
    /// Adapts online via tracking which detector was more accurate.
    alpha: f32,
    hopfield_accuracy_ema: f32,
    sae_accuracy_ema: f32,

    /// Adaptive threshold via EMA-Welford stats on free energy.
    /// α=0.002 ≈ 500-sample window (~4 min at 2Hz).
    threshold_stats: EmaWelford,

    /// Recent pressure for niche assignment.
    recent_pressure: f32,
}

impl Default for EvolvedAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolvedAnomalyDetector {
    pub fn new() -> Self {
        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_1337);
        let population = (0..POP_SIZE).map(|_| SaeGenome::random(&mut rng)).collect();

        Self {
            normaliser: OnlineNormaliser::new(),
            hopfield: HopfieldMemory::new(),
            population,
            rng,
            sample_count: 0,
            alpha: 0.5,
            hopfield_accuracy_ema: 0.5,
            sae_accuracy_ema: 0.5,
            threshold_stats: EmaWelford::new(0.002),
            recent_pressure: 0.0,
        }
    }

    /// Feed a raw telemetry vector and return anomaly score in [0.0, 1.0].
    /// 0.0 = normal, >0.5 = notable deviation, >0.8 = severe anomaly.
    ///
    /// The score is the free energy F = α·E_hopfield + (1-α)·E_sae,
    /// normalised against the running distribution of F values.
    pub fn score(&mut self, raw: &[f32; N_FEATURES], pressure: f32) -> f64 {
        self.recent_pressure = pressure;
        let x = self.normaliser.update_and_normalise(raw);
        self.sample_count += 1;

        // Warmup: need at least 30 samples for meaningful normalisation.
        if self.sample_count < 30 {
            self.hopfield.store(&x);
            for ind in &mut self.population {
                ind.train_step(&x);
            }
            return 0.0;
        }

        // 1. Hopfield energy score.
        let e_hopfield = self.hopfield.energy(&x);

        // 2. SAE ensemble: compute MSE once per individual (avoid triple forward pass).
        let mut mse_cache = [0.0f32; POP_SIZE];
        let mut weighted_mse = 0.0f32;
        let mut total_fitness = 0.0f32;
        for (i, ind) in self.population.iter().enumerate() {
            let mse = ind.mse(&x);
            mse_cache[i] = mse;
            let w = ind.fitness.max(0.01);
            weighted_mse += mse * w;
            total_fitness += w;
        }
        let e_sae = sigmoid(weighted_mse / total_fitness.max(0.01));

        // 3. Free energy fusion (Friston): F = α·hopfield + (1-α)·sae.
        let free_energy = self.alpha * e_hopfield + (1.0 - self.alpha) * e_sae;

        // 4. Adaptive threshold: normalise against running distribution.
        self.threshold_stats.update(free_energy);
        let z_score = if self.threshold_stats.count > 50 {
            (free_energy - self.threshold_stats.mean) / self.threshold_stats.std()
        } else {
            0.0
        };

        // 5. Online training (only when not anomalous — avoid poisoning).
        //    Reuse mse_cache from step 2 instead of recomputing forward().
        let is_anomaly = z_score > THRESHOLD_SIGMA;
        if !is_anomaly {
            self.hopfield.store(&x);
            for (i, ind) in self.population.iter_mut().enumerate() {
                ind.update_fitness(mse_cache[i]);
                ind.train_step(&x);
            }
        }

        // 6. Adapt fusion weight α: track which detector is more stable.
        self.hopfield_accuracy_ema = 0.99 * self.hopfield_accuracy_ema + 0.01 * (1.0 - e_hopfield);
        self.sae_accuracy_ema = 0.99 * self.sae_accuracy_ema + 0.01 * (1.0 - e_sae);
        let total_acc = self.hopfield_accuracy_ema + self.sae_accuracy_ema;
        if total_acc > 0.0 {
            self.alpha = self.hopfield_accuracy_ema / total_acc;
        }

        // 7. Evolution step (Darwinian selection with MAP-Elites speciation).
        if self.sample_count.is_multiple_of(EVOLVE_INTERVAL) {
            self.evolve();
        }

        // Map z-score to [0, 1] via sigmoid: z=0→0.5, z=3→0.95, z=-3→0.05.
        // Shift so that normal (z=0) maps to ~0.0.
        let score = sigmoid(z_score - 1.5); // z=1.5→0.5, z=4.5→0.95
        (score as f64).clamp(0.0, 1.0)
    }

    /// Darwinian evolution step: tournament selection + mutation within niches.
    fn evolve(&mut self) {
        let pressure_zone = classify_pressure_zone(self.recent_pressure);

        // Assign niches.
        let niches: Vec<usize> = self
            .population
            .iter()
            .map(|ind| niche_of(pressure_zone, classify_feature_focus(ind.feature_mask)))
            .collect();

        // Find worst individual (lowest fitness, excluding elite per niche).
        let mut niche_best: [Option<(usize, f32)>; 6] = [None; 6];
        for (i, &n) in niches.iter().enumerate() {
            let entry = &mut niche_best[n];
            match entry {
                Some((_, best_f)) if self.population[i].fitness > *best_f => {
                    *entry = Some((i, self.population[i].fitness));
                }
                None => {
                    *entry = Some((i, self.population[i].fitness));
                }
                _ => {}
            }
        }
        let elite_set: Vec<usize> = niche_best
            .iter()
            .filter_map(|e| e.map(|(idx, _)| idx))
            .collect();

        // Tournament selection: pick 2 random non-elite, replace loser with mutant of winner.
        let non_elite: Vec<usize> = (0..POP_SIZE).filter(|i| !elite_set.contains(i)).collect();

        if non_elite.len() >= 2 {
            let a = non_elite[self.rng.usize(non_elite.len())];
            let mut b = a;
            while b == a {
                b = non_elite[self.rng.usize(non_elite.len())];
            }
            let (winner, loser) = if self.population[a].fitness >= self.population[b].fitness {
                (a, b)
            } else {
                (b, a)
            };
            self.mutate_into(winner, loser);
        }
    }

    /// Create a mutant copy of `src` and place it at index `dst`.
    fn mutate_into(&mut self, src: usize, dst: usize) {
        // Copy weights from winner.
        let src_mask = self.population[src].feature_mask;
        let src_lr = self.population[src].lr;

        // Mutate feature mask: flip 1-2 random bits.
        let mut new_mask = src_mask;
        let flips = if self.rng.uniform() < 0.3 { 2 } else { 1 };
        for _ in 0..flips {
            new_mask ^= 1 << self.rng.usize(N_FEATURES);
        }
        if new_mask.count_ones() < 4 {
            new_mask = src_mask; // preserve minimum features
        }

        // Mutate learning rate: log-normal perturbation.
        let new_lr = (src_lr * (1.0 + 0.1 * self.rng.gaussian())).clamp(1e-5, 0.01);

        // Copy weights and add noise (weight mutation).
        self.population[dst].feature_mask = new_mask;
        self.population[dst].lr = new_lr;
        self.population[dst].fitness = self.population[src].fitness * 0.8; // inheritance discount
        self.population[dst].fitness_count = 0;

        for h in 0..HIDDEN {
            for f in 0..N_FEATURES {
                self.population[dst].w_enc[h][f] =
                    self.population[src].w_enc[h][f] + 0.02 * self.rng.gaussian();
                self.population[dst].rms_w_enc[h][f] = 1.0; // reset optimizer state
            }
            self.population[dst].b_enc[h] = self.population[src].b_enc[h];
            self.population[dst].rms_b_enc[h] = 1.0;
        }
        for f in 0..N_FEATURES {
            for h in 0..HIDDEN {
                self.population[dst].w_dec[f][h] =
                    self.population[src].w_dec[f][h] + 0.02 * self.rng.gaussian();
                self.population[dst].rms_w_dec[f][h] = 1.0;
            }
            self.population[dst].b_dec[f] = self.population[src].b_dec[f];
            self.population[dst].rms_b_dec[f] = 1.0;
        }
    }

    /// Number of samples processed.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Whether the detector has enough data to produce meaningful scores.
    pub fn is_ready(&self) -> bool {
        self.sample_count >= 50
    }

    /// Current fusion weight (0=pure SAE, 1=pure Hopfield).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f32; N_FEATURES], b: &[f32; N_FEATURES]) -> f32 {
    neon_dot16_f32(a, b)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_vector(seed: f32) -> [f32; N_FEATURES] {
        let mut v = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            // Stable pattern: sinusoidal with small noise.
            v[i] = (seed + i as f32 * 0.3).sin() * 0.5 + 0.5;
        }
        v
    }

    fn make_anomalous_vector() -> [f32; N_FEATURES] {
        // Anomaly: inverted pattern (negative direction) — maximally different
        // from the normal sinusoidal baseline in dot-product space.
        let mut v = [0.0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            v[i] = -(1.0 + i as f32 * 0.3).sin() * 2.0;
        }
        v
    }

    #[test]
    fn warmup_returns_zero() {
        let mut det = EvolvedAnomalyDetector::new();
        for i in 0..29 {
            let v = make_normal_vector(i as f32 * 0.1);
            assert_eq!(det.score(&v, 0.3), 0.0, "warmup sample {i} should be 0");
        }
    }

    #[test]
    fn detector_learns_normal_baseline() {
        let mut det = EvolvedAnomalyDetector::new();
        // Train on 200 normal samples.
        for i in 0..200 {
            let v = make_normal_vector(i as f32 * 0.05);
            let _ = det.score(&v, 0.4);
        }
        // Normal sample should have low score.
        let normal = make_normal_vector(10.0);
        let s = det.score(&normal, 0.4);
        assert!(s < 0.6, "normal sample should score low, got {s}");
    }

    #[test]
    fn anomaly_scores_higher_than_normal() {
        let mut det = EvolvedAnomalyDetector::new();
        // Train on 200 normal samples.
        for i in 0..200 {
            let v = make_normal_vector(i as f32 * 0.05);
            let _ = det.score(&v, 0.4);
        }
        let normal_score = det.score(&make_normal_vector(10.0), 0.4);
        let anomaly_score = det.score(&make_anomalous_vector(), 0.4);
        assert!(
            anomaly_score > normal_score,
            "anomaly ({anomaly_score}) should score higher than normal ({normal_score})"
        );
    }

    #[test]
    fn hopfield_empty_returns_zero() {
        let h = HopfieldMemory::new();
        let x = [0.5f32; N_FEATURES];
        assert_eq!(h.energy(&x), 0.0);
    }

    #[test]
    fn hopfield_exact_match_low_energy() {
        let mut h = HopfieldMemory::new();
        let x = make_normal_vector(1.0);
        h.store(&x);
        let e = h.energy(&x);
        // Exact match should have very low energy.
        assert!(e < 0.5, "exact match energy should be low, got {e}");
    }

    #[test]
    fn hopfield_mismatch_higher_energy() {
        let mut h = HopfieldMemory::new();
        for i in 0..10 {
            h.store(&make_normal_vector(i as f32 * 0.2));
        }
        let normal_e = h.energy(&make_normal_vector(0.5));
        let anomaly_e = h.energy(&make_anomalous_vector());
        assert!(
            anomaly_e > normal_e,
            "mismatch ({anomaly_e}) should have higher energy than normal ({normal_e})"
        );
    }

    #[test]
    fn sae_forward_runs() {
        let mut rng = Rng::new(42);
        let sae = SaeGenome::random(&mut rng);
        let x = make_normal_vector(1.0);
        let (recon, hidden) = sae.forward(&x);
        // Reconstruction should have same dimension.
        assert_eq!(recon.len(), N_FEATURES);
        // TopK: at most TOP_K non-zero hidden activations.
        let active = hidden.iter().filter(|&&v| v > 0.0).count();
        assert!(active <= TOP_K, "expected ≤{TOP_K} active, got {active}");
    }

    #[test]
    fn sae_training_reduces_mse() {
        let mut rng = Rng::new(42);
        let mut sae = SaeGenome::random(&mut rng);
        let x = make_normal_vector(1.0);
        let mse_before = sae.mse(&x);
        for _ in 0..100 {
            sae.train_step(&x);
        }
        let mse_after = sae.mse(&x);
        assert!(
            mse_after < mse_before,
            "training should reduce MSE: {mse_before} → {mse_after}"
        );
    }

    #[test]
    fn evolution_runs_without_panic() {
        let mut det = EvolvedAnomalyDetector::new();
        for i in 0..EVOLVE_INTERVAL + 1 {
            let v = make_normal_vector(i as f32 * 0.1);
            let _ = det.score(&v, 0.5);
        }
        // If we got here, evolution didn't panic.
        assert!(det.sample_count() > EVOLVE_INTERVAL);
    }

    #[test]
    fn feature_mask_zeros_unused() {
        let mut rng = Rng::new(42);
        let mut sae = SaeGenome::random(&mut rng);
        sae.feature_mask = 0b0000_0000_0000_1111; // only first 4 features
        let x = [1.0f32; N_FEATURES];
        let masked = sae.mask_input(&x);
        for i in 4..N_FEATURES {
            assert_eq!(masked[i], 0.0, "feature {i} should be masked");
        }
        for i in 0..4 {
            assert_eq!(masked[i], 1.0, "feature {i} should be active");
        }
    }

    #[test]
    fn is_ready_after_warmup() {
        let mut det = EvolvedAnomalyDetector::new();
        assert!(!det.is_ready());
        for i in 0..50 {
            let v = make_normal_vector(i as f32);
            let _ = det.score(&v, 0.3);
        }
        assert!(det.is_ready());
    }

    #[test]
    fn normaliser_centres_data() {
        let mut norm = OnlineNormaliser::new();
        // Feed 100 samples centred at 5.0 with std ≈ 1.0.
        let mut rng = Rng::new(123);
        for _ in 0..100 {
            let mut v = [0.0f32; N_FEATURES];
            for f in 0..N_FEATURES {
                v[f] = 5.0 + rng.gaussian();
            }
            let _ = norm.update_and_normalise(&v);
        }
        // Normalised output of the mean should be near 0.
        let mean_vec = [5.0f32; N_FEATURES];
        let normed = norm.update_and_normalise(&mean_vec);
        for i in 0..N_FEATURES {
            assert!(
                normed[i].abs() < 1.0,
                "feature {i} normalised mean should be near 0, got {}",
                normed[i]
            );
        }
    }
}
