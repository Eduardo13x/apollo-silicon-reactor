//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Compressor-Aware Memory Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests whether compressor_aware correctly decides Freeze/Hint/Skip
//! across realistic memory profiles. The goal is to minimize:
//!   - SIGCONT regret (freeze a process that was expensive to resume)
//!   - Missed freezes (skip a process that was cheap to freeze)
//!   - Unnecessary hints (hint when freeze would have been fine)
//!
//! Target file: src/engine/compressor_aware.rs

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::compressor_aware::{
        decide_enhanced, decide_memory_action, compressor_efficiency_score,
        MemoryAction, ProcessMemoryProfile, TempProfile,
    };

    fn profile(phys_mb: u64, compressed_mb: u64, purgeable_mb: u64) -> ProcessMemoryProfile {
        let phys = phys_mb * 1024 * 1024;
        let compressed = compressed_mb * 1024 * 1024;
        let purgeable = purgeable_mb * 1024 * 1024;
        ProcessMemoryProfile {
            pid: 1,
            phys_footprint: phys,
            compressed_bytes: compressed,
            purgeable_bytes: purgeable,
            compression_ratio: if compressed > 0 {
                (phys + compressed) as f64 / phys.max(1) as f64
            } else {
                1.0
            },
            working_set_bytes: phys,
            resident_bytes: phys,
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: BASIC DECISION MATRIX
    // ══════════════════════════════════════════════════════════════════════════

    /// M01: High compression (3:1), small footprint → Freeze (cheap decompress).
    #[test]
    fn m01_high_compression_small_footprint_freezes() {
        let p = profile(100, 200, 0); // ratio ≈ 3.0
        assert_eq!(
            decide_memory_action(&p, 0.50, 0.0),
            MemoryAction::Freeze,
            "3:1 compression = cheap freeze. ratio={}",
            p.compression_ratio
        );
    }

    /// M02: Low compression (1.1:1), 500MB footprint, normal pressure → Hint.
    /// Freezing this would cause swap I/O (incompressible + large).
    #[test]
    fn m02_low_compression_large_footprint_hints() {
        let p = profile(500, 50, 0); // ratio ≈ 1.1
        let action = decide_memory_action(&p, 0.50, 0.0);
        assert_eq!(
            action,
            MemoryAction::PressureHint,
            "1.1:1 ratio + 500MB = expensive freeze → hint. ratio={}",
            p.compression_ratio
        );
    }

    /// M03: Low compression + large footprint under EMERGENCY pressure → Freeze.
    /// Even though freeze is expensive, swap is the lesser evil vs OOM.
    #[test]
    fn m03_emergency_overrides_low_compression() {
        let p = profile(500, 50, 0); // ratio ≈ 1.1
        assert_eq!(
            decide_memory_action(&p, 0.90, 0.0),
            MemoryAction::Freeze,
            "Emergency (0.90) must freeze even with bad ratio"
        );
    }

    /// M04: Lots of purgeable memory → Hint is enough (kernel discards for free).
    #[test]
    fn m04_purgeable_prefers_hint() {
        let p = profile(300, 100, 80); // 80MB purgeable
        assert_eq!(
            decide_memory_action(&p, 0.70, 0.0),
            MemoryAction::PressureHint,
            "80MB purgeable → hint is enough, no need to freeze"
        );
    }

    /// M05: Thrashing (high faults) + decent compression → Freeze to break loop.
    #[test]
    fn m05_thrashing_compressible_freezes() {
        let p = profile(200, 150, 0); // ratio ≈ 1.75
        assert_eq!(
            decide_memory_action(&p, 0.60, 100.0),
            MemoryAction::Freeze,
            "Thrashing + ratio 1.75 = freeze breaks the decompress loop"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: ENHANCED DECISION (TEMPERATURE + WSS + SLC)
    // ══════════════════════════════════════════════════════════════════════════

    /// M06: WSS fits in SLC share → Freeze is free (pages stay cached).
    /// Even with bad compression ratio, SLC residence means zero decompression.
    #[test]
    fn m06_slc_fit_overrides_bad_ratio() {
        let p = profile(400, 20, 0); // bad ratio 1.05
        // WSS = 500KB, 2 active processes → SLC share = 4MB → fits
        let action = decide_enhanced(&p, None, Some(500_000), 2, 0.50, 0.0);
        assert_eq!(action, MemoryAction::Freeze, "WSS fits in SLC → free freeze regardless of ratio");
    }

    /// M07: Mostly compressed process (70% pages compressed) → Skip.
    /// Freezing would force swap (compressor pages evicted to disk).
    #[test]
    fn m07_mostly_compressed_skips() {
        let p = profile(500, 300, 0);
        let temp = TempProfile {
            pct_hot: 0.10,
            pct_dram: 0.20,
            pct_compressed: 0.70,
            sample_count: 8,
        };
        assert_eq!(
            decide_enhanced(&p, Some(&temp), None, 10, 0.60, 0.0),
            MemoryAction::Skip,
            "70% compressed → freeze would cause swap storm"
        );
    }

    /// M08: Actively hot process (85% hot pages) → Hint (not freeze).
    /// Process is using its pages right now — freeze would evict cache.
    #[test]
    fn m08_hot_process_gets_hint() {
        let p = profile(300, 0, 0);
        let temp = TempProfile {
            pct_hot: 0.85,
            pct_dram: 0.15,
            pct_compressed: 0.0,
            sample_count: 8,
        };
        assert_eq!(
            decide_enhanced(&p, Some(&temp), None, 10, 0.50, 0.0),
            MemoryAction::PressureHint,
            "85% hot pages → process is active, hint is gentler"
        );
    }

    /// M09: No enhanced data → must fall through to legacy decision.
    #[test]
    fn m09_no_enhanced_data_falls_through() {
        let p = profile(100, 200, 0); // 3:1 ratio
        let enhanced = decide_enhanced(&p, None, None, 10, 0.50, 0.0);
        let legacy = decide_memory_action(&p, 0.50, 0.0);
        assert_eq!(enhanced, legacy, "No enhanced data → must match legacy decision");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: EFFICIENCY SCORING & EDGE CASES
    // ══════════════════════════════════════════════════════════════════════════

    /// M10: High compression ratio → high efficiency score.
    /// 4:1 ratio should score near 1.0 (compressor is very effective).
    #[test]
    fn m10_high_ratio_high_efficiency() {
        let p = profile(100, 300, 0); // ratio = 4.0
        let eff = compressor_efficiency_score(&p);
        assert!(eff > 0.7, "4:1 ratio should have efficiency > 0.7, got {}", eff);
    }

    /// M11: Zero compression → efficiency = 1.0 (not burdening compressor).
    #[test]
    fn m11_no_compression_full_efficiency() {
        let p = profile(200, 0, 0);
        let eff = compressor_efficiency_score(&p);
        assert_eq!(eff, 1.0, "No compressed bytes → efficiency must be 1.0");
    }

    /// M12: Thrashing + LOW compression → Hint (freeze would cause swap storm).
    /// Different from M05 — here ratio is too low to benefit from freeze.
    #[test]
    fn m12_thrashing_incompressible_hints() {
        let p = profile(500, 50, 0); // ratio ≈ 1.1
        assert_eq!(
            decide_memory_action(&p, 0.60, 100.0),
            MemoryAction::PressureHint,
            "Thrashing + low ratio = freeze would cause swap storm → hint"
        );
    }

    /// M13: Warm process (50% DRAM, 20% hot, 30% compressed) with decent ratio.
    /// Enhanced should fall through to legacy and freeze (ratio ≥ 2.0).
    #[test]
    fn m13_warm_process_decent_ratio_freezes() {
        let p = profile(200, 200, 0); // ratio = 2.0
        let temp = TempProfile {
            pct_hot: 0.20,
            pct_dram: 0.50,
            pct_compressed: 0.30,
            sample_count: 8,
        };
        // pct_compressed ≤ 0.60, pct_hot ≤ 0.80 → falls through to legacy
        let action = decide_enhanced(&p, Some(&temp), None, 10, 0.50, 0.0);
        assert_eq!(
            action,
            MemoryAction::Freeze,
            "Warm process with 2:1 ratio → legacy freeze is correct"
        );
    }

    /// M14: SLC check must use DAMON WSS, not phys_footprint.
    /// phys = 400MB (won't fit SLC), but DAMON WSS = 1MB (hot working set is tiny).
    #[test]
    fn m14_damon_wss_overrides_footprint() {
        let p = profile(400, 0, 0); // 400MB phys
        // WSS = 1MB, 4 processes → SLC share = 2MB → fits
        let action = decide_enhanced(&p, None, Some(1_000_000), 4, 0.50, 0.0);
        assert_eq!(action, MemoryAction::Freeze, "DAMON WSS 1MB fits SLC despite 400MB phys");
    }

    /// M15: Borderline compression ratio (1.5) + moderate footprint (200MB).
    /// No thrashing, normal pressure. Should Freeze (ratio just meets threshold).
    #[test]
    fn m15_borderline_ratio_freezes() {
        let p = profile(200, 100, 0); // ratio = 1.5
        // The legacy path: not purgeable, not thrashing, ratio < 2.0,
        // but ratio >= 1.5 and footprint is moderate (200MB fits the fallthrough case)
        let action = decide_memory_action(&p, 0.50, 0.0);
        assert_eq!(
            action,
            MemoryAction::Freeze,
            "Ratio 1.5 + 200MB footprint should freeze (fallthrough). ratio={}",
            p.compression_ratio
        );
    }
}
