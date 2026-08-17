//! Bounded probe for the Core ML lane's cost against its benefit.
//!
//! The lane exists to be *faster* than arithmetic Apollo already has: the
//! deterministic `cpu_oracle_predict` produces the same four outputs, and
//! `configured_coreml_model_matches_the_cpu_oracle` pins them to within 1e-4.
//! So Core ML buys no accuracy here. What it could buy is latency, by routing a
//! forward pass onto the ANE or GPU.
//!
//! Whether that happens is not observable — Core ML publishes no per-inference
//! dispatch target, which is why `AneObservation::Unsupported` exists. This
//! probe therefore measures the only thing that *is* observable and is also the
//! thing that matters: end-to-end latency of the configured lane against the
//! oracle it would replace, on identical inputs.
//!
//! It is deliberately not on the normal path. It runs only when
//! `APOLLO_COREML_MODEL_PATH` is set, so CI and the daemon never pay for it.
//!
//! Run with:
//!   APOLLO_COREML_MODEL_PATH=/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel \
//!     cargo test -p apollo-engine --test coreml_backend_probe -- --nocapture

use apollo_engine::engine::coreml_predictor::{
    cpu_oracle_predict, CoreMlPredictor, PredictorBackend, TemporalFeatureVector,
    TEMPORAL_FEATURE_COUNT,
};
use std::time::Instant;

const WARMUP: usize = 200;
const SAMPLES: usize = 2_000;

/// Distinct inputs so neither lane can be measured on a single cached value.
fn probe_inputs() -> Vec<TemporalFeatureVector> {
    (0..64)
        .map(|index| {
            let mut values = [0.0_f32; TEMPORAL_FEATURE_COUNT];
            for (slot, value) in values.iter_mut().enumerate() {
                *value = (((index * 7 + slot * 13) % 97) as f32) / 97.0;
            }
            TemporalFeatureVector::new(values)
        })
        .collect()
}

fn percentiles(mut samples: Vec<u128>) -> (u128, u128) {
    samples.sort_unstable();
    let at = |q: f64| {
        let rank = ((samples.len() as f64) * q).ceil() as usize;
        samples[rank.clamp(1, samples.len()) - 1]
    };
    (at(0.50), at(0.95))
}

#[test]
fn coreml_lane_latency_against_the_cpu_oracle_it_would_replace() {
    if std::env::var_os("APOLLO_COREML_MODEL_PATH").is_none() {
        eprintln!("probe skipped: APOLLO_COREML_MODEL_PATH is unset");
        return;
    }
    let predictor = CoreMlPredictor::new();
    let status = predictor.status();
    if status.backend != PredictorBackend::CoreMl {
        eprintln!("probe skipped: Core ML unavailable ({:?})", status.reason);
        return;
    }
    let inputs = probe_inputs();

    for index in 0..WARMUP {
        let features = &inputs[index % inputs.len()];
        std::hint::black_box(predictor.predict(features));
        std::hint::black_box(cpu_oracle_predict(features));
    }

    let mut coreml_ns = Vec::with_capacity(SAMPLES);
    let mut oracle_ns = Vec::with_capacity(SAMPLES);
    for index in 0..SAMPLES {
        let features = &inputs[index % inputs.len()];
        let start = Instant::now();
        std::hint::black_box(predictor.predict(features));
        coreml_ns.push(start.elapsed().as_nanos());

        let start = Instant::now();
        std::hint::black_box(cpu_oracle_predict(features));
        oracle_ns.push(start.elapsed().as_nanos());
    }

    let (coreml_p50, coreml_p95) = percentiles(coreml_ns);
    let (oracle_p50, oracle_p95) = percentiles(oracle_ns);

    println!("configured_backend = {:?}", status.configured_backend);
    println!("ane_observation    = {}", status.ane_observation.as_str());
    println!("core ml   p50 {coreml_p50}ns  p95 {coreml_p95}ns");
    println!("cpu oracle p50 {oracle_p50}ns  p95 {oracle_p95}ns");
    println!(
        "ratio      p50 {:.1}x  p95 {:.1}x",
        coreml_p50 as f64 / oracle_p50.max(1) as f64,
        coreml_p95 as f64 / oracle_p95.max(1) as f64
    );

    // The probe reports; it does not legislate which lane should win. The one
    // thing it does assert is that the two lanes still agree, because a latency
    // comparison between lanes that compute different things is meaningless.
    for features in inputs.iter().take(8) {
        let coreml = predictor.predict(features).as_array();
        let oracle = cpu_oracle_predict(features).as_array();
        for (left, right) in coreml.into_iter().zip(oracle) {
            assert!(
                (left - right).abs() <= 0.000_1,
                "lanes diverged: coreml={left} oracle={right}"
            );
        }
    }
}
