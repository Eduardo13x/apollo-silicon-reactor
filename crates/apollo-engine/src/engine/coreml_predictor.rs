//! Bounded temporal predictor foundation with an honest Core ML escape hatch.
//!
//! The CPU oracle is the reference implementation. Core ML is optional and is
//! only used when the macOS bridge can load a model whose input schema and
//! embedded hashes match this module. The default build has no model artifact,
//! so it reports Core ML as unavailable and stays on the oracle.

#[cfg(target_os = "macos")]
use std::ffi::{c_char, c_void, CStr};
use std::sync::Mutex;

pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_TEMPORAL_FEATURES: usize = 256;
pub const TEMPORAL_FEATURE_COUNT: usize = 16;
pub const PREDICTION_OUTPUT_COUNT: usize = 4;

pub const TEMPORAL_FEATURE_NAMES: [&str; TEMPORAL_FEATURE_COUNT] = [
    "load",
    "load_delta",
    "transition",
    "transition_delta",
    "pressure",
    "pressure_delta",
    "p95",
    "p95_delta",
    "cpu_utilization",
    "memory_pressure",
    "io_pressure",
    "thermal_pressure",
    "run_queue",
    "active_work",
    "sample_age",
    "load_transition_coupling",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalFeatureSchema {
    pub version: u32,
    pub feature_count: usize,
    pub max_feature_count: usize,
    pub hash: u64,
}

pub const TEMPORAL_SCHEMA: TemporalFeatureSchema = TemporalFeatureSchema {
    version: SCHEMA_VERSION,
    feature_count: TEMPORAL_FEATURE_COUNT,
    max_feature_count: MAX_TEMPORAL_FEATURES,
    hash: TEMPORAL_SCHEMA_HASH,
};

const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

const fn fnv1a64_u32(mut hash: u64, value: u32) -> u64 {
    let mut shift = 0;
    while shift < 32 {
        hash ^= ((value >> shift) & 0xff) as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        shift += 8;
    }
    hash
}

const fn fnv1a64_u64(mut hash: u64, value: u64) -> u64 {
    hash = fnv1a64_u32(hash, value as u32);
    fnv1a64_u32(hash, (value >> 32) as u32)
}

pub const TEMPORAL_SCHEMA_HASH: u64 = fnv1a64(
    b"apollo.temporal.v1\0load\0load_delta\0transition\0transition_delta\0pressure\0pressure_delta\0p95\0p95_delta\0cpu_utilization\0memory_pressure\0io_pressure\0thermal_pressure\0run_queue\0active_work\0sample_age\0load_transition_coupling\0",
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSchemaError {
    WrongLength { expected: usize, actual: usize },
    NonFinite { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalFeatureVector {
    version: u32,
    values: [f32; TEMPORAL_FEATURE_COUNT],
}

impl TemporalFeatureVector {
    /// Builds a version-one vector, sanitizing invalid values and bounding all
    /// features to the normalized [-1, 1] model domain.
    pub fn new(values: [f32; TEMPORAL_FEATURE_COUNT]) -> Self {
        let mut sanitized = [0.0_f32; TEMPORAL_FEATURE_COUNT];
        for (destination, source) in sanitized.iter_mut().zip(values) {
            *destination = sanitize_feature(source);
        }
        Self {
            version: SCHEMA_VERSION,
            values: sanitized,
        }
    }

    pub fn try_from_slice(values: &[f32]) -> Result<Self, FeatureSchemaError> {
        if values.len() != TEMPORAL_FEATURE_COUNT {
            return Err(FeatureSchemaError::WrongLength {
                expected: TEMPORAL_FEATURE_COUNT,
                actual: values.len(),
            });
        }
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(FeatureSchemaError::NonFinite { index });
        }
        let mut copy = [0.0_f32; TEMPORAL_FEATURE_COUNT];
        copy.copy_from_slice(values);
        Ok(Self::new(copy))
    }

    pub fn from_observations(
        current: TemporalObservation,
        previous: Option<TemporalObservation>,
    ) -> Self {
        let previous = previous.unwrap_or(current);
        Self::new([
            current.load,
            current.load - previous.load,
            current.transition,
            current.transition - previous.transition,
            current.pressure,
            current.pressure - previous.pressure,
            current.p95,
            current.p95 - previous.p95,
            current.cpu_utilization,
            current.memory_pressure,
            current.io_pressure,
            current.thermal_pressure,
            current.run_queue,
            current.active_work,
            current.sample_age,
            (current.load - previous.load) * (current.transition - previous.transition),
        ])
    }

    pub const fn schema_version(self) -> u32 {
        self.version
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub const fn as_array(self) -> [f32; TEMPORAL_FEATURE_COUNT] {
        self.values
    }
}

impl Default for TemporalFeatureVector {
    fn default() -> Self {
        Self::new([0.0; TEMPORAL_FEATURE_COUNT])
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TemporalObservation {
    pub load: f32,
    pub transition: f32,
    pub pressure: f32,
    pub p95: f32,
    pub cpu_utilization: f32,
    pub memory_pressure: f32,
    pub io_pressure: f32,
    pub thermal_pressure: f32,
    pub run_queue: f32,
    pub active_work: f32,
    pub sample_age: f32,
}

const MODEL_BIASES: [f32; PREDICTION_OUTPUT_COUNT] = [0.06, 0.04, 0.05, 0.06];

// Output order is load, transition, pressure, p95. All weights are bounded
// constants; the final clamp is part of the model contract.
const MODEL_WEIGHTS: [[f32; TEMPORAL_FEATURE_COUNT]; PREDICTION_OUTPUT_COUNT] = [
    [
        0.56, 0.16, 0.00, 0.00, 0.06, 0.02, 0.04, 0.00, 0.12, 0.02, 0.00, 0.00, 0.04, 0.06, -0.02,
        0.02,
    ],
    [
        0.10, 0.12, 0.40, 0.18, 0.04, 0.06, 0.02, 0.02, 0.04, 0.00, 0.02, 0.00, 0.04, 0.02, -0.01,
        0.06,
    ],
    [
        0.10, 0.06, 0.08, 0.04, 0.46, 0.18, 0.04, 0.02, 0.02, 0.16, 0.08, 0.10, 0.04, 0.04, 0.00,
        0.02,
    ],
    [
        0.12, 0.04, 0.16, 0.08, 0.18, 0.08, 0.26, 0.14, 0.04, 0.06, 0.04, 0.04, 0.02, 0.02, 0.02,
        0.04,
    ],
];

const fn calculate_model_hash() -> u64 {
    let mut hash = fnv1a64(b"apollo.fixed-bounded-linear-model.v1\0");
    let mut output = 0;
    while output < PREDICTION_OUTPUT_COUNT {
        hash = fnv1a64_u32(hash, MODEL_BIASES[output].to_bits());
        let mut feature = 0;
        while feature < TEMPORAL_FEATURE_COUNT {
            hash = fnv1a64_u32(hash, MODEL_WEIGHTS[output][feature].to_bits());
            feature += 1;
        }
        output += 1;
    }
    fnv1a64_u64(hash, TEMPORAL_SCHEMA_HASH)
}

pub const MODEL_HASH: u64 = calculate_model_hash();

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Prediction {
    pub load: f32,
    pub transition: f32,
    pub pressure: f32,
    pub p95: f32,
}

impl Prediction {
    pub fn from_array(values: [f32; PREDICTION_OUTPUT_COUNT]) -> Self {
        Self {
            load: bounded_output(values[0]),
            transition: bounded_output(values[1]),
            pressure: bounded_output(values[2]),
            p95: bounded_output(values[3]),
        }
    }

    pub const fn as_array(self) -> [f32; PREDICTION_OUTPUT_COUNT] {
        [self.load, self.transition, self.pressure, self.p95]
    }

    pub fn is_finite(self) -> bool {
        self.as_array().iter().all(|value| value.is_finite())
    }
}

pub fn cpu_oracle_predict(features: &TemporalFeatureVector) -> Prediction {
    let values = features.as_slice();
    let mut outputs = [0.0_f32; PREDICTION_OUTPUT_COUNT];
    for output in 0..PREDICTION_OUTPUT_COUNT {
        let mut value = MODEL_BIASES[output];
        for (feature, weight) in values.iter().zip(MODEL_WEIGHTS[output]) {
            value += *feature * weight;
        }
        outputs[output] = bounded_output(value);
    }
    Prediction::from_array(outputs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CoreMlBackend {
    CpuAndNeuralEngine = 1,
    All = 2,
    CpuOnly = 3,
}

impl CoreMlBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuAndNeuralEngine => "cpu-and-neural-engine",
            Self::All => "all",
            Self::CpuOnly => "cpu-only",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorBackend {
    CoreMl,
    CpuOracle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictorStatus {
    pub backend: PredictorBackend,
    pub requested_backend: CoreMlBackend,
    pub effective_backend: Option<CoreMlBackend>,
    pub model_available: bool,
    pub ane_execution_measured: bool,
    pub schema_hash: u64,
    pub model_hash: u64,
    pub reason: Option<String>,
}

impl PredictorStatus {
    fn cpu_oracle(reason: impl Into<String>) -> Self {
        Self {
            backend: PredictorBackend::CpuOracle,
            requested_backend: CoreMlBackend::CpuAndNeuralEngine,
            effective_backend: None,
            model_available: false,
            ane_execution_measured: false,
            schema_hash: TEMPORAL_SCHEMA_HASH,
            model_hash: MODEL_HASH,
            reason: Some(reason.into()),
        }
    }
}

enum PredictorImplementation {
    CpuOracle,
    #[cfg(target_os = "macos")]
    CoreMl(NativeCoreMlContext),
}

pub struct CoreMlPredictor {
    state: Mutex<PredictorState>,
}

struct PredictorState {
    implementation: PredictorImplementation,
    status: PredictorStatus,
}

impl CoreMlPredictor {
    /// Attempts Core ML with CPU+ANE, then All, then CPU-only. With no model
    /// path configured, this intentionally returns an oracle-backed predictor.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            return Self::try_coreml();
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::oracle_with_reason("Core ML unavailable on this platform")
        }
    }

    pub fn cpu_oracle() -> Self {
        Self::oracle_with_reason("Core ML disabled; using deterministic CPU oracle")
    }

    pub fn status(&self) -> PredictorStatus {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.status.clone()
    }

    /// Runs at most one inference at a time. Native failures permanently
    /// demote this instance to the deterministic CPU oracle.
    pub fn predict(&self, features: &TemporalFeatureVector) -> Prediction {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(target_os = "macos")]
        if let PredictorImplementation::CoreMl(context) = &state.implementation {
            let mut output = [0.0_f32; PREDICTION_OUTPUT_COUNT];
            let result = unsafe {
                apollo_coreml_predict(
                    context.pointer(),
                    features.as_slice().as_ptr(),
                    TEMPORAL_FEATURE_COUNT as u32,
                    output.as_mut_ptr(),
                )
            };
            if result == 0 && output.iter().all(|value| value.is_finite()) {
                return Prediction::from_array(output);
            }

            let implementation = std::mem::replace(
                &mut state.implementation,
                PredictorImplementation::CpuOracle,
            );
            drop(implementation);
            state.status = PredictorStatus::cpu_oracle(
                "Core ML inference failed; using deterministic CPU oracle",
            );
        }
        cpu_oracle_predict(features)
    }

    fn oracle_with_reason(reason: &str) -> Self {
        Self {
            state: Mutex::new(PredictorState {
                implementation: PredictorImplementation::CpuOracle,
                status: PredictorStatus::cpu_oracle(reason),
            }),
        }
    }

    #[cfg(target_os = "macos")]
    fn try_coreml() -> Self {
        let mut native_status = NativeCoreMlStatus::default();
        let context = unsafe {
            apollo_coreml_create(
                TEMPORAL_SCHEMA_HASH,
                MODEL_HASH,
                TEMPORAL_FEATURE_COUNT as u32,
                &mut native_status,
            )
        };
        let Some(context) = std::ptr::NonNull::new(context) else {
            let reason = native_status_reason(&native_status).unwrap_or_else(|| {
                "Core ML model unavailable; using deterministic CPU oracle".into()
            });
            return Self::oracle_with_reason(&reason);
        };

        let status = PredictorStatus {
            backend: PredictorBackend::CoreMl,
            requested_backend: coreml_backend(native_status.requested_backend)
                .unwrap_or(CoreMlBackend::CpuAndNeuralEngine),
            effective_backend: coreml_backend(native_status.effective_backend),
            model_available: native_status.model_available != 0,
            // Core ML's computeUnits is a permission/configuration, not proof
            // of execution. The bridge never sets this without measured data.
            ane_execution_measured: native_status.ane_execution_measured != 0,
            schema_hash: TEMPORAL_SCHEMA_HASH,
            model_hash: MODEL_HASH,
            reason: native_status_reason(&native_status),
        };
        Self {
            state: Mutex::new(PredictorState {
                implementation: PredictorImplementation::CoreMl(NativeCoreMlContext { context }),
                status,
            }),
        }
    }
}

impl Default for CoreMlPredictor {
    fn default() -> Self {
        Self::new()
    }
}

fn sanitize_feature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn bounded_output(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else if value > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[cfg(target_os = "macos")]
const NATIVE_REASON_CAPACITY: usize = 512;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCoreMlStatus {
    requested_backend: u32,
    effective_backend: u32,
    model_available: u32,
    ane_execution_measured: u32,
    reason: [c_char; NATIVE_REASON_CAPACITY],
}

#[cfg(target_os = "macos")]
impl Default for NativeCoreMlStatus {
    fn default() -> Self {
        Self {
            requested_backend: CoreMlBackend::CpuAndNeuralEngine as u32,
            effective_backend: 0,
            model_available: 0,
            ane_execution_measured: 0,
            reason: [0; NATIVE_REASON_CAPACITY],
        }
    }
}

#[cfg(target_os = "macos")]
struct NativeCoreMlContext {
    context: std::ptr::NonNull<c_void>,
}

#[cfg(target_os = "macos")]
unsafe impl Send for NativeCoreMlContext {}

#[cfg(target_os = "macos")]
impl NativeCoreMlContext {
    fn pointer(&self) -> *mut c_void {
        self.context.as_ptr()
    }
}

#[cfg(target_os = "macos")]
impl Drop for NativeCoreMlContext {
    fn drop(&mut self) {
        unsafe { apollo_coreml_destroy(self.context.as_ptr()) };
    }
}

#[cfg(target_os = "macos")]
fn coreml_backend(value: u32) -> Option<CoreMlBackend> {
    match value {
        1 => Some(CoreMlBackend::CpuAndNeuralEngine),
        2 => Some(CoreMlBackend::All),
        3 => Some(CoreMlBackend::CpuOnly),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn native_status_reason(status: &NativeCoreMlStatus) -> Option<String> {
    let reason = unsafe { CStr::from_ptr(status.reason.as_ptr()) };
    let reason = reason.to_string_lossy();
    (!reason.is_empty()).then(|| reason.into_owned())
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn apollo_coreml_create(
        expected_schema_hash: u64,
        expected_model_hash: u64,
        feature_count: u32,
        status: *mut NativeCoreMlStatus,
    ) -> *mut c_void;
    fn apollo_coreml_destroy(context: *mut c_void);
    fn apollo_coreml_predict(
        context: *mut c_void,
        features: *const f32,
        feature_count: u32,
        output: *mut f32,
    ) -> i32;
}
