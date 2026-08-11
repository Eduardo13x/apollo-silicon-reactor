use serde::{Deserialize, Serialize};

use crate::engine::installation_identity::InstallationId;
use crate::engine::telemetry_medallion::{HardwareRegime, TelemetryContextSummary};

const MAX_LIVE_AGE_SECS: i64 = 30;
const MAX_FUTURE_SKEW_SECS: i64 = 5;
const MAX_CONTEXT_TEXT_BYTES: usize = 64;
const MIN_TEMP_C: f64 = -20.0;
const MAX_TEMP_C: f64 = 150.0;
const MAX_COMPONENT_WATTS: f64 = 500.0;
/// `PressureComponents::total_boost()` is an additive diagnostic, not a
/// fraction. Its current physical maximum is 1.74; retain headroom for new
/// factors while still rejecting corrupt magnitudes.
const MAX_PRESSURE_TOTAL_BOOST: f64 = 4.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextTier {
    #[default]
    Rejected,
    Silver,
    Gold,
}

impl ContextTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
            Self::Silver => "silver",
            Self::Gold => "gold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContextReason {
    NonFinite = 0,
    OutOfRange = 1,
    InvalidIdentity = 2,
    Stale = 3,
    FutureTimestamp = 4,
    Temporal = 5,
    ForeignHardware = 6,
    Coherence = 7,
    RequiredCollector = 8,
    UnknownInstallation = 9,
    UnknownHardware = 10,
}

impl ContextReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NonFinite => "non_finite",
            Self::OutOfRange => "out_of_range",
            Self::InvalidIdentity => "invalid_identity",
            Self::Stale => "stale",
            Self::FutureTimestamp => "future_timestamp",
            Self::Temporal => "temporal",
            Self::ForeignHardware => "foreign_hardware",
            Self::Coherence => "coherence",
            Self::RequiredCollector => "required_collector",
            Self::UnknownInstallation => "unknown_installation",
            Self::UnknownHardware => "unknown_hardware",
        }
    }
}

/// Closed field vocabulary for numeric context admission. A u64 bitset keeps
/// per-snapshot diagnostics allocation-free while still allowing the medallion
/// to aggregate each offending sensor independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ContextField {
    MemoryPressure,
    MemoryPressureRaw,
    CompressorPressure,
    CpuGlobalUsage,
    CpuMeanBusy,
    CpuMaxBusy,
    CpuPeggedFraction,
    StallFraction,
    UsedRamFraction,
    ThermalScore,
    FluidityScore,
    PerceptualLatencyScore,
    SchedulerJitterP95Ms,
    TopProcessCpu,
    WindowserverCpuFraction,
    SignalPressureSmooth,
    SignalPOom30s,
    SignalUrgency,
    SignalEntropyAnomaly,
    SignalTransformerAnomaly,
    ArousalLevel,
    MarkovPredictionConfidence,
    PressureTotalBoost,
    SignalPressureVelocity,
    SwapDeltaBytesPerSec,
    NaturalDrift,
    NarsDriftScore,
    ThrashingScore,
    RefaultDeltaPerSec,
    NetworkRetransmitsPerK,
    NetworkListenDropRate,
    MarkovPredictionEtaSecs,
    UserIdleSecs,
    PClusterTempC,
    EClusterTempC,
    GpuTempC,
    NandTempC,
    PackageWatts,
    CpuWatts,
    GpuWatts,
    DramWatts,
    AneWatts,
    PClusterUtil,
    EClusterUtil,
    AneUtilPct,
    BatteryWatts,
    BatteryPercent,
}

impl ContextField {
    pub const ALL: [Self; 47] = [
        Self::MemoryPressure,
        Self::MemoryPressureRaw,
        Self::CompressorPressure,
        Self::CpuGlobalUsage,
        Self::CpuMeanBusy,
        Self::CpuMaxBusy,
        Self::CpuPeggedFraction,
        Self::StallFraction,
        Self::UsedRamFraction,
        Self::ThermalScore,
        Self::FluidityScore,
        Self::PerceptualLatencyScore,
        Self::SchedulerJitterP95Ms,
        Self::TopProcessCpu,
        Self::WindowserverCpuFraction,
        Self::SignalPressureSmooth,
        Self::SignalPOom30s,
        Self::SignalUrgency,
        Self::SignalEntropyAnomaly,
        Self::SignalTransformerAnomaly,
        Self::ArousalLevel,
        Self::MarkovPredictionConfidence,
        Self::PressureTotalBoost,
        Self::SignalPressureVelocity,
        Self::SwapDeltaBytesPerSec,
        Self::NaturalDrift,
        Self::NarsDriftScore,
        Self::ThrashingScore,
        Self::RefaultDeltaPerSec,
        Self::NetworkRetransmitsPerK,
        Self::NetworkListenDropRate,
        Self::MarkovPredictionEtaSecs,
        Self::UserIdleSecs,
        Self::PClusterTempC,
        Self::EClusterTempC,
        Self::GpuTempC,
        Self::NandTempC,
        Self::PackageWatts,
        Self::CpuWatts,
        Self::GpuWatts,
        Self::DramWatts,
        Self::AneWatts,
        Self::PClusterUtil,
        Self::EClusterUtil,
        Self::AneUtilPct,
        Self::BatteryWatts,
        Self::BatteryPercent,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::MemoryPressure => "memory_pressure",
            Self::MemoryPressureRaw => "memory_pressure_raw",
            Self::CompressorPressure => "compressor_pressure",
            Self::CpuGlobalUsage => "cpu_global_usage",
            Self::CpuMeanBusy => "cpu_mean_busy",
            Self::CpuMaxBusy => "cpu_max_busy",
            Self::CpuPeggedFraction => "cpu_pegged_fraction",
            Self::StallFraction => "stall_fraction",
            Self::UsedRamFraction => "used_ram_fraction",
            Self::ThermalScore => "thermal_score",
            Self::FluidityScore => "fluidity_score",
            Self::PerceptualLatencyScore => "perceptual_latency_score",
            Self::SchedulerJitterP95Ms => "scheduler_jitter_p95_ms",
            Self::TopProcessCpu => "top_process_cpu",
            Self::WindowserverCpuFraction => "windowserver_cpu_fraction",
            Self::SignalPressureSmooth => "signal_pressure_smooth",
            Self::SignalPOom30s => "signal_p_oom_30s",
            Self::SignalUrgency => "signal_urgency",
            Self::SignalEntropyAnomaly => "signal_entropy_anomaly",
            Self::SignalTransformerAnomaly => "signal_transformer_anomaly",
            Self::ArousalLevel => "arousal_level",
            Self::MarkovPredictionConfidence => "markov_prediction_confidence",
            Self::PressureTotalBoost => "pressure_total_boost",
            Self::SignalPressureVelocity => "signal_pressure_velocity",
            Self::SwapDeltaBytesPerSec => "swap_delta_bytes_per_sec",
            Self::NaturalDrift => "natural_drift",
            Self::NarsDriftScore => "nars_drift_score",
            Self::ThrashingScore => "thrashing_score",
            Self::RefaultDeltaPerSec => "refault_delta_per_sec",
            Self::NetworkRetransmitsPerK => "network_retransmits_per_k",
            Self::NetworkListenDropRate => "network_listen_drop_rate",
            Self::MarkovPredictionEtaSecs => "markov_prediction_eta_secs",
            Self::UserIdleSecs => "user_idle_secs",
            Self::PClusterTempC => "p_cluster_temp_c",
            Self::EClusterTempC => "e_cluster_temp_c",
            Self::GpuTempC => "gpu_temp_c",
            Self::NandTempC => "nand_temp_c",
            Self::PackageWatts => "package_watts",
            Self::CpuWatts => "cpu_watts",
            Self::GpuWatts => "gpu_watts",
            Self::DramWatts => "dram_watts",
            Self::AneWatts => "ane_watts",
            Self::PClusterUtil => "p_cluster_util",
            Self::EClusterUtil => "e_cluster_util",
            Self::AneUtilPct => "ane_util_pct",
            Self::BatteryWatts => "battery_watts",
            Self::BatteryPercent => "battery_percent",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ContextFieldSet(u64);

impl ContextFieldSet {
    fn insert(&mut self, field: ContextField) {
        self.0 |= 1_u64 << field as u8;
    }

    pub fn contains(self, field: ContextField) -> bool {
        self.0 & (1_u64 << field as u8) != 0
    }

    pub fn iter(self) -> impl Iterator<Item = ContextField> {
        ContextField::ALL
            .into_iter()
            .filter(move |field| self.contains(*field))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextFieldViolation {
    pub field: ContextField,
    pub reason: ContextReason,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ValidationDiagnostics {
    fields: ContextFieldSet,
    primary: Option<ContextFieldViolation>,
}

impl ValidationDiagnostics {
    fn record(&mut self, field: ContextField, reason: ContextReason, value: f64) {
        self.fields.insert(field);
        if self.primary.is_none() {
            self.primary = Some(ContextFieldViolation {
                field,
                reason,
                value,
            });
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContextReasonSet(u16);

impl ContextReasonSet {
    pub fn insert(&mut self, reason: ContextReason) {
        self.0 |= 1_u16 << reason as u8;
    }

    pub fn contains(self, reason: ContextReason) -> bool {
        self.0 & (1_u16 << reason as u8) != 0
    }

    fn intersects(self, mask: u16) -> bool {
        self.0 & mask != 0
    }

    fn has_hard_rejection(self) -> bool {
        self.intersects(
            bit(ContextReason::NonFinite)
                | bit(ContextReason::OutOfRange)
                | bit(ContextReason::InvalidIdentity)
                | bit(ContextReason::Stale)
                | bit(ContextReason::FutureTimestamp)
                | bit(ContextReason::Temporal)
                | bit(ContextReason::ForeignHardware)
                | bit(ContextReason::Coherence),
        )
    }

    fn has_degradation(self) -> bool {
        self.intersects(
            bit(ContextReason::RequiredCollector)
                | bit(ContextReason::UnknownInstallation)
                | bit(ContextReason::UnknownHardware),
        )
    }
}

const fn bit(reason: ContextReason) -> u16 {
    1_u16 << reason as u8
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextAdmission {
    pub tier: ContextTier,
    pub quality: f64,
    pub reasons: ContextReasonSet,
    pub hardware_regime: HardwareRegime,
    pub installation_id: InstallationId,
    pub local_epoch: bool,
    pub violating_fields: ContextFieldSet,
    pub primary_violation: Option<ContextFieldViolation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextReasonCounters {
    pub non_finite: u64,
    pub out_of_range: u64,
    pub identity: u64,
    pub stale: u64,
    pub temporal: u64,
    pub foreign_hardware: u64,
    pub coherence: u64,
    pub collector: u64,
    pub unknown_installation: u64,
    pub unknown_hardware: u64,
}

impl ContextReasonCounters {
    pub fn record(&mut self, reasons: ContextReasonSet) {
        record_if(&mut self.non_finite, reasons, ContextReason::NonFinite);
        record_if(&mut self.out_of_range, reasons, ContextReason::OutOfRange);
        record_if(&mut self.identity, reasons, ContextReason::InvalidIdentity);
        if reasons.contains(ContextReason::Stale)
            || reasons.contains(ContextReason::FutureTimestamp)
        {
            self.stale = self.stale.saturating_add(1);
        }
        record_if(&mut self.temporal, reasons, ContextReason::Temporal);
        record_if(
            &mut self.foreign_hardware,
            reasons,
            ContextReason::ForeignHardware,
        );
        record_if(&mut self.coherence, reasons, ContextReason::Coherence);
        record_if(
            &mut self.collector,
            reasons,
            ContextReason::RequiredCollector,
        );
        record_if(
            &mut self.unknown_installation,
            reasons,
            ContextReason::UnknownInstallation,
        );
        record_if(
            &mut self.unknown_hardware,
            reasons,
            ContextReason::UnknownHardware,
        );
    }
}

fn record_if(counter: &mut u64, reasons: ContextReasonSet, reason: ContextReason) {
    if reasons.contains(reason) {
        *counter = counter.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ContextAdmissionInput<'a> {
    context: &'a TelemetryContextSummary,
    previous: Option<&'a TelemetryContextSummary>,
    now_unix: i64,
    installation_id: InstallationId,
    local_epoch: bool,
}

impl<'a> ContextAdmissionInput<'a> {
    pub fn live(
        context: &'a TelemetryContextSummary,
        previous: Option<&'a TelemetryContextSummary>,
        now_unix: i64,
        installation_id: InstallationId,
    ) -> Self {
        Self {
            context,
            previous,
            now_unix,
            installation_id,
            local_epoch: true,
        }
    }
}

pub fn classify(input: ContextAdmissionInput<'_>) -> ContextAdmission {
    let context = input.context;
    let hardware_regime = HardwareRegime::from_context(context);
    let mut reasons = ContextReasonSet::default();
    let mut diagnostics = ValidationDiagnostics::default();

    validate_required_numbers(context, &mut reasons, &mut diagnostics);
    validate_optional_numbers(context, &mut reasons, &mut diagnostics);
    validate_identity(context, &mut reasons);
    validate_coherence(context, hardware_regime, &mut reasons);
    validate_time(&input, &mut reasons);

    if !input.installation_id.is_known() {
        reasons.insert(ContextReason::UnknownInstallation);
    }
    if !hardware_regime.is_known() {
        reasons.insert(ContextReason::UnknownHardware);
    }
    if !context.collector_pressure_alive || !context.reactor_healthy {
        reasons.insert(ContextReason::RequiredCollector);
    }

    let structural_ok = !reasons.intersects(
        bit(ContextReason::NonFinite)
            | bit(ContextReason::OutOfRange)
            | bit(ContextReason::InvalidIdentity)
            | bit(ContextReason::Coherence),
    );
    let temporal_ok = !reasons.intersects(
        bit(ContextReason::Stale)
            | bit(ContextReason::FutureTimestamp)
            | bit(ContextReason::Temporal),
    );
    let origin_ok = input.installation_id.is_known()
        && hardware_regime.is_known()
        && !reasons.contains(ContextReason::ForeignHardware);
    let collectors_ok = !reasons.contains(ContextReason::RequiredCollector);
    let optional_coverage = optional_coverage(context);
    let mut quality = if structural_ok { 0.40 } else { 0.0 }
        + if temporal_ok { 0.20 } else { 0.0 }
        + if origin_ok { 0.20 } else { 0.0 }
        + if collectors_ok { 0.15 } else { 0.0 }
        + 0.05 * optional_coverage;

    let tier = if reasons.has_hard_rejection() {
        quality = quality.min(0.40);
        ContextTier::Rejected
    } else if reasons.has_degradation() {
        ContextTier::Silver
    } else {
        ContextTier::Gold
    };

    ContextAdmission {
        tier,
        quality: quality.clamp(0.0, 1.0),
        reasons,
        hardware_regime,
        installation_id: input.installation_id,
        local_epoch: input.local_epoch,
        violating_fields: diagnostics.fields,
        primary_violation: diagnostics.primary,
    }
}

fn validate_required_numbers(
    context: &TelemetryContextSummary,
    reasons: &mut ContextReasonSet,
    diagnostics: &mut ValidationDiagnostics,
) {
    let fractions = [
        (ContextField::MemoryPressure, context.memory_pressure),
        (ContextField::MemoryPressureRaw, context.memory_pressure_raw),
        (
            ContextField::CompressorPressure,
            context.compressor_pressure,
        ),
        (ContextField::CpuGlobalUsage, context.cpu_global_usage),
        (ContextField::CpuMeanBusy, context.cpu_mean_busy),
        (ContextField::CpuMaxBusy, context.cpu_max_busy),
        (ContextField::CpuPeggedFraction, context.cpu_pegged_fraction),
        (ContextField::StallFraction, context.stall_fraction),
        (ContextField::UsedRamFraction, context.used_ram_fraction),
        (ContextField::ThermalScore, context.thermal_score),
        (ContextField::FluidityScore, context.fluidity_score),
        (
            ContextField::PerceptualLatencyScore,
            context.perceptual_latency_score,
        ),
        (ContextField::TopProcessCpu, context.top_process_cpu),
        (
            ContextField::WindowserverCpuFraction,
            context.windowserver_cpu_fraction,
        ),
        (
            ContextField::SignalPressureSmooth,
            context.signal_pressure_smooth,
        ),
        (ContextField::SignalPOom30s, context.signal_p_oom_30s),
        (ContextField::SignalUrgency, context.signal_urgency),
        (
            ContextField::SignalTransformerAnomaly,
            context.signal_transformer_anomaly,
        ),
        (ContextField::ArousalLevel, context.arousal_level),
        (
            ContextField::MarkovPredictionConfidence,
            context.markov_prediction_confidence,
        ),
    ];
    for (field, value) in fractions {
        finite_range(field, value, 0.0, 1.0, reasons, diagnostics);
    }
    // Entropy anomaly is a signed z-like novelty signal, not a fraction. Real
    // contention routinely exceeds 1.0 and is intentionally consumed at >2.0.
    finite(
        ContextField::SignalEntropyAnomaly,
        context.signal_entropy_anomaly,
        reasons,
        diagnostics,
    );
    finite_min(
        ContextField::SchedulerJitterP95Ms,
        context.scheduler_jitter_p95_ms,
        0.0,
        reasons,
        diagnostics,
    );
    finite_range(
        ContextField::PressureTotalBoost,
        context.pressure_total_boost,
        0.0,
        MAX_PRESSURE_TOTAL_BOOST,
        reasons,
        diagnostics,
    );

    let signed = [
        (
            ContextField::SignalPressureVelocity,
            context.signal_pressure_velocity,
        ),
        (
            ContextField::SwapDeltaBytesPerSec,
            context.swap_delta_bytes_per_sec,
        ),
        (ContextField::NaturalDrift, context.natural_drift),
        (ContextField::NarsDriftScore, context.nars_drift_score),
    ];
    for (field, value) in signed {
        finite(field, value, reasons, diagnostics);
    }

    let non_negative = [
        (ContextField::ThrashingScore, context.thrashing_score),
        (
            ContextField::RefaultDeltaPerSec,
            context.refault_delta_per_sec,
        ),
        (
            ContextField::NetworkRetransmitsPerK,
            context.network_retransmits_per_k,
        ),
        (
            ContextField::NetworkListenDropRate,
            context.network_listen_drop_rate,
        ),
        (
            ContextField::MarkovPredictionEtaSecs,
            context.markov_prediction_eta_secs,
        ),
        (ContextField::UserIdleSecs, context.user_idle_secs),
    ];
    for (field, value) in non_negative {
        finite_min(field, value, 0.0, reasons, diagnostics);
    }
}

fn validate_optional_numbers(
    context: &TelemetryContextSummary,
    reasons: &mut ContextReasonSet,
    diagnostics: &mut ValidationDiagnostics,
) {
    for (field, value) in [
        (ContextField::PClusterTempC, context.p_cluster_temp_c),
        (ContextField::EClusterTempC, context.e_cluster_temp_c),
        (ContextField::GpuTempC, context.gpu_temp_c),
        (ContextField::NandTempC, context.nand_temp_c),
    ]
    .into_iter()
    .filter_map(|(field, value)| value.map(|value| (field, value)))
    {
        finite_range(field, value, MIN_TEMP_C, MAX_TEMP_C, reasons, diagnostics);
    }
    for (field, value) in [
        (ContextField::PackageWatts, context.package_watts),
        (ContextField::CpuWatts, context.cpu_watts),
        (ContextField::GpuWatts, context.gpu_watts),
        (ContextField::DramWatts, context.dram_watts),
        (ContextField::AneWatts, context.ane_watts),
    ]
    .into_iter()
    .filter_map(|(field, value)| value.map(|value| (field, value)))
    {
        finite_range(field, value, 0.0, MAX_COMPONENT_WATTS, reasons, diagnostics);
    }
    for (field, value) in [
        (ContextField::PClusterUtil, context.p_cluster_util),
        (ContextField::EClusterUtil, context.e_cluster_util),
    ]
    .into_iter()
    .filter_map(|(field, value)| value.map(|value| (field, value)))
    {
        finite_range(field, value, 0.0, 100.0, reasons, diagnostics);
    }
    if let Some(value) = context.ane_util_pct {
        finite_range(
            ContextField::AneUtilPct,
            value,
            0.0,
            100.0,
            reasons,
            diagnostics,
        );
    }
    if let Some(value) = context.battery_watts {
        finite(ContextField::BatteryWatts, value, reasons, diagnostics);
    }
    if context.battery_percent.is_some_and(|value| value > 100) {
        reasons.insert(ContextReason::OutOfRange);
        diagnostics.record(
            ContextField::BatteryPercent,
            ContextReason::OutOfRange,
            context.battery_percent.unwrap_or_default() as f64,
        );
    }
}

fn validate_identity(context: &TelemetryContextSummary, reasons: &mut ContextReasonSet) {
    for value in [
        context.workload.as_str(),
        context.effective_profile.as_str(),
        context.pressure_dominant_factor.as_str(),
    ] {
        if value.is_empty() || value.len() > MAX_CONTEXT_TEXT_BYTES {
            reasons.insert(ContextReason::InvalidIdentity);
        }
    }
    if context
        .foreground_app
        .as_ref()
        .is_some_and(|value| value.len() > 256)
    {
        reasons.insert(ContextReason::InvalidIdentity);
    }
}

fn validate_coherence(
    context: &TelemetryContextSummary,
    hardware_regime: HardwareRegime,
    reasons: &mut ContextReasonSet,
) {
    if context.total_ram_bytes == 0 || context.cpu_core_count == 0 {
        reasons.insert(ContextReason::Coherence);
    }
    if context.used_ram_bytes > context.total_ram_bytes
        || context.free_ram_bytes > context.total_ram_bytes
        || (context.swap_total_bytes == 0 && context.swap_used_bytes > 0)
        || (context.swap_total_bytes > 0 && context.swap_used_bytes > context.swap_total_bytes)
        || (context.disk_count > 0
            && (context.disk_total_bytes == 0
                || context.disk_available_bytes > context.disk_total_bytes))
        || context.top_process_rss_bytes > context.total_process_rss_bytes
    {
        reasons.insert(ContextReason::Coherence);
    }

    let regime_cores = hardware_regime
        .p_core_count
        .saturating_add(hardware_regime.e_core_count);
    if regime_cores > 0 && regime_cores != context.cpu_core_count {
        reasons.insert(ContextReason::ForeignHardware);
    }
}

fn validate_time(input: &ContextAdmissionInput<'_>, reasons: &mut ContextReasonSet) {
    let timestamp = input.context.timestamp_unix;
    if timestamp < input.now_unix.saturating_sub(MAX_LIVE_AGE_SECS) {
        reasons.insert(ContextReason::Stale);
    }
    if timestamp > input.now_unix.saturating_add(MAX_FUTURE_SKEW_SECS) {
        reasons.insert(ContextReason::FutureTimestamp);
    }
    if let Some(previous) = input.previous {
        if input.context.cycle < previous.cycle
            || input.context.timestamp_unix < previous.timestamp_unix
        {
            reasons.insert(ContextReason::Temporal);
        }
    }
}

fn finite(
    field: ContextField,
    value: f64,
    reasons: &mut ContextReasonSet,
    diagnostics: &mut ValidationDiagnostics,
) {
    if !value.is_finite() {
        reasons.insert(ContextReason::NonFinite);
        diagnostics.record(field, ContextReason::NonFinite, value);
    }
}

fn finite_min(
    field: ContextField,
    value: f64,
    min: f64,
    reasons: &mut ContextReasonSet,
    diagnostics: &mut ValidationDiagnostics,
) {
    if !value.is_finite() {
        reasons.insert(ContextReason::NonFinite);
        diagnostics.record(field, ContextReason::NonFinite, value);
    } else if value < min {
        reasons.insert(ContextReason::OutOfRange);
        diagnostics.record(field, ContextReason::OutOfRange, value);
    }
}

fn finite_range(
    field: ContextField,
    value: f64,
    min: f64,
    max: f64,
    reasons: &mut ContextReasonSet,
    diagnostics: &mut ValidationDiagnostics,
) {
    if !value.is_finite() {
        reasons.insert(ContextReason::NonFinite);
        diagnostics.record(field, ContextReason::NonFinite, value);
    } else if !(min..=max).contains(&value) {
        reasons.insert(ContextReason::OutOfRange);
        diagnostics.record(field, ContextReason::OutOfRange, value);
    }
}

fn optional_coverage(context: &TelemetryContextSummary) -> f64 {
    let present = [
        context.p_cluster_temp_c.is_some(),
        context.e_cluster_temp_c.is_some(),
        context.gpu_temp_c.is_some(),
        context.nand_temp_c.is_some(),
        context.p_cluster_util.is_some(),
        context.e_cluster_util.is_some(),
        context.package_watts.is_some(),
        context.cpu_watts.is_some(),
        context.gpu_watts.is_some(),
        context.dram_watts.is_some(),
        context.ane_watts.is_some(),
        context.ane_util_pct.is_some(),
        context.battery_percent.is_some(),
        context.battery_watts.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    present as f64 / 14.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::installation_identity::InstallationId;
    use crate::engine::telemetry_medallion::TelemetryContextSummary;
    use chrono::Utc;

    const LOCAL_ID: InstallationId = InstallationId(0x1020_3040_5060_7080);

    fn clean_context() -> TelemetryContextSummary {
        TelemetryContextSummary {
            cycle: 10,
            timestamp_unix: Utc::now().timestamp(),
            workload: "idle".to_string(),
            memory_pressure: 0.25,
            memory_pressure_raw: 0.25,
            compressor_pressure: 0.10,
            cpu_global_usage: 0.20,
            cpu_mean_busy: 0.20,
            cpu_max_busy: 0.40,
            cpu_pegged_fraction: 0.0,
            cpu_core_count: 10,
            used_ram_fraction: 0.50,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            used_ram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 8 * 1024 * 1024 * 1024,
            swap_total_bytes: 4 * 1024 * 1024 * 1024,
            disk_count: 1,
            disk_total_bytes: 1_000_000_000_000,
            disk_available_bytes: 500_000_000_000,
            thermal_score: 0.0,
            fluidity_score: 0.95,
            effective_profile: "balanced-root".to_string(),
            pressure_dominant_factor: "memory".to_string(),
            collector_pressure_alive: true,
            reactor_healthy: true,
            p_core_count: 4,
            e_core_count: 6,
            signal_pressure_smooth: 0.25,
            signal_p_oom_30s: 0.01,
            signal_urgency: 0.10,
            signal_entropy_anomaly: 0.0,
            signal_transformer_anomaly: 0.0,
            arousal_level: 0.10,
            markov_prediction_confidence: 0.0,
            markov_prediction_eta_secs: 0.0,
            predictive_intervention: "Observe".to_string(),
            ..TelemetryContextSummary::default()
        }
    }

    fn classify_live(context: &TelemetryContextSummary) -> ContextAdmission {
        classify(ContextAdmissionInput::live(
            context,
            None,
            context.timestamp_unix,
            LOCAL_ID,
        ))
    }

    fn assert_rejected(context: TelemetryContextSummary, reason: ContextReason) {
        let admission = classify_live(&context);
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(reason));
    }

    #[test]
    fn rejects_every_non_finite_required_family() {
        let mutations: &[fn(&mut TelemetryContextSummary)] = &[
            |c| c.memory_pressure = f64::NAN,
            |c| c.cpu_global_usage = f64::INFINITY,
            |c| c.fluidity_score = f64::NEG_INFINITY,
            |c| c.signal_pressure_smooth = f64::NAN,
            |c| c.natural_drift = f64::NAN,
            |c| c.markov_prediction_confidence = f64::NAN,
        ];
        for mutate in mutations {
            let mut context = clean_context();
            mutate(&mut context);
            let admission = classify(ContextAdmissionInput::live(
                &context,
                None,
                context.timestamp_unix,
                LOCAL_ID,
            ));
            assert_eq!(admission.tier, ContextTier::Rejected);
            assert!(admission.reasons.contains(ContextReason::NonFinite));
        }
    }

    #[test]
    fn rejects_cross_signal_and_temporal_contradictions() {
        let mut context = clean_context();
        context.used_ram_bytes = context.total_ram_bytes + 1;
        assert_rejected(context, ContextReason::Coherence);

        let mut context = clean_context();
        context.swap_used_bytes = context.swap_total_bytes + 1;
        assert_rejected(context, ContextReason::Coherence);

        let previous = clean_context();
        let mut regressed = clean_context();
        regressed.cycle = previous.cycle - 1;
        let admission = classify(ContextAdmissionInput::live(
            &regressed,
            Some(&previous),
            regressed.timestamp_unix,
            LOCAL_ID,
        ));
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::Temporal));
    }

    #[test]
    fn optional_sensor_absence_does_not_block_gold() {
        let mut context = clean_context();
        context.p_cluster_temp_c = None;
        context.e_cluster_temp_c = None;
        context.gpu_temp_c = None;
        context.ane_watts = None;
        context.ane_util_pct = None;
        context.collector_smc_alive = false;
        assert_eq!(classify_live(&context).tier, ContextTier::Gold);
    }

    #[test]
    fn missing_required_live_authority_is_silver() {
        let mut context = clean_context();
        context.collector_pressure_alive = false;
        assert_eq!(classify_live(&context).tier, ContextTier::Silver);

        let context = clean_context();
        let admission = classify(ContextAdmissionInput::live(
            &context,
            None,
            context.timestamp_unix,
            InstallationId::UNKNOWN,
        ));
        assert_eq!(admission.tier, ContextTier::Silver);
    }

    #[test]
    fn accepts_real_collector_utilization_units() {
        let mut context = clean_context();
        context.p_cluster_util = Some(100.0);
        context.e_cluster_util = Some(57.25);
        context.ane_util_pct = Some(100.0);
        assert_eq!(classify_live(&context).tier, ContextTier::Gold);
    }

    #[test]
    fn accepts_raw_additive_pressure_boost_above_one() {
        let mut context = clean_context();
        // effective_pressure::PressureComponents intentionally preserves the
        // uncapped sum for observability; all current factors max at 1.74.
        context.pressure_total_boost = 1.74;

        let admission = classify_live(&context);

        assert_eq!(admission.tier, ContextTier::Gold);
        assert!(
            !admission.reasons.contains(ContextReason::OutOfRange),
            "a valid additive boost must not be treated as a fraction"
        );

        context.pressure_total_boost = MAX_PRESSURE_TOTAL_BOOST + 0.01;
        let admission = classify_live(&context);
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::OutOfRange));
        assert_eq!(
            admission.primary_violation.map(|violation| violation.field),
            Some(ContextField::PressureTotalBoost)
        );
        assert!(admission
            .violating_fields
            .contains(ContextField::PressureTotalBoost));
    }

    #[test]
    fn accepts_entropy_novelty_above_fraction_range() {
        let mut context = clean_context();
        context.signal_entropy_anomaly = 5.0;

        let admission = classify_live(&context);

        assert_eq!(admission.tier, ContextTier::Gold);
        assert!(!admission.reasons.contains(ContextReason::OutOfRange));
    }

    #[test]
    fn rejects_invalid_latency_signals() {
        let mut context = clean_context();
        context.perceptual_latency_score = 1.01;
        assert_rejected(context, ContextReason::OutOfRange);

        let mut context = clean_context();
        context.scheduler_jitter_p95_ms = -0.01;
        assert_rejected(context, ContextReason::OutOfRange);
    }

    #[test]
    fn reports_each_invalid_numeric_field_without_growing_storage() {
        let mut context = clean_context();
        context.memory_pressure = 1.01;
        context.gpu_watts = Some(f64::NAN);

        let admission = classify_live(&context);

        assert!(admission
            .violating_fields
            .contains(ContextField::MemoryPressure));
        assert!(admission.violating_fields.contains(ContextField::GpuWatts));
        assert_eq!(std::mem::size_of::<ContextFieldSet>(), 8);
    }

    #[test]
    fn classifier_has_fixed_storage_and_no_history_growth() {
        assert_eq!(std::mem::size_of::<ContextReasonSet>(), 2);
        let context = clean_context();
        for _ in 0..1_000_000 {
            assert_eq!(classify_live(&context).tier, ContextTier::Gold);
        }
    }

    #[test]
    fn accepts_time_boundaries_and_same_second_progress() {
        let now = Utc::now().timestamp();
        let mut previous = clean_context();
        previous.timestamp_unix = now - MAX_LIVE_AGE_SECS;
        let mut current = previous.clone();
        current.cycle += 1;
        let admission = classify(ContextAdmissionInput::live(
            &current,
            Some(&previous),
            now,
            LOCAL_ID,
        ));
        assert_eq!(admission.tier, ContextTier::Gold);

        current.timestamp_unix = now + MAX_FUTURE_SKEW_SECS;
        assert_eq!(
            classify(ContextAdmissionInput::live(
                &current,
                Some(&previous),
                now,
                LOCAL_ID,
            ))
            .tier,
            ContextTier::Gold
        );
    }

    #[test]
    fn rejects_stale_future_and_regressed_timestamps() {
        let now = Utc::now().timestamp();
        let mut context = clean_context();
        context.timestamp_unix = now - MAX_LIVE_AGE_SECS - 1;
        let admission = classify(ContextAdmissionInput::live(&context, None, now, LOCAL_ID));
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::Stale));

        context.timestamp_unix = now + MAX_FUTURE_SKEW_SECS + 1;
        let admission = classify(ContextAdmissionInput::live(&context, None, now, LOCAL_ID));
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::FutureTimestamp));

        let previous = clean_context();
        context = previous.clone();
        context.cycle += 1;
        context.timestamp_unix -= 1;
        let admission = classify(ContextAdmissionInput::live(
            &context,
            Some(&previous),
            previous.timestamp_unix,
            LOCAL_ID,
        ));
        assert_eq!(admission.tier, ContextTier::Rejected);
        assert!(admission.reasons.contains(ContextReason::Temporal));
    }

    #[test]
    fn rejects_optional_values_only_when_present_and_impossible() {
        let mut context = clean_context();
        context.p_cluster_temp_c = Some(MAX_TEMP_C + 0.1);
        assert_rejected(context, ContextReason::OutOfRange);

        let mut context = clean_context();
        context.package_watts = Some(f64::NAN);
        assert_rejected(context, ContextReason::NonFinite);

        let mut context = clean_context();
        context.p_cluster_util = Some(100.01);
        assert_rejected(context, ContextReason::OutOfRange);

        let mut context = clean_context();
        context.battery_watts = Some(-12.0);
        assert_eq!(classify_live(&context).tier, ContextTier::Gold);
    }

    #[test]
    fn rejects_live_hardware_contradiction_but_degrades_unknown_hardware() {
        let mut context = clean_context();
        context.cpu_core_count = 8;
        assert_rejected(context, ContextReason::ForeignHardware);

        let mut context = clean_context();
        context.p_core_count = 0;
        context.e_core_count = 0;
        let admission = classify_live(&context);
        assert_eq!(admission.tier, ContextTier::Silver);
        assert!(admission.reasons.contains(ContextReason::UnknownHardware));
    }

    #[test]
    fn reason_counters_are_stable_and_saturating() {
        let mut reasons = ContextReasonSet::default();
        reasons.insert(ContextReason::NonFinite);
        reasons.insert(ContextReason::RequiredCollector);
        reasons.insert(ContextReason::FutureTimestamp);
        assert_eq!(reasons.0, 0b1_0001_0001);

        let mut counters = ContextReasonCounters {
            non_finite: u64::MAX,
            ..ContextReasonCounters::default()
        };
        counters.record(reasons);
        assert_eq!(counters.non_finite, u64::MAX);
        assert_eq!(counters.collector, 1);
        assert_eq!(counters.stale, 1);
    }
}
