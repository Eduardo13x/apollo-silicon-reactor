//! Contract for the ledger-to-lab endpoint wire.
//!
//! Covers the three failures that kept the lab at `would N / open 0 / Gold 0`:
//! the endpoint contract was a hard-coded `false`, the lab's action keys never
//! matched the ones production writes to `DecisionLedger`, and no adapter
//! turned a `ResolvedDecisionEpisode` into a `TimedPairEndpoint`.

use apollo_engine::engine::decision_ledger::{
    DecisionEnvelope, DecisionId, DecisionLifecycle, ExecutionDisposition, ExecutionReceipt,
    ReceiptAttribution, ResolvedDecisionEpisode,
};
use apollo_engine::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationKey, ExplorationMetadata,
    ExplorationMode, ExplorationOrigin, HardwareIdentity, ProbeCorrelation, TerminalDiagnostic,
};
use apollo_engine::engine::microexperiment_actions::{
    canonical_action_key, family_horizon_cycles, parse_action_key, ActionKeyError, ActionVariant,
    LEGACY_MARKOV_KEY,
};
use apollo_engine::engine::microexperiment_endpoints::{
    EndpointUtilitySample, MicroexperimentEndpointAdapter, OBSERVATION_LIVENESS_CYCLES,
};
use apollo_engine::engine::microexperiment_lab::{
    ArmKind, CandidateDisposition, HorizonClosure, LabPhase, MicroexperimentLab, PairCandidate,
    PairDirective, PairGates, PairGoldRecord, RollbackClosure, TimedEndpointDisposition,
    TimedPairEndpoint,
};
use apollo_engine::engine::telemetry_medallion::ActuatorFamily;

const WASHOUT: u32 = 2;

fn origin() -> ExplorationOrigin {
    ExplorationOrigin {
        installation_id: 0xA110,
        hardware: HardwareIdentity {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        },
    }
}

fn qos_candidate(sequence: u64) -> PairCandidate {
    PairCandidate {
        sequence,
        origin: origin(),
        family: ActuatorFamily::InteractionQos,
        action_class: ActionClass::InteractionForeground,
        treatment_arm: ExplorationArm::InteractionQosStandard,
        context: ExplorationContext::Interactive,
        action_key: canonical_action_key(
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosStandard,
        )
        .expect("catalogued key")
        .to_string(),
        stratum_hash: 0x51,
        horizon_cycles: 4,
        washout_cycles: WASHOUT,
        minimum_effect_micros: 500,
    }
}

fn treatment_metadata() -> ExplorationMetadata {
    ExplorationMetadata {
        correlation: ProbeCorrelation(11),
        family: ActuatorFamily::InteractionQos,
        key: ExplorationKey {
            family: ActuatorFamily::InteractionQos,
            mode: ExplorationMode::Treatment,
            arm: ExplorationArm::InteractionQosStandard,
            action_class: ActionClass::InteractionForeground,
            context: ExplorationContext::Interactive,
        },
        arm: ExplorationArm::InteractionQosStandard,
        treatment: true,
        committed: true,
        cancelled: None,
    }
}

fn control_metadata() -> ExplorationMetadata {
    ExplorationMetadata {
        correlation: ProbeCorrelation(12),
        family: ActuatorFamily::InteractionQos,
        key: ExplorationKey {
            family: ActuatorFamily::InteractionQos,
            mode: ExplorationMode::Control,
            arm: ExplorationArm::InteractionQosStandard,
            action_class: ActionClass::InteractionForeground,
            context: ExplorationContext::Interactive,
        },
        arm: ExplorationArm::InteractionQosStandard,
        treatment: false,
        committed: false,
        cancelled: None,
    }
}

/// Shape of what `DecisionLedger::archive` hands back at the tail of a cycle.
fn episode(
    id: u64,
    action_key: &str,
    lifecycle: DecisionLifecycle,
    settled_cycle: u64,
    exploration: Option<ExplorationMetadata>,
) -> ResolvedDecisionEpisode {
    let disposition = match lifecycle {
        DecisionLifecycle::Applied => Some(ExecutionDisposition::Applied),
        DecisionLifecycle::Reverted => Some(ExecutionDisposition::Reverted),
        DecisionLifecycle::Failed => Some(ExecutionDisposition::Failed),
        DecisionLifecycle::NoOp => Some(ExecutionDisposition::NoOp),
        DecisionLifecycle::Blocked => Some(ExecutionDisposition::Blocked),
        _ => None,
    };
    let attribution = ReceiptAttribution::local("acceleration-lease");
    ResolvedDecisionEpisode {
        id: DecisionId(id),
        lifecycle: DecisionLifecycle::Settled,
        settled_cycle,
        authority_eligible: lifecycle == DecisionLifecycle::Applied && exploration.is_none(),
        envelope: DecisionEnvelope {
            id: DecisionId(id),
            action_key: action_key.to_string(),
            target: "pid:4242".to_string(),
            proposed_cycle: settled_cycle,
            lifecycle,
            terminal_attribution: Some(attribution.clone()),
            receipt: disposition.map(|disposition| ExecutionReceipt {
                receipt_id: id,
                disposition,
                observed_cycle: settled_cycle,
                attribution: Some(attribution),
                detail: String::new(),
            }),
            exploration,
            ..DecisionEnvelope::default()
        },
    }
}

fn adapter_with_live_path(cycle: u64) -> MicroexperimentEndpointAdapter {
    let mut adapter = MicroexperimentEndpointAdapter::new(origin(), 7);
    // A delivered episode, not an empty poll: liveness now means the ledger
    // handed something over.
    adapter.observe_episodes(&[live_probe_episode(cycle)], cycle);
    adapter
}

/// A minimal settled episode used only to prove the observation route carries
/// traffic. Its action key is deliberately outside the experiment catalog, so
/// it can never bind to an arm or contribute evidence.
fn live_probe_episode(cycle: u64) -> ResolvedDecisionEpisode {
    episode(
        900_000 + cycle,
        "boost:unrelated-liveness-probe",
        DecisionLifecycle::Applied,
        cycle,
        None,
    )
}

fn open_pair(lab: &mut MicroexperimentLab, candidate: PairCandidate) {
    lab.force_phase_for_test(LabPhase::Active);
    let disposition = lab
        .consider_candidate(candidate, PairGates::healthy_enabled(), 1_000)
        .expect("candidate admitted");
    assert!(matches!(disposition, CandidateDisposition::Opened(_)));
}

fn issue_one(lab: &mut MicroexperimentLab, cycle: u64) -> PairDirective {
    let mut directives = lab.issue_ready_arms(cycle, PairGates::healthy_enabled());
    assert_eq!(directives.len(), 1, "exactly one arm is issued at a time");
    directives.remove(0)
}

/// Pair order is assigned by the lab's own sequence, so a treatment-first pair
/// is obtained by opening one pair ahead of it and taking the second directive.
fn issue_treatment_first(lab: &mut MicroexperimentLab, cycle: u64) -> PairDirective {
    open_pair(lab, qos_candidate(1));
    open_pair(lab, qos_candidate(2));
    let directives = lab.issue_ready_arms(cycle, PairGates::healthy_enabled());
    directives
        .into_iter()
        .find(|directive| directive.arm == ArmKind::Treatment)
        .expect("the second pair runs treatment first")
}

/// Exploration metadata production attaches for the given arm role.
fn metadata_for(arm: ArmKind) -> ExplorationMetadata {
    match arm {
        ArmKind::Control => control_metadata(),
        ArmKind::Treatment => treatment_metadata(),
    }
}

/// Terminal lifecycle a clean arm of this role resolves with.
fn clean_lifecycle_for(arm: ArmKind) -> DecisionLifecycle {
    match arm {
        ArmKind::Control => DecisionLifecycle::NoOp,
        ArmKind::Treatment => DecisionLifecycle::Reverted,
    }
}

/// Drive one arm from directive to endpoint through the real adapter.
fn observe_arm(
    adapter: &mut MicroexperimentEndpointAdapter,
    directive: &PairDirective,
    decision_id: u64,
    lifecycle: DecisionLifecycle,
    exploration: Option<ExplorationMetadata>,
    utility_micros: i64,
) -> Vec<TimedPairEndpoint> {
    let settle_cycle = directive.issued_cycle + 1;
    adapter.register_directives(std::slice::from_ref(directive), directive.issued_cycle);
    adapter.observe_episodes(
        &[episode(
            decision_id,
            &directive.action_key,
            lifecycle,
            settle_cycle,
            exploration,
        )],
        settle_cycle,
    );
    let resolved_cycle = directive.complete_not_before_cycle;
    adapter.observe_utilities(
        &[EndpointUtilitySample {
            decision_id,
            utility_micros,
            resolved_cycle,
            confounded: false,
        }],
        resolved_cycle,
    );
    adapter.drain_ready()
}

// ── Action key identity ──────────────────────────────────────────────────────

#[test]
fn production_markov_key_is_the_canonical_candidate_key() {
    // Production writes `markov_prewarm:predicted_app` (daemon_markov_tick.rs
    // `markov_event`). The lab must generate the same identity.
    let key = canonical_action_key(
        ActuatorFamily::MarkovPrewarm,
        ActionClass::MarkovPredictedApp,
        ExplorationArm::MarkovCacheOnly,
    )
    .expect("catalogued");
    assert_eq!(key, "markov_prewarm:predicted_app");
    assert!(parse_action_key(key)
        .expect("parses")
        .matches(parse_action_key("markov_prewarm:predicted_app").unwrap()));
}

#[test]
fn legacy_markov_key_is_deliberately_retired_not_silently_migrated() {
    assert_eq!(
        parse_action_key(LEGACY_MARKOV_KEY),
        Err(ActionKeyError::LegacyRetired)
    );
}

#[test]
fn interaction_variants_are_distinct_identities() {
    let short = parse_action_key("interaction_qos:foreground@short").unwrap();
    let standard = parse_action_key("interaction_qos:foreground@standard").unwrap();
    let long = parse_action_key("interaction_qos:foreground@long").unwrap();
    assert_eq!(short.variant, ActionVariant::Short);
    assert_eq!(standard.variant, ActionVariant::Standard);
    assert_eq!(long.variant, ActionVariant::Long);
    assert!(!short.matches(standard));
    assert!(!standard.matches(long));
}

#[test]
fn a_similar_key_from_another_family_never_matches() {
    let qos = parse_action_key("interaction_qos:foreground@standard").unwrap();
    let markov = parse_action_key("markov_prewarm:predicted_app").unwrap();
    assert!(!qos.matches(markov));
    assert_eq!(
        parse_action_key("boost:Editor"),
        Err(ActionKeyError::UnknownFamily)
    );
}

#[test]
fn no_prefix_fallback_lets_the_parent_key_close_a_variant_experiment() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);

    adapter.register_directives(std::slice::from_ref(&directive), 10);
    // Production's non-exploratory key. It is a real key, but a different
    // identity, and must not be accepted for an `@standard` experiment.
    adapter.observe_episodes(
        &[episode(
            77,
            "interaction_qos:foreground",
            DecisionLifecycle::Reverted,
            11,
            Some(treatment_metadata()),
        )],
        11,
    );
    assert_eq!(adapter.counters().bound_decisions, 0);
    assert_eq!(adapter.counters().routine_unclaimed, 1);
}

#[test]
fn horizons_agree_with_the_measuring_module() {
    assert_eq!(
        family_horizon_cycles(ActuatorFamily::InteractionQos),
        Some(30)
    );
    assert_eq!(
        family_horizon_cycles(ActuatorFamily::MarkovPrewarm),
        Some(120)
    );
}

// ── Endpoint contract ────────────────────────────────────────────────────────

#[test]
fn contract_is_not_ready_until_the_observation_path_proves_itself_live() {
    let adapter = MicroexperimentEndpointAdapter::new(origin(), 7);
    assert!(!adapter.contract_ready(1, 7), "no ledger batch seen yet");

    let mut adapter = adapter_with_live_path(10);
    assert!(adapter.contract_ready(10, 7));
    assert!(adapter.contract_ready(10 + OBSERVATION_LIVENESS_CYCLES, 7));
    assert!(
        !adapter.contract_ready(11 + OBSERVATION_LIVENESS_CYCLES, 7),
        "a stalled ledger must retract the contract"
    );
    assert!(
        !adapter.contract_ready(10, 8),
        "another daemon generation is never ready"
    );
    // An empty batch is the ledger being polled, not the ledger delivering.
    // It used to refresh liveness, which let `contract_ready` — and through it
    // the lab's whole admission path — report a healthy observation route on a
    // daemon that had never ingested a single episode.
    adapter.observe_episodes(&[], 40);
    assert!(
        !adapter.contract_ready(40, 7),
        "an empty batch must not resurrect a stalled observation path"
    );

    // A batch that actually carries an episode does refresh it.
    adapter.observe_episodes(&[live_probe_episode(41)], 41);
    assert!(
        adapter.contract_ready(41, 7),
        "a delivered episode is what liveness means"
    );
}

#[test]
fn an_unknown_origin_never_reports_a_ready_contract() {
    let mut adapter = MicroexperimentEndpointAdapter::new(ExplorationOrigin::default(), 7);
    adapter.observe_episodes(&[], 3);
    assert!(!adapter.contract_ready(3, 7));
}

#[test]
fn a_ready_contract_without_observations_never_forges_a_closure() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);

    for cycle in 11..=14 {
        adapter.observe_episodes(&[], cycle);
        assert!(adapter.drain_ready().is_empty());
    }
    assert_eq!(lab.metrics().pair_gold_total, 0);
    assert!(lab.drain_pair_gold().is_empty());
}

#[test]
fn an_episode_without_a_real_decision_id_is_rejected() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);

    adapter.observe_episodes(
        &[episode(
            0,
            &directive.action_key,
            DecisionLifecycle::Reverted,
            11,
            Some(treatment_metadata()),
        )],
        11,
    );
    assert_eq!(adapter.counters().bound_decisions, 0);
    assert_eq!(adapter.counters().rejected_incomplete_metadata, 1);
}

#[test]
fn imported_cancelled_and_uncommitted_observations_are_rejected_by_authority() {
    let cases: [(
        &str,
        Box<dyn Fn(ResolvedDecisionEpisode) -> ResolvedDecisionEpisode>,
    ); 4] = [
        (
            "imported attribution",
            Box::new(|mut episode: ResolvedDecisionEpisode| {
                episode.envelope.terminal_attribution = Some(ReceiptAttribution::Imported);
                if let Some(receipt) = episode.envelope.receipt.as_mut() {
                    receipt.attribution = Some(ReceiptAttribution::Imported);
                }
                episode
            }),
        ),
        (
            "cancelled probe",
            Box::new(|mut episode: ResolvedDecisionEpisode| {
                if let Some(metadata) = episode.envelope.exploration.as_mut() {
                    metadata.cancelled = Some(TerminalDiagnostic::ReleaseFailed);
                }
                episode
            }),
        ),
        (
            "uncommitted treatment",
            Box::new(|mut episode: ResolvedDecisionEpisode| {
                if let Some(metadata) = episode.envelope.exploration.as_mut() {
                    metadata.committed = false;
                }
                episode
            }),
        ),
        (
            "lease still held",
            Box::new(|mut episode: ResolvedDecisionEpisode| {
                episode.envelope.lifecycle = DecisionLifecycle::Applied;
                episode
            }),
        ),
    ];

    for (label, mutate) in cases {
        let mut adapter = adapter_with_live_path(1);
        let mut lab = MicroexperimentLab::cold_start(origin());
        open_pair(&mut lab, qos_candidate(1));
        let directive = issue_one(&mut lab, 10);
        adapter.register_directives(std::slice::from_ref(&directive), 10);
        let base = episode(
            5,
            &directive.action_key,
            DecisionLifecycle::Reverted,
            11,
            Some(treatment_metadata()),
        );
        adapter.observe_episodes(&[mutate(base)], 11);
        assert_eq!(
            adapter.counters().bound_decisions,
            0,
            "{label} must not bind"
        );
        assert_eq!(adapter.counters().rejected_authority, 1, "{label}");
    }
}

#[test]
fn a_natural_observation_is_never_admitted_as_a_control() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let directive = issue_treatment_first(&mut lab, 10);

    let mut natural = control_metadata();
    natural.key.mode = ExplorationMode::Natural;
    natural.key.arm = ExplorationArm::NaturalObservation;
    natural.key.action_class = ActionClass::Natural;
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(
        &[episode(
            9,
            &directive.action_key,
            DecisionLifecycle::NoOp,
            11,
            Some(natural),
        )],
        11,
    );
    assert_eq!(adapter.counters().bound_decisions, 0);
    assert_eq!(adapter.counters().rejected_authority, 1);
}

#[test]
fn a_duplicate_observation_binds_exactly_once() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    open_pair(&mut lab, qos_candidate(2));
    let directives = lab.issue_ready_arms(10, PairGates::healthy_enabled());
    assert_eq!(directives.len(), 2);
    let directive = directives[0].clone();
    // A second pair keeps an arm outstanding after the first binds, so the
    // repeat observation is still examined instead of skipped as idle.
    adapter.register_directives(&directives, 10);

    let observation = episode(
        31,
        &directive.action_key,
        clean_lifecycle_for(directive.arm),
        11,
        Some(metadata_for(directive.arm)),
    );
    adapter.observe_episodes(std::slice::from_ref(&observation), 11);
    adapter.observe_episodes(std::slice::from_ref(&observation), 12);
    assert_eq!(adapter.counters().bound_decisions, 1);
    assert_eq!(adapter.counters().rejected_duplicate, 1);

    adapter.observe_utilities(
        &[EndpointUtilitySample {
            decision_id: 31,
            utility_micros: 1_000,
            resolved_cycle: directive.complete_not_before_cycle,
            confounded: false,
        }],
        directive.complete_not_before_cycle,
    );
    assert_eq!(adapter.drain_ready().len(), 1);
}

#[test]
fn an_expired_observation_never_contaminates_the_pair() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);

    let late = directive.expires_after_cycle + 1;
    adapter.observe_episodes(
        &[episode(
            41,
            &directive.action_key,
            clean_lifecycle_for(directive.arm),
            late,
            Some(metadata_for(directive.arm)),
        )],
        late,
    );
    assert_eq!(adapter.counters().bound_decisions, 0);
    assert_eq!(adapter.counters().rejected_expired, 1);
    assert!(adapter.drain_ready().is_empty());
}

#[test]
fn an_observation_predating_its_directive_is_rejected() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);

    adapter.observe_episodes(
        &[episode(
            42,
            &directive.action_key,
            clean_lifecycle_for(directive.arm),
            directive.issued_cycle - 1,
            Some(metadata_for(directive.arm)),
        )],
        11,
    );
    assert_eq!(adapter.counters().bound_decisions, 0);
    assert_eq!(adapter.counters().rejected_expired, 1);
}

#[test]
fn a_restart_never_reuses_old_observations_as_new_evidence() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(
        &[episode(
            51,
            &directive.action_key,
            clean_lifecycle_for(directive.arm),
            11,
            Some(metadata_for(directive.arm)),
        )],
        11,
    );
    assert_eq!(adapter.counters().bound_decisions, 1);

    // A restart is a new generation: nothing carries over, and the previous
    // generation's directives are not registered again.
    let mut restarted = MicroexperimentEndpointAdapter::new(origin(), 8);
    restarted.observe_episodes(
        &[episode(
            51,
            &directive.action_key,
            clean_lifecycle_for(directive.arm),
            11,
            Some(metadata_for(directive.arm)),
        )],
        11,
    );
    assert_eq!(restarted.counters().bound_decisions, 0);
    // A fresh generation has no outstanding arm, so the batch is skipped
    // wholesale — the old observation is not evidence for anything.
    assert_eq!(restarted.counters().episodes_skipped_idle, 1);
    assert_eq!(restarted.counters().routine_unclaimed, 0);
    assert!(restarted.drain_ready().is_empty());
    assert!(!restarted.contract_ready(11, 7));
}

#[test]
fn queues_stay_bounded_under_a_flood_of_directives_and_episodes() {
    use apollo_engine::engine::microexperiment_endpoints::MAX_OUTSTANDING_ARMS;

    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    lab.force_phase_for_test(LabPhase::Active);
    let mut directives = Vec::new();
    for sequence in 1..=(MAX_OUTSTANDING_ARMS as u64 + 8) {
        if lab
            .consider_candidate(
                qos_candidate(sequence),
                PairGates::healthy_enabled(),
                1_000 + sequence,
            )
            .is_ok()
        {
            directives.extend(lab.issue_ready_arms(10, PairGates::healthy_enabled()));
        }
    }
    adapter.register_directives(&directives, 10);
    assert!(adapter.outstanding_len() <= MAX_OUTSTANDING_ARMS);
}

// ── Pair lifecycle ───────────────────────────────────────────────────────────

/// The acceptance path: open -> real endpoints -> closed -> exactly one Gold.
#[test]
fn a_valid_pair_closes_into_exactly_one_pair_gold_record() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let candidate = qos_candidate(1);
    open_pair(&mut lab, candidate.clone());

    // Sequence 1 assigns control first.
    let first = issue_one(&mut lab, 10);
    assert_eq!(first.arm, ArmKind::Control);
    let endpoints = observe_arm(
        &mut adapter,
        &first,
        101,
        DecisionLifecycle::NoOp,
        Some(control_metadata()),
        1_000,
    );
    assert_eq!(endpoints.len(), 1);
    assert_eq!(endpoints[0].endpoint.decision_id, 101);
    assert_eq!(endpoints[0].endpoint.horizon, HorizonClosure::Complete);
    assert!(matches!(
        lab.record_timed_endpoint(endpoints[0].clone()),
        Ok(TimedEndpointDisposition::Progress(_))
    ));

    // Washout is real elapsed cycles, not a free pass.
    let mut cycle = first.complete_not_before_cycle;
    for _ in 0..=WASHOUT {
        cycle += 1;
        lab.advance_cycle(cycle, 2_000 + cycle, PairGates::healthy_enabled());
    }

    let second = issue_one(&mut lab, cycle);
    assert_eq!(second.arm, ArmKind::Treatment);
    assert!(second.rollback_required);
    let endpoints = observe_arm(
        &mut adapter,
        &second,
        202,
        DecisionLifecycle::Reverted,
        Some(treatment_metadata()),
        4_000,
    );
    assert_eq!(endpoints.len(), 1);
    assert_eq!(endpoints[0].endpoint.decision_id, 202);
    assert_eq!(endpoints[0].endpoint.rollback, RollbackClosure::Succeeded);
    assert!(matches!(
        lab.record_timed_endpoint(endpoints[0].clone()),
        Ok(TimedEndpointDisposition::Progress(
            apollo_engine::engine::microexperiment_lab::PairProgress::ReadyToClose
        ))
    ));

    let closure = lab.close_pair(first.pair_id).expect("pair closes");
    assert_eq!(
        closure.evidence,
        apollo_engine::engine::microexperiment_lab::EvidenceClosure::PairGold
    );
    assert_eq!(closure.effect_micros, 3_000);
    assert!(closure.effective);

    let gold: Vec<PairGoldRecord> = lab.drain_pair_gold();
    assert_eq!(gold.len(), 1);
    assert_eq!(gold[0].control_decision_id, 101);
    assert_eq!(gold[0].treatment_decision_id, 202);
    assert_eq!(gold[0].action_key, candidate.action_key);
    assert!(lab.drain_pair_gold().is_empty(), "gold drains exactly once");
    assert_eq!(lab.metrics().pair_gold_total, 1);
}

#[test]
fn a_failed_rollback_is_reflected_and_never_becomes_gold() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let directive = issue_treatment_first(&mut lab, 10);
    assert_eq!(directive.arm, ArmKind::Treatment);

    let endpoints = observe_arm(
        &mut adapter,
        &directive,
        303,
        DecisionLifecycle::Failed,
        Some(treatment_metadata()),
        4_000,
    );
    assert_eq!(endpoints.len(), 1);
    assert_eq!(endpoints[0].endpoint.rollback, RollbackClosure::Failed);
    assert_eq!(adapter.counters().rollback_failed, 1);

    let disposition = lab
        .record_timed_endpoint(endpoints[0].clone())
        .expect("endpoint is handled");
    let TimedEndpointDisposition::Invalidated(record) = disposition else {
        panic!("a failed rollback must invalidate the pair");
    };
    assert_eq!(
        record.reason,
        apollo_engine::engine::microexperiment_lab::PairInvalidationReason::FailedRollback
    );
    assert_eq!(lab.metrics().pair_gold_total, 0);
    assert!(lab.drain_pair_gold().is_empty());
}

#[test]
fn a_pending_utility_leaves_the_arm_open_and_produces_no_gold() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(
        &[episode(
            401,
            &directive.action_key,
            DecisionLifecycle::NoOp,
            11,
            Some(control_metadata()),
        )],
        11,
    );

    assert_eq!(adapter.counters().bound_decisions, 1);
    assert_eq!(adapter.counters().pending_utility, 1);
    assert!(adapter.drain_ready().is_empty());
    assert_eq!(lab.metrics().pair_gold_total, 0);
}

#[test]
fn an_arm_whose_horizon_lapsed_expires_instead_of_closing() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    assert_eq!(adapter.outstanding_len(), 1);

    adapter.prune(directive.expires_after_cycle + 1);
    assert_eq!(adapter.outstanding_len(), 0);
    assert_eq!(adapter.counters().expired_arms, 1);

    let invalidations = lab.advance_cycle(
        directive.expires_after_cycle + 1,
        9_000,
        PairGates::healthy_enabled(),
    );
    assert_eq!(invalidations.len(), 1);
    assert_eq!(
        invalidations[0].reason,
        apollo_engine::engine::microexperiment_lab::PairInvalidationReason::DeadlineExpired
    );
    assert_eq!(lab.metrics().pair_gold_total, 0);
}

#[test]
fn a_confounded_utility_sample_closes_the_endpoint_as_confounded() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(
        &[episode(
            501,
            &directive.action_key,
            DecisionLifecycle::NoOp,
            11,
            Some(control_metadata()),
        )],
        11,
    );
    adapter.observe_utilities(
        &[EndpointUtilitySample {
            decision_id: 501,
            utility_micros: 10,
            resolved_cycle: directive.complete_not_before_cycle,
            confounded: true,
        }],
        directive.complete_not_before_cycle,
    );
    let endpoints = adapter.drain_ready();
    assert_eq!(endpoints[0].endpoint.horizon, HorizonClosure::Confounded);

    let disposition = lab
        .record_timed_endpoint(endpoints[0].clone())
        .expect("handled");
    assert!(matches!(
        disposition,
        TimedEndpointDisposition::Invalidated(_)
    ));
    assert_eq!(lab.metrics().pair_gold_total, 0);
}

#[test]
fn forgetting_a_pair_drops_every_arm_binding_and_endpoint_for_it() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(
        &[episode(
            601,
            &directive.action_key,
            DecisionLifecycle::NoOp,
            11,
            Some(control_metadata()),
        )],
        11,
    );
    assert_eq!(adapter.bound_len(), 1);

    adapter.forget_pair(directive.pair_id);
    assert_eq!(adapter.bound_len(), 0);
    assert_eq!(adapter.outstanding_len(), 0);

    adapter.observe_utilities(
        &[EndpointUtilitySample {
            decision_id: 601,
            utility_micros: 5_000,
            resolved_cycle: directive.complete_not_before_cycle,
            confounded: false,
        }],
        directive.complete_not_before_cycle,
    );
    assert!(adapter.drain_ready().is_empty());
}

// ── Control arm handoff ──────────────────────────────────────────────────────

/// The withhold mask is a process-global static, so all of its behaviour is
/// asserted in one test to keep the parallel test runner deterministic.
#[test]
fn the_control_withhold_mask_is_advisory_bounded_and_catalogue_only() {
    use apollo_engine::engine::microexperiment_endpoints::{
        control_withhold_requested, publish_control_withholds,
    };

    let standard = parse_action_key("interaction_qos:foreground@standard").unwrap();
    let long = parse_action_key("interaction_qos:foreground@long").unwrap();
    let markov = parse_action_key("markov_prewarm:predicted_app").unwrap();
    let bare = parse_action_key("interaction_qos:foreground").unwrap();

    // Default is "never withhold".
    publish_control_withholds(&[]);
    for action in [standard, long, markov, bare] {
        assert!(!control_withhold_requested(action));
    }

    publish_control_withholds(&[standard]);
    assert!(control_withhold_requested(standard));
    assert!(
        !control_withhold_requested(long),
        "variants are independent"
    );
    assert!(
        !control_withhold_requested(markov),
        "families are independent"
    );
    assert!(
        !control_withhold_requested(bare),
        "the parent key has no withhold bit"
    );

    // Republishing replaces rather than accumulates: a closed pair stops
    // requesting a withhold on the next cycle.
    publish_control_withholds(&[markov]);
    assert!(control_withhold_requested(markov));
    assert!(!control_withhold_requested(standard));

    publish_control_withholds(&[]);
    assert!(!control_withhold_requested(markov));
}

#[test]
fn only_open_control_arms_are_published_as_withhold_requests() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let control = issue_one(&mut lab, 10);
    assert_eq!(control.arm, ArmKind::Control);

    assert!(
        adapter.outstanding_control_actions().is_empty(),
        "nothing is requested before the directive is registered"
    );
    adapter.register_directives(std::slice::from_ref(&control), 10);
    assert_eq!(
        adapter.outstanding_control_actions(),
        vec![parse_action_key(&control.action_key).unwrap()]
    );

    // Once the control arm is observed, the request disappears.
    adapter.observe_episodes(
        &[episode(
            701,
            &control.action_key,
            DecisionLifecycle::NoOp,
            11,
            Some(control_metadata()),
        )],
        11,
    );
    assert!(adapter.outstanding_control_actions().is_empty());
}

#[test]
fn a_treatment_arm_never_asks_an_owner_to_withhold() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    let treatment = issue_treatment_first(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&treatment), 10);
    assert!(adapter
        .outstanding_control_actions()
        .iter()
        .all(|action| action.family == ActuatorFamily::InteractionQos));
    assert_eq!(
        adapter.outstanding_control_actions().len(),
        0,
        "only the treatment arm was registered"
    );
}

#[test]
fn the_exploration_catalog_admits_a_control_arm_for_both_lab_families() {
    use apollo_engine::engine::exploration_scheduler::ExplorationCandidate;

    for (family, arm, class, context) in [
        (
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosShort,
            ActionClass::InteractionForeground,
            ExplorationContext::Interactive,
        ),
        (
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosStandard,
            ActionClass::InteractionForeground,
            ExplorationContext::Interactive,
        ),
        (
            ActuatorFamily::InteractionQos,
            ExplorationArm::InteractionQosLong,
            ActionClass::InteractionForeground,
            ExplorationContext::Interactive,
        ),
        (
            ActuatorFamily::MarkovPrewarm,
            ExplorationArm::MarkovCacheOnly,
            ActionClass::MarkovPredictedApp,
            ExplorationContext::Background,
        ),
    ] {
        assert!(
            ExplorationCandidate::new(
                family,
                ExplorationMode::Control,
                arm,
                class,
                context,
                origin()
            )
            .is_ok(),
            "{family:?}/{arm:?} must offer a control arm"
        );
    }

    // Widening the catalog must not admit a mismatched arm or class.
    assert!(ExplorationCandidate::new(
        ActuatorFamily::InteractionQos,
        ExplorationMode::Control,
        ExplorationArm::MarkovCacheOnly,
        ActionClass::InteractionForeground,
        ExplorationContext::Interactive,
        origin()
    )
    .is_err());
    assert!(ExplorationCandidate::new(
        ActuatorFamily::MarkovPrewarm,
        ExplorationMode::Control,
        ExplorationArm::MarkovCacheOnly,
        ActionClass::InteractionForeground,
        ExplorationContext::Background,
        origin()
    )
    .is_err());
}

// ── Rejection counters stay diagnostic ───────────────────────────────────────

/// Regression guard for a defect production surfaced: with no arm open, every
/// episode of the cycle was classified as a rejection (374 key + 266 authority
/// = all 640 observed), burying the one signal those counters exist for.
#[test]
fn routine_traffic_never_inflates_the_rejection_counters() {
    let mut adapter = adapter_with_live_path(1);

    // A realistic cycle batch: mostly uncatalogued families, plus real
    // interaction_qos and markov decisions that no arm is waiting on.
    let batch = vec![
        episode(1, "boost:Editor", DecisionLifecycle::Applied, 2, None),
        episode(2, "throttle:Helper", DecisionLifecycle::Applied, 2, None),
        episode(3, "freeze:Worker", DecisionLifecycle::NoOp, 2, None),
        episode(4, "coordinated:batch", DecisionLifecycle::Applied, 2, None),
        episode(
            5,
            "interaction_qos:foreground@standard",
            DecisionLifecycle::Reverted,
            2,
            Some(treatment_metadata()),
        ),
        episode(
            6,
            "markov_prewarm:predicted_app",
            DecisionLifecycle::NoOp,
            2,
            None,
        ),
    ];

    // No arm outstanding: the whole batch is skipped, not classified.
    adapter.observe_episodes(&batch, 2);
    let idle = adapter.counters();
    // 6 from this batch plus the one `adapter_with_live_path` delivers to prove
    // the route carries traffic — it lands with no arm outstanding, so it is
    // skipped exactly like the rest.
    assert_eq!(idle.episodes_skipped_idle, 7);
    assert_eq!(idle.observed_episodes, 0);
    assert_eq!(idle.rejected_action_mismatch, 0);
    assert_eq!(idle.rejected_authority, 0);
    assert_eq!(idle.routine_unclaimed, 0);

    // With an arm open, uncatalogued families are counted as routine traffic
    // and still never touch a rejection counter.
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);
    adapter.observe_episodes(&batch, 11);

    let busy = adapter.counters();
    assert_eq!(busy.observed_episodes, 6);
    assert_eq!(
        busy.uncatalogued_episodes, 4,
        "boost/throttle/freeze/coordinated"
    );
    assert_eq!(
        busy.rejected_action_mismatch, 0,
        "an uncatalogued family is not a key mismatch"
    );
    // The open arm is Control; the two catalogued episodes are a treatment and
    // an unattributed markov no-op, so neither can bind.
    assert_eq!(busy.bound_decisions, 0);
    // The treatment episode arrives for an action an experiment IS watching,
    // but cannot fill the Control role that is open — a genuine experimental
    // anomaly, not passing traffic. Under the old single `unknown_arm` bucket
    // it was indistinguishable from the daemon's routine work.
    // The two catalogued episodes now separate instead of sharing one bucket:
    //  - the interaction_qos treatment names an action an experiment IS
    //    watching but cannot fill the open Control role: a real anomaly;
    //  - the markov no-op belongs to a family no arm is waiting on at all:
    //    ordinary traffic.
    // Under the old single `unknown_arm` counter both read as failures, and
    // production showed 7,809 of them.
    assert_eq!(busy.invalid_experimental, 1, "treatment for a control arm");
    assert_eq!(busy.routine_unclaimed, 1, "another family, nobody waiting");
    assert_eq!(busy.rejected_authority, 0);
}

/// A genuine key mismatch must remain visible rather than being absorbed as
/// routine traffic — this is the counter that would have caught the original
/// `markov:cache-only` vs `markov_prewarm:predicted_app` break.
#[test]
fn a_retired_key_on_an_open_arm_is_reported_not_absorbed() {
    let mut adapter = adapter_with_live_path(1);
    let mut lab = MicroexperimentLab::cold_start(origin());
    open_pair(&mut lab, qos_candidate(1));
    let directive = issue_one(&mut lab, 10);
    adapter.register_directives(std::slice::from_ref(&directive), 10);

    adapter.observe_episodes(
        &[episode(
            9,
            LEGACY_MARKOV_KEY,
            DecisionLifecycle::NoOp,
            11,
            Some(control_metadata()),
        )],
        11,
    );
    let counters = adapter.counters();
    assert_eq!(counters.bound_decisions, 0);
    assert_eq!(
        counters.uncatalogued_episodes, 1,
        "the retired key is refused and surfaces separately from real traffic"
    );
    assert_eq!(counters.rejected_authority, 0);
}
