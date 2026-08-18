//! Canonical action identity shared by microexperiment generation, actuation
//! and observation.
//!
//! The Microexperiment Lab, the production actuators and `DecisionLedger` all
//! name actions with the same `family:class[@variant]` string. Before this
//! module each side spelled that identity by hand, so a lab candidate keyed
//! `markov:cache-only` could never join the `markov_prewarm:predicted_app`
//! decisions its own pipeline actually emits.
//!
//! Matching is structural (family + action class + variant), never a prefix
//! test: `interaction_qos:foreground` is the non-exploratory production key and
//! must not silently satisfy an `interaction_qos:foreground@standard`
//! experiment. Everything outside the closed catalog is an explicit error, so a
//! new or renamed actuator key fails loudly instead of pairing by accident.

use crate::engine::exploration_scheduler::{ActionClass, ExplorationArm};
use crate::engine::telemetry_medallion::ActuatorFamily;

/// Retired lab key. It never existed at any actuator boundary, so it is
/// rejected rather than aliased onto the canonical Markov identity.
pub const LEGACY_MARKOV_KEY: &str = "markov:cache-only";

/// Bounded variant suffix. `None` is a distinct identity, not a wildcard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ActionVariant {
    None,
    Short,
    Standard,
    Long,
}

impl ActionVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Short => "short",
            Self::Standard => "standard",
            Self::Long => "long",
        }
    }
}

/// Structured identity of one catalogued action key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CanonicalAction {
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub variant: ActionVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKeyError {
    /// Empty, oversized, or not `family:class` shaped.
    Malformed,
    /// Family segment is not part of the experiment catalog.
    UnknownFamily,
    /// Class segment is not part of the experiment catalog for that family.
    UnknownClass,
    /// Variant suffix is present but not a catalogued band.
    UnknownVariant,
    /// Key was emitted by an older Apollo revision and carries no authority.
    LegacyRetired,
}

/// Longest key the catalog can emit; keeps parsing bounded.
const MAX_KEY_BYTES: usize = 96;

/// Canonical key for one catalogued `(family, class, arm)` triple.
///
/// Returns `None` for combinations the lab may reason about but that carry no
/// stable production key. `Boost`/`BoostOmission` is deliberately absent:
/// production boost decisions are keyed per process (`boost:Editor`), so they
/// have no catalog-wide identity an experiment could address.
pub fn canonical_action_key(
    family: ActuatorFamily,
    action_class: ActionClass,
    arm: ExplorationArm,
) -> Option<&'static str> {
    match (family, action_class, arm) {
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosShort,
        ) => Some("interaction_qos:foreground@short"),
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosStandard,
        ) => Some("interaction_qos:foreground@standard"),
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosLong,
        ) => Some("interaction_qos:foreground@long"),
        (
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            ExplorationArm::MarkovCacheOnly,
        ) => Some("markov_prewarm:predicted_app"),
        _ => None,
    }
}

/// Outcome horizon in daemon cycles for one catalogued family.
///
/// These mirror `telemetry_medallion::decision_episode_spec`, which is the
/// module that actually resolves a decision into measured evidence. A pair
/// whose horizon disagreed with the medallion would expire before its own
/// utility sample existed.
pub fn family_horizon_cycles(family: ActuatorFamily) -> Option<u32> {
    match family {
        ActuatorFamily::InteractionQos => Some(30),
        ActuatorFamily::MarkovPrewarm => Some(120),
        _ => None,
    }
}

/// Single normalization entry point for every action key that crosses the
/// experiment boundary.
pub fn parse_action_key(key: &str) -> Result<CanonicalAction, ActionKeyError> {
    if key == LEGACY_MARKOV_KEY {
        return Err(ActionKeyError::LegacyRetired);
    }
    if key.is_empty() || key.len() > MAX_KEY_BYTES {
        return Err(ActionKeyError::Malformed);
    }
    let Some((family_segment, remainder)) = key.split_once(':') else {
        return Err(ActionKeyError::Malformed);
    };
    if family_segment.is_empty() || remainder.is_empty() {
        return Err(ActionKeyError::Malformed);
    }
    let (class_segment, variant_segment) = match remainder.split_once('@') {
        Some((class, variant)) => (class, Some(variant)),
        None => (remainder, None),
    };
    if class_segment.is_empty() {
        return Err(ActionKeyError::Malformed);
    }
    let family = match family_segment {
        "interaction_qos" => ActuatorFamily::InteractionQos,
        "markov_prewarm" => ActuatorFamily::MarkovPrewarm,
        _ => return Err(ActionKeyError::UnknownFamily),
    };
    let action_class = match (family, class_segment) {
        (ActuatorFamily::InteractionQos, "foreground") => ActionClass::InteractionForeground,
        (ActuatorFamily::MarkovPrewarm, "predicted_app") => ActionClass::MarkovPredictedApp,
        _ => return Err(ActionKeyError::UnknownClass),
    };
    let variant = match variant_segment {
        None => ActionVariant::None,
        Some("short") if family == ActuatorFamily::InteractionQos => ActionVariant::Short,
        Some("standard") if family == ActuatorFamily::InteractionQos => ActionVariant::Standard,
        Some("long") if family == ActuatorFamily::InteractionQos => ActionVariant::Long,
        Some(_) => return Err(ActionKeyError::UnknownVariant),
    };
    Ok(CanonicalAction {
        family,
        action_class,
        variant,
    })
}

impl CanonicalAction {
    /// Exploration arm that produces this exact identity, when one exists.
    pub fn arm(self) -> Option<ExplorationArm> {
        match (self.family, self.action_class, self.variant) {
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ActionVariant::Short,
            ) => Some(ExplorationArm::InteractionQosShort),
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ActionVariant::Standard,
            ) => Some(ExplorationArm::InteractionQosStandard),
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ActionVariant::Long,
            ) => Some(ExplorationArm::InteractionQosLong),
            (
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ActionVariant::None,
            ) => Some(ExplorationArm::MarkovCacheOnly),
            _ => None,
        }
    }

    /// True only when both identities name the same catalogued action. There
    /// is no prefix or parent-key fallback.
    pub fn matches(self, other: Self) -> bool {
        self == other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogued_keys_round_trip_through_the_canonical_identity() {
        for (family, class, arm) in [
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ExplorationArm::InteractionQosShort,
            ),
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ExplorationArm::InteractionQosStandard,
            ),
            (
                ActuatorFamily::InteractionQos,
                ActionClass::InteractionForeground,
                ExplorationArm::InteractionQosLong,
            ),
            (
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
            ),
        ] {
            let key = canonical_action_key(family, class, arm).expect("catalogued key");
            let parsed = parse_action_key(key).expect("parsed key");
            assert_eq!(parsed.family, family);
            assert_eq!(parsed.action_class, class);
            assert_eq!(parsed.arm(), Some(arm));
        }
    }

    #[test]
    fn production_markov_key_is_the_canonical_identity_and_the_legacy_key_is_retired() {
        assert_eq!(
            canonical_action_key(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly
            ),
            Some("markov_prewarm:predicted_app")
        );
        assert_eq!(
            parse_action_key(LEGACY_MARKOV_KEY),
            Err(ActionKeyError::LegacyRetired)
        );
    }

    #[test]
    fn bare_parent_key_never_collides_with_a_variant_key() {
        let bare = parse_action_key("interaction_qos:foreground").expect("bare key parses");
        let standard =
            parse_action_key("interaction_qos:foreground@standard").expect("variant key parses");
        assert_eq!(bare.variant, ActionVariant::None);
        assert_eq!(standard.variant, ActionVariant::Standard);
        assert!(!bare.matches(standard));
        assert!(!standard.matches(bare));
    }

    #[test]
    fn same_family_different_variant_and_different_family_never_match() {
        let short = parse_action_key("interaction_qos:foreground@short").unwrap();
        let long = parse_action_key("interaction_qos:foreground@long").unwrap();
        let markov = parse_action_key("markov_prewarm:predicted_app").unwrap();
        assert!(!short.matches(long));
        assert!(!short.matches(markov));
    }

    #[test]
    fn uncatalogued_and_malformed_keys_are_typed_errors() {
        assert_eq!(
            parse_action_key("boost:Editor"),
            Err(ActionKeyError::UnknownFamily)
        );
        assert_eq!(
            parse_action_key("interaction_qos:background"),
            Err(ActionKeyError::UnknownClass)
        );
        assert_eq!(
            parse_action_key("interaction_qos:foreground@turbo"),
            Err(ActionKeyError::UnknownVariant)
        );
        assert_eq!(
            parse_action_key("markov_prewarm:predicted_app@short"),
            Err(ActionKeyError::UnknownVariant)
        );
        assert_eq!(parse_action_key(""), Err(ActionKeyError::Malformed));
        assert_eq!(
            parse_action_key("interaction_qos"),
            Err(ActionKeyError::Malformed)
        );
        assert_eq!(parse_action_key("a:"), Err(ActionKeyError::Malformed));
        assert_eq!(
            parse_action_key(&"x".repeat(MAX_KEY_BYTES + 1)),
            Err(ActionKeyError::Malformed)
        );
    }

    #[test]
    fn uncatalogued_families_expose_no_horizon_and_no_canonical_key() {
        assert_eq!(family_horizon_cycles(ActuatorFamily::Boost), None);
        assert_eq!(
            canonical_action_key(
                ActuatorFamily::Boost,
                ActionClass::BoostBackground,
                ExplorationArm::BoostOmission
            ),
            None
        );
    }
}
