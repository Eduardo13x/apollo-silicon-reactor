//! Immutable, identity-bound world snapshots for models and optional compute.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::engine::network_flow::NetworkWorldObservation;
use crate::engine::webflow_controller::WebWorldObservation;

pub const WORLD_STATE_SCHEMA_VERSION: u16 = 3;
pub const FEATURE_CAPACITY: usize = 256;
pub const RETAINED_WORLD_REVISIONS: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureStore {
    pub schema_version: u16,
    values: Vec<f32>,
}

impl FeatureStore {
    pub fn try_new(schema_version: u16, values: Vec<f32>) -> Result<Self, &'static str> {
        if schema_version == 0 || values.len() > FEATURE_CAPACITY {
            return Err("invalid feature schema or capacity");
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("feature vector contains non-finite value");
        }
        Ok(Self {
            schema_version,
            values,
        })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldIdentity {
    pub daemon_epoch: u64,
    pub revision: u64,
    pub workload_id: u64,
    pub capability_revision: u64,
    pub thermal_revision: u64,
    pub process_revision: u64,
    pub session_revision: u64,
    pub kill_switch: bool,
    pub sleeping: bool,
}

impl WorldIdentity {
    pub fn accepts_result_for(self, current: Self) -> bool {
        self == current
    }

    pub fn accepts_recent_result_for(self, current: Self, maximum_age: u64) -> bool {
        self.daemon_epoch == current.daemon_epoch
            && self.workload_id == current.workload_id
            && self.capability_revision == current.capability_revision
            && self.thermal_revision == current.thermal_revision
            && self.process_revision == current.process_revision
            && self.session_revision == current.session_revision
            && self.kill_switch == current.kill_switch
            && self.sleeping == current.sleeping
            && current.revision >= self.revision
            && current.revision.saturating_sub(self.revision) <= maximum_age
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldStateSnapshot {
    pub schema_version: u16,
    pub identity: WorldIdentity,
    pub event_watermark: u64,
    pub features: FeatureStore,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web: Option<WebWorldObservation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkWorldObservation>,
}

impl WorldStateSnapshot {
    pub fn new(
        identity: WorldIdentity,
        event_watermark: u64,
        features: FeatureStore,
    ) -> Result<Self, &'static str> {
        if identity.daemon_epoch == 0 || identity.revision == 0 || identity.capability_revision == 0
        {
            return Err("world identity is incomplete");
        }
        Ok(Self {
            schema_version: WORLD_STATE_SCHEMA_VERSION,
            identity,
            event_watermark,
            features,
            web: None,
            network: None,
        })
    }

    pub fn with_web(mut self, web: Option<WebWorldObservation>) -> Self {
        self.web = if self.identity.kill_switch || self.identity.sleeping {
            None
        } else {
            web
        };
        self
    }

    pub fn with_network(mut self, network: Option<NetworkWorldObservation>) -> Self {
        self.network = if self.identity.kill_switch || self.identity.sleeping {
            None
        } else {
            network
        };
        self
    }
}

struct PublishedWorld {
    current: Arc<WorldStateSnapshot>,
    retained: VecDeque<Arc<WorldStateSnapshot>>,
}

pub struct WorldStatePublisher {
    state: RwLock<PublishedWorld>,
}

impl WorldStatePublisher {
    pub fn new(initial: WorldStateSnapshot) -> Self {
        let current = Arc::new(initial);
        let mut retained = VecDeque::with_capacity(RETAINED_WORLD_REVISIONS);
        retained.push_back(Arc::clone(&current));
        Self {
            state: RwLock::new(PublishedWorld { current, retained }),
        }
    }

    pub fn latest(&self) -> Arc<WorldStateSnapshot> {
        Arc::clone(
            &self
                .state
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .current,
        )
    }

    pub fn publish(
        &self,
        mut snapshot: WorldStateSnapshot,
    ) -> Result<Arc<WorldStateSnapshot>, &'static str> {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if snapshot.identity.daemon_epoch != state.current.identity.daemon_epoch {
            return Err("daemon epoch mismatch");
        }
        if snapshot.identity.revision <= state.current.identity.revision {
            return Err("world revision must increase");
        }
        if snapshot.identity.session_revision != state.current.identity.session_revision
            || snapshot.identity.kill_switch
            || snapshot.identity.sleeping
        {
            snapshot.web = None;
            snapshot.network = None;
        }
        let next = Arc::new(snapshot);
        state.current = Arc::clone(&next);
        state.retained.push_back(Arc::clone(&next));
        while state.retained.len() > RETAINED_WORLD_REVISIONS {
            state.retained.pop_front();
        }
        Ok(next)
    }

    pub fn retained_len(&self) -> usize {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retained
            .len()
    }

    pub fn by_revision(&self, revision: u64) -> Option<Arc<WorldStateSnapshot>> {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retained
            .iter()
            .find(|snapshot| snapshot.identity.revision == revision)
            .map(Arc::clone)
    }
}
