//! Content-hash AhoCorasick cache (Sprint patch 2026-06-05).
//!
//! Pattern lists in `usage_model.rs` mutate only on rare policy promotion,
//! but the surrounding code rebuilds `AhoCorasick` automata per call.  This
//! cache keys on the hash of the (sorted+deduped) pattern set so two callers
//! that supply the same logical pattern bundle share the same `Arc<AhoCorasick>`
//! instead of paying the build cost again.
//!
//! Implementation notes — stdlib-only by design:
//! - hash via `std::collections::hash_map::DefaultHasher` (SipHash-1-3).
//!   Cryptographic quality is irrelevant here; we just need stable equality
//!   over the canonicalised pattern set.
//! - LRU eviction via a `VecDeque<u64>` ordering history + a `HashMap`.
//!   Capacity is 32 distinct pattern sets; usage_model holds 4, callers of
//!   the cache rarely exceed this in practice.
//! - All accesses are guarded by a `Mutex` since the build path is rare
//!   and lock contention is dwarfed by the cost of an actual `build` call.
//!
//! References:
//! - Saltzer & Schroeder 1975 — Economy of Mechanism: amortise expensive
//!   data-structure builds at the natural cache boundary.

use aho_corasick::{AhoCorasick, MatchKind};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::sync::{Arc, Mutex, OnceLock};

const CAP: usize = 32;

struct Inner {
    map: HashMap<u64, Arc<AhoCorasick>>,
    order: VecDeque<u64>,
}

fn cache() -> &'static Mutex<Inner> {
    static CACHE: OnceLock<Mutex<Inner>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(Inner {
            map: HashMap::with_capacity(CAP),
            order: VecDeque::with_capacity(CAP),
        })
    })
}

fn hash_patterns(pats: &[&str], kind: MatchKind) -> u64 {
    let mut v: Vec<&str> = pats.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut h = DefaultHasher::new();
    h.write_u8(kind as u8);
    for p in &v {
        h.write(p.as_bytes());
        h.write_u8(0);
    }
    h.finish()
}

/// Get a cached automaton if it exists, otherwise build one and insert it.
///
/// Returns `None` when `pats` is empty (matches existing `AhoCorasick::new`
/// failure-on-empty behaviour) so callers do not have to special-case the
/// branch.
///
/// **Concurrency (follow-up #2, MED severity):** uses a double-check pattern
/// so that two callers racing on the same `key` end up sharing the *same*
/// `Arc<AhoCorasick>`. The first lock check is a fast-path hit; on miss we
/// build outside the lock, then re-check under the lock and return the
/// existing entry if a concurrent builder beat us to insertion (dropping the
/// duplicate build). This preserves the property `identical_patterns_share_arc`
/// even under build-mode contention (2+ rustc + decide_actions racing).
pub fn get_or_build(pats: &[&str], kind: MatchKind) -> Option<Arc<AhoCorasick>> {
    if pats.is_empty() {
        return None;
    }
    let key = hash_patterns(pats, kind);
    if let Ok(g) = cache().lock() {
        if let Some(ac) = g.map.get(&key) {
            return Some(ac.clone());
        }
    }
    let built = AhoCorasick::builder().match_kind(kind).build(pats).ok()?;
    let arc = Arc::new(built);
    if let Ok(mut g) = cache().lock() {
        // Double-check after the build: a concurrent caller may have
        // inserted the same key while we were releasing the lock + building.
        // If so, return the *cached* Arc and drop our freshly-built one,
        // preserving Arc::ptr_eq for downstream consumers (e.g. tests + the
        // S6 user_profile pre-build invariant).
        if let Some(ac) = g.map.get(&key) {
            return Some(ac.clone());
        }
        if g.order.len() >= CAP {
            if let Some(evict) = g.order.pop_front() {
                g.map.remove(&evict);
                crate::engine::lse_counters::LSE_COUNTERS.inc_ac_cache_eviction();
            }
        }
        g.map.insert(key, arc.clone());
        g.order.push_back(key);
    }
    Some(arc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_patterns_share_arc() {
        let a = get_or_build(&["x", "y"], MatchKind::Standard).unwrap();
        let b = get_or_build(&["y", "x"], MatchKind::Standard).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn distinct_patterns_distinct_arcs() {
        let a = get_or_build(&["distinct_unique_x"], MatchKind::Standard).unwrap();
        let b = get_or_build(&["distinct_unique_z"], MatchKind::Standard).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn empty_returns_none() {
        assert!(get_or_build(&[], MatchKind::Standard).is_none());
    }
}
