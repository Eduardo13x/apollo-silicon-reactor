//! apollo-optimizer — facade crate for binaries and integration tests.
//!
//! Sprint 5 Mes 0: this lib is a thin facade that re-exports
//! `apollo-engine` so legacy paths `apollo_optimizer::engine::...`
//! in integration tests continue to resolve.
//!
//! Discipline:
//! - This file MUST stay this size (one re-export, no other modules).
//! - New logic goes in `apollo-engine` (lib) or `src/bin/<name>/` (bin).
//! - If you're tempted to add `pub mod foo` here, you're rebuilding the
//!   monolith. Don't.
//!
//! See docs/superpowers/specs/2026-05-09-workspace-split-design.md.

pub use apollo_engine::*;
