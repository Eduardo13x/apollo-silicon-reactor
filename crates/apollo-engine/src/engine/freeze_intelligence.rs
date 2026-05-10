//! Universal freeze/thaw intelligence using NARS beliefs and process classification.
//!
//! App-agnostic: works for any freezeable process, not just Chromium renderers.
//! Replaces per-module `DriftDetector` usage for freeze-safety tracking.
//!
//! # Process Categories
//!
//! Every process name maps to a *freeze category* — a stable string that
//! groups processes with similar freeze-safety characteristics. NARS beliefs
//! are tracked per category, not per individual process name, so evidence
//! accumulates across all processes of the same type.
//!
//! | Category             | Examples                                              |
//! |----------------------|-------------------------------------------------------|
//! | `chromium-renderer`  | "Brave Browser Helper (Renderer)", "Slack Helper (Renderer)" |
//! | `chromium-gpu`       | "Brave Browser Helper (GPU)", "Code Helper (GPU)"    |
//! | `chromium-plugin`    | "* Helper (Plugin)"                                  |
//! | `ide-lsp`            | sourcekit-lsp, clangd, ccls, rust-analyzer            |
//! | `xpc-service`        | Any process name containing "XPCService"              |
//! | `app-helper`         | "* Helper" (plain, no qualifier)                     |
//! | `media-helper`       | Spotify Helper, Music Helper, etc.                   |
//! | `generic`            | Everything else                                       |
//!
//! # Pre-thaw Hints
//!
//! Given a Markov prediction for the next foreground app, `pre_thaw_hint()`
//! returns the process categories that should be thawed before the switch.
//! This is purely app-agnostic — it works for IDEs, media players, browsers,
//! and any other app that spawns helper subprocesses.
//!
//! # NARS Belief Semantics
//!
//! Freeze confidence ∈ [0, 1]:
//!  - Default 0.70 — assume freezing is safe until proven otherwise.
//!  - Drops toward 0.0 as failures accumulate (process died while frozen).
//!  - Recovers naturally via NARS Bayesian forgetting (confidence decay).
//!  - Below 0.35: skip freezing for processes in this category.
//!
//! [Pei Wang 2013] Non-Axiomatic Reasoning System §3.3.3 — Revision Rule.
//! [McGaugh 2004] Amygdala-modulated memory consolidation — arousal weighting.

use crate::engine::nars_belief::{DriftDetector, Salience};

// ── Default freeze confidence (no prior evidence) ────────────────────────────

/// Default freeze confidence for any category with no observed outcomes yet.
/// Conservative: assume safe (0.70) rather than refusing (0.0) until we learn.
const DEFAULT_CONFIDENCE: f32 = 0.70;

/// Minimum confidence to permit freezing a process in this category.
/// Below this threshold, skip freeze — too many failures observed.
const MIN_FREEZE_CONFIDENCE: f32 = 0.35;

// ── FreezeIntelligence ───────────────────────────────────────────────────────

/// Universal cognitive layer for freeze/thaw decisions.
///
/// Tracks NARS freeze-safety beliefs per process category (not per individual
/// process name), so evidence accumulates across all processes of the same type.
/// Supports any freezeable process — Chromium renderers, IDE LSP servers,
/// XPC services, generic helpers, etc.
pub struct FreezeIntelligence {
    beliefs: DriftDetector,
}

impl Default for FreezeIntelligence {
    fn default() -> Self {
        Self::new()
    }
}

impl FreezeIntelligence {
    pub fn new() -> Self {
        Self {
            beliefs: DriftDetector::new(),
        }
    }

    // ── Process classification ────────────────────────────────────────────────

    /// Classify any process name into a stable freeze category.
    ///
    /// Categories are coarse-grained by design: NARS beliefs accumulate per
    /// category, so evidence transfers across all processes of the same type.
    ///
    /// Rules are applied in priority order (most specific first).
    ///
    /// Limitations: name-only. Cannot distinguish "apple-owned" (path /
    /// codesign-derived) or "companion-of-fg" (foreground-context-derived).
    /// Use `classify_full` when those signals are available — they preserve
    /// NARS truth-value calibration for the legacy categories by routing
    /// today's structurally-protected processes (apple-owned, companion) to
    /// their own categories instead of misclassifying them as `background-noise`
    /// or similar.
    pub fn classify(name: &str) -> &'static str {
        // Chromium/Electron helper variants (most specific first)
        if name.ends_with("Helper (Renderer)") {
            return "chromium-renderer";
        }
        if name.ends_with("Helper (GPU)") {
            return "chromium-gpu";
        }
        if name.ends_with("Helper (Plugin)") {
            return "chromium-plugin";
        }

        // IDE language server helpers
        if matches!(name, "sourcekit-lsp" | "clangd" | "ccls" | "rust-analyzer") {
            return "ide-lsp";
        }

        // XPC services — Apple's inter-process communication layer
        if name.contains("XPCService") {
            return "xpc-service";
        }

        // Media player helpers
        if name.contains("Spotify Helper")
            || name.contains("Music Helper")
            || name.contains("Podcasts Helper")
            || name.contains("QuickTime Helper")
        {
            return "media-helper";
        }

        // Generic plain helpers (no role qualifier)
        if name.ends_with(" Helper") {
            return "app-helper";
        }

        "generic"
    }

    /// Classify with full structural context (Sprint Coalition 2026-05-10).
    ///
    /// Preserves NARS belief calibration for the legacy categories by routing
    /// structurally-protected processes to their own categories. Without this,
    /// after Sprint Coalition deployed (commits a381c6b..1ab6bdb), every
    /// process now under apple-owned guard or active-coalition envelope was
    /// still being classified as `background-noise` / `app-helper` /
    /// `generic` — but never throttled. NARS DriftDetector would slowly
    /// corrupt truth-values for those categories: "we say throttle works on
    /// background-noise, but observations show 0% throttle attempts ever
    /// land on these procs". Adversarial review NotebookLM round-3 2026-05-10.
    ///
    /// Priority order (specific first, structural fallback for everything else):
    ///   1. legacy name-pattern classify() — chromium-renderer, ide-lsp,
    ///      xpc-service, media-helper, app-helper. These categories carry
    ///      years of accumulated NARS evidence about freeze semantics, so a
    ///      renderer that happens to also be a fg-companion still routes to
    ///      `chromium-renderer` (the more specific belief).
    ///   2. `apple-owned`  — overrides only when legacy returned `generic`.
    ///      Origin-based: SIP path or codesign Apple chain.
    ///   3. `companion-of-fg` — overrides only when legacy returned `generic`
    ///      AND apple-owned didn't match. Statistical, name-keyed via Lift.
    ///   4. fall through to `generic`.
    ///
    /// Routing today's structurally-protected `generic` processes to
    /// `apple-owned` / `companion-of-fg` instead preserves NARS truth-value
    /// calibration for the legacy categories.
    pub fn classify_full(
        name: &str,
        pid: Option<u32>,
        fg_app: Option<&str>,
        companion_graph: Option<&crate::engine::companion_graph::CompanionGraph>,
    ) -> &'static str {
        let legacy = Self::classify(name);
        if legacy != "generic" {
            return legacy;
        }
        // Layer 2: apple-owned (path/codesign — structural).
        if let Some(p) = pid {
            if crate::engine::apple_owned::is_apple_owned(p) {
                return "apple-owned";
            }
        }
        // Layer 3: companion-of-fg (statistical, name-keyed).
        if let (Some(fg), Some(g)) = (fg_app, companion_graph) {
            if g.is_companion_of(fg, name) {
                return "companion-of-fg";
            }
        }
        "generic"
    }

    /// Same as `observe` but uses `classify_full` so apple-owned and
    /// companion-of-fg processes accumulate evidence in their OWN NARS
    /// categories instead of corrupting the legacy ones.
    pub fn observe_full(
        &mut self,
        process_name: &str,
        pid: Option<u32>,
        fg_app: Option<&str>,
        companion_graph: Option<&crate::engine::companion_graph::CompanionGraph>,
        success: bool,
        salience: f32,
    ) {
        let category = Self::classify_full(process_name, pid, fg_app, companion_graph);
        self.beliefs.observe_salient(
            category,
            success,
            Salience {
                arousal: salience.clamp(0.0, 1.0),
                valence: if success { 0.5 } else { -0.5 },
            },
        );
    }

    // ── Belief update ─────────────────────────────────────────────────────────

    /// Observe a freeze/thaw outcome for any process.
    ///
    /// `success = true` if the process survived the freeze (alive after thaw).
    /// `success = false` if the process died or became unresponsive while frozen.
    /// `salience` — arousal weight for this event (0.0 = routine, 1.0 = crisis).
    ///
    /// Automatically routes to the correct category via `classify()`.
    pub fn observe(&mut self, process_name: &str, success: bool, salience: f32) {
        let category = Self::classify(process_name);
        self.beliefs.observe_salient(
            category,
            success,
            Salience {
                arousal: salience.clamp(0.0, 1.0),
                valence: if success { 0.5 } else { -0.5 },
            },
        );
    }

    // ── Belief query ──────────────────────────────────────────────────────────

    /// Returns freeze confidence [0.0, 1.0] for a process category.
    ///
    /// Default `DEFAULT_CONFIDENCE` (0.70) when no evidence exists yet.
    /// Drops toward 0.0 as failures accumulate.
    pub fn confidence_for_category(&self, category: &str) -> f32 {
        self.beliefs
            .belief(category)
            .map(|tv| tv.frequency)
            .unwrap_or(DEFAULT_CONFIDENCE)
    }

    /// Returns freeze confidence for a process (looks up category automatically).
    pub fn confidence(&self, process_name: &str) -> f32 {
        self.confidence_for_category(Self::classify(process_name))
    }

    /// Should we attempt to freeze this process based on learned outcomes?
    ///
    /// Returns `false` if accumulated failures have dropped confidence below
    /// `MIN_FREEZE_CONFIDENCE`. Returns `true` by default (no evidence = safe).
    pub fn should_freeze(&self, process_name: &str) -> bool {
        self.confidence(process_name) >= MIN_FREEZE_CONFIDENCE
    }

    // ── Pre-thaw hints ────────────────────────────────────────────────────────

    /// Given a Markov-predicted next foreground app name, return which process
    /// categories should be pre-thawed before the switch happens.
    ///
    /// App-agnostic: works for browsers, IDEs, media players, and any other app.
    /// Returns a `Vec<&'static str>` of category names to thaw.
    ///
    /// [Altmann & Trafton 2002] Pre-activate resources before predicted task switch.
    pub fn pre_thaw_hint(predicted_app: &str) -> Vec<&'static str> {
        // Chromium/Electron browser or Electron app (Slack, Discord, Code, Cursor…)
        // These spawn "Helper (Renderer)" subprocesses.
        const CHROMIUM_APPS: &[&str] = &[
            "Brave Browser",
            "Google Chrome",
            "Microsoft Edge",
            "Arc",
            "Vivaldi",
            "Opera",
            "Chromium",
            "Slack",
            "Code",
            "Cursor",
            "Discord",
            "Notion",
            "Linear",
            "Figma",
        ];
        if CHROMIUM_APPS
            .iter()
            .any(|&a| predicted_app == a || predicted_app.starts_with(a))
        {
            return vec!["chromium-renderer", "chromium-gpu"];
        }

        // IDE / code editors that spawn LSP helpers
        const IDE_APPS: &[&str] = &["Xcode", "Nova", "Zed", "CLion", "PyCharm", "GoLand"];
        if IDE_APPS
            .iter()
            .any(|&a| predicted_app == a || predicted_app.starts_with(a))
        {
            return vec!["ide-lsp", "app-helper"];
        }

        // Media players
        const MEDIA_APPS: &[&str] = &[
            "Spotify",
            "Music",
            "Podcasts",
            "QuickTime Player",
            "IINA",
            "VLC",
        ];
        if MEDIA_APPS
            .iter()
            .any(|&a| predicted_app == a || predicted_app.starts_with(a))
        {
            return vec!["media-helper", "app-helper"];
        }

        // For any other app: pre-thaw generic app helpers that share the app name.
        // This is a best-effort hint — the daemon filters by actual process name.
        vec!["app-helper"]
    }

    // ── Inner belief access ───────────────────────────────────────────────────

    /// Access the underlying DriftDetector for NARS drift tracking.
    pub fn drift_detector(&self) -> &DriftDetector {
        &self.beliefs
    }

    /// Mutable access to the underlying DriftDetector (for Bayesian forgetting etc.).
    pub fn drift_detector_mut(&mut self) -> &mut DriftDetector {
        &mut self.beliefs
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify() ────────────────────────────────────────────────────────────

    #[test]
    fn classify_chromium_renderer() {
        assert_eq!(
            FreezeIntelligence::classify("Brave Browser Helper (Renderer)"),
            "chromium-renderer"
        );
        assert_eq!(
            FreezeIntelligence::classify("Slack Helper (Renderer)"),
            "chromium-renderer"
        );
        assert_eq!(
            FreezeIntelligence::classify("Code Helper (Renderer)"),
            "chromium-renderer"
        );
    }

    #[test]
    fn classify_chromium_gpu() {
        assert_eq!(
            FreezeIntelligence::classify("Brave Browser Helper (GPU)"),
            "chromium-gpu"
        );
        assert_eq!(
            FreezeIntelligence::classify("Cursor Helper (GPU)"),
            "chromium-gpu"
        );
    }

    #[test]
    fn classify_ide_lsp() {
        assert_eq!(FreezeIntelligence::classify("sourcekit-lsp"), "ide-lsp");
        assert_eq!(FreezeIntelligence::classify("clangd"), "ide-lsp");
        assert_eq!(FreezeIntelligence::classify("ccls"), "ide-lsp");
        assert_eq!(FreezeIntelligence::classify("rust-analyzer"), "ide-lsp");
    }

    #[test]
    fn classify_xpc_service() {
        assert_eq!(
            FreezeIntelligence::classify("com.apple.WebKit.WebContent.XPCService"),
            "xpc-service"
        );
        assert_eq!(
            FreezeIntelligence::classify("com.example.MyApp.XPCService"),
            "xpc-service"
        );
    }

    #[test]
    fn classify_app_helper() {
        assert_eq!(
            FreezeIntelligence::classify("Spotify Helper"),
            // "Spotify Helper" hits the media-helper rule first via contains()
            // Wait — "Spotify Helper" contains "Spotify Helper" → media-helper
            "media-helper"
        );
        assert_eq!(FreezeIntelligence::classify("SomeApp Helper"), "app-helper");
    }

    #[test]
    fn classify_generic() {
        assert_eq!(FreezeIntelligence::classify("launchd"), "generic");
        assert_eq!(FreezeIntelligence::classify("kernel_task"), "generic");
        assert_eq!(FreezeIntelligence::classify("bash"), "generic");
    }

    // ── confidence() and observe() ────────────────────────────────────────────

    #[test]
    fn confidence_default_is_0_70() {
        let fi = FreezeIntelligence::new();
        // No observations → default confidence
        assert!(
            (fi.confidence("Brave Browser Helper (Renderer)") - 0.70).abs() < 1e-6,
            "default confidence should be 0.70, got {}",
            fi.confidence("Brave Browser Helper (Renderer)")
        );
    }

    #[test]
    fn confidence_drops_on_failures() {
        let mut fi = FreezeIntelligence::new();
        // Record repeated failures for chromium-renderer category
        let name = "Brave Browser Helper (Renderer)";
        for _ in 0..10 {
            fi.observe(name, false, 0.5);
        }
        let conf = fi.confidence(name);
        assert!(
            conf < 0.5,
            "confidence should drop significantly after 10 failures, got {}",
            conf
        );
    }

    #[test]
    fn confidence_stays_high_on_successes() {
        let mut fi = FreezeIntelligence::new();
        let name = "Slack Helper (Renderer)";
        for _ in 0..5 {
            fi.observe(name, true, 0.3);
        }
        let conf = fi.confidence(name);
        assert!(
            conf >= 0.6,
            "confidence should remain high after 5 successes, got {}",
            conf
        );
    }

    // ── should_freeze() ────────────────────────────────────────────────────────

    #[test]
    fn should_freeze_true_by_default() {
        let fi = FreezeIntelligence::new();
        assert!(fi.should_freeze("Brave Browser Helper (Renderer)"));
        assert!(fi.should_freeze("sourcekit-lsp"));
        assert!(fi.should_freeze("SomeApp Helper"));
    }

    #[test]
    fn should_freeze_blocks_low_confidence() {
        let mut fi = FreezeIntelligence::new();
        let name = "Brave Browser Helper (Renderer)";
        // Drive confidence below MIN_FREEZE_CONFIDENCE (0.35)
        // Many high-arousal failures should push frequency toward 0
        for _ in 0..20 {
            fi.observe(name, false, 0.9);
        }
        assert!(
            !fi.should_freeze(name),
            "should_freeze must return false when confidence is too low"
        );
    }

    // ── pre_thaw_hint() ────────────────────────────────────────────────────────

    #[test]
    fn pre_thaw_hint_chromium_returns_renderer_and_gpu() {
        let hint = FreezeIntelligence::pre_thaw_hint("Brave Browser");
        assert!(
            hint.contains(&"chromium-renderer"),
            "missing chromium-renderer"
        );
        assert!(hint.contains(&"chromium-gpu"), "missing chromium-gpu");
    }

    #[test]
    fn pre_thaw_hint_ide_returns_lsp_category() {
        let hint = FreezeIntelligence::pre_thaw_hint("Xcode");
        assert!(hint.contains(&"ide-lsp"), "Xcode should hint ide-lsp");
        assert!(hint.contains(&"app-helper"), "Xcode should hint app-helper");
    }

    #[test]
    fn pre_thaw_hint_media_returns_media_helper() {
        let hint = FreezeIntelligence::pre_thaw_hint("Spotify");
        assert!(
            hint.contains(&"media-helper"),
            "Spotify should hint media-helper"
        );
    }

    #[test]
    fn pre_thaw_hint_unknown_app_returns_app_helper() {
        let hint = FreezeIntelligence::pre_thaw_hint("SomeRandomApp");
        assert_eq!(hint, vec!["app-helper"]);
    }

    // ── cross-category isolation ───────────────────────────────────────────────

    #[test]
    fn failures_in_one_category_do_not_affect_another() {
        let mut fi = FreezeIntelligence::new();
        // Destroy confidence for ide-lsp
        for _ in 0..20 {
            fi.observe("sourcekit-lsp", false, 0.9);
        }
        // chromium-renderer should be unaffected
        assert!(
            fi.should_freeze("Brave Browser Helper (Renderer)"),
            "chromium-renderer confidence should be independent of ide-lsp failures"
        );
    }

    #[test]
    fn classify_full_returns_apple_owned_for_pid_zero() {
        // pid 0 = kernel_task → is_apple_owned returns true → category apple-owned
        let cat = FreezeIntelligence::classify_full("kernel_task", Some(0), None, None);
        assert_eq!(cat, "apple-owned");
    }

    #[test]
    fn classify_full_legacy_wins_over_new_categories() {
        // Renderer name → legacy classify returns "chromium-renderer".
        // Even with apple-owned pid signal, legacy wins because it's more
        // specific (carries chromium-specific NARS evidence).
        let cat = FreezeIntelligence::classify_full(
            "Brave Browser Helper (Renderer)",
            Some(0), // would be apple-owned, but legacy is more specific
            None,
            None,
        );
        assert_eq!(cat, "chromium-renderer");
    }

    #[test]
    fn classify_full_returns_companion_when_graph_matches() {
        use crate::engine::companion_graph::CompanionGraph;
        let mut g = CompanionGraph::new();
        // Build a Brave session that makes Slack a companion (lift > 2.0).
        for c in 0..200 {
            g.observe_cycle(Some("Brave"), &["Slack".into()], c);
        }
        for c in 200..1000 {
            g.observe_cycle(Some("Other"), &["unrelated".into()], c);
        }
        // Slack while Brave is fg → companion-of-fg, NOT app-helper.
        let cat = FreezeIntelligence::classify_full(
            "Slack",
            None, // no pid → skip apple_owned check
            Some("Brave"),
            Some(&g),
        );
        assert_eq!(cat, "companion-of-fg");
    }

    #[test]
    fn observe_full_routes_evidence_to_new_categories() {
        // Simulate 30 successful observations on apple-owned processes.
        // Legacy `observe` would route them to `generic` (or whatever the
        // name-based classification said). `observe_full` routes them to
        // `apple-owned`, preserving `generic` calibration.
        let mut fi = FreezeIntelligence::new();
        for _ in 0..30 {
            // pid 0 = kernel_task, ensures apple-owned routing.
            fi.observe_full("kernel_task", Some(0), None, None, true, 0.5);
        }
        // apple-owned belief now has 30 successful observations → high
        // confidence (>= default 0.70).
        let apple_conf = fi.confidence_for_category("apple-owned");
        assert!(
            apple_conf >= 0.70,
            "apple-owned should accumulate evidence, got {apple_conf}"
        );
        // generic stays at default — was NOT polluted.
        let gen_conf = fi.confidence_for_category("generic");
        assert!(
            (gen_conf - 0.70).abs() < 0.01,
            "generic should remain at default 0.70, got {gen_conf}"
        );
    }
}
