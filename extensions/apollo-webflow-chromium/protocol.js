(function (root) {
  'use strict';

  const MAX_MESSAGE_BYTES = 16 * 1024;
  const PHASES = new Set(['started', 'committed', 'dom-ready', 'loaded', 'settled', 'failed', 'abandoned']);
  const SOURCES = new Set(['extension-lifecycle', 'extension-vitals']);
  const ERROR_CLASSES = new Set(['network', 'name-resolution', 'connection', 'tls', 'timeout', 'cancelled', 'browser', 'unknown']);

  function randomOpaqueId(fill = (bytes) => crypto.getRandomValues(bytes)) {
    const bytes = new Uint8Array(16);
    fill(bytes);
    if (bytes.every((byte) => byte === 0)) throw new Error('opaque ID must be nonzero');
    return Array.from(bytes);
  }

  function boundedInteger(value, maximum) {
    if (!Number.isFinite(value)) return undefined;
    return Math.min(maximum, Math.max(0, Math.round(value)));
  }

  /// v2 adds the per-interaction fields. Must match
  /// `WEBFLOW_SCHEMA_VERSION` in crates/apollo-engine/src/engine/webflow_types.rs.
  const SCHEMA_VERSION = 2;
  /// Must match manifest.json. Sent so the daemon can say "EXT v1 STALE"
  /// instead of showing unexplained zeros.
  const EXTENSION_VERSION = '2.0.2';
  /// Bit positions mirror FeatureCapabilities in webflow_types.rs.
  const CAP_INTERACTION_GROUPING = 1 << 0;
  const CAP_COMPONENT_BREAKDOWN = 1 << 1;
  const CAP_TRANSPORT_TIMING = 1 << 2;
  const CAP_FAST_PROBE = 1 << 3;
  const FEATURE_CAPABILITIES =
    CAP_INTERACTION_GROUPING | CAP_COMPONENT_BREAKDOWN | CAP_TRANSPORT_TIMING;

  function normalizeMetrics(input = {}) {
    const output = {};
    const timing = {
      ttfbMs: 'ttfb_ms', domReadyMs: 'dom_ready_ms', loadMs: 'load_ms',
      lcpMs: 'lcp_ms', eventDurationMs: 'event_duration_ms',
      inpEstimateMs: 'inp_estimate_ms',
      inputDelayTotalMs: 'input_delay_total_ms',
      processingTotalMs: 'processing_total_ms',
      presentationTotalMs: 'presentation_total_ms',
      longTaskTotalMs: 'long_task_total_ms',
    };
    for (const [source, target] of Object.entries(timing)) {
      const value = boundedInteger(input[source], 120_000);
      if (value !== undefined) output[target] = value;
    }
    const cls = boundedInteger(input.clsMilli, 100_000);
    if (cls !== undefined) output.cls_milli = cls;
    for (const [source, target] of [
      ['longTaskCount', 'long_task_count'], ['resourceCount', 'resource_count'],
      ['interactionCount', 'interaction_count'], ['interactionsDropped', 'interactions_dropped'],
    ]) {
      const value = boundedInteger(input[source], 100_000);
      if (value !== undefined) output[target] = value;
    }
    const transfer = boundedInteger(input.transferBytes, 16 * 1024 * 1024 * 1024);
    if (transfer !== undefined) output.transfer_bytes = transfer;
    if (ERROR_CLASSES.has(input.errorClass)) output.error_class = input.errorClass;
    return output;
  }

  function normalizeTransport(input) {
    if (!input || typeof input !== 'object') return undefined;
    const output = {};
    for (const [source, target] of [
      ['contentSendStartedAtMs', 'content_send_started_at_ms'],
      ['serviceWorkerReceivedAtMs', 'service_worker_received_at_ms'],
      ['nativeMessageStartedAtMs', 'native_message_started_at_ms'],
    ]) {
      const value = boundedInteger(input[source], Number.MAX_SAFE_INTEGER);
      if (value !== undefined) output[target] = value;
    }
    const depth = boundedInteger(input.tabQueueDepth, 100_000);
    if (depth !== undefined) output.tab_queue_depth = depth;
    if (typeof input.serviceWorkerColdStart === 'boolean') {
      output.service_worker_cold_start = input.serviceWorkerColdStart;
    }
    return Object.keys(output).length > 0 ? output : undefined;
  }

  function validId(value) {
    return Array.isArray(value)
      && value.length === 16
      && value.some((byte) => byte !== 0)
      && value.every((byte) => Number.isInteger(byte) && byte >= 0 && byte <= 255);
  }

  function buildEvent(input) {
    if (!validId(input.browserSessionId) || !validId(input.tabSessionId) || !validId(input.navigationId)) {
      throw new Error('invalid opaque identity');
    }
    if (!Number.isSafeInteger(input.sequence) || input.sequence <= 0) throw new Error('invalid sequence');
    if (!PHASES.has(input.phase) || !SOURCES.has(input.source)) throw new Error('invalid event kind');
    const event = {
      schema_version: SCHEMA_VERSION,
      browser_session_id: input.browserSessionId,
      tab_session_id: input.tabSessionId,
      navigation_id: input.navigationId,
      sequence: input.sequence,
      phase: input.phase,
      source: input.source,
      metrics: normalizeMetrics(input.metrics),
    };
    event.producer_kind = 'chromium-extension';
    event.extension_version = EXTENSION_VERSION;
    event.feature_capabilities = input.featureCapabilities === undefined
      ? FEATURE_CAPABILITIES
      : (input.featureCapabilities | 0);
    const transport = normalizeTransport(input.transport);
    if (transport !== undefined) event.transport = transport;
    if (input.siteBucket !== undefined) {
      if (!validId(input.siteBucket)) throw new Error('invalid site bucket');
      event.site_bucket = input.siteBucket;
    }
    return event;
  }

  function encodeBounded(event) {
    const bytes = new TextEncoder().encode(JSON.stringify(event));
    if (bytes.byteLength === 0 || bytes.byteLength > MAX_MESSAGE_BYTES) {
      throw new Error('WebFlow message outside native bound');
    }
    return bytes;
  }

  class NavigationTracker {
    constructor(capacity = 64, random = randomOpaqueId) {
      this.capacity = capacity;
      this.random = random;
      this.tabs = new Map();
      this.tabSessions = new Map();
    }

    get size() { return this.tabs.size; }
    get(tabId) { return this.tabs.get(tabId); }

    start(tabId) {
      const abandoned = this.tabs.get(tabId);
      if (!this.tabSessions.has(tabId)) this.tabSessions.set(tabId, this.random());
      this.tabs.delete(tabId);
      const record = {
        tabId,
        tabSessionId: this.tabSessions.get(tabId),
        navigationId: this.random(),
      };
      this.tabs.set(tabId, record);
      while (this.tabs.size > this.capacity) {
        const oldest = this.tabs.keys().next().value;
        this.tabs.delete(oldest);
        this.tabSessions.delete(oldest);
      }
      return { ...record, abandoned };
    }

    close(tabId) {
      const record = this.tabs.get(tabId);
      this.tabs.delete(tabId);
      this.tabSessions.delete(tabId);
      return record;
    }
  }


  /// Fold PerformanceEventTiming entries into interactions.
  ///
  /// One interaction emits several entries (pointerdown/pointerup/click); Web
  /// Vitals defines its latency as the longest of them, and the components must
  /// come from that same entry so the three parts stay reconcilable with the
  /// total. Entries with `interactionId === 0` are not interactions at all.
  const MAX_TRACKED_INTERACTIONS = 256;
  const MIN_INTERACTIONS_FOR_PERCENTILE = 50;

  function foldInteractions(entries, state) {
    const interactions = state && state.interactions ? state.interactions : new Map();
    let dropped = state && state.dropped ? state.dropped : 0;
    for (const entry of entries || []) {
      const id = entry && entry.interactionId;
      if (!id) continue;
      if (interactions.size >= MAX_TRACKED_INTERACTIONS && !interactions.has(id)) {
        dropped += 1;
        continue;
      }
      const duration = Math.max(0, Math.round(entry.duration || 0));
      const prior = interactions.get(id);
      if (prior && prior.totalMs >= duration) {
        prior.entryCount += 1;
        continue;
      }
      const startTime = entry.startTime || 0;
      const processingStart = entry.processingStart || 0;
      const processingEnd = entry.processingEnd || 0;
      interactions.set(id, {
        totalMs: duration,
        inputDelayMs: Math.max(0, Math.round(processingStart - startTime)),
        processingMs: Math.max(0, Math.round(processingEnd - processingStart)),
        presentationMs: Math.max(0, Math.round(startTime + duration - processingEnd)),
        entryCount: prior ? prior.entryCount + 1 : 1,
        cancelable: Boolean(entry.cancelable),
      });
    }
    return { interactions, dropped };
  }

  function inpEstimateMs(interactions) {
    const values = Array.from(interactions.values(), (i) => i.totalMs).sort((a, b) => a - b);
    if (values.length === 0) return undefined;
    if (values.length < MIN_INTERACTIONS_FOR_PERCENTILE) return values[values.length - 1];
    const rank = Math.ceil(values.length * 0.98);
    return values[Math.min(values.length, Math.max(1, rank)) - 1];
  }

  function componentTotals(interactions) {
    let inputDelay = 0; let processing = 0; let presentation = 0;
    for (const i of interactions.values()) {
      inputDelay += i.inputDelayMs;
      processing += i.processingMs;
      presentation += i.presentationMs;
    }
    return { inputDelay, processing, presentation };
  }

  const api = { MAX_MESSAGE_BYTES, SCHEMA_VERSION, EXTENSION_VERSION,
    FEATURE_CAPABILITIES, CAP_FAST_PROBE, normalizeTransport, foldInteractions, inpEstimateMs, componentTotals,
    MAX_TRACKED_INTERACTIONS, MIN_INTERACTIONS_FOR_PERCENTILE, NavigationTracker, buildEvent, encodeBounded, normalizeMetrics, randomOpaqueId };
  root.ApolloWebFlowProtocol = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
