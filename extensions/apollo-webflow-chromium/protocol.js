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

  function normalizeMetrics(input = {}) {
    const output = {};
    const timing = {
      ttfbMs: 'ttfb_ms', domReadyMs: 'dom_ready_ms', loadMs: 'load_ms',
      lcpMs: 'lcp_ms', inpMs: 'inp_ms', longTaskTotalMs: 'long_task_total_ms',
    };
    for (const [source, target] of Object.entries(timing)) {
      const value = boundedInteger(input[source], 120_000);
      if (value !== undefined) output[target] = value;
    }
    const cls = boundedInteger(input.clsMilli, 100_000);
    if (cls !== undefined) output.cls_milli = cls;
    for (const [source, target] of [['longTaskCount', 'long_task_count'], ['resourceCount', 'resource_count']]) {
      const value = boundedInteger(input[source], 100_000);
      if (value !== undefined) output[target] = value;
    }
    const transfer = boundedInteger(input.transferBytes, 16 * 1024 * 1024 * 1024);
    if (transfer !== undefined) output.transfer_bytes = transfer;
    if (ERROR_CLASSES.has(input.errorClass)) output.error_class = input.errorClass;
    return output;
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
      schema_version: 1,
      browser_session_id: input.browserSessionId,
      tab_session_id: input.tabSessionId,
      navigation_id: input.navigationId,
      sequence: input.sequence,
      phase: input.phase,
      source: input.source,
      metrics: normalizeMetrics(input.metrics),
    };
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

  const api = { MAX_MESSAGE_BYTES, NavigationTracker, buildEvent, encodeBounded, normalizeMetrics, randomOpaqueId };
  root.ApolloWebFlowProtocol = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
