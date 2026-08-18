'use strict';

const EVENT_DURATION_TAIL_THRESHOLD_MS = 40;

const metrics = {
  lcpMs: undefined,
  eventDurationMs: undefined,
  clsMilli: 0,
  longTaskCount: 0,
  longTaskTotalMs: 0,
  resourceCount: 0,
  transferBytes: 0,
};
let quietTimer;
const startedAt = performance.now();

function resetQuietTimer() {
  clearTimeout(quietTimer);
  quietTimer = setTimeout(report, 750);
}

function observe(type, callback, options = {}) {
  try {
    const observer = new PerformanceObserver((list) => callback(list.getEntries()));
    observer.observe({ type, buffered: true, ...options });
  } catch (_) {}
}

observe('largest-contentful-paint', (entries) => {
  const last = entries.at(-1);
  if (last) metrics.lcpMs = last.startTime;
});
// Interactions, grouped by interactionId — not individual entries.
//
// The folding, percentile and component maths live in protocol.js so they are
// unit-testable outside the browser; this file only wires them to the observer.
// The previous collector took max(entry.duration) above a 40 ms threshold and
// called it INP: that conflates the several entries of one interaction, counts
// interactionId === 0 (scroll, not an interaction), and hides every fast one.
const P = globalThis.ApolloWebFlowProtocol;
let folded = { interactions: new Map(), dropped: 0 };

observe('event', (entries) => {
  folded = P.foldInteractions(entries, folded);
  resetQuietTimer();
}, { durationThreshold: 0 });
observe('first-input', (entries) => {
  folded = P.foldInteractions(entries, folded);
});

function eventDurationTailMs() {
  // Same quantity the old collector published, under its true name: the worst
  // interaction above the legacy threshold. Kept so the operator retains a
  // continuous series while the corrected one warms up. Not comparable to
  // inp_estimate_ms — different definitions, different populations.
  let tail;
  for (const i of folded.interactions.values()) {
    if (i.totalMs >= EVENT_DURATION_TAIL_THRESHOLD_MS) {
      tail = Math.max(tail || 0, i.totalMs);
    }
  }
  return tail;
}

function report() {
  const navigation = performance.getEntriesByType('navigation')[0];
  const components = P.componentTotals(folded.interactions);
  const payload = {
    ...metrics,
    eventDurationMs: eventDurationTailMs(),
    inpEstimateMs: P.inpEstimateMs(folded.interactions),
    interactionCount: folded.interactions.size,
    interactionsDropped: folded.dropped,
    inputDelayTotalMs: components.inputDelay,
    processingTotalMs: components.processing,
    presentationTotalMs: components.presentation,
    ttfbMs: navigation ? navigation.responseStart : undefined,
    domReadyMs: navigation ? navigation.domContentLoadedEventEnd : undefined,
    loadMs: navigation ? navigation.loadEventEnd : undefined,
  };
  chrome.runtime.sendMessage({ type: 'apollo-webflow-vitals', metrics: payload }).catch(() => {});
}

// A navigation or a closing tab ends the interaction population: carrying it
// across would attribute one page's latency to another.
addEventListener('pagehide', () => {
  report();
  folded = { interactions: new Map(), dropped: 0 };
}, { capture: true });

addEventListener('load', resetQuietTimer, { once: true });
setTimeout(report, Math.max(0, 10_000 - performance.now() + startedAt));
