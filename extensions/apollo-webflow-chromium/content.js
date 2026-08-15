'use strict';

const metrics = {
  lcpMs: undefined,
  inpMs: undefined,
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
observe('event', (entries) => {
  for (const entry of entries) metrics.inpMs = Math.max(metrics.inpMs || 0, entry.duration || 0);
}, { durationThreshold: 40 });
observe('layout-shift', (entries) => {
  for (const entry of entries) {
    if (!entry.hadRecentInput) metrics.clsMilli += Math.round((entry.value || 0) * 1000);
  }
});
observe('longtask', (entries) => {
  metrics.longTaskCount += entries.length;
  for (const entry of entries) metrics.longTaskTotalMs += Math.round(entry.duration || 0);
  resetQuietTimer();
});
observe('resource', (entries) => {
  metrics.resourceCount += entries.length;
  for (const entry of entries) {
    metrics.transferBytes += Math.max(0, Number(entry.transferSize || entry.encodedBodySize || 0));
  }
  resetQuietTimer();
});

function report() {
  const navigation = performance.getEntriesByType('navigation')[0];
  const payload = {
    ...metrics,
    ttfbMs: navigation ? navigation.responseStart : undefined,
    domReadyMs: navigation ? navigation.domContentLoadedEventEnd : undefined,
    loadMs: navigation ? navigation.loadEventEnd : undefined,
  };
  chrome.runtime.sendMessage({ type: 'apollo-webflow-vitals', metrics: payload }).catch(() => {});
}

addEventListener('load', resetQuietTimer, { once: true });
setTimeout(report, Math.max(0, 10_000 - performance.now() + startedAt));
