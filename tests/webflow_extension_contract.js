const test = require('node:test');
const assert = require('node:assert/strict');

const {
  MAX_MESSAGE_BYTES,
  NavigationTracker,
  buildEvent,
  encodeBounded,
  normalizeMetrics,
  randomOpaqueId,
} = require('../extensions/apollo-webflow-chromium/protocol.js');

const FORBIDDEN = ['url', 'origin', 'title', 'text', 'cookie', 'header', 'body', 'dom', 'host', 'path'];

function ids() {
  return {
    browserSessionId: Array(16).fill(1),
    tabSessionId: Array(16).fill(2),
    navigationId: Array(16).fill(3),
  };
}

test('opaque IDs are sixteen nonzero bytes', () => {
  const id = randomOpaqueId((bytes) => bytes.fill(7));
  assert.equal(id.length, 16);
  assert.ok(id.every((byte) => byte === 7));
  assert.throws(() => randomOpaqueId((bytes) => bytes.fill(0)));
});

test('wire event contains only the closed privacy-safe schema', () => {
  const event = buildEvent({
    ...ids(),
    sequence: 1,
    phase: 'started',
    source: 'extension-lifecycle',
    siteBucket: Array(16).fill(9),
    metrics: { ttfbMs: 20, resourceCount: 7, transferBytes: 12_000 },
  });
  const encoded = JSON.stringify(event).toLowerCase();
  for (const forbidden of FORBIDDEN) {
    assert.equal(encoded.includes(`"${forbidden}"`), false, forbidden);
  }
  // The closed set grows only with producer identity — never with content.
  assert.deepEqual(Object.keys(event).sort(), [
    'browser_session_id',
    'extension_version',
    'feature_capabilities',
    'metrics',
    'navigation_id',
    'phase',
    'producer_kind',
    'schema_version',
    'sequence',
    'site_bucket',
    'source',
    'tab_session_id',
  ]);
  assert.ok(encodeBounded(event).byteLength <= MAX_MESSAGE_BYTES);
});

test('metrics are clamped and unknown fields are discarded', () => {
  const metrics = normalizeMetrics({
    ttfbMs: -9,
    loadMs: 999_999,
    clsMilli: 150_000,
    resourceCount: 999_999,
    transferBytes: Number.MAX_SAFE_INTEGER,
    title: 'must disappear',
    rawUrl: 'https://example.test/private',
  });
  assert.deepEqual(metrics, {
    ttfb_ms: 0,
    load_ms: 120_000,
    cls_milli: 100_000,
    resource_count: 100_000,
    transfer_bytes: 17_179_869_184,
  });
});

test('oversized or malformed events fail closed', () => {
  assert.throws(() => buildEvent({ ...ids(), sequence: 0, phase: 'started', source: 'extension-lifecycle' }));
  assert.throws(() => buildEvent({ ...ids(), sequence: 1, phase: 'invented', source: 'extension-lifecycle' }));
  const huge = buildEvent({ ...ids(), sequence: 1, phase: 'started', source: 'extension-lifecycle' });
  huge.metrics = { padding: 'x'.repeat(MAX_MESSAGE_BYTES) };
  assert.throws(() => encodeBounded(huge));
});

test('navigation tracker abandons replacement and remains bounded to 64 tabs', () => {
  const tracker = new NavigationTracker(64, () => Array(16).fill(5));
  const first = tracker.start(1);
  const replacement = tracker.start(1);
  assert.deepEqual(replacement.abandoned.navigationId, first.navigationId);
  for (let tabId = 2; tabId <= 80; tabId += 1) tracker.start(tabId);
  assert.equal(tracker.size, 64);
  assert.equal(tracker.get(1), undefined);
});

test('extension sources never contain forbidden content APIs', () => {
  const fs = require('node:fs');
  for (const file of ['background.js', 'content.js']) {
    const source = fs.readFileSync(`extensions/apollo-webflow-chromium/${file}`, 'utf8').toLowerCase();
    for (const forbidden of ['document.body', 'innertext', 'outerhtml', 'cookies.get', 'webrequest']) {
      assert.equal(source.includes(forbidden), false, `${file}: ${forbidden}`);
    }
  }
});

test('manifest public key pins the expected native-host extension ID', () => {
  const crypto = require('node:crypto');
  const manifest = require('../extensions/apollo-webflow-chromium/manifest.json');
  const digest = crypto.createHash('sha256').update(Buffer.from(manifest.key, 'base64')).digest('hex').slice(0, 32);
  const extensionId = digest.replace(/[0-9a-f]/g, (nibble) => String.fromCharCode(97 + Number.parseInt(nibble, 16)));
  assert.equal(extensionId, 'mhagiddoeecedoknmhdlhghdnglglbhp');
});

// ── 0A-obs: interaction folding ─────────────────────────────────────────────
// The previous collector took max(entry.duration) over entries above a 40 ms
// threshold and published it as INP. These pin the corrected semantics.

const {
  foldInteractions,
  inpEstimateMs,
  componentTotals,
  MAX_TRACKED_INTERACTIONS,
} = require('../extensions/apollo-webflow-chromium/protocol.js');

function entry(overrides = {}) {
  return {
    interactionId: 1,
    startTime: 1000,
    processingStart: 1010,
    processingEnd: 1030,
    duration: 48,
    cancelable: true,
    ...overrides,
  };
}

test('entries sharing an interactionId fold into one interaction', () => {
  const { interactions } = foldInteractions([
    entry({ duration: 16 }),
    entry({ duration: 48 }),
    entry({ duration: 32 }),
  ]);
  assert.equal(interactions.size, 1);
  assert.equal(interactions.get(1).totalMs, 48, 'the longest entry defines the latency');
  assert.equal(interactions.get(1).entryCount, 3);
});

test('interactionId 0 is not an interaction and is discarded', () => {
  const { interactions } = foldInteractions([
    entry({ interactionId: 0, duration: 900 }),
    entry({ interactionId: undefined, duration: 900 }),
  ]);
  assert.equal(interactions.size, 0);
  assert.equal(inpEstimateMs(interactions), undefined);
});

test('consecutive clicks are separate interactions, not one maximum', () => {
  const { interactions } = foldInteractions([
    entry({ interactionId: 1, duration: 20 }),
    entry({ interactionId: 2, duration: 300 }),
    entry({ interactionId: 3, duration: 24 }),
  ]);
  assert.equal(interactions.size, 3);
  assert.equal(inpEstimateMs(interactions), 300, 'worst interaction below the percentile floor');
});

test('repeated keys each count as their own interaction', () => {
  const entries = Array.from({ length: 12 }, (_, index) =>
    entry({ interactionId: index + 1, duration: 8 + index }));
  const { interactions } = foldInteractions(entries);
  assert.equal(interactions.size, 12);
});

test('fast interactions are represented, not filtered away', () => {
  const { interactions } = foldInteractions([entry({ duration: 8 })]);
  assert.equal(interactions.size, 1, 'no 40 ms threshold hides them');
  assert.equal(inpEstimateMs(interactions), 8);
});

test('components reconcile with the total for the winning entry', () => {
  const { interactions } = foldInteractions([
    entry({ startTime: 1000, processingStart: 1012, processingEnd: 1040, duration: 60 }),
  ]);
  const i = interactions.get(1);
  assert.equal(i.inputDelayMs, 12);
  assert.equal(i.processingMs, 28);
  assert.equal(i.presentationMs, 20);
  assert.equal(i.inputDelayMs + i.processingMs + i.presentationMs, i.totalMs);
});

test('a cancelled event keeps its flag so it can be excluded downstream', () => {
  const { interactions } = foldInteractions([entry({ cancelable: false })]);
  assert.equal(interactions.get(1).cancelable, false);
});

test('an entry with no paint still yields nonnegative components', () => {
  const { interactions } = foldInteractions([
    entry({ startTime: 1000, processingStart: 1010, processingEnd: 1200, duration: 100 }),
  ]);
  const i = interactions.get(1);
  assert.ok(i.presentationMs >= 0, 'processingEnd past the paint must not go negative');
});

test('late entries update an existing interaction only when longer', () => {
  const first = foldInteractions([entry({ duration: 200 })]);
  const second = foldInteractions([entry({ duration: 40 })], first);
  assert.equal(second.interactions.get(1).totalMs, 200);
  const third = foldInteractions([entry({ duration: 260 })], second);
  assert.equal(third.interactions.get(1).totalMs, 260);
});

test('duplicate identical entries do not inflate the interaction count', () => {
  const duplicated = [entry(), entry(), entry()];
  const { interactions } = foldInteractions(duplicated);
  assert.equal(interactions.size, 1);
});

test('the tracked population is bounded against a hostile page', () => {
  const entries = Array.from({ length: MAX_TRACKED_INTERACTIONS + 50 }, (_, index) =>
    entry({ interactionId: index + 1, duration: 10 }));
  const { interactions, dropped } = foldInteractions(entries);
  assert.equal(interactions.size, MAX_TRACKED_INTERACTIONS);
  assert.equal(dropped, 50, 'refusals are counted, not silent');
});

test('component totals sum across interactions', () => {
  const { interactions } = foldInteractions([
    entry({ interactionId: 1, startTime: 0, processingStart: 10, processingEnd: 30, duration: 50 }),
    entry({ interactionId: 2, startTime: 0, processingStart: 5, processingEnd: 25, duration: 40 }),
  ]);
  const totals = componentTotals(interactions);
  assert.equal(totals.inputDelay, 15);
  assert.equal(totals.processing, 40);
  assert.equal(totals.presentation, 35);
});

test('every event declares its producer identity and capabilities', () => {
  const event = buildEvent({ ...ids(), sequence: 1, phase: 'settled', source: 'extension-vitals', metrics: {} });
  assert.equal(event.schema_version, 2);
  assert.equal(event.producer_kind, 'chromium-extension');
  assert.equal(event.extension_version, require('../extensions/apollo-webflow-chromium/protocol.js').EXTENSION_VERSION);
  const caps = event.feature_capabilities;
  assert.ok(caps & 0b001, 'interaction grouping');
  assert.ok(caps & 0b010, 'component breakdown');
  assert.ok(caps & 0b100, 'transport timing');
});

test('the declared extension version matches the manifest', () => {
  const manifest = require('../extensions/apollo-webflow-chromium/manifest.json');
  const P = require('../extensions/apollo-webflow-chromium/protocol.js');
  assert.equal(manifest.version, P.EXTENSION_VERSION,
    'a drifting version would make the daemon report a wrong producer');
});

test('the content script gets the protocol helpers it depends on', () => {
  // content.js reads folding/percentile helpers from globalThis. An isolated
  // world does not inherit the service worker's importScripts, so protocol.js
  // must be injected alongside it — and before it.
  const fs = require('node:fs');
  const background = fs.readFileSync('extensions/apollo-webflow-chromium/background.js', 'utf8');
  const match = background.match(/js:\s*\[([^\]]+)\]/);
  assert.ok(match, 'registerContentScripts must declare a js list');
  const files = match[1].split(',').map((s) => s.trim().replace(/['"]/g, ''));
  assert.ok(files.includes('protocol.js'), 'protocol.js must be injected');
  assert.ok(
    files.indexOf('protocol.js') < files.indexOf('content.js'),
    'protocol.js must load before content.js',
  );

  const content = fs.readFileSync('extensions/apollo-webflow-chromium/content.js', 'utf8');
  assert.ok(
    content.includes('collectorReady'),
    'content.js must degrade quietly when the helpers are absent',
  );
});

test('the content script survives an orphaned extension context', () => {
  // Reloading the extension leaves content scripts running in open tabs;
  // sendMessage then throws synchronously and .catch() never sees it.
  const content = require('node:fs')
    .readFileSync('extensions/apollo-webflow-chromium/content.js', 'utf8');
  assert.ok(content.includes('chrome.runtime?.id'), 'must check the context is alive');
  assert.match(content, /try\s*\{[\s\S]*sendMessage[\s\S]*catch/, 'and guard the throw');
});
