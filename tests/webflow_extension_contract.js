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
  assert.deepEqual(Object.keys(event).sort(), [
    'browser_session_id',
    'metrics',
    'navigation_id',
    'phase',
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
