'use strict';

importScripts('protocol.js');

const P = globalThis.ApolloWebFlowProtocol;
const NATIVE_HOST = 'com.eduardocortez.apollo_webflow';
const ROTATION_MS = 30 * 24 * 60 * 60 * 1000;
const browserSessionId = P.randomOpaqueId();
const tracker = new P.NavigationTracker(64);
const tabQueues = new Map();
let sequence = 0;
let nativePort = null;
// A fresh service worker global means MV3 tore the previous one down: the
// first event after that pays the cold start we are trying to measure.
let servedAnEvent = false;

function nextSequence() {
  sequence += 1;
  if (!Number.isSafeInteger(sequence)) sequence = 1;
  return sequence;
}

function postNative(event) {
  try {
    P.encodeBounded(event);
    if (!nativePort) {
      nativePort = chrome.runtime.connectNative(NATIVE_HOST);
      nativePort.onDisconnect.addListener(() => { nativePort = null; });
    }
    nativePort.postMessage(event);
  } catch (_) {
    nativePort = null;
  }
}

function enqueue(tabId, operation) {
  const prior = tabQueues.get(tabId) || Promise.resolve();
  const next = prior.then(operation, operation).catch(() => {});
  tabQueues.set(tabId, next);
  next.finally(() => {
    if (tabQueues.get(tabId) === next) tabQueues.delete(tabId);
  });
}

async function hmacSiteBucket(rawUrl) {
  let origin;
  try { origin = new URL(rawUrl).origin; } catch (_) { return undefined; }
  if (!origin || origin === 'null') return undefined;
  const now = Date.now();
  const stored = await chrome.storage.local.get(['siteSecret', 'siteSecretCreated']);
  let secret = stored.siteSecret;
  if (!Array.isArray(secret) || secret.length !== 32 || now - Number(stored.siteSecretCreated || 0) >= ROTATION_MS) {
    secret = Array.from(crypto.getRandomValues(new Uint8Array(32)));
    await chrome.storage.local.set({ siteSecret: secret, siteSecretCreated: now });
  }
  const key = await crypto.subtle.importKey(
    'raw', new Uint8Array(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'],
  );
  const digest = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(origin));
  origin = undefined;
  const bucket = Array.from(new Uint8Array(digest).slice(0, 16));
  return bucket.some((byte) => byte !== 0) ? bucket : undefined;
}

async function emit(record, phase, source, rawUrl, metrics = {}, transport = undefined) {
  if (!record) return;
  const siteBucket = rawUrl ? await hmacSiteBucket(rawUrl) : undefined;
  postNative(P.buildEvent({
    browserSessionId,
    tabSessionId: record.tabSessionId,
    navigationId: record.navigationId,
    sequence: nextSequence(),
    phase,
    source,
    siteBucket,
    metrics,
    transport: transport
      ? { ...transport, nativeMessageStartedAtMs: Date.now() }
      : undefined,
  }));
}

chrome.webNavigation.onBeforeNavigate.addListener((details) => {
  if (details.frameId !== 0) return;
  const started = tracker.start(details.tabId);
  enqueue(details.tabId, async () => {
    if (started.abandoned) await emit(started.abandoned, 'abandoned', 'extension-lifecycle');
    await emit(started, 'started', 'extension-lifecycle', details.url);
  });
});

for (const [event, phase] of [
  [chrome.webNavigation.onCommitted, 'committed'],
  [chrome.webNavigation.onDOMContentLoaded, 'dom-ready'],
  [chrome.webNavigation.onCompleted, 'loaded'],
]) {
  event.addListener((details) => {
    if (details.frameId !== 0) return;
    const record = tracker.get(details.tabId);
    enqueue(details.tabId, () => emit(record, phase, 'extension-lifecycle'));
  });
}

chrome.webNavigation.onErrorOccurred.addListener((details) => {
  if (details.frameId !== 0) return;
  const record = tracker.get(details.tabId);
  enqueue(details.tabId, () => emit(record, 'failed', 'extension-lifecycle', undefined, {
    errorClass: classifyError(details.error),
  }));
});

chrome.tabs.onRemoved.addListener((tabId) => {
  const record = tracker.close(tabId);
  enqueue(tabId, () => emit(record, 'abandoned', 'extension-lifecycle'));
});

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message?.type !== 'apollo-webflow-vitals' || !sender.tab) return;
  const record = tracker.get(sender.tab.id);
  const transport = {
    contentSendStartedAtMs: message.contentSendStartedAtMs,
    serviceWorkerReceivedAtMs: Date.now(),
    tabQueueDepth: tabQueues.size,
    serviceWorkerColdStart: !servedAnEvent,
  };
  servedAnEvent = true;
  enqueue(sender.tab.id, () =>
    emit(record, 'settled', 'extension-vitals', undefined, message.metrics, transport));
});

function classifyError(error) {
  const value = String(error || '').toLowerCase();
  if (value.includes('name_not_resolved')) return 'name-resolution';
  if (value.includes('connection')) return 'connection';
  if (value.includes('ssl') || value.includes('cert')) return 'tls';
  if (value.includes('timed_out')) return 'timeout';
  if (value.includes('aborted')) return 'cancelled';
  return value ? 'network' : 'unknown';
}

async function registerVitals() {
  const allowed = await chrome.permissions.contains({ origins: ['<all_urls>'] });
  if (!allowed) return;
  try {
    await chrome.scripting.unregisterContentScripts({ ids: ['apollo-webflow-vitals'] });
  } catch (_) {}
  await chrome.scripting.registerContentScripts([{
    id: 'apollo-webflow-vitals',
    matches: ['<all_urls>'],
    // protocol.js first: content.js reads the folding helpers from it, and an
    // isolated world does not inherit the service worker's importScripts.
    js: ['protocol.js', 'content.js'],
    runAt: 'document_start',
    persistAcrossSessions: true,
  }]);
}

chrome.action.onClicked.addListener(async () => {
  const granted = await chrome.permissions.request({ origins: ['<all_urls>'] });
  if (granted) await registerVitals();
});
chrome.runtime.onStartup.addListener(registerVitals);
chrome.runtime.onInstalled.addListener(registerVitals);
