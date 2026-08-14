# Apollo Universal NetworkFlow

**Status:** Approved extension to the WebFlow design

**Date:** 2026-08-14

**Goal:** Extend Apollo's interactive network acceleration beyond Chromium without intercepting, proxying, classifying, or persisting network content.

## Decision

Apollo uses two complementary evidence paths behind one actuation authority:

1. `WebFlow` remains the exact path for browsers with the optional extension.
2. `NetworkFlow` is the universal fallback for any foreground application, including Safari, Firefox, media clients, chat applications, cloud clients, launchers, package managers, and native apps.
3. Exact WebFlow evidence supersedes generic NetworkFlow evidence for the same foreground workload. The two paths never create duplicate leases.
4. Both paths emit advisory intents. The existing `ReflexBroker`, live process identity checks, effect ledger, TTL, budgets, rollback, and protected-process rules remain authoritative.

```text
exact browser lifecycle ---------------------> WebFlowIntent ---+
foreground PID + interaction + sockets + TCP -> NetworkIntent --+-> one arbiter -> existing reversible actions
```

## Universal Evidence

`NetworkFlowController` consumes a bounded per-cycle sample containing:

- current foreground PID identity availability;
- recent user interaction;
- whether the foreground process family owns open sockets;
- kernel TCP byte/connection deltas and freshness;
- exact WebFlow activity, when present;
- memory pressure, thermal, low-power, sleep, session, and kill-switch state.

Open sockets alone are insufficient because many applications keep idle connections. Global traffic alone is insufficient because background sync may be unrelated to the foreground app. A generic episode requires their conjunction with foreground interaction and fresh traffic. The evidence remains explicitly `inferred`; Apollo never claims per-process byte attribution it cannot measure.

Background transfers continue to receive Apollo's existing freeze/purge/media protections. They are not promoted to interactive QoS merely for holding sockets.

## Controller

- Start after a fresh qualifying sample.
- Initial generic TTL: 1,200 ms.
- Renew only while evidence remains fresh and constraints remain clear.
- Generic continuous hard cap: 12 seconds, followed by a 2-second cooldown.
- Exact WebFlow uses its existing 2-second initial TTL and 15-second cap.
- Exact WebFlow activity suppresses generic intent for that foreground workload.
- Target/session/sleep/wake changes close the generic episode immediately.
- Pressure >= 85%, serious thermal state, critical battery, or kill switch blocks admission and releases optional effects.

## Closed Action Catalog

NetworkFlow may request only the same reversible foreground lanes already owned by Apollo:

- interactive task QoS;
- bounded reversible `nice` fallback;
- release of Apollo-owned I/O restrictions;
- bounded family boost where existing safety classification allows it.

It cannot freeze, throttle, purge, kill, fetch, prefetch, rewrite packets, modify TLS, change DNS, install a proxy, or write sysctls. `SysctlGovernor` remains the only TCP sysctl authority.

## Privacy and Persistence

NetworkFlow does not observe destinations, ports, URLs, packet payloads, headers, account identifiers, or content. Its world observation contains only bounded numeric state, confidence, freshness, target availability, throughput bands, and counters. Raw generic episodes are not persisted.

## Metrics and Honesty

Dashboard counters separate `exact-web` and `inferred-network` proposals, admissions, applications, skips, vetoes, expirations, and failures. Generic evidence does not earn AIS credit by existing. Only a real actuator receipt and a matched improved outcome may reach causal Gold.

## Complexity and Overhead

- Foreground family socket probing is bounded to at most 16 already-known family PIDs.
- TCP counters use Apollo's direct kernel collector, never a subprocess.
- The controller performs O(1) state transitions with one active generic foreground episode.
- TCP sampling runs at a bounded one-second cadence and is reused by the existing governor and metrics.
- No full process scan, socket table walk, JSON persistence, or model wait is added to the cycle.

## Tests

- Idle sockets, traffic without interaction, and interaction without traffic never create an intent.
- Any foreground app can create an inferred intent when all evidence agrees.
- Exact WebFlow suppresses the generic path without cancelling the exact lease.
- Stale samples, PID/session changes, sleep, pressure, thermal, battery, and kill switch release or block.
- TTL, hard cap, cooldown, no duplicate actuation, and legacy behavior are deterministic.
- The module contains no content/network interception APIs and no direct actuator calls.
