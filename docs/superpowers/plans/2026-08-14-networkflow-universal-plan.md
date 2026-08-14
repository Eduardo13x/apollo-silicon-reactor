# Apollo Universal NetworkFlow Implementation Plan

**Goal:** Add a bounded all-app inferred network path beside exact browser WebFlow and merge both at Apollo's existing acceleration lease authority.

## Tasks

1. Add `network_flow.rs` with pure controller contracts, confidence, TTL, hard-cap, cooldown, constraints, and tests.
2. Expose a fresh one-second direct-kernel TCP sample from `NetworkMonitor` and test its age/throughput semantics.
3. Add `networkflow_tick.rs` to combine foreground identity, interaction, bounded family socket evidence, TCP activity, and exact-WebFlow suppression.
4. Publish a bounded numeric `WorldStateSnapshot.network` observation and Event Mesh summary without app names or destinations.
5. Extend the acceleration lease selector with `WebNavigation` and `NetworkActivity` hints. Preserve one active lease and all existing safety/rollback gates.
6. Add separate exact/inferred counters and compact dashboard fields.
7. Run focused controller, world-state, reflex, daemon, and legacy tests before the full release build and guarded deploy.

## Acceptance

- Safari or any native foreground app with interaction, sockets, and fresh traffic can produce a generic intent.
- An idle Electron app with sockets cannot produce an intent.
- Exact Chromium evidence takes precedence and cannot double-apply.
- No new proxy, packet, URL, DNS, TLS, prefetch, freeze, throttle, purge, kill, or direct sysctl path exists.
- Main-cycle p95 remains below 75 ms and idle CPU regression remains below one percentage point.
