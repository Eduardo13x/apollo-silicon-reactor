/*
 * ioreport_bridge.c — C shim for IOReport private framework.
 *
 * IOReport is the private IOKit API used internally by powermetrics.
 * It gives direct access to:
 *   - Per-cluster CPU utilization (P-cores vs E-cores separately)
 *   - GPU utilization and performance states
 *   - ANE (Neural Engine) utilization
 *   - Per-component power in milliwatts (CPU, GPU, DRAM, package)
 *
 * IOReportIterate uses Objective-C blocks, which cannot be created
 * directly in Rust. This bridge wraps it with a plain C callback.
 *
 * Linking: -lIOReport (/usr/lib/libIOReport.dylib) + CoreFoundation.
 *
 * References:
 *   - asitop (open source, Python): github.com/tlkh/asitop
 *   - Stats.app (open source, Swift): github.com/exelban/stats
 *   - XNU source: powermetrics uses the same API set
 */

#include <CoreFoundation/CoreFoundation.h>
#include <stdint.h>
#include <string.h>

/* ── IOReport opaque types ─────────────────────────────────────────────────── */

typedef void *IOReportSampleRef;
typedef void *IOReportSubscriptionRef;

/* ── IOReport function declarations (from /usr/lib/libIOReport.dylib) ──────── */

extern IOReportSubscriptionRef IOReportCreateSubscription(
    void *ctx,
    CFMutableDictionaryRef channels,
    CFMutableDictionaryRef *subChannels,
    uint64_t channel_id,
    CFDictionaryRef options);

extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(
    CFStringRef group, CFStringRef subgroup,
    void *a, void *b, void *c);

extern CFMutableDictionaryRef IOReportMergeChannels(
    CFDictionaryRef a, CFDictionaryRef b, void *c);

extern CFDictionaryRef IOReportCreateSamples(
    IOReportSubscriptionRef sub,
    CFMutableDictionaryRef channels,
    CFErrorRef *error);

extern CFDictionaryRef IOReportCreateSamplesDelta(
    CFDictionaryRef s1, CFDictionaryRef s2, CFErrorRef *error);

extern void IOReportIterate(
    CFDictionaryRef samples,
    int (^callback)(IOReportSampleRef));

extern CFStringRef IOReportChannelGetChannelName(IOReportSampleRef sample);
extern CFStringRef IOReportChannelGetDriverName(IOReportSampleRef sample);
extern int64_t     IOReportSimpleGetIntegerValue(IOReportSampleRef sample, int *err);
extern int32_t     IOReportStateGetCount(IOReportSampleRef sample);
extern double      IOReportStateGetDutyCycle(IOReportSampleRef sample, int32_t idx);
extern CFStringRef IOReportStateGetNameForIndex(IOReportSampleRef sample, int32_t idx);
extern long        IOReportGetChannelCount(CFDictionaryRef channels);

/* ── Apollo bridge types ────────────────────────────────────────────────────── */

/* Maximum performance states per channel (Apple Silicon has ≤16) */
#define APOLLO_MAX_STATES 32

typedef struct {
    char    driver[128];
    char    channel[256];
    int64_t value;          /* integer value (non-state channels) */
    int32_t state_count;    /* >0 means duty_cycles/state_names are valid */
    double  duty_cycles[APOLLO_MAX_STATES];
    char    state_names[APOLLO_MAX_STATES][64];
} ApolloIOReportChannel;

typedef void (*ApolloIOReportCallback)(const ApolloIOReportChannel *ch, void *ctx);

/* ── Bridge functions ────────────────────────────────────────────────────────── */

/*
 * apollo_ioreport_iterate — wraps IOReportIterate (block) with a plain C callback.
 *
 * Extracts channel name, driver, value, and per-state duty cycles.
 * Called once per delta sample to populate ApolloIOReportChannel.
 */
void apollo_ioreport_iterate(
    CFDictionaryRef samples,
    ApolloIOReportCallback cb,
    void *ctx)
{
    if (!samples || !cb) return;

    IOReportIterate(samples, ^int(IOReportSampleRef sample) {
        ApolloIOReportChannel ch;
        memset(&ch, 0, sizeof(ch));

        CFStringRef driver_ref  = IOReportChannelGetDriverName(sample);
        CFStringRef channel_ref = IOReportChannelGetChannelName(sample);

        if (driver_ref) {
            CFStringGetCString(driver_ref, ch.driver, sizeof(ch.driver),
                               kCFStringEncodingUTF8);
        }
        if (channel_ref) {
            CFStringGetCString(channel_ref, ch.channel, sizeof(ch.channel),
                               kCFStringEncodingUTF8);
        }

        ch.state_count = IOReportStateGetCount(sample);

        if (ch.state_count > 0) {
            /* Performance state distribution (e.g., CPU cluster freq states) */
            int32_t n = ch.state_count < APOLLO_MAX_STATES
                      ? ch.state_count : APOLLO_MAX_STATES;
            for (int32_t i = 0; i < n; i++) {
                ch.duty_cycles[i] = IOReportStateGetDutyCycle(sample, i);
                CFStringRef name = IOReportStateGetNameForIndex(sample, i);
                if (name) {
                    CFStringGetCString(name, ch.state_names[i],
                                       sizeof(ch.state_names[i]),
                                       kCFStringEncodingUTF8);
                }
            }
        } else {
            /* Simple integer value (e.g., mW reading) */
            int err = 0;
            ch.value = IOReportSimpleGetIntegerValue(sample, &err);
        }

        cb(&ch, ctx);
        return 0; /* kIOReportIterOk */
    });
}

/*
 * apollo_ioreport_create_subscription — create subscription for given group names.
 *
 * group_names: array of C strings (e.g., "CPU Stats", "Energy Model")
 * group_count: number of groups
 * out_channels: receives the merged CFMutableDictionaryRef (caller must CFRelease)
 *
 * Returns IOReportSubscriptionRef or NULL on failure.
 */
IOReportSubscriptionRef apollo_ioreport_create_subscription(
    CFMutableDictionaryRef *out_channels,
    const char **group_names,
    int group_count)
{
    CFMutableDictionaryRef all_channels = NULL;

    for (int i = 0; i < group_count; i++) {
        CFStringRef group = CFStringCreateWithCString(
            kCFAllocatorDefault, group_names[i], kCFStringEncodingUTF8);
        if (!group) continue;

        CFMutableDictionaryRef ch =
            IOReportCopyChannelsInGroup(group, NULL, NULL, NULL, NULL);
        CFRelease(group);

        if (!ch) continue;

        if (all_channels == NULL) {
            all_channels = ch;
        } else {
            CFMutableDictionaryRef merged =
                IOReportMergeChannels(all_channels, ch, NULL);
            CFRelease(all_channels);
            CFRelease(ch);
            all_channels = merged;
        }
    }

    if (!all_channels) {
        *out_channels = NULL;
        return NULL;
    }

    CFMutableDictionaryRef sub_channels = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(
        NULL, all_channels, &sub_channels, 0, NULL);

    *out_channels = all_channels;
    return sub;
}

/*
 * apollo_ioreport_sample — take an instantaneous sample.
 * Returns CFDictionaryRef that must be CFRelease'd by caller.
 */
CFDictionaryRef apollo_ioreport_sample(
    IOReportSubscriptionRef sub,
    CFMutableDictionaryRef channels)
{
    return IOReportCreateSamples(sub, channels, NULL);
}

/*
 * apollo_ioreport_delta — compute delta between two samples.
 * Returns CFDictionaryRef that must be CFRelease'd by caller.
 */
CFDictionaryRef apollo_ioreport_delta(CFDictionaryRef s1, CFDictionaryRef s2)
{
    if (!s1 || !s2) return NULL;
    return IOReportCreateSamplesDelta(s1, s2, NULL);
}

/*
 * apollo_ioreport_release — release a CFTypeRef returned by this bridge.
 */
void apollo_ioreport_release(void *ref)
{
    if (ref) CFRelease((CFTypeRef)ref);
}
