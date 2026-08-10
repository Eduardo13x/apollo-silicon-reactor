#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <string.h>

typedef struct {
    float expected_gain;
    float sigma;
    uint32_t candidate_seed;
    uint32_t reserved;
} ApolloGpuCandidate;

typedef struct {
    uint32_t candidate_count;
    uint32_t samples_per_candidate;
    uint32_t seed;
    uint32_t reserved;
} ApolloGpuConfig;

@interface ApolloGpuImaginationContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@end

@implementation ApolloGpuImaginationContext
@end

static const char *kApolloGpuSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Candidate {
    float expected_gain;
    float sigma;
    uint candidate_seed;
    uint reserved;
};

struct Config {
    uint candidate_count;
    uint samples_per_candidate;
    uint seed;
    uint reserved;
};

inline uint mix_bits(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

inline float uniform01(uint value) {
    return float(mix_bits(value) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

kernel void apollo_imagine(
    device const Candidate *candidates [[buffer(0)]],
    device float *out_gain [[buffer(1)]],
    constant Config &config [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = config.candidate_count * config.samples_per_candidate;
    if (tid >= total || config.samples_per_candidate == 0) {
        return;
    }
    const uint candidate_index = tid / config.samples_per_candidate;
    const uint sample_index = tid - candidate_index * config.samples_per_candidate;
    const Candidate candidate = candidates[candidate_index];
    uint state = config.seed ^ candidate.candidate_seed ^
                 (sample_index * 0x9e3779b9u) ^ (candidate_index * 0x85ebca6bu);

    // Six uniforms form a cheap, bounded normal approximation. It keeps the
    // CPU reference and Metal kernel bit-for-bit comparable without trig.
    float noise = -3.0f;
    for (uint lane = 0; lane < 6; ++lane) {
        noise += uniform01(state + lane * 0x27d4eb2du);
    }
    out_gain[tid] = clamp(candidate.expected_gain + candidate.sigma * noise,
                          -1.0f, 1.0f);
}
)METAL";

static void apollo_copy_error(char *out, size_t capacity, NSString *message) {
    if (out == NULL || capacity == 0) {
        return;
    }
    const char *text = message.UTF8String;
    if (text == NULL) {
        text = "unknown Metal error";
    }
    const size_t length = strnlen(text, capacity - 1);
    memcpy(out, text, length);
    out[length] = '\0';
}

extern "C" void *apollo_gpu_imagination_create(char *error_out,
                                                  size_t error_capacity) {
    @autoreleasepool {
        if (error_out != NULL && error_capacity > 0) {
            error_out[0] = '\0';
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            apollo_copy_error(error_out, error_capacity, @"no default Metal device");
            return NULL;
        }
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:kApolloGpuSource];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:nil
                                                       error:&error];
        if (library == nil || error != nil) {
            apollo_copy_error(error_out, error_capacity,
                              error.localizedDescription ?: @"Metal library compilation failed");
            return NULL;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"apollo_imagine"];
        if (function == nil) {
            apollo_copy_error(error_out, error_capacity,
                              @"apollo_imagine kernel not found");
            return NULL;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (pipeline == nil || queue == nil || error != nil) {
            apollo_copy_error(error_out, error_capacity,
                              error.localizedDescription ?: @"Metal pipeline creation failed");
            return NULL;
        }
        ApolloGpuImaginationContext *context = [ApolloGpuImaginationContext new];
        context.device = device;
        context.queue = queue;
        context.pipeline = pipeline;
        return (__bridge_retained void *)context;
    }
}

extern "C" void apollo_gpu_imagination_destroy(void *raw_context) {
    if (raw_context == NULL) {
        return;
    }
    @autoreleasepool {
        CFBridgingRelease(raw_context);
    }
}

extern "C" int apollo_gpu_imagination_device_name(void *raw_context,
                                                    char *out,
                                                    size_t capacity) {
    if (raw_context == NULL || out == NULL || capacity == 0) {
        return -1;
    }
    @autoreleasepool {
        ApolloGpuImaginationContext *context =
            (__bridge ApolloGpuImaginationContext *)raw_context;
        const char *name = context.device.name.UTF8String;
        if (name == NULL) {
            return -2;
        }
        const size_t length = strnlen(name, capacity - 1);
        memcpy(out, name, length);
        out[length] = '\0';
        return 0;
    }
}

extern "C" int apollo_gpu_imagination_run(void *raw_context,
                                            const ApolloGpuCandidate *candidates,
                                            uint32_t candidate_count,
                                            uint32_t samples_per_candidate,
                                            uint32_t seed,
                                            float *out_gain,
                                            uint64_t *gpu_time_ns) {
    if (raw_context == NULL || candidates == NULL || out_gain == NULL ||
        candidate_count == 0 || samples_per_candidate == 0) {
        return -1;
    }
    const uint64_t total = (uint64_t)candidate_count * samples_per_candidate;
    if (total > UINT32_MAX || total > SIZE_MAX / sizeof(float)) {
        return -2;
    }

    @autoreleasepool {
        ApolloGpuImaginationContext *context =
            (__bridge ApolloGpuImaginationContext *)raw_context;
        const NSUInteger candidate_bytes =
            (NSUInteger)candidate_count * sizeof(ApolloGpuCandidate);
        const NSUInteger output_bytes = (NSUInteger)total * sizeof(float);
        id<MTLBuffer> candidate_buffer =
            [context.device newBufferWithBytes:candidates
                                        length:candidate_bytes
                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer =
            [context.device newBufferWithLength:output_bytes
                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (candidate_buffer == nil || output_buffer == nil ||
            command_buffer == nil || encoder == nil) {
            return -3;
        }

        ApolloGpuConfig config = {candidate_count, samples_per_candidate, seed, 0};
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:candidate_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBytes:&config length:sizeof(config) atIndex:2];

        const NSUInteger max_threads = context.pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger execution_width = context.pipeline.threadExecutionWidth;
        const NSUInteger group_width = MIN(MAX(execution_width, 1u),
                                           MIN(max_threads, 256u));
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return -4;
        }
        memcpy(out_gain, output_buffer.contents, output_bytes);
        if (gpu_time_ns != NULL) {
            const CFTimeInterval elapsed =
                command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            *gpu_time_ns = elapsed > 0.0 ? (uint64_t)(elapsed * 1000000000.0) : 0;
        }
        return 0;
    }
}
