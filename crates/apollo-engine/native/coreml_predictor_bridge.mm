#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

static const uint32_t kApolloCpuAndNeuralEngine = 1;
static const uint32_t kApolloAll = 2;
static const uint32_t kApolloCpuOnly = 3;
static const uint32_t kApolloMaxFeatureCount = 256;
static const size_t kApolloReasonCapacity = 512;
static NSString *const kApolloInputName = @"temporal_features";
static NSString *const kApolloSchemaHashKey = @"apollo_schema_hash";
static NSString *const kApolloModelHashKey = @"apollo_model_hash";

/// How much is known about where an inference actually executed.
/// `kApolloAneUnsupported` is the honest state on current macOS: Core ML
/// accepts a compute-unit request at load and then routes each inference
/// itself without publishing the unit it chose.
enum : uint32_t {
    kApolloAneUnsupported = 0,
    kApolloAneUnavailable = 1,
    kApolloAneMeasuredIdle = 2,
    kApolloAneMeasuredActive = 3,
};

typedef struct {
    uint32_t requested_backend;
    /// Compute units Core ML accepted at model load. A configuration, not an
    /// observation of where work ran.
    uint32_t configured_backend;
    uint32_t model_available;
    /// One of the kApolloAne* codes above. Never a bool: "not implemented"
    /// must not share a value with "measured, and the ANE stayed idle".
    uint32_t ane_observation;
    char reason[kApolloReasonCapacity];
} ApolloCoreMlStatus;

@interface ApolloCoreMlContext : NSObject {
@public
    std::mutex inference_mutex;
}
@property(nonatomic, strong) MLModel *model;
@property(nonatomic, assign) uint32_t configured_backend;
@property(nonatomic, assign) uint32_t feature_count;
@end

@implementation ApolloCoreMlContext
@end

static void apollo_copy_reason(ApolloCoreMlStatus *status, NSString *message) {
    if (status == nullptr || message == nil) {
        return;
    }
    const char *text = message.UTF8String;
    if (text == nullptr) {
        text = "unknown Core ML error";
    }
    const size_t length = strnlen(text, kApolloReasonCapacity - 1);
    memcpy(status->reason, text, length);
    status->reason[length] = '\0';
}

static NSString *apollo_error_message(NSError *error, NSString *fallback) {
    if (error != nil && error.localizedDescription.length > 0) {
        return error.localizedDescription;
    }
    return fallback;
}

static bool apollo_read_hash(id value, uint64_t *hash) {
    if (value == nil || hash == nullptr) {
        return false;
    }
    if ([value isKindOfClass:[NSNumber class]]) {
        *hash = [(NSNumber *)value unsignedLongLongValue];
        return true;
    }
    if (![value isKindOfClass:[NSString class]]) {
        return false;
    }
    const char *text = [(NSString *)value UTF8String];
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *hash = static_cast<uint64_t>(parsed);
    return true;
}

static bool apollo_shape_matches(NSArray<NSNumber *> *shape, uint32_t expected_count) {
    if (shape == nil || shape.count == 0) {
        return false;
    }
    uint64_t product = 1;
    for (NSNumber *dimension in shape) {
        const NSInteger value = dimension.integerValue;
        if (value <= 0 || product > UINT64_MAX / static_cast<uint64_t>(value)) {
            return false;
        }
        product *= static_cast<uint64_t>(value);
    }
    return product == expected_count;
}

static bool apollo_validate_model(MLModel *model,
                                  uint64_t expected_schema_hash,
                                  uint64_t expected_model_hash,
                                  uint32_t feature_count,
                                  NSString **failure) {
    if (model == nil) {
        *failure = @"Core ML returned no model";
        return false;
    }

    NSDictionary *metadata = model.modelDescription.metadata;
    NSDictionary *user_defined = metadata[MLModelCreatorDefinedKey];
    uint64_t schema_hash = 0;
    uint64_t model_hash = 0;
    if (![user_defined isKindOfClass:[NSDictionary class]] ||
        !apollo_read_hash(user_defined[kApolloSchemaHashKey], &schema_hash) ||
        schema_hash != expected_schema_hash) {
        *failure = [NSString stringWithFormat:
            @"Core ML schema hash mismatch (expected 0x%016llx)",
            static_cast<unsigned long long>(expected_schema_hash)];
        return false;
    }
    if (!apollo_read_hash(user_defined[kApolloModelHashKey], &model_hash) ||
        model_hash != expected_model_hash) {
        *failure = [NSString stringWithFormat:
            @"Core ML model hash mismatch (expected 0x%016llx)",
            static_cast<unsigned long long>(expected_model_hash)];
        return false;
    }

    MLFeatureDescription *input = model.modelDescription.inputDescriptionsByName[kApolloInputName];
    if (input == nil || input.type != MLFeatureTypeMultiArray ||
        !apollo_shape_matches(input.multiArrayConstraint.shape, feature_count)) {
        *failure = [NSString stringWithFormat:
            @"Core ML input schema mismatch for %@ (%u features)", kApolloInputName, feature_count];
        return false;
    }

    NSArray<NSString *> *output_names = @[@"load", @"transition", @"pressure", @"p95"];
    for (NSString *name in output_names) {
        if (model.modelDescription.outputDescriptionsByName[name] == nil) {
            *failure = [NSString stringWithFormat:@"Core ML output schema missing %@", name];
            return false;
        }
    }
    return true;
}

static NSURL *apollo_model_url(NSString **failure) {
    NSString *path = NSProcessInfo.processInfo.environment[@"APOLLO_COREML_MODEL_PATH"];
    if (path == nil || path.length == 0) {
        path = @"/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel";
    }
    if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
        *failure = [NSString stringWithFormat:@"Core ML model path does not exist: %@", path];
        return nil;
    }
    return [NSURL fileURLWithPath:path];
}

static MLModel *apollo_load_model(NSURL *url, MLComputeUnits units, NSError **error) {
    NSURL *compiled_url = url;
    if ([url.pathExtension.lowercaseString isEqualToString:@"mlmodel"]) {
        compiled_url = [MLModel compileModelAtURL:url error:error];
        if (compiled_url == nil) {
            return nil;
        }
    }
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = units;
    return [MLModel modelWithContentsOfURL:compiled_url configuration:configuration error:error];
}

extern "C" void *apollo_coreml_create(uint64_t expected_schema_hash,
                                        uint64_t expected_model_hash,
                                        uint32_t feature_count,
                                        ApolloCoreMlStatus *status) {
    if (status != nullptr) {
        memset(status, 0, sizeof(*status));
        status->requested_backend = kApolloCpuAndNeuralEngine;
    }

    if (feature_count == 0 || feature_count > kApolloMaxFeatureCount) {
        apollo_copy_reason(status, @"Core ML feature count exceeds the versioned 256-feature bound");
        return nullptr;
    }

    @autoreleasepool {
        NSString *failure = nil;
        NSURL *url = apollo_model_url(&failure);
        if (url == nil) {
            apollo_copy_reason(status, failure);
            return nullptr;
        }

        const MLComputeUnits requested_units[] = {
            MLComputeUnitsCPUAndNeuralEngine,
            MLComputeUnitsAll,
            MLComputeUnitsCPUOnly,
        };
        const uint32_t backend_values[] = {
            kApolloCpuAndNeuralEngine,
            kApolloAll,
            kApolloCpuOnly,
        };
        for (size_t index = 0; index < sizeof(requested_units) / sizeof(requested_units[0]); ++index) {
            NSError *error = nil;
            MLModel *model = apollo_load_model(url, requested_units[index], &error);
            NSString *validation_failure = nil;
            if (model != nil && apollo_validate_model(model,
                                                       expected_schema_hash,
                                                       expected_model_hash,
                                                       feature_count,
                                                       &validation_failure)) {
                ApolloCoreMlContext *context = [ApolloCoreMlContext new];
                context.model = model;
                context.configured_backend = backend_values[index];
                context.feature_count = feature_count;
                if (status != nullptr) {
                    // What Core ML accepted, not where inference will run.
                    status->configured_backend = backend_values[index];
                    status->model_available = 1;
                    // A compute-unit request is not measured proof of ANE use,
                    // and Core ML exposes no per-inference dispatch target.
                    status->ane_observation = kApolloAneUnsupported;
                }
                return (__bridge_retained void *)context;
            }
            failure = validation_failure != nil
                ? validation_failure
                : apollo_error_message(error, @"Core ML model could not be loaded");
        }
        apollo_copy_reason(status, failure);
        return nullptr;
    }
}

extern "C" void apollo_coreml_destroy(void *raw_context) {
    if (raw_context == nullptr) {
        return;
    }
    (void)CFBridgingRelease(raw_context);
}

static bool apollo_feature_scalar(MLFeatureValue *value, float *output) {
    if (value == nil || output == nullptr) {
        return false;
    }
    double scalar = 0.0;
    switch (value.type) {
        case MLFeatureTypeDouble:
            scalar = value.doubleValue;
            break;
        case MLFeatureTypeInt64:
            scalar = static_cast<double>(value.int64Value);
            break;
        case MLFeatureTypeMultiArray: {
            MLMultiArray *array = value.multiArrayValue;
            if (array == nil || array.count != 1 || array.dataPointer == nullptr) {
                return false;
            }
            if (array.dataType == MLMultiArrayDataTypeFloat32) {
                scalar = static_cast<double>(*static_cast<float *>(array.dataPointer));
            } else if (array.dataType == MLMultiArrayDataTypeDouble) {
                scalar = *static_cast<double *>(array.dataPointer);
            } else {
                return false;
            }
            break;
        }
        default:
            return false;
    }
    if (!std::isfinite(scalar)) {
        return false;
    }
    *output = static_cast<float>(std::fmin(1.0, std::fmax(0.0, scalar)));
    return std::isfinite(*output);
}

extern "C" int32_t apollo_coreml_predict(void *raw_context,
                                           const float *features,
                                           uint32_t feature_count,
                                           float *output) {
    if (raw_context == nullptr || features == nullptr || output == nullptr || feature_count == 0 ||
        feature_count > kApolloMaxFeatureCount) {
        return -1;
    }
    ApolloCoreMlContext *context = (__bridge ApolloCoreMlContext *)raw_context;
    if (feature_count != context.feature_count) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(context->inference_mutex);

    @autoreleasepool {
        for (uint32_t index = 0; index < feature_count; ++index) {
            if (!std::isfinite(features[index])) {
                return -1;
            }
        }
        NSError *error = nil;
        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithShape:@[@(feature_count)]
            dataType:MLMultiArrayDataTypeFloat32
            error:&error];
        if (input_array == nil || input_array.dataPointer == nullptr) {
            return -2;
        }
        memcpy(input_array.dataPointer, features, sizeof(float) * feature_count);
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{kApolloInputName: input_value}
            error:&error];
        if (provider == nil) {
            return -3;
        }
        id<MLFeatureProvider> prediction = [context.model predictionFromFeatures:provider error:&error];
        if (prediction == nil) {
            return -4;
        }

        NSArray<NSString *> *output_names = @[@"load", @"transition", @"pressure", @"p95"];
        float bounded[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (NSUInteger index = 0; index < output_names.count; ++index) {
            MLFeatureValue *value = [prediction featureValueForName:output_names[index]];
            if (!apollo_feature_scalar(value, &bounded[index])) {
                return -5;
            }
        }
        memcpy(output, bounded, sizeof(bounded));
        return 0;
    }
}
