// src/metal/engine/gr_dispatch.mm - validated GR Metal encoders.
#import <Metal/Metal.h>

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "embedding_encode_internal.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kCvecThreads = 256u;
constexpr std::uint32_t kDownThreads = 256u;
constexpr std::uint32_t kNormThreads = 256u;
constexpr std::uint32_t kUpThreads = 256u;
constexpr std::uint32_t kPostThreads = 256u;
constexpr std::uint32_t kDownSiluThreads = 256u;
constexpr std::uint32_t kPreGatedThreads = 256u;
constexpr std::uint32_t kRMSNormThreads = 1024u;
constexpr std::uint32_t kDownTileValues = 512u;
constexpr std::uint32_t kBroadcastThreads = 256u;
constexpr std::uint32_t kBroadcastGroupsX = HC_DIM / kBroadcastThreads;
constexpr std::uint64_t kBroadcastIndexCapacity =
    std::uint64_t{1u} << 32u;
constexpr std::uint32_t kBroadcastMaxRows =
    static_cast<std::uint32_t>(kBroadcastIndexCapacity / HC_DIM);

struct RMSNormParams {
    std::uint32_t n_cols;
    std::uint32_t gamma_period;
    float epsilon;
};
static_assert(sizeof(RMSNormParams) == 3u * sizeof(std::uint32_t));

struct GRCvecParams {
    std::uint32_t layer;
    std::uint32_t token_count;
    std::uint32_t mode;
    std::uint32_t write;
};
static_assert(sizeof(GRCvecParams) == 16u);

struct GRNormParams {
    std::uint32_t apply;
};
static_assert(sizeof(GRNormParams) == sizeof(std::uint32_t));

struct GRDownParams {
    std::uint32_t token_count;
};
static_assert(sizeof(GRDownParams) == sizeof(std::uint32_t));

struct GRUpParams {
    std::uint32_t token_count;
    std::uint32_t apply;
};
static_assert(sizeof(GRUpParams) == 2u * sizeof(std::uint32_t));

struct Binding {
    BufferRef buffer;
    NSUInteger index;
    std::string_view name;
};

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes,
                     std::size_t alignment, std::string_view name,
                     std::string_view family, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || buffer.offset % alignment != 0u) {
        error = std::string(family) + " " + std::string(name) +
                " buffer is missing, undersized or misaligned";
        return false;
    }
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = std::string(family) + " " + std::string(name) +
                " view exceeds its Metal allocation";
        return false;
    }
    return true;
}

bool validate_cpu_backing(BufferRef buffer, std::string_view name,
                          std::string_view family, std::string& error) {
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || native.contents == nullptr) {
        error = std::string(family) + " " + std::string(name) +
                " buffer has no CPU-visible backing metadata";
        return false;
    }
    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = std::string(family) + " " + std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }
    return true;
}

id<MTLCommandBuffer> validate_command_buffer(
    Context& context, CommandBuffer& command_buffer,
    std::string_view family, std::string& error) {
    void* const command_handle = command_buffer.unsafe_native_handle();
    if (command_handle == nullptr) {
        error = std::string(family) + " command buffer is unavailable";
        return nil;
    }
    if (!command_buffer.belongs_to(context)) {
        error = std::string(family) +
                " command buffer belongs to a different Context";
        return nil;
    }
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command_handle;
    if (native_command == nil ||
        native_command.status != MTLCommandBufferStatusNotEnqueued) {
        error = std::string(family) +
                " work must be encoded before command-buffer submission";
        return nil;
    }
    return native_command;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.length,
                               right.offset, right.length);
    const bool cpu_backing_overlap = address_ranges_overlap(
        reinterpret_cast<std::uintptr_t>(left.contents), left.length,
        reinterpret_cast<std::uintptr_t>(right.contents), right.length);
    const bool gpu_backing_overlap = address_ranges_overlap(
        left.gpu_address, left.length, right.gpu_address, right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

template <class Address>
bool broadcast_ranges_overlap(Address left, std::size_t left_bytes,
                              Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool broadcast_buffer_views_overlap(BufferRef left, BufferRef right) {
    const bool same_buffer_overlap =
        left.handle == right.handle &&
        broadcast_ranges_overlap(left.offset, left.length,
                                 right.offset, right.length);
    const bool cpu_backing_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        broadcast_ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left.length,
            reinterpret_cast<std::uintptr_t>(right.contents), right.length);
    const bool gpu_backing_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        broadcast_ranges_overlap(left.gpu_address, left.length,
                                 right.gpu_address, right.length);
    return same_buffer_overlap || cpu_backing_overlap || gpu_backing_overlap;
}

bool validate_pipeline(Context& context, const char* kernel_name,
                       std::string_view family, std::uint32_t threads,
                       std::uint32_t threadgroup_x,
                       std::uint32_t threadgroup_y,
                       std::uint32_t threadgroup_z, std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0;
    const KernelInfo* const manifest = kernel_manifest(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (std::string_view(manifest[index].name) == kernel_name) {
            info = &manifest[index];
            break;
        }
    }
    if (info == nullptr || info->threads != threads ||
        info->threadgroup_x != threadgroup_x ||
        info->threadgroup_y != threadgroup_y ||
        info->threadgroup_z != threadgroup_z) {
        error = std::string(family) + " threadgroup does not match its manifest";
        return false;
    }
    void* const pipeline_handle = context.pipeline(kernel_name);
    if (pipeline_handle == nullptr) {
        error = std::string(family) + " pipeline is unavailable";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_handle;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        error = std::string(family) + " threadgroup exceeds the device limit";
        return false;
    }
    return true;
}

bool validate_cvec_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_cvec", "GR cvec",
                             kCvecThreads, kCvecThreads, 1u, 1u, error);
}

bool validate_norm_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_norm", "GR norm",
                             kNormThreads, kNormThreads, 1u, 1u, error);
}

bool validate_down_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_down", "GR down",
                             kDownThreads, kDownThreads, 1u, 1u, error);
}

bool validate_up_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_up", "GR up",
                             kUpThreads, kUpThreads, 1u, 1u, error);
}

bool validate_post_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_post", "GR post",
                             kPostThreads, kPostThreads, 1u, 1u, error);
}

bool validate_down_silu_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_down_silu",
                             "GR down SiLU", kDownSiluThreads,
                             kDownSiluThreads, 1u, 1u, error);
}

bool validate_pre_gated_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_gr_pre_gated",
                             "GR pre-gated", kPreGatedThreads,
                             kPreGatedThreads, 1u, 1u, error);
}

bool validate_rms_norm_pipeline(Context& context, std::string& error) {
    return validate_pipeline(context, "strata_rms_norm_w",
                             "Weighted RMS norm", kRMSNormThreads,
                             kRMSNormThreads, 1u, 1u, error);
}

bool checked_float_bytes(std::uint32_t rows, std::uint32_t n_cols,
                         std::size_t& bytes) {
    const std::size_t maximum_elements =
        std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (n_cols != 0u &&
        static_cast<std::size_t>(rows) > maximum_elements / n_cols) {
        return false;
    }
    const std::size_t elements =
        static_cast<std::size_t>(rows) * n_cols;
    if (elements > maximum_elements) return false;
    bytes = elements * sizeof(float);
    return true;
}

bool validate_weighted_rms_norm(
    Context& context, std::uint32_t rows, std::uint32_t n_cols,
    std::uint32_t gamma_period, float epsilon,
    const WeightedRMSNormBuffers& buffers, std::size_t& input_bytes,
    std::size_t& gamma_bytes, std::string& error) {
    if (rows == 0u) {
        error = "Weighted RMS norm row count must be positive";
        return false;
    }
    if (n_cols == 0u) {
        error = "Weighted RMS norm column count must be positive";
        return false;
    }
    if (gamma_period == 0u) {
        error = "Weighted RMS norm gamma period must be positive";
        return false;
    }
    const bool supported_shape =
        (n_cols == 2560u &&
         (gamma_period == 1u || gamma_period == 4u)) ||
        (n_cols == 10240u && gamma_period == 1u);
    if (!supported_shape) {
        error = "Weighted RMS norm supports (2560, gp=1|4) or (10240, gp=1)";
        return false;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        error = "Weighted RMS norm epsilon must be finite and positive";
        return false;
    }
    if (!validate_rms_norm_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_rms_norm_w");
    if (pipeline == nil) {
        error = "Weighted RMS norm pipeline is unavailable";
        return false;
    }
    if (pipeline.staticThreadgroupMemoryLength >
        context.device_facts().max_threadgroup_memory_length) {
        error = "Weighted RMS norm threadgroup memory exceeds the device limit";
        return false;
    }
    if (!checked_float_bytes(rows, n_cols, input_bytes) ||
        !checked_float_bytes(gamma_period, n_cols, gamma_bytes)) {
        error = "Weighted RMS norm buffer size overflows addressable memory";
        return false;
    }

    const std::array bindings{
        buffers.input, buffers.gamma, buffers.output,
    };
    const std::array required_bytes{input_bytes, gamma_bytes, input_bytes};
    constexpr std::array<std::string_view, 3> names{
        "input", "gamma", "output",
    };
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index], required_bytes[index],
                             alignof(float), names[index],
                             "Weighted RMS norm", error)) {
            return false;
        }
    }

    const bool exact_in_place_view =
        buffers.input.handle == buffers.output.handle &&
        buffers.input.offset == buffers.output.offset &&
        buffers.input.length == buffers.output.length;
    if (buffer_views_overlap(buffers.input, buffers.output) &&
        !exact_in_place_view) {
        error = "Weighted RMS norm allows only exact input/output aliasing";
        return false;
    }
    if (buffer_views_overlap(buffers.gamma, buffers.output)) {
        error = "Weighted RMS norm output must not overlap gamma";
        return false;
    }
    return true;
}

}  // namespace

namespace detail {

bool prepare_gr_broadcast(Context& context, std::uint32_t rows,
                          const GRBroadcastBuffers& buffers,
                          GRBroadcastDispatch& dispatch,
                          std::string& error) {
    if (rows == 0u || rows > kBroadcastMaxRows) {
        error = "GR broadcast row count is outside the 32-bit shader index range";
        return false;
    }

    std::size_t embedding_bytes = 0u;
    std::size_t streams_bytes = 0u;
    const auto checked_uint_bytes = [](std::uint32_t row_count,
                                       std::uint32_t elements_per_row,
                                       std::size_t& bytes) {
        const std::size_t maximum_elements =
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t);
        if (elements_per_row != 0u &&
            static_cast<std::size_t>(row_count) >
                maximum_elements / elements_per_row) {
            return false;
        }
        const std::size_t elements =
            static_cast<std::size_t>(row_count) * elements_per_row;
        if (elements > maximum_elements) return false;
        bytes = elements * sizeof(std::uint32_t);
        return true;
    };
    if (!checked_uint_bytes(rows, N_EMBD, embedding_bytes) ||
        !checked_uint_bytes(rows, HC_DIM, streams_bytes)) {
        error = "GR broadcast buffer size overflows host address space";
        return false;
    }
    if (!validate_pipeline(context, "strata_gr_broadcast", "GR broadcast",
                           kBroadcastThreads, kBroadcastThreads, 1u, 1u,
                           error)) {
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_gr_broadcast");
    if (pipeline == nil) {
        error = "GR broadcast pipeline is unavailable";
        return false;
    }

    if (!validate_buffer(context, buffers.embedding, embedding_bytes,
                         alignof(std::uint32_t), "embedding", "GR broadcast",
                         error) ||
        !validate_cpu_backing(buffers.embedding, "embedding", "GR broadcast",
                              error) ||
        !validate_buffer(context, buffers.streams, streams_bytes,
                         alignof(std::uint32_t), "streams", "GR broadcast",
                         error) ||
        !validate_cpu_backing(buffers.streams, "streams", "GR broadcast",
                              error)) {
        return false;
    }
    if (broadcast_buffer_views_overlap(buffers.embedding, buffers.streams)) {
        error = "GR broadcast input and output views must not overlap";
        return false;
    }

    dispatch = {rows, buffers};
    error.clear();
    return true;
}

void encode_gr_broadcast(Context& context,
                         id<MTLComputeCommandEncoder> encoder,
                         const GRBroadcastDispatch& dispatch) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_gr_broadcast");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.embedding.handle
                offset:dispatch.buffers.embedding.offset
               atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dispatch.buffers.streams.handle
                offset:dispatch.buffers.streams.offset
               atIndex:1u];
    [encoder dispatchThreadgroups:
                 MTLSizeMake(kBroadcastGroupsX, dispatch.rows, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kBroadcastThreads, 1u, 1u)];
}

}  // namespace detail

bool encode_gr_broadcast(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows,
                         const GRBroadcastBuffers& buffers,
                         std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    detail::GRBroadcastDispatch dispatch;
    if (!detail::prepare_gr_broadcast(context, rows, buffers, dispatch,
                                      error)) {
        return false;
    }
    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR broadcast",
                                error);
    if (native_command == nil) return false;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> native_encoder =
            begin_compute_encoder(context, native_command);
        if (native_encoder == nil) {
            error = "cannot create GR broadcast encoder";
            return false;
        }
        native_encoder.label = @"GR stream broadcast";
        detail::encode_gr_broadcast(context, native_encoder, dispatch);
        end_compute_encoder(context, native_encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_broadcast(Context& context, Encoder& encoder,
                         std::uint32_t rows,
                         const GRBroadcastBuffers& buffers,
                         std::string& error) {
    error.clear();
    detail::GRBroadcastDispatch dispatch;
    if (!detail::prepare_gr_broadcast(context, rows, buffers, dispatch,
                                      error)) {
        return false;
    }
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t streams_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::array buffer_bindings{
        EncoderBufferBinding{buffers.embedding, 0u, embedding_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.streams, 1u, streams_bytes,
                             alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_broadcast", {kBroadcastGroupsX, rows, 1u},
        {kBroadcastThreads, 1u, 1u}, buffer_bindings, bytes,
        threadgroup_memory, error);
}

bool encode_gr_cvec(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, std::uint32_t layer,
                    GRControlMode mode, bool write,
                    const GRControlVectorBuffers& buffers,
                    std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR cvec row count must be in 1..8";
        return false;
    }
    if (layer >= N_LAYER) {
        error = "GR cvec layer must be in 0..47";
        return false;
    }
    if (mode != GRControlMode::project && mode != GRControlMode::add) {
        error = "GR cvec mode must be project or add";
        return false;
    }
    if (!validate_cvec_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_cvec");
    if (pipeline == nil) {
        error = "GR cvec pipeline is unavailable";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t direction_bytes =
        static_cast<std::size_t>(N_LAYER) * N_EMBD * sizeof(float);
    const std::size_t layer_scale_bytes =
        static_cast<std::size_t>(N_LAYER) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.direction, 1u, "direction"},
        Binding{buffers.layer_scale, 2u, "layer-scale"},
        Binding{buffers.enabled, 3u, "enabled"},
        Binding{buffers.block_out, 4u, "block-output"},
        Binding{buffers.inject_prev, 5u, "injection"},
    };
    const std::array required_bytes{
        residual_bytes, direction_bytes, layer_scale_bytes,
        sizeof(std::int32_t), block_bytes, inject_bytes,
    };
    const std::array alignments{
        alignof(float), alignof(float), alignof(float), alignof(std::int32_t),
        alignof(float), alignof(float),
    };
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR cvec", error)) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR cvec buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR cvec", error);
    if (native_command == nil) return false;

    const GRCvecParams params{
        layer, rows, static_cast<std::uint32_t>(mode), write ? 1u : 0u};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR cvec encoder";
            return false;
        }
        encoder.label = @"GR control vector";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder setBytes:&params length:sizeof(params) atIndex:6u];
        [encoder dispatchThreadgroups:MTLSizeMake(HC, rows, 1u)
                  threadsPerThreadgroup:MTLSizeMake(kCvecThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_cvec(Context& context, Encoder& encoder,
                    std::uint32_t rows, std::uint32_t layer,
                    GRControlMode mode, bool write,
                    const GRControlVectorBuffers& buffers,
                    std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR cvec row count must be in 1..8";
        return false;
    }
    if (layer >= N_LAYER) {
        error = "GR cvec layer must be in 0..47";
        return false;
    }
    if (mode != GRControlMode::project && mode != GRControlMode::add) {
        error = "GR cvec mode must be project or add";
        return false;
    }
    if (!validate_cvec_pipeline(context, error)) return false;
    if (context.pipeline("strata_gr_cvec") == nullptr) {
        error = "GR cvec pipeline is unavailable";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t direction_bytes =
        static_cast<std::size_t>(N_LAYER) * N_EMBD * sizeof(float);
    const std::size_t layer_scale_bytes =
        static_cast<std::size_t>(N_LAYER) * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.direction, 1u, "direction"},
        Binding{buffers.layer_scale, 2u, "layer-scale"},
        Binding{buffers.enabled, 3u, "enabled"},
        Binding{buffers.block_out, 4u, "block-output"},
        Binding{buffers.inject_prev, 5u, "injection"},
    };
    const std::array required_bytes{
        residual_bytes, direction_bytes, layer_scale_bytes,
        sizeof(std::int32_t), block_bytes, inject_bytes,
    };
    const std::array alignments{
        alignof(float), alignof(float), alignof(float), alignof(std::int32_t),
        alignof(float), alignof(float),
    };
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR cvec", error)) {
            return false;
        }
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR cvec buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    const GRCvecParams params{
        layer, rows, static_cast<std::uint32_t>(mode), write ? 1u : 0u};
    const std::array encoder_bindings{
        EncoderBufferBinding{buffers.residual, 0u, residual_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.direction, 1u, direction_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.layer_scale, 2u, layer_scale_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.enabled, 3u, sizeof(std::int32_t),
                             alignof(std::int32_t)},
        EncoderBufferBinding{buffers.block_out, 4u, block_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.inject_prev, 5u, inject_bytes,
                             alignof(float)},
    };
    const std::array bytes{
        EncoderBytesBinding{&params, sizeof(params), 6u},
    };
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_cvec", {HC, rows, 1u}, {kCvecThreads, 1u, 1u},
        encoder_bindings, bytes, threadgroup_memory, error);
}

bool encode_gr_norm(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, bool apply,
                    const GRNormBuffers& buffers, std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR norm row count must be in 1..8";
        return false;
    }
    if (!validate_norm_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_norm");
    if (pipeline == nil) {
        error = "GR norm pipeline is unavailable";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.norm, 2u, "norm"},
        Binding{buffers.inject_prev, 3u, "injection"},
        Binding{buffers.xn, 4u, "normalized-output"},
        Binding{buffers.rs_out, 5u, "scale-output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, norm_bytes, injection_bytes,
        residual_bytes, injection_bytes,
    };
    constexpr std::size_t kFloatAlignment = alignof(float);
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], kFloatAlignment,
                             bindings[index].name, "GR norm", error)) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR norm buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR norm", error);
    if (native_command == nil) return false;

    const GRNormParams params{apply ? 1u : 0u};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR norm encoder";
            return false;
        }
        encoder.label = @"GR normalization";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder setBytes:&params length:sizeof(params) atIndex:6u];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, HC, 1u)
                  threadsPerThreadgroup:MTLSizeMake(kNormThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_norm(Context& context, Encoder& encoder,
                    std::uint32_t rows, bool apply,
                    const GRNormBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR norm row count must be in 1..8";
        return false;
    }
    if (!validate_norm_pipeline(context, error)) return false;
    if (context.pipeline("strata_gr_norm") == nullptr) {
        error = "GR norm pipeline is unavailable";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.norm, 2u, "norm"},
        Binding{buffers.inject_prev, 3u, "injection"},
        Binding{buffers.xn, 4u, "normalized-output"},
        Binding{buffers.rs_out, 5u, "scale-output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, norm_bytes, injection_bytes,
        residual_bytes, injection_bytes,
    };
    constexpr std::size_t kFloatAlignment = alignof(float);
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], kFloatAlignment,
                             bindings[index].name, "GR norm", error)) {
            return false;
        }
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR norm buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    const GRNormParams params{apply ? 1u : 0u};
    const std::array encoder_bindings{
        EncoderBufferBinding{buffers.residual, 0u, residual_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.block_out, 1u, block_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.norm, 2u, norm_bytes, alignof(float)},
        EncoderBufferBinding{buffers.inject_prev, 3u, injection_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.xn, 4u, residual_bytes, alignof(float)},
        EncoderBufferBinding{buffers.rs_out, 5u, injection_bytes,
                             alignof(float)},
    };
    const std::array bytes{
        EncoderBytesBinding{&params, sizeof(params), 6u},
    };
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_norm", {rows, HC, 1u}, {kNormThreads, 1u, 1u},
        encoder_bindings, bytes, threadgroup_memory, error);
}

bool encode_gr_down(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, const GRDownBuffers& buffers,
                    std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR down row count must be in 1..8";
        return false;
    }
    if (!validate_down_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_down");
    if (pipeline == nil) {
        error = "GR down pipeline is unavailable";
        return false;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.xn, 0u, "normalized-input"},
        Binding{buffers.hc_down, 1u, "down-weights"},
        Binding{buffers.hc_inject, 2u, "injection-weights"},
        Binding{buffers.lo, 3u, "low-rank-output"},
        Binding{buffers.inj_out, 4u, "injection-output"},
    };
    const std::array required_bytes{
        input_bytes, down_weight_bytes, inject_weight_bytes,
        low_rank_bytes, inject_bytes,
    };
    const std::array<std::size_t, 5u> alignments{
        alignof(float), 16u, 16u, alignof(float), alignof(float)};
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR down", error)) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR down buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    const std::size_t threadgroup_memory_bytes =
        static_cast<std::size_t>(rows) * kDownTileValues * sizeof(float);
    const DeviceFacts facts = context.device_facts();
    if (threadgroup_memory_bytes % 16u != 0u ||
        threadgroup_memory_bytes > facts.max_threadgroup_memory_length) {
        error = "GR down threadgroup memory exceeds the device limit";
        return false;
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR down", error);
    if (native_command == nil) return false;

    const GRDownParams params{rows};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR down encoder";
            return false;
        }
        encoder.label = @"GR down and injection projections";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder setBytes:&params length:sizeof(params) atIndex:5u];
        [encoder setThreadgroupMemoryLength:threadgroup_memory_bytes atIndex:0u];
        [encoder dispatchThreadgroups:MTLSizeMake(41u, 1u, 1u)
                  threadsPerThreadgroup:MTLSizeMake(kDownThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_down(Context& context, Encoder& encoder, std::uint32_t rows,
                    const GRDownBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR down row count must be in 1..8";
        return false;
    }
    if (!validate_down_pipeline(context, error)) return false;
    if (context.pipeline("strata_gr_down") == nullptr) {
        error = "GR down pipeline is unavailable";
        return false;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t inject_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.xn, 0u, "normalized-input"},
        Binding{buffers.hc_down, 1u, "down-weights"},
        Binding{buffers.hc_inject, 2u, "injection-weights"},
        Binding{buffers.lo, 3u, "low-rank-output"},
        Binding{buffers.inj_out, 4u, "injection-output"},
    };
    const std::array required_bytes{
        input_bytes, down_weight_bytes, inject_weight_bytes,
        low_rank_bytes, inject_bytes,
    };
    const std::array<std::size_t, 5u> alignments{
        alignof(float), 16u, 16u, alignof(float), alignof(float)};
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR down", error)) {
            return false;
        }
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR down buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    const std::size_t threadgroup_memory_bytes =
        static_cast<std::size_t>(rows) * kDownTileValues * sizeof(float);
    if (threadgroup_memory_bytes % 16u != 0u ||
        threadgroup_memory_bytes >
            context.device_facts().max_threadgroup_memory_length) {
        error = "GR down threadgroup memory exceeds the device limit";
        return false;
    }

    const GRDownParams params{rows};
    const std::array encoder_bindings{
        EncoderBufferBinding{buffers.xn, 0u, input_bytes, alignof(float)},
        EncoderBufferBinding{buffers.hc_down, 1u, down_weight_bytes, 16u},
        EncoderBufferBinding{buffers.hc_inject, 2u, inject_weight_bytes, 16u},
        EncoderBufferBinding{buffers.lo, 3u, low_rank_bytes, alignof(float)},
        EncoderBufferBinding{buffers.inj_out, 4u, inject_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{
        &params, sizeof(params), 5u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, threadgroup_memory_bytes},
    };
    return encoder.dispatch(
        "strata_gr_down", {41u, 1u, 1u}, {kDownThreads, 1u, 1u},
        encoder_bindings, byte_bindings, threadgroup_memory, error);
}

bool encode_gr_up(Context& context, CommandBuffer& command_buffer,
                  std::uint32_t rows, bool apply,
                  const GRUpBuffers& buffers, std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR up row count must be in 1..8";
        return false;
    }
    if (!validate_up_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_up");
    if (pipeline == nil) {
        error = "GR up pipeline is unavailable";
        return false;
    }
    if (pipeline.staticThreadgroupMemoryLength >
        context.device_facts().max_threadgroup_memory_length) {
        error = "GR up threadgroup memory exceeds the device limit";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK *
        sizeof(std::uint16_t);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.inject_prev, 2u, "injection"},
        Binding{buffers.norm, 3u, "norm"},
        Binding{buffers.rs, 4u, "reciprocal-scale"},
        Binding{buffers.lo, 5u, "low-rank-input"},
        Binding{buffers.hc_up, 6u, "up-weights"},
        Binding{buffers.mixed, 7u, "mixed-output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, injection_bytes, norm_bytes,
        injection_bytes, low_rank_bytes, weight_bytes, block_bytes,
    };
    const std::array<std::size_t, 8u> alignments{
        alignof(float), alignof(float), alignof(float), alignof(float),
        alignof(float), alignof(float), 16u, alignof(float),
    };
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR up", error)) {
            return false;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR up buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR up", error);
    if (native_command == nil) return false;

    const GRUpParams params{rows, apply ? 1u : 0u};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR up encoder";
            return false;
        }
        encoder.label = @"GR up projection and mixer";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder setBytes:&params length:sizeof(params) atIndex:8u];
        [encoder dispatchThreadgroups:MTLSizeMake(160u, 1u, 1u)
                  threadsPerThreadgroup:MTLSizeMake(kUpThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_up(Context& context, Encoder& encoder, std::uint32_t rows,
                  bool apply, const GRUpBuffers& buffers,
                  std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR up row count must be in 1..8";
        return false;
    }
    if (!validate_up_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_up");
    if (pipeline == nil) {
        error = "GR up pipeline is unavailable";
        return false;
    }
    if (pipeline.staticThreadgroupMemoryLength >
        context.device_facts().max_threadgroup_memory_length) {
        error = "GR up threadgroup memory exceeds the device limit";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK *
        sizeof(std::uint16_t);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.inject_prev, 2u, "injection"},
        Binding{buffers.norm, 3u, "norm"},
        Binding{buffers.rs, 4u, "reciprocal-scale"},
        Binding{buffers.lo, 5u, "low-rank-input"},
        Binding{buffers.hc_up, 6u, "up-weights"},
        Binding{buffers.mixed, 7u, "mixed-output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, injection_bytes, norm_bytes,
        injection_bytes, low_rank_bytes, weight_bytes, block_bytes,
    };
    const std::array<std::size_t, 8u> alignments{
        alignof(float), alignof(float), alignof(float), alignof(float),
        alignof(float), alignof(float), 16u, alignof(float),
    };
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignments[index],
                             bindings[index].name, "GR up", error)) {
            return false;
        }
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR up buffer views overlap within one dispatch";
                return false;
            }
        }
    }

    const GRUpParams params{rows, apply ? 1u : 0u};
    const std::array encoder_bindings{
        EncoderBufferBinding{buffers.residual, 0u, residual_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.block_out, 1u, block_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.inject_prev, 2u, injection_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.norm, 3u, norm_bytes, alignof(float)},
        EncoderBufferBinding{buffers.rs, 4u, injection_bytes, alignof(float)},
        EncoderBufferBinding{buffers.lo, 5u, low_rank_bytes, alignof(float)},
        EncoderBufferBinding{buffers.hc_up, 6u, weight_bytes, 16u},
        EncoderBufferBinding{buffers.mixed, 7u, block_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{
        &params, sizeof(params), 8u};
    const std::array byte_bindings{params_binding};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_up", {160u, 1u, 1u}, {kUpThreads, 1u, 1u},
        encoder_bindings, byte_bindings, threadgroup_memory, error);
}

bool encode_gr_post(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, const GRPostBuffers& buffers,
                    std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR post row count must be in 1..8";
        return false;
    }
    if (!validate_post_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline("strata_gr_post");
    if (pipeline == nil) {
        error = "GR post pipeline is unavailable";
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.inject, 2u, "injection"},
        Binding{buffers.output, 3u, "output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, injection_bytes, residual_bytes,
    };
    constexpr std::size_t kFloatAlignment = alignof(float);
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], kFloatAlignment,
                             bindings[index].name, "GR post", error)) {
            return false;
        }
    }
    const BufferRef residual = bindings[0].buffer;
    const BufferRef output = bindings[3].buffer;
    const bool exact_in_place_view =
        residual.handle == output.handle &&
        residual.offset == output.offset && residual.length == output.length;
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (earlier == 0u && index == 3u && exact_in_place_view) continue;
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR post permits only an exact residual/output alias";
                return false;
            }
        }
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR post", error);
    if (native_command == nil) return false;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR post encoder";
            return false;
        }
        encoder.label = @"GR native post write";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(40u, rows, 1u)
                  threadsPerThreadgroup:MTLSizeMake(kPostThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_post(Context& context, Encoder& encoder,
                    std::uint32_t rows, const GRPostBuffers& buffers,
                    std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR post row count must be in 1..8";
        return false;
    }
    if (!validate_post_pipeline(context, error)) return false;

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::array bindings{
        Binding{buffers.residual, 0u, "residual"},
        Binding{buffers.block_out, 1u, "block-output"},
        Binding{buffers.inject, 2u, "injection"},
        Binding{buffers.output, 3u, "output"},
    };
    const std::array required_bytes{
        residual_bytes, block_bytes, injection_bytes, residual_bytes,
    };
    constexpr std::size_t kFloatAlignment = alignof(float);
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], kFloatAlignment,
                             bindings[index].name, "GR post", error)) {
            return false;
        }
    }
    const BufferRef residual = buffers.residual;
    const BufferRef output = buffers.output;
    const bool exact_in_place_view =
        residual.handle == output.handle &&
        residual.offset == output.offset && residual.length == output.length;
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (earlier == 0u && index == 3u && exact_in_place_view) continue;
            if (buffer_views_overlap(bindings[index].buffer,
                                     bindings[earlier].buffer)) {
                error = "GR post permits only an exact residual/output alias";
                return false;
            }
        }
    }

    const std::array buffer_bindings{
        EncoderBufferBinding{buffers.residual, 0u, residual_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.block_out, 1u, block_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.inject, 2u, injection_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.output, 3u, residual_bytes,
                             alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_post", {40u, rows, 1u}, {kPostThreads, 1u, 1u},
        buffer_bindings, bytes, threadgroup_memory, error);
}

bool encode_gr_down_silu(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows, BufferRef low_rank,
                         std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR down SiLU row count must be in 1..8";
        return false;
    }
    if (!validate_down_silu_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_gr_down_silu");
    if (pipeline == nil) {
        error = "GR down SiLU pipeline is unavailable";
        return false;
    }

    const std::uint32_t value_count = rows * HC_LOW_RANK;
    const std::size_t required_bytes =
        static_cast<std::size_t>(value_count) * sizeof(float);
    if (!validate_buffer(context, low_rank, required_bytes, alignof(float),
                         "low-rank", "GR down SiLU", error)) {
        return false;
    }
    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR down SiLU",
                                error);
    if (native_command == nil) return false;

    const std::uint32_t group_count =
        (value_count + kDownSiluThreads - 1u) / kDownSiluThreads;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR down SiLU encoder";
            return false;
        }
        encoder.label = @"GR down projection scale and SiLU";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)low_rank.handle
                    offset:low_rank.offset
                   atIndex:0u];
        [encoder setBytes:&value_count
                   length:sizeof(value_count)
                  atIndex:1u];
        [encoder dispatchThreadgroups:MTLSizeMake(group_count, 1u, 1u)
                  threadsPerThreadgroup:
                      MTLSizeMake(kDownSiluThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_pre_gated(Context& context, CommandBuffer& command_buffer,
                         std::uint32_t rows,
                         const GRPreGatedBuffers& buffers,
                         std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR pre-gated row count must be in 1..8";
        return false;
    }
    if (!validate_pre_gated_pipeline(context, error)) return false;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_gr_pre_gated");
    if (pipeline == nil) {
        error = "GR pre-gated pipeline is unavailable";
        return false;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::array bindings{
        Binding{buffers.xn, 0u, "normalized-input"},
        Binding{buffers.gate, 1u, "gate"},
        Binding{buffers.mixed, 2u, "mixed-output"},
    };
    const std::array required_bytes{
        input_bytes, input_bytes, output_bytes,
    };
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignof(float),
                             bindings[index].name, "GR pre-gated", error)) {
            return false;
        }
    }
    if (buffer_views_overlap(buffers.mixed, buffers.xn) ||
        buffer_views_overlap(buffers.mixed, buffers.gate)) {
        error = "GR pre-gated output must not overlap either input";
        return false;
    }

    id<MTLCommandBuffer> native_command =
        validate_command_buffer(context, command_buffer, "GR pre-gated",
                                error);
    if (native_command == nil) return false;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(context, native_command);
        if (encoder == nil) {
            error = "cannot create GR pre-gated encoder";
            return false;
        }
        encoder.label = @"GR head mixer pre-gated";
        [encoder setComputePipelineState:pipeline];
        for (const Binding& binding : bindings) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                        offset:binding.buffer.offset
                       atIndex:binding.index];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(10u, rows, 1u)
                  threadsPerThreadgroup:
                      MTLSizeMake(kPreGatedThreads, 1u, 1u)];
        end_compute_encoder(context, encoder);
    }
    error.clear();
    return true;
}

bool encode_gr_down_silu(Context& context, Encoder& encoder,
                         std::uint32_t rows, BufferRef low_rank,
                         std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR down SiLU row count must be in 1..8";
        return false;
    }
    if (!validate_down_silu_pipeline(context, error)) return false;
    if (context.pipeline("strata_gr_down_silu") == nullptr) {
        error = "GR down SiLU pipeline is unavailable";
        return false;
    }

    const std::uint32_t value_count = rows * HC_LOW_RANK;
    const std::size_t required_bytes =
        static_cast<std::size_t>(value_count) * sizeof(float);
    if (!validate_buffer(context, low_rank, required_bytes, alignof(float),
                         "low-rank", "GR down SiLU", error)) {
        return false;
    }

    const std::uint32_t group_count =
        (value_count + kDownSiluThreads - 1u) / kDownSiluThreads;
    const std::array buffer_bindings{
        EncoderBufferBinding{low_rank, 0u, required_bytes, alignof(float)},
    };
    const EncoderBytesBinding count_binding{
        &value_count, sizeof(value_count), 1u};
    const std::array byte_bindings{count_binding};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_down_silu", {group_count, 1u, 1u},
        {kDownSiluThreads, 1u, 1u}, buffer_bindings, byte_bindings,
        threadgroup_memory, error);
}

bool encode_gr_pre_gated(Context& context, Encoder& encoder,
                         std::uint32_t rows,
                         const GRPreGatedBuffers& buffers,
                         std::string& error) {
    error.clear();
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR pre-gated row count must be in 1..8";
        return false;
    }
    if (!validate_pre_gated_pipeline(context, error)) return false;
    if (context.pipeline("strata_gr_pre_gated") == nullptr) {
        error = "GR pre-gated pipeline is unavailable";
        return false;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::array bindings{
        Binding{buffers.xn, 0u, "normalized-input"},
        Binding{buffers.gate, 1u, "gate"},
        Binding{buffers.mixed, 2u, "mixed-output"},
    };
    const std::array required_bytes{
        input_bytes, input_bytes, output_bytes,
    };
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
        if (!validate_buffer(context, bindings[index].buffer,
                             required_bytes[index], alignof(float),
                             bindings[index].name, "GR pre-gated", error)) {
            return false;
        }
    }
    if (buffer_views_overlap(buffers.mixed, buffers.xn) ||
        buffer_views_overlap(buffers.mixed, buffers.gate)) {
        error = "GR pre-gated output must not overlap either input";
        return false;
    }

    const std::array encoder_bindings{
        EncoderBufferBinding{buffers.xn, 0u, input_bytes, alignof(float)},
        EncoderBufferBinding{buffers.gate, 1u, input_bytes, alignof(float)},
        EncoderBufferBinding{buffers.mixed, 2u, output_bytes, alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_gr_pre_gated", {10u, rows, 1u},
        {kPreGatedThreads, 1u, 1u}, encoder_bindings, bytes,
        threadgroup_memory, error);
}

bool encode_weighted_rms_norm(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const WeightedRMSNormBuffers& buffers, std::string& error) {
    error.clear();
    std::size_t input_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    if (!validate_weighted_rms_norm(
            context, rows, n_cols, gamma_period, epsilon, buffers,
            input_bytes, gamma_bytes, error)) {
        return false;
    }
    const RMSNormParams params{n_cols, gamma_period, epsilon};
    const std::array buffer_bindings{
        EncoderBufferBinding{
            buffers.input, 0u, input_bytes, alignof(float)},
        EncoderBufferBinding{
            buffers.gamma, 1u, gamma_bytes, alignof(float)},
        EncoderBufferBinding{
            buffers.output, 2u, input_bytes, alignof(float)},
    };
    const EncoderBytesBinding params_binding{&params, sizeof(params), 3u};
    const std::array byte_bindings{params_binding};
    return encoder.dispatch(
        "strata_rms_norm_w",
        {rows, 1u, 1u},
        {kRMSNormThreads, 1u, 1u}, buffer_bindings, byte_bindings, {}, error);
}

bool encode_weighted_rms_norm(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_cols, std::uint32_t gamma_period, float epsilon,
    const WeightedRMSNormBuffers& buffers, std::string& error) {
    error.clear();
    std::size_t input_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    if (!validate_weighted_rms_norm(
            context, rows, n_cols, gamma_period, epsilon, buffers,
            input_bytes, gamma_bytes, error)) {
        return false;
    }
    return Encoder::encode(
        context, command_buffer, "Weighted RMS norm",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_weighted_rms_norm(
                context, encoder, rows, n_cols, gamma_period, epsilon,
                buffers, encode_error);
        },
        error);
}

bool encode_gr_read(Context& context, Encoder& encoder, std::uint32_t rows,
                    bool apply, const GRReadBuffers& buffers,
                    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "GR read pipeline requires an ordered concurrent-group boundary";
        return false;
    }
    if (rows == 0u || rows > VERIFY_MAX_T) {
        error = "GR read row count must be in 1..8";
        return false;
    }

    const std::array views{
        buffers.residual, buffers.block_out, buffers.inject_prev, buffers.norm,
        buffers.normalized, buffers.rs, buffers.hc_down, buffers.hc_inject,
        buffers.lo, buffers.inject_out, buffers.hc_up, buffers.mixed,
    };
    for (std::size_t index = 0u; index < views.size(); ++index) {
        for (std::size_t earlier = 0u; earlier < index; ++earlier) {
            if (buffer_views_overlap(views[index], views[earlier])) {
                error = "GR read pipeline buffer views overlap across dispatches";
                return false;
            }
        }
    }

    const GRNormBuffers norm_buffers{
        buffers.residual, buffers.block_out, buffers.norm,
        buffers.inject_prev, buffers.normalized, buffers.rs};
    if (!encode_gr_norm(context, encoder, rows, apply, norm_buffers, error)) {
        return false;
    }

    const GRDownBuffers down_buffers{
        buffers.normalized, buffers.hc_down, buffers.hc_inject,
        buffers.lo, buffers.inject_out};
    if (!encode_gr_down(context, encoder, rows, down_buffers, error)) {
        return false;
    }

    const GRUpBuffers up_buffers{
        buffers.residual, buffers.block_out, buffers.inject_prev,
        buffers.norm, buffers.rs, buffers.lo, buffers.hc_up,
        buffers.mixed};
    return encode_gr_up(context, encoder, rows, apply, up_buffers, error);
}

bool encode_gr_read(Context& context, CommandBuffer& command_buffer,
                    std::uint32_t rows, bool apply,
                    const GRReadBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "GR read pipeline",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_gr_read(context, encoder, rows, apply, buffers,
                                  encode_error);
        },
        error);
}

}  // namespace strata::metal
