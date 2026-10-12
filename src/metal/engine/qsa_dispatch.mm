// Validated QSA kernel encoders.
#import <Metal/Metal.h>

#include "strata/metal/qsa_dispatch.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"
#include "metal_fence.hpp"
#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kFwhtThreads = 256u;
constexpr std::uint32_t kKvAppendThreads = 256u;
static_assert(
    static_cast<std::uint64_t>(kPrefillMaxShaderRows) * N_HEAD_KV <=
    std::numeric_limits<std::uint32_t>::max());
constexpr std::uint32_t kIndexerSnapshotThreads = 384u;
constexpr std::uint32_t kIndexerAppendThreads = 128u;
constexpr std::uint32_t kSelectionScoresThreads = 256u;
constexpr std::uint32_t kSelectionTopkRegThreads = 1024u;
constexpr std::uint32_t kSelectionTopkMemThreads = 256u;
constexpr std::uint32_t kPrefillSelectionGroupRows = 256u;
constexpr std::uint32_t kQsaGateThreads = 256u;
constexpr std::uint32_t kQsaAttentionThreads = 256u;
constexpr std::uint32_t kQsaIndexerLayers = INDEXER_SNAPSHOT_LAYERS;
constexpr std::uint32_t kSelectionTopkRegMaxBlocks = 33792u;
constexpr std::uint32_t kQsaAttentionChunkCells = 64u;
constexpr std::uint32_t kQsaAttentionMaxCells = 262144u;
constexpr std::size_t kIndexerTailFloats = 3u * INDEXER_HEAD_DIM;
constexpr std::size_t kFwhtRowBytes = 256u * sizeof(float);
constexpr std::size_t kRopeCellBytes = 32u * 2u * sizeof(float);
constexpr std::size_t kRopeAlignment = 2u * alignof(float);
constexpr std::size_t kSelectionVectorAlignment = 16u;

struct Params {
    std::uint32_t token_rows;
    std::uint32_t mode;
};
static_assert(sizeof(Params) == 2u * sizeof(std::uint32_t));

struct FwhtParams {
    std::uint32_t rows;
};
static_assert(sizeof(FwhtParams) == sizeof(std::uint32_t));

struct KvAppendF16Params {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
};
static_assert(sizeof(KvAppendF16Params) == 2u * sizeof(std::uint32_t));

struct KvAppendI8Params {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
};
static_assert(sizeof(KvAppendI8Params) == 2u * sizeof(std::uint32_t));

struct KvAppendQ4Params {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
};
static_assert(sizeof(KvAppendQ4Params) == 2u * sizeof(std::uint32_t));

struct KvAppendK8V4Params {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
};
static_assert(sizeof(KvAppendK8V4Params) == 2u * sizeof(std::uint32_t));

struct IndexerSnapshotParams {
    std::uint64_t tail_layer_stride_floats;
    std::uint64_t block_pos_layer_stride_ints;
};
static_assert(sizeof(IndexerSnapshotParams) == 2u * sizeof(std::uint64_t));

struct IndexerAppendParams {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
    std::uint32_t mode;
    std::uint32_t qsa_layer_index;
    std::int32_t position0;
    float rope_zero_cos;
    float rope_zero_sin;
};
static_assert(sizeof(IndexerAppendParams) == 7u * sizeof(std::uint32_t));

struct PrefillIndexerAppendParams {
    PrefillParams prefill;
    float rope_zero_cos;
    float rope_zero_sin;
};
static_assert(sizeof(PrefillIndexerAppendParams) == 6u * sizeof(std::uint32_t));

struct SelectionScoresParams {
    std::uint32_t token_rows;
    std::uint32_t max_blocks;
    std::uint32_t grid_count_x;
};
static_assert(sizeof(SelectionScoresParams) == 3u * sizeof(std::uint32_t));

struct SelectionTopkParams {
    std::uint32_t token_rows;
    std::uint32_t max_blocks;
};
static_assert(sizeof(SelectionTopkParams) == 2u * sizeof(std::uint32_t));

struct QsaGateParams {
    std::uint32_t token_rows;
};
static_assert(sizeof(QsaGateParams) == sizeof(std::uint32_t));

struct QsaGateDispatch {
    std::size_t attn_bytes = 0u;
    std::size_t q_full_bytes = 0u;
    QsaGateParams params{};
};

struct QsaAttentionParams {
    std::uint32_t token_rows;
    std::uint32_t cell_capacity;
    std::uint32_t id_row_stride;
    std::uint32_t chunk_count;
};
static_assert(sizeof(QsaAttentionParams) == 4u * sizeof(std::uint32_t));

struct Binding {
    BufferRef buffer;
    NSUInteger index;
};

struct NormRopeDispatch {
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::uint64_t threadgroups = 0u;
    Params params{};
};

struct FwhtDispatch {
    std::size_t values_bytes = 0u;
    std::uint32_t threadgroups = 0u;
    FwhtParams params{};
};

struct IndexerSnapshotDispatch {
    std::size_t tail_stride_bytes = 0u;
    std::size_t block_pos_stride_bytes = 0u;
    std::size_t tail_required_bytes = 0u;
    std::size_t block_pos_required_bytes = 0u;
    std::size_t tail_snapshot_bytes = 0u;
    std::size_t block_pos_snapshot_bytes = 0u;
    IndexerSnapshotParams params{};
};

struct IndexerAppendDispatch {
    std::size_t raw_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t rope_required_bytes = 0u;
    std::size_t tail_bytes = 0u;
    std::size_t dead_bytes = 0u;
    std::size_t pooled_bytes = 0u;
    std::size_t tail_snapshot_bytes = 0u;
    std::size_t block_pos_snapshot_bytes = 0u;
    std::size_t tail_snapshot_binding_bytes = 0u;
    std::size_t block_pos_snapshot_binding_bytes = 0u;
    BufferRef tail_snapshot_binding;
    BufferRef block_pos_snapshot_binding;
    IndexerAppendParams params{};
};

struct PrefillIndexerAppendDispatch {
    std::size_t raw_bytes = 0u;
    std::size_t gamma_bytes = 0u;
    std::size_t rope_required_bytes = 0u;
    std::size_t tail_bytes = 0u;
    std::size_t dead_bytes = 0u;
    std::size_t pooled_bytes = 0u;
    std::uint32_t first_rows = 0u;
    std::uint32_t block_count = 0u;
    PrefillIndexerAppendParams params{};
};

struct SelectionScoresDispatch {
    std::size_t pooled_bytes = 0u;
    std::size_t dead_bytes = 0u;
    std::size_t query_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t scores_bytes = 0u;
    std::uint32_t grid_count_x = 0u;
    SelectionScoresParams params{};
};

struct SelectionTopkDispatch {
    std::size_t scores_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t ids_bytes = 0u;
    std::string_view kernel_name;
    std::uint32_t threads = 0u;
    std::uint32_t histogram_banks = 0u;
    SelectionTopkParams params{};
};

struct PrefillSelectionDispatch {
    std::size_t max_blocks = 0u;
    std::size_t pooled_bytes = 0u;
    std::size_t dead_bytes = 0u;
    std::size_t query_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t scores_bytes = 0u;
    std::size_t ids_bytes = 0u;
    std::size_t query_row_bytes = 0u;
    std::size_t step_row_bytes = 0u;
    std::size_t score_row_bytes = 0u;
    std::size_t id_row_bytes = 0u;
};

struct QsaAttentionChunkDispatch {
    const char* kernel_name = nullptr;
    std::size_t query_bytes = 0u;
    std::size_t steps_bytes = 0u;
    std::size_t ids_bytes = 0u;
    std::size_t key_data_bytes = 0u;
    std::size_t key_scale_bytes = 0u;
    std::size_t value_data_bytes = 0u;
    std::size_t value_scale_bytes = 0u;
    std::size_t key_scale_binding_bytes = 0u;
    std::size_t value_scale_binding_bytes = 0u;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t key_data_alignment = 0u;
    std::size_t key_scale_binding_alignment = 0u;
    std::size_t value_data_alignment = 0u;
    std::size_t value_scale_binding_alignment = 0u;
    std::uint32_t chunk_count = 0u;
    BufferRef key_scale_binding;
    BufferRef value_scale_binding;
    QsaAttentionParams params{};
};

struct QsaAttentionMergeDispatch {
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    std::size_t output_bytes = 0u;
    QsaAttentionParams params{};
};

bool overlaps(std::uintptr_t a, std::size_t a_size,
              std::uintptr_t b, std::size_t b_size) {
    if (a == 0u || b == 0u || a_size == 0u || b_size == 0u) return false;
    return a <= b ? b - a < a_size : a - b < b_size;
}

bool buffer_overlaps(BufferRef left, std::size_t left_bytes,
                     BufferRef right, std::size_t right_bytes) {
    if (left.handle == right.handle &&
        overlaps(left.offset, left_bytes, right.offset, right_bytes)) {
        return true;
    }
    if (overlaps(reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
                 reinterpret_cast<std::uintptr_t>(right.contents),
                 right_bytes)) {
        return true;
    }
    return overlaps(static_cast<std::uintptr_t>(left.gpu_address), left_bytes,
                    static_cast<std::uintptr_t>(right.gpu_address),
                    right_bytes);
}

bool same_view(BufferRef left, BufferRef right) {
    return left.handle == right.handle && left.offset == right.offset &&
           left.length == right.length;
}

bool validate_buffer(Context& context, BufferRef buffer,
                     std::size_t required_bytes, std::size_t alignment,
                     std::string_view name, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = "QSA " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes || buffer.offset % alignment != 0u) {
        error = "QSA " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    id<MTLBuffer> native = (__bridge id<MTLBuffer>)buffer.handle;
    if (native == nil || buffer.offset > native.length ||
        buffer.length > native.length - buffer.offset) {
        error = "QSA " + std::string(name) +
                " view exceeds its Metal allocation";
        return false;
    }

    const std::uintptr_t native_contents =
        reinterpret_cast<std::uintptr_t>(native.contents);
    if (native.contents == nullptr ||
        native_contents >
            std::numeric_limits<std::uintptr_t>::max() - buffer.offset ||
        buffer.contents !=
            reinterpret_cast<void*>(native_contents + buffer.offset)) {
        error = "QSA " + std::string(name) +
                " CPU backing metadata does not match its Metal view";
        return false;
    }

    const std::uint64_t native_gpu_address = native.gpuAddress;
    if (native_gpu_address == 0u) {
        if (buffer.gpu_address != 0u) {
            error = "QSA " + std::string(name) +
                    " GPU address metadata is forged";
            return false;
        }
    } else if (native_gpu_address >
                   std::numeric_limits<std::uint64_t>::max() - buffer.offset ||
               buffer.gpu_address != native_gpu_address + buffer.offset) {
        error = "QSA " + std::string(name) +
                " GPU address metadata does not match its Metal view";
        return false;
    }
    return true;
}

bool validate_kv_step_positions(const StepRow* step_rows,
                                std::uint32_t token_rows,
                                std::uint32_t cell_capacity,
                                std::string_view format,
                                std::string& error) {
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position = step_rows[row].pos;
        if (position < 0 ||
            static_cast<std::uint32_t>(position) >= cell_capacity) {
            error = "QSA " + std::string(format) +
                    " KV step position is outside the cell capacity";
            return false;
        }
    }
    if (token_rows <= VERIFY_MAX_T) {
        for (std::uint32_t row = 0u; row < token_rows; ++row) {
            for (std::uint32_t earlier = 0u; earlier < row; ++earlier) {
                if (step_rows[earlier].pos == step_rows[row].pos) {
                    error = "QSA " + std::string(format) +
                            " KV step positions must be unique";
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<std::int32_t> positions;
    try {
        positions.reserve(token_rows);
    } catch (const std::bad_alloc&) {
        error = "QSA " + std::string(format) +
                " KV position validation allocation failed";
        return false;
    }
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        positions.push_back(step_rows[row].pos);
    }
    std::sort(positions.begin(), positions.end());
    if (std::adjacent_find(positions.begin(), positions.end()) !=
        positions.end()) {
        error = "QSA " + std::string(format) +
                " KV step positions must be unique";
        return false;
    }
    return true;
}

bool validate_pipeline(Context& context, std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == "strata_qsa_norm_rope") {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kThreads ||
        info->threadgroup_x != kThreads || info->threadgroup_y != 1u ||
        info->threadgroup_z != 1u) {
        error = "QSA norm/RoPE threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_norm_rope");
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kThreads) {
        error = "QSA norm/RoPE pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_fwht_pipeline(Context& context, std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == "strata_qsa_fwht256") {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kFwhtThreads ||
        info->threadgroup_x != kFwhtThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA FWHT256 threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_fwht256");
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kFwhtThreads) {
        error = "QSA FWHT256 pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_kv_append_f16_pipeline(Context& context, std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_kv_append_f16";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kKvAppendThreads ||
        info->threadgroup_x != kKvAppendThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA fp16 KV append threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kKvAppendThreads) {
        error = "QSA fp16 KV append pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_kv_append_i8_pipeline(Context& context, std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_kv_append_i8";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kKvAppendThreads ||
        info->threadgroup_x != kKvAppendThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA int8 KV append threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kKvAppendThreads) {
        error = "QSA int8 KV append pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_kv_append_q4_pipeline(Context& context, std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_kv_append_q4";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kKvAppendThreads ||
        info->threadgroup_x != kKvAppendThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA q4 KV append threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kKvAppendThreads) {
        error = "QSA q4 KV append pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_kv_append_k8v4_pipeline(Context& context,
                                      std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_kv_append_k8v4";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kKvAppendThreads ||
        info->threadgroup_x != kKvAppendThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA k8v4 KV append threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kKvAppendThreads) {
        error = "QSA k8v4 KV append pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_indexer_snapshot_pipeline(Context& context,
                                        std::string& error) {
    constexpr std::string_view kernel_name = "strata_idx_snapshot";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kIndexerSnapshotThreads ||
        info->threadgroup_x != kIndexerSnapshotThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA indexer snapshot threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kIndexerSnapshotThreads) {
        error = "QSA indexer snapshot pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_indexer_append_pipeline(Context& context,
                                     std::string& error) {
    constexpr std::string_view kernel_name = "strata_idx_append";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kIndexerAppendThreads ||
        info->threadgroup_x != kIndexerAppendThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA indexer append threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kIndexerAppendThreads) {
        error = "QSA indexer append pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_prefill_indexer_append_pipelines(Context& context,
                                               std::string& error) {
    constexpr std::array<std::string_view, 3> kernel_names{
        "strata_idx_append_first", "strata_idx_append_blocks",
        "strata_idx_append_tail"};
    const KernelInfo* manifest = nullptr;
    std::size_t count = 0u;
    manifest = kernel_manifest(count);
    for (const std::string_view kernel_name : kernel_names) {
        const KernelInfo* info = nullptr;
        for (std::size_t i = 0u; i < count; ++i) {
            if (std::string_view(manifest[i].name) == kernel_name) {
                info = &manifest[i];
                break;
            }
        }
        if (info == nullptr || info->threads != kIndexerAppendThreads ||
            info->threadgroup_x != kIndexerAppendThreads ||
            info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
            error = "QSA Prefill indexer append threadgroup does not match "
                    "its manifest: " +
                    std::string(kernel_name);
            return false;
        }
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)context.pipeline(
                kernel_name.data());
        if (pipeline == nil ||
            pipeline.maxTotalThreadsPerThreadgroup <
                kIndexerAppendThreads) {
            error = "QSA Prefill indexer append pipeline is unavailable or "
                    "below its threadgroup limit: " +
                    std::string(kernel_name);
            return false;
        }
    }
    return true;
}

bool validate_selection_scores_pipeline(Context& context,
                                        std::string& error) {
    constexpr std::string_view kernel_name = "strata_sel_scores";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kSelectionScoresThreads ||
        info->threadgroup_x != kSelectionScoresThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA selection-score threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kSelectionScoresThreads) {
        error = "QSA selection-score pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool validate_selection_topk_pipeline(Context& context,
                                      std::string_view kernel_name,
                                      std::uint32_t threads,
                                      std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != threads ||
        info->threadgroup_x != threads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA top-k threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < threads) {
        error = "QSA top-k pipeline is unavailable or below its threadgroup "
                "limit";
        return false;
    }
    return true;
}

bool validate_qsa_gate_pipeline(Context& context, std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_gate";
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kQsaGateThreads ||
        info->threadgroup_x != kQsaGateThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA gate threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data());
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kQsaGateThreads) {
        error = "QSA gate pipeline is unavailable or below its threadgroup "
                "limit";
        return false;
    }
    return true;
}

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& product) {
    product = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            product > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        product *= factor;
    }
    return true;
}

const char* qsa_attention_kernel_name(QsaAttentionMode mode) {
    switch (mode) {
        case QsaAttentionMode::f16:
            return "strata_attn_chunk_f16";
        case QsaAttentionMode::i8:
            return "strata_attn_chunk_i8";
        case QsaAttentionMode::q4:
            return "strata_attn_chunk_q4";
        case QsaAttentionMode::k8v4:
            return "strata_attn_chunk_k8v4";
    }
    return nullptr;
}

bool validate_qsa_attention_pipeline(Context& context,
                                     const char* kernel_name,
                                     std::string& error) {
    const KernelInfo* info = nullptr;
    std::size_t count = 0u;
    const KernelInfo* manifest = kernel_manifest(count);
    for (std::size_t i = 0u; i < count; ++i) {
        if (std::string_view(manifest[i].name) == kernel_name) {
            info = &manifest[i];
            break;
        }
    }
    if (info == nullptr || info->threads != kQsaAttentionThreads ||
        info->threadgroup_x != kQsaAttentionThreads ||
        info->threadgroup_y != 1u || info->threadgroup_z != 1u) {
        error = "QSA attention threadgroup does not match its manifest";
        return false;
    }
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)context.pipeline(kernel_name);
    if (pipeline == nil ||
        pipeline.maxTotalThreadsPerThreadgroup < kQsaAttentionThreads) {
        error = "QSA attention pipeline is unavailable or below its "
                "threadgroup limit";
        return false;
    }
    return true;
}

bool qsa_attention_extents(std::uint32_t token_rows,
                           std::uint32_t id_row_stride,
                           std::uint32_t& chunk_count,
                           std::size_t& partial_acc_bytes,
                           std::size_t& partial_stats_bytes,
                           std::size_t& output_bytes) {
    if (token_rows == 0u || token_rows > VERIFY_MAX_T ||
        id_row_stride == 0u ||
        id_row_stride > kQsaAttentionMaxCells) {
        return false;
    }
    chunk_count =
        (id_row_stride - 1u) / kQsaAttentionChunkCells + 1u;
    const std::size_t slots = static_cast<std::size_t>(token_rows) *
        N_HEAD_KV * chunk_count;
    return checked_product(
               {slots, N_HEAD_Q / N_HEAD_KV, HEAD_DIM, sizeof(float)},
               partial_acc_bytes) &&
           checked_product(
               {slots, N_HEAD_Q / N_HEAD_KV, sizeof(float)},
               partial_stats_bytes) &&
           checked_product(
               {static_cast<std::size_t>(token_rows), N_HEAD_Q, HEAD_DIM,
                sizeof(float)},
               output_bytes);
}

id<MTLCommandBuffer> validate_command_buffer(
    Context& context, CommandBuffer& command_buffer, std::string& error) {
    void* handle = command_buffer.unsafe_native_handle();
    if (handle == nullptr) {
        error = "QSA command buffer is unavailable";
        return nil;
    }
    if (!command_buffer.belongs_to(context)) {
        error = "QSA command buffer belongs to a different Context";
        return nil;
    }
    id<MTLCommandBuffer> native = (__bridge id<MTLCommandBuffer>)handle;
    if (native == nil ||
        native.status != MTLCommandBufferStatusNotEnqueued) {
        error = "QSA work must be encoded before command-buffer submission";
        return nil;
    }
    return native;
}

bool strided_view_overlaps(BufferRef strided, std::size_t stride_bytes,
                           std::size_t region_bytes, BufferRef other,
                           std::size_t other_bytes) {
    for (std::uint32_t layer = 0u; layer < kQsaIndexerLayers; ++layer) {
        const std::size_t relative_offset =
            static_cast<std::size_t>(layer) * stride_bytes;
        const std::uintptr_t contents =
            reinterpret_cast<std::uintptr_t>(strided.contents);
        if (strided.offset >
                std::numeric_limits<std::uint64_t>::max() - relative_offset ||
            contents >
                std::numeric_limits<std::uintptr_t>::max() - relative_offset ||
            (strided.gpu_address != 0u &&
             strided.gpu_address >
                 std::numeric_limits<std::uint64_t>::max() -
                     relative_offset)) {
            return true;
        }

        const BufferRef region = strided.slice(relative_offset, region_bytes);
        if (!region) return true;
        if (buffer_overlaps(region, region_bytes, other, other_bytes)) {
            return true;
        }
    }
    return false;
}

bool prepare_norm_rope_dispatch(Context& context, std::uint32_t token_rows,
                               QsaNormRopeMode mode,
                               const QsaNormRopeBuffers& buffers,
                               NormRopeDispatch& dispatch,
                               std::uint32_t row_limit,
                               std::string_view row_limit_error,
                               std::string& error) {
    error.clear();
    if (token_rows == 0u) {
        error = row_limit == VERIFY_MAX_T
            ? "QSA token row count must be in 1..8"
            : "Prefill QSA token row count must be positive";
        return false;
    }
    if (token_rows > row_limit) {
        error = row_limit_error;
        return false;
    }
    if (mode != QsaNormRopeMode::key &&
        mode != QsaNormRopeMode::query &&
        mode != QsaNormRopeMode::indexer_query) {
        error = "QSA norm/RoPE mode must be key, query or indexer query";
        return false;
    }

    std::uint32_t heads_per_token = 0u;
    std::uint32_t width = 0u;
    std::uint32_t input_stride = 0u;
    switch (mode) {
        case QsaNormRopeMode::key:
            heads_per_token = N_HEAD_KV;
            width = HEAD_DIM;
            input_stride = HEAD_DIM;
            break;
        case QsaNormRopeMode::query:
            heads_per_token = N_HEAD_Q;
            width = HEAD_DIM;
            input_stride = 2u * HEAD_DIM;
            break;
        case QsaNormRopeMode::indexer_query:
            heads_per_token = INDEXER_HEADS;
            width = INDEXER_HEAD_DIM;
            input_stride = INDEXER_HEAD_DIM;
            break;
    }

    const std::size_t input_elements =
        static_cast<std::size_t>(token_rows) * heads_per_token;
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (input_elements >
            std::numeric_limits<std::size_t>::max() / input_stride ||
        input_elements * input_stride >
            std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        input_elements >
            std::numeric_limits<std::size_t>::max() / width ||
        input_elements * width >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        error = "QSA norm/RoPE buffer size overflows host address space";
        return false;
    }
    input_bytes = input_elements * input_stride * sizeof(float);
    output_bytes = input_elements * width * sizeof(float);
    const std::size_t steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    const std::size_t gamma_bytes =
        static_cast<std::size_t>(width) * sizeof(float);

    if (!validate_buffer(context, buffers.input, input_bytes, alignof(float),
                         "norm input", error) ||
        !validate_buffer(context, buffers.gamma, gamma_bytes, alignof(float),
                         "norm gamma", error) ||
        !validate_buffer(context, buffers.steps, steps_bytes,
                         alignof(StepRow), "step rows", error) ||
        !validate_buffer(context, buffers.output, output_bytes,
                         alignof(float), "norm output", error) ||
        !validate_buffer(context, buffers.rope, kRopeCellBytes,
                         kRopeAlignment, "RoPE table", error)) {
        return false;
    }
    if (buffers.rope.length % kRopeCellBytes != 0u) {
        error = "QSA RoPE table length must contain complete 32-pair rows";
        return false;
    }
    const std::size_t rope_cells =
        static_cast<std::size_t>(buffers.rope.length) / kRopeCellBytes;
    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int32_t position = step_rows[row].pos;
        if (position < 0 ||
            static_cast<std::size_t>(position) >= rope_cells) {
            error = "QSA step position is outside the RoPE table";
            return false;
        }
    }

    const bool input_output_overlap =
        buffer_overlaps(buffers.input, input_bytes,
                        buffers.output, output_bytes);
    if (input_output_overlap &&
        (mode == QsaNormRopeMode::query ||
         !same_view(buffers.input, buffers.output))) {
        error = mode == QsaNormRopeMode::query
            ? "QSA query input and compact output must be disjoint"
            : "QSA in-place input/output must be the exact same view";
        return false;
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> other_inputs{{
        {buffers.gamma, gamma_bytes},
        {buffers.steps, steps_bytes},
        {buffers.rope, static_cast<std::size_t>(buffers.rope.length)}}};
    for (const auto& [input, bytes] : other_inputs) {
        if (buffer_overlaps(buffers.output, output_bytes, input, bytes)) {
            error = "QSA norm/RoPE output must not overlap gamma, steps or "
                    "the RoPE table";
            return false;
        }
    }
    if (!validate_pipeline(context, error)) return false;

    dispatch.input_bytes = input_bytes;
    dispatch.output_bytes = output_bytes;
    dispatch.gamma_bytes = gamma_bytes;
    dispatch.steps_bytes = steps_bytes;
    dispatch.threadgroups =
        static_cast<std::uint64_t>(token_rows) * heads_per_token;
    dispatch.params =
        Params{token_rows, static_cast<std::uint32_t>(mode)};
    error.clear();
    return true;
}

bool encode_kernel(Context& context, CommandBuffer& command_buffer,
                   const std::array<Binding, 5>& bindings,
                   const Params& params, MTLSize grid,
                   std::string& error) {
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA norm/RoPE compute encoder";
        return false;
    }
    encoder.label = @"QSA normalization and RoPE";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_norm_rope")];
    for (const Binding& binding : bindings) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                    offset:binding.buffer.offset
                   atIndex:binding.index];
    }
    [encoder setBytes:&params length:sizeof(params) atIndex:5u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0u];
    [encoder setThreadgroupMemoryLength:HEAD_DIM * sizeof(float) atIndex:1u];
    [encoder dispatchThreadgroups:grid
              threadsPerThreadgroup:MTLSizeMake(kThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

}  // namespace

bool encode_qsa_norm_rope(Context& context, CommandBuffer& command_buffer,
                          std::uint32_t token_rows, QsaNormRopeMode mode,
                          const QsaNormRopeBuffers& buffers,
                          std::string& error) {
    NormRopeDispatch dispatch;
    if (!prepare_norm_rope_dispatch(context, token_rows, mode, buffers,
                                    dispatch, VERIFY_MAX_T,
                                    "QSA token row count must be in 1..8",
                                    error)) {
        return false;
    }
    const std::array<Binding, 5> bindings{{
        {buffers.input, 0u}, {buffers.gamma, 1u}, {buffers.steps, 2u},
        {buffers.rope, 3u}, {buffers.output, 4u}}};
    return encode_kernel(
        context, command_buffer, bindings, dispatch.params,
        MTLSizeMake(dispatch.threadgroups, 1u, 1u), error);
}

bool encode_qsa_norm_rope(Context& context, Encoder& encoder,
                          std::uint32_t token_rows, QsaNormRopeMode mode,
                          const QsaNormRopeBuffers& buffers,
                          std::string& error) {
    NormRopeDispatch dispatch;
    if (!prepare_norm_rope_dispatch(context, token_rows, mode, buffers,
                                    dispatch, VERIFY_MAX_T,
                                    "QSA token row count must be in 1..8",
                                    error)) {
        return false;
    }
    const std::array<EncoderBufferBinding, 5> bindings{{
        {buffers.input, 0u, dispatch.input_bytes, alignof(float)},
        {buffers.gamma, 1u, dispatch.gamma_bytes, alignof(float)},
        {buffers.steps, 2u, dispatch.steps_bytes, alignof(StepRow)},
        {buffers.rope, 3u, buffers.rope.length, kRopeAlignment},
        {buffers.output, 4u, dispatch.output_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 5u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, 8u * sizeof(float)},
        EncoderThreadgroupMemory{1u, HEAD_DIM * sizeof(float)},
    };
    return encoder.dispatch(
        "strata_qsa_norm_rope",
        {static_cast<std::size_t>(dispatch.threadgroups), 1u, 1u},
        {kThreads, 1u, 1u}, bindings, byte_bindings, threadgroup_memory,
        error);
}

bool encode_prefill_qsa_norm_rope(
    Context& context, Encoder& encoder, std::uint32_t token_rows,
    QsaNormRopeMode mode, const QsaNormRopeBuffers& buffers,
    std::string& error) {
    NormRopeDispatch dispatch;
    if (!prepare_norm_rope_dispatch(
            context, token_rows, mode, buffers, dispatch,
            kPrefillMaxShaderRows,
            "Prefill QSA token row count exceeds the shader-row limit",
            error)) {
        return false;
    }
    const std::array<EncoderBufferBinding, 5> bindings{{
        {buffers.input, 0u, dispatch.input_bytes, alignof(float)},
        {buffers.gamma, 1u, dispatch.gamma_bytes, alignof(float)},
        {buffers.steps, 2u, dispatch.steps_bytes, alignof(StepRow)},
        {buffers.rope, 3u, buffers.rope.length, kRopeAlignment},
        {buffers.output, 4u, dispatch.output_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 5u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, 8u * sizeof(float)},
        EncoderThreadgroupMemory{1u, HEAD_DIM * sizeof(float)},
    };
    return encoder.dispatch(
        "strata_qsa_norm_rope",
        {static_cast<std::size_t>(dispatch.threadgroups), 1u, 1u},
        {kThreads, 1u, 1u}, bindings, byte_bindings, threadgroup_memory,
        error);
}

bool encode_prefill_qsa_norm_rope(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, QsaNormRopeMode mode,
    const QsaNormRopeBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill QSA norm/RoPE",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_norm_rope(
                context, encoder, token_rows, mode, buffers, encode_error);
        },
        error);
}

static bool encode_qsa_kv_append_f16_impl(
    Context& context, CommandBuffer* command_buffer,
    Encoder* shared_encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t row_limit, std::string_view row_limit_error,
    const QsaKvAppendF16Buffers& buffers, std::string& error) {
    if (token_rows == 0u || token_rows > row_limit) {
        error = row_limit_error;
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % 4u != 0u ||
        static_cast<std::uint64_t>(cell_capacity) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) + 1u) {
        error = "QSA fp16 KV append cell capacity must be a positive "
                "multiple of four addressable by StepRow";
        return false;
    }

    constexpr std::size_t kCurrentRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(float);
    constexpr std::size_t kPoolRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(std::uint16_t);
    const std::size_t current_rows =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV;
    if (current_rows >
            std::numeric_limits<std::size_t>::max() / kCurrentRowBytes ||
        static_cast<std::size_t>(cell_capacity) >
            std::numeric_limits<std::size_t>::max() /
                (N_HEAD_KV * kPoolRowBytes)) {
        error = "QSA fp16 KV append buffer size overflows host address space";
        return false;
    }
    const std::size_t current_bytes = current_rows * kCurrentRowBytes;
    const std::size_t pool_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kPoolRowBytes;
    const std::size_t steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    if (!validate_buffer(context, buffers.keys, current_bytes, alignof(float),
                         "fp16 KV key input", error) ||
        !validate_buffer(context, buffers.values, current_bytes,
                         alignof(float), "fp16 KV value input", error) ||
        !validate_buffer(context, buffers.steps, steps_bytes,
                         alignof(StepRow), "fp16 KV step rows", error) ||
        !validate_buffer(context, buffers.key_pool, pool_bytes,
                         alignof(std::uint16_t), "fp16 KV key pool", error) ||
        !validate_buffer(context, buffers.value_pool, pool_bytes,
                         alignof(std::uint16_t), "fp16 KV value pool",
                         error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    if (!validate_kv_step_positions(step_rows, token_rows, cell_capacity,
                                    "fp16", error)) {
        return false;
    }

    if (buffer_overlaps(buffers.key_pool, pool_bytes,
                        buffers.value_pool, pool_bytes)) {
        error = "QSA fp16 KV key and value pools must be disjoint";
        return false;
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> inputs{{
        {buffers.keys, current_bytes},
        {buffers.values, current_bytes},
        {buffers.steps, steps_bytes}}};
    for (const auto& [input, bytes] : inputs) {
        if (buffer_overlaps(buffers.key_pool, pool_bytes, input, bytes) ||
            buffer_overlaps(buffers.value_pool, pool_bytes, input, bytes)) {
            error = "QSA fp16 KV pools must not overlap K/V inputs or steps";
            return false;
        }
    }

    const KvAppendF16Params params{token_rows, cell_capacity};
    if (shared_encoder != nullptr) {
        if (!validate_kv_append_f16_pipeline(context, error)) return false;
        const std::array<EncoderBufferBinding, 5> bindings{{
            {buffers.keys, 0u, current_bytes, alignof(float)},
            {buffers.values, 1u, current_bytes, alignof(float)},
            {buffers.steps, 2u, steps_bytes, alignof(StepRow)},
            {buffers.key_pool, 3u, pool_bytes, alignof(std::uint16_t)},
            {buffers.value_pool, 4u, pool_bytes, alignof(std::uint16_t)},
        }};
        const std::array<EncoderBytesBinding, 1> byte_bindings{{
            EncoderBytesBinding{&params, sizeof(params), 5u},
        }};
        const std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
        return shared_encoder->dispatch(
            "strata_qsa_kv_append_f16",
            {static_cast<std::size_t>(token_rows) * N_HEAD_KV, 1u, 1u},
            {kKvAppendThreads, 1u, 1u}, bindings, byte_bindings,
            threadgroup_memory, error);
    }
    if (command_buffer == nullptr) {
        error = "QSA f16 KV append has no dispatch target";
        return false;
    }
    auto encoding_lock = command_buffer->lock_for_encoding();
    if (!validate_kv_append_f16_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, *command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA fp16 KV append compute encoder";
        return false;
    }

    encoder.label = @"QSA fp16 KV append";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_kv_append_f16")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.keys.handle
                offset:buffers.keys.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.values.handle
                offset:buffers.values.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_pool.handle
                offset:buffers.key_pool.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_pool.handle
                offset:buffers.value_pool.offset atIndex:4u];
    [encoder setBytes:&params length:sizeof(params) atIndex:5u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(static_cast<NSUInteger>(token_rows) * N_HEAD_KV, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(kKvAppendThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_kv_append_f16(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_f16_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA fp16 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_qsa_kv_append_f16(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_f16_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA fp16 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_f16(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_f16_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA fp16 KV append row count exceeds the shader-row limit",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_f16(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendF16Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_f16_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA fp16 KV append row count exceeds the shader-row limit",
        buffers, error);
}

static bool encode_qsa_kv_append_i8_impl(
    Context& context, CommandBuffer* command_buffer,
    Encoder* shared_encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t row_limit, std::string_view row_limit_error,
    const QsaKvAppendI8Buffers& buffers, std::string& error) {
    if (token_rows == 0u || token_rows > row_limit) {
        error = row_limit_error;
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % 4u != 0u ||
        static_cast<std::uint64_t>(cell_capacity) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) + 1u) {
        error = "QSA int8 KV append cell capacity must be a positive "
                "multiple of four addressable by StepRow";
        return false;
    }

    constexpr std::size_t kCurrentRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(float);
    constexpr std::size_t kCodePoolRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(std::int8_t);
    constexpr std::size_t kScalePoolRowBytes =
        4u * sizeof(std::uint16_t);
    const std::size_t current_rows =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (current_rows > max_size / kCurrentRowBytes ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kCodePoolRowBytes) ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kScalePoolRowBytes)) {
        error = "QSA int8 KV append buffer size overflows host address space";
        return false;
    }
    const std::size_t current_bytes = current_rows * kCurrentRowBytes;
    const std::size_t code_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kCodePoolRowBytes;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kScalePoolRowBytes;
    const std::size_t steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    if (!validate_buffer(context, buffers.keys, current_bytes, alignof(float),
                         "int8 KV key input", error) ||
        !validate_buffer(context, buffers.values, current_bytes,
                         alignof(float), "int8 KV value input", error) ||
        !validate_buffer(context, buffers.steps, steps_bytes,
                         alignof(StepRow), "int8 KV step rows", error) ||
        !validate_buffer(context, buffers.key_codes, code_bytes,
                         alignof(std::int8_t), "int8 KV key codes", error) ||
        !validate_buffer(context, buffers.key_scales, scale_bytes,
                         alignof(std::uint16_t), "int8 KV key scales",
                         error) ||
        !validate_buffer(context, buffers.value_codes, code_bytes,
                         alignof(std::int8_t), "int8 KV value codes",
                         error) ||
        !validate_buffer(context, buffers.value_scales, scale_bytes,
                         alignof(std::uint16_t), "int8 KV value scales",
                         error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    if (!validate_kv_step_positions(step_rows, token_rows, cell_capacity,
                                    "int8", error)) {
        return false;
    }

    const std::array<std::pair<BufferRef, std::size_t>, 4> outputs{{
        {buffers.key_codes, code_bytes},
        {buffers.key_scales, scale_bytes},
        {buffers.value_codes, code_bytes},
        {buffers.value_scales, scale_bytes}}};
    for (std::size_t first = 0u; first < outputs.size(); ++first) {
        for (std::size_t second = first + 1u; second < outputs.size();
             ++second) {
            if (buffer_overlaps(outputs[first].first, outputs[first].second,
                                outputs[second].first, outputs[second].second)) {
                error = "QSA int8 KV output arrays must be disjoint";
                return false;
            }
        }
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> inputs{{
        {buffers.keys, current_bytes},
        {buffers.values, current_bytes},
        {buffers.steps, steps_bytes}}};
    for (const auto& [input, bytes] : inputs) {
        for (const auto& [output, output_bytes] : outputs) {
            if (buffer_overlaps(output, output_bytes, input, bytes)) {
                error = "QSA int8 KV outputs must not overlap K/V inputs or "
                        "steps";
                return false;
            }
        }
    }

    const KvAppendI8Params params{token_rows, cell_capacity};
    if (shared_encoder != nullptr) {
        if (!validate_kv_append_i8_pipeline(context, error)) return false;
        const std::array<EncoderBufferBinding, 7> bindings{{
            {buffers.keys, 0u, current_bytes, alignof(float)},
            {buffers.values, 1u, current_bytes, alignof(float)},
            {buffers.steps, 2u, steps_bytes, alignof(StepRow)},
            {buffers.key_codes, 3u, code_bytes, alignof(std::int8_t)},
            {buffers.key_scales, 4u, scale_bytes, alignof(std::uint16_t)},
            {buffers.value_codes, 5u, code_bytes, alignof(std::int8_t)},
            {buffers.value_scales, 6u, scale_bytes, alignof(std::uint16_t)},
        }};
        const std::array<EncoderBytesBinding, 1> byte_bindings{{
            EncoderBytesBinding{&params, sizeof(params), 7u},
        }};
        const std::array threadgroup_memory{
            EncoderThreadgroupMemory{0u, 8u * sizeof(float)},
            EncoderThreadgroupMemory{1u, 8u * sizeof(float)},
        };
        return shared_encoder->dispatch(
            "strata_qsa_kv_append_i8",
            {static_cast<std::size_t>(token_rows) * N_HEAD_KV, 1u, 1u},
            {kKvAppendThreads, 1u, 1u}, bindings, byte_bindings,
            threadgroup_memory, error);
    }
    if (command_buffer == nullptr) {
        error = "QSA i8 KV append has no dispatch target";
        return false;
    }
    auto encoding_lock = command_buffer->lock_for_encoding();
    if (!validate_kv_append_i8_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, *command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA int8 KV append compute encoder";
        return false;
    }

    encoder.label = @"QSA int8 KV append";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_kv_append_i8")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.keys.handle
                offset:buffers.keys.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.values.handle
                offset:buffers.values.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_codes.handle
                offset:buffers.key_codes.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_scales.handle
                offset:buffers.key_scales.offset atIndex:4u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_codes.handle
                offset:buffers.value_codes.offset atIndex:5u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_scales.handle
                offset:buffers.value_scales.offset atIndex:6u];
    [encoder setBytes:&params length:sizeof(params) atIndex:7u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:1u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(static_cast<NSUInteger>(token_rows) * N_HEAD_KV, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(kKvAppendThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_kv_append_i8(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_i8_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA int8 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_qsa_kv_append_i8(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_i8_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA int8 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_i8(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_i8_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA int8 KV append row count exceeds the shader-row limit",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_i8(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendI8Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_i8_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA int8 KV append row count exceeds the shader-row limit",
        buffers, error);
}

static bool encode_qsa_kv_append_q4_impl(
    Context& context, CommandBuffer* command_buffer,
    Encoder* shared_encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t row_limit, std::string_view row_limit_error,
    const QsaKvAppendQ4Buffers& buffers, std::string& error) {
    if (token_rows == 0u || token_rows > row_limit) {
        error = row_limit_error;
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % 4u != 0u ||
        static_cast<std::uint64_t>(cell_capacity) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) + 1u) {
        error = "QSA q4 KV append cell capacity must be a positive multiple "
                "of four addressable by StepRow";
        return false;
    }

    constexpr std::size_t kCurrentRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(float);
    constexpr std::size_t kQ4PoolRowBytes =
        static_cast<std::size_t>(HEAD_DIM / 32u) * sizeof(block_q4_0);
    const std::size_t current_rows =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (current_rows > max_size / kCurrentRowBytes ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kQ4PoolRowBytes)) {
        error = "QSA q4 KV append buffer size overflows host address space";
        return false;
    }
    const std::size_t current_bytes = current_rows * kCurrentRowBytes;
    const std::size_t pool_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kQ4PoolRowBytes;
    const std::size_t steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    if (!validate_buffer(context, buffers.keys, current_bytes, alignof(float),
                         "q4 KV key input", error) ||
        !validate_buffer(context, buffers.values, current_bytes,
                         alignof(float), "q4 KV value input", error) ||
        !validate_buffer(context, buffers.steps, steps_bytes,
                         alignof(StepRow), "q4 KV step rows", error) ||
        !validate_buffer(context, buffers.key_pool, pool_bytes,
                         alignof(block_q4_0), "q4 KV key pool", error) ||
        !validate_buffer(context, buffers.value_pool, pool_bytes,
                         alignof(block_q4_0), "q4 KV value pool", error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    if (!validate_kv_step_positions(step_rows, token_rows, cell_capacity,
                                    "q4", error)) {
        return false;
    }

    if (buffer_overlaps(buffers.key_pool, pool_bytes,
                        buffers.value_pool, pool_bytes)) {
        error = "QSA q4 KV key and value pools must be disjoint";
        return false;
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> inputs{{
        {buffers.keys, current_bytes},
        {buffers.values, current_bytes},
        {buffers.steps, steps_bytes}}};
    for (const auto& [input, bytes] : inputs) {
        if (buffer_overlaps(buffers.key_pool, pool_bytes, input, bytes) ||
            buffer_overlaps(buffers.value_pool, pool_bytes, input, bytes)) {
            error = "QSA q4 KV pools must not overlap K/V inputs or steps";
            return false;
        }
    }

    const KvAppendQ4Params params{token_rows, cell_capacity};
    if (shared_encoder != nullptr) {
        if (shared_encoder->concurrent_group_open()) {
            error = "QSA q4 KV append must follow FWHT outside an open "
                    "concurrent dispatch group";
            return false;
        }
        if (!validate_kv_append_q4_pipeline(context, error)) return false;
        const std::array<EncoderBufferBinding, 5> bindings{{
            {buffers.keys, 0u, current_bytes, alignof(float)},
            {buffers.values, 1u, current_bytes, alignof(float)},
            {buffers.steps, 2u, steps_bytes, alignof(StepRow)},
            {buffers.key_pool, 3u, pool_bytes, alignof(block_q4_0)},
            {buffers.value_pool, 4u, pool_bytes, alignof(block_q4_0)},
        }};
        const std::array<EncoderBytesBinding, 1> byte_bindings{{
            EncoderBytesBinding{&params, sizeof(params), 5u},
        }};
        const std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
        return shared_encoder->dispatch(
            "strata_qsa_kv_append_q4",
            {static_cast<std::size_t>(token_rows) * N_HEAD_KV, 1u, 1u},
            {kKvAppendThreads, 1u, 1u}, bindings, byte_bindings,
            threadgroup_memory, error);
    }
    if (command_buffer == nullptr) {
        error = "QSA q4 KV append has no dispatch target";
        return false;
    }
    auto encoding_lock = command_buffer->lock_for_encoding();
    if (!validate_kv_append_q4_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, *command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA q4 KV append compute encoder";
        return false;
    }

    encoder.label = @"QSA q4 KV append";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_kv_append_q4")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.keys.handle
                offset:buffers.keys.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.values.handle
                offset:buffers.values.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_pool.handle
                offset:buffers.key_pool.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_pool.handle
                offset:buffers.value_pool.offset atIndex:4u];
    [encoder setBytes:&params length:sizeof(params) atIndex:5u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(static_cast<NSUInteger>(token_rows) * N_HEAD_KV, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(kKvAppendThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_kv_append_q4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_q4_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA q4 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_qsa_kv_append_q4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_q4_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA q4 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_q4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_q4_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA q4 KV append row count exceeds the shader-row limit",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_q4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendQ4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_q4_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA q4 KV append row count exceeds the shader-row limit",
        buffers, error);
}

static bool encode_qsa_kv_append_k8v4_impl(
    Context& context, CommandBuffer* command_buffer,
    Encoder* shared_encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t row_limit, std::string_view row_limit_error,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error) {
    constexpr std::string_view kernel_name = "strata_qsa_kv_append_k8v4";
    if (token_rows == 0u || token_rows > row_limit) {
        error = row_limit_error;
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % 4u != 0u ||
        static_cast<std::uint64_t>(cell_capacity) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) + 1u) {
        error = "QSA k8v4 KV append cell capacity must be a positive "
                "multiple of four addressable by StepRow";
        return false;
    }

    constexpr std::size_t kCurrentRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(float);
    constexpr std::size_t kCodePoolRowBytes =
        static_cast<std::size_t>(HEAD_DIM) * sizeof(std::int8_t);
    constexpr std::size_t kScalePoolRowBytes =
        4u * sizeof(std::uint16_t);
    constexpr std::size_t kQ4PoolRowBytes =
        static_cast<std::size_t>(HEAD_DIM / 32u) * sizeof(block_q4_0);
    const std::size_t current_rows =
        static_cast<std::size_t>(token_rows) * N_HEAD_KV;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (current_rows > max_size / kCurrentRowBytes ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kCodePoolRowBytes) ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kScalePoolRowBytes) ||
        static_cast<std::size_t>(cell_capacity) >
            max_size / (N_HEAD_KV * kQ4PoolRowBytes)) {
        error = "QSA k8v4 KV append buffer size overflows host address space";
        return false;
    }
    const std::size_t current_bytes = current_rows * kCurrentRowBytes;
    const std::size_t code_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kCodePoolRowBytes;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kScalePoolRowBytes;
    const std::size_t value_pool_bytes =
        static_cast<std::size_t>(cell_capacity) *
        N_HEAD_KV * kQ4PoolRowBytes;
    const std::size_t steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    if (!validate_buffer(context, buffers.keys, current_bytes, alignof(float),
                         "k8v4 KV key input", error) ||
        !validate_buffer(context, buffers.values, current_bytes,
                         alignof(float), "k8v4 KV value input", error) ||
        !validate_buffer(context, buffers.steps, steps_bytes,
                         alignof(StepRow), "k8v4 KV step rows", error) ||
        !validate_buffer(context, buffers.key_codes, code_bytes,
                         alignof(std::int8_t), "k8v4 KV key codes", error) ||
        !validate_buffer(context, buffers.key_scales, scale_bytes,
                         alignof(std::uint16_t), "k8v4 KV key scales",
                         error) ||
        !validate_buffer(context, buffers.value_pool, value_pool_bytes,
                         alignof(block_q4_0), "k8v4 KV value pool", error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    if (!validate_kv_step_positions(step_rows, token_rows, cell_capacity,
                                    "k8v4", error)) {
        return false;
    }

    const std::array<std::pair<BufferRef, std::size_t>, 3> outputs{{
        {buffers.key_codes, code_bytes},
        {buffers.key_scales, scale_bytes},
        {buffers.value_pool, value_pool_bytes}}};
    for (std::size_t first = 0u; first < outputs.size(); ++first) {
        for (std::size_t second = first + 1u; second < outputs.size();
             ++second) {
            if (buffer_overlaps(outputs[first].first, outputs[first].second,
                                outputs[second].first, outputs[second].second)) {
                error = "QSA k8v4 KV output arrays must be disjoint";
                return false;
            }
        }
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> inputs{{
        {buffers.keys, current_bytes},
        {buffers.values, current_bytes},
        {buffers.steps, steps_bytes}}};
    for (std::size_t first = 0u; first < inputs.size(); ++first) {
        for (std::size_t second = first + 1u; second < inputs.size();
             ++second) {
            if (buffer_overlaps(inputs[first].first, inputs[first].second,
                                inputs[second].first, inputs[second].second)) {
                error = "QSA k8v4 KV input arrays must be disjoint";
                return false;
            }
        }
    }
    for (const auto& [input, bytes] : inputs) {
        for (const auto& [output, output_bytes] : outputs) {
            if (buffer_overlaps(output, output_bytes, input, bytes)) {
                error = "QSA k8v4 KV outputs must not overlap K/V inputs or "
                        "steps";
                return false;
            }
        }
    }

    const KvAppendK8V4Params params{token_rows, cell_capacity};
    if (shared_encoder != nullptr) {
        if (shared_encoder->concurrent_group_open()) {
            error = "QSA k8v4 KV append must follow V FWHT outside an open "
                    "concurrent dispatch group";
            return false;
        }
        if (!validate_kv_append_k8v4_pipeline(context, error)) return false;
        const std::array<EncoderBufferBinding, 6> bindings{{
            {buffers.keys, 0u, current_bytes, alignof(float)},
            {buffers.values, 1u, current_bytes, alignof(float)},
            {buffers.steps, 2u, steps_bytes, alignof(StepRow)},
            {buffers.key_codes, 3u, code_bytes, alignof(std::int8_t)},
            {buffers.key_scales, 4u, scale_bytes, alignof(std::uint16_t)},
            {buffers.value_pool, 5u, value_pool_bytes, alignof(block_q4_0)},
        }};
        const std::array<EncoderBytesBinding, 1> byte_bindings{{
            EncoderBytesBinding{&params, sizeof(params), 6u},
        }};
        const std::array threadgroup_memory{
            EncoderThreadgroupMemory{0u, 8u * sizeof(float)},
        };
        return shared_encoder->dispatch(
            "strata_qsa_kv_append_k8v4",
            {static_cast<std::size_t>(token_rows) * N_HEAD_KV, 1u, 1u},
            {kKvAppendThreads, 1u, 1u}, bindings, byte_bindings,
            threadgroup_memory, error);
    }
    if (command_buffer == nullptr) {
        error = "QSA k8v4 KV append has no dispatch target";
        return false;
    }
    auto encoding_lock = command_buffer->lock_for_encoding();
    if (!validate_kv_append_k8v4_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, *command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA k8v4 KV append compute encoder";
        return false;
    }

    encoder.label = @"QSA k8v4 KV append";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            kernel_name.data())];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.keys.handle
                offset:buffers.keys.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.values.handle
                offset:buffers.values.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_codes.handle
                offset:buffers.key_codes.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_scales.handle
                offset:buffers.key_scales.offset atIndex:4u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_pool.handle
                offset:buffers.value_pool.offset atIndex:5u];
    [encoder setBytes:&params length:sizeof(params) atIndex:6u];
    [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(static_cast<NSUInteger>(token_rows) * N_HEAD_KV, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(kKvAppendThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_kv_append_k8v4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_k8v4_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA k8v4 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_qsa_kv_append_k8v4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_k8v4_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        VERIFY_MAX_T, "QSA k8v4 KV append row count must be in 1..8",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_k8v4(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_k8v4_impl(
        context, &command_buffer, nullptr, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA k8v4 KV append row count exceeds the shader-row limit",
        buffers, error);
}

bool encode_prefill_qsa_kv_append_k8v4(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaKvAppendK8V4Buffers& buffers, std::string& error) {
    return encode_qsa_kv_append_k8v4_impl(
        context, nullptr, &encoder, token_rows, cell_capacity,
        kPrefillMaxShaderRows,
        "Prefill QSA k8v4 KV append row count exceeds the shader-row limit",
        buffers, error);
}

bool prepare_fwht_dispatch(Context& context, std::uint32_t rows,
                           BufferRef values, FwhtDispatch& dispatch,
                           std::string& error) {
    error.clear();
    if (rows == 0u) {
        error = "QSA FWHT256 row count must be positive";
        return false;
    }
    if (static_cast<std::size_t>(rows) >
        std::numeric_limits<std::size_t>::max() / kFwhtRowBytes) {
        error = "QSA FWHT256 buffer size overflows host address space";
        return false;
    }
    const std::size_t required_bytes =
        static_cast<std::size_t>(rows) * kFwhtRowBytes;
    if (!validate_buffer(context, values, required_bytes, alignof(float),
                         "FWHT values", error)) {
        return false;
    }

    dispatch.values_bytes = required_bytes;
    dispatch.threadgroups =
        rows / 8u + (rows % 8u == 0u ? 0u : 1u);
    dispatch.params = FwhtParams{rows};
    error.clear();
    return true;
}

static bool prepare_indexer_snapshot_dispatch(
    Context& context, std::uint64_t tail_layer_stride_floats,
    std::uint64_t block_pos_layer_stride_ints,
    const QsaIndexerSnapshotBuffers& buffers,
    IndexerSnapshotDispatch& dispatch, std::string& error) {
    error.clear();
    const auto strided_extent = [](std::uint64_t stride_elements,
                                   std::size_t element_bytes,
                                   std::size_t region_elements,
                                   std::size_t& stride_bytes,
                                   std::size_t& required_bytes) {
        const std::size_t max_size =
            std::numeric_limits<std::size_t>::max();
        if (stride_elements < region_elements ||
            stride_elements > max_size / element_bytes) {
            return false;
        }
        stride_bytes =
            static_cast<std::size_t>(stride_elements) * element_bytes;
        const std::size_t region_bytes = region_elements * element_bytes;
        const std::size_t preceding_layers = kQsaIndexerLayers - 1u;
        if (stride_bytes != 0u &&
            preceding_layers > (max_size - region_bytes) / stride_bytes) {
            return false;
        }
        required_bytes = preceding_layers * stride_bytes + region_bytes;
        return true;
    };

    if (!strided_extent(tail_layer_stride_floats, sizeof(float),
                        kIndexerTailFloats, dispatch.tail_stride_bytes,
                        dispatch.tail_required_bytes)) {
        error = "QSA indexer tail stride is too small or its extent overflows";
        return false;
    }
    if (!strided_extent(block_pos_layer_stride_ints, sizeof(std::int32_t),
                        1u, dispatch.block_pos_stride_bytes,
                        dispatch.block_pos_required_bytes)) {
        error = "QSA indexer block-position stride is too small or its extent "
                "overflows";
        return false;
    }

    dispatch.tail_snapshot_bytes =
        kQsaIndexerLayers * kIndexerTailFloats * sizeof(float);
    dispatch.block_pos_snapshot_bytes =
        kQsaIndexerLayers * sizeof(std::int32_t);
    if (!validate_buffer(context, buffers.tails,
                         dispatch.tail_required_bytes, alignof(float),
                         "indexer tails", error) ||
        !validate_buffer(context, buffers.block_positions,
                         dispatch.block_pos_required_bytes,
                         alignof(std::int32_t), "indexer block positions",
                         error) ||
        !validate_buffer(context, buffers.tail_snapshot,
                         dispatch.tail_snapshot_bytes, alignof(float),
                         "indexer tail snapshot", error) ||
        !validate_buffer(context, buffers.block_pos_snapshot,
                         dispatch.block_pos_snapshot_bytes,
                         alignof(std::int32_t),
                         "indexer block-position snapshot", error)) {
        return false;
    }

    if (buffer_overlaps(buffers.tail_snapshot,
                        dispatch.tail_snapshot_bytes,
                        buffers.block_pos_snapshot,
                        dispatch.block_pos_snapshot_bytes) ||
        strided_view_overlaps(
            buffers.tails, dispatch.tail_stride_bytes,
            kIndexerTailFloats * sizeof(float), buffers.tail_snapshot,
            dispatch.tail_snapshot_bytes) ||
        strided_view_overlaps(
            buffers.tails, dispatch.tail_stride_bytes,
            kIndexerTailFloats * sizeof(float), buffers.block_pos_snapshot,
            dispatch.block_pos_snapshot_bytes) ||
        strided_view_overlaps(
            buffers.block_positions, dispatch.block_pos_stride_bytes,
            sizeof(std::int32_t), buffers.tail_snapshot,
            dispatch.tail_snapshot_bytes) ||
        strided_view_overlaps(
            buffers.block_positions, dispatch.block_pos_stride_bytes,
            sizeof(std::int32_t), buffers.block_pos_snapshot,
            dispatch.block_pos_snapshot_bytes)) {
        error = "QSA indexer snapshot outputs overlap each other or an input";
        return false;
    }

    dispatch.params = IndexerSnapshotParams{
        tail_layer_stride_floats, block_pos_layer_stride_ints};
    error.clear();
    return true;
}

bool encode_qsa_fwht256(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, BufferRef values,
                        std::string& error) {
    FwhtDispatch dispatch;
    if (!prepare_fwht_dispatch(context, rows, values, dispatch, error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_fwht_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA FWHT256 compute encoder";
        return false;
    }
    encoder.label = @"QSA FWHT256";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_fwht256")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)values.handle
                offset:values.offset
               atIndex:0u];
    [encoder setBytes:&dispatch.params
               length:sizeof(dispatch.params)
              atIndex:1u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(dispatch.threadgroups, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(kFwhtThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_fwht256(Context& context, Encoder& encoder,
                        std::uint32_t rows, BufferRef values,
                        std::string& error) {
    FwhtDispatch dispatch;
    if (!prepare_fwht_dispatch(context, rows, values, dispatch, error)) {
        return false;
    }
    if (!validate_fwht_pipeline(context, error)) return false;

    const std::array<EncoderBufferBinding, 1> bindings{{
        {values, 0u, dispatch.values_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 1u};
    const std::array byte_bindings{params_binding};
    const std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_qsa_fwht256", {dispatch.threadgroups, 1u, 1u},
        {kFwhtThreads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

bool encode_qsa_indexer_snapshot(
    Context& context, CommandBuffer& command_buffer,
    std::uint64_t tail_layer_stride_floats,
    std::uint64_t block_pos_layer_stride_ints,
    const QsaIndexerSnapshotBuffers& buffers, std::string& error) {
    IndexerSnapshotDispatch dispatch;
    if (!prepare_indexer_snapshot_dispatch(
            context, tail_layer_stride_floats, block_pos_layer_stride_ints,
            buffers, dispatch, error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_indexer_snapshot_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA indexer snapshot compute encoder";
        return false;
    }
    encoder.label = @"QSA indexer snapshot";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_idx_snapshot")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.tails.handle
                offset:buffers.tails.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.block_positions.handle
                offset:buffers.block_positions.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.tail_snapshot.handle
                offset:buffers.tail_snapshot.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.block_pos_snapshot.handle
                offset:buffers.block_pos_snapshot.offset atIndex:3u];
    [encoder setBytes:&dispatch.params
               length:sizeof(dispatch.params) atIndex:4u];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kIndexerSnapshotThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_indexer_snapshot(
    Context& context, Encoder& encoder,
    std::uint64_t tail_layer_stride_floats,
    std::uint64_t block_pos_layer_stride_ints,
    const QsaIndexerSnapshotBuffers& buffers, std::string& error) {
    IndexerSnapshotDispatch dispatch;
    if (!prepare_indexer_snapshot_dispatch(
            context, tail_layer_stride_floats, block_pos_layer_stride_ints,
            buffers, dispatch, error)) {
        return false;
    }
    if (!validate_indexer_snapshot_pipeline(context, error)) return false;

    const std::array<EncoderBufferBinding, 4> bindings{{
        {buffers.tails, 0u, dispatch.tail_required_bytes, alignof(float)},
        {buffers.block_positions, 1u, dispatch.block_pos_required_bytes,
         alignof(std::int32_t)},
        {buffers.tail_snapshot, 2u, dispatch.tail_snapshot_bytes,
         alignof(float)},
        {buffers.block_pos_snapshot, 3u, dispatch.block_pos_snapshot_bytes,
         alignof(std::int32_t)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 4u};
    const std::array byte_bindings{params_binding};
    const std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_idx_snapshot", {1u, 1u, 1u},
        {kIndexerSnapshotThreads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

static bool prepare_indexer_append_dispatch(
    Context& context, std::uint32_t token_rows,
    std::uint32_t cell_capacity, QsaIndexerAppendMode mode,
    std::uint32_t qsa_layer_index, float rope_zero_cos, float rope_zero_sin,
    const QsaIndexerAppendBuffers& buffers, IndexerAppendDispatch& dispatch,
    std::string& error) {
    error.clear();
    if (token_rows == 0u || token_rows > VERIFY_MAX_T) {
        error = "QSA indexer append row count must be in 1..8";
        return false;
    }
    if (mode != QsaIndexerAppendMode::window &&
        mode != QsaIndexerAppendMode::commit) {
        error = "QSA indexer append mode must be window or commit";
        return false;
    }
    if (qsa_layer_index >= kQsaIndexerLayers) {
        error = "QSA indexer append layer index must be in 0..11";
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        cell_capacity >
            static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        !std::isfinite(rope_zero_cos) || !std::isfinite(rope_zero_sin)) {
        error = "QSA indexer append capacity or angle-zero pair is invalid";
        return false;
    }

    constexpr std::size_t raw_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t tail_bytes =
        3u * static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t dead_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t block_pos_bytes = sizeof(std::int32_t);
    constexpr std::size_t tail_snapshot_bytes =
        static_cast<std::size_t>(kQsaIndexerLayers) *
        3u * INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t block_pos_snapshot_bytes =
        static_cast<std::size_t>(kQsaIndexerLayers) * sizeof(std::int32_t);

    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t pool_rows =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    constexpr std::size_t pool_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    if (pool_rows > maximum / pool_row_bytes) {
        error = "QSA indexer append buffer size overflows host address space";
        return false;
    }
    dispatch.raw_bytes =
        static_cast<std::size_t>(token_rows) * raw_row_bytes;
    dispatch.steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    dispatch.gamma_bytes = gamma_bytes;
    dispatch.pooled_bytes = pool_rows * pool_row_bytes;
    dispatch.tail_bytes = tail_bytes;
    dispatch.dead_bytes = dead_bytes;
    dispatch.tail_snapshot_bytes = tail_snapshot_bytes;
    dispatch.block_pos_snapshot_bytes = block_pos_snapshot_bytes;

    const std::uint64_t rope_rows =
        buffers.rope.length / kRopeCellBytes;
    const std::uint64_t last_block =
        (static_cast<std::uint64_t>(cell_capacity) - 1u) /
        INDEXER_BLOCK_CELLS;
    const std::uint64_t required_rope_rows =
        last_block * INDEXER_BLOCK_CELLS + 1u;
    if (buffers.rope.length % kRopeCellBytes != 0u ||
        required_rope_rows > maximum / kRopeCellBytes) {
        error = "QSA indexer RoPE table has an invalid row extent";
        return false;
    }
    dispatch.rope_required_bytes =
        static_cast<std::size_t>(required_rope_rows) * kRopeCellBytes;

    if (!validate_buffer(context, buffers.raw_keys, dispatch.raw_bytes,
                         alignof(float), "indexer raw keys", error) ||
        !validate_buffer(context, buffers.steps, dispatch.steps_bytes,
                         alignof(StepRow), "indexer step rows", error) ||
        !validate_buffer(context, buffers.gamma, dispatch.gamma_bytes,
                         alignof(float), "indexer gamma", error) ||
        !validate_buffer(context, buffers.rope, dispatch.rope_required_bytes,
                         kRopeAlignment, "indexer RoPE table", error) ||
        !validate_buffer(context, buffers.tail, dispatch.tail_bytes,
                         alignof(float), "indexer tail", error) ||
        !validate_buffer(context, buffers.dead, dispatch.dead_bytes,
                         alignof(float), "indexer dead key", error) ||
        !validate_buffer(context, buffers.pooled, dispatch.pooled_bytes,
                         alignof(float), "indexer pooled keys", error) ||
        !validate_buffer(context, buffers.block_position, block_pos_bytes,
                         alignof(std::int32_t),
                         "indexer block position", error)) {
        return false;
    }
    if (mode == QsaIndexerAppendMode::commit &&
        (!validate_buffer(context, buffers.tail_snapshot,
                          dispatch.tail_snapshot_bytes, alignof(float),
                          "indexer tail snapshot", error) ||
         !validate_buffer(context, buffers.block_pos_snapshot,
                          dispatch.block_pos_snapshot_bytes,
                          alignof(std::int32_t),
                          "indexer block-position snapshot", error))) {
        return false;
    }
    if (mode == QsaIndexerAppendMode::commit &&
        buffer_overlaps(buffers.tail_snapshot,
                        dispatch.tail_snapshot_bytes,
                        buffers.block_pos_snapshot,
                        dispatch.block_pos_snapshot_bytes)) {
        error = "QSA indexer commit snapshots must be disjoint";
        return false;
    }
    if (buffers.rope.length % kRopeCellBytes != 0u) {
        error = "QSA indexer RoPE table length must contain complete "
                "32-pair rows";
        return false;
    }
    if (rope_rows < required_rope_rows) {
        error = "QSA indexer RoPE table does not cover the cell capacity";
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    const std::int32_t position0 = step_rows[0].pos;
    if (position0 < 0 ||
        static_cast<std::uint64_t>(position0) + token_rows > cell_capacity) {
        error = "QSA indexer append step positions exceed the cell capacity";
        return false;
    }
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::int64_t expected =
            static_cast<std::int64_t>(position0) + row;
        if (step_rows[row].pos != expected) {
            error = "QSA indexer append step positions must be contiguous "
                    "and increasing";
            return false;
        }
    }

    const std::array<std::pair<BufferRef, std::size_t>, 4> outputs{{
        {buffers.tail, dispatch.tail_bytes},
        {buffers.dead, dispatch.dead_bytes},
        {buffers.pooled, dispatch.pooled_bytes},
        {buffers.block_position, block_pos_bytes},
    }};
    for (std::size_t left = 0u; left < outputs.size(); ++left) {
        for (std::size_t right = left + 1u; right < outputs.size(); ++right) {
            if (buffer_overlaps(outputs[left].first, outputs[left].second,
                                outputs[right].first, outputs[right].second)) {
                error = "QSA indexer append outputs must be disjoint";
                return false;
            }
        }
    }

    const std::array<std::pair<BufferRef, std::size_t>, 6> inputs{{
        {buffers.raw_keys, dispatch.raw_bytes},
        {buffers.steps, dispatch.steps_bytes},
        {buffers.gamma, dispatch.gamma_bytes},
        {buffers.rope, dispatch.rope_required_bytes},
        {buffers.tail_snapshot,
         mode == QsaIndexerAppendMode::commit
             ? dispatch.tail_snapshot_bytes
             : 0u},
        {buffers.block_pos_snapshot,
         mode == QsaIndexerAppendMode::commit
             ? dispatch.block_pos_snapshot_bytes
             : 0u},
    }};
    for (const auto& [output, output_bytes] : outputs) {
        for (const auto& [input, input_bytes] : inputs) {
            if (buffer_overlaps(output, output_bytes, input, input_bytes)) {
                error = "QSA indexer append outputs must not overlap inputs "
                        "or commit snapshots";
                return false;
            }
        }
    }

    const bool commit = mode == QsaIndexerAppendMode::commit;
    dispatch.tail_snapshot_binding =
        commit ? buffers.tail_snapshot : buffers.tail;
    dispatch.block_pos_snapshot_binding =
        commit ? buffers.block_pos_snapshot : buffers.block_position;
    dispatch.tail_snapshot_binding_bytes =
        commit ? dispatch.tail_snapshot_bytes : dispatch.tail_bytes;
    dispatch.block_pos_snapshot_binding_bytes =
        commit ? dispatch.block_pos_snapshot_bytes : block_pos_bytes;
    dispatch.params = IndexerAppendParams{
        token_rows,
        cell_capacity,
        static_cast<std::uint32_t>(mode),
        qsa_layer_index,
        position0,
        rope_zero_cos,
        rope_zero_sin,
    };
    error.clear();
    return true;
}

static bool prepare_prefill_indexer_append_dispatch(
    Context& context, const PrefillParams& prefill,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    const QsaPrefillIndexerAppendBuffers& buffers,
    PrefillIndexerAppendDispatch& dispatch, std::string& error) {
    error.clear();
    if (prefill.T == 0u || prefill.T > kPrefillMaxShaderRows) {
        error = "Prefill QSA indexer append row count exceeds the "
                "shader-row limit";
        return false;
    }
    if (cell_capacity == 0u || cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        cell_capacity >
            static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())) {
        error = "Prefill QSA indexer cell capacity is invalid";
        return false;
    }
    if (!std::isfinite(rope_zero_cos) || !std::isfinite(rope_zero_sin)) {
        error = "Prefill QSA indexer angle-zero pair must be finite";
        return false;
    }

    const std::uint64_t end =
        static_cast<std::uint64_t>(prefill.p0) + prefill.T;
    if (prefill.p0 >= cell_capacity || end > cell_capacity ||
        end > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int32_t>::max())) {
        error = "Prefill QSA indexer chunk exceeds the cell capacity";
        return false;
    }
    const std::uint64_t expected_idx_f =
        prefill.p0 == 0u
            ? INDEXER_BLOCK_CELLS
            : (static_cast<std::uint64_t>(prefill.p0) +
               INDEXER_BLOCK_CELLS - 1u) &
                  ~static_cast<std::uint64_t>(
                      INDEXER_BLOCK_CELLS - 1u);
    const std::uint32_t expected_idx_e = static_cast<std::uint32_t>(
        end & ~static_cast<std::uint64_t>(INDEXER_BLOCK_CELLS - 1u));
    if (expected_idx_f > std::numeric_limits<std::uint32_t>::max() ||
        prefill.idx_f != expected_idx_f ||
        prefill.idx_e != expected_idx_e) {
        error = "Prefill QSA indexer parameters do not match the chunk";
        return false;
    }
    constexpr std::size_t raw_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t tail_bytes =
        3u * static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t dead_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t block_pos_bytes = sizeof(std::int32_t);
    constexpr std::size_t pool_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    static_assert(
        static_cast<std::uint64_t>(kPrefillMaxShaderRows) *
            raw_row_bytes <=
        std::numeric_limits<std::size_t>::max());
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t pool_rows =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    if (pool_rows > maximum / pool_row_bytes) {
        error = "Prefill QSA indexer pool size overflows host address space";
        return false;
    }
    dispatch.raw_bytes =
        static_cast<std::size_t>(prefill.T) * raw_row_bytes;
    dispatch.gamma_bytes = gamma_bytes;
    dispatch.tail_bytes = tail_bytes;
    dispatch.dead_bytes = dead_bytes;
    dispatch.pooled_bytes = pool_rows * pool_row_bytes;
    dispatch.first_rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(prefill.idx_f, end) - prefill.p0);
    dispatch.block_count =
        prefill.idx_e > prefill.idx_f
            ? (prefill.idx_e - prefill.idx_f) / INDEXER_BLOCK_CELLS
            : 0u;
    dispatch.params = {prefill, rope_zero_cos, rope_zero_sin};

    const std::uint64_t rope_rows =
        buffers.rope.length / kRopeCellBytes;
    const std::uint64_t last_block =
        (static_cast<std::uint64_t>(cell_capacity) - 1u) /
        INDEXER_BLOCK_CELLS;
    const std::uint64_t required_rope_rows =
        last_block * INDEXER_BLOCK_CELLS + 1u;
    if (buffers.rope.length % kRopeCellBytes != 0u ||
        required_rope_rows > maximum / kRopeCellBytes) {
        error = "Prefill QSA indexer RoPE table has an invalid row extent";
        return false;
    }
    dispatch.rope_required_bytes =
        static_cast<std::size_t>(required_rope_rows) * kRopeCellBytes;
    if (!validate_buffer(context, buffers.raw_keys, dispatch.raw_bytes,
                         alignof(float), "Prefill indexer raw keys", error) ||
        !validate_buffer(context, buffers.gamma, dispatch.gamma_bytes,
                         alignof(float), "Prefill indexer gamma", error) ||
        !validate_buffer(context, buffers.rope,
                         dispatch.rope_required_bytes, kRopeAlignment,
                         "Prefill indexer RoPE table", error) ||
        !validate_buffer(context, buffers.tail, dispatch.tail_bytes,
                         alignof(float), "Prefill indexer tail", error) ||
        !validate_buffer(context, buffers.dead, dispatch.dead_bytes,
                         alignof(float), "Prefill indexer dead key", error) ||
        !validate_buffer(context, buffers.pooled, dispatch.pooled_bytes,
                         alignof(float), "Prefill indexer pooled keys",
                         error) ||
        !validate_buffer(context, buffers.block_position, block_pos_bytes,
                         alignof(std::int32_t),
                         "Prefill indexer block position", error)) {
        return false;
    }
    if (rope_rows < required_rope_rows) {
        error = "Prefill QSA indexer RoPE table does not cover the cell "
                "capacity";
        return false;
    }

    const std::array<std::pair<BufferRef, std::size_t>, 4> outputs{{
        {buffers.tail, dispatch.tail_bytes},
        {buffers.dead, dispatch.dead_bytes},
        {buffers.pooled, dispatch.pooled_bytes},
        {buffers.block_position, block_pos_bytes},
    }};
    for (std::size_t left = 0u; left < outputs.size(); ++left) {
        for (std::size_t right = left + 1u; right < outputs.size(); ++right) {
            if (buffer_overlaps(outputs[left].first, outputs[left].second,
                                outputs[right].first, outputs[right].second)) {
                error = "Prefill QSA indexer outputs must be disjoint";
                return false;
            }
        }
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> inputs{{
        {buffers.raw_keys, dispatch.raw_bytes},
        {buffers.gamma, dispatch.gamma_bytes},
        {buffers.rope, dispatch.rope_required_bytes},
    }};
    for (const auto& [output, output_bytes] : outputs) {
        for (const auto& [input, input_bytes] : inputs) {
            if (buffer_overlaps(output, output_bytes, input, input_bytes)) {
                error = "Prefill QSA indexer outputs must not overlap inputs";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

bool encode_qsa_indexer_append(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    QsaIndexerAppendMode mode, std::uint32_t qsa_layer_index,
    float rope_zero_cos, float rope_zero_sin,
    const QsaIndexerAppendBuffers& buffers, std::string& error) {
    IndexerAppendDispatch dispatch;
    if (!prepare_indexer_append_dispatch(
            context, token_rows, cell_capacity, mode, qsa_layer_index,
            rope_zero_cos, rope_zero_sin, buffers, dispatch, error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_indexer_append_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA indexer append compute encoder";
        return false;
    }
    encoder.label = mode == QsaIndexerAppendMode::window
        ? @"QSA indexer append window"
        : @"QSA indexer append commit";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_idx_append")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.raw_keys.handle
                offset:buffers.raw_keys.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.gamma.handle
                offset:buffers.gamma.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.rope.handle
                offset:buffers.rope.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.tail.handle
                offset:buffers.tail.offset atIndex:4u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.dead.handle
                offset:buffers.dead.offset atIndex:5u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.pooled.handle
                offset:buffers.pooled.offset atIndex:6u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.block_position.handle
                offset:buffers.block_position.offset atIndex:7u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)dispatch.tail_snapshot_binding.handle
                offset:dispatch.tail_snapshot_binding.offset atIndex:8u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)dispatch.block_pos_snapshot_binding.handle
                offset:dispatch.block_pos_snapshot_binding.offset atIndex:9u];
    [encoder setBytes:&dispatch.params
               length:sizeof(dispatch.params) atIndex:10u];
    [encoder setThreadgroupMemoryLength:4u * sizeof(float) atIndex:0u];
    [encoder setThreadgroupMemoryLength:
        INDEXER_HEAD_DIM * sizeof(float) atIndex:1u];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kIndexerAppendThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_indexer_append(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    QsaIndexerAppendMode mode, std::uint32_t qsa_layer_index,
    float rope_zero_cos, float rope_zero_sin,
    const QsaIndexerAppendBuffers& buffers, std::string& error) {
    IndexerAppendDispatch dispatch;
    if (!prepare_indexer_append_dispatch(
            context, token_rows, cell_capacity, mode, qsa_layer_index,
            rope_zero_cos, rope_zero_sin, buffers, dispatch, error)) {
        return false;
    }
    if (!validate_indexer_append_pipeline(context, error)) return false;

    const std::array<EncoderBufferBinding, 10> bindings{{
        {buffers.raw_keys, 0u, dispatch.raw_bytes, alignof(float)},
        {buffers.steps, 1u, dispatch.steps_bytes, alignof(StepRow)},
        {buffers.gamma, 2u, dispatch.gamma_bytes, alignof(float)},
        {buffers.rope, 3u, dispatch.rope_required_bytes, kRopeAlignment},
        {buffers.tail, 4u, dispatch.tail_bytes, alignof(float)},
        {buffers.dead, 5u, dispatch.dead_bytes, alignof(float)},
        {buffers.pooled, 6u, dispatch.pooled_bytes, alignof(float)},
        {buffers.block_position, 7u, sizeof(std::int32_t),
         alignof(std::int32_t)},
        {dispatch.tail_snapshot_binding, 8u,
         dispatch.tail_snapshot_binding_bytes, alignof(float)},
        {dispatch.block_pos_snapshot_binding, 9u,
         dispatch.block_pos_snapshot_binding_bytes, alignof(std::int32_t)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 10u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{0u, 4u * sizeof(float)},
        EncoderThreadgroupMemory{
            1u, static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float)},
    };
    return encoder.dispatch(
        "strata_idx_append", {1u, 1u, 1u},
        {kIndexerAppendThreads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

bool encode_prefill_qsa_indexer_append(
    Context& context, Encoder& encoder, const PrefillParams& prefill,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    const QsaPrefillIndexerAppendBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA indexer append requires an ordered concurrent-"
                "group boundary";
        return false;
    }
    PrefillIndexerAppendDispatch dispatch;
    if (!prepare_prefill_indexer_append_dispatch(
            context, prefill, cell_capacity, rope_zero_cos, rope_zero_sin,
            buffers, dispatch, error) ||
        !validate_prefill_indexer_append_pipelines(context, error)) {
        return false;
    }

    const std::array<EncoderThreadgroupMemory, 2> threadgroup_memory{{
        {0u, 4u * sizeof(float)},
        {1u, static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 7u};
    const std::array<EncoderBytesBinding, 1> byte_bindings{params_binding};
    if (dispatch.first_rows != 0u) {
        const std::array<EncoderBufferBinding, 7> bindings{{
            {buffers.raw_keys, 0u, dispatch.raw_bytes, alignof(float)},
            {buffers.gamma, 1u, dispatch.gamma_bytes, alignof(float)},
            {buffers.rope, 2u, dispatch.rope_required_bytes,
             kRopeAlignment},
            {buffers.tail, 3u, dispatch.tail_bytes, alignof(float)},
            {buffers.dead, 4u, dispatch.dead_bytes, alignof(float)},
            {buffers.pooled, 5u, dispatch.pooled_bytes, alignof(float)},
            {buffers.block_position, 6u, sizeof(std::int32_t),
             alignof(std::int32_t)},
        }};
        if (!encoder.dispatch(
                "strata_idx_append_first", {1u, 1u, 1u},
                {kIndexerAppendThreads, 1u, 1u}, bindings, byte_bindings,
                threadgroup_memory, error)) {
            return false;
        }
    }
    if (dispatch.block_count != 0u) {
        const std::array<EncoderBufferBinding, 6> bindings{{
            {buffers.raw_keys, 0u, dispatch.raw_bytes, alignof(float)},
            {buffers.gamma, 1u, dispatch.gamma_bytes, alignof(float)},
            {buffers.rope, 2u, dispatch.rope_required_bytes,
             kRopeAlignment},
            {buffers.dead, 3u, dispatch.dead_bytes, alignof(float)},
            {buffers.pooled, 4u, dispatch.pooled_bytes, alignof(float)},
            {buffers.block_position, 5u, sizeof(std::int32_t),
             alignof(std::int32_t)},
        }};
        const EncoderBytesBinding blocks_params_binding{
            &dispatch.params, sizeof(dispatch.params), 6u};
        const std::array<EncoderBytesBinding, 1> blocks_byte_bindings{
            blocks_params_binding};
        if (!encoder.dispatch(
                "strata_idx_append_blocks",
                {dispatch.block_count, 1u, 1u},
                {kIndexerAppendThreads, 1u, 1u}, bindings,
                blocks_byte_bindings, threadgroup_memory, error)) {
            return false;
        }
    }

    const std::array<EncoderBufferBinding, 2> tail_bindings{{
        {buffers.raw_keys, 0u, dispatch.raw_bytes, alignof(float)},
        {buffers.tail, 1u, dispatch.tail_bytes, alignof(float)},
    }};
    const EncoderBytesBinding tail_params_binding{
        &dispatch.params, sizeof(dispatch.params), 2u};
    const std::array<EncoderBytesBinding, 1> tail_byte_bindings{
        tail_params_binding};
    const std::array<EncoderThreadgroupMemory, 0> no_threadgroup_memory{};
    return encoder.dispatch(
        "strata_idx_append_tail", {INDEXER_BLOCK_CELLS - 1u, 1u, 1u},
        {kIndexerAppendThreads, 1u, 1u}, tail_bindings,
        tail_byte_bindings, no_threadgroup_memory, error);
}

bool encode_prefill_qsa_indexer_append(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& prefill, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    const QsaPrefillIndexerAppendBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill QSA indexer append",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_indexer_append(
                context, encoder, prefill, cell_capacity, rope_zero_cos,
                rope_zero_sin, buffers, encode_error);
        },
        error);
}

bool encode_qsa_indexer_commit_batch(
    Context& context, Encoder& encoder, std::uint32_t accepted_rows,
    std::uint32_t cell_capacity, float rope_zero_cos, float rope_zero_sin,
    std::span<const QsaIndexerAppendBuffers> layers, std::string& error) {
    error.clear();
    if (accepted_rows == 0u || accepted_rows > VERIFY_MAX_T) {
        error = "QSA indexer commit batch row count must be in 1..8";
        return false;
    }
    if (layers.size() != kQsaIndexerLayers) {
        error = "QSA indexer commit batch requires exactly 12 layers";
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error =
            "QSA indexer commit batch requires an ordered concurrent-group boundary";
        return false;
    }

    struct BufferExtent {
        BufferRef view;
        std::size_t bytes;
        std::uint32_t layer;
        const char* name;
    };
    std::vector<BufferExtent> outputs;
    outputs.reserve(static_cast<std::size_t>(kQsaIndexerLayers) * 4u);
    std::vector<BufferExtent> inputs;
    inputs.reserve(static_cast<std::size_t>(kQsaIndexerLayers) * 6u);
    for (std::uint32_t layer = 0u; layer < kQsaIndexerLayers; ++layer) {
        IndexerAppendDispatch dispatch;
        if (!prepare_indexer_append_dispatch(
                context, accepted_rows, cell_capacity,
                QsaIndexerAppendMode::commit, layer, rope_zero_cos,
                rope_zero_sin, layers[layer], dispatch, error)) {
            error = "QSA indexer commit batch layer " +
                    std::to_string(layer) + ": " + error;
            return false;
        }
        if (layer != 0u &&
            (!same_view(layers[layer].steps, layers[0].steps) ||
             !same_view(layers[layer].rope, layers[0].rope) ||
             !same_view(layers[layer].tail_snapshot,
                        layers[0].tail_snapshot) ||
             !same_view(layers[layer].block_pos_snapshot,
                        layers[0].block_pos_snapshot))) {
            error = "QSA indexer commit batch must share steps, RoPE and "
                    "snapshot views across all layers";
            return false;
        }

        outputs.push_back(
            {layers[layer].tail, dispatch.tail_bytes, layer, "tail"});
        outputs.push_back(
            {layers[layer].dead, dispatch.dead_bytes, layer, "dead"});
        outputs.push_back(
            {layers[layer].pooled, dispatch.pooled_bytes, layer, "pooled"});
        outputs.push_back({layers[layer].block_position,
                           sizeof(std::int32_t), layer, "block position"});

        inputs.push_back({layers[layer].raw_keys, dispatch.raw_bytes, layer,
                          "raw keys"});
        inputs.push_back(
            {layers[layer].steps, dispatch.steps_bytes, layer, "step rows"});
        inputs.push_back(
            {layers[layer].gamma, dispatch.gamma_bytes, layer, "gamma"});
        inputs.push_back({layers[layer].rope, dispatch.rope_required_bytes,
                          layer, "RoPE"});
        inputs.push_back({layers[layer].tail_snapshot,
                          dispatch.tail_snapshot_bytes, layer,
                          "tail snapshot"});
        inputs.push_back({layers[layer].block_pos_snapshot,
                          dispatch.block_pos_snapshot_bytes, layer,
                          "block-position snapshot"});
    }

    for (std::size_t output_index = 0u;
         output_index < outputs.size(); ++output_index) {
        const BufferExtent& output = outputs[output_index];
        for (std::size_t prior = 0u; prior < output_index; ++prior) {
            const BufferExtent& other = outputs[prior];
            if (buffer_overlaps(output.view, output.bytes, other.view,
                                other.bytes)) {
                error = "QSA indexer commit batch layer " +
                        std::to_string(output.layer) + " " + output.name +
                        " overlaps layer " + std::to_string(other.layer) +
                        " " + other.name;
                return false;
            }
        }
        for (const BufferExtent& input : inputs) {
            if (buffer_overlaps(output.view, output.bytes, input.view,
                                input.bytes)) {
                error = "QSA indexer commit batch layer " +
                        std::to_string(output.layer) + " " + output.name +
                        " overlaps layer " + std::to_string(input.layer) +
                        " " + input.name;
                return false;
            }
        }
    }
    if (!validate_indexer_append_pipeline(context, error)) return false;

    for (std::uint32_t layer = 0u; layer < kQsaIndexerLayers; ++layer) {
        if (!encode_qsa_indexer_append(
                context, encoder, accepted_rows, cell_capacity,
                QsaIndexerAppendMode::commit, layer, rope_zero_cos,
                rope_zero_sin, layers[layer], error)) {
            error = "QSA indexer commit batch layer " +
                    std::to_string(layer) + ": " + error;
            return false;
        }
    }
    error.clear();
    return true;
}

bool encode_qsa_indexer_commit_batch(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t accepted_rows, std::uint32_t cell_capacity,
    float rope_zero_cos, float rope_zero_sin,
    std::span<const QsaIndexerAppendBuffers> layers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "QSA indexer accepted-prefix commit",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_qsa_indexer_commit_batch(
                context, encoder, accepted_rows, cell_capacity,
                rope_zero_cos, rope_zero_sin, layers, encode_error);
        },
        error);
}

static bool prepare_selection_scores_dispatch(
    Context& context, std::uint32_t token_rows,
    std::uint32_t cell_capacity,
    const QsaSelectionScoresBuffers& buffers,
    SelectionScoresDispatch& dispatch, std::string& error) {
    error.clear();
    if (token_rows == 0u || token_rows > VERIFY_MAX_T) {
        error = "QSA selection-score row count must be in 1..8";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        cell_capacity >
            static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())) {
        error = "QSA selection-score cell capacity must be a positive "
                "multiple of four addressable by StepRow";
        return false;
    }

    constexpr std::size_t query_row_bytes =
        static_cast<std::size_t>(INDEXER_HEADS) * INDEXER_HEAD_DIM *
        sizeof(float);
    constexpr std::size_t key_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t score_element_bytes = sizeof(float);
    const std::size_t max_size =
        std::numeric_limits<std::size_t>::max();
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    if (max_blocks > max_size / key_row_bytes ||
        max_blocks > max_size / score_element_bytes ||
        static_cast<std::size_t>(token_rows) >
            max_size / (max_blocks * score_element_bytes)) {
        error = "QSA selection-score buffer size overflows host address space";
        return false;
    }
    dispatch.pooled_bytes = max_blocks * key_row_bytes;
    dispatch.dead_bytes = key_row_bytes;
    dispatch.query_bytes =
        static_cast<std::size_t>(token_rows) * query_row_bytes;
    dispatch.steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    dispatch.scores_bytes =
        static_cast<std::size_t>(token_rows) * max_blocks *
        score_element_bytes;
    if (!validate_buffer(context, buffers.pooled, dispatch.pooled_bytes,
                         kSelectionVectorAlignment,
                         "selection pooled keys", error) ||
        !validate_buffer(context, buffers.dead, dispatch.dead_bytes,
                         kSelectionVectorAlignment,
                         "selection dead key", error) ||
        !validate_buffer(context, buffers.indexer_queries,
                         dispatch.query_bytes,
                         alignof(float), "selection indexer queries", error) ||
        !validate_buffer(context, buffers.steps, dispatch.steps_bytes,
                         alignof(StepRow), "selection step rows", error) ||
        !validate_buffer(context, buffers.scores, dispatch.scores_bytes,
                         alignof(float), "selection scores", error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    std::uint32_t top = 0u;
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const StepRow& step = step_rows[row];
        if (step.pos < 0 ||
            static_cast<std::uint32_t>(step.pos) >= cell_capacity ||
            step.n_kv != step.pos + 1 ||
            step.n_bid != step.n_kv / INDEXER_BLOCK_CELLS ||
            step.width != std::min<std::int32_t>(
                step.n_kv,
                static_cast<std::int32_t>(
                    INDEXER_SELECTION_WIDTH_MAX))) {
            error = "QSA selection-score StepRow is inconsistent with its "
                    "position";
            return false;
        }
        top = std::max(top, static_cast<std::uint32_t>(step.n_bid));
    }

    const std::array<std::pair<BufferRef, std::size_t>, 4> inputs{{
        {buffers.pooled, dispatch.pooled_bytes},
        {buffers.dead, dispatch.dead_bytes},
        {buffers.indexer_queries, dispatch.query_bytes},
        {buffers.steps, dispatch.steps_bytes},
    }};
    for (const auto& [input, input_bytes] : inputs) {
        if (buffer_overlaps(buffers.scores, dispatch.scores_bytes,
                            input, input_bytes)) {
            error = "QSA selection scores must not overlap an input";
            return false;
        }
    }

    const std::uint32_t active_blocks = top + 1u;
    dispatch.grid_count_x = std::min(
        256u, active_blocks / 8u + (active_blocks % 8u == 0u ? 0u : 1u));
    dispatch.params = SelectionScoresParams{
        token_rows, static_cast<std::uint32_t>(max_blocks),
        dispatch.grid_count_x};
    error.clear();
    return true;
}

static bool prepare_prefill_selection_dispatch(
    Context& context, const PrefillParams& prefill,
    std::uint32_t cell_capacity,
    const QsaPrefillSelectionBuffers& buffers,
    PrefillSelectionDispatch& dispatch, std::string& error) {
    error.clear();
    if (prefill.T == 0u || prefill.T > kPrefillMaxShaderRows) {
        error = "Prefill QSA selection row count exceeds the shader-row "
                "limit";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        cell_capacity > static_cast<std::uint32_t>(
                            std::numeric_limits<std::int32_t>::max())) {
        error = "Prefill QSA selection cell capacity is invalid";
        return false;
    }

    const std::uint64_t end =
        static_cast<std::uint64_t>(prefill.p0) + prefill.T;
    if (prefill.p0 >= cell_capacity || end > cell_capacity ||
        end > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int32_t>::max())) {
        error = "Prefill QSA selection chunk exceeds the cell capacity";
        return false;
    }
    const std::uint64_t expected_idx_f =
        prefill.p0 == 0u
            ? INDEXER_BLOCK_CELLS
            : (static_cast<std::uint64_t>(prefill.p0) +
               INDEXER_BLOCK_CELLS - 1u) &
                  ~static_cast<std::uint64_t>(
                      INDEXER_BLOCK_CELLS - 1u);
    const std::uint32_t expected_idx_e = static_cast<std::uint32_t>(
        end & ~static_cast<std::uint64_t>(INDEXER_BLOCK_CELLS - 1u));
    if (expected_idx_f > std::numeric_limits<std::uint32_t>::max() ||
        prefill.idx_f != expected_idx_f ||
        prefill.idx_e != expected_idx_e) {
        error = "Prefill QSA selection parameters do not match the chunk";
        return false;
    }

    constexpr std::size_t key_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t query_row_bytes =
        static_cast<std::size_t>(INDEXER_HEADS) * key_row_bytes;
    constexpr std::size_t step_row_bytes = sizeof(StepRow);
    constexpr std::size_t id_row_bytes =
        static_cast<std::size_t>(INDEXER_SELECTION_WIDTH_MAX) *
        sizeof(std::int32_t);
    dispatch.max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    if (dispatch.max_blocks >
            std::numeric_limits<std::uint32_t>::max() ||
        !checked_product({dispatch.max_blocks, key_row_bytes},
                         dispatch.pooled_bytes) ||
        !checked_product({key_row_bytes}, dispatch.dead_bytes) ||
        !checked_product({prefill.T, query_row_bytes},
                         dispatch.query_bytes) ||
        !checked_product({prefill.T, step_row_bytes},
                         dispatch.steps_bytes) ||
        !checked_product(
            {kPrefillSelectionGroupRows, dispatch.max_blocks, sizeof(float)},
            dispatch.scores_bytes) ||
        !checked_product({prefill.T, id_row_bytes}, dispatch.ids_bytes) ||
        !checked_product({dispatch.max_blocks, sizeof(float)},
                         dispatch.score_row_bytes)) {
        error = "Prefill QSA selection buffer size overflows host address "
                "space";
        return false;
    }
    dispatch.query_row_bytes = query_row_bytes;
    dispatch.step_row_bytes = step_row_bytes;
    dispatch.id_row_bytes = id_row_bytes;

    if (!validate_buffer(context, buffers.pooled, dispatch.pooled_bytes,
                         kSelectionVectorAlignment,
                         "Prefill selection pooled keys", error) ||
        !validate_buffer(context, buffers.dead, dispatch.dead_bytes,
                         kSelectionVectorAlignment,
                         "Prefill selection dead key", error) ||
        !validate_buffer(context, buffers.indexer_queries,
                         dispatch.query_bytes, alignof(float),
                         "Prefill selection indexer queries", error) ||
        !validate_buffer(context, buffers.steps, dispatch.steps_bytes,
                         alignof(StepRow), "Prefill selection step rows",
                         error) ||
        !validate_buffer(context, buffers.scores, dispatch.scores_bytes,
                         alignof(float), "Prefill selection score scratch",
                         error) ||
        !validate_buffer(context, buffers.ids, dispatch.ids_bytes,
                         alignof(std::int32_t),
                         "Prefill selection cell IDs", error)) {
        return false;
    }

    static_assert(sizeof(PrefillStepRow) == sizeof(StepRow));
    const auto* steps =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    for (std::uint32_t row = 0u; row < prefill.T; ++row) {
        const std::uint64_t position =
            static_cast<std::uint64_t>(prefill.p0) + row;
        const std::int32_t n_kv = static_cast<std::int32_t>(position + 1u);
        const StepRow expected{
            static_cast<std::int32_t>(position),
            n_kv,
            n_kv / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(
                n_kv,
                static_cast<std::int32_t>(
                    INDEXER_SELECTION_WIDTH_MAX)),
        };
        if (steps[row].pos != expected.pos ||
            steps[row].n_kv != expected.n_kv ||
            steps[row].n_bid != expected.n_bid ||
            steps[row].width != expected.width) {
            error = "Prefill QSA selection StepRows must match the chunk's "
                    "contiguous positions";
            return false;
        }
    }

    const std::array<std::pair<BufferRef, std::size_t>, 4> inputs{{
        {buffers.pooled, dispatch.pooled_bytes},
        {buffers.dead, dispatch.dead_bytes},
        {buffers.indexer_queries, dispatch.query_bytes},
        {buffers.steps, dispatch.steps_bytes},
    }};
    const std::array<std::pair<BufferRef, std::size_t>, 2> outputs{{
        {buffers.scores, dispatch.scores_bytes},
        {buffers.ids, dispatch.ids_bytes},
    }};
    if (buffer_overlaps(outputs[0].first, outputs[0].second,
                        outputs[1].first, outputs[1].second)) {
        error = "Prefill QSA selection score and ID outputs must be "
                "disjoint";
        return false;
    }
    for (const auto& [output, output_bytes] : outputs) {
        for (const auto& [input, input_bytes] : inputs) {
            if (buffer_overlaps(output, output_bytes, input, input_bytes)) {
                error = "Prefill QSA selection outputs must not overlap "
                        "inputs";
                return false;
            }
        }
    }

    const bool register_form =
        dispatch.max_blocks <= kSelectionTopkRegMaxBlocks;
    if (!validate_selection_scores_pipeline(context, error) ||
        !validate_selection_topk_pipeline(
            context,
            register_form ? "strata_sel_topk_reg"
                          : "strata_sel_topk_mem",
            register_form ? kSelectionTopkRegThreads
                          : kSelectionTopkMemThreads,
            error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_qsa_selection_scores(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionScoresBuffers& buffers, std::string& error) {
    SelectionScoresDispatch dispatch;
    if (!prepare_selection_scores_dispatch(
            context, token_rows, cell_capacity, buffers, dispatch, error)) {
        return false;
    }
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_selection_scores_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA selection-score compute encoder";
        return false;
    }
    encoder.label = @"QSA selection scores";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_sel_scores")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.pooled.handle
                offset:buffers.pooled.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.dead.handle
                offset:buffers.dead.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.indexer_queries.handle
                offset:buffers.indexer_queries.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:3u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.scores.handle
                offset:buffers.scores.offset atIndex:4u];
    [encoder setBytes:&dispatch.params length:sizeof(dispatch.params)
               atIndex:5u];
    [encoder setThreadgroupMemoryLength:
        static_cast<NSUInteger>(VERIFY_MAX_T) * INDEXER_HEADS *
            INDEXER_HEAD_DIM * sizeof(float)
        atIndex:0u];
    [encoder setThreadgroupMemoryLength:
        static_cast<NSUInteger>(VERIFY_MAX_T) * sizeof(StepRow)
        atIndex:1u];
    [encoder dispatchThreadgroups:MTLSizeMake(dispatch.grid_count_x, 1u, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(kSelectionScoresThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_selection_scores(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionScoresBuffers& buffers, std::string& error) {
    SelectionScoresDispatch dispatch;
    if (!prepare_selection_scores_dispatch(
            context, token_rows, cell_capacity, buffers, dispatch, error)) {
        return false;
    }
    if (!validate_selection_scores_pipeline(context, error)) return false;

    const std::array<EncoderBufferBinding, 5> bindings{{
        {buffers.pooled, 0u, dispatch.pooled_bytes,
         kSelectionVectorAlignment},
        {buffers.dead, 1u, dispatch.dead_bytes,
         kSelectionVectorAlignment},
        {buffers.indexer_queries, 2u, dispatch.query_bytes,
         alignof(float)},
        {buffers.steps, 3u, dispatch.steps_bytes, alignof(StepRow)},
        {buffers.scores, 4u, dispatch.scores_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 5u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{
            0u, static_cast<std::size_t>(VERIFY_MAX_T) * INDEXER_HEADS *
                    INDEXER_HEAD_DIM * sizeof(float)},
        EncoderThreadgroupMemory{
            1u, static_cast<std::size_t>(VERIFY_MAX_T) * sizeof(StepRow)},
    };
    return encoder.dispatch(
        "strata_sel_scores", {dispatch.grid_count_x, 1u, 1u},
        {kSelectionScoresThreads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

static bool prepare_selection_topk_dispatch(
    Context& context, std::uint32_t token_rows,
    std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers,
    SelectionTopkDispatch& dispatch, std::string& error,
    std::uint32_t row_limit = VERIFY_MAX_T) {
    error.clear();
    if (token_rows == 0u || token_rows > row_limit) {
        error = row_limit == VERIFY_MAX_T
            ? "QSA top-k row count must be in 1..8"
            : "Prefill QSA top-k row count exceeds its 256-row batch limit";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        cell_capacity >
            static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max())) {
        error = "QSA top-k cell capacity must be a positive multiple of four "
                "addressable by StepRow";
        return false;
    }

    const std::size_t max_size =
        std::numeric_limits<std::size_t>::max();
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    constexpr std::size_t score_element_bytes = sizeof(float);
    constexpr std::size_t id_row_bytes =
        static_cast<std::size_t>(INDEXER_SELECTION_WIDTH_MAX) *
        sizeof(std::int32_t);
    if (max_blocks > max_size / score_element_bytes ||
        static_cast<std::size_t>(token_rows) >
            max_size / (max_blocks * score_element_bytes) ||
        static_cast<std::size_t>(token_rows) >
            max_size / id_row_bytes) {
        error = "QSA top-k buffer size overflows host address space";
        return false;
    }
    dispatch.scores_bytes =
        static_cast<std::size_t>(token_rows) * max_blocks *
        score_element_bytes;
    dispatch.steps_bytes =
        static_cast<std::size_t>(token_rows) * sizeof(StepRow);
    dispatch.ids_bytes =
        static_cast<std::size_t>(token_rows) * id_row_bytes;
    if (!validate_buffer(context, buffers.scores, dispatch.scores_bytes,
                         alignof(float), "top-k block scores", error) ||
        !validate_buffer(context, buffers.steps, dispatch.steps_bytes,
                         alignof(StepRow), "top-k step rows", error) ||
        !validate_buffer(context, buffers.ids, dispatch.ids_bytes,
                         alignof(std::int32_t), "top-k cell ids", error)) {
        return false;
    }

    const auto* step_rows =
        reinterpret_cast<const StepRow*>(buffers.steps.contents);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const StepRow& step = step_rows[row];
        if (step.pos < 0 ||
            static_cast<std::uint32_t>(step.pos) >= cell_capacity ||
            step.n_kv != step.pos + 1 ||
            step.n_bid !=
                step.n_kv /
                    static_cast<std::int32_t>(INDEXER_BLOCK_CELLS) ||
            step.width != std::min<std::int32_t>(
                              step.n_kv,
                              static_cast<std::int32_t>(
                                  INDEXER_SELECTION_WIDTH_MAX))) {
            error = "QSA top-k StepRow is inconsistent with its position";
            return false;
        }
    }
    if (buffer_overlaps(buffers.ids, dispatch.ids_bytes,
                        buffers.scores, dispatch.scores_bytes) ||
        buffer_overlaps(buffers.ids, dispatch.ids_bytes,
                        buffers.steps, dispatch.steps_bytes)) {
        error = "QSA top-k id output must not overlap scores or steps";
        return false;
    }

    const bool register_form =
        max_blocks <= kSelectionTopkRegMaxBlocks;
    dispatch.kernel_name = register_form
        ? "strata_sel_topk_reg"
        : "strata_sel_topk_mem";
    dispatch.threads = register_form
        ? kSelectionTopkRegThreads
        : kSelectionTopkMemThreads;
    dispatch.histogram_banks = register_form ? 8u : 1u;
    dispatch.params = SelectionTopkParams{
        token_rows, static_cast<std::uint32_t>(max_blocks)};
    error.clear();
    return true;
}

bool encode_qsa_selection_topk(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers, std::string& error) {
    SelectionTopkDispatch dispatch;
    if (!prepare_selection_topk_dispatch(
            context, token_rows, cell_capacity, buffers, dispatch, error)) {
        return false;
    }
    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_selection_topk_pipeline(
            context, dispatch.kernel_name, dispatch.threads, error)) {
        return false;
    }
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;

    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA top-k compute encoder";
        return false;
    }
    encoder.label = dispatch.histogram_banks == 8u
        ? @"QSA selection top-k register"
        : @"QSA selection top-k memory";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            dispatch.kernel_name.data())];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.scores.handle
                offset:buffers.scores.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.ids.handle
                offset:buffers.ids.offset atIndex:2u];
    [encoder setBytes:&dispatch.params length:sizeof(dispatch.params)
               atIndex:3u];
    [encoder setThreadgroupMemoryLength:
        static_cast<NSUInteger>(dispatch.histogram_banks) * 256u *
            sizeof(std::uint32_t)
        atIndex:0u];
    [encoder setThreadgroupMemoryLength:
        static_cast<NSUInteger>(dispatch.threads) * sizeof(std::uint32_t)
        atIndex:1u];
    [encoder setThreadgroupMemoryLength:
        static_cast<NSUInteger>(dispatch.threads) * sizeof(std::uint32_t)
        atIndex:2u];
    [encoder dispatchThreadgroups:MTLSizeMake(token_rows, 1u, 1u)
              threadsPerThreadgroup:
                  MTLSizeMake(dispatch.threads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

static bool encode_selection_topk_encoder(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers, std::string& error,
    std::uint32_t row_limit = VERIFY_MAX_T) {
    SelectionTopkDispatch dispatch;
    if (!prepare_selection_topk_dispatch(
            context, token_rows, cell_capacity, buffers, dispatch, error,
            row_limit)) {
        return false;
    }
    if (!validate_selection_topk_pipeline(
            context, dispatch.kernel_name, dispatch.threads, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 3> bindings{{
        {buffers.scores, 0u, dispatch.scores_bytes, alignof(float)},
        {buffers.steps, 1u, dispatch.steps_bytes, alignof(StepRow)},
        {buffers.ids, 2u, dispatch.ids_bytes, alignof(std::int32_t)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 3u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{
            0u, static_cast<std::size_t>(dispatch.histogram_banks) * 256u *
                    sizeof(std::uint32_t)},
        EncoderThreadgroupMemory{
            1u, static_cast<std::size_t>(dispatch.threads) *
                    sizeof(std::uint32_t)},
        EncoderThreadgroupMemory{
            2u, static_cast<std::size_t>(dispatch.threads) *
                    sizeof(std::uint32_t)},
    };
    return encoder.dispatch(
        dispatch.kernel_name, {token_rows, 1u, 1u},
        {dispatch.threads, 1u, 1u}, bindings, byte_bindings,
        threadgroup_memory, error);
}

bool encode_qsa_selection_topk(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    const QsaSelectionTopkBuffers& buffers, std::string& error) {
    return encode_selection_topk_encoder(
        context, encoder, token_rows, cell_capacity, buffers, error);
}

bool encode_prefill_qsa_selection(
    Context& context, Encoder& encoder, const PrefillParams& prefill,
    std::uint32_t cell_capacity,
    const QsaPrefillSelectionBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA selection requires an ordered concurrent-group "
                "boundary";
        return false;
    }
    PrefillSelectionDispatch dispatch;
    if (!prepare_prefill_selection_dispatch(
            context, prefill, cell_capacity, buffers, dispatch, error)) {
        return false;
    }

    for (std::uint32_t group_base = 0u; group_base < prefill.T;
         group_base += kPrefillSelectionGroupRows) {
        const std::uint32_t group_rows = std::min(
            kPrefillSelectionGroupRows, prefill.T - group_base);
        for (std::uint32_t local_base = 0u; local_base < group_rows;
             local_base += VERIFY_MAX_T) {
            const std::uint32_t score_rows =
                std::min(VERIFY_MAX_T, group_rows - local_base);
            const std::uint32_t input_row = group_base + local_base;
            const std::size_t query_offset =
                static_cast<std::size_t>(input_row) *
                dispatch.query_row_bytes;
            const std::size_t step_offset =
                static_cast<std::size_t>(input_row) *
                dispatch.step_row_bytes;
            const std::size_t score_offset =
                static_cast<std::size_t>(local_base) *
                dispatch.score_row_bytes;
            const std::size_t query_bytes =
                static_cast<std::size_t>(score_rows) *
                dispatch.query_row_bytes;
            const std::size_t step_bytes =
                static_cast<std::size_t>(score_rows) *
                dispatch.step_row_bytes;
            const std::size_t score_bytes =
                static_cast<std::size_t>(score_rows) *
                dispatch.score_row_bytes;
            const QsaSelectionScoresBuffers score_buffers{
                buffers.pooled,
                buffers.dead,
                buffers.indexer_queries.slice(query_offset, query_bytes),
                buffers.steps.slice(step_offset, step_bytes),
                buffers.scores.slice(score_offset, score_bytes),
            };
            if (!encode_qsa_selection_scores(
                    context, encoder, score_rows, cell_capacity,
                    score_buffers, error)) {
                return false;
            }
        }

        const std::size_t group_score_bytes =
            static_cast<std::size_t>(group_rows) *
            dispatch.score_row_bytes;
        const std::size_t group_step_offset =
            static_cast<std::size_t>(group_base) *
            dispatch.step_row_bytes;
        const std::size_t group_step_bytes =
            static_cast<std::size_t>(group_rows) *
            dispatch.step_row_bytes;
        const std::size_t group_id_offset =
            static_cast<std::size_t>(group_base) * dispatch.id_row_bytes;
        const std::size_t group_id_bytes =
            static_cast<std::size_t>(group_rows) * dispatch.id_row_bytes;
        const QsaSelectionTopkBuffers topk_buffers{
            buffers.scores.slice(0u, group_score_bytes),
            buffers.steps.slice(group_step_offset, group_step_bytes),
            buffers.ids.slice(group_id_offset, group_id_bytes),
        };
        if (!encode_selection_topk_encoder(
                context, encoder, group_rows, cell_capacity, topk_buffers,
                error, kPrefillSelectionGroupRows)) {
            return false;
        }
    }
    error.clear();
    return true;
}

bool encode_prefill_qsa_selection(
    Context& context, CommandBuffer& command_buffer,
    const PrefillParams& prefill, std::uint32_t cell_capacity,
    const QsaPrefillSelectionBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill QSA selection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_selection(
                context, encoder, prefill, cell_capacity, buffers,
                encode_error);
        },
        error);
}

static bool prepare_qsa_gate_dispatch(
    Context& context, std::uint32_t token_rows,
    const QsaGateBuffers& buffers, QsaGateDispatch& dispatch,
    std::string& error) {
    error.clear();
    if (token_rows == 0u || token_rows > VERIFY_MAX_T) {
        error = "QSA gate row count must be in 1..8";
        return false;
    }

    constexpr std::size_t attn_row_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    constexpr std::size_t q_full_row_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * sizeof(float);
    if (!checked_product({token_rows, attn_row_bytes}, dispatch.attn_bytes) ||
        !checked_product({token_rows, q_full_row_bytes},
                         dispatch.q_full_bytes)) {
        error = "QSA gate buffer size overflows host address space";
        return false;
    }
    if (!validate_buffer(context, buffers.attn, dispatch.attn_bytes,
                         alignof(float),
                         "gate attention input", error) ||
        !validate_buffer(context, buffers.q_full, dispatch.q_full_bytes,
                         alignof(float), "gate q_full input", error) ||
        !validate_buffer(context, buffers.output, dispatch.attn_bytes,
                         alignof(float), "gate output", error)) {
        return false;
    }
    if (buffer_overlaps(buffers.output, dispatch.attn_bytes,
                        buffers.attn, dispatch.attn_bytes) ||
        buffer_overlaps(buffers.output, dispatch.attn_bytes,
                        buffers.q_full, dispatch.q_full_bytes)) {
        error = "QSA gate output must be disjoint from attn and q_full";
        return false;
    }
    dispatch.params = QsaGateParams{token_rows};
    error.clear();
    return true;
}

bool encode_qsa_gate(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t token_rows,
                     const QsaGateBuffers& buffers, std::string& error) {
    QsaGateDispatch dispatch;
    if (!prepare_qsa_gate_dispatch(context, token_rows, buffers, dispatch,
                                   error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_qsa_gate_pipeline(context, error)) return false;
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA gate compute encoder";
        return false;
    }

    encoder.label = @"QSA attention gate";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_qsa_gate")];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.attn.handle
                offset:buffers.attn.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.q_full.handle
                offset:buffers.q_full.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.output.handle
                offset:buffers.output.offset atIndex:2u];
    [encoder setBytes:&dispatch.params length:sizeof(dispatch.params)
               atIndex:3u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(N_HEAD_Q, token_rows, 1u)
              threadsPerThreadgroup:MTLSizeMake(kQsaGateThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_gate(Context& context, Encoder& encoder,
                     std::uint32_t token_rows,
                     const QsaGateBuffers& buffers, std::string& error) {
    QsaGateDispatch dispatch;
    if (!prepare_qsa_gate_dispatch(context, token_rows, buffers, dispatch,
                                   error)) {
        return false;
    }
    if (!validate_qsa_gate_pipeline(context, error)) return false;

    const std::array<EncoderBufferBinding, 3> bindings{{
        {buffers.attn, 0u, dispatch.attn_bytes, alignof(float)},
        {buffers.q_full, 1u, dispatch.q_full_bytes, alignof(float)},
        {buffers.output, 2u, dispatch.attn_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 3u};
    const std::array byte_bindings{params_binding};
    return encoder.dispatch(
        "strata_qsa_gate", {N_HEAD_Q, token_rows, 1u},
        {kQsaGateThreads, 1u, 1u}, bindings, byte_bindings, {}, error);
}

static bool prepare_qsa_attention_chunk_dispatch(
    Context& context, QsaAttentionMode mode, std::uint32_t token_rows,
    std::uint32_t cell_capacity, std::uint32_t id_row_stride,
    const QsaAttentionChunkBuffers& buffers,
    QsaAttentionChunkDispatch& dispatch, std::string& error) {
    error.clear();
    dispatch.kernel_name = qsa_attention_kernel_name(mode);
    if (dispatch.kernel_name == nullptr) {
        error = "QSA attention KV mode is invalid";
        return false;
    }
    if (cell_capacity == 0u ||
        cell_capacity > kQsaAttentionMaxCells ||
        cell_capacity % INDEXER_BLOCK_CELLS != 0u ||
        id_row_stride == 0u) {
        error = "QSA attention cell capacity or id row stride is invalid";
        return false;
    }

    if (!qsa_attention_extents(
            token_rows, id_row_stride, dispatch.chunk_count,
            dispatch.partial_acc_bytes, dispatch.partial_stats_bytes,
            dispatch.output_bytes)) {
        error = "QSA attention row count or scratch extent is invalid";
        return false;
    }

    constexpr std::size_t query_row_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    std::size_t pool_rows = 0u;
    if (!checked_product({token_rows, query_row_bytes},
                         dispatch.query_bytes) ||
        !checked_product({token_rows, sizeof(StepRow)},
                         dispatch.steps_bytes) ||
        !checked_product({token_rows, id_row_stride, sizeof(std::int32_t)},
                         dispatch.ids_bytes) ||
        !checked_product({cell_capacity, N_HEAD_KV}, pool_rows)) {
        error = "QSA attention input extent overflows host address space";
        return false;
    }

    std::size_t i8_code_bytes = 0u;
    std::size_t i8_scale_bytes = 0u;
    std::size_t q4_block_bytes = 0u;
    if (!checked_product({pool_rows, HEAD_DIM}, i8_code_bytes) ||
        !checked_product({pool_rows, 4u, sizeof(std::uint16_t)},
                         i8_scale_bytes) ||
        !checked_product({pool_rows, 8u, sizeof(block_q4_0)},
                         q4_block_bytes)) {
        error = "QSA attention KV extent overflows host address space";
        return false;
    }

    bool key_uses_scales = false;
    bool value_uses_scales = false;
    dispatch.key_data_alignment = alignof(std::uint8_t);
    dispatch.value_data_alignment = alignof(std::uint8_t);
    switch (mode) {
        case QsaAttentionMode::f16:
            if (!checked_product(
                    {pool_rows, HEAD_DIM, sizeof(std::uint16_t)},
                    dispatch.key_data_bytes) ||
                !checked_product(
                    {pool_rows, HEAD_DIM, sizeof(std::uint16_t)},
                    dispatch.value_data_bytes)) {
                error = "QSA attention fp16 KV extent overflows";
                return false;
            }
            dispatch.key_data_alignment = alignof(std::uint16_t);
            dispatch.value_data_alignment = alignof(std::uint16_t);
            break;
        case QsaAttentionMode::i8:
            dispatch.key_data_bytes = i8_code_bytes;
            dispatch.key_scale_bytes = i8_scale_bytes;
            dispatch.value_data_bytes = i8_code_bytes;
            dispatch.value_scale_bytes = i8_scale_bytes;
            key_uses_scales = true;
            value_uses_scales = true;
            dispatch.key_data_alignment = alignof(std::int8_t);
            dispatch.value_data_alignment = alignof(std::int8_t);
            break;
        case QsaAttentionMode::q4:
            dispatch.key_data_bytes = q4_block_bytes;
            dispatch.value_data_bytes = q4_block_bytes;
            dispatch.key_data_alignment = alignof(block_q4_0);
            dispatch.value_data_alignment = alignof(block_q4_0);
            break;
        case QsaAttentionMode::k8v4:
            dispatch.key_data_bytes = i8_code_bytes;
            dispatch.key_scale_bytes = i8_scale_bytes;
            dispatch.value_data_bytes = q4_block_bytes;
            key_uses_scales = true;
            dispatch.key_data_alignment = alignof(std::int8_t);
            dispatch.value_data_alignment = alignof(block_q4_0);
            break;
    }

    if (!validate_buffer(context, buffers.query, dispatch.query_bytes,
                         alignof(float), "attention query", error) ||
        !validate_buffer(context, buffers.steps, dispatch.steps_bytes,
                         alignof(StepRow), "attention steps", error) ||
        !validate_buffer(context, buffers.ids, dispatch.ids_bytes,
                         alignof(std::int32_t), "attention ids", error) ||
        !validate_buffer(context, buffers.key_data,
                         dispatch.key_data_bytes,
                         dispatch.key_data_alignment,
                         "attention key pool", error) ||
        (key_uses_scales &&
         !validate_buffer(context, buffers.key_scales,
                          dispatch.key_scale_bytes,
                          alignof(std::uint16_t),
                          "attention key scales", error)) ||
        !validate_buffer(context, buffers.value_data,
                         dispatch.value_data_bytes,
                         dispatch.value_data_alignment,
                         "attention value pool", error) ||
        (value_uses_scales &&
         !validate_buffer(context, buffers.value_scales,
                          dispatch.value_scale_bytes,
                          alignof(std::uint16_t),
                          "attention value scales", error)) ||
        !validate_buffer(context, buffers.scratch.partial_acc,
                         dispatch.partial_acc_bytes, alignof(float),
                         "attention partial accumulators", error) ||
        !validate_buffer(context, buffers.scratch.partial_m,
                         dispatch.partial_stats_bytes, alignof(float),
                         "attention partial maxima", error) ||
        !validate_buffer(context, buffers.scratch.partial_l,
                         dispatch.partial_stats_bytes, alignof(float),
                         "attention partial denominators", error)) {
        return false;
    }

    const auto* steps =
        static_cast<const StepRow*>(buffers.steps.contents);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const StepRow& step = steps[row];
        if (step.pos < 0 ||
            static_cast<std::uint32_t>(step.pos) >= cell_capacity ||
            step.n_kv != step.pos + 1 ||
            step.n_bid != step.n_kv /
                              static_cast<std::int32_t>(
                                  INDEXER_BLOCK_CELLS) ||
            step.width <= 0 ||
            static_cast<std::uint32_t>(step.width) > id_row_stride ||
            step.width > step.n_kv) {
            error = "QSA attention StepRow is inconsistent with its "
                    "cell position or id row";
            return false;
        }
    }

    const std::array<std::pair<BufferRef, std::size_t>, 7> reads{{
        {buffers.query, dispatch.query_bytes},
        {buffers.steps, dispatch.steps_bytes},
        {buffers.ids, dispatch.ids_bytes},
        {buffers.key_data, dispatch.key_data_bytes},
        {buffers.key_scales, dispatch.key_scale_bytes},
        {buffers.value_data, dispatch.value_data_bytes},
        {buffers.value_scales, dispatch.value_scale_bytes},
    }};
    const std::array<std::pair<BufferRef, std::size_t>, 3> writes{{
        {buffers.scratch.partial_acc, dispatch.partial_acc_bytes},
        {buffers.scratch.partial_m, dispatch.partial_stats_bytes},
        {buffers.scratch.partial_l, dispatch.partial_stats_bytes},
    }};
    for (std::size_t write = 0u; write < writes.size(); ++write) {
        for (std::size_t read = 0u; read < reads.size(); ++read) {
            if (buffer_overlaps(writes[write].first, writes[write].second,
                                reads[read].first, reads[read].second)) {
                error = "QSA attention scratch outputs must be disjoint "
                        "from all inputs";
                return false;
            }
        }
        for (std::size_t other = write + 1u; other < writes.size(); ++other) {
            if (buffer_overlaps(writes[write].first, writes[write].second,
                                writes[other].first, writes[other].second)) {
                error = "QSA attention scratch outputs must be disjoint "
                        "from one another";
                return false;
            }
        }
    }

    dispatch.key_scale_binding =
        key_uses_scales ? buffers.key_scales : buffers.key_data;
    dispatch.key_scale_binding_bytes =
        key_uses_scales ? dispatch.key_scale_bytes : dispatch.key_data_bytes;
    dispatch.key_scale_binding_alignment =
        key_uses_scales ? alignof(std::uint16_t)
                        : dispatch.key_data_alignment;
    dispatch.value_scale_binding =
        value_uses_scales ? buffers.value_scales : buffers.value_data;
    dispatch.value_scale_binding_bytes =
        value_uses_scales ? dispatch.value_scale_bytes
                          : dispatch.value_data_bytes;
    dispatch.value_scale_binding_alignment =
        value_uses_scales ? alignof(std::uint16_t)
                          : dispatch.value_data_alignment;
    dispatch.params = QsaAttentionParams{
        token_rows, cell_capacity, id_row_stride, dispatch.chunk_count};
    error.clear();
    return true;
}

static bool prepare_qsa_attention_merge_dispatch(
    Context& context, std::uint32_t token_rows,
    std::uint32_t id_row_stride,
    const QsaAttentionMergeBuffers& buffers,
    QsaAttentionMergeDispatch& dispatch, std::string& error) {
    error.clear();
    std::uint32_t chunk_count = 0u;
    if (!qsa_attention_extents(
            token_rows, id_row_stride, chunk_count,
            dispatch.partial_acc_bytes, dispatch.partial_stats_bytes,
            dispatch.output_bytes)) {
        error = "QSA attention merge row count or extent is invalid";
        return false;
    }
    if (!validate_buffer(context, buffers.scratch.partial_acc,
                         dispatch.partial_acc_bytes, alignof(float),
                         "attention partial accumulators", error) ||
        !validate_buffer(context, buffers.scratch.partial_m,
                         dispatch.partial_stats_bytes, alignof(float),
                         "attention partial maxima", error) ||
        !validate_buffer(context, buffers.scratch.partial_l,
                         dispatch.partial_stats_bytes, alignof(float),
                         "attention partial denominators", error) ||
        !validate_buffer(context, buffers.output, dispatch.output_bytes,
                         alignof(float), "attention output", error)) {
        return false;
    }

    const std::array<std::pair<BufferRef, std::size_t>, 3> reads{{
        {buffers.scratch.partial_acc, dispatch.partial_acc_bytes},
        {buffers.scratch.partial_m, dispatch.partial_stats_bytes},
        {buffers.scratch.partial_l, dispatch.partial_stats_bytes},
    }};
    for (std::size_t left = 0u; left < reads.size(); ++left) {
        for (std::size_t right = left + 1u; right < reads.size(); ++right) {
            if (buffer_overlaps(reads[left].first, reads[left].second,
                                reads[right].first, reads[right].second)) {
                error = "QSA attention scratch views must be disjoint";
                return false;
            }
        }
        if (buffer_overlaps(buffers.output, dispatch.output_bytes,
                            reads[left].first, reads[left].second)) {
            error = "QSA attention output must be disjoint from scratch";
            return false;
        }
    }

    dispatch.params =
        QsaAttentionParams{token_rows, 0u, id_row_stride, chunk_count};
    error.clear();
    return true;
}

bool encode_qsa_attention_chunk(
    Context& context, CommandBuffer& command_buffer, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaAttentionChunkBuffers& buffers, std::string& error) {
    QsaAttentionChunkDispatch dispatch;
    if (!prepare_qsa_attention_chunk_dispatch(
            context, mode, token_rows, cell_capacity, id_row_stride, buffers,
            dispatch, error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_qsa_attention_pipeline(
            context, dispatch.kernel_name, error)) {
        return false;
    }
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA attention chunk compute encoder";
        return false;
    }

    encoder.label = @"QSA sparse attention chunks";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            dispatch.kernel_name)];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.query.handle
                offset:buffers.query.offset atIndex:0u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.steps.handle
                offset:buffers.steps.offset atIndex:1u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.ids.handle
                offset:buffers.ids.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.key_data.handle
                offset:buffers.key_data.offset atIndex:3u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)dispatch.key_scale_binding.handle
                offset:dispatch.key_scale_binding.offset atIndex:4u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.value_data.handle
                offset:buffers.value_data.offset atIndex:5u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)dispatch.value_scale_binding.handle
                offset:dispatch.value_scale_binding.offset atIndex:6u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_acc.handle
                offset:buffers.scratch.partial_acc.offset atIndex:7u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_m.handle
                offset:buffers.scratch.partial_m.offset atIndex:8u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_l.handle
                offset:buffers.scratch.partial_l.offset atIndex:9u];
    [encoder setBytes:&dispatch.params length:sizeof(dispatch.params)
               atIndex:10u];
    [encoder setThreadgroupMemoryLength:
        (N_HEAD_Q / N_HEAD_KV) * HEAD_DIM * sizeof(float) atIndex:0u];
    [encoder setThreadgroupMemoryLength:
        (N_HEAD_Q / N_HEAD_KV) * kQsaAttentionChunkCells * sizeof(float)
        atIndex:1u];
    [encoder setThreadgroupMemoryLength:
        kQsaAttentionChunkCells * sizeof(std::uint32_t) atIndex:2u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(dispatch.chunk_count, N_HEAD_KV, token_rows)
              threadsPerThreadgroup:MTLSizeMake(
                  kQsaAttentionThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_attention_chunk(
    Context& context, Encoder& encoder, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaAttentionChunkBuffers& buffers, std::string& error) {
    QsaAttentionChunkDispatch dispatch;
    if (!prepare_qsa_attention_chunk_dispatch(
            context, mode, token_rows, cell_capacity, id_row_stride, buffers,
            dispatch, error)) {
        return false;
    }
    if (!validate_qsa_attention_pipeline(
            context, dispatch.kernel_name, error)) {
        return false;
    }

    std::vector<EncoderBufferBinding> bindings;
    bindings.reserve(10u);
    bindings.push_back(
        {buffers.query, 0u, dispatch.query_bytes, alignof(float)});
    bindings.push_back(
        {buffers.steps, 1u, dispatch.steps_bytes, alignof(StepRow)});
    bindings.push_back(
        {buffers.ids, 2u, dispatch.ids_bytes, alignof(std::int32_t)});
    bindings.push_back({buffers.key_data, 3u, dispatch.key_data_bytes,
                        dispatch.key_data_alignment});
    if (dispatch.key_scale_bytes != 0u) {
        bindings.push_back(
            {dispatch.key_scale_binding, 4u,
             dispatch.key_scale_binding_bytes,
             dispatch.key_scale_binding_alignment});
    }
    bindings.push_back({buffers.value_data, 5u, dispatch.value_data_bytes,
                        dispatch.value_data_alignment});
    if (dispatch.value_scale_bytes != 0u) {
        bindings.push_back(
            {dispatch.value_scale_binding, 6u,
             dispatch.value_scale_binding_bytes,
             dispatch.value_scale_binding_alignment});
    }
    bindings.push_back(
        {buffers.scratch.partial_acc, 7u, dispatch.partial_acc_bytes,
         alignof(float)});
    bindings.push_back(
        {buffers.scratch.partial_m, 8u, dispatch.partial_stats_bytes,
         alignof(float)});
    bindings.push_back(
        {buffers.scratch.partial_l, 9u, dispatch.partial_stats_bytes,
         alignof(float)});
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 10u};
    const std::array byte_bindings{params_binding};
    const std::array threadgroup_memory{
        EncoderThreadgroupMemory{
            0u, static_cast<std::size_t>(N_HEAD_Q / N_HEAD_KV) * HEAD_DIM *
                    sizeof(float)},
        EncoderThreadgroupMemory{
            1u, static_cast<std::size_t>(N_HEAD_Q / N_HEAD_KV) *
                    kQsaAttentionChunkCells * sizeof(float)},
        EncoderThreadgroupMemory{
            2u, static_cast<std::size_t>(kQsaAttentionChunkCells) *
                    sizeof(std::uint32_t)},
    };
    return encoder.dispatch(
        dispatch.kernel_name,
        {dispatch.chunk_count, N_HEAD_KV, token_rows},
        {kQsaAttentionThreads, 1u, 1u},
        std::span<const EncoderBufferBinding>(bindings), byte_bindings,
        threadgroup_memory, error);
}

bool encode_qsa_attention_merge(
    Context& context, CommandBuffer& command_buffer,
    std::uint32_t token_rows, std::uint32_t id_row_stride,
    const QsaAttentionMergeBuffers& buffers, std::string& error) {
    QsaAttentionMergeDispatch dispatch;
    if (!prepare_qsa_attention_merge_dispatch(
            context, token_rows, id_row_stride, buffers, dispatch, error)) {
        return false;
    }

    auto encoding_lock = command_buffer.lock_for_encoding();
    if (!validate_qsa_attention_pipeline(
            context, "strata_attn_merge", error)) {
        return false;
    }
    id<MTLCommandBuffer> native =
        validate_command_buffer(context, command_buffer, error);
    if (native == nil) return false;
    id<MTLComputeCommandEncoder> encoder =
        begin_compute_encoder(context, native);
    if (encoder == nil) {
        error = "cannot create QSA attention merge compute encoder";
        return false;
    }

    encoder.label = @"QSA sparse attention merge";
    [encoder setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)context.pipeline(
            "strata_attn_merge")];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_acc.handle
                offset:buffers.scratch.partial_acc.offset atIndex:0u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_m.handle
                offset:buffers.scratch.partial_m.offset atIndex:1u];
    [encoder setBuffer:
        (__bridge id<MTLBuffer>)buffers.scratch.partial_l.handle
                offset:buffers.scratch.partial_l.offset atIndex:2u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffers.output.handle
                offset:buffers.output.offset atIndex:3u];
    [encoder setBytes:&dispatch.params length:sizeof(dispatch.params)
               atIndex:4u];
    [encoder dispatchThreadgroups:
        MTLSizeMake(N_HEAD_Q, token_rows, 1u)
              threadsPerThreadgroup:MTLSizeMake(
                  kQsaAttentionThreads, 1u, 1u)];
    end_compute_encoder(context, encoder);
    error.clear();
    return true;
}

bool encode_qsa_attention_merge(
    Context& context, Encoder& encoder,
    std::uint32_t token_rows, std::uint32_t id_row_stride,
    const QsaAttentionMergeBuffers& buffers, std::string& error) {
    QsaAttentionMergeDispatch dispatch;
    if (!prepare_qsa_attention_merge_dispatch(
            context, token_rows, id_row_stride, buffers, dispatch, error)) {
        return false;
    }
    if (!validate_qsa_attention_pipeline(
            context, "strata_attn_merge", error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 4> bindings{{
        {buffers.scratch.partial_acc, 0u, dispatch.partial_acc_bytes,
         alignof(float)},
        {buffers.scratch.partial_m, 1u, dispatch.partial_stats_bytes,
         alignof(float)},
        {buffers.scratch.partial_l, 2u, dispatch.partial_stats_bytes,
         alignof(float)},
        {buffers.output, 3u, dispatch.output_bytes, alignof(float)},
    }};
    const EncoderBytesBinding params_binding{
        &dispatch.params, sizeof(dispatch.params), 4u};
    const std::array byte_bindings{params_binding};
    return encoder.dispatch(
        "strata_attn_merge", {N_HEAD_Q, token_rows, 1u},
        {kQsaAttentionThreads, 1u, 1u}, bindings, byte_bindings, {}, error);
}

bool encode_qsa_selection_attention(
    Context& context, Encoder& encoder, QsaAttentionMode mode,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride,
    const QsaSelectionAttentionBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "QSA selection-attention must be encoded outside a concurrent "
                "dispatch group";
        return false;
    }
    if (id_row_stride != INDEXER_SELECTION_WIDTH_MAX) {
        error = "QSA selection-attention ID stride must be 2051";
        return false;
    }
    if (!same_view(buffers.selection_scores.steps,
                   buffers.selection_topk.steps) ||
        !same_view(buffers.selection_scores.steps,
                   buffers.attention_chunk.steps)) {
        error = "QSA selection-attention stages must share one StepRow view";
        return false;
    }
    if (!same_view(buffers.selection_scores.scores,
                   buffers.selection_topk.scores)) {
        error = "QSA selection-attention stages must share one score view";
        return false;
    }
    if (!same_view(buffers.selection_topk.ids,
                   buffers.attention_chunk.ids)) {
        error = "QSA selection-attention stages must share one ID view";
        return false;
    }
    if (!same_view(buffers.attention_chunk.scratch.partial_acc,
                   buffers.attention_merge.scratch.partial_acc) ||
        !same_view(buffers.attention_chunk.scratch.partial_m,
                   buffers.attention_merge.scratch.partial_m) ||
        !same_view(buffers.attention_chunk.scratch.partial_l,
                   buffers.attention_merge.scratch.partial_l)) {
        error = "QSA attention chunk and merge must share the same scratch views";
        return false;
    }

    SelectionScoresDispatch selection_scores_dispatch;
    SelectionTopkDispatch selection_topk_dispatch;
    QsaAttentionChunkDispatch attention_chunk_dispatch;
    QsaAttentionMergeDispatch attention_merge_dispatch;
    if (!prepare_selection_scores_dispatch(
            context, token_rows, cell_capacity,
            buffers.selection_scores, selection_scores_dispatch, error) ||
        !prepare_selection_topk_dispatch(
            context, token_rows, cell_capacity,
            buffers.selection_topk, selection_topk_dispatch, error) ||
        !prepare_qsa_attention_chunk_dispatch(
            context, mode, token_rows, cell_capacity, id_row_stride,
            buffers.attention_chunk, attention_chunk_dispatch, error) ||
        !prepare_qsa_attention_merge_dispatch(
            context, token_rows, id_row_stride,
            buffers.attention_merge, attention_merge_dispatch, error)) {
        return false;
    }

    struct NamedView {
        BufferRef buffer;
        std::size_t bytes;
        std::string_view name;
    };
    const std::array<NamedView, 6> writes{{
        {buffers.selection_scores.scores,
         selection_scores_dispatch.scores_bytes, "selection scores"},
        {buffers.selection_topk.ids,
         selection_topk_dispatch.ids_bytes, "top-k IDs"},
        {buffers.attention_chunk.scratch.partial_acc,
         attention_chunk_dispatch.partial_acc_bytes,
         "attention partial accumulators"},
        {buffers.attention_chunk.scratch.partial_m,
         attention_chunk_dispatch.partial_stats_bytes,
         "attention partial maxima"},
        {buffers.attention_chunk.scratch.partial_l,
         attention_chunk_dispatch.partial_stats_bytes,
         "attention partial denominators"},
        {buffers.attention_merge.output,
         attention_merge_dispatch.output_bytes, "merged attention"},
    }};
    const std::array<NamedView, 16> reads{{
        {buffers.selection_scores.pooled,
         selection_scores_dispatch.pooled_bytes, "pooled keys"},
        {buffers.selection_scores.dead,
         selection_scores_dispatch.dead_bytes, "dead key"},
        {buffers.selection_scores.indexer_queries,
         selection_scores_dispatch.query_bytes, "indexer queries"},
        {buffers.selection_scores.steps,
         selection_scores_dispatch.steps_bytes, "selection StepRows"},
        {buffers.selection_topk.scores,
         selection_topk_dispatch.scores_bytes, "top-k scores"},
        {buffers.selection_topk.steps,
         selection_topk_dispatch.steps_bytes, "top-k StepRows"},
        {buffers.attention_chunk.query,
         attention_chunk_dispatch.query_bytes, "attention queries"},
        {buffers.attention_chunk.steps,
         attention_chunk_dispatch.steps_bytes, "attention StepRows"},
        {buffers.attention_chunk.ids,
         attention_chunk_dispatch.ids_bytes, "attention IDs"},
        {buffers.attention_chunk.key_data,
         attention_chunk_dispatch.key_data_bytes, "attention key data"},
        {buffers.attention_chunk.key_scales,
         attention_chunk_dispatch.key_scale_bytes, "attention key scales"},
        {buffers.attention_chunk.value_data,
         attention_chunk_dispatch.value_data_bytes, "attention value data"},
        {buffers.attention_chunk.value_scales,
         attention_chunk_dispatch.value_scale_bytes,
         "attention value scales"},
        {buffers.attention_merge.scratch.partial_acc,
         attention_merge_dispatch.partial_acc_bytes,
         "merge partial accumulators"},
        {buffers.attention_merge.scratch.partial_m,
         attention_merge_dispatch.partial_stats_bytes,
         "merge partial maxima"},
        {buffers.attention_merge.scratch.partial_l,
         attention_merge_dispatch.partial_stats_bytes,
         "merge partial denominators"},
    }};
    for (std::size_t write = 0u; write < writes.size(); ++write) {
        for (std::size_t other = write + 1u; other < writes.size(); ++other) {
            if (buffer_overlaps(writes[write].buffer, writes[write].bytes,
                                writes[other].buffer, writes[other].bytes)) {
                error = "QSA selection-attention stage outputs must be "
                        "disjoint";
                return false;
            }
        }
        for (std::size_t read = 0u; read < reads.size(); ++read) {
            const bool producer_consumer_pair =
                (write == 0u && read == 4u &&
                 same_view(writes[write].buffer, reads[read].buffer)) ||
                (write == 1u && read == 8u &&
                 same_view(writes[write].buffer, reads[read].buffer)) ||
                (write == 2u && read == 13u &&
                 same_view(writes[write].buffer, reads[read].buffer)) ||
                (write == 3u && read == 14u &&
                 same_view(writes[write].buffer, reads[read].buffer)) ||
                (write == 4u && read == 15u &&
                 same_view(writes[write].buffer, reads[read].buffer));
            if (!producer_consumer_pair &&
                buffer_overlaps(writes[write].buffer, writes[write].bytes,
                                reads[read].buffer, reads[read].bytes)) {
                error = "QSA selection-attention stage outputs must not "
                        "overlap unrelated stage inputs";
                return false;
            }
        }
    }

    if (!encode_qsa_selection_scores(
            context, encoder, token_rows, cell_capacity,
            buffers.selection_scores, error)) {
        return false;
    }
    if (!encode_qsa_selection_topk(
            context, encoder, token_rows, cell_capacity,
            buffers.selection_topk, error)) {
        return false;
    }
    if (!encode_qsa_attention_chunk(
            context, encoder, mode, token_rows, cell_capacity,
            id_row_stride, buffers.attention_chunk, error)) {
        return false;
    }
    return encode_qsa_attention_merge(
        context, encoder, token_rows, id_row_stride,
        buffers.attention_merge, error);
}

}  // namespace strata::metal
