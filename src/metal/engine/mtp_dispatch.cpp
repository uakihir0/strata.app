// src/metal/engine/mtp_dispatch.cpp - validated MTP stage encoders.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "../kernels/common/params.h"
#include "../kernels/common/types.h"

namespace strata::metal {
namespace {

constexpr std::uint32_t kMtpKvProjectionWidth = 512u;
constexpr std::uint32_t kMtpKvHeadRows = 2u;
constexpr std::uint32_t kMtpKvHeadWidth = 256u;
constexpr std::uint32_t kMtpQueryProjectionWidth =
    N_HEAD_Q * 2u * HEAD_DIM;
constexpr std::uint32_t kMtpQueryHeadWidth =
    N_HEAD_Q * HEAD_DIM;
constexpr std::uint32_t kMtpOutputProjectionInputWidth =
    N_HEAD_Q * HEAD_DIM;
constexpr std::uint32_t kMtpRouterExperts = 512u;
constexpr std::uint32_t kMtpRouterTopK = 10u;
constexpr std::uint32_t kMtpRoutedExpertTopK = 10u;
constexpr std::uint32_t kMtpRoutedExpertGroups = 40u;
constexpr std::uint32_t kMtpRoutedExpertThreads = 256u;
constexpr std::uint32_t kMtpRoutedExpertChunkValues = 32u;
constexpr std::uint32_t kMtpQ8BlockValues = 32u;
constexpr std::uint32_t kMtpQ81BlockBytes = 36u;
constexpr std::uint32_t kMtpAddStreamsThreads = 256u;
constexpr std::uint32_t kMtpAddStreamsGroupsX =
    HC_DIM / kMtpAddStreamsThreads;
constexpr std::uint32_t kMtpWindowIdsThreads = 256u;
constexpr std::uint32_t kMtpWindowIdsGroups = 8u;
constexpr std::uint32_t kMtpWindowIdsStride =
    kMtpWindowIdsThreads * kMtpWindowIdsGroups;
constexpr std::uint32_t kMtpAttentionChunkCells = 64u;
constexpr std::size_t kMtpAttentionReservedScratchFloats = 64u;
constexpr std::uint32_t kMtpMaxRows =
    static_cast<std::uint32_t>(sizeof(MtpParams{}.tokens) /
                               sizeof(MtpParams{}.tokens[0]));

static_assert(HC_DIM % kMtpAddStreamsThreads == 0u);
static_assert(sizeof(MtpParams) == 236u);
static_assert(offsetof(MtpParams, T) == 0u);
static_assert(sizeof(Q81) == kMtpQ81BlockBytes);
static_assert(kMtpQueryProjectionWidth == 12'288u);
static_assert(kMtpQueryHeadWidth == 6'144u);
static_assert(kMtpOutputProjectionInputWidth == 6'144u);
static_assert(kMtpWindowIdsStride == 2'048u);
static_assert(kMtpRoutedExpertTopK == kMtpRouterTopK);
static_assert(kMtpRoutedExpertTopK * N_FF == 6'400u);
static_assert(kMtpExpertsFileBytes == 707'788'800u);

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes,
                   Address right, std::size_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(ReadOnlyBufferRef input, std::size_t input_bytes,
              BufferRef output, std::size_t output_bytes) {
    const void* const input_contents = input.contents();
    const bool cpu_overlap =
        input_contents != nullptr && output.contents != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(input_contents), input_bytes,
            reinterpret_cast<std::uintptr_t>(output.contents), output_bytes);
    const std::uint64_t input_address = input.gpu_address();
    const bool gpu_overlap =
        input_address != 0u && output.gpu_address != 0u &&
        ranges_overlap(input_address, input_bytes, output.gpu_address,
                       output_bytes);
    return cpu_overlap || gpu_overlap;
}

bool overlaps(BufferRef left, std::size_t left_bytes, BufferRef right,
              std::size_t right_bytes) {
    const bool handle_overlap =
        left.handle != nullptr && left.handle == right.handle &&
        ranges_overlap(left.offset, left_bytes, right.offset, right_bytes);
    const bool cpu_overlap =
        left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(
            reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
            reinterpret_cast<std::uintptr_t>(right.contents), right_bytes);
    const bool gpu_overlap =
        left.gpu_address != 0u && right.gpu_address != 0u &&
        ranges_overlap(left.gpu_address, left_bytes, right.gpu_address,
                       right_bytes);
    return handle_overlap || cpu_overlap || gpu_overlap;
}

bool overlaps(ReadOnlyBufferRef left, std::size_t left_bytes,
              ReadOnlyBufferRef right, std::size_t right_bytes) {
    const void* const left_contents = left.contents();
    const void* const right_contents = right.contents();
    const bool cpu_overlap =
        left_contents != nullptr && right_contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left_bytes,
                       reinterpret_cast<std::uintptr_t>(right_contents),
                       right_bytes);
    const std::uint64_t left_address = left.gpu_address();
    const std::uint64_t right_address = right.gpu_address();
    const bool gpu_overlap =
        left_address != 0u && right_address != 0u &&
        ranges_overlap(left_address, left_bytes, right_address, right_bytes);
    return cpu_overlap || gpu_overlap;
}

bool validate_read_only_buffer(Context& context, ReadOnlyBufferRef buffer,
                               std::size_t required_bytes,
                               std::size_t alignment, const char* name,
                               const char* operation, std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(operation) + " " + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length() < required_bytes ||
        buffer.offset() % alignment != 0u) {
        error = std::string(operation) + " " + name +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_output_buffer(Context& context, BufferRef buffer,
                            std::size_t required_bytes,
                            const char* operation, const char* name,
                            std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(operation) + " " + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes ||
        buffer.offset % alignof(float) != 0u) {
        error = std::string(operation) + " " + name +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_buffer_view(Context& context, BufferRef buffer,
                          std::size_t required_bytes, std::size_t alignment,
                          const char* operation, const char* name,
                          std::string& error) {
    if (!buffer || !context.buffers().owns(buffer)) {
        error = std::string(operation) + " " + name +
                " buffer is not owned by this Context";
        return false;
    }
    if (buffer.length < required_bytes ||
        alignment == 0u || buffer.offset % alignment != 0u) {
        error = std::string(operation) + " " + name +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

bool same_region(KvRegion left, KvRegion right) {
    return left.offset == right.offset && left.bytes == right.bytes;
}

bool same_mtp_kv_layout(const MtpKvLayout& left,
                        const MtpKvLayout& right) {
    return left.requested_format == right.requested_format &&
           left.storage_format == right.storage_format &&
           left.requested_cells == right.requested_cells &&
           left.cell_capacity == right.cell_capacity &&
           left.kv_rows == right.kv_rows &&
           left.allocation_bytes == right.allocation_bytes &&
           same_region(left.k, right.k) &&
           same_region(left.k_scale, right.k_scale) &&
           same_region(left.v, right.v) &&
           same_region(left.v_scale, right.v_scale);
}

bool checked_size_multiply(std::size_t left, std::size_t right,
                           std::size_t& result) {
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_size_add(std::size_t left, std::size_t right,
                      std::size_t& result) {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool mtp_kv_views_match_layout(Context& context,
                              const MtpKvLayout& layout,
                              const MtpKvViews& views,
                              std::string& error,
                              const char* operation =
                                  "MTP attention K/V append") {
    const std::array<std::pair<const char*, BufferRef>, 4> named_views{{
        {"K cache", views.k},
        {"K scales", views.k_scale},
        {"V cache", views.v},
        {"V scales", views.v_scale},
    }};
    const std::array<KvRegion, 4> regions{
        layout.k, layout.k_scale, layout.v, layout.v_scale};

    BufferRef allocation;
    KvRegion allocation_region{};
    for (std::size_t index = 0u; index < regions.size(); ++index) {
        const BufferRef view = named_views[index].second;
        const KvRegion region = regions[index];
        if (region.bytes == 0u) {
            if (view) {
                error = std::string(operation) + " " +
                        named_views[index].first +
                        " view must be empty for this KV format";
                return false;
            }
            continue;
        }
        if (!validate_buffer_view(context, view, region.bytes, 1u, operation,
                                  named_views[index].first, error)) {
            return false;
        }
        if (view.length != region.bytes) {
            error = std::string(operation) + " " +
                    named_views[index].first +
                    " view must match its exact layout extent";
            return false;
        }
        if (!allocation) {
            allocation = view;
            allocation_region = region;
        } else {
            if (allocation.handle != view.handle ||
                view.offset < region.offset ||
                allocation.offset < allocation_region.offset ||
                view.offset - region.offset !=
                    allocation.offset - allocation_region.offset) {
                error = std::string(operation) +
                        " cache views do not share the supplied layout allocation";
                return false;
            }
            if ((allocation.gpu_address == 0u) !=
                    (view.gpu_address == 0u) ||
                (allocation.gpu_address != 0u &&
                 (allocation.gpu_address < allocation_region.offset ||
                  view.gpu_address < region.offset ||
                  view.gpu_address - region.offset !=
                      allocation.gpu_address -
                          allocation_region.offset))) {
                error = std::string(operation) +
                        " cache GPU views do not match the supplied layout";
                return false;
            }
            if (allocation.contents != nullptr || view.contents != nullptr) {
                if (allocation.contents == nullptr || view.contents == nullptr) {
                    error = std::string(operation) +
                            " cache CPU mappings are inconsistent";
                    return false;
                }
                const auto allocation_address =
                    reinterpret_cast<std::uintptr_t>(allocation.contents);
                const auto view_address =
                    reinterpret_cast<std::uintptr_t>(view.contents);
                if (allocation_address < allocation_region.offset ||
                    view_address < region.offset ||
                    view_address - region.offset !=
                        allocation_address - allocation_region.offset) {
                    error = std::string(operation) +
                            " cache CPU views do not match the supplied layout";
                    return false;
                }
            }
        }
    }
    if (!allocation) {
        error = std::string(operation) + " has no cache allocation";
        return false;
    }
    return true;
}

bool read_mtp_rows(Context& context, ReadOnlyBufferRef params,
                   const char* operation, std::uint32_t& rows,
                   std::string& error) {
    if (!params || !context.buffers().owns(params)) {
        error = std::string(operation) +
                " parameter block is not owned by this Context";
        return false;
    }
    if (params.length() < sizeof(MtpParams) ||
        params.offset() % alignof(MtpParams) != 0u) {
        error = std::string(operation) +
                " parameter block is undersized or misaligned";
        return false;
    }
    const void* const contents = params.contents();
    if (contents == nullptr) {
        error = std::string(operation) +
                " parameter block has no CPU-visible contents";
        return false;
    }
    std::memcpy(&rows, contents, sizeof(rows));
    if (rows == 0u || rows > kMtpMaxRows) {
        error = std::string(operation) + " T must be in 1..8";
        return false;
    }
    return true;
}

std::uint32_t compiled_mtp_columns(std::uint32_t rows) {
    if (rows <= 1u) return 1u;
    if (rows <= 2u) return 2u;
    if (rows <= 4u) return 4u;
    return 8u;
}

bool validate_mtp_kv_projection(
    Context& context, const MtpAttentionKvProjectionBuffers& buffers,
    std::uint32_t& rows, std::uint32_t& compiled_columns,
    std::string& error) {
    constexpr char operation[] = "MTP attention K/V projection";
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }

    compiled_columns = compiled_mtp_columns(rows);
    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t q8_1_bytes =
        static_cast<std::size_t>(rows) * (N_EMBD / kMtpQ8BlockValues) *
        kMtpQ81BlockBytes;
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kMtpKvProjectionWidth) *
        (N_EMBD / kMtpQ8BlockValues) * sizeof(block_q8_0);
    const std::size_t output_bytes = static_cast<std::size_t>(rows) *
                                     kMtpKvProjectionWidth * sizeof(float);
    const std::size_t sink_floats =
        std::max<std::size_t>(
            1u, static_cast<std::size_t>(compiled_columns - rows) *
                    kMtpKvProjectionWidth);
    const std::size_t sink_bytes = sink_floats * sizeof(float);

    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error) ||
        !validate_output_buffer(context, buffers.mixed, input_bytes,
                                operation, "mixed", error) ||
        !validate_output_buffer(context, buffers.xq, q8_1_bytes,
                                operation, "xq", error) ||
        !validate_read_only_buffer(context, buffers.key_weights, weight_bytes,
                                   alignof(std::uint16_t), "key weights",
                                   operation,
                                   error) ||
        !validate_read_only_buffer(context, buffers.value_weights,
                                   weight_bytes, alignof(std::uint16_t),
                                   "value weights", operation, error) ||
        !validate_output_buffer(context, buffers.kcur, output_bytes,
                                operation, "kcur", error) ||
        !validate_output_buffer(context, buffers.vcur, output_bytes,
                                operation, "vcur", error) ||
        !validate_output_buffer(context, buffers.sink, sink_bytes,
                                operation, "sink", error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    if (!active_rows) {
        error = std::string(operation) +
                " cannot form its active-row metadata view";
        return false;
    }

    const std::array<std::pair<const char*, BufferRef>, 4> writes{{
        {"xq", buffers.xq},
        {"kcur", buffers.kcur},
        {"vcur", buffers.vcur},
        {"sink", buffers.sink},
    }};
    const std::array<std::size_t, 4> write_bytes{
        q8_1_bytes, output_bytes, output_bytes, sink_bytes};
    const std::array<std::pair<const char*, ReadOnlyBufferRef>, 3> reads{{
        {"mixed", context.buffers().read_only(buffers.mixed)},
        {"key weights", buffers.key_weights},
        {"value weights", buffers.value_weights},
    }};
    const std::array<std::size_t, 3> read_bytes{
        input_bytes, weight_bytes, weight_bytes};

    for (std::size_t write_index = 0u; write_index < writes.size();
         ++write_index) {
        for (std::size_t other_write = write_index + 1u;
             other_write < writes.size(); ++other_write) {
            if (overlaps(writes[write_index].second, write_bytes[write_index],
                         writes[other_write].second,
                         write_bytes[other_write])) {
                error = std::string(operation) + " writable views " +
                        writes[write_index].first + " and " +
                        writes[other_write].first + " overlap";
                return false;
            }
        }
        if (overlaps(buffers.params, sizeof(MtpParams),
                     writes[write_index].second, write_bytes[write_index])) {
            error = std::string(operation) + " parameter block overlaps " +
                    writes[write_index].first;
            return false;
        }
        if (overlaps(active_rows, sizeof(std::uint32_t),
                     writes[write_index].second,
                     write_bytes[write_index])) {
            error = std::string(operation) + " writable view " +
                    writes[write_index].first +
                    " overlaps active-row metadata";
            return false;
        }
        for (std::size_t read_index = 0u; read_index < reads.size();
             ++read_index) {
            if (overlaps(reads[read_index].second, read_bytes[read_index],
                         writes[write_index].second,
                         write_bytes[write_index])) {
                error = std::string(operation) + " writable view " +
                        writes[write_index].first + " overlaps read-only input " +
                        reads[read_index].first;
                return false;
            }
        }
    }
    for (std::size_t read_index = 0u; read_index < reads.size(); ++read_index) {
        if (overlaps(buffers.params, sizeof(MtpParams),
                     reads[read_index].second, read_bytes[read_index])) {
            error = std::string(operation) +
                    " parameter block overlaps read-only input " +
                    reads[read_index].first;
            return false;
        }
    }
    return true;
}

bool validate_mtp_kv_append(
    Context& context, const MtpAttentionKvAppendBuffers& buffers,
    std::uint32_t& rows, BufferRef& steps, std::size_t& current_bytes,
    std::string& error) {
    constexpr char operation[] = "MTP attention K/V append";
    const ReadOnlyBufferRef params =
        context.buffers().read_only(buffers.params);
    if (!read_mtp_rows(context, params, operation, rows, error) ||
        !validate_buffer_view(context, buffers.params, sizeof(MtpParams),
                             alignof(MtpParams), operation, "params",
                             error)) {
        return false;
    }

    MtpKvLayout canonical_layout;
    std::string layout_error;
    if (!make_mtp_kv_layout(buffers.kv_layout.requested_cells,
                            buffers.kv_layout.requested_format,
                            canonical_layout, layout_error) ||
        !same_mtp_kv_layout(buffers.kv_layout, canonical_layout)) {
        error = std::string(operation) +
                " KV layout is invalid or not canonical";
        if (!layout_error.empty()) error += ": " + layout_error;
        return false;
    }
    if ((canonical_layout.storage_format == KvFormat::Fp16 &&
         buffers.kv_rot) ||
        (canonical_layout.storage_format == KvFormat::Q4_0 &&
         !buffers.kv_rot)) {
        error = std::string(operation) +
                " kv_rot does not match the selected MTP KV format";
        return false;
    }
    if (canonical_layout.storage_format != KvFormat::Fp16 &&
        canonical_layout.storage_format != KvFormat::Int8 &&
        canonical_layout.storage_format != KvFormat::Q4_0) {
        error = std::string(operation) +
                " storage format is not supported by the MTP cache";
        return false;
    }
    if (!mtp_kv_views_match_layout(context, canonical_layout,
                                   buffers.kv_views, error)) {
        return false;
    }

    current_bytes =
        static_cast<std::size_t>(rows) * kMtpKvProjectionWidth *
        sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(kMtpKvHeadWidth) * sizeof(float);
    if (!validate_buffer_view(context, buffers.kcur, current_bytes,
                              alignof(float), operation, "kcur", error) ||
        !validate_buffer_view(context, buffers.vcur, current_bytes,
                              alignof(float), operation, "vcur", error) ||
        !validate_buffer_view(context, buffers.k_norm, norm_bytes,
                              alignof(float), operation, "K norm", error) ||
        !buffers.rope || !context.buffers().owns(buffers.rope)) {
        if (error.empty()) {
            error = std::string(operation) +
                    " RoPE table is not owned by this Context";
        }
        return false;
    }
    const std::size_t step_bytes =
        static_cast<std::size_t>(rows) * sizeof(StepRow);
    steps = buffers.params.slice(offsetof(MtpParams, step), step_bytes);
    if (!steps) {
        error = std::string(operation) +
                " cannot form its active StepRow view";
        return false;
    }
    const auto* const step_rows =
        reinterpret_cast<const StepRow*>(steps.contents);
    if (step_rows == nullptr) {
        error = std::string(operation) +
                " StepRows have no CPU-visible contents";
        return false;
    }
    std::array<std::int32_t, kMtpMaxRows> positions{};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::int32_t position = step_rows[row].pos;
        if (position < 0 ||
            static_cast<std::uint32_t>(position) >=
                canonical_layout.cell_capacity) {
            error = std::string(operation) +
                    " StepRow position is outside the KV capacity";
            return false;
        }
        for (std::uint32_t prior = 0u; prior < row; ++prior) {
            if (positions[prior] == position) {
                error = std::string(operation) +
                        " StepRow positions must be unique";
                return false;
            }
        }
        positions[row] = position;
    }

    if (!validate_buffer_view(context, buffers.rope, 1u, 1u, operation,
                              "RoPE", error)) {
        return false;
    }

    const std::array<std::pair<const char*, BufferRef>, 6> writable{{
        {"kcur", buffers.kcur},
        {"vcur", buffers.vcur},
        {"K cache", buffers.kv_views.k},
        {"K scales", buffers.kv_views.k_scale},
        {"V cache", buffers.kv_views.v},
        {"V scales", buffers.kv_views.v_scale},
    }};
    const std::array<std::size_t, 6> writable_bytes{
        current_bytes, current_bytes, canonical_layout.k.bytes,
        canonical_layout.k_scale.bytes, canonical_layout.v.bytes,
        canonical_layout.v_scale.bytes};
    for (std::size_t index = 0u; index < writable.size(); ++index) {
        if (writable_bytes[index] == 0u) continue;
        if (overlaps(buffers.params, sizeof(MtpParams),
                     writable[index].second, writable_bytes[index]) ||
            overlaps(buffers.k_norm, norm_bytes, writable[index].second,
                     writable_bytes[index]) ||
            overlaps(buffers.rope, buffers.rope.length,
                     writable[index].second, writable_bytes[index])) {
            error = std::string(operation) + " writable view " +
                    writable[index].first +
                    " overlaps params, K norm or the RoPE table";
            return false;
        }
        for (std::size_t other = index + 1u; other < writable.size(); ++other) {
            if (writable_bytes[other] != 0u &&
                overlaps(writable[index].second, writable_bytes[index],
                         writable[other].second, writable_bytes[other])) {
                error = std::string(operation) + " writable views " +
                        writable[index].first + " and " +
                        writable[other].first + " overlap";
                return false;
            }
        }
    }
    if (overlaps(buffers.params, sizeof(MtpParams), buffers.k_norm,
                 norm_bytes) ||
        overlaps(buffers.params, sizeof(MtpParams), buffers.rope,
                 buffers.rope.length)) {
        error = std::string(operation) +
                " parameter block overlaps K norm or the RoPE table";
        return false;
    }
    return true;
}

bool validate_mtp_query_projection(
    Context& context, const MtpAttentionQueryProjectionBuffers& buffers,
    ReadOnlyBufferRef& active_rows, std::string& error) {
    constexpr char operation[] = "MTP attention query projection";
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }
    const std::size_t activation_bytes =
        static_cast<std::size_t>(N_EMBD / kMtpQ8BlockValues) *
        kMtpQ81BlockBytes;
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kMtpQueryProjectionWidth) *
        (N_EMBD / kMtpQ8BlockValues) * sizeof(block_q8_0);
    const std::size_t output_bytes =
        static_cast<std::size_t>(kMtpQueryProjectionWidth) * sizeof(float);
    constexpr std::size_t sink_bytes = sizeof(float);

    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error) ||
        !validate_buffer_view(context, buffers.xq, activation_bytes, 256u,
                              operation, "stage-8 xq", error) ||
        !validate_read_only_buffer(context, buffers.q_weights, weight_bytes,
                                   alignof(std::uint16_t), "q_proj weights",
                                   operation, error) ||
        !validate_output_buffer(context, buffers.q_full, output_bytes,
                                operation, "q_full", error) ||
        !validate_output_buffer(context, buffers.sink, sink_bytes,
                                operation, "sink", error)) {
        return false;
    }

    active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    if (!active_rows) {
        error = std::string(operation) +
                " cannot form its active-row metadata view";
        return false;
    }

    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.xq);
    const std::array<std::pair<const char*, BufferRef>, 2> outputs{{
        {"q_full", buffers.q_full},
        {"sink", buffers.sink},
    }};
    const std::array<std::size_t, 2> output_extents{
        output_bytes, sink_bytes};
    for (std::size_t output = 0u; output < outputs.size(); ++output) {
        if (overlaps(buffers.params, sizeof(MtpParams),
                     outputs[output].second, output_extents[output]) ||
            overlaps(activations, activation_bytes, outputs[output].second,
                     output_extents[output]) ||
            overlaps(buffers.q_weights, weight_bytes,
                     outputs[output].second, output_extents[output])) {
            error = std::string(operation) + " " + outputs[output].first +
                    " overlaps an input";
            return false;
        }
    }
    if (overlaps(buffers.q_full, output_bytes, buffers.sink, sink_bytes)) {
        error = std::string(operation) +
                " q_full and sink outputs overlap";
        return false;
    }
    if (overlaps(buffers.params, sizeof(MtpParams), activations,
                 activation_bytes) ||
        overlaps(buffers.params, sizeof(MtpParams), buffers.q_weights,
                 weight_bytes)) {
        error = std::string(operation) +
                " parameter block overlaps a projection input";
        return false;
    }
    return true;
}

bool validate_mtp_query_norm(
    Context& context, const MtpAttentionQueryNormBuffers& buffers,
    BufferRef& active_steps, std::string& error) {
    constexpr char operation[] = "MTP attention query norm";
    const ReadOnlyBufferRef params =
        context.buffers().read_only(buffers.params);
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }

    constexpr std::size_t q_full_bytes =
        static_cast<std::size_t>(kMtpQueryProjectionWidth) * sizeof(float);
    constexpr std::size_t q_norm_bytes = HEAD_DIM * sizeof(float);
    constexpr std::size_t qcur_bytes =
        static_cast<std::size_t>(kMtpQueryHeadWidth) * sizeof(float);
    constexpr std::size_t step_bytes = sizeof(StepRow);
    if (!validate_buffer_view(context, buffers.params, sizeof(MtpParams),
                              alignof(MtpParams), operation, "params",
                              error) ||
        !validate_buffer_view(context, buffers.q_full, q_full_bytes,
                              alignof(float), operation, "q_full", error) ||
        !validate_buffer_view(context, buffers.q_norm, q_norm_bytes,
                              alignof(float), operation, "q_norm", error) ||
        !validate_buffer_view(context, buffers.rope, 1u, 1u, operation,
                              "RoPE", error) ||
        !validate_output_buffer(context, buffers.qcur, qcur_bytes, operation,
                                "qcur", error)) {
        return false;
    }

    active_steps = buffers.params.slice(offsetof(MtpParams, step), step_bytes);
    if (!active_steps) {
        error = std::string(operation) +
                " cannot form its active StepRow view";
        return false;
    }

    const std::array<std::pair<const char*, BufferRef>, 4> views{{
        {"q_full", buffers.q_full},
        {"q_norm", buffers.q_norm},
        {"RoPE", buffers.rope},
        {"qcur", buffers.qcur},
    }};
    const std::array<std::size_t, 4> extents{
        q_full_bytes, q_norm_bytes, buffers.rope.length, qcur_bytes};
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (overlaps(buffers.params, sizeof(MtpParams), views[i].second,
                     extents[i])) {
            error = std::string(operation) + " parameter block overlaps " +
                    views[i].first;
            return false;
        }
    }
    return true;
}

bool validate_mtp_window_ids(Context& context,
                             const MtpWindowIdsBuffers& buffers,
                             BufferRef& active_ids,
                             std::uint32_t& active_width,
                             std::string& error) {
    constexpr char operation[] = "MTP window IDs";
    if (buffers.window == 0u) {
        error = std::string(operation) +
                " requires a positive finite window";
        return false;
    }
    if (buffers.window >
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max())) {
        error = std::string(operation) +
                " window exceeds the StepRow integer range";
        return false;
    }

    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }
    if (!validate_read_only_buffer(
            context, buffers.params, sizeof(MtpParams), alignof(MtpParams),
            "params", operation, error) ||
        !validate_output_buffer(context, buffers.mtp_ids,
                                sizeof(std::int32_t), operation, "mtp_ids",
                                error)) {
        return false;
    }
    if (buffers.mtp_ids.length % sizeof(std::int32_t) != 0u) {
        error = std::string(operation) +
                " mtp_ids length must contain whole int32 elements";
        return false;
    }
    const std::size_t id_capacity =
        buffers.mtp_ids.length / sizeof(std::int32_t);
    if (buffers.window > id_capacity) {
        error = std::string(operation) +
                " window exceeds the mtp_ids capacity";
        return false;
    }

    StepRow step{};
    const auto* const params_contents =
        static_cast<const std::byte*>(buffers.params.contents());
    std::memcpy(&step, params_contents + offsetof(MtpParams, step),
                sizeof(step));
    if (step.n_kv <= 0) {
        error = std::string(operation) +
                " StepRow n_kv must be positive";
        return false;
    }
    active_width = std::min(static_cast<std::uint32_t>(step.n_kv),
                            buffers.window);
    if (step.width < 0 ||
        static_cast<std::uint32_t>(step.width) != active_width) {
        error = std::string(operation) +
                " StepRow width does not match n_kv and the finite window";
        return false;
    }
    if (active_width > id_capacity) {
        error = std::string(operation) +
                " active ID prefix exceeds its buffer extent";
        return false;
    }
    const std::size_t active_bytes =
        static_cast<std::size_t>(active_width) * sizeof(std::int32_t);
    active_ids = buffers.mtp_ids.slice(0u, active_bytes);
    if (!active_ids) {
        error = std::string(operation) +
                " cannot form the active ID prefix view";
        return false;
    }
    if (overlaps(buffers.params, sizeof(MtpParams), active_ids,
                 active_bytes)) {
        error = std::string(operation) +
                " active ID output overlaps the parameter block";
        return false;
    }
    return true;
}

bool validate_mtp_attention(Context& context,
                            const MtpAttentionBuffers& buffers,
                            QsaAttentionMode& mode, BufferRef& steps,
                            QsaAttentionScratch& scratch,
                            std::size_t& scratch_active_bytes,
                            std::size_t& attention_bytes,
                            std::string& error) {
    constexpr char operation[] = "MTP attention";
    const ReadOnlyBufferRef params =
        context.buffers().read_only(buffers.params);
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }
    if (!validate_buffer_view(context, buffers.params, sizeof(MtpParams),
                              alignof(MtpParams), operation, "params",
                              error)) {
        return false;
    }

    MtpKvLayout canonical_layout;
    std::string layout_error;
    if (!make_mtp_kv_layout(buffers.kv_layout.requested_cells,
                            buffers.kv_layout.requested_format,
                            canonical_layout, layout_error) ||
        !same_mtp_kv_layout(buffers.kv_layout, canonical_layout)) {
        error = std::string(operation) +
                " KV layout is invalid or not canonical";
        if (!layout_error.empty()) error += ": " + layout_error;
        return false;
    }
    if ((canonical_layout.storage_format == KvFormat::Fp16 &&
         buffers.kv_rot) ||
        (canonical_layout.storage_format == KvFormat::Q4_0 &&
         !buffers.kv_rot)) {
        error = std::string(operation) +
                " kv_rot does not match the selected MTP KV format";
        return false;
    }
    switch (canonical_layout.storage_format) {
        case KvFormat::Fp16:
            mode = QsaAttentionMode::f16;
            break;
        case KvFormat::Int8:
            // MTP K8v4 requests are represented by the canonical int8 store.
            mode = QsaAttentionMode::i8;
            break;
        case KvFormat::Q4_0:
            mode = QsaAttentionMode::q4;
            break;
        case KvFormat::K8v4:
            error = std::string(operation) +
                    " storage format is not supported by the MTP cache";
            return false;
    }
    if (!mtp_kv_views_match_layout(context, canonical_layout,
                                   buffers.kv_views, error, operation)) {
        return false;
    }

    if (buffers.id_row_stride == 0u ||
        buffers.id_row_stride > kKvMaximumCells ||
        buffers.id_row_stride % kMtpAttentionChunkCells != 0u) {
        error = std::string(operation) +
                " ID row stride must be a positive multiple of 64 within "
                "the MTP KV cell limit";
        return false;
    }

    constexpr std::size_t query_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    const std::size_t id_bytes =
        static_cast<std::size_t>(buffers.id_row_stride) *
        sizeof(std::int32_t);
    const std::size_t chunk_count =
        buffers.id_row_stride / kMtpAttentionChunkCells;
    std::size_t accumulator_floats = 0u;
    std::size_t stats_floats = 0u;
    std::size_t accumulator_bytes = 0u;
    std::size_t stats_bytes = 0u;
    std::size_t both_stats_bytes = 0u;
    std::size_t required_scratch_bytes = 0u;
    std::size_t reserved_tail_bytes = 0u;
    if (!checked_size_multiply(
            chunk_count,
            static_cast<std::size_t>(N_HEAD_KV) *
                (N_HEAD_Q / N_HEAD_KV) * HEAD_DIM,
            accumulator_floats) ||
        !checked_size_multiply(
            chunk_count,
            static_cast<std::size_t>(N_HEAD_KV) *
                (N_HEAD_Q / N_HEAD_KV),
            stats_floats) ||
        !checked_size_multiply(accumulator_floats, sizeof(float),
                               accumulator_bytes) ||
        !checked_size_multiply(stats_floats, sizeof(float), stats_bytes) ||
        !checked_size_multiply(stats_bytes, 2u, both_stats_bytes) ||
        !checked_size_multiply(kMtpAttentionReservedScratchFloats,
                               sizeof(float), reserved_tail_bytes) ||
        !checked_size_add(accumulator_bytes, both_stats_bytes,
                          scratch_active_bytes) ||
        !checked_size_add(scratch_active_bytes, reserved_tail_bytes,
                          required_scratch_bytes) ||
        !checked_size_multiply(
            static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM, sizeof(float),
            attention_bytes)) {
        error = std::string(operation) +
                " active buffer extent overflows host address space";
        return false;
    }

    if (!validate_buffer_view(context, buffers.qcur, query_bytes,
                              alignof(float), operation, "qcur", error) ||
        !validate_buffer_view(context, buffers.mtp_ids, id_bytes,
                              alignof(std::int32_t), operation, "mtp_ids",
                              error) ||
        !validate_buffer_view(context, buffers.att_scratch,
                              required_scratch_bytes, alignof(float),
                              operation, "att_scratch", error) ||
        !validate_buffer_view(context, buffers.attn, attention_bytes,
                              alignof(float), operation, "attn", error)) {
        return false;
    }

    steps = buffers.params.slice(offsetof(MtpParams, step), sizeof(StepRow));
    if (!steps) {
        error = std::string(operation) +
                " cannot form its active StepRow view";
        return false;
    }
    scratch.partial_acc =
        buffers.att_scratch.slice(0u, accumulator_bytes);
    scratch.partial_m =
        buffers.att_scratch.slice(accumulator_bytes, stats_bytes);
    scratch.partial_l =
        buffers.att_scratch.slice(accumulator_bytes + stats_bytes,
                                  stats_bytes);
    if (!scratch.partial_acc || !scratch.partial_m || !scratch.partial_l) {
        error = std::string(operation) +
                " cannot form the chunk scratch views";
        return false;
    }

    struct ActiveView {
        const char* name;
        BufferRef view;
        std::size_t bytes;
    };
    const std::array<ActiveView, 7> inputs{{
        {"params", buffers.params, sizeof(MtpParams)},
        {"qcur", buffers.qcur, query_bytes},
        {"mtp_ids", buffers.mtp_ids, id_bytes},
        {"K cache", buffers.kv_views.k, canonical_layout.k.bytes},
        {"K scales", buffers.kv_views.k_scale,
         canonical_layout.k_scale.bytes},
        {"V cache", buffers.kv_views.v, canonical_layout.v.bytes},
        {"V scales", buffers.kv_views.v_scale,
         canonical_layout.v_scale.bytes},
    }};
    const std::array<ActiveView, 2> outputs{{
        {"attention scratch", buffers.att_scratch, required_scratch_bytes},
        {"attn", buffers.attn, attention_bytes},
    }};
    if (overlaps(buffers.att_scratch, required_scratch_bytes,
                 buffers.attn, attention_bytes)) {
        error = std::string(operation) +
                " attn output overlaps the attention scratch region";
        return false;
    }
    for (std::size_t write = 0u; write < outputs.size(); ++write) {
        for (std::size_t read = 0u; read < inputs.size(); ++read) {
            if (overlaps(inputs[read].view, inputs[read].bytes,
                         outputs[write].view, outputs[write].bytes)) {
                error = std::string(operation) + " " + outputs[write].name +
                        " overlaps input " + inputs[read].name;
                return false;
            }
        }
    }
    const std::array<std::pair<BufferRef, std::size_t>, 3> active_scratch{{
        {scratch.partial_acc, accumulator_bytes},
        {scratch.partial_m, stats_bytes},
        {scratch.partial_l, stats_bytes},
    }};
    for (std::size_t left = 0u; left < active_scratch.size(); ++left) {
        if (overlaps(buffers.attn, attention_bytes,
                     active_scratch[left].first,
                     active_scratch[left].second)) {
            error = std::string(operation) +
                    " attn output overlaps chunk scratch";
            return false;
        }
        for (std::size_t right = left + 1u;
             right < active_scratch.size(); ++right) {
            if (overlaps(active_scratch[left].first,
                         active_scratch[left].second,
                         active_scratch[right].first,
                         active_scratch[right].second)) {
                error = std::string(operation) +
                        " chunk scratch views overlap";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

bool validate_mtp_attention_output_projection(
    Context& context, const MtpAttentionOutputProjectionBuffers& buffers,
    ReadOnlyBufferRef& active_rows, std::string& error) {
    constexpr char operation[] = "MTP attention output projection";
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }

    constexpr std::size_t attention_bytes =
        static_cast<std::size_t>(kMtpOutputProjectionInputWidth) *
        sizeof(float);
    constexpr std::size_t q_full_bytes =
        static_cast<std::size_t>(kMtpQueryProjectionWidth) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t sink_bytes = sizeof(float);
    constexpr std::size_t activation_bytes =
        (static_cast<std::size_t>(kMtpOutputProjectionInputWidth) /
         kMtpQ8BlockValues) *
        kMtpQ81BlockBytes;
    constexpr std::size_t weight_blocks =
        static_cast<std::size_t>(N_EMBD) *
        (kMtpOutputProjectionInputWidth / kMtpQ8BlockValues);
    constexpr std::size_t weight_bytes =
        weight_blocks * sizeof(block_q8_0);

    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error) ||
        !validate_read_only_buffer(
            context, context.buffers().read_only(buffers.attn),
            attention_bytes, alignof(float), "attn", operation, error) ||
        !validate_read_only_buffer(
            context, context.buffers().read_only(buffers.q_full),
            q_full_bytes, alignof(float), "q_full", operation, error) ||
        !validate_buffer_view(context, buffers.attn32, attention_bytes,
                              256u, operation, "attn32", error) ||
        !validate_buffer_view(context, buffers.xq, activation_bytes, 256u,
                              operation, "stage-14 xq", error) ||
        !validate_read_only_buffer(context, buffers.o_weights, weight_bytes,
                                   alignof(block_q8_0), "o_proj weights",
                                   operation, error) ||
        !validate_buffer_view(context, buffers.bo, output_bytes,
                              alignof(float), operation, "bo", error) ||
        !validate_buffer_view(context, buffers.sink, sink_bytes,
                              alignof(float), operation, "sink", error)) {
        return false;
    }

    active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    if (!active_rows) {
        error = std::string(operation) +
                " cannot form its active-row metadata view";
        return false;
    }

    struct InputView {
        const char* name;
        ReadOnlyBufferRef view;
        std::size_t bytes;
    };
    const std::array<InputView, 4> inputs{{
        {"params", buffers.params, sizeof(MtpParams)},
        {"attn", context.buffers().read_only(buffers.attn), attention_bytes},
        {"q_full", context.buffers().read_only(buffers.q_full), q_full_bytes},
        {"o_proj weights", buffers.o_weights, weight_bytes},
    }};
    struct OutputView {
        const char* name;
        BufferRef view;
        std::size_t bytes;
    };
    const std::array<OutputView, 4> outputs{{
        {"attn32", buffers.attn32, attention_bytes},
        {"stage-14 xq", buffers.xq, activation_bytes},
        {"bo", buffers.bo, output_bytes},
        {"sink", buffers.sink, sink_bytes},
    }};
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            if (overlaps(input.view, input.bytes, output.view,
                         output.bytes)) {
                error = std::string(operation) + " " + output.name +
                        " overlaps input " + input.name;
                return false;
            }
        }
    }
    for (std::size_t input = 1u; input < inputs.size(); ++input) {
        if (overlaps(inputs[0].view, inputs[0].bytes, inputs[input].view,
                     inputs[input].bytes)) {
            error = std::string(operation) +
                    " parameter block overlaps input " + inputs[input].name;
            return false;
        }
    }
    for (std::size_t left = 0u; left < outputs.size(); ++left) {
        for (std::size_t right = left + 1u; right < outputs.size(); ++right) {
            if (overlaps(outputs[left].view, outputs[left].bytes,
                         outputs[right].view, outputs[right].bytes)) {
                error = std::string(operation) + " outputs " +
                        outputs[left].name + " and " + outputs[right].name +
                        " overlap";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_mtp_add_streams(Context& context, Encoder& encoder,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error) {
    error.clear();
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, "MTP add streams", rows,
                       error)) {
        return false;
    }

    const std::size_t h2_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t e2_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    if (!validate_read_only_buffer(context, buffers.h2, h2_bytes,
                                   alignof(float), "h2",
                                   "MTP add streams", error) ||
        !validate_read_only_buffer(context, buffers.e2, e2_bytes,
                                   alignof(float), "e2",
                                   "MTP add streams", error) ||
        !validate_output_buffer(context, buffers.output, h2_bytes,
                                "MTP add streams", "output", error)) {
        return false;
    }

    if (overlaps(buffers.params, sizeof(MtpParams), buffers.output, h2_bytes) ||
        overlaps(buffers.h2, h2_bytes, buffers.output, h2_bytes) ||
        overlaps(buffers.e2, e2_bytes, buffers.output, h2_bytes)) {
        error = "MTP add streams output overlaps a read-only input";
        return false;
    }

    const std::array read_only_bindings{
        EncoderReadOnlyBufferBinding{buffers.params, 0u, sizeof(MtpParams),
                                     alignof(MtpParams)},
        EncoderReadOnlyBufferBinding{buffers.h2, 1u, h2_bytes,
                                     alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.e2, 2u, e2_bytes,
                                     alignof(float)},
    };
    const std::array writable_bindings{
        EncoderBufferBinding{buffers.output, 3u, h2_bytes, alignof(float)},
    };
    constexpr std::array<EncoderBytesBinding, 0> bytes{};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_mtp_add_streams", {kMtpAddStreamsGroupsX, rows, 1u},
        {kMtpAddStreamsThreads, 1u, 1u}, writable_bindings,
        read_only_bindings, bytes, threadgroup_memory, error);
}

bool encode_mtp_add_streams(Context& context, CommandBuffer& command_buffer,
                            const MtpAddStreamsBuffers& buffers,
                            std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP add streams",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_add_streams(context, encoder, buffers,
                                          encode_error);
        },
        error);
}

bool encode_mtp_attention_read(Context& context, Encoder& encoder,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP attention read requires an ordered concurrent-group boundary";
        return false;
    }

    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, "MTP attention read", rows,
                       error)) {
        return false;
    }

    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    const std::size_t injection_bytes =
        static_cast<std::size_t>(rows) * HC * sizeof(float);
    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * HC_DIM * sizeof(float);
    const std::size_t low_rank_bytes =
        static_cast<std::size_t>(rows) * HC_LOW_RANK * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::array<std::pair<BufferRef, std::size_t>, 11> active_views{{
        {buffers.residual, residual_bytes},
        {buffers.dead_h2_scratch, block_bytes + injection_bytes},
        {buffers.norm, static_cast<std::size_t>(HC_DIM) * sizeof(float)},
        {buffers.normalized,
         residual_bytes},
        {buffers.rs, injection_bytes},
        {buffers.hc_down, weight_bytes},
        {buffers.hc_inject,
         static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t)},
        {buffers.lo, low_rank_bytes},
        {buffers.inject_out, injection_bytes},
        {buffers.hc_up, weight_bytes},
        {buffers.mixed, block_bytes},
    }};
    for (const auto& [view, active_bytes] : active_views) {
        if (overlaps(buffers.params, sizeof(MtpParams), view,
                     active_bytes)) {
            error = "MTP attention read view overlaps the parameter block";
            return false;
        }
    }

    if (!buffers.dead_h2_scratch ||
        !context.buffers().owns(buffers.dead_h2_scratch)) {
        error = "MTP attention read dead h2 scratch is not owned by this Context";
        return false;
    }
    if (buffers.dead_h2_scratch.length < block_bytes + injection_bytes ||
        buffers.dead_h2_scratch.offset % alignof(float) != 0u) {
        error = "MTP attention read dead h2 scratch is undersized or misaligned";
        return false;
    }
    const BufferRef unused_block_out =
        buffers.dead_h2_scratch.slice(0u, block_bytes);
    const BufferRef unused_injection =
        buffers.dead_h2_scratch.slice(block_bytes, injection_bytes);
    if (!unused_block_out || !unused_injection) {
        error = "MTP attention read cannot form its dead h2 scratch views";
        return false;
    }

    const GRReadBuffers gr_buffers{
        buffers.residual,
        unused_block_out,
        unused_injection,
        buffers.norm,
        buffers.normalized,
        buffers.rs,
        buffers.hc_down,
        buffers.hc_inject,
        buffers.lo,
        buffers.inject_out,
        buffers.hc_up,
        buffers.mixed,
    };
    return encode_gr_read(context, encoder, rows, false, gr_buffers, error);
}

bool encode_mtp_attention_read(Context& context,
                               CommandBuffer& command_buffer,
                               const MtpAttentionReadBuffers& buffers,
                               std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_read(context, encoder, buffers,
                                            encode_error);
        },
        error);
}

bool encode_mtp_attention_kv_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP attention K/V projection requires an ordered concurrent-group boundary";
        return false;
    }

    std::uint32_t rows = 0u;
    std::uint32_t compiled_columns = 0u;
    if (!validate_mtp_kv_projection(
            context, buffers, rows, compiled_columns, error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.xq);
    if (!encode_q8_1(context, encoder, rows, N_EMBD,
                     {buffers.mixed, buffers.xq}, error)) {
        return false;
    }
    if (!encode_q8_0_matvec(
            context, encoder, N_EMBD, kMtpKvProjectionWidth, rows,
            compiled_columns,
            {activations, buffers.key_weights, active_rows, buffers.kcur,
             buffers.sink},
            error)) {
        return false;
    }
    return encode_q8_0_matvec(
        context, encoder, N_EMBD, kMtpKvProjectionWidth, rows,
        compiled_columns,
        {activations, buffers.value_weights, active_rows, buffers.vcur,
         buffers.sink},
        error);
}

bool encode_mtp_attention_kv_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvProjectionBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention K/V projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_kv_projection(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_attention_kv_append(
    Context& context, Encoder& encoder,
    const MtpAttentionKvAppendBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP attention K/V append requires an ordered concurrent-group boundary";
        return false;
    }

    std::uint32_t rows = 0u;
    BufferRef steps;
    std::size_t current_bytes = 0u;
    if (!validate_mtp_kv_append(context, buffers, rows, steps,
                                current_bytes, error)) {
        return false;
    }

    const std::uint32_t head_rows = rows * kMtpKvHeadRows;
    const QsaNormRopeBuffers norm_rope{
        buffers.kcur, buffers.k_norm, steps, buffers.rope, buffers.kcur};
    if (!encode_qsa_norm_rope(context, encoder, rows,
                              QsaNormRopeMode::key, norm_rope, error)) {
        return false;
    }
    if (buffers.kv_rot) {
        if (!encode_qsa_fwht256(context, encoder, head_rows, buffers.kcur,
                                error) ||
            !encode_qsa_fwht256(context, encoder, head_rows, buffers.vcur,
                                error)) {
            return false;
        }
    }

    switch (buffers.kv_layout.storage_format) {
        case KvFormat::Fp16:
            return encode_qsa_kv_append_f16(
                context, encoder, rows, buffers.kv_layout.cell_capacity,
                {buffers.kcur, buffers.vcur, steps, buffers.kv_views.k,
                 buffers.kv_views.v},
                error);
        case KvFormat::Int8:
            return encode_qsa_kv_append_i8(
                context, encoder, rows, buffers.kv_layout.cell_capacity,
                {buffers.kcur, buffers.vcur, steps, buffers.kv_views.k,
                 buffers.kv_views.k_scale, buffers.kv_views.v,
                 buffers.kv_views.v_scale},
                error);
        case KvFormat::Q4_0:
            return encode_qsa_kv_append_q4(
                context, encoder, rows, buffers.kv_layout.cell_capacity,
                {buffers.kcur, buffers.vcur, steps, buffers.kv_views.k,
                 buffers.kv_views.v},
                error);
        case KvFormat::K8v4:
            break;
    }
    error = "MTP attention K/V append has an unsupported storage format";
    return false;
}

bool encode_mtp_attention_kv_append(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionKvAppendBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention K/V append",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_kv_append(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_attention_query_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryProjectionBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP attention query projection requires an ordered concurrent-group boundary";
        return false;
    }
    ReadOnlyBufferRef active_rows;
    if (!validate_mtp_query_projection(context, buffers, active_rows,
                                       error)) {
        return false;
    }
    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.xq);
    return encode_q8_0_matvec(
        context, encoder, N_EMBD, kMtpQueryProjectionWidth, 1u, 1u,
        {activations, buffers.q_weights, active_rows, buffers.q_full,
         buffers.sink},
        error);
}

bool encode_mtp_attention_query_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryProjectionBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention query projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_query_projection(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_attention_query_norm(
    Context& context, Encoder& encoder,
    const MtpAttentionQueryNormBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP attention query norm requires an ordered "
                "concurrent-group boundary";
        return false;
    }
    BufferRef active_steps;
    if (!validate_mtp_query_norm(context, buffers, active_steps, error)) {
        return false;
    }

    const QsaNormRopeBuffers norm_buffers{
        buffers.q_full, buffers.q_norm, active_steps, buffers.rope,
        buffers.qcur};
    if (!encode_qsa_norm_rope(context, encoder, 1u,
                              QsaNormRopeMode::query, norm_buffers, error)) {
        return false;
    }
    if (buffers.kv_rot &&
        !encode_qsa_fwht256(context, encoder, N_HEAD_Q, buffers.qcur, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_mtp_attention_query_norm(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionQueryNormBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention query norm",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_query_norm(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_window_ids(Context& context, Encoder& encoder,
                           const MtpWindowIdsBuffers& buffers,
                           std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP window IDs require an ordered concurrent-group boundary";
        return false;
    }
    BufferRef active_ids;
    std::uint32_t active_width = 0u;
    if (!validate_mtp_window_ids(context, buffers, active_ids, active_width,
                                 error)) {
        return false;
    }

    const EncoderReadOnlyBufferBinding params_binding{
        buffers.params, 0u, sizeof(MtpParams), alignof(MtpParams)};
    const EncoderBufferBinding ids_binding{
        active_ids, 1u,
        static_cast<std::size_t>(active_width) * sizeof(std::int32_t),
        alignof(std::int32_t)};
    const EncoderBytesBinding window_binding{
        &buffers.window, sizeof(buffers.window), 2u};
    const std::array buffers_binding{ids_binding};
    const std::array read_only_bindings{params_binding};
    const std::array bytes_bindings{window_binding};
    constexpr std::array<EncoderThreadgroupMemory, 0> threadgroup_memory{};
    return encoder.dispatch(
        "strata_mtp_window_ids", {kMtpWindowIdsGroups, 1u, 1u},
        {kMtpWindowIdsThreads, 1u, 1u}, buffers_binding,
        read_only_bindings, bytes_bindings, threadgroup_memory, error);
}

bool encode_mtp_window_ids(Context& context, CommandBuffer& command_buffer,
                           const MtpWindowIdsBuffers& buffers,
                           std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP window IDs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_window_ids(context, encoder, buffers,
                                         encode_error);
        },
        error);
}

bool encode_mtp_attention(Context& context, Encoder& encoder,
                          const MtpAttentionBuffers& buffers,
                          std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error =
            "MTP attention requires an ordered concurrent-group boundary";
        return false;
    }

    QsaAttentionMode mode{};
    BufferRef steps;
    QsaAttentionScratch scratch;
    std::size_t scratch_active_bytes = 0u;
    std::size_t attention_bytes = 0u;
    if (!validate_mtp_attention(context, buffers, mode, steps, scratch,
                                scratch_active_bytes, attention_bytes,
                                error)) {
        return false;
    }
    (void)scratch_active_bytes;
    (void)attention_bytes;

    const QsaAttentionChunkBuffers chunk_buffers{
        buffers.qcur, steps, buffers.mtp_ids,
        buffers.kv_views.k, buffers.kv_views.k_scale,
        buffers.kv_views.v, buffers.kv_views.v_scale, scratch};
    if (!encode_qsa_attention_chunk(
            context, encoder, mode, 1u, buffers.kv_layout.cell_capacity,
            buffers.id_row_stride, chunk_buffers, error)) {
        return false;
    }
    const QsaAttentionMergeBuffers merge_buffers{scratch, buffers.attn};
    if (!encode_qsa_attention_merge(context, encoder, 1u,
                                    buffers.id_row_stride, merge_buffers,
                                    error)) {
        return false;
    }
    if (buffers.kv_rot &&
        !encode_qsa_fwht256(context, encoder, N_HEAD_Q, buffers.attn,
                            error)) {
        return false;
    }
    error.clear();
    return true;
}

bool encode_mtp_attention(Context& context, CommandBuffer& command_buffer,
                          const MtpAttentionBuffers& buffers,
                          std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention(context, encoder, buffers,
                                        encode_error);
        },
        error);
}

bool encode_mtp_attention_output_projection(
    Context& context, Encoder& encoder,
    const MtpAttentionOutputProjectionBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP attention output projection requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    ReadOnlyBufferRef active_rows;
    if (!validate_mtp_attention_output_projection(
            context, buffers, active_rows, error)) {
        return false;
    }

    if (!encode_qsa_gate(
            context, encoder, 1u,
            {buffers.attn, buffers.q_full, buffers.attn32}, error) ||
        !encode_q8_1(
            context, encoder, 1u, kMtpOutputProjectionInputWidth,
            {buffers.attn32, buffers.xq}, error)) {
        return false;
    }
    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.xq);
    return encode_q8_0_matvec(
        context, encoder, kMtpOutputProjectionInputWidth, N_EMBD, 1u, 1u,
        {activations, buffers.o_weights, active_rows, buffers.bo,
         buffers.sink},
        error);
}

bool encode_mtp_attention_output_projection(
    Context& context, CommandBuffer& command_buffer,
    const MtpAttentionOutputProjectionBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP attention output projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_attention_output_projection(
                context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_feed_forward_read(Context& context, Encoder& encoder,
                                  const MtpFeedForwardReadBuffers& buffers,
                                  std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP feed-forward read requires an ordered "
                "concurrent-group boundary";
        return false;
    }

    constexpr char operation[] = "MTP feed-forward read";
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }
    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error)) {
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
    const std::size_t down_weight_bytes =
        static_cast<std::size_t>(HC_LOW_RANK) * HC_DIM *
        sizeof(std::uint16_t);
    const std::size_t inject_weight_bytes =
        static_cast<std::size_t>(HC) * HC_DIM * sizeof(std::uint16_t);
    const std::size_t up_weight_bytes =
        static_cast<std::size_t>(HC_DIM) * HC_LOW_RANK *
        sizeof(std::uint16_t);
    const std::array<std::pair<const char*, BufferRef>, 12> active_views{{
        {"residual", buffers.residual},
        {"bo", buffers.bo},
        {"inj", buffers.inj},
        {"norm", buffers.norm},
        {"normalized", buffers.normalized},
        {"rs", buffers.rs},
        {"hc_down", buffers.hc_down},
        {"hc_inject", buffers.hc_inject},
        {"lo", buffers.lo},
        {"inject_out", buffers.inject_out},
        {"hc_up", buffers.hc_up},
        {"mixed", buffers.mixed},
    }};
    const std::array<std::size_t, 12> active_bytes{
        residual_bytes,
        block_bytes,
        injection_bytes,
        norm_bytes,
        residual_bytes,
        injection_bytes,
        down_weight_bytes,
        inject_weight_bytes,
        low_rank_bytes,
        injection_bytes,
        up_weight_bytes,
        block_bytes,
    };
    for (std::size_t index = 0u; index < active_views.size(); ++index) {
        if (overlaps(buffers.params, sizeof(MtpParams),
                     active_views[index].second, active_bytes[index])) {
            error = std::string(operation) +
                    " parameter block overlaps active view " +
                    active_views[index].first;
            return false;
        }
    }

    const GRReadBuffers gr_buffers{
        buffers.residual,
        buffers.bo,
        buffers.inj,
        buffers.norm,
        buffers.normalized,
        buffers.rs,
        buffers.hc_down,
        buffers.hc_inject,
        buffers.lo,
        buffers.inject_out,
        buffers.hc_up,
        buffers.mixed,
    };
    return encode_gr_read(context, encoder, rows, true, gr_buffers, error);
}

bool encode_mtp_feed_forward_read(
    Context& context, CommandBuffer& command_buffer,
    const MtpFeedForwardReadBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP feed-forward read",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_feed_forward_read(context, encoder, buffers,
                                                encode_error);
        },
        error);
}

bool encode_mtp_router(Context& context, Encoder& encoder,
                       const MtpRouterBuffers& buffers,
                       std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP router requires an ordered concurrent-group boundary";
        return false;
    }

    constexpr char operation[] = "MTP router";
    std::uint32_t rows = 0u;
    if (!read_mtp_rows(context, buffers.params, operation, rows, error)) {
        return false;
    }
    if (rows != 1u) {
        error = std::string(operation) +
                " requires T=1 for the full-layer pass";
        return false;
    }
    if (!validate_read_only_buffer(context, buffers.params,
                                   sizeof(MtpParams), alignof(MtpParams),
                                   "params", operation, error)) {
        return false;
    }

    const ReadOnlyBufferRef active_rows =
        buffers.params.slice(offsetof(MtpParams, T), sizeof(std::uint32_t));
    if (!active_rows) {
        error = std::string(operation) +
                " could not form the active-row count view";
        return false;
    }

    constexpr std::size_t mixed_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t router_weight_bytes =
        static_cast<std::size_t>(kMtpRouterExperts) * N_EMBD *
        sizeof(std::uint16_t);
    constexpr std::size_t logits_bytes =
        static_cast<std::size_t>(kMtpRouterExperts) * sizeof(float);
    constexpr std::size_t ids_bytes =
        static_cast<std::size_t>(kMtpRouterTopK) * sizeof(std::int32_t);
    constexpr std::size_t route_weight_bytes =
        static_cast<std::size_t>(kMtpRouterTopK) * sizeof(float);
    constexpr std::size_t sink_bytes = sizeof(float);

    const std::array<std::pair<const char*, ReadOnlyBufferRef>, 2> inputs{{
        {"mixed", buffers.mixed},
        {"router_weights", buffers.router_weights},
    }};
    const std::array<std::size_t, 2> input_bytes{
        mixed_bytes, router_weight_bytes};
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        const std::size_t alignment =
            index == 0u ? 2u * alignof(float) : alignof(std::uint32_t);
        if (!validate_read_only_buffer(
                context, inputs[index].second, input_bytes[index], alignment,
                inputs[index].first, operation, error)) {
            return false;
        }
        if (overlaps(buffers.params, sizeof(MtpParams),
                     inputs[index].second, input_bytes[index])) {
            error = std::string(operation) +
                    " parameter block overlaps active view " +
                    inputs[index].first;
            return false;
        }
    }
    if (overlaps(buffers.mixed, mixed_bytes, buffers.router_weights,
                 router_weight_bytes)) {
        error = std::string(operation) +
                " mixed activations overlap router weights";
        return false;
    }

    if (!validate_output_buffer(context, buffers.logits_r, logits_bytes,
                                operation, "logits_r", error) ||
        !validate_buffer_view(context, buffers.ids, ids_bytes,
                              alignof(std::int32_t), operation, "ids",
                              error) ||
        !validate_buffer_view(context, buffers.weights, route_weight_bytes,
                              alignof(float), operation, "weights", error) ||
        !validate_output_buffer(context, buffers.matvec_sink, sink_bytes,
                                operation, "matvec sink", error)) {
        return false;
    }

    const std::array<std::pair<const char*, BufferRef>, 4> outputs{{
        {"logits_r", buffers.logits_r},
        {"ids", buffers.ids},
        {"weights", buffers.weights},
        {"matvec_sink", buffers.matvec_sink},
    }};
    const std::array<std::size_t, 4> output_bytes{
        logits_bytes, ids_bytes, route_weight_bytes, sink_bytes};
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
        if (overlaps(buffers.params, sizeof(MtpParams),
                     outputs[index].second, output_bytes[index])) {
            error = std::string(operation) +
                    " parameter block overlaps active view " +
                    outputs[index].first;
            return false;
        }
        for (std::size_t input = 0u; input < inputs.size(); ++input) {
            if (overlaps(inputs[input].second, input_bytes[input],
                         outputs[index].second, output_bytes[index])) {
                error = std::string(operation) + " output " +
                        outputs[index].first + " overlaps active input " +
                        inputs[input].first;
                return false;
            }
        }
        for (std::size_t prior = 0u; prior < index; ++prior) {
            if (overlaps(outputs[index].second, output_bytes[index],
                         outputs[prior].second, output_bytes[prior])) {
                error = std::string(operation) + " outputs " +
                        outputs[index].first + " and " +
                        outputs[prior].first + " overlap";
                return false;
            }
        }
    }

    const ReadOnlyBF16MatVecBuffers matvec_buffers{
        buffers.mixed, buffers.router_weights, active_rows,
        buffers.logits_r, buffers.matvec_sink};
    if (!encode_bf16_matvec(context, encoder, N_EMBD, kMtpRouterExperts,
                            1u, 1u, matvec_buffers, error)) {
        return false;
    }

    const ReadOnlyBufferRef logits =
        context.buffers().read_only(buffers.logits_r);
    if (!logits) {
        error = std::string(operation) +
                " could not form the route-logits input view";
        return false;
    }
    const MoeRouteBuffers route_buffers{
        logits, buffers.ids, buffers.weights};
    return encode_moe_route(context, encoder, 1u, kMtpRouterExperts, false,
                            route_buffers, error);
}

bool encode_mtp_router(Context& context, CommandBuffer& command_buffer,
                       const MtpRouterBuffers& buffers,
                       std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP router",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_router(context, encoder, buffers, encode_error);
        },
        error);
}

bool encode_mtp_routed_experts(
    Context& context, Encoder& encoder,
    const MtpRoutedExpertsBuffers& buffers, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MTP routed experts require an ordered concurrent-group boundary";
        return false;
    }

    constexpr char operation[] = "MTP routed experts";
    constexpr std::size_t mixed_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t ids_bytes =
        static_cast<std::size_t>(kMtpRoutedExpertTopK) *
        sizeof(std::int32_t);
    constexpr std::size_t expert_blob_bytes =
        static_cast<std::size_t>(kMtpExpertsFileBytes);
    constexpr std::size_t q8_mixed_bytes =
        static_cast<std::size_t>(N_EMBD);
    constexpr std::size_t q8_mixed_scales_bytes =
        static_cast<std::size_t>(N_EMBD / kMtpRoutedExpertChunkValues) *
        sizeof(float);
    constexpr std::size_t routed_hidden_values =
        static_cast<std::size_t>(kMtpRoutedExpertTopK) * N_FF;
    constexpr std::size_t routed_hidden_bytes =
        routed_hidden_values * sizeof(float);
    constexpr std::size_t routed_hidden_q8_bytes = routed_hidden_values;
    constexpr std::size_t routed_hidden_scales_bytes =
        routed_hidden_values / kMtpRoutedExpertChunkValues * sizeof(float);
    constexpr std::size_t parts_bytes =
        static_cast<std::size_t>(kMtpRoutedExpertTopK) * N_EMBD *
        sizeof(float);

    const std::array<std::pair<const char*, ReadOnlyBufferRef>, 3> inputs{{
        {"mixed", buffers.mixed},
        {"route IDs", buffers.ids},
        {"expert blob", buffers.expert_blob},
    }};
    const std::array<std::size_t, 3> input_bytes{
        mixed_bytes, ids_bytes, expert_blob_bytes};
    const std::array<std::size_t, 3> input_alignments{
        alignof(float), alignof(std::int32_t), alignof(std::uint32_t)};
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        if (!validate_read_only_buffer(
                context, inputs[index].second, input_bytes[index],
                input_alignments[index], inputs[index].first, operation,
                error)) {
            return false;
        }
        for (std::size_t prior = 0u; prior < index; ++prior) {
            if (overlaps(inputs[index].second, input_bytes[index],
                         inputs[prior].second, input_bytes[prior])) {
                error = std::string(operation) + " inputs " +
                        inputs[index].first + " and " + inputs[prior].first +
                        " overlap";
                return false;
            }
        }
    }

    const std::array<std::pair<const char*, BufferRef>, 8> outputs{{
        {"mixed Q8 codes", buffers.mq},
        {"mixed Q8 scales", buffers.ms},
        {"gate rows", buffers.gate},
        {"up rows", buffers.up},
        {"SwiGLU rows", buffers.eh},
        {"SwiGLU Q8 codes", buffers.hq},
        {"SwiGLU Q8 scales", buffers.hs},
        {"down parts", buffers.parts},
    }};
    const std::array<std::size_t, 8> output_bytes{
        q8_mixed_bytes, q8_mixed_scales_bytes, routed_hidden_bytes,
        routed_hidden_bytes, routed_hidden_bytes, routed_hidden_q8_bytes,
        routed_hidden_scales_bytes, parts_bytes};
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
        if (!validate_buffer_view(
                context, outputs[index].second, output_bytes[index],
                alignof(std::uint32_t), operation, outputs[index].first,
                error)) {
            return false;
        }
        for (std::size_t input = 0u; input < inputs.size(); ++input) {
            if (overlaps(inputs[input].second, input_bytes[input],
                         outputs[index].second, output_bytes[index])) {
                error = std::string(operation) + " output " +
                        outputs[index].first + " overlaps input " +
                        inputs[input].first;
                return false;
            }
        }
        for (std::size_t prior = 0u; prior < index; ++prior) {
            if (overlaps(outputs[index].second, output_bytes[index],
                         outputs[prior].second, output_bytes[prior])) {
                error = std::string(operation) + " outputs " +
                        outputs[index].first + " and " +
                        outputs[prior].first + " overlap";
                return false;
            }
        }
    }

    constexpr std::array<EncoderThreadgroupMemory, 0> no_threadgroup_memory{};
    const std::uint32_t mixed_width = N_EMBD;
    const EncoderBytesBinding mixed_width_binding{
        &mixed_width, sizeof(mixed_width), 3u};
    const std::array mixed_quantize_read_only{
        EncoderReadOnlyBufferBinding{buffers.mixed, 0u, mixed_bytes,
                                     alignof(float)},
    };
    const std::array mixed_quantize_writable{
        EncoderBufferBinding{buffers.mq, 1u, q8_mixed_bytes,
                             alignof(std::uint32_t)},
        EncoderBufferBinding{buffers.ms, 2u, q8_mixed_scales_bytes,
                             alignof(float)},
    };
    if (!encoder.dispatch(
            "strata_q_q8_0_scaled", {1u, 1u, 1u}, {128u, 1u, 1u},
            mixed_quantize_writable, mixed_quantize_read_only,
            std::span<const EncoderBytesBinding>(&mixed_width_binding, 1u),
            no_threadgroup_memory, error)) {
        return false;
    }

    const std::array routed_read_only{
        EncoderReadOnlyBufferBinding{buffers.ids, 0u, ids_bytes,
                                     alignof(std::int32_t)},
        EncoderReadOnlyBufferBinding{buffers.expert_blob, 1u,
                                     expert_blob_bytes,
                                     alignof(std::uint32_t)},
        EncoderReadOnlyBufferBinding{context.buffers().read_only(buffers.mq),
                                     2u, q8_mixed_bytes,
                                     alignof(std::uint32_t)},
        EncoderReadOnlyBufferBinding{context.buffers().read_only(buffers.ms),
                                     3u, q8_mixed_scales_bytes,
                                     alignof(float)},
    };
    const std::array routed_writable{
        EncoderBufferBinding{buffers.gate, 4u, routed_hidden_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.up, 5u, routed_hidden_bytes,
                             alignof(float)},
    };
    if (!encoder.dispatch("strata_mtp_gu_s2",
                          {kMtpRoutedExpertGroups,
                           kMtpRoutedExpertTopK, 1u},
                          {kMtpRoutedExpertThreads, 1u, 1u},
                          routed_writable, routed_read_only, {}, {},
                          error)) {
        return false;
    }

    const auto gate_read_only = context.buffers().read_only(buffers.gate);
    const auto up_read_only = context.buffers().read_only(buffers.up);
    if (!gate_read_only || !up_read_only) {
        error = std::string(operation) +
                " could not form read-only gate/up views";
        return false;
    }
    const std::array swiglu_read_only{
        EncoderReadOnlyBufferBinding{gate_read_only, 0u,
                                     routed_hidden_bytes, alignof(float)},
        EncoderReadOnlyBufferBinding{up_read_only, 1u,
                                     routed_hidden_bytes, alignof(float)},
    };
    const std::array swiglu_writable{
        EncoderBufferBinding{buffers.eh, 2u, routed_hidden_bytes,
                             alignof(float)},
    };
    if (!encoder.dispatch("strata_mtp_swiglu_s2",
                          {25u, 1u, 1u}, {256u, 1u, 1u},
                          swiglu_writable, swiglu_read_only, {}, {},
                          error)) {
        return false;
    }

    const std::uint32_t hidden_width =
        static_cast<std::uint32_t>(routed_hidden_values);
    const EncoderBytesBinding hidden_width_binding{
        &hidden_width, sizeof(hidden_width), 3u};
    const auto eh_read_only = context.buffers().read_only(buffers.eh);
    if (!eh_read_only) {
        error = std::string(operation) +
                " could not form a read-only SwiGLU view";
        return false;
    }
    const std::array hidden_quantize_read_only{
        EncoderReadOnlyBufferBinding{eh_read_only, 0u, routed_hidden_bytes,
                                     alignof(float)},
    };
    const std::array hidden_quantize_writable{
        EncoderBufferBinding{buffers.hq, 1u, routed_hidden_q8_bytes,
                             alignof(std::uint32_t)},
        EncoderBufferBinding{buffers.hs, 2u, routed_hidden_scales_bytes,
                             alignof(float)},
    };
    if (!encoder.dispatch(
            "strata_q_q8_0_scaled", {2u, 1u, 1u}, {128u, 1u, 1u},
            hidden_quantize_writable, hidden_quantize_read_only,
            std::span<const EncoderBytesBinding>(&hidden_width_binding, 1u),
            no_threadgroup_memory, error)) {
        return false;
    }

    const auto hq_read_only = context.buffers().read_only(buffers.hq);
    const auto hs_read_only = context.buffers().read_only(buffers.hs);
    if (!hq_read_only || !hs_read_only) {
        error = std::string(operation) +
                " could not form read-only quantized activation views";
        return false;
    }
    const std::array down_read_only{
        EncoderReadOnlyBufferBinding{buffers.ids, 0u, ids_bytes,
                                     alignof(std::int32_t)},
        EncoderReadOnlyBufferBinding{buffers.expert_blob, 1u,
                                     expert_blob_bytes,
                                     alignof(std::uint32_t)},
        EncoderReadOnlyBufferBinding{hq_read_only, 2u,
                                     routed_hidden_q8_bytes,
                                     alignof(std::uint32_t)},
        EncoderReadOnlyBufferBinding{hs_read_only, 3u,
                                     routed_hidden_scales_bytes,
                                     alignof(float)},
    };
    const std::array down_writable{
        EncoderBufferBinding{buffers.parts, 4u, parts_bytes,
                             alignof(float)},
    };
    return encoder.dispatch("strata_mtp_down_s2",
                            {kMtpRoutedExpertGroups,
                             kMtpRoutedExpertTopK, 1u},
                            {kMtpRoutedExpertThreads, 1u, 1u},
                            down_writable, down_read_only, {}, {},
                            error);
}

bool encode_mtp_routed_experts(
    Context& context, CommandBuffer& command_buffer,
    const MtpRoutedExpertsBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MTP routed experts",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_routed_experts(context, encoder, buffers,
                                             encode_error);
        },
        error);
}

}  // namespace strata::metal
