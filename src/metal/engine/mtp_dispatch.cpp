// src/metal/engine/mtp_dispatch.cpp - validated MTP stage encoders.
#include "strata/metal/mtp_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
constexpr std::uint32_t kMtpQ8BlockValues = 32u;
constexpr std::uint32_t kMtpQ81BlockBytes = 36u;
constexpr std::uint32_t kMtpAddStreamsThreads = 256u;
constexpr std::uint32_t kMtpAddStreamsGroupsX =
    HC_DIM / kMtpAddStreamsThreads;
constexpr std::uint32_t kMtpMaxRows =
    static_cast<std::uint32_t>(sizeof(MtpParams{}.tokens) /
                               sizeof(MtpParams{}.tokens[0]));

static_assert(HC_DIM % kMtpAddStreamsThreads == 0u);
static_assert(sizeof(MtpParams) == 236u);
static_assert(offsetof(MtpParams, T) == 0u);
static_assert(sizeof(Q81) == kMtpQ81BlockBytes);
static_assert(kMtpQueryProjectionWidth == 12'288u);
static_assert(kMtpQueryHeadWidth == 6'144u);

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

bool mtp_kv_views_match_layout(Context& context,
                              const MtpKvLayout& layout,
                              const MtpKvViews& views,
                              std::string& error) {
    constexpr char operation[] = "MTP attention K/V append";
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

}  // namespace strata::metal
