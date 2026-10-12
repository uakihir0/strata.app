// Ordered Prefill QSA input projections.
#include "strata/metal/prefill_qsa_projection_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include "../kernels/common/params.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);

struct MemoryRange {
    void* handle = nullptr;
    std::uintptr_t cpu_address = 0u;
    std::uint64_t gpu_address = 0u;
    std::size_t offset = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

bool checked_bytes(std::size_t rows, std::size_t values_per_row,
                   std::size_t element_bytes, std::size_t& bytes) {
    if (element_bytes == 0u ||
        values_per_row >
            std::numeric_limits<std::size_t>::max() / element_bytes) {
        return false;
    }
    const std::size_t row_bytes = values_per_row * element_bytes;
    if (rows != 0u &&
        row_bytes > std::numeric_limits<std::size_t>::max() / rows) {
        return false;
    }
    bytes = rows * row_bytes;
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

MemoryRange memory_range(BufferRef view, std::size_t bytes,
                         std::string_view name) {
    return {view.handle, reinterpret_cast<std::uintptr_t>(view.contents),
            view.gpu_address, view.offset, bytes, name};
}

MemoryRange memory_range(ReadOnlyBufferRef view, std::size_t bytes,
                         std::string_view name) {
    return {nullptr, reinterpret_cast<std::uintptr_t>(view.contents()),
            view.gpu_address(), view.offset(), bytes, name};
}

bool ranges_overlap(const MemoryRange& left, const MemoryRange& right) {
    const bool same_allocation =
        left.handle != nullptr && left.handle == right.handle &&
        address_ranges_overlap(left.offset, left.bytes, right.offset,
                               right.bytes);
    const bool same_cpu_backing =
        address_ranges_overlap(left.cpu_address, left.bytes,
                               right.cpu_address, right.bytes);
    const bool same_gpu_backing =
        address_ranges_overlap(left.gpu_address, left.bytes,
                               right.gpu_address, right.bytes);
    return same_allocation || same_cpu_backing || same_gpu_backing;
}

template <class View>
bool validate_view(Context& context, View view, std::size_t required_bytes,
                   std::size_t alignment, std::string_view name,
                   std::string& error);

template <>
bool validate_view<BufferRef>(Context& context, BufferRef view,
                              std::size_t required_bytes,
                              std::size_t alignment, std::string_view name,
                              std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill QSA projection " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || view.offset % alignment != 0u) {
        error = "Prefill QSA projection " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

template <>
bool validate_view<ReadOnlyBufferRef>(Context& context, ReadOnlyBufferRef view,
                                      std::size_t required_bytes,
                                      std::size_t alignment,
                                      std::string_view name,
                                      std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill QSA projection " + std::string(name) +
                " buffer is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || view.offset() % alignment != 0u) {
        error = "Prefill QSA projection " + std::string(name) +
                " buffer is undersized or misaligned";
        return false;
    }
    return true;
}

void make_params(std::uint32_t rows, std::uint32_t output_width,
                 PrefillGemmMPPParams& params) {
    params = {rows, output_width, N_EMBD, output_width, 0u, false};
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       PrefillBF16X2Mode mode,
                       const PrefillQsaProjectionBuffers& buffers,
                       PrefillQsaProjectionBufferSizes& sizes,
                       std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill QSA projection row count exceeds the shader-safe "
                "range";
        return false;
    }
    if (mode != PrefillBF16X2Mode::off &&
        mode != PrefillBF16X2Mode::all_projections &&
        mode != PrefillBF16X2Mode::except_gr) {
        error = "Prefill QSA projection BF16X2 mode is invalid";
        return false;
    }
    const bool use_bf16x2 = mode != PrefillBF16X2Mode::off;
    if (use_bf16x2 != static_cast<bool>(buffers.mixed_bf_lo)) {
        error = use_bf16x2
                    ? "Prefill QSA projection BF16X2 mode requires "
                      "mixed_bf_lo"
                    : "Prefill QSA projection mode off requires an empty "
                      "mixed_bf_lo view";
        return false;
    }

    const std::uint32_t key_value_width = N_HEAD_KV * HEAD_DIM;
    const std::uint32_t query_width = N_HEAD_Q * 2u * HEAD_DIM;
    const std::uint32_t indexer_query_width =
        INDEXER_HEADS * INDEXER_HEAD_DIM;
    make_params(rows, key_value_width, sizes.attn_k_params);
    make_params(rows, key_value_width, sizes.attn_v_params);
    make_params(rows, query_width, sizes.attn_q_params);
    make_params(rows, INDEXER_HEAD_DIM, sizes.indexer_k_params);
    make_params(rows, indexer_query_width, sizes.indexer_q_params);

    if (!prefill_native_projection_buffer_sizes(
            sizes.attn_k_params, buffers.attn_k_format, sizes.attn_k, error) ||
        !prefill_native_projection_buffer_sizes(
            sizes.attn_v_params, buffers.attn_v_format, sizes.attn_v, error) ||
        !prefill_native_projection_buffer_sizes(
            sizes.attn_q_params, buffers.attn_q_format, sizes.attn_q, error)) {
        return false;
    }

    const std::size_t rows_size = rows;
    if (!checked_bytes(rows_size, N_EMBD, kHalfBytes,
                       sizes.mixed_bf_bytes) ||
        !checked_bytes(N_EMBD, INDEXER_HEAD_DIM, kHalfBytes,
                       sizes.indexer_k_weights_bytes) ||
        !checked_bytes(N_EMBD, indexer_query_width, kHalfBytes,
                       sizes.indexer_q_weights_bytes) ||
        !checked_bytes(rows_size, INDEXER_HEAD_DIM, kFloatBytes,
                       sizes.indexer_k_output_bytes) ||
        !checked_bytes(rows_size, indexer_query_width, kFloatBytes,
                       sizes.indexer_q_output_bytes)) {
        error = "Prefill QSA projection buffer size overflows size_t";
        return false;
    }
    sizes.mixed_bf_lo_bytes =
        use_bf16x2 ? sizes.mixed_bf_bytes : 0u;
    sizes.dequantized_f16_bytes =
        std::max({sizes.attn_k.dequantized_f16_bytes,
                  sizes.attn_v.dequantized_f16_bytes,
                  sizes.attn_q.dequantized_f16_bytes});

    if (!validate_view(context, buffers.mixed_h,
                       sizes.attn_q.x_f16_bytes, kHalfBytes, "mixed_h",
                       error) ||
        !validate_view(context, buffers.mixed_bf, sizes.mixed_bf_bytes,
                       kHalfBytes, "mixed_bf", error) ||
        (use_bf16x2 &&
         !validate_view(context, buffers.mixed_bf_lo,
                        sizes.mixed_bf_lo_bytes, kHalfBytes, "mixed_bf_lo",
                        error)) ||
        !validate_view(context, buffers.attn_k_weights,
                       sizes.attn_k.weights_bytes,
                       sizes.attn_k.weights_alignment, "attn_k weights",
                       error) ||
        !validate_view(context, buffers.attn_v_weights,
                       sizes.attn_v.weights_bytes,
                       sizes.attn_v.weights_alignment, "attn_v weights",
                       error) ||
        !validate_view(context, buffers.attn_q_weights,
                       sizes.attn_q.weights_bytes,
                       sizes.attn_q.weights_alignment, "attn_q weights",
                       error) ||
        !validate_view(context, buffers.indexer_k_weights,
                       sizes.indexer_k_weights_bytes, kHalfBytes,
                       "indexer.k_proj weights", error) ||
        !validate_view(context, buffers.indexer_q_weights,
                       sizes.indexer_q_weights_bytes, kHalfBytes,
                       "indexer.q_proj weights", error) ||
        !validate_view(context, buffers.dequantized_f16,
                       sizes.dequantized_f16_bytes, kHalfBytes,
                       "dequantized_f16 scratch", error) ||
        !validate_view(context, buffers.key,
                       sizes.attn_k.y_f32_bytes, kFloatBytes, "Kc", error) ||
        !validate_view(context, buffers.value,
                       sizes.attn_v.y_f32_bytes, kFloatBytes, "Vc", error) ||
        !validate_view(context, buffers.query_full,
                       sizes.attn_q.y_f32_bytes, kFloatBytes, "Qf", error) ||
        !validate_view(context, buffers.indexer_key,
                       sizes.indexer_k_output_bytes, kFloatBytes, "idx_raw",
                       error) ||
        !validate_view(context, buffers.indexer_query,
                       sizes.indexer_q_output_bytes, kFloatBytes, "q_idx",
                       error)) {
        return false;
    }

    const std::array<MemoryRange, 6u> writable{
        memory_range(buffers.dequantized_f16, sizes.dequantized_f16_bytes,
                     "dequantized_f16 scratch"),
        memory_range(buffers.key, sizes.attn_k.y_f32_bytes, "Kc"),
        memory_range(buffers.value, sizes.attn_v.y_f32_bytes, "Vc"),
        memory_range(buffers.query_full, sizes.attn_q.y_f32_bytes, "Qf"),
        memory_range(buffers.indexer_key, sizes.indexer_k_output_bytes,
                     "idx_raw"),
        memory_range(buffers.indexer_query, sizes.indexer_q_output_bytes,
                     "q_idx")};
    const std::array<MemoryRange, 8u> read_only{
        memory_range(buffers.mixed_h, sizes.attn_q.x_f16_bytes, "mixed_h"),
        memory_range(buffers.mixed_bf, sizes.mixed_bf_bytes, "mixed_bf"),
        memory_range(buffers.mixed_bf_lo, sizes.mixed_bf_lo_bytes,
                     "mixed_bf_lo"),
        memory_range(buffers.attn_k_weights, sizes.attn_k.weights_bytes,
                     "attn_k weights"),
        memory_range(buffers.attn_v_weights, sizes.attn_v.weights_bytes,
                     "attn_v weights"),
        memory_range(buffers.attn_q_weights, sizes.attn_q.weights_bytes,
                     "attn_q weights"),
        memory_range(buffers.indexer_k_weights,
                     sizes.indexer_k_weights_bytes, "indexer.k_proj weights"),
        memory_range(buffers.indexer_q_weights,
                     sizes.indexer_q_weights_bytes, "indexer.q_proj weights")};

    for (std::size_t output = 0u; output < writable.size(); ++output) {
        for (std::size_t earlier = 0u; earlier < output; ++earlier) {
            if (ranges_overlap(writable[output], writable[earlier])) {
                error = "Prefill QSA projection writable stages overlap: " +
                        std::string(writable[earlier].name) + " and " +
                        std::string(writable[output].name);
                return false;
            }
        }
        for (const MemoryRange& input : read_only) {
            if (ranges_overlap(writable[output], input)) {
                error = "Prefill QSA projection writable " +
                        std::string(writable[output].name) +
                        " overlaps read-only " + std::string(input.name);
                return false;
            }
        }
    }
    return true;
}

BufferRef active_view(BufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
}

ReadOnlyBufferRef active_view(ReadOnlyBufferRef view, std::size_t bytes) {
    return view.slice(0u, bytes);
}

}  // namespace

bool prefill_qsa_projection_buffer_sizes(
    std::uint32_t rows, PrefillQsaProjectionBufferSizes& sizes,
    const PrefillQsaProjectionBuffers& buffers, std::string& error) {
    error.clear();
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill QSA projection row count exceeds the shader-safe "
                "range";
        return false;
    }

    PrefillQsaProjectionBufferSizes candidate{};
    make_params(rows, N_HEAD_KV * HEAD_DIM, candidate.attn_k_params);
    make_params(rows, N_HEAD_KV * HEAD_DIM, candidate.attn_v_params);
    make_params(rows, N_HEAD_Q * 2u * HEAD_DIM, candidate.attn_q_params);
    make_params(rows, INDEXER_HEAD_DIM, candidate.indexer_k_params);
    make_params(rows, INDEXER_HEADS * INDEXER_HEAD_DIM,
                candidate.indexer_q_params);
    if (!prefill_native_projection_buffer_sizes(
            candidate.attn_k_params, buffers.attn_k_format, candidate.attn_k,
            error) ||
        !prefill_native_projection_buffer_sizes(
            candidate.attn_v_params, buffers.attn_v_format, candidate.attn_v,
            error) ||
        !prefill_native_projection_buffer_sizes(
            candidate.attn_q_params, buffers.attn_q_format, candidate.attn_q,
            error)) {
        return false;
    }
    if (!checked_bytes(rows, N_EMBD, kHalfBytes,
                       candidate.mixed_bf_bytes) ||
        !checked_bytes(N_EMBD, INDEXER_HEAD_DIM, kHalfBytes,
                       candidate.indexer_k_weights_bytes) ||
        !checked_bytes(N_EMBD, INDEXER_HEADS * INDEXER_HEAD_DIM, kHalfBytes,
                       candidate.indexer_q_weights_bytes) ||
        !checked_bytes(rows, INDEXER_HEAD_DIM, kFloatBytes,
                       candidate.indexer_k_output_bytes) ||
        !checked_bytes(rows, INDEXER_HEADS * INDEXER_HEAD_DIM, kFloatBytes,
                       candidate.indexer_q_output_bytes)) {
        error = "Prefill QSA projection buffer size overflows size_t";
        return false;
    }
    candidate.mixed_bf_lo_bytes = candidate.mixed_bf_bytes;
    candidate.dequantized_f16_bytes =
        std::max({candidate.attn_k.dequantized_f16_bytes,
                  candidate.attn_v.dequantized_f16_bytes,
                  candidate.attn_q.dequantized_f16_bytes});
    sizes = candidate;
    return true;
}

bool encode_prefill_qsa_projection(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillQsaProjectionBuffers& buffers,
    std::string& error, PrefillGemmBackend* selected_backend) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill QSA projection requires an ordered "
                "concurrent-group boundary";
        return false;
    }
    PrefillQsaProjectionBufferSizes sizes;
    if (!validate_dispatch(context, rows, mode, buffers, sizes, error)) {
        return false;
    }

    const ReadOnlyBufferRef mixed_h =
        active_view(buffers.mixed_h, sizes.attn_q.x_f16_bytes);
    const ReadOnlyBufferRef mixed_bf =
        active_view(buffers.mixed_bf, sizes.mixed_bf_bytes);
    const BufferRef dequantized_f16 =
        active_view(buffers.dequantized_f16, sizes.dequantized_f16_bytes);
    const BufferRef key =
        active_view(buffers.key, sizes.attn_k.y_f32_bytes);
    const BufferRef value =
        active_view(buffers.value, sizes.attn_v.y_f32_bytes);
    const BufferRef query_full =
        active_view(buffers.query_full, sizes.attn_q.y_f32_bytes);
    const ReadOnlyBufferRef attn_k_weights =
        active_view(buffers.attn_k_weights, sizes.attn_k.weights_bytes);
    const ReadOnlyBufferRef attn_v_weights =
        active_view(buffers.attn_v_weights, sizes.attn_v.weights_bytes);
    const ReadOnlyBufferRef attn_q_weights =
        active_view(buffers.attn_q_weights, sizes.attn_q.weights_bytes);

    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    if (!encode_prefill_native_projection(
            context, encoder, sizes.attn_k_params, buffers.attn_k_format,
            {mixed_h, attn_k_weights,
             active_view(dequantized_f16,
                         sizes.attn_k.dequantized_f16_bytes),
             key},
            error, &selected) ||
        !encode_prefill_native_projection(
            context, encoder, sizes.attn_v_params, buffers.attn_v_format,
            {mixed_h, attn_v_weights,
             active_view(dequantized_f16,
                         sizes.attn_v.dequantized_f16_bytes),
             value},
            error, &selected) ||
        !encode_prefill_native_projection(
            context, encoder, sizes.attn_q_params, buffers.attn_q_format,
            {mixed_h, attn_q_weights,
             active_view(dequantized_f16,
                         sizes.attn_q.dequantized_f16_bytes),
             query_full},
            error, &selected)) {
        return false;
    }

    const ReadOnlyBufferRef indexer_k_weights =
        active_view(buffers.indexer_k_weights,
                    sizes.indexer_k_weights_bytes);
    const ReadOnlyBufferRef indexer_q_weights =
        active_view(buffers.indexer_q_weights,
                    sizes.indexer_q_weights_bytes);
    const BufferRef indexer_key =
        active_view(buffers.indexer_key, sizes.indexer_k_output_bytes);
    const BufferRef indexer_query =
        active_view(buffers.indexer_query, sizes.indexer_q_output_bytes);
    if (mode == PrefillBF16X2Mode::off) {
        if (!encode_prefill_gemm(
                context, encoder, sizes.indexer_k_params,
                {mixed_bf, indexer_k_weights, indexer_key},
                PrefillGemmFormat::BF16, error, &selected) ||
            !encode_prefill_gemm(
                context, encoder, sizes.indexer_q_params,
                {mixed_bf, indexer_q_weights, indexer_query},
                PrefillGemmFormat::BF16, error, &selected)) {
            return false;
        }
    } else {
        const ReadOnlyBufferRef mixed_bf_lo =
            active_view(buffers.mixed_bf_lo, sizes.mixed_bf_lo_bytes);
        if (!encode_prefill_gemm_bf16x2(
                context, encoder, sizes.indexer_k_params,
                {mixed_bf, mixed_bf_lo, indexer_k_weights, indexer_key},
                error, &selected) ||
            !encode_prefill_gemm_bf16x2(
                context, encoder, sizes.indexer_q_params,
                {mixed_bf, mixed_bf_lo, indexer_q_weights, indexer_query},
                error, &selected)) {
            return false;
        }
    }
    if (selected_backend != nullptr) *selected_backend = selected;
    return true;
}

bool encode_prefill_qsa_projection(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillQsaProjectionBuffers& buffers,
    std::string& error, PrefillGemmBackend* selected_backend) {
    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    const bool encoded = Encoder::encode(
        context, command_buffer, "Prefill QSA input projections",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_projection(
                context, encoder, rows, mode, buffers, encode_error,
                &selected);
        },
        error);
    if (encoded && selected_backend != nullptr) {
        *selected_backend = selected;
    }
    return encoded;
}

}  // namespace strata::metal
