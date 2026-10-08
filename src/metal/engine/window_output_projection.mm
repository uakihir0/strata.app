// Ordered Window-head quantization and immutable vocabulary projection.
#include "strata/metal/window_output_projection.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/window_scratch.hpp"
#include "../kernels/common/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

namespace strata::metal {
namespace {

constexpr std::uint32_t kGgmlTypeQ5K = 13u;
constexpr std::uint32_t kGgmlTypeQ6K = 14u;
constexpr std::uint32_t kGgmlTypeIQ4XS = 23u;
constexpr std::uint32_t kGgmlValuesPerKBlock = 256u;
constexpr std::uint32_t kQ8_1ValuesPerBlock = 32u;
constexpr std::size_t kQ8_1BlockBytes = 36u;

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& result) {
    result = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

template <class Address>
bool address_ranges_overlap(Address left, std::size_t left_bytes,
                            Address right, std::size_t right_bytes) {
    if (left == 0 || right == 0) return false;
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool buffer_ranges_overlap(BufferRef left, std::size_t left_bytes,
                           BufferRef right, std::size_t right_bytes) {
    return (left.handle == right.handle &&
            address_ranges_overlap(left.offset, left_bytes,
                                   right.offset, right_bytes)) ||
           address_ranges_overlap(
               reinterpret_cast<std::uintptr_t>(left.contents), left_bytes,
               reinterpret_cast<std::uintptr_t>(right.contents), right_bytes) ||
           address_ranges_overlap(left.gpu_address, left_bytes,
                                  right.gpu_address, right_bytes);
}

bool read_only_buffer_ranges_overlap(ReadOnlyBufferRef left,
                                    std::size_t left_bytes,
                                    BufferRef right,
                                    std::size_t right_bytes) {
    return address_ranges_overlap(
               reinterpret_cast<std::uintptr_t>(left.contents()), left_bytes,
               reinterpret_cast<std::uintptr_t>(right.contents), right_bytes) ||
           address_ranges_overlap(left.gpu_address(), left_bytes,
                                  right.gpu_address, right_bytes);
}

bool require_disjoint(ReadOnlyBufferRef input, std::size_t input_bytes,
                      BufferRef output, std::size_t output_bytes,
                      const char* label, std::string& error) {
    if (read_only_buffer_ranges_overlap(input, input_bytes, output,
                                        output_bytes)) {
        error = std::string("Window output projection ") + label +
                " overlaps a read-only input";
        return false;
    }
    return true;
}

bool resolve_output_weight(const FullModeStorage& full_mode,
                           FullModeTensorView& output_weight,
                           MatVecWeightFormat& format,
                           std::size_t& required_weight_bytes,
                           std::string& error) {
    if (!full_mode.dense_tensor_view("output.weight", output_weight, error)) {
        error = "Window output projection cannot resolve output.weight: " +
                error;
        return false;
    }
    if (!window_output_weight_format(output_weight.ggml_type,
                                     output_weight.shape, format, error)) {
        error = "Window output projection has unsupported output.weight: " +
                error;
        return false;
    }

    std::size_t block_bytes = 0u;
    switch (format) {
        case MatVecWeightFormat::q5_k:
            block_bytes = sizeof(block_q5_K);
            break;
        case MatVecWeightFormat::q6_k:
            block_bytes = sizeof(block_q6_K);
            break;
        case MatVecWeightFormat::iq4_xs:
            block_bytes = sizeof(block_iq4_xs);
            break;
        default:
            error = "Window output projection selected an unsupported weight format";
            return false;
    }
    if (!checked_product(
            {kWindowOutputVocabularySize,
             kWindowOutputInputWidth / kGgmlValuesPerKBlock, block_bytes},
            required_weight_bytes)) {
        error = "Window output projection weight extent overflows host size";
        return false;
    }
    return true;
}

std::uint32_t compiled_columns(std::uint32_t rows) {
    std::uint32_t columns = 1u;
    while (columns < rows) columns <<= 1u;
    return columns;
}

bool validate_projection_views(Context& context,
                               const FullModeTensorView& output_weight,
                               std::size_t required_weight_bytes,
                               std::uint32_t rows,
                               const WindowOutputProjectionBuffers& buffers,
                               std::size_t& input_bytes,
                               std::size_t& q8_1_bytes,
                               std::size_t& logits_bytes,
                               std::size_t& sink_bytes,
                               std::string& error) {
    if (rows == 0u || rows > kWindowScratchTokenRows) {
        error = "Window output projection row count must be in 1..8";
        return false;
    }
    const std::uint32_t columns = compiled_columns(rows);
    if (!checked_product({rows, kWindowOutputInputWidth, sizeof(float)},
                         input_bytes) ||
        !checked_product(
            {rows, kWindowOutputInputWidth / kQ8_1ValuesPerBlock,
             kQ8_1BlockBytes},
            q8_1_bytes) ||
        !checked_product({rows, kWindowOutputVocabularySize, sizeof(float)},
                         logits_bytes) ||
        !checked_product(
            {columns == rows
                 ? 1u
                 : static_cast<std::size_t>(columns - rows) *
                       kWindowOutputVocabularySize,
             sizeof(float)},
            sink_bytes)) {
        error = "Window output projection scratch extent overflows host size";
        return false;
    }

    if (!context.buffers().owns(output_weight.buffer)) {
        error = "Window output projection weight is not owned by this Context";
        return false;
    }
    if (output_weight.buffer.length() < required_weight_bytes) {
        error = "Window output projection weight view is undersized";
        return false;
    }
    if (output_weight.buffer.gpu_address() % alignof(std::uint16_t) != 0u) {
        error = "Window output projection weight view is misaligned";
        return false;
    }
    if (!context.buffers().owns(buffers.mixed) ||
        !context.buffers().owns(buffers.q8_1) ||
        !context.buffers().owns(buffers.active_rows) ||
        !context.buffers().owns(buffers.logits) ||
        !context.buffers().owns(buffers.matvec_sink)) {
        error = "Window output projection scratch or row count is not owned by this Context";
        return false;
    }
    if (buffers.mixed.length < input_bytes ||
        buffers.q8_1.length < q8_1_bytes ||
        buffers.logits.length < logits_bytes ||
        buffers.matvec_sink.length < sink_bytes ||
        buffers.active_rows.length() < sizeof(std::uint32_t) ||
        buffers.active_rows.contents() == nullptr) {
        error = "Window output projection scratch or row-count view is undersized";
        return false;
    }

    std::uint32_t active_rows = 0u;
    std::memcpy(&active_rows, buffers.active_rows.contents(),
                sizeof(active_rows));
    if (active_rows != rows) {
        error = "Window output projection active-row count does not match rows";
        return false;
    }

    if (buffer_ranges_overlap(buffers.mixed, input_bytes, buffers.q8_1,
                              q8_1_bytes)) {
        error = "Window output projection Q8_1 scratch overlaps mixed input";
        return false;
    }
    if (read_only_buffer_ranges_overlap(output_weight.buffer, required_weight_bytes,
                                        buffers.q8_1, q8_1_bytes) ||
        read_only_buffer_ranges_overlap(buffers.active_rows,
                                        sizeof(std::uint32_t), buffers.q8_1,
                                        q8_1_bytes)) {
        error = "Window output projection Q8_1 output overlaps a later read-only input";
        return false;
    }
    if (!require_disjoint(output_weight.buffer, required_weight_bytes,
                          buffers.logits, logits_bytes, "logits", error) ||
        !require_disjoint(output_weight.buffer, required_weight_bytes,
                          buffers.matvec_sink, sink_bytes, "sink", error) ||
        !require_disjoint(buffers.active_rows, sizeof(std::uint32_t),
                          buffers.logits, logits_bytes, "logits", error) ||
        !require_disjoint(buffers.active_rows, sizeof(std::uint32_t),
                          buffers.matvec_sink, sink_bytes, "sink", error) ||
        buffer_ranges_overlap(buffers.mixed, input_bytes, buffers.logits,
                              logits_bytes) ||
        buffer_ranges_overlap(buffers.mixed, input_bytes,
                              buffers.matvec_sink, sink_bytes) ||
        buffer_ranges_overlap(buffers.logits, logits_bytes,
                              buffers.matvec_sink, sink_bytes)) {
        if (error.empty()) {
            error = "Window output projection outputs overlap mixed input or each other";
        }
        return false;
    }
    if (!require_disjoint(output_weight.buffer, required_weight_bytes,
                          buffers.q8_1, q8_1_bytes, "Q8_1 scratch", error) ||
        !require_disjoint(buffers.active_rows, sizeof(std::uint32_t),
                          buffers.q8_1, q8_1_bytes, "Q8_1 scratch", error) ||
        buffer_ranges_overlap(buffers.q8_1, q8_1_bytes, buffers.logits,
                             logits_bytes) ||
        buffer_ranges_overlap(buffers.q8_1, q8_1_bytes,
                              buffers.matvec_sink, sink_bytes) ||
        !require_disjoint(output_weight.buffer, required_weight_bytes,
                          buffers.mixed, input_bytes, "mixed input", error) ||
        !require_disjoint(buffers.active_rows, sizeof(std::uint32_t),
                          buffers.mixed, input_bytes, "mixed input", error)) {
        return false;
    }
    return true;
}

bool encode_projection(Context& context, Encoder& encoder,
                       const FullModeStorage& full_mode, std::uint32_t rows,
                       const WindowOutputProjectionBuffers& buffers,
                       std::string& error) {
    FullModeTensorView output_weight;
    MatVecWeightFormat format{};
    std::size_t required_weight_bytes = 0u;
    if (!resolve_output_weight(full_mode, output_weight, format,
                               required_weight_bytes, error)) {
        return false;
    }

    std::size_t input_bytes = 0u;
    std::size_t q8_1_bytes = 0u;
    std::size_t logits_bytes = 0u;
    std::size_t sink_bytes = 0u;
    if (!validate_projection_views(context, output_weight,
                                   required_weight_bytes, rows, buffers,
                                   input_bytes, q8_1_bytes, logits_bytes,
                                   sink_bytes, error)) {
        return false;
    }
    if (encoder.concurrent_group_open()) {
        error = "Window output projection cannot encode inside a concurrent dispatch group";
        return false;
    }

    const ReadOnlyBufferRef activations =
        context.buffers().read_only(buffers.q8_1);
    if (!activations) {
        error = "Window output projection cannot borrow its Q8_1 scratch read-only";
        return false;
    }
    if (!encode_q8_1(context, encoder, rows, kWindowOutputInputWidth,
                      {buffers.mixed, buffers.q8_1}, error)) {
        error = "Window output projection Q8_1 encoding failed: " + error;
        return false;
    }
    const ReadOnlyMatVecBuffers matvec_buffers{
        activations, output_weight.buffer, buffers.active_rows, buffers.logits,
        buffers.matvec_sink};
    if (!encode_read_only_matvec(
            context, encoder, format, kWindowOutputInputWidth,
            kWindowOutputVocabularySize, rows, compiled_columns(rows),
            matvec_buffers, error)) {
        error = "Window output projection mat-vec failed: " + error;
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool window_output_weight_format(
    std::uint32_t ggml_type, std::span<const std::uint64_t> shape,
    MatVecWeightFormat& format, std::string& error) {
    format = {};
    error.clear();
    if (shape.size() != 2u ||
        shape[0] != kWindowOutputInputWidth ||
        shape[1] != kWindowOutputVocabularySize) {
        error = "output.weight shape must be [2560, 248320]";
        return false;
    }
    switch (ggml_type) {
        case kGgmlTypeQ5K:
            format = MatVecWeightFormat::q5_k;
            return true;
        case kGgmlTypeQ6K:
            format = MatVecWeightFormat::q6_k;
            return true;
        case kGgmlTypeIQ4XS:
            format = MatVecWeightFormat::iq4_xs;
            return true;
        default:
            error = "output.weight GGML type must be Q5_K, Q6_K or IQ4_XS";
            return false;
    }
}

bool encode_window_output_projection(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    std::uint32_t rows, const WindowOutputProjectionBuffers& buffers,
    std::string& error) {
    return encode_projection(context, encoder, full_mode, rows, buffers,
                             error);
}

bool encode_window_output_projection(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, std::uint32_t rows,
    const WindowOutputProjectionBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window output projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_projection(context, encoder, full_mode, rows,
                                     buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
