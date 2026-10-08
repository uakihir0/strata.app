// Resolve FullMode head weights and compose mixer plus vocabulary projection.
#include "strata/metal/window_output_head.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/window_head_mixer.hpp"
#include "strata/metal/window_output_projection.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kGgmlTypeF32 = 0u;
constexpr std::uint32_t kGgmlTypeBF16 = 30u;
constexpr std::uint32_t kHeadWidth = 10'240u;
constexpr std::uint32_t kHeadLowRank = 320u;
constexpr std::size_t kQ81BlockBytes = 36u;
constexpr std::size_t kQ81BlockValues = 32u;
constexpr std::size_t kWindowStreamCount = 4u;

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

bool resolve_head_tensor(const FullModeStorage& full_mode, Context& context,
                         const char* name, std::uint32_t expected_type,
                         std::vector<std::uint64_t> expected_shape,
                         FullModeTensorView& output, std::string& error) {
    if (!full_mode.dense_tensor_view(name, output, error)) {
        error = std::string("Window output head cannot resolve ") + name +
                ": " + error;
        return false;
    }
    if (output.ggml_type != expected_type ||
        output.shape != expected_shape ||
        !context.buffers().owns(output.buffer)) {
        error = std::string("Window output head has invalid FullMode tensor ") +
                name;
        return false;
    }
    return true;
}

bool get_scratch_view(Context& context, WindowScratchStorage& scratch,
                      WindowScratchRegion region, const char* name,
                      BufferRef& output, std::string& error) {
    if (!scratch.view(region, output, error) ||
        !context.buffers().owns(output)) {
        error = std::string("Window output head cannot resolve scratch ") +
                name + ": " + error;
        return false;
    }
    return true;
}

bool encode_output_head(Context& context, Encoder& encoder,
                        const FullModeStorage& full_mode,
                        WindowScratchStorage& scratch, std::uint32_t rows,
                        const WindowOutputHeadBuffers& buffers,
                        WindowOutputHeadResult& output,
                        std::string& error) {
    output = {};
    if (rows == 0u || rows > kWindowScratchTokenRows) {
        error = "Window output head row count must be in 1..8";
        return false;
    }
    if (!scratch.is_valid()) {
        error = "Window output head scratch storage is invalid";
        return false;
    }

    FullModeTensorView norm_gamma;
    FullModeTensorView down_weights;
    FullModeTensorView up_weights;
    if (!resolve_head_tensor(full_mode, context, "output_hc_norm.weight",
                             kGgmlTypeF32, {kHeadWidth}, norm_gamma, error) ||
        !resolve_head_tensor(full_mode, context, "output_hc_down.weight",
                             kGgmlTypeBF16, {kHeadWidth, kHeadLowRank},
                             down_weights, error) ||
        !resolve_head_tensor(full_mode, context, "output_hc_up.weight",
                             kGgmlTypeBF16, {kHeadLowRank, kHeadWidth},
                             up_weights, error)) {
        return false;
    }

    const WindowScratchLayout& layout = scratch.layout();
    BufferRef normalized;
    BufferRef low_rank;
    BufferRef gate;
    BufferRef mixed;
    BufferRef q8_1;
    BufferRef logits;
    BufferRef matvec_sink;
    if (!get_scratch_view(context, scratch, layout.head_normalized,
                          "head_normalized", normalized, error) ||
        !get_scratch_view(context, scratch, layout.head_low_rank,
                          "head_low_rank", low_rank, error) ||
        !get_scratch_view(context, scratch, layout.head_gate, "head_gate",
                          gate, error) ||
        !get_scratch_view(context, scratch, layout.head_mixed, "head_mixed",
                          mixed, error) ||
        !get_scratch_view(context, scratch, layout.head_q8_1, "head_q8_1",
                          q8_1, error) ||
        !get_scratch_view(context, scratch, layout.head_logits, "head_logits",
                          logits, error) ||
        !get_scratch_view(context, scratch, layout.matvec_sink, "matvec_sink",
                          matvec_sink, error)) {
        return false;
    }

    const std::size_t residual_bytes =
        static_cast<std::size_t>(rows) * kWindowStreamCount *
        kWindowOutputInputWidth * sizeof(float);
    const std::size_t q8_1_bytes =
        static_cast<std::size_t>(rows) *
        (kWindowOutputInputWidth / kQ81BlockValues) * kQ81BlockBytes;
    const std::size_t logits_bytes =
        static_cast<std::size_t>(rows) * kWindowOutputVocabularySize *
        sizeof(float);
    if (!context.buffers().owns(buffers.residual) ||
        buffers.residual.length < residual_bytes) {
        error = "Window output head residual view is undersized or unowned";
        return false;
    }
    if (buffer_ranges_overlap(buffers.residual, residual_bytes, q8_1,
                              q8_1_bytes) ||
        buffer_ranges_overlap(buffers.residual, residual_bytes, logits,
                              logits_bytes)) {
        error = "Window output head residual overlaps later projection output scratch";
        return false;
    }

    const WindowHeadMixerBuffers mixer_buffers{
        buffers.residual, norm_gamma.buffer, down_weights.buffer,
        up_weights.buffer, buffers.active_rows, normalized, low_rank, gate,
        mixed, matvec_sink};
    if (!encode_window_head_mixer(context, encoder, rows, mixer_buffers,
                                  error)) {
        return false;
    }

    const WindowOutputProjectionBuffers projection_buffers{
        mixed, q8_1, buffers.active_rows, logits, matvec_sink};
    if (!encode_window_output_projection(
            context, encoder, full_mode, rows, projection_buffers, error)) {
        return false;
    }

    const std::size_t active_logits_bytes =
        static_cast<std::size_t>(rows) * kWindowOutputVocabularySize *
        sizeof(float);
    output.logits = logits.slice(0u, active_logits_bytes);
    if (!output.logits) {
        error = "Window output head could not publish the active logits rows";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_window_output_head(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    WindowScratchStorage& scratch, std::uint32_t rows,
    const WindowOutputHeadBuffers& buffers, WindowOutputHeadResult& output,
    std::string& error) {
    return encode_output_head(context, encoder, full_mode, scratch, rows,
                              buffers, output, error);
}

bool encode_window_output_head(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, WindowScratchStorage& scratch,
    std::uint32_t rows, const WindowOutputHeadBuffers& buffers,
    WindowOutputHeadResult& output, std::string& error) {
    output = {};
    error.clear();
    return Encoder::encode(
        context, command_buffer, "Window FullMode output head",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_output_head(context, encoder, full_mode, scratch,
                                      rows, buffers, output, encode_error);
        },
        error);
}

}  // namespace strata::metal
