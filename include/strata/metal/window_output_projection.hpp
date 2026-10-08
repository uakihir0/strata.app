// GPU encoding for the final vocabulary projection of a Window.
#pragma once

#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/window.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace strata::metal {

class Encoder;

inline constexpr std::uint32_t kWindowOutputInputWidth = 2'560u;
inline constexpr std::uint32_t kWindowOutputVocabularySize = 248'320u;

struct WindowOutputProjectionBuffers {
    BufferRef mixed;
    BufferRef q8_1;
    ReadOnlyBufferRef active_rows;
    BufferRef logits;
    BufferRef matvec_sink;
};

/// Map the validated output.weight GGML type and dimensions to the immutable
/// dense mat-vec format. The expected GGUF shape is [2560, 248320].
bool window_output_weight_format(
    std::uint32_t ggml_type, std::span<const std::uint64_t> shape,
    MatVecWeightFormat& format, std::string& error);

/// Quantize the final Window head rows to Q8_1 and project them through the
/// live FullMode `output.weight` tensor in one ordered Encoder segment.
/// `full_mode` and all scratch views must remain alive through command
/// completion. The caller-owned form can fail after quantization was encoded;
/// propagate false from its enclosing Encoder::encode body so the segment is
/// discarded. Call outside an open concurrent-dispatch group.
bool encode_window_output_projection(
    Context& context, Encoder& encoder, const FullModeStorage& full_mode,
    std::uint32_t rows, const WindowOutputProjectionBuffers& buffers,
    std::string& error);

/// Standalone transactional form. Any validation or dispatch failure discards
/// the unsubmitted command buffer, including an already-encoded quantizer.
bool encode_window_output_projection(
    Context& context, CommandBuffer& command_buffer,
    const FullModeStorage& full_mode, std::uint32_t rows,
    const WindowOutputProjectionBuffers& buffers, std::string& error);

}  // namespace strata::metal
