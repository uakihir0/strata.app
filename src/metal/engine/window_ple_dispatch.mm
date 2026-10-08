// Ordered Window layer-0 write and PLE entry encoding.
#include "strata/metal/window_ple_dispatch.hpp"

#include "strata/metal/encoder.hpp"

namespace strata::metal {
namespace {

bool same_view(BufferRef left, BufferRef right) {
    return left && right && left.handle == right.handle &&
           left.contents == right.contents &&
           left.gpu_address == right.gpu_address &&
           left.length == right.length && left.offset == right.offset;
}

}  // namespace

bool encode_window_ple_entry(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowPLEEntryBuffers& buffers,
    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window PLE entry requires an ordered concurrent-group boundary";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "Window PLE entry row count must be in 1..8";
        return false;
    }

    const GRPostBuffers previous_write{
        buffers.ple.post.residual,
        buffers.previous_block_out,
        buffers.previous_inject,
        buffers.ple.post.residual,
    };
    if (!encode_gr_post(context, encoder, rows, previous_write, error)) {
        return false;
    }
    return encode_ple_block(context, encoder, rows, key_format,
                            compiled_columns, history_update, buffers.ple,
                            error);
}

bool encode_window_ple_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowPLEEntryBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window PLE entry",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_ple_entry(
                context, encoder, rows, key_format, compiled_columns,
                history_update, buffers, encode_error);
        },
        error);
}

bool encode_window_moe_ple_entry(
    Context& context, Encoder& encoder, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowMoePLEEntryBuffers& buffers,
    std::string& error) {
    error.clear();
    if (!same_view(buffers.previous_gr_read.residual,
                   buffers.ple.post.residual)) {
        error = "Window MoE PLE entry must reuse the previous GR residual";
        return false;
    }

    const WindowPLEEntryBuffers entry{
        buffers.previous_ffn.shared.output,
        buffers.previous_gr_read.inject_out,
        buffers.ple,
    };
    return encode_window_ple_entry(
        context, encoder, rows, key_format, compiled_columns,
        history_update, entry, error);
}

bool encode_window_moe_ple_entry(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    MatVecWeightFormat key_format, std::uint32_t compiled_columns,
    PLEHistoryUpdate history_update, const WindowMoePLEEntryBuffers& buffers,
    std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window pending MoE PLE entry",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_ple_entry(
                context, encoder, rows, key_format, compiled_columns,
                history_update, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
