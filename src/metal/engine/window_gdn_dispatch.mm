// Ordered speculative Window GDN block encoding.
#include "strata/metal/window_gdn_dispatch.hpp"

#include "strata/metal/encoder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "../kernels/common/params.h"

namespace strata::metal {
namespace {

struct NamedView {
    BufferRef view;
    std::string_view name;
};

bool ranges_overlap(std::uint64_t left, std::size_t left_length,
                    std::uint64_t right, std::size_t right_length) {
    if (left_length == 0u || right_length == 0u) return false;
    if (left <= right) {
        return right - left < left_length;
    }
    return left - right < right_length;
}

bool views_overlap(BufferRef left, BufferRef right) {
    if (left.handle != nullptr && left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length)) {
        return true;
    }
    if (left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length)) {
        return true;
    }
    return left.gpu_address != 0u && right.gpu_address != 0u &&
           ranges_overlap(left.gpu_address, left.length, right.gpu_address,
                          right.length);
}

std::uint32_t compiled_columns(std::uint32_t rows) {
    if (rows <= 1u) return 1u;
    if (rows <= 2u) return 2u;
    if (rows <= 4u) return 4u;
    return 8u;
}

bool validate_views(Context& context, const WindowGDNBlockBuffers& buffers,
                    std::string& error) {
    const std::array views{
        NamedView{buffers.mixed, "mixed"},
        NamedView{buffers.history, "history"},
        NamedView{buffers.state, "state"},
        NamedView{buffers.qkv_weights, "qkv weights"},
        NamedView{buffers.conv_weights, "convolution weights"},
        NamedView{buffers.alpha, "alpha weights"},
        NamedView{buffers.beta_weights, "beta weights"},
        NamedView{buffers.dt_bias, "dt bias"},
        NamedView{buffers.a, "a"},
        NamedView{buffers.gate_weights, "gate weights"},
        NamedView{buffers.gamma, "gamma"},
        NamedView{buffers.output_weights, "output weights"},
        NamedView{buffers.xq, "xq"},
        NamedView{buffers.qkv, "qkv"},
        NamedView{buffers.h, "h"},
        NamedView{buffers.gate, "gate"},
        NamedView{buffers.beta, "beta"},
        NamedView{buffers.z, "z"},
        NamedView{buffers.y, "y"},
        NamedView{buffers.yq, "yq"},
        NamedView{buffers.block_out, "block output"},
        NamedView{buffers.sink, "matvec sink"},
        NamedView{buffers.active_rows, "active rows"},
    };
    for (std::size_t i = 0u; i < views.size(); ++i) {
        if (!views[i].view || !context.buffers().owns(views[i].view)) {
            error = "Window GDN " + std::string(views[i].name) +
                    " buffer is not owned by this Context";
            return false;
        }
        for (std::size_t j = 0u; j < i; ++j) {
            if (views_overlap(views[i].view, views[j].view)) {
                error = "Window GDN " + std::string(views[i].name) +
                        " buffer overlaps " + std::string(views[j].name);
                return false;
            }
        }
    }
    return true;
}

bool q8_1_weight_format(MatVecWeightFormat format) {
    return format != MatVecWeightFormat::bf16;
}

}  // namespace

bool encode_window_gdn_block(Context& context, Encoder& encoder,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Window GDN block requires an ordered concurrent-group boundary";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "Window GDN block row count must be in 1..8";
        return false;
    }
    if (!q8_1_weight_format(buffers.qkv_format) ||
        !q8_1_weight_format(buffers.gate_format) ||
        !q8_1_weight_format(buffers.output_format)) {
        error = "Window GDN Q8_1 matvec stages require quantized weight formats";
        return false;
    }
    if (!validate_views(context, buffers, error)) return false;

    const std::uint32_t columns = compiled_columns(rows);
    const MatVecBuffers qkv_matvec{buffers.xq, buffers.qkv_weights,
                                   buffers.qkv, buffers.sink,
                                   buffers.active_rows};
    const MatVecBuffers gate_matvec{buffers.xq, buffers.gate_weights,
                                    buffers.z, buffers.sink,
                                    buffers.active_rows};
    const MatVecBuffers output_matvec{buffers.yq, buffers.output_weights,
                                      buffers.block_out, buffers.sink,
                                      buffers.active_rows};

    if (!encode_q8_1(context, encoder, rows, N_EMBD,
                     {buffers.mixed, buffers.xq}, error)) {
        return false;
    }
    if (!encode_matvec(context, encoder, buffers.qkv_format, N_EMBD,
                       GDN_CONV_CHANNELS, rows, columns, qkv_matvec, error)) {
        return false;
    }
    if (!encode_gdn_conv_l2(context, encoder, rows,
                            {buffers.history, buffers.qkv,
                             buffers.conv_weights, buffers.h},
                            error)) {
        return false;
    }
    if (!encode_gdn_ab(context, encoder, rows,
                       {buffers.mixed, buffers.alpha, buffers.beta_weights,
                        buffers.dt_bias, buffers.a, buffers.gate, buffers.beta},
                       error)) {
        return false;
    }
    if (!encode_matvec(context, encoder, buffers.gate_format, N_EMBD,
                       GDN_VALUE_DIM, rows, columns, gate_matvec, error)) {
        return false;
    }
    if (!encode_gdn_step_norm(
            context, encoder, rows, GDNStateMode::window,
            {buffers.state, buffers.h, buffers.gate, buffers.beta, buffers.z,
             buffers.gamma, buffers.y},
            error)) {
        return false;
    }
    if (!encode_q8_1(context, encoder, rows, GDN_VALUE_DIM,
                     {buffers.y, buffers.yq}, error)) {
        return false;
    }
    return encode_matvec(context, encoder, buffers.output_format,
                         GDN_VALUE_DIM, N_EMBD, rows, columns, output_matvec,
                         error);
}

bool encode_window_gdn_block(Context& context, CommandBuffer& command_buffer,
                             std::uint32_t rows,
                             const WindowGDNBlockBuffers& buffers,
                             std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Window GDN block",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_gdn_block(context, encoder, rows, buffers,
                                           encode_error);
        },
        error);
}

}  // namespace strata::metal
