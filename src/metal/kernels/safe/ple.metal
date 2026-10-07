// PLE operations (docs/mac/04-metal-kernels.md §5.3).
#include "math.h"
#include "params.h"
#include "reduce.h"

using namespace metal;

[[max_total_threads_per_threadgroup(512)]]
kernel void strata_ple_gate(device const float* key [[buffer(0)]],
                            device const float* query [[buffer(1)]],
                            device float* gate [[buffer(2)]],
                            constant float& scale [[buffer(3)]],
                            uint3 tid3 [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
    const uint row_stream = group.x;
    const uint row = row_stream / HC;
    const uint stream = row_stream % HC;
    const uint tid = tid3.x;
    const ulong offset = ulong(row) * HC_DIM + ulong(stream) * N_EMBD;

    float sum = 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        const uint col = tid + 512u * j;
        const float product =
            col < N_EMBD ? key[offset + col] * query[offset + col] : 0.0f;
        sum = sum + product;
    }

    threadgroup float partials[16];
    sum = bfly_sum(sum);
    if (lane == 0u) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        float total = lane < 16u ? partials[lane] : 0.0f;
        total = bfly_sum(total);
        if (lane == 0u) {
            const float scaled = fma(scale, total, 0.0f);
            const float magnitude = sqrt(fmax(fabs(scaled), 1e-6f));
            const float sign = float((scaled > 0.0f) - (scaled < 0.0f));
            gate[row_stream] = sigmoid_k(sign * magnitude);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_ple_broadcast(device const float* value [[buffer(0)]],
                                 device const float* gate [[buffer(1)]],
                                 device float* gated [[buffer(2)]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 group [[threadgroup_position_in_grid]]) {
    const uint index = group.x * 256u + tid3.x;
    const uint row = group.y;
    const uint stream = index / N_EMBD;
    const uint col = index % N_EMBD;
    const ulong value_index = ulong(row) * N_EMBD + col;
    const ulong gate_index = ulong(row) * HC + stream;
    const ulong output_index = ulong(row) * HC_DIM + index;
    gated[output_index] = value[value_index] * gate[gate_index];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_ple_conv_residual(device const float* ple_norm [[buffer(0)]],
                                     device const float* hist [[buffer(1)]],
                                     device const half* conv1d [[buffer(2)]],
                                     device const float* gated [[buffer(3)]],
                                     device float* residual [[buffer(4)]],
                                     uint3 tid3 [[thread_position_in_threadgroup]],
                                     uint3 group [[threadgroup_position_in_grid]]) {
    const uint channel = group.x * 256u + tid3.x;
    const uint row = group.y;
    const ulong row_offset = ulong(row) * HC_DIM + channel;

    float sum = 0.0f;
    for (uint tap = 0u; tap < 4u; ++tap) {
        const int norm_row = int(row) - int(PLE_HISTORY_ROWS) + int(3u * tap);
        const float value =
            norm_row >= 0
                ? ple_norm[ulong(norm_row) * HC_DIM + channel]
                : hist[ulong(channel) * PLE_HISTORY_ROWS +
                       uint(int(PLE_HISTORY_ROWS) + norm_row)];
        const float term =
            value * float(conv1d[ulong(channel) * 4u + tap]);
        sum = tap == 0u ? term : sum + term;
    }
    const float activation = silu_k(sum);
    residual[row_offset] = residual[row_offset] + (gated[row_offset] + activation);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_ple_history(device float* hist [[buffer(0)]],
                              device const float* ple_norm [[buffer(1)]],
                              constant uint& n [[buffer(2)]],
                              uint3 tid3 [[thread_position_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]]) {
    const uint channel = group.x * 256u + tid3.x;
    thread float next_history[PLE_HISTORY_ROWS];
    for (uint row = 0u; row < PLE_HISTORY_ROWS; ++row) {
        const int norm_row = int(n) - int(PLE_HISTORY_ROWS) + int(row);
        if (norm_row >= 0) {
            next_history[row] =
                ple_norm[ulong(norm_row) * HC_DIM + channel];
        } else {
            const uint old_row = n + row;
            next_history[row] =
                hist[ulong(channel) * PLE_HISTORY_ROWS + old_row];
        }
    }
    for (uint row = 0u; row < PLE_HISTORY_ROWS; ++row) {
        hist[ulong(channel) * PLE_HISTORY_ROWS + row] = next_history[row];
    }
}
