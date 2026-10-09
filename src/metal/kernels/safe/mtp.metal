// MTP drafter kernels whose arithmetic follows docs/mac/04 §5.11.
#include "types.h"
#include "params.h"

using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_add_streams(
    constant MtpParams* params [[buffer(0)]],
    device const float* h2 [[buffer(1)]],
    device const float* e2 [[buffer(2)]],
    device float* R [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (group.y >= params->T) return;

    const uint value = 256u * group.x + tid3.x;
    if (value >= HC_DIM) return;

    const uint row = group.y;
    const uint output_index = row * HC_DIM + value;
    const uint embedding_index = row * N_EMBD + value % N_EMBD;
    R[output_index] = h2[output_index] + e2[embedding_index];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_mtp_window_ids(
    constant MtpParams* params [[buffer(0)]],
    device int* mtp_ids [[buffer(1)]],
    constant uint& window [[buffer(2)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const uint n_kv = uint(params->step[0].n_kv);
    const uint start = n_kv > window ? n_kv - window : 0u;
    const uint width = n_kv - start;

    for (uint index = 256u * group.x + tid3.x; index < width;
         index += 2048u) {
        mtp_ids[index] = int(start + index);
    }
}
