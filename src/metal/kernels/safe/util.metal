// src/metal/kernels/safe/util.metal - copies and conversions (docs/mac/04-metal-kernels.md, section 5.13).
// Each kernel reads its row count from the caller's parameter word `rows` (bound at the block's T); widths and
// strides are setBytes constants. Threads past the end return.
#include "strata_common.h"
#include "math.h"
using namespace metal;

struct UtilCopyArgs {
    uint cols;
    uint src_stride;  // in floats; 0 repeats source row 0 for every destination row
    uint dst_stride;  // in floats
};

struct UtilBF16Args {
    uint cols;
    uint has_lo;      // nonzero writes the low BF16 part
};

[[max_total_threads_per_threadgroup(256)]]
kernel void strata_util_copy_strided(device const float* src [[buffer(0)]],
                                     device float* dst [[buffer(1)]],
                                     constant uint* rows [[buffer(2)]],
                                     constant UtilCopyArgs& args [[buffer(3)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
    const uint i = 256u * group + tid;
    const uint r = i / args.cols;
    const uint c = i - r * args.cols;
    if (r >= rows[0]) return;
    dst[r * args.dst_stride + c] = src[r * args.src_stride + c];
}

// Float32 to float16 with the rounding and special-value rules measured in appendix A.10: round to nearest even,
// subnormal halves kept, overflow to infinity from 65520, float32 denormals become signed zeros and every NaN becomes
// 0x7e00.
[[max_total_threads_per_threadgroup(256)]]
kernel void strata_util_to_f16(device const float* x [[buffer(0)]],
                               device ushort* y [[buffer(1)]],
                               constant uint* rows [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
    const uint i = 256u * group + tid;
    if (i >= rows[0] * cols) return;
    y[i] = as_type<ushort>(half(x[i]));
}

// Float32 to BF16 round-to-nearest-even (section 3.6). The low part is the BF16 of the rounding residual x - hi.
[[max_total_threads_per_threadgroup(256)]]
kernel void strata_util_to_bf16(device const float* x [[buffer(0)]],
                                device ushort* y [[buffer(1)]],
                                device ushort* ylo [[buffer(2)]],
                                constant uint* rows [[buffer(3)]],
                                constant UtilBF16Args& args [[buffer(4)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]]) {
    const uint i = 256u * group + tid;
    if (i >= rows[0] * args.cols) return;
    const ushort hi = f32_to_bf16_rne(x[i]);
    y[i] = hi;
    if (args.has_lo != 0u) {
        ylo[i] = f32_to_bf16_rne(x[i] - bf16_to_f32(hi));
    }
}
