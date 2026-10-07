// Binding and ordering checks for doc 04, sections 4 and 8.
//
// 1. addr:     kernels read weights through a table of uint64 GPU addresses (buffer.gpuAddress + offset), with
//              32-B-aligned offsets, from a buffer that is only in a residency set (no setBuffer, no useResource).
// 2. offset:   a buffer bound with setBuffer:offset: at a 32-B-aligned (not 256-B-aligned) offset.
// 3. order:    Untracked buffers. A chain of dependent dispatches in one encoder: serial dispatch type,
//              concurrent with a buffer barrier after every dispatch, and concurrent with no barrier.
// 4. fence:    two command buffers on one queue, Untracked buffers, ordered by an MTLFence.
// 5. cost:     GPU time of 1,000 small dependent dispatches under each policy, and of independent pairs.
//
// Build: clang++ -std=c++17 -O2 -fobjc-arc binding.mm -framework Metal -framework Foundation -o binding
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ExpertAddr { device const uchar* role[3]; };

// One thread per entry: read a float at byte 0, a half at byte 2 and a uint at byte 4 of each role's blob.
kernel void addr_read(device const ExpertAddr* addr [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float s = 0.0f;
    for (int r = 0; r < 3; ++r) {
        device const uchar* p = addr[i].role[r];
        float f = *(device const float*)(p);
        half h = *(device const half*)(p + 4);
        uint u = *(device const uint*)(p + 8);
        s += f + float(h) + float(u & 0xffu);
    }
    out[i] = s;
}

// Reads a half array bound at an offset.
kernel void offset_read(device const half* w [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = float(w[i]);
}

// y = x * 1.0001 + 1, ping-pong between two buffers.
kernel void step(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant uint& n [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = fma(x[i], 1.0001f, 1.0f);
}
)MSL";

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pso(const char* name) {
    NSError* err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

static id<MTLBuffer> buf(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (!b) { fprintf(stderr, "alloc %zu failed\n", bytes); exit(1); }
    return b;
}

static double gpu_ms(id<MTLCommandBuffer> cb) { return (cb.GPUEndTime - cb.GPUStartTime) * 1e3; }

enum Policy { SERIAL, CONC_BARRIER, CONC_NONE, CONC_PAIRS };
static const char* policy_name[] = {"serial", "concurrent+barrier", "concurrent, no barrier", "concurrent, barrier per pair"};

// Encodes `chain` dependent steps over A/B. With pairs = true, every step is two independent dispatches
// (halves of the array) and the barrier goes after the pair.
static double run_chain(id<MTLComputePipelineState> p, id<MTLBuffer> A, id<MTLBuffer> B, uint32_t n, int chain,
                        Policy pol, bool pairs, bool* ok) {
    float* a = (float*)A.contents;
    for (uint32_t i = 0; i < n; ++i) a[i] = (float)(i % 1000);
    memset(B.contents, 0, n * 4);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e =
        pol == SERIAL ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial]
                      : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [e setComputePipelineState:p];
    for (int s = 0; s < chain; ++s) {
        id<MTLBuffer> x = (s % 2 == 0) ? A : B;
        id<MTLBuffer> y = (s % 2 == 0) ? B : A;
        if (!pairs) {
            [e setBuffer:x offset:0 atIndex:0];
            [e setBuffer:y offset:0 atIndex:1];
            [e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            uint32_t h = n / 2;
            for (int half = 0; half < 2; ++half) {
                [e setBuffer:x offset:half * h * 4 atIndex:0];
                [e setBuffer:y offset:half * h * 4 atIndex:1];
                [e setBytes:&h length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(h, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        if (pol == CONC_BARRIER || pol == CONC_PAIRS) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    // CPU reference
    std::vector<float> r(n);
    for (uint32_t i = 0; i < n; ++i) r[i] = (float)(i % 1000);
    for (int s = 0; s < chain; ++s)
        for (uint32_t i = 0; i < n; ++i) r[i] = std::fma(r[i], 1.0001f, 1.0f);
    const float* out = (const float*)((chain % 2 == 0) ? A.contents : B.contents);
    size_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) bad += memcmp(&out[i], &r[i], 4) != 0;
    *ok = bad == 0;
    return gpu_ms(cb);
}

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("device %s\n", dev.name.UTF8String);

        // ---- 1. addr table through a residency set only
        {
            const size_t data_bytes = 64ull << 20;
            const uint32_t n = 4096;
            id<MTLBuffer> D = buf(data_bytes);
            id<MTLBuffer> T = buf(n * 24);
            id<MTLBuffer> O = buf(n * 4);
            uint8_t* d = (uint8_t*)D.contents;
            memset(d, 0, data_bytes);
            uint64_t* t = (uint64_t*)T.contents;
            std::vector<float> ref(n, 0.0f);
            for (uint32_t i = 0; i < n; ++i) {
                for (int r = 0; r < 3; ++r) {
                    // 32-B-aligned offsets spread over the buffer, not 16-byte aligned to anything larger
                    uint64_t off = (((uint64_t)i * 3 + r) * 5413ull * 32ull + 32ull) % (data_bytes - 64);
                    off &= ~31ull;
                    float f = (float)(i * 3 + r);
                    __fp16 h = (__fp16)(0.5f * r);
                    uint32_t u = (i + r) & 0xff;
                    memcpy(d + off, &f, 4);
                    memcpy(d + off + 4, &h, 2);
                    memcpy(d + off + 8, &u, 4);
                    t[i * 3 + r] = D.gpuAddress + off;
                    ref[i] += f + (float)h + (float)u;
                }
            }
            MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
            rd.initialCapacity = 4;
            id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rd error:&err];
            if (!rs) { fprintf(stderr, "residency set: %s\n", err.localizedDescription.UTF8String); return 1; }
            [rs addAllocation:D];
            [rs commit];
            [rs requestResidency];
            [q addResidencySet:rs];
            id<MTLComputePipelineState> p = pso("addr_read");
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:T offset:0 atIndex:0];
            [e setBuffer:O offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const float* o = (const float*)O.contents;
            size_t bad = 0;
            for (uint32_t i = 0; i < n; ++i) bad += o[i] != ref[i];
            printf("1 addr table (residency set only, 32-B offsets): %s (%zu/%u wrong), cb status %ld\n",
                   bad ? "FAIL" : "ok", bad, n, (long)cb.status);
            [q removeResidencySet:rs];
        }

        // ---- 2. setBuffer:offset: at a 32-B-aligned offset
        {
            const uint32_t n = 1024;
            id<MTLBuffer> W = buf(1 << 20);
            id<MTLBuffer> O = buf(n * 4);
            __fp16* w = (__fp16*)((uint8_t*)W.contents + 32 * 37);
            for (uint32_t i = 0; i < n; ++i) w[i] = (__fp16)(i * 0.25f);
            id<MTLComputePipelineState> p = pso("offset_read");
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:W offset:32 * 37 atIndex:0];
            [e setBuffer:O offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const float* o = (const float*)O.contents;
            size_t bad = 0;
            for (uint32_t i = 0; i < n; ++i) bad += o[i] != i * 0.25f;
            printf("2 setBuffer offset 1184 (32-B aligned): %s (%zu wrong)\n", bad ? "FAIL" : "ok", bad);
        }

        id<MTLComputePipelineState> ps = pso("step");

        // ---- 3. ordering of a dependent chain, Untracked buffers
        {
            const uint32_t n = 1u << 20;
            id<MTLBuffer> A = buf(n * 4), B = buf(n * 4);
            for (int trial = 0; trial < 3; ++trial) {
                for (Policy pol : {SERIAL, CONC_BARRIER, CONC_NONE}) {
                    bool ok = false;
                    double ms = run_chain(ps, A, B, n, 64, pol, false, &ok);
                    printf("3 order, 64 dependent dispatches of 1M floats, %-24s: %s (%.3f ms)\n", policy_name[pol],
                           ok ? "exact" : "WRONG", ms);
                }
            }
        }

        // ---- 4. MTLFence across two command buffers, Untracked
        {
            const uint32_t n = 1u << 22;
            id<MTLBuffer> A = buf(n * 4), B = buf(n * 4), C = buf(n * 4);
            id<MTLFence> f = [dev newFence];
            int wrong_runs = 0, wrong_nofence = 0;
            for (int trial = 0; trial < 20; ++trial) {
                for (int use_fence = 1; use_fence >= 0; --use_fence) {
                    float* a = (float*)A.contents;
                    for (uint32_t i = 0; i < n; ++i) a[i] = (float)(i & 1023);
                    memset(B.contents, 0, n * 4);
                    memset(C.contents, 0, n * 4);
                    id<MTLCommandBuffer> cb1 = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e1 = [cb1 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [e1 setComputePipelineState:ps];
                    // 8 steps A->B->A..., ends in A after an even count; then one more A->B
                    for (int s = 0; s < 9; ++s) {
                        [e1 setBuffer:(s % 2 ? B : A) offset:0 atIndex:0];
                        [e1 setBuffer:(s % 2 ? A : B) offset:0 atIndex:1];
                        [e1 setBytes:&n length:4 atIndex:2];
                        [e1 dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [e1 memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    if (use_fence) [e1 updateFence:f];
                    [e1 endEncoding];
                    id<MTLCommandBuffer> cb2 = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    if (use_fence) [e2 waitForFence:f];
                    [e2 setComputePipelineState:ps];
                    [e2 setBuffer:B offset:0 atIndex:0];
                    [e2 setBuffer:C offset:0 atIndex:1];
                    [e2 setBytes:&n length:4 atIndex:2];
                    [e2 dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [e2 endEncoding];
                    [cb1 commit];
                    [cb2 commit];
                    [cb2 waitUntilCompleted];
                    std::vector<float> r(n);
                    size_t bad = 0;
                    const float* c = (const float*)C.contents;
                    for (uint32_t i = 0; i < n; ++i) {
                        float v = (float)(i & 1023);
                        for (int s = 0; s < 10; ++s) v = std::fma(v, 1.0001f, 1.0f);
                        bad += memcmp(&c[i], &v, 4) != 0;
                    }
                    if (bad) { if (use_fence) ++wrong_runs; else ++wrong_nofence; }
                }
            }
            printf("4 fence across command buffers: %d/20 wrong with the fence, %d/20 wrong without\n", wrong_runs,
                   wrong_nofence);
        }

        // ---- 5. cost of 1,000 small dependent dispatches, and of independent pairs
        {
            const uint32_t n = 16384;  // 64 threadgroups of 256: a small kernel
            id<MTLBuffer> A = buf(n * 4), B = buf(n * 4);
            for (int trial = 0; trial < 3; ++trial) {
                for (Policy pol : {SERIAL, CONC_BARRIER, CONC_NONE}) {
                    bool ok = false;
                    double ms = run_chain(ps, A, B, n, 1000, pol, false, &ok);
                    printf("5 cost, 1000 dispatches of 16K floats, %-24s: %.3f ms (%.2f us each)%s\n",
                           policy_name[pol], ms, ms, ok ? "" : " [result wrong, expected without barriers]");
                }
                bool ok = false;
                double ms_pairs_serial = run_chain(ps, A, B, n, 500, SERIAL, true, &ok);
                printf("5 cost, 500 steps x 2 independent halves, serial                  : %.3f ms%s\n",
                       ms_pairs_serial, ok ? "" : " WRONG");
                double ms_pairs = run_chain(ps, A, B, n, 500, CONC_PAIRS, true, &ok);
                printf("5 cost, 500 steps x 2 independent halves, concurrent, barrier/pair: %.3f ms%s\n", ms_pairs,
                       ok ? "" : " WRONG");
            }
            const uint32_t nb = 1u << 20;
            id<MTLBuffer> A2 = buf(nb * 4), B2 = buf(nb * 4);
            for (int trial = 0; trial < 2; ++trial) {
                bool ok = false;
                double s1 = run_chain(ps, A2, B2, nb, 200, SERIAL, false, &ok);
                double s2 = run_chain(ps, A2, B2, nb, 200, CONC_BARRIER, false, &ok);
                double s3 = run_chain(ps, A2, B2, nb, 100, SERIAL, true, &ok);
                double s4 = run_chain(ps, A2, B2, nb, 100, CONC_PAIRS, true, &ok);
                printf("5 cost, 1M floats: 200 serial %.3f ms, 200 conc+barrier %.3f ms, 100x2 serial %.3f ms, "
                       "100x2 conc pair-barrier %.3f ms\n", s1, s2, s3, s4);
            }
        }
    }
    return 0;
}
