// (1) Does CPU+GPU concurrent reading beat GPU alone on unified memory? (2) Cost of dependent dispatches in one encoder.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cstdio>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void rd(device const uint4* s [[buffer(0)]], device uint* o [[buffer(1)]], constant ulong& n [[buffer(2)]], uint g [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {"
      " uint4 a = 0; for (ulong i = g; i < n; i += gs) a ^= s[i]; if ((a.x ^ a.y ^ a.z ^ a.w) == 0x1234567u) o[0] = 1; }\n"
      "kernel void tiny(device float* x [[buffer(0)]], uint g [[thread_position_in_grid]]) { x[g] = x[g] * 0.999f + 1.0f; }"
      options:nil error:&err];
    id<MTLComputePipelineState> rd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rd"] error:&err];
    id<MTLComputePipelineState> tiny = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tiny"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    size_t gbytes = 8ul << 30;
    id<MTLBuffer> g = [dev newBufferWithLength:gbytes options:MTLResourceStorageModeShared];
    memset(g.contents, 1, gbytes);
    id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    size_t cbytes = 4ul << 30;
    std::vector<uint64_t> c(cbytes / 8, 3);

    auto gpu_pass = [&]() {
      uint64_t n = gbytes / 16;
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:rd]; [e setBuffer:g offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
      [e setBytes:&n length:8 atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(1280,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      return cb.GPUEndTime - cb.GPUStartTime;
    };
    gpu_pass();
    double ga = 1e9; for (int i = 0; i < 5; i++) ga = std::min(ga, gpu_pass());
    printf("GPU alone: %.1f GB/s\n", gbytes / ga / 1e9);

    for (int threads : {4, 8, 12}) {
      std::atomic<bool> stop{false}; std::atomic<uint64_t> cpu_bytes{0};
      std::vector<std::thread> th;
      for (int t = 0; t < threads; t++) th.emplace_back([&, t] {
        size_t n = c.size() / threads; const uint64_t* p = c.data() + t * n; uint64_t a = 0;
        while (!stop) { for (size_t i = 0; i < n; i += 4) a ^= p[i] ^ p[i+1] ^ p[i+2] ^ p[i+3]; cpu_bytes += n * 8; }
        if (a == 42) printf("."); });
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      uint64_t b0 = cpu_bytes; double t0 = now();
      double gt = 0; int passes = 6; for (int i = 0; i < passes; i++) gt += gpu_pass();
      double wall = now() - t0; uint64_t b1 = cpu_bytes;
      stop = true; for (auto& x : th) x.join();
      double gbs = gbytes * passes / gt / 1e9, cbs = (b1 - b0) / wall / 1e9;
      printf("concurrent, %2d CPU threads: GPU %.1f GB/s + CPU %.1f GB/s = %.1f GB/s total\n", threads, gbs, cbs, gbs + cbs);
    }

    // dependent tiny dispatches in one compute encoder (serial dispatch type = implicit barriers)
    id<MTLBuffer> x = [dev newBufferWithLength:4096 * 4 options:MTLResourceStorageModeShared];
    for (int n : {100, 1000}) {
      double best = 1e9;
      for (int r = 0; r < 5; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:tiny]; [e setBuffer:x offset:0 atIndex:0];
        double c0 = now();
        for (int i = 0; i < n; i++) [e dispatchThreads:MTLSizeMake(4096,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        double enc = now() - c0;
        [cb commit]; [cb waitUntilCompleted];
        double t = cb.GPUEndTime - cb.GPUStartTime;
        if (t < best) best = t;
        if (r == 4) printf("%4d dependent dispatches: GPU %.3f ms (%.2f us each), CPU encode %.3f ms\n", n, best * 1e3, best / n * 1e6, enc * 1e3);
      }
    }
    // empty command buffer round trip latency
    double best = 1e9;
    for (int r = 0; r < 50; r++) { double t0 = now(); id<MTLCommandBuffer> cb = [q commandBuffer]; [cb commit]; [cb waitUntilCompleted]; best = std::min(best, now() - t0); }
    printf("empty command buffer commit->complete: %.1f us\n", best * 1e6);
  }
  return 0;
}
