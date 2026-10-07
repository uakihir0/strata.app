// Feasibility micro-benchmarks for a Metal port of Strata (unified-memory bandwidth, mmap->MTLBuffer, gather).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <cstdio>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void stream_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                        constant ulong& n16 [[buffer(2)]],
                        uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  uint4 acc = 0;
  for (ulong i = gid; i < n16; i += gsz) acc ^= src[i];
  uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}
// gather: each threadgroup reads one block of blk16 uint4 at offsets[tg]
kernel void gather_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                        device const ulong* offs [[buffer(2)]], constant uint& blk16 [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                        uint tsz [[threads_per_threadgroup]]) {
  device const uint4* p = src + offs[tg];
  uint4 acc = 0;
  for (uint i = lid; i < blk16; i += tsz) acc ^= p[i];
  uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}
)";

static double run_stream(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                         id<MTLBuffer> buf, size_t bytes, id<MTLBuffer> out, int reps) {
  double best = 0;
  uint64_t n16 = bytes / 16;
  for (int r = 0; r < reps; r++) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:buf offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&n16 length:8 atIndex:2];
    NSUInteger tg = 1024, ntg = 20 * 64;  // plenty of threads in flight
    [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    double t = cb.GPUEndTime - cb.GPUStartTime;
    double gbs = bytes / t / 1e9;
    if (gbs > best) best = gbs;
  }
  return best;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", dev.name.UTF8String);
    printf("hasUnifiedMemory: %d\n", (int)dev.hasUnifiedMemory);
    printf("recommendedMaxWorkingSetSize: %.2f GiB\n", dev.recommendedMaxWorkingSetSize / 1073741824.0);
    printf("maxBufferLength: %.2f GiB\n", dev.maxBufferLength / 1073741824.0);
    printf("maxThreadgroupMemoryLength: %lu B\n", (unsigned long)dev.maxThreadgroupMemoryLength);
    printf("supports Apple10: %d Apple11: %d Metal4: %d\n", (int)[dev supportsFamily:MTLGPUFamilyApple10],
           (int)[dev supportsFamily:MTLGPUFamilyApple11], (int)[dev supportsFamily:MTLGPUFamilyMetal4]);
    NSError* err = nil;
    MTLCompileOptions* opt = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if (!lib) { printf("compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream_read"] error:&err];
    id<MTLComputePipelineState> pg = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gather_read"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];

    // 1) Streaming read of a shared (unified) buffer.
    for (size_t gib : {1ul, 4ul, 8ul}) {
      size_t bytes = gib << 30;
      id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      memset(buf.contents, 0x5a, bytes);
      printf("GPU stream read, shared buffer %zu GiB: %.1f GB/s\n", gib, run_stream(dev, q, ps, buf, bytes, out, 8));
    }

    // 2) Gather of expert-sized blocks (Q2_0 expert ~1.38 MB) from a 16 GiB pool: 48 layers x 10 experts x W tokens.
    {
      size_t pool = 16ul << 30;
      id<MTLBuffer> buf = [dev newBufferWithLength:pool options:MTLResourceStorageModeShared];
      memset(buf.contents, 0x33, pool);
      const uint32_t blk = 1376256;  // 1.3125 MiB, 16B-aligned
      uint32_t blk16 = blk / 16;
      for (int nblk : {480, 1920}) {
        std::mt19937_64 rng(42);
        id<MTLBuffer> offs = [dev newBufferWithLength:nblk * 8 options:MTLResourceStorageModeShared];
        uint64_t* o = (uint64_t*)offs.contents;
        for (int i = 0; i < nblk; i++) o[i] = (rng() % ((pool - blk) / 4096)) * 4096 / 16;
        double best = 0;
        for (int r = 0; r < 8; r++) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pg];
          [e setBuffer:buf offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
          [e setBuffer:offs offset:0 atIndex:2]; [e setBytes:&blk16 length:4 atIndex:3];
          [e dispatchThreadgroups:MTLSizeMake(nblk,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          double t = cb.GPUEndTime - cb.GPUStartTime;
          double gbs = (double)nblk * blk / t / 1e9;
          if (gbs > best) best = gbs;
        }
        printf("GPU gather %d random 1.31 MiB blocks from 16 GiB: %.1f GB/s (%.2f ms per pass)\n", nblk, best,
               (double)nblk * blk / best / 1e6);
      }
    }

    // 3) CPU multi-threaded read bandwidth (all cores), to compare with the GPU on the same memory.
    {
      size_t bytes = 4ul << 30;
      std::vector<uint64_t> v(bytes / 8, 0x1111);
      unsigned nt = std::thread::hardware_concurrency();
      for (unsigned threads : {6u, nt}) {
        double best = 0;
        for (int r = 0; r < 5; r++) {
          std::vector<std::thread> th; std::vector<uint64_t> res(threads);
          auto t0 = std::chrono::steady_clock::now();
          for (unsigned t = 0; t < threads; t++) th.emplace_back([&, t] {
            size_t n = v.size() / threads, b = t * n; uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            const uint64_t* p = v.data() + b;
            for (size_t i = 0; i + 4 <= n; i += 4) { a0 ^= p[i]; a1 ^= p[i+1]; a2 ^= p[i+2]; a3 ^= p[i+3]; }
            res[t] = a0 ^ a1 ^ a2 ^ a3; });
          for (auto& x : th) x.join();
          double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          double gbs = bytes / s / 1e9; if (gbs > best) best = gbs;
          if (res[0] == 42) printf(".");
        }
        printf("CPU read, %u threads: %.1f GB/s\n", threads, best);
      }
    }

    // 4) mmap'd file wrapped as a no-copy MTLBuffer (the n-gram table / low-RAM expert path).
    if (argc > 1) {
      int fd = open(argv[1], O_RDONLY);
      struct stat st; fstat(fd, &st);
      size_t len = (size_t)st.st_size & ~((size_t)16383);
      void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
      id<MTLBuffer> fb = [dev newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared deallocator:nil];
      printf("mmap no-copy MTLBuffer of %.1f GiB: %s\n", len / 1073741824.0, fb ? "OK" : "FAILED");
      if (fb) {
        // first pass may fault pages in from SSD/page cache; report both
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        uint64_t n16 = len / 16;
        [e setComputePipelineState:ps]; [e setBuffer:fb offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:&n16 length:8 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1280,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        printf("  first GPU pass over mmap buffer: %.1f GB/s (status %ld)\n", len / (cb.GPUEndTime - cb.GPUStartTime) / 1e9, (long)cb.status);
        printf("  warm GPU passes over mmap buffer: %.1f GB/s\n", run_stream(dev, q, ps, fb, len, out, 5));
      }
    }
  }
  return 0;
}
