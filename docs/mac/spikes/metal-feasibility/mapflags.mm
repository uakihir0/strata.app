// Checks which file mappings newBufferWithBytesNoCopy accepts, and that the GPU reads the file's bytes through them.
// Build: clang++ -std=c++17 -O2 -fobjc-arc mapflags.mm -framework Metal -framework Foundation -o mapflags
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void xor_words(device const uint* src [[buffer(0)]], device atomic_uint* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                      uint gsz [[threads_per_grid]]) {
  uint acc = 0;
  for (uint i = gid; i < n; i += gsz) acc ^= src[i];
  atomic_fetch_xor_explicit(out, acc, memory_order_relaxed);
}
)";

static uint32_t gpu_xor(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> b,
                        size_t bytes) {
  id<MTLBuffer> out = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
  *(uint32_t*)out.contents = 0;
  uint32_t n = (uint32_t)(bytes / 4);
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:b offset:0 atIndex:0];
  [e setBuffer:out offset:0 atIndex:1];
  [e setBytes:&n length:4 atIndex:2];
  [e dispatchThreads:MTLSizeMake(65536, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) printf("  command buffer error: %s\n", cb.error.localizedDescription.UTF8String);
  return *(uint32_t*)out.contents;
}

static uint32_t cpu_xor(const uint8_t* p, size_t bytes) {
  uint32_t acc = 0;
  for (size_t i = 0; i < bytes / 4; i++) acc ^= ((const uint32_t*)p)[i];
  return acc;
}

int main() {
  @autoreleasepool {
    const size_t kSize = 256ull << 20, kPage = 16384;
    const char* path = "/tmp/strata_mapflags.bin";
    {
      std::vector<uint32_t> v(kSize / 4);
      uint32_t x = 2463534242u;
      for (auto& w : v) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; w = x; }
      int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
      write(fd, v.data(), kSize);
      close(fd);
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xor_words"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    int fd = open(path, O_RDONLY);

    struct Case { const char* name; int flags; size_t off; size_t len; };
    Case cases[] = {
        {"MAP_SHARED  whole", MAP_SHARED, 0, kSize},
        {"MAP_PRIVATE whole", MAP_PRIVATE, 0, kSize},
        {"MAP_PRIVATE window at 16 KiB x 1001, 64 MiB", MAP_PRIVATE, kPage * 1001, 64ull << 20},
        {"MAP_SHARED  window at 16 KiB x 1001, 64 MiB", MAP_SHARED, kPage * 1001, 64ull << 20},
        {"MAP_SHARED  window at 4 KiB offset (expect nil)", MAP_SHARED, 4096, 64ull << 20},
        {"MAP_SHARED  window, length not 16 KiB multiple (expect nil)", MAP_SHARED, 0, (64ull << 20) + 4096},
    };
    for (auto& c : cases) {
      uint8_t* base = (uint8_t*)mmap(nullptr, kSize, PROT_READ, c.flags, fd, 0);
      if (base == MAP_FAILED) { printf("%s: mmap failed\n", c.name); continue; }
      id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:base + c.off length:c.len
                                              options:MTLResourceStorageModeShared deallocator:nil];
      if (!b) {
        printf("%-60s buffer=nil\n", c.name);
      } else {
        uint32_t g = gpu_xor(dev, q, ps, b, c.len), h = cpu_xor(base + c.off, c.len);
        printf("%-60s buffer=ok gpu=%08x cpu=%08x %s\n", c.name, g, h, g == h ? "MATCH" : "MISMATCH");
      }
      b = nil;
      munmap(base, kSize);
    }
    close(fd);
    unlink(path);
  }
  return 0;
}
