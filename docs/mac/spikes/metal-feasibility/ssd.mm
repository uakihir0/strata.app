// SSD / page-cache behaviour for a Metal port: cold random reads, and whether a no-copy MTLBuffer over an mmap wires it.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <cstdio>
#include <cstring>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void make_cold_file(const char* path, size_t bytes) {
  int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
  fcntl(fd, F_NOCACHE, 1);
  size_t chunk = 64ul << 20;
  std::vector<uint8_t> buf(chunk);
  std::mt19937_64 rng(7);
  for (auto& b : buf) b = (uint8_t)rng();
  double t0 = now();
  for (size_t off = 0; off < bytes; off += chunk) { buf[0] ^= 1; pwrite(fd, buf.data(), chunk, off); }
  fsync(fd);
  printf("wrote %s (%.0f GiB) uncached at %.2f GB/s\n", path, bytes / 1073741824.0, bytes / (now() - t0) / 1e9);
  close(fd);
}

static double resident_frac(void* p, size_t len) {
  size_t pg = getpagesize(), n = len / pg;
  std::vector<char> v(n);
  mincore(p, len, v.data());
  size_t r = 0; for (char c : v) r += (c & 1);
  return (double)r / n;
}

static void rand_reads(const char* path, size_t file_bytes, size_t blk, int threads, int per_thread) {
  std::atomic<size_t> total{0};
  std::vector<std::thread> th;
  std::vector<double> lat(threads * per_thread);
  double t0 = now();
  for (int t = 0; t < threads; t++) th.emplace_back([&, t] {
    int fd = open(path, O_RDONLY); fcntl(fd, F_NOCACHE, 1);
    void* b = nullptr; posix_memalign(&b, 16384, blk);
    std::mt19937_64 rng(1000 + t);
    for (int i = 0; i < per_thread; i++) {
      size_t off = (rng() % ((file_bytes - blk) / 16384)) * 16384;
      double a = now();
      ssize_t r = pread(fd, b, blk, off);
      lat[t * per_thread + i] = now() - a;
      if (r > 0) total += r;
    }
    free(b); close(fd); });
  for (auto& x : th) x.join();
  double s = now() - t0;
  std::sort(lat.begin(), lat.end());
  printf("cold pread %7zu B x %4d, %2d threads: %6.2f GB/s, %8.0f reads/s, p50 %.3f ms, p99 %.3f ms\n", blk,
         threads * per_thread, threads, total / s / 1e9, threads * per_thread / s, lat[lat.size() / 2] * 1e3,
         lat[lat.size() * 99 / 100] * 1e3);
}

int main() {
  @autoreleasepool {
    const size_t GB8 = 8ul << 30;
    const char* A = "/tmp/strata-spike/coldA.bin";
    const char* B = "/tmp/strata-spike/coldB.bin";
    make_cold_file(A, GB8);
    make_cold_file(B, GB8);

    // Metal: no-copy buffer over a cold mmap; touch 1 MiB; how much becomes resident?
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    int fd = open(B, O_RDONLY);
    void* p = mmap(nullptr, GB8, PROT_READ, MAP_SHARED, fd, 0);
    printf("coldB resident before: %.3f\n", resident_frac(p, GB8));
    double t0 = now();
    id<MTLBuffer> fb = [dev newBufferWithBytesNoCopy:p length:GB8 options:MTLResourceStorageModeShared deallocator:nil];
    printf("newBufferWithBytesNoCopy(8 GiB cold mmap): %s in %.3f s; resident after create: %.3f\n", fb ? "OK" : "FAIL",
           now() - t0, resident_frac(p, GB8));
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void touch(device const uint4* s [[buffer(0)]], device uint* o [[buffer(1)]], constant ulong& base [[buffer(2)]], uint g [[thread_position_in_grid]]) {"
      " uint4 v = s[base + g]; if ((v.x ^ v.y) == 0x1234567u) o[0] = 1; }" options:nil error:&err];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    // touch 1 MiB in the middle
    uint64_t base = (GB8 / 2) / 16;
    t0 = now();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps]; [e setBuffer:fb offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&base length:8 atIndex:2];
    [e dispatchThreads:MTLSizeMake(65536,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    printf("GPU touch of 1 MiB in cold 8 GiB no-copy buffer: %.3f s wall, status %ld, error %s; resident after: %.3f\n",
           now() - t0, (long)cb.status, cb.error ? cb.error.localizedDescription.UTF8String : "none", resident_frac(p, GB8));

    // SSD random reads (cold, F_NOCACHE) on file A
    for (int th : {1, 4, 8, 16}) rand_reads(A, GB8, 1376256, th, 64);
    for (int th : {1, 8, 32}) rand_reads(A, GB8, 16384, th, 512);
  }
  return 0;
}
