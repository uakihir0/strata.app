// Measured matmul throughput on this GPU with Metal Performance Primitives (tensor ops), half x half -> float.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
static NSString* src = @R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mm(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]],
               constant int3& mnk [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, M));
  auto tB = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(N, K));
  auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tgid.y * 64);
  auto mB = tB.slice(tgid.x * 32, 0);
  auto mC = tC.slice(tgid.x * 32, tgid.y * 64);
  op.run(mA, mB, mC);
}
kernel void mm64(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]],
               constant int3& mnk [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, M));
  auto tB = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(N, K));
  auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto a = tA.slice(0, tgid.y * 64); auto b = tB.slice(tgid.x * 64, 0); auto c = tC.slice(tgid.x * 64, tgid.y * 64); op.run(a, b, c);
}
kernel void mmi8(device char* A [[buffer(0)]], device char* B [[buffer(1)]], device int* C [[buffer(2)]],
               constant int3& mnk [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = mnk.x, N = mnk.y, K = mnk.z;
  auto tA = tensor<device int8_t,  dextents<int32_t, 2>, tensor_inline>((device int8_t*)A, dextents<int32_t, 2>(K, M));
  auto tB = tensor<device int8_t,  dextents<int32_t, 2>, tensor_inline>((device int8_t*)B, dextents<int32_t, 2>(N, K));
  auto tC = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>((device int32_t*)C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto a = tA.slice(0, tgid.y * 64); auto b = tB.slice(tgid.x * 32, 0); auto c = tC.slice(tgid.x * 32, tgid.y * 64); op.run(a, b, c);
}
)";
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    const char* fname = getenv("K") ? getenv("K") : "mm";
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:fname]] error:&err];
    int tn = strcmp(fname, "mm64") == 0 ? 64 : 32; bool i8 = strcmp(fname, "mmi8") == 0;
    if (!ps) { printf("pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int s : {2048, 4096, 8192}) {
      int M = s, N = s, K = 4096;
      id<MTLBuffer> A = [dev newBufferWithLength:(size_t)M*K*2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> B = [dev newBufferWithLength:(size_t)K*N*2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> C = [dev newBufferWithLength:(size_t)M*N*4 options:MTLResourceStorageModeShared];
      __fp16* a = (__fp16*)A.contents; __fp16* b = (__fp16*)B.contents;
      for (size_t i = 0; i < (size_t)M*K; i++) a[i] = (__fp16)((i % 7) * 0.01f);
      for (size_t i = 0; i < (size_t)K*N; i++) b[i] = (__fp16)((i % 5) * 0.01f);
      int mnk[4] = {M, N, K, 0};
      double best = 1e9;
      for (int r = 0; r < 6; r++) {
        memset(C.contents, 0, (size_t)M*N*4);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
        [e setBytes:mnk length:16 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(N/tn, M/64, 1) threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth*4,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("err %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
      }
      // check one element
      float* c = (float*)C.contents; double ref = 0;
      for (int k = 0; k < K; k++) ref += (double)a[(size_t)0*K + k] * (double)b[(size_t)k*N + 0];
      printf("%s: M=N=%d K=%d: %.2f TOPS (C[0,0]=%.4f ref %.4f)\n", fname, s, K,
             2.0 * M * N * K / best / 1e12, c[0], ref);
    }
  }
  return 0;
}
