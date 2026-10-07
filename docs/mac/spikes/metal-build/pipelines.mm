// Library load, pipeline build and MTLBinaryArchive reuse (docs/mac/08-build-and-distribution.md, section 10.1).
//   ./pipelines strata.metallib /tmp/archive.bin     first run: every pipeline compiles and the archive is written
//   ./pipelines strata.metallib /tmp/archive.bin     second run: every pipeline comes from the archive
//   head -c 32768 /dev/urandom > /tmp/bad.bin; ./pipelines strata.metallib /tmp/bad.bin   an invalid archive
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>

static double ms(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

int main(int argc, char** argv) { @autoreleasepool {
  if (argc != 3) { std::fprintf(stderr, "usage: pipelines LIB.metallib ARCHIVE\n"); return 2; }
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  NSError* e = nil;
  auto t = std::chrono::steady_clock::now();
  id<MTLLibrary> lib = [d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
  if (!lib) { std::printf("library failed: %s\n", e.localizedDescription.UTF8String); return 1; }
  std::printf("library %.2f ms, functions:", ms(t));
  for (NSString* n in lib.functionNames) std::printf(" %s", n.UTF8String);
  std::printf("\n");

  NSURL* au = [NSURL fileURLWithPath:@(argv[2])];
  MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
  if ([[NSFileManager defaultManager] fileExistsAtPath:au.path]) ad.url = au;
  id<MTLBinaryArchive> ar = [d newBinaryArchiveWithDescriptor:ad error:&e];
  if (!ar) {   // the engine's rule: log, start an empty archive, never fail
    std::printf("archive open failed: %s\n", e.localizedDescription.UTF8String);
    ad.url = nil;
    ar = [d newBinaryArchiveWithDescriptor:ad error:&e];
  }
  for (NSString* n in lib.functionNames) {
    MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = [lib newFunctionWithName:n];
    pd.binaryArchives = @[ ar ];
    t = std::chrono::steady_clock::now();
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:pd
                                                                     options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                  reflection:nil error:&e];
    if (p) { std::printf("%s: from archive %.2f ms\n", n.UTF8String, ms(t)); continue; }
    t = std::chrono::steady_clock::now();
    p = [d newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&e];
    std::printf("%s: compiled %.2f ms, threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu\n", n.UTF8String,
                ms(t), (unsigned long)p.threadExecutionWidth, (unsigned long)p.maxTotalThreadsPerThreadgroup);
    if (![ar addComputePipelineFunctionsWithDescriptor:pd error:&e])
      std::printf("add failed: %s\n", e.localizedDescription.UTF8String);
  }
  if (![ar serializeToURL:au error:&e]) std::printf("serialize failed: %s\n", e.localizedDescription.UTF8String);
  std::printf("macOS %s\n", [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String);
}}
