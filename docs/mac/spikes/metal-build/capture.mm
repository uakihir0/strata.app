// Programmatic GPU capture without the variable in the environment (docs/mac/08-build-and-distribution.md, section 10.2).
//   ./capture /tmp/t.gputrace          sets MTL_CAPTURE_ENABLED=1 itself before the device exists: started=1
//   ./capture /tmp/t.gputrace nosetenv  does not: started=0 unless the shell exported MTL_CAPTURE_ENABLED=1
// The output must not exist: rm -rf /tmp/t.gputrace between runs.
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) { @autoreleasepool {
  if (argc < 2) { std::fprintf(stderr, "usage: capture OUT.gputrace [nosetenv]\n"); return 2; }
  const bool set = !(argc > 2 && std::strcmp(argv[2], "nosetenv") == 0);
  if (set) setenv("MTL_CAPTURE_ENABLED", "1", 1);   // must happen before the first Metal call
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  MTLCaptureManager* m = [MTLCaptureManager sharedCaptureManager];
  BOOL ok = [m supportsDestination:MTLCaptureDestinationGPUTraceDocument];
  id<MTLCommandQueue> q = [d newCommandQueue];
  MTLCaptureDescriptor* cd = [MTLCaptureDescriptor new];
  cd.captureObject = q;
  cd.destination = MTLCaptureDestinationGPUTraceDocument;
  cd.outputURL = [NSURL fileURLWithPath:@(argv[1])];
  NSError* e = nil;
  BOOL st = ok ? [m startCaptureWithDescriptor:cd error:&e] : NO;
  if (st) {
    id<MTLCommandBuffer> b = [q commandBuffer];
    [b commit];
    [b waitUntilCompleted];
    [m stopCapture];
  }
  std::printf("setenv=%d supports=%d started=%d %s\n", set, ok, st, e ? e.localizedDescription.UTF8String : "");
  return st ? 0 : 1;
}}
