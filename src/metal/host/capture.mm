// src/metal/host/capture.mm - programmatic GPU capture of the command queue (docs/mac/08 section 10.2).
#include "context_impl.h"

#import <Foundation/Foundation.h>

namespace strata::metal {

bool Context::begin_capture(std::string& err) {
    if (impl_->capture.empty() || impl_->capturing) return true;
    MTLCaptureManager* m = [MTLCaptureManager sharedCaptureManager];
    if (![m supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        err = "GPU capture is not available (MTL_CAPTURE_ENABLED must be set before Metal starts)";
        return false;
    }
    MTLCaptureDescriptor* d = [MTLCaptureDescriptor new];
    d.captureObject = impl_->queue;
    d.destination = MTLCaptureDestinationGPUTraceDocument;
    d.outputURL = [NSURL fileURLWithPath:@(impl_->capture.c_str())];
    NSError* e = nil;
    if (![m startCaptureWithDescriptor:d error:&e]) {
        err = "cannot start the GPU capture: " + std::string(e != nil ? e.localizedDescription.UTF8String : "unknown");
        return false;
    }
    impl_->capturing = true;
    return true;
}

void Context::end_capture() {
    if (!impl_->capturing) return;
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
    impl_->capturing = false;
}

}  // namespace strata::metal
