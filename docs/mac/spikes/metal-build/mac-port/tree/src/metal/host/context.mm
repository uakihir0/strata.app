// src/metal/host/context.mm - see include/strata/metal/context.hpp.
#include "context_impl.h"

#include "strata/metal/pipeline_cache.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>

#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

namespace strata::metal {

static const KernelInfo kManifest[] = {
#include "kernel_manifest.inc"
};

const KernelInfo* kernel_manifest(std::size_t& count) {
    count = sizeof kManifest / sizeof kManifest[0];
    return kManifest;
}

std::string sha256_hex(const void* p, std::size_t n) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(p, static_cast<CC_LONG>(n), d);
    static const char hex[] = "0123456789abcdef";
    std::string s;
    for (unsigned char c : d) { s += hex[c >> 4]; s += hex[c & 15]; }
    return s;
}

static std::string executable_dir() {
    char raw[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) != 0) return ".";
    char real[PATH_MAX];
    std::string p = realpath(raw, real) != nullptr ? real : raw;
    const std::size_t slash = p.rfind('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

static std::string ns(NSError* e) { return e != nil ? e.localizedDescription.UTF8String : "unknown error"; }

bool build_pipelines(Context::Impl& im, const std::string& metallib_sha256, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string why;
    const std::string path = pipeline_archive_path(metallib_sha256, im.device.name.UTF8String, why);
    id<MTLBinaryArchive> ar = nil;
    NSError* e = nil;
    if (!path.empty()) {
        MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
        if (access(path.c_str(), R_OK) == 0) ad.url = [NSURL fileURLWithPath:@(path.c_str())];
        ar = [im.device newBinaryArchiveWithDescriptor:ad error:&e];
        if (ar == nil && ad.url != nil) {   // a damaged or foreign archive: start an empty one (section 10.1)
            std::fprintf(stderr, "strata: ignoring the pipeline cache %s: %s\n", path.c_str(), ns(e).c_str());
            ad.url = nil;
            ar = [im.device newBinaryArchiveWithDescriptor:ad error:&e];
        }
        why = ar != nil ? path : "cannot create an archive: " + ns(e);
    }
    im.stats.archive = why;

    std::size_t n = 0;
    const KernelInfo* k = kernel_manifest(n);
    std::vector<MTLComputePipelineDescriptor*> descs(n);
    std::vector<id<MTLComputePipelineState>> made(n);
    std::vector<std::size_t> misses;
    for (std::size_t i = 0; i < n; ++i) {
        id<MTLFunction> f = [im.library newFunctionWithName:@(k[i].name)];
        if (f == nil) { err = std::string("strata.metallib has no kernel ") + k[i].name; return false; }
        MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
        d.computeFunction = f;
        d.label = @(k[i].name);
        if (ar != nil) {
            d.binaryArchives = @[ ar ];
            made[i] = [im.device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                            reflection:nil error:nil];
        }
        descs[i] = d;
        if (made[i] != nil) ++im.stats.from_archive;
        else misses.push_back(i);
    }
    if (!misses.empty()) {
        std::fprintf(stderr, "strata: compiling GPU kernels for this Mac (once per engine update)\n");
        std::vector<NSError*> errors(n);
        id<MTLComputePipelineState> __strong* out = made.data();
        NSError* __strong* out_err = errors.data();
        dispatch_group_t g = dispatch_group_create();
        for (std::size_t i : misses) {
            dispatch_group_enter(g);
            [im.device newComputePipelineStateWithDescriptor:descs[i] options:MTLPipelineOptionNone
                completionHandler:^(id<MTLComputePipelineState> p, MTLComputePipelineReflection*, NSError* pe) {
                    out[i] = p;
                    out_err[i] = pe;
                    dispatch_group_leave(g);
                }];
        }
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        for (std::size_t i : misses) {
            if (made[i] == nil) { err = std::string("cannot build kernel ") + k[i].name + ": " + ns(errors[i]); return false; }
            ++im.stats.compiled;
            if (ar != nil && ![ar addComputePipelineFunctionsWithDescriptor:descs[i] error:&e])
                std::fprintf(stderr, "strata: pipeline cache: cannot add %s: %s\n", k[i].name, ns(e).c_str());
        }
        if (ar != nil) {   // write beside, then rename: a concurrent engine never reads half a file
            const std::string tmp = path + ".tmp." + std::to_string(getpid());
            if ([ar serializeToURL:[NSURL fileURLWithPath:@(tmp.c_str())] error:&e]) {
                if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
            } else {
                std::fprintf(stderr, "strata: pipeline cache: cannot write %s: %s\n", path.c_str(), ns(e).c_str());
            }
        }
    }
    if (!path.empty()) prune_pipeline_archives(path);
    for (std::size_t i = 0; i < n; ++i) {
        if (made[i].threadExecutionWidth != 32) {
            err = std::string("kernel ") + k[i].name + " has SIMD width " + std::to_string(made[i].threadExecutionWidth) +
                  ", Strata's kernels need 32";
            return false;
        }
        if (made[i].maxTotalThreadsPerThreadgroup < k[i].threads) {
            err = std::string("kernel ") + k[i].name + " allows " +
                  std::to_string(made[i].maxTotalThreadsPerThreadgroup) + " threads per threadgroup, it needs " +
                  std::to_string(k[i].threads);
            return false;
        }
        im.pipelines[k[i].name] = made[i];
    }
    im.stats.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

std::unique_ptr<Context> Context::open(const ContextOptions& o, std::string& err) {
    @autoreleasepool {
        // the capture manager reads MTL_CAPTURE_ENABLED when Metal starts, so it must be set first (section 10.2)
        if (!o.capture.empty()) setenv("MTL_CAPTURE_ENABLED", "1", 1);
        auto im = std::make_unique<Impl>();
        im->capture = o.capture;
        im->device = MTLCreateSystemDefaultDevice();
        if (im->device == nil) { err = "no Metal device"; return nullptr; }
        if (!im->device.hasUnifiedMemory) { err = "the GPU has no unified memory; Strata needs Apple silicon"; return nullptr; }
        if (![im->device supportsFamily:MTLGPUFamilyApple7] || ![im->device supportsFamily:MTLGPUFamilyMetal4]) {
            err = std::string("the GPU (") + im->device.name.UTF8String + ") is older than Apple7 / Metal 4; Strata needs M1 or newer";
            return nullptr;
        }
        im->queue = [im->device newCommandQueue];
        if (o.pipelines) {
            const std::string path = o.metallib.empty() ? executable_dir() + "/strata.metallib" : o.metallib;
            NSData* data = [NSData dataWithContentsOfFile:@(path.c_str())];
            if (data == nil) { err = "cannot read " + path + " (strata.metallib ships next to strata)"; return nullptr; }
            const std::string h = sha256_hex(data.bytes, data.length);
            if (!o.metallib_sha256.empty() && h != o.metallib_sha256) {
                err = "strata.metallib does not belong to this engine (" + path + " has SHA-256 " + h.substr(0, 16) +
                      "..., the engine was built with " + o.metallib_sha256.substr(0, 16) + "...); reinstall the engine";
                return nullptr;
            }
            dispatch_data_t dd = dispatch_data_create(data.bytes, data.length, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError* e = nil;
            im->library = [im->device newLibraryWithData:dd error:&e];
            if (im->library == nil) { err = "cannot load " + path + ": " + ns(e); return nullptr; }
            if (!build_pipelines(*im, h, err)) return nullptr;
        }
        return std::unique_ptr<Context>(new Context(std::move(im)));
    }
}

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Context::~Context() { end_capture(); }
std::string Context::device_name() const { return impl_->device.name.UTF8String; }
const PipelineStats& Context::pipeline_stats() const { return impl_->stats; }
void* Context::device() const { return (__bridge void*) impl_->device; }
void* Context::queue() const { return (__bridge void*) impl_->queue; }
void* Context::pipeline(const char* name) const {
    auto it = impl_->pipelines.find(name);
    return it == impl_->pipelines.end() ? nullptr : (__bridge void*) it->second;
}

bool Context::run_selftest(std::string& err) {
    @autoreleasepool {
        const NSUInteger n = 1024;
        id<MTLBuffer> b = [impl_->device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        float* x = static_cast<float*>(b.contents);
        for (NSUInteger i = 0; i < n; ++i) x[i] = static_cast<float>(i) * 0.001f;
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        cb.label = @"selftest";
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (const char* k : {"strata_selftest_add", "strata_selftest_exp"}) {
            [enc setComputePipelineState:impl_->pipelines.at(k)];
            [enc setBuffer:b offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { err = "selftest command buffer failed"; return false; }
        for (NSUInteger i = 0; i < n; ++i) {
            const float want = std::exp(static_cast<float>(i) * 0.001f * 2.0f + 1.0f);
            if (std::fabs(x[i] - want) > 1e-4f * want) {
                err = "selftest: element " + std::to_string(i) + " is " + std::to_string(x[i]) + ", want " + std::to_string(want);
                return false;
            }
        }
        return true;
    }
}

}  // namespace strata::metal
