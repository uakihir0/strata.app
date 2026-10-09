// src/metal/host/context.mm - see include/strata/metal/context.hpp.
#include "context_impl.h"

#include "strata/metal/pipeline_cache.hpp"
#include "../engine/metal_fence.hpp"
#include "../kernels/common/types.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <IOKit/IOKitLib.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>

#include <chrono>
#include <cmath>
#include <climits>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <algorithm>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::metal {

std::mutex& command_buffer_submission_mutex() {
    static std::mutex mutex;
    return mutex;
}

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

static std::uint64_t io_gpu_core_count(id<MTLDevice> device) {
    io_iterator_t services = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &services) !=
        KERN_SUCCESS) {
        return 0;
    }
    std::uint64_t fallback = 0;
    std::uint64_t matched = 0;
    io_service_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(services)) != IO_OBJECT_NULL) {
        std::uint64_t registry_id = 0;
        const bool is_device =
            IORegistryEntryGetRegistryEntryID(service, &registry_id) == KERN_SUCCESS && registry_id == device.registryID;
        CFTypeRef property =
            IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (property != nullptr && CFGetTypeID(property) == CFNumberGetTypeID()) {
            std::int32_t count = 0;
            if (CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberSInt32Type, &count) && count > 0) {
                if (is_device) matched = static_cast<std::uint64_t>(count);
                else if (fallback == 0) fallback = static_cast<std::uint64_t>(count);
            }
        }
        if (property != nullptr) CFRelease(property);
        IOObjectRelease(service);
        if (matched != 0) break;
    }
    IOObjectRelease(services);
    return matched != 0 ? matched : fallback;
}

static std::uint64_t physical_memory_bytes() {
    std::uint64_t bytes = 0;
    std::size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 && size == sizeof(bytes) ? bytes : 0;
}

static std::string macos_version() {
    char version[64] = {};
    std::size_t size = sizeof(version);
    if (sysctlbyname("kern.osproductversion", version, &size, nullptr, 0) != 0 || size == 0) return {};
    version[sizeof(version) - 1] = '\0';
    return version;
}

static bool metal_validation_enabled() {
    const char* api_validation = std::getenv("MTL_DEBUG_LAYER");
    const char* shader_validation = std::getenv("MTL_SHADER_VALIDATION");
    return (api_validation != nullptr && std::strcmp(api_validation, "1") == 0) ||
           (shader_validation != nullptr && std::strcmp(shader_validation, "1") == 0);
}

static bool metal_error_options_enabled() {
    const char* configured = std::getenv("STRATA_METAL_ERRORS");
    if (configured != nullptr) return std::strcmp(configured, "1") == 0;
#if defined(NDEBUG)
    return false;
#else
    return true;
#endif
}

static void add_device_families(id<MTLDevice> device, DeviceFacts& facts) {
    const std::pair<const char*, MTLGPUFamily> supported[] = {
        {"apple7", MTLGPUFamilyApple7},   {"apple8", MTLGPUFamilyApple8}, {"apple9", MTLGPUFamilyApple9},
        {"apple10", MTLGPUFamilyApple10}, {"metal3", MTLGPUFamilyMetal3}, {"metal4", MTLGPUFamilyMetal4},
    };
    for (const auto& entry : supported) {
        if ([device supportsFamily:entry.second]) facts.families.emplace_back(entry.first);
    }
}

bool build_pipelines(Context::Impl& im, const std::string& metallib_sha256, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string why;
    const bool validation = metal_validation_enabled();
    const std::string path = validation ? std::string() :
        pipeline_archive_path(metallib_sha256, im.device.name.UTF8String, why);
    if (validation) why = "disabled while Metal validation is enabled";
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
    std::vector<MTLComputePipelineReflection*> reflections(n);
    std::vector<bool> enabled(n, true);
    std::vector<std::size_t> misses;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t name_length = std::strlen(k[i].name);
        const bool requires_apple10 =
            name_length >= 4u &&
            std::strcmp(k[i].name + name_length - 4u, "_mpp") == 0;
        if (requires_apple10 &&
            ![im.device supportsFamily:MTLGPUFamilyApple10]) {
            enabled[i] = false;
            continue;
        }
        id<MTLFunction> f = [im.library newFunctionWithName:@(k[i].name)];
        if (f == nil) { err = std::string("strata.metallib has no kernel ") + k[i].name; return false; }
        MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
        d.computeFunction = f;
        d.label = @(k[i].name);
        if (ar != nil) {
            d.binaryArchives = @[ ar ];
            MTLComputePipelineReflection* __autoreleasing reflection = nil;
            made[i] = [im.device
                newComputePipelineStateWithDescriptor:d
                                               options:MTLPipelineOptionFailOnBinaryArchiveMiss |
                                                       MTLPipelineOptionBindingInfo
                                            reflection:&reflection
                                                 error:nil];
            reflections[i] = reflection;
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
        MTLComputePipelineReflection* __strong* reflection_out = reflections.data();
        dispatch_group_t g = dispatch_group_create();
        for (std::size_t i : misses) {
            dispatch_group_enter(g);
            [im.device
                newComputePipelineStateWithDescriptor:descs[i]
                                               options:MTLPipelineOptionBindingInfo
                                     completionHandler:^(id<MTLComputePipelineState> p,
                                                         MTLComputePipelineReflection* reflection,
                                                         NSError* pe) {
                out[i] = p;
                out_err[i] = pe;
                reflection_out[i] = reflection;
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
        if (!enabled[i]) continue;
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
        if (reflections[i] == nil) {
            err = std::string("cannot reflect resource bindings for kernel ") + k[i].name;
            return false;
        }
        KernelBindingSchema schema;
        for (id<MTLBinding> binding in reflections[i].bindings) {
            if (!binding.isUsed) continue;
            switch (binding.type) {
                case MTLBindingTypeBuffer:
                    schema.buffers.push_back(static_cast<std::uint32_t>(binding.index));
                    break;
                case MTLBindingTypeThreadgroupMemory:
                    schema.threadgroup_memory.push_back(
                        static_cast<std::uint32_t>(binding.index));
                    break;
                default:
                    schema.supported = false;
                    break;
            }
        }
        std::sort(schema.buffers.begin(), schema.buffers.end());
        std::sort(schema.threadgroup_memory.begin(), schema.threadgroup_memory.end());
        im.kernel_bindings[k[i].name] = std::move(schema);
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
        im->max_buffer_length =
            static_cast<std::uint64_t>(im->device.maxBufferLength);
        if (o.max_buffer_length_cap != 0 &&
            o.max_buffer_length_cap < im->max_buffer_length) {
            im->max_buffer_length = o.max_buffer_length_cap;
        }
        im->facts.gpu_core_count = io_gpu_core_count(im->device);
        im->facts.working_set_bytes = im->device.recommendedMaxWorkingSetSize;
        im->facts.max_buffer_length = im->max_buffer_length;
        im->facts.max_threadgroup_memory_length =
            im->device.maxThreadgroupMemoryLength;
        im->facts.physical_memory_bytes = physical_memory_bytes();
        im->facts.macos_version = macos_version();
        add_device_families(im->device, im->facts);
        im->queue = [im->device newCommandQueue];
        if (im->queue == nil) { err = "cannot create a Metal command queue"; return nullptr; }
        im->hazard_fence = [im->device newFence];
        if (im->hazard_fence == nil) {
            err = "cannot create the Metal queue hazard fence";
            return nullptr;
        }
        im->hazard_fence.label = @"strata queue hazard fence";
        id<MTLCommandBuffer> fence_seed = [im->queue commandBuffer];
        if (fence_seed == nil) {
            err = "cannot create a command buffer to initialize the Metal queue hazard fence";
            return nullptr;
        }
        fence_seed.label = @"strata queue hazard fence initialization";
        id<MTLBlitCommandEncoder> fence_seed_encoder =
            [fence_seed blitCommandEncoder];
        if (fence_seed_encoder == nil) {
            err = "cannot create a blit encoder to initialize the Metal queue hazard fence";
            return nullptr;
        }
        [fence_seed_encoder updateFence:im->hazard_fence];
        [fence_seed_encoder endEncoding];
        [fence_seed commit];
        [fence_seed waitUntilCompleted];
        if (fence_seed.status != MTLCommandBufferStatusCompleted) {
            err = "cannot initialize the Metal queue hazard fence";
            return nullptr;
        }
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
        auto buffers = std::unique_ptr<Buffers>(
            new Buffers((__bridge void*)im->device, (__bridge void*)im->queue,
                        im->max_buffer_length, err));
        if (!err.empty()) return nullptr;
        return std::unique_ptr<Context>(new Context(std::move(im), std::move(buffers)));
    }
}

Context::Context(std::unique_ptr<Impl> impl, std::unique_ptr<Buffers> buffers)
    : impl_(std::move(impl)), buffers_(std::move(buffers)) {}
Context::~Context() {
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    bool drained = impl_ == nullptr;
    if (impl_ && impl_->queue != nil) {
        // Queue ordering makes this fence complete after every earlier committed command buffer. Keep
        // Context-owned buffers and the residency set alive until that fence has completed.
        id<MTLCommandBuffer> drain = [impl_->queue commandBuffer];
        if (drain != nil) {
            drain.label = @"strata context teardown";
            [drain commit];
            [drain waitUntilCompleted];
            drained = true;
        } else {
            std::fprintf(stderr, "strata: cannot drain Metal command queue before Context teardown\n");
        }
    } else if (impl_) {
        std::fprintf(stderr, "strata: Metal command queue is unavailable during Context teardown\n");
    }
    end_capture();
    if (!drained) {
        std::fprintf(stderr,
                     "strata: cannot safely destroy Metal Context because queue drain failed; terminating process\n");
        std::fflush(stderr);
        // Some no-copy buffers refer to mappings owned outside Context. A process-wide quarantine of Metal
        // objects cannot keep those mappings alive if a caller unmaps them after this destructor returns.
        // Fail closed before returning control to code that could release GPU-visible memory.
        _exit(1);
    }
}
std::string Context::device_name() const { return impl_->device.name.UTF8String; }
const DeviceFacts& Context::device_facts() const { return impl_->facts; }
const PipelineStats& Context::pipeline_stats() const { return impl_->stats; }
void* Context::device() const { return (__bridge void*) impl_->device; }
void* Context::queue() const { return (__bridge void*) impl_->queue; }
void* Context::hazard_fence() const noexcept {
    return (__bridge void*)impl_->hazard_fence;
}

struct CommandBuffer::SubmissionState {
    void* queue_handle = nullptr;
    id<MTLCommandBuffer> native_command_buffer = nil;
    std::uint64_t creation_order = 0u;
    bool enqueued = false;
    bool committed = false;
    bool discarded = false;
};

namespace {

std::atomic<std::uint64_t> g_command_buffer_creation_order{1u};

}  // namespace

CommandBuffer::CommandBuffer(void* retained_handle, void* queue_handle)
    : handle_(retained_handle), queue_handle_(queue_handle),
      submission_state_(std::make_shared<SubmissionState>()) {
    submission_state_->queue_handle = queue_handle;
    submission_state_->native_command_buffer =
        (__bridge id<MTLCommandBuffer>)retained_handle;
    submission_state_->creation_order =
        g_command_buffer_creation_order.fetch_add(1u, std::memory_order_relaxed);
}

bool CommandBuffer::belongs_to(const Context& context) const {
    const void* const context_queue = context.queue();
    return queue_handle_ != nullptr && queue_handle_ == context_queue;
}
bool CommandBuffer::depends_on(CommandBuffer& producer, std::string& err) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (&producer == this) {
        err = "a Metal command buffer cannot depend on itself";
        return false;
    }
    if (handle_ == nullptr || discarded_ || committed_ || enqueued_) {
        err = "cannot add a dependency to a submitted or discarded command buffer";
        return false;
    }
    if (producer.queue_handle_ == nullptr ||
        producer.queue_handle_ != queue_handle_) {
        err = "Metal command-buffer dependencies must use the same queue";
        return false;
    }
    if (!submission_state_ || !producer.submission_state_ ||
        producer.submission_state_->creation_order >=
            submission_state_->creation_order) {
        err = "Metal command-buffer dependency must refer to an earlier buffer";
        return false;
    }
    if (producer.discarded_ || producer.submission_state_->discarded) {
        err = "cannot depend on a discarded Metal command buffer";
        return false;
    }
    bool already_registered = false;
    for (const auto& dependency : dependencies_) {
        if (dependency == producer.submission_state_) {
            already_registered = true;
            break;
        }
    }
    if (!already_registered) dependencies_.push_back(producer.submission_state_);
    err.clear();
    return true;
}
bool CommandBuffer::wait_until_completed(std::string& err) const {
    id<MTLCommandBuffer> command_buffer = nil;
    {
        std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
        std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
        if (handle_ == nullptr || !committed_) {
            err = "cannot wait for an uncommitted Metal command buffer";
            return false;
        }
        command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
    }
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted ||
        command_buffer.error != nil) {
        err = "Metal command buffer did not complete successfully";
        if (command_buffer.error != nil) err += ": " + ns(command_buffer.error);
        return false;
    }
    err.clear();
    return true;
}
std::unique_lock<std::recursive_mutex> CommandBuffer::lock_for_encoding() const {
    return std::unique_lock<std::recursive_mutex>(encoding_mutex_);
}

struct CommandBuffer::CompletionState {
    std::mutex mutex;
    std::vector<std::function<void(bool)>> handlers;
    bool delivered = false;

    void complete(bool success) noexcept {
        std::vector<std::function<void(bool)>> ready;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (delivered) return;
            delivered = true;
            ready.swap(handlers);
        }
        for (auto& handler : ready) {
            if (!handler) continue;
            try {
                handler(success);
            } catch (...) {
                std::fprintf(stderr, "strata: Metal command-buffer completion handler threw an exception\n");
                std::fflush(stderr);
            }
        }
    }
};

CommandBuffer::~CommandBuffer() {
    discard();
    if (handle_ != nullptr) {
        id<MTLCommandBuffer> command_buffer = (__bridge_transfer id<MTLCommandBuffer>)handle_;
        (void)command_buffer;
    }
}
bool CommandBuffer::add_discard_handler(std::function<void()> handler) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    if (!can_encode()) return false;
    discard_handlers_.emplace_back(std::move(handler));
    return true;
}
bool CommandBuffer::add_completion_handler(std::function<void(bool)> handler,
                                           std::string& err) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (!handler || !can_encode()) {
        err = "cannot register a completion handler on a committed, enqueued, or discarded command buffer";
        return false;
    }
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
    if (!completion_state_) {
        completion_state_ = std::make_shared<CompletionState>();
        const std::shared_ptr<CompletionState> state = completion_state_;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            state->complete(completed.status == MTLCommandBufferStatusCompleted);
        }];
    }
    {
        std::lock_guard<std::mutex> lock(completion_state_->mutex);
        if (completion_state_->delivered) {
            err = "Metal command buffer completed before its completion handler was registered";
            return false;
        }
        completion_state_->handlers.emplace_back(std::move(handler));
    }
    err.clear();
    return true;
}
bool CommandBuffer::can_encode() const noexcept {
    if (handle_ == nullptr || committed_ || discarded_ || enqueued_) return false;
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
    return command_buffer.status == MTLCommandBufferStatusNotEnqueued;
}
bool CommandBuffer::enqueue(std::string& err) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (handle_ == nullptr || discarded_ || committed_) {
        err = "cannot enqueue a committed or discarded Metal command buffer";
        return false;
    }
    if (enqueued_) {
        err.clear();
        return true;
    }
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
    if (command_buffer.status == MTLCommandBufferStatusEnqueued) {
        err = "Metal command buffer was enqueued outside its Strata wrapper";
        return false;
    }
    if (command_buffer.status != MTLCommandBufferStatusNotEnqueued) {
        committed_ = true;
        submission_state_->committed = true;
        err = "Metal command buffer was already committed";
        return false;
    }
    if (!validate_submission(err)) return false;
    [command_buffer enqueue];
    enqueued_ = true;
    submission_state_->enqueued = true;
    notify_submission(false);
    err.clear();
    return true;
}
bool CommandBuffer::commit(std::string& err) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (handle_ == nullptr || discarded_) {
        err = "cannot commit a discarded Metal command buffer";
        return false;
    }
    if (committed_) {
        err = "Metal command buffer was already committed";
        return false;
    }
    id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
    const MTLCommandBufferStatus status = command_buffer.status;
    if (status != MTLCommandBufferStatusNotEnqueued &&
        status != MTLCommandBufferStatusEnqueued) {
        committed_ = true;
        submission_state_->committed = true;
        err = "Metal command buffer was committed outside its Strata wrapper";
        return false;
    }
    if ((status == MTLCommandBufferStatusEnqueued) != enqueued_) {
        err = "Metal command buffer submission state does not match its Strata wrapper";
        return false;
    }
    if (!validate_submission(err)) return false;
    [command_buffer commit];
    committed_ = true;
    submission_state_->committed = true;
    notify_submission(true);
    submission_handlers_.clear();
    discard_handlers_.clear();
    err.clear();
    return true;
}
bool CommandBuffer::encode_unsubmitted(
    const std::function<bool(void*, std::string&)>& encode,
    std::string& err, bool discard_on_failure) {
    if (!encode) {
        err = "Metal command-buffer encoder callback is empty";
        return false;
    }

    std::unique_lock<std::recursive_mutex> encoding_lock(encoding_mutex_);
    id<MTLCommandBuffer> command_buffer = nil;
    {
        std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
        if (handle_ == nullptr) {
            err = "Metal command buffer is unavailable";
            return false;
        }
        if (discarded_) {
            err = "cannot encode a discarded Metal command buffer";
            return false;
        }
        if (committed_) {
            err = "Metal command-buffer work must be encoded before command-buffer submission";
            return false;
        }
        command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
        if (command_buffer.status != MTLCommandBufferStatusNotEnqueued || enqueued_) {
            err = "Metal command-buffer work must be encoded before command-buffer submission";
            return false;
        }
    }

    bool encoded = false;
    try {
        encoded = encode(handle_, err);
    } catch (const std::exception& exception) {
        err = std::string("Metal command-buffer encoding threw: ") + exception.what();
    } catch (...) {
        err = "Metal command-buffer encoding threw an unknown exception";
    }
    if (encoded) {
        std::lock_guard<std::mutex> submission_lock(command_buffer_submission_mutex());
        if (command_buffer.status == MTLCommandBufferStatusNotEnqueued) {
            err.clear();
            return true;
        }
        err = "Metal command buffer was submitted outside its Strata wrapper while encoding";
    } else if (err.empty()) {
        err = "Metal command-buffer encoding failed";
    }

    if (!discard_on_failure) return false;

    std::vector<std::function<void()>> discarded_handlers;
    std::shared_ptr<CompletionState> completion;
    std::unique_lock<std::mutex> submission_lock(command_buffer_submission_mutex());
    // Wrapper submissions cannot race the callback because they use the same
    // per-command-buffer mutex. Preserve the native-handle escape hatch rules.
    if (command_buffer.status != MTLCommandBufferStatusNotEnqueued) {
        return false;
    }
    discarded_ = true;
    submission_state_->discarded = true;
    discarded_handlers.swap(discard_handlers_);
    completion = completion_state_;
    submission_lock.unlock();
    for (auto& handler : discarded_handlers) {
        if (!handler) continue;
        try {
            handler();
        } catch (...) {
            std::fprintf(stderr, "strata: Metal command-buffer discard handler threw an exception\n");
            std::fflush(stderr);
        }
    }
    if (completion) completion->complete(false);
    return false;
}
void CommandBuffer::discard() noexcept {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    std::unique_lock<std::mutex> submission_lock(command_buffer_submission_mutex());
    if (committed_ || discarded_) return;
    if (handle_ != nullptr) {
        id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)handle_;
        if (command_buffer.status == MTLCommandBufferStatusEnqueued) {
            // Metal has no cancellation API for an enqueued command buffer. Submit it so it cannot
            // block the queue forever and leave registered event signals pending forever.
            std::string validation_error;
            try {
                if (!validate_submission(validation_error)) {
                    std::fprintf(stderr, "strata: cannot safely submit enqueued Metal command buffer: %s\n",
                                 validation_error.c_str());
                    std::fflush(stderr);
                    _exit(1);
                }
            } catch (...) {
                std::fprintf(stderr, "strata: cannot validate enqueued Metal command buffer during destruction\n");
                std::fflush(stderr);
                _exit(1);
            }
            committed_ = true;
            submission_state_->committed = true;
            [command_buffer commit];
            notify_submission(true);
            submission_handlers_.clear();
            discard_handlers_.clear();
            return;
        }
        if (command_buffer.status != MTLCommandBufferStatusNotEnqueued) {
            committed_ = true;
            submission_state_->committed = true;
            const std::shared_ptr<CompletionState> completion = completion_state_;
            submission_lock.unlock();
            if (completion) completion->complete(false);
            return;
        }
    }
    discarded_ = true;
    submission_state_->discarded = true;
    std::vector<std::function<void()>> discarded_handlers;
    discarded_handlers.swap(discard_handlers_);
    const std::shared_ptr<CompletionState> completion = completion_state_;
    submission_lock.unlock();
    for (auto& handler : discarded_handlers) {
        if (!handler) continue;
        try {
            handler();
        } catch (...) {
            std::fprintf(stderr, "strata: Metal command-buffer discard handler threw an exception\n");
            std::fflush(stderr);
        }
    }
    if (completion) completion->complete(false);
}
bool CommandBuffer::add_submission_handler(SubmissionHandler handler, std::string& err) {
    std::lock_guard<std::recursive_mutex> encoding_lock(encoding_mutex_);
    if (!can_encode() || !handler.validate) {
        err = "cannot register submission checks on a committed, enqueued, or discarded command buffer";
        return false;
    }
    submission_handlers_.emplace_back(std::move(handler));
    err.clear();
    return true;
}
bool CommandBuffer::validate_submission(std::string& err) const {
    for (const auto& dependency : dependencies_) {
        if (dependency->discarded) {
            err = "cannot submit Metal command buffer because a dependency was discarded";
            return false;
        }
        if (!dependency->committed) {
            err = "Metal command-buffer dependency must be committed first";
            return false;
        }
        id<MTLCommandBuffer> producer = dependency->native_command_buffer;
        if (producer == nil ||
            producer.status != MTLCommandBufferStatusCompleted ||
            producer.error != nil) {
            err = "Metal command-buffer dependency must complete successfully first";
            if (producer != nil && producer.error != nil) {
                err += ": " + ns(producer.error);
            }
            return false;
        }
    }
    for (const auto& handler : submission_handlers_) {
        if (handler.validate && !handler.validate(err)) return false;
    }
    err.clear();
    return true;
}
void CommandBuffer::notify_submission(bool committed) {
    for (const auto& handler : submission_handlers_) {
        if (handler.notify) handler.notify(committed);
    }
}
std::unique_ptr<CommandBuffer> Context::make_command_buffer(const std::string& label, std::string& err) const {
    id<MTLCommandBuffer> command_buffer = nil;
    @autoreleasepool {
        MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
        if (metal_error_options_enabled()) {
            descriptor.errorOptions =
                MTLCommandBufferErrorOptionEncoderExecutionStatus;
        }
        command_buffer =
            [impl_->queue commandBufferWithDescriptor:descriptor];
        if (command_buffer == nil) {
            err = "cannot create Metal command buffer \"" + label + "\"";
            return {};
        }
        command_buffer.label = @(label.c_str());
        const std::string label_copy = label;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            @autoreleasepool {
                for (id<MTLFunctionLog> log in completed.logs) {
                    NSString* detail = [log description];
                    const char* encoder = log.encoderLabel.UTF8String;
                    if (encoder != nullptr && encoder[0] != '\0') {
                        detail = [NSString
                            stringWithFormat:@"encoder \"%s\": %@",
                                             encoder,
                                             detail ?: @"unknown error"];
                    }
                    std::fprintf(
                        stderr, "strata: GPU error in \"%s\": %s\n",
                        label_copy.c_str(),
                        detail.UTF8String != nullptr
                            ? detail.UTF8String
                            : "unknown Metal validation error");
                    std::fflush(stderr);
                }
                if (completed.status != MTLCommandBufferStatusError) return;
                NSError* error = completed.error;
                NSString* detail =
                    error != nil
                        ? [NSString
                              stringWithFormat:@"%@/%ld/%@", error.domain,
                                               (long)error.code,
                                               error.localizedDescription
                                                   ?: @"unknown error"]
                        : @"unknown error";
                std::fprintf(stderr, "strata: GPU error in \"%s\": %s\n",
                             label_copy.c_str(),
                             detail.UTF8String ?: "unknown error");
                std::fflush(stderr);
            }
        }];
    }
    err.clear();
    return std::unique_ptr<CommandBuffer>(
        new CommandBuffer((__bridge_retained void*)command_buffer, (__bridge void*)impl_->queue));
}
Buffers& Context::buffers() { return *buffers_; }
const Buffers& Context::buffers() const { return *buffers_; }
std::unique_ptr<Events> Context::make_events(const std::string& label, std::string& err) const {
    auto events = std::unique_ptr<Events>(
        new Events((__bridge void*)impl_->device, (__bridge void*)impl_->queue, label, err));
    if (!err.empty()) return nullptr;
    return events;
}
void* Context::pipeline(const char* name) const {
    auto it = impl_->pipelines.find(name);
    return it == impl_->pipelines.end() ? nullptr : (__bridge void*) it->second;
}
bool Context::validate_pipeline_bindings(
    std::string_view name, std::span<const std::uint32_t> buffer_indices,
    std::span<const std::uint32_t> threadgroup_memory_indices,
    std::string& err) const {
    const auto found = impl_->kernel_bindings.find(std::string(name));
    if (found == impl_->kernel_bindings.end()) {
        err = "Metal kernel binding reflection is unavailable";
        return false;
    }
    if (!found->second.supported) {
        err = "Metal kernel uses resource bindings unsupported by the shared Encoder";
        return false;
    }
    std::vector<std::uint32_t> actual_buffers(buffer_indices.begin(),
                                              buffer_indices.end());
    std::vector<std::uint32_t> actual_threadgroup_memory(
        threadgroup_memory_indices.begin(), threadgroup_memory_indices.end());
    std::sort(actual_buffers.begin(), actual_buffers.end());
    std::sort(actual_threadgroup_memory.begin(), actual_threadgroup_memory.end());
    if (actual_buffers != found->second.buffers ||
        actual_threadgroup_memory != found->second.threadgroup_memory) {
        err = "Metal kernel resource slots do not match reflected bindings";
        return false;
    }
    err.clear();
    return true;
}

bool Context::run_selftest(std::string& err) {
    @autoreleasepool {
        const NSUInteger n = 1024;
        id<MTLBuffer> b = [impl_->device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        if (b == nil || b.contents == nullptr) { err = "cannot allocate selftest buffer"; return false; }
        constexpr NSUInteger layout_count = 41;
        id<MTLBuffer> layout =
            [impl_->device newBufferWithLength:layout_count * sizeof(std::uint32_t)
                                       options:MTLResourceStorageModeShared];
        if (layout == nil || layout.contents == nullptr) {
            err = "cannot allocate selftest layout buffer";
            return false;
        }
        float* x = static_cast<float*>(b.contents);
        for (NSUInteger i = 0; i < n; ++i) x[i] = static_cast<float>(i) * 0.001f;
        auto command = make_command_buffer("selftest", err);
        if (command == nullptr) return false;
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
        id<MTLComputeCommandEncoder> enc = begin_compute_encoder(*this, cb);
        if (enc == nil) {
            command->discard();
            err = "cannot create selftest compute encoder";
            return false;
        }
        for (const char* k : {"strata_selftest_add", "strata_selftest_exp"}) {
            [enc setComputePipelineState:impl_->pipelines.at(k)];
            [enc setBuffer:b offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc setComputePipelineState:impl_->pipelines.at("strata_selftest_layout")];
        [enc setBuffer:layout offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        end_compute_encoder(*this, enc);
        if (!command->commit(err)) return false;
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { err = "selftest command buffer failed"; return false; }
        for (NSUInteger i = 0; i < n; ++i) {
            const float want = std::exp(static_cast<float>(i) * 0.001f * 2.0f + 1.0f);
            if (std::fabs(x[i] - want) > 1e-4f * want) {
                err = "selftest: element " + std::to_string(i) + " is " + std::to_string(x[i]) + ", want " + std::to_string(want);
                return false;
            }
        }
        const std::array<std::uint32_t, layout_count> want = {
            sizeof(StepRow), sizeof(SampRow), sizeof(WindowParams), sizeof(PrefillParams), sizeof(MtpParams),
            offsetof(StepRow, pos), offsetof(StepRow, n_kv), offsetof(StepRow, n_bid), offsetof(StepRow, width),
            offsetof(SampRow, temperature), offsetof(SampRow, top_p), offsetof(SampRow, min_p),
            offsetof(SampRow, penalty_repeat), offsetof(SampRow, penalty_freq), offsetof(SampRow, penalty_present),
            offsetof(SampRow, top_k), offsetof(SampRow, min_keep), offsetof(SampRow, greedy),
            offsetof(SampRow, counter), offsetof(SampRow, hist_off), offsetof(SampRow, hist_len),
            offsetof(WindowParams, T), offsetof(WindowParams, nc), offsetof(WindowParams, pos0),
            offsetof(WindowParams, n_keep), offsetof(WindowParams, seed_lo), offsetof(WindowParams, seed_hi),
            offsetof(WindowParams, tokens), offsetof(WindowParams, step), offsetof(WindowParams, samp),
            offsetof(PrefillParams, T), offsetof(PrefillParams, p0), offsetof(PrefillParams, idx_f),
            offsetof(PrefillParams, idx_e),
            offsetof(MtpParams, T), offsetof(MtpParams, hid), offsetof(MtpParams, a), offsetof(MtpParams, j),
            offsetof(MtpParams, tokens), offsetof(MtpParams, step), offsetof(MtpParams, samp),
        };
        const auto* actual = static_cast<const std::uint32_t*>(layout.contents);
        for (NSUInteger i = 0; i < layout_count; ++i) {
            if (actual[i] != want[i]) {
                err = "selftest: Metal parameter layout item " + std::to_string(i) + " is " +
                      std::to_string(actual[i]) + ", want " + std::to_string(want[i]);
                return false;
            }
        }
        return true;
    }
}

bool Context::warm_no_copy_windows(const std::vector<BufferRef>& windows, std::string& err) {
    @autoreleasepool {
        constexpr std::size_t kPageBytes = 16 * 1024;
        constexpr NSUInteger kThreads = 256;
        if (windows.empty()) {
            err.clear();
            return true;
        }
        const auto pipeline_it = impl_->pipelines.find("strata_window_warm");
        if (pipeline_it == impl_->pipelines.end() || pipeline_it->second == nil) {
            err = "Metal window warm-up pipeline is unavailable";
            return false;
        }
        for (const BufferRef& window : windows) {
            if (!window || window.length == 0 || window.length % kPageBytes != 0 ||
                window.length / kPageBytes > std::numeric_limits<std::uint32_t>::max()) {
                err = "Metal window warm-up needs non-empty 16 KiB-aligned buffers";
                return false;
            }
            id<MTLBuffer> native = (__bridge id<MTLBuffer>)window.handle;
            if (native == nil || window.offset > native.length ||
                window.length > native.length - window.offset) {
                err = "Metal window warm-up view exceeds its buffer";
                return false;
            }
        }

        BufferRef checksum = buffers_->allocate("window warm-up checksum", sizeof(std::uint32_t), err);
        if (!checksum) return false;
        std::memset(checksum.contents, 0, sizeof(std::uint32_t));
        if (!buffers_->commit_residency(err)) return false;
        auto command = make_command_buffer("window warm-up", err);
        if (!command) return false;
        id<MTLCommandBuffer> native_command =
            (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
        id<MTLComputeCommandEncoder> encoder =
            begin_compute_encoder(*this, native_command);
        if (encoder == nil) {
            command->discard();
            err = "cannot create Metal window warm-up encoder";
            return false;
        }
        encoder.label = @"strata window warm-up";
        [encoder setComputePipelineState:pipeline_it->second];
        [encoder setBuffer:(__bridge id<MTLBuffer>)checksum.handle
                    offset:checksum.offset atIndex:1];
        for (const BufferRef& window : windows) {
            const std::uint32_t pages = static_cast<std::uint32_t>(window.length / kPageBytes);
            [encoder setBuffer:(__bridge id<MTLBuffer>)window.handle
                        offset:window.offset atIndex:0];
            [encoder setBytes:&pages length:sizeof(pages) atIndex:2];
            const NSUInteger groups = (static_cast<NSUInteger>(pages) + kThreads - 1) / kThreads;
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
        }
        end_compute_encoder(*this, encoder);
        if (!command->commit(err)) return false;
        [native_command waitUntilCompleted];
        if (native_command.status != MTLCommandBufferStatusCompleted) {
            NSError* command_error = native_command.error;
            const char* description = command_error.localizedDescription.UTF8String;
            err = "Metal window warm-up command buffer failed";
            if (description != nullptr && description[0] != '\0') err += ": " + std::string(description);
            return false;
        }
        err.clear();
        return true;
    }
}

}  // namespace strata::metal
