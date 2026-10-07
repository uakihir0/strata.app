// include/strata/metal/context.hpp - the Metal device, strata.metallib and every pipeline (docs/mac/06 section 2.3,
// docs/mac/08 sections 5 and 10).  Plain C++: the Objective-C++ lives in src/metal/host/*.mm.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::metal {

enum class KernelMath { safe, fast };

struct KernelInfo {
    const char* name;
    KernelMath math;
    unsigned threads;   ///< the largest threadgroup the kernel is dispatched with
};

/// The kernels of src/metal/kernels/manifest.txt, in manifest order.
const KernelInfo* kernel_manifest(std::size_t& count);

struct ContextOptions {
    std::string metallib;          ///< empty: strata.metallib next to the executable
    std::string metallib_sha256;   ///< the hash the engine embeds; empty skips the check (--metallib)
    bool pipelines = true;         ///< false: device checks only (--plan-only, --list-devices)
    std::string capture;           ///< --metal-capture / STRATA_METAL_CAPTURE: the .gputrace to write
};

struct PipelineStats {
    unsigned from_archive = 0;
    unsigned compiled = 0;
    double ms = 0;
    std::string archive;   ///< the archive file, or why there is none
};

class Context {
public:
    /// Null and a one-line reason in `err` when any check of docs/mac/06 section 2.3 fails.
    static std::unique_ptr<Context> open(const ContextOptions& o, std::string& err);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    std::string device_name() const;
    const PipelineStats& pipeline_stats() const;
    /// The pipeline of a manifest kernel, as an opaque id<MTLComputePipelineState>; null if not built.
    void* pipeline(const char* name) const;
    void* device() const;
    void* queue() const;

    bool begin_capture(std::string& err);   ///< no-op (true) without ContextOptions::capture
    void end_capture();

    /// Skeleton only: one dispatch of strata_selftest_add and strata_selftest_exp over 1024 floats, checked.
    bool run_selftest(std::string& err);

    struct Impl;
private:
    explicit Context(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// SHA-256 of a byte range as 64 lower-case hex digits.
std::string sha256_hex(const void* p, std::size_t n);

}  // namespace strata::metal
