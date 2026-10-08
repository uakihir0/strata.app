// include/strata/metal/context.hpp - the Metal device, strata.metallib and every pipeline (docs/mac/06 section 2.3,
// docs/mac/08 sections 5 and 10).  Plain C++: the Objective-C++ lives in src/metal/host/*.mm.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::metal {

class Buffers;
class Events;
class Context;

enum class KernelMath { safe, fast };

struct KernelInfo {
    const char* name;
    KernelMath math;
    unsigned threads;   ///< total threads in the declared threadgroup
    unsigned threadgroup_x;
    unsigned threadgroup_y;
    unsigned threadgroup_z;
};

/// The kernels of src/metal/kernels/manifest.txt, in manifest order.
const KernelInfo* kernel_manifest(std::size_t& count);

struct ContextOptions {
    std::string metallib;          ///< empty: strata.metallib next to the executable
    std::string metallib_sha256;   ///< the hash the engine embeds; empty skips the check (--metallib)
    bool pipelines = true;         ///< false: device checks only (--plan-only, --list-devices)
    std::string capture;           ///< --metal-capture / STRATA_METAL_CAPTURE: the .gputrace to write
    /// Test/simulation cap; zero uses hardware limits and a nonzero value can only reduce the limit.
    std::uint64_t max_buffer_length_cap = 0;
};

namespace detail {

constexpr bool gpu_address_with_offset(std::uint64_t base,
                                       std::size_t offset,
                                       std::uint64_t& address) noexcept {
    if (base == 0) {
        address = 0;
        return true;
    }
    if (offset > std::numeric_limits<std::uint64_t>::max() - base) return false;
    address = base + static_cast<std::uint64_t>(offset);
    return true;
}

}  // namespace detail

struct PipelineStats {
    unsigned from_archive = 0;
    unsigned compiled = 0;
    double ms = 0;
    std::string archive;   ///< the archive file, or why there is none
};

struct DeviceFacts {
    std::uint64_t gpu_core_count = 0;     ///< zero means IORegistry did not report it
    std::uint64_t working_set_bytes = 0;
    std::uint64_t max_buffer_length = 0;
    std::uint64_t max_threadgroup_memory_length = 0;
    std::uint64_t physical_memory_bytes = 0;
    std::string macos_version;
    std::vector<std::string> families;
};

/// Shared validity state for copies of a Context-owned buffer view.
class BufferValidity {
private:
    BufferValidity(void* handle, void* contents, std::uint64_t gpu_address,
                   std::size_t allocation_length, std::uint64_t allocation_identity)
        : handle_(handle),
          contents_(reinterpret_cast<std::uintptr_t>(contents)),
          gpu_address_(gpu_address),
          allocation_length_(allocation_length),
          allocation_identity_(allocation_identity) {}

    bool valid_for(const void* handle, const void* contents,
                   std::uint64_t gpu_address, std::size_t view_length,
                   std::size_t offset) const noexcept {
        std::uint64_t expected_gpu_address = 0;
        if (handle == nullptr || handle_ != handle || allocation_identity_ == 0 ||
            !live_.load(std::memory_order_acquire) || offset > allocation_length_ ||
            view_length > allocation_length_ - offset ||
            (gpu_address_ != 0 &&
             allocation_length_ >
                 std::numeric_limits<std::uint64_t>::max() - gpu_address_) ||
            !detail::gpu_address_with_offset(gpu_address_, offset,
                                             expected_gpu_address) ||
            gpu_address != expected_gpu_address) {
            return false;
        }
        if (contents_ == 0) return contents == nullptr;
        if (allocation_length_ >
                std::numeric_limits<std::uintptr_t>::max() - contents_ ||
            offset > std::numeric_limits<std::uintptr_t>::max() - contents_) {
            return false;
        }
        return reinterpret_cast<std::uintptr_t>(contents) == contents_ + offset;
    }

    void invalidate() noexcept { live_.store(false, std::memory_order_release); }

    void* handle_ = nullptr;
    std::uintptr_t contents_ = 0;
    std::uint64_t gpu_address_ = 0;
    std::size_t allocation_length_ = 0;
    std::uint64_t allocation_identity_ = 0;
    std::atomic<bool> live_{true};

    friend class Buffers;
    friend struct BufferRef;
};

/// A borrowed view into a Context-owned MTLBuffer. Copies share the allocation's
/// validity token; `operator bool` becomes false when an enclosing transaction
/// rolls the allocation back or the Context is destroyed. Check the view before
/// reading its raw handle or contents pointer.
struct BufferRef {
    BufferRef() = default;

    void* handle = nullptr;          ///< opaque id<MTLBuffer>
    void* contents = nullptr;        ///< CPU-visible pointer at offset
    std::uint64_t gpu_address = 0;   ///< GPU address at offset
    std::size_t length = 0;          ///< bytes available from this view
    std::size_t offset = 0;          ///< offset into handle

    explicit operator bool() const noexcept {
        return validity_ != nullptr &&
               handle == issued_handle_ && contents == issued_contents_ &&
               gpu_address == issued_gpu_address_ && length <= issued_length_ &&
               offset == issued_offset_ &&
               validity_->valid_for(handle, contents, gpu_address, length, offset);
    }

    /// Return a bounded view relative to this view, or an empty view on failure.
    BufferRef slice(std::size_t relative_offset, std::size_t view_length) const noexcept {
        std::uint64_t sliced_gpu_address = 0;
        if (!*this || relative_offset > length || view_length > length - relative_offset ||
            relative_offset > std::numeric_limits<std::size_t>::max() - offset ||
            !detail::gpu_address_with_offset(gpu_address, relative_offset,
                                             sliced_gpu_address)) {
            return {};
        }
        void* sliced_contents = nullptr;
        if (contents != nullptr) {
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(contents);
            if (relative_offset > std::numeric_limits<std::uintptr_t>::max() - base) return {};
            sliced_contents = reinterpret_cast<void*>(base + relative_offset);
        }
        return BufferRef(handle, sliced_contents, sliced_gpu_address,
                         view_length, offset + relative_offset, validity_);
    }

private:
    BufferRef(void* buffer_handle, void* cpu_contents, std::uint64_t gpu_address,
              std::size_t view_length, std::size_t buffer_offset,
              std::shared_ptr<const BufferValidity> validity)
        : handle(buffer_handle),
          contents(cpu_contents),
          gpu_address(gpu_address),
          length(view_length),
          offset(buffer_offset),
          issued_handle_(buffer_handle),
          issued_contents_(cpu_contents),
          issued_gpu_address_(gpu_address),
          issued_length_(view_length),
          issued_offset_(buffer_offset),
          validity_(std::move(validity)) {}

    void* issued_handle_ = nullptr;
    void* issued_contents_ = nullptr;
    std::uint64_t issued_gpu_address_ = 0;
    std::size_t issued_length_ = 0;
    std::size_t issued_offset_ = 0;
    std::shared_ptr<const BufferValidity> validity_;

    friend class Buffers;
};

/// A borrowed Context-owned buffer view for reading model weights. Unlike
/// BufferRef, its CPU mapping is exposed only as const data. Copies retain the
/// same generation-checked validity token but do not own the MTLBuffer.
class ReadOnlyBufferRef {
public:
    ReadOnlyBufferRef() = default;

    explicit operator bool() const noexcept { return static_cast<bool>(buffer_); }
    const void* contents() const noexcept {
        return buffer_ ? static_cast<const void*>(buffer_.contents) : nullptr;
    }
    std::uint64_t gpu_address() const noexcept {
        return buffer_ ? buffer_.gpu_address : 0;
    }
    std::size_t length() const noexcept { return buffer_ ? buffer_.length : 0; }
    std::size_t offset() const noexcept { return buffer_ ? buffer_.offset : 0; }

    /// Return a bounded read-only subview, or an empty view on failure.
    ReadOnlyBufferRef slice(std::size_t relative_offset,
                            std::size_t view_length) const noexcept {
        return *this ? ReadOnlyBufferRef(buffer_.slice(relative_offset, view_length))
                     : ReadOnlyBufferRef{};
    }

private:
    explicit ReadOnlyBufferRef(BufferRef buffer) : buffer_(std::move(buffer)) {}
    BufferRef buffer_;

    friend class Buffers;
    friend class FullModeStorage;
};

/// Owns a Metal command buffer until this C++ handle is destroyed. Use enqueue()
/// and commit() to control submission. Destroying a NotEnqueued handle discards
/// it; Metal cannot cancel an Enqueued handle, so destruction commits it.
/// Encode all work before enqueueing, and commit or discard it before destroying
/// the Context that created it.
class CommandBuffer {
public:
    ~CommandBuffer();
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    bool enqueue(std::string& err);
    bool commit(std::string& err);
    void discard() noexcept;
    /// Register that this command buffer reads or otherwise depends on work
    /// encoded by `producer`. The producer must be created first, submitted,
    /// and complete successfully before this command buffer can be submitted.
    bool depends_on(CommandBuffer& producer, std::string& err);
    /// Wait for a committed command buffer and return false if Metal reports an
    /// execution error. Required before submitting a command buffer that
    /// depends on this one.
    bool wait_until_completed(std::string& err) const;
    /// Hold this command buffer against concurrent enqueue, commit or discard
    /// while validating and encoding one engine operation. Do not submit or
    /// discard the command buffer from the same scope.
    std::unique_lock<std::recursive_mutex> lock_for_encoding() const;
    /// Run an encoder callback while holding this command buffer's encoding
    /// lock. The callback receives the native command buffer and must not
    /// submit or discard this wrapper. If `discard_on_failure` is true, a false
    /// return or exception discards it before another thread can submit it.
    bool encode_unsubmitted(
        const std::function<bool(void*, std::string&)>& encode,
        std::string& err, bool discard_on_failure = false);
    /// True when this command buffer was created from `context`'s queue.
    bool belongs_to(const Context& context) const;
    /// Run exactly once when this command buffer completes, or with false if a
    /// NotEnqueued command buffer is discarded. Register before enqueue/commit.
    bool add_completion_handler(std::function<void(bool)> handler, std::string& err);
    /// UNSAFE escape hatch for Objective-C++ encoding and diagnostics. Never
    /// enqueue or commit the returned command buffer directly; doing so bypasses
    /// event-order validation. Submit through this wrapper.
    void* unsafe_native_handle() const noexcept { return discarded_ ? nullptr : handle_; }

private:
    struct SubmissionHandler {
        std::function<bool(std::string&)> validate;
        std::function<void(bool)> notify;
    };
    struct SubmissionState;
    struct CompletionState;
    explicit CommandBuffer(void* retained_handle, void* queue_handle);
    bool can_encode() const noexcept;
    bool add_discard_handler(std::function<void()> handler);
    bool add_submission_handler(SubmissionHandler handler, std::string& err);
    bool validate_submission(std::string& err) const;
    void notify_submission(bool committed);
    void* handle_ = nullptr;
    void* queue_handle_ = nullptr;
    mutable std::recursive_mutex encoding_mutex_;
    bool committed_ = false;
    bool discarded_ = false;
    bool enqueued_ = false;
    std::uint64_t event_operation_count_ = 0;
    std::vector<std::function<void()>> discard_handlers_;
    std::vector<SubmissionHandler> submission_handlers_;
    std::shared_ptr<SubmissionState> submission_state_;
    std::vector<std::shared_ptr<SubmissionState>> dependencies_;
    std::shared_ptr<CompletionState> completion_state_;
    friend class Context;
    friend class Events;
};

class Context {
public:
    /// Null and a one-line reason in `err` when any check of docs/mac/06 section 2.3 fails.
    static std::unique_ptr<Context> open(const ContextOptions& o, std::string& err);
    /// Drains committed work on this Context's queue before releasing its buffers and residency set.
    /// Callers must commit or discard every command buffer before destroying the Context.
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    std::string device_name() const;
    /// Immutable facts captured once when this Context opens.
    const DeviceFacts& device_facts() const;
    const PipelineStats& pipeline_stats() const;
    /// The pipeline of a manifest kernel, as an opaque id<MTLComputePipelineState>; null if not built.
    void* pipeline(const char* name) const;
    /// Require the supplied resource slots to match the active Metal function bindings.
    bool validate_pipeline_bindings(
        std::string_view name, std::span<const std::uint32_t> buffer_indices,
        std::span<const std::uint32_t> threadgroup_memory_indices,
        std::string& err) const;
    void* device() const;
    void* queue() const;
    /// Opaque MTLFence shared by every encoder on this Context's command queue.
    void* hazard_fence() const noexcept;
    std::unique_ptr<CommandBuffer> make_command_buffer(const std::string& label, std::string& err) const;
    Buffers& buffers();
    const Buffers& buffers() const;
    std::unique_ptr<Events> make_events(const std::string& label, std::string& err) const;

    bool begin_capture(std::string& err);   ///< no-op (true) without ContextOptions::capture
    void end_capture();

    /// Skeleton only: one dispatch of strata_selftest_add and strata_selftest_exp over 1024 floats, checked.
    bool run_selftest(std::string& err);
    /// Touch one 16-byte word per 16 KiB page in each no-copy file window.
    bool warm_no_copy_windows(const std::vector<BufferRef>& windows, std::string& err);

    struct Impl;
private:
    explicit Context(std::unique_ptr<Impl> impl, std::unique_ptr<Buffers> buffers);
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<Buffers> buffers_;
};

/// Shared MTLBuffers and aligned no-copy windows, all retained and labeled by the Context.
class Buffers {
public:
    /// Roll back buffers registered during a failed storage-open attempt.
    /// Buffer operations performed while a transaction is live are serialized.
    class Transaction {
    public:
        explicit Transaction(Buffers& buffers);
        ~Transaction();
        Transaction(const Transaction&) = delete;
        Transaction& operator=(const Transaction&) = delete;
        Transaction(Transaction&&) = delete;
        Transaction& operator=(Transaction&&) = delete;

        void commit() noexcept;

    private:
        Buffers* buffers_ = nullptr;
        std::unique_lock<std::recursive_mutex> lock_;
        std::size_t allocation_count_ = 0;
        std::size_t backing_lifetime_count_ = 0;
        bool committed_ = false;
    };

    ~Buffers();
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

    Transaction transaction();
    BufferRef allocate(const std::string& label, std::size_t bytes, std::string& err, bool poison = false);
    BufferRef wrap_no_copy(const std::string& label, void* address, std::size_t bytes, std::string& err);
    /// Retain the mapping owner until the Context releases this no-copy buffer.
    BufferRef wrap_no_copy(const std::string& label, void* address, std::size_t bytes,
                           std::shared_ptr<void> backing_lifetime, std::string& err);
    bool commit_residency(std::string& err);
    std::size_t allocation_count() const;
    std::uint64_t allocated_size() const;
    /// True when this generation-checked view references an MTLBuffer retained by this pool.
    bool owns(BufferRef buffer) const;
    /// Read-only view equivalent; validates its private underlying generation token.
    bool owns(ReadOnlyBufferRef buffer) const;
    /// Monotonic identity for a live owned buffer handle; zero when it is not owned.
    std::uint64_t allocation_identity(BufferRef buffer) const;
    /// Also verifies that the handle still refers to the same allocation generation.
    bool owns(BufferRef buffer, std::uint64_t identity) const;

private:
    struct Impl;
    Buffers(void* device, void* queue, std::uint64_t max_buffer_length,
            std::string& err);
    std::unique_ptr<Impl> impl_;
    friend class Context;
    friend class BufferArena;
};

/// A fixed-address bump allocator over one shared MTLBuffer.
class BufferArena {
public:
    BufferArena(Buffers& buffers, const std::string& label, std::size_t capacity,
                std::string& err, bool poison = false);
    BufferRef allocate(std::size_t bytes, std::size_t alignment, std::string& err);
    std::size_t capacity() const noexcept;
    std::size_t used() const noexcept;

private:
    BufferRef backing_;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
};

/// A monotonic MTLSharedEvent used for host/GPU ordering. Value reservation and
/// host/GPU signal encoding are serialized. GPU signals must be submitted in
/// value order on the same queue, and a GPU wait cannot enter the queue before
/// its pending GPU producer. Discarding a NotEnqueued command buffer cancels
/// its pending event operations.
class Events {
public:
    ~Events();
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;

    std::uint64_t reserve_value();
    std::uint64_t signaled_value() const;
    bool signal_host(std::uint64_t value, std::string& err);
    bool wait(std::uint64_t value, std::uint64_t timeout_ms, std::string& err) const;
    bool encode_wait(CommandBuffer& command_buffer, std::uint64_t value, std::string& err) const;
    bool encode_signal(CommandBuffer& command_buffer, std::uint64_t value, std::string& err) const;

private:
    struct Impl;
    Events(void* device, void* queue, const std::string& label, std::string& err);
    std::shared_ptr<Impl> impl_;
    friend class Context;
};

/// SHA-256 of a byte range as 64 lower-case hex digits.
std::string sha256_hex(const void* p, std::size_t n);

}  // namespace strata::metal
