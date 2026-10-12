// include/strata/metal/encoder.hpp - validated, policy-aware Metal dispatch encoding.
#pragma once

#include "strata/metal/context.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace strata::metal {

class ExpertLeaseSet;

struct EncoderSize {
    std::size_t x = 1;
    std::size_t y = 1;
    std::size_t z = 1;
};

struct EncoderBufferBinding {
    BufferRef buffer;
    std::uint32_t index = 0;
    std::size_t required_bytes = 1;
    std::size_t alignment = 1;
};

/// A read-only buffer argument. Use only for MSL `const` buffer parameters;
/// one dispatch may not bind an overlapping writable view of the same buffer.
struct EncoderReadOnlyBufferBinding {
    ReadOnlyBufferRef buffer;
    std::uint32_t index = 0;
    std::size_t required_bytes = 1;
    std::size_t alignment = 1;
};

/// A read-only Metal resource accessed through a GPU address rather than a
/// shader buffer argument. Declaring it keeps untracked allocations resident
/// and visible to the command encoder.
struct EncoderResourceBinding {
    ReadOnlyBufferRef buffer;
    std::size_t required_bytes = 1;
    std::size_t alignment = 1;
};

struct EncoderBytesBinding {
    /// Copied inline payload; the total across a segment is limited to 4096 bytes.
    const void* bytes = nullptr;
    std::size_t length = 0;
    std::uint32_t index = 0;
};

struct EncoderThreadgroupMemory {
    std::uint32_t index = 0;
    std::size_t length = 0;
};

/// A single compute-encoder segment. Encode a segment with `encode`, then add
/// validated manifest dispatches and declare independent dispatch groups.
class Encoder {
public:
    using Body = std::function<bool(Encoder&, std::string&)>;

    ~Encoder();
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    /// Create one labeled compute encoder under the Context fence. A failed
    /// body, invalid dispatch, or exception discards the unsubmitted command
    /// buffer so partially encoded work cannot be submitted.
    static bool encode(Context& context, CommandBuffer& command_buffer,
                       std::string_view label, const Body& body,
                       std::string& error);

    bool dispatch(std::string_view kernel, EncoderSize grid,
                  EncoderSize threads_per_group,
                  std::span<const EncoderBufferBinding> buffers,
                  std::span<const EncoderBytesBinding> bytes,
                  std::span<const EncoderThreadgroupMemory> threadgroup_memory,
                  std::string& error);
    /// Overload for kernels with explicitly read-only MSL buffer inputs.
    bool dispatch(std::string_view kernel, EncoderSize grid,
                  EncoderSize threads_per_group,
                  std::span<const EncoderBufferBinding> buffers,
                  std::span<const EncoderReadOnlyBufferBinding> read_only_buffers,
                  std::span<const EncoderBytesBinding> bytes,
                  std::span<const EncoderThreadgroupMemory> threadgroup_memory,
                  std::string& error);

    /// Declare read-only resources dereferenced through GPU addresses.
    bool use_resources(std::span<const EncoderResourceBinding> resources,
                       std::string& error);

    /// Reserve caller-owned staging bytes once per pending command buffer.
    /// Reusing an overlapping range in that command buffer is rejected.
    bool claim_unique_staging_range(BufferRef buffer,
                                    std::size_t required_bytes,
                                    std::size_t alignment,
                                    std::string& error);

    bool group_begin(std::string& error);
    bool group_end(std::string& error);

    /// Return whether concurrent dispatches are currently inside a group,
    /// where automatic inter-dispatch buffer barriers are suppressed.
    bool concurrent_group_open() const noexcept;

    /// Return whether this unsubmitted segment has encoded at least one
    /// compute dispatch. Useful to verify coordinator rollback after a prefix.
    bool has_encoded_dispatches() const noexcept;

private:
    struct Impl;
    explicit Encoder(std::unique_ptr<Impl> impl);
    const void* command_buffer_identity() const noexcept;
    std::unique_ptr<Impl> impl_;

    friend class ExpertLeaseSet;
};

}  // namespace strata::metal
