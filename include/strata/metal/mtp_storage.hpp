// include/strata/metal/mtp_storage.hpp - mapped MTP files owned by a Metal Context.
#pragma once

#include "strata/metal/context.hpp"
#include "strata/metal/mtp.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

/// Map the inspected MTP files into bounded, warmed no-copy Metal buffers.
///
/// Context retains each mapping and the MTP generation lock until its buffers
/// are released. Context must outlive this object. If an enclosing
/// Buffers::Transaction rolls back these windows, accessors return empty views.
class MtpStorage {
public:
    static std::unique_ptr<MtpStorage> open(Context& context,
                                            const MtpRuntimeSources& sources,
                                            std::string& error);
    ~MtpStorage();
    MtpStorage(const MtpStorage&) = delete;
    MtpStorage& operator=(const MtpStorage&) = delete;

    const MtpRuntimeLayout& layout() const noexcept;
    const std::vector<std::uint32_t>& draft_vocab_ids() const noexcept;
    BufferRef expert(std::uint32_t expert_index) const noexcept;
    BufferRef dense_weight(const std::string& tensor_name) const noexcept;
    BufferRef expert_window(std::size_t window_index) const noexcept;
    BufferRef dense_window(std::size_t window_index) const noexcept;

private:
    struct Impl;
    explicit MtpStorage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
