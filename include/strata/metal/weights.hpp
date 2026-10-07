// include/strata/metal/weights.hpp - device-free inspection of model and pack metadata.
#pragma once

#include "strata/metal/geometry.hpp"
#include "strata/metal/window_plan.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace strata::metal {

struct MtpRuntimeSources;

struct WeightStoreOptions {
    std::string native_shard;
    std::string pack_dir = "pack/full";
    std::string ple_gguf;
    std::string mtp_dir;
    std::string family;
    std::string model;
};

/// Check one tensor against the production Qwen role/type allowlist.
bool validate_model_tensor_type(std::string_view tensor_name, std::uint32_t tensor_type,
                                std::string& error);

/// Validate GGUF headers and the native pack, then derive the exact facts used by Geometry and Plan.
/// Tensor payloads are not read. The returned Geometry owns metadata only.
class WeightStore {
public:
    static bool inspect(const WeightStoreOptions& options, Geometry& geometry, std::string& error,
                        WindowSources* window_sources = nullptr,
                        MtpRuntimeSources* mtp_sources = nullptr);

private:
    static bool acquire_inspection_lock(const std::string& path,
                                        WindowSourceFileLock& result,
                                        std::string& error, bool create);
};

}  // namespace strata::metal
