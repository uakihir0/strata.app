// Device-independent Prefill GEMM backend policy (docs/mac/04 §6.2).
#include "strata/metal/prefill_gemm_dispatch.hpp"

namespace strata::metal {
namespace {

constexpr std::uint32_t kMinimumMPPColumns = 64u;

}  // namespace

bool resolve_prefill_gemm_backend(
    bool apple10, std::uint32_t n, std::string_view setting,
    PrefillGemmBackend& backend, std::string& error) {
    if (n == 0u) {
        error = "STRATA_METAL_GEMM selection requires n > 0";
        return false;
    }
    if (!setting.empty() && setting != "mpp" && setting != "simd") {
        error = "STRATA_METAL_GEMM must be 'mpp' or 'simd'";
        return false;
    }

    PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
    if (n >= kMinimumMPPColumns) {
        if (setting == "mpp") {
            if (!apple10) {
                error = "STRATA_METAL_GEMM=mpp requires an Apple10 GPU";
                return false;
            }
            selected = PrefillGemmBackend::MPP;
        } else if (setting == "simd") {
            selected = PrefillGemmBackend::SIMD;
        } else if (apple10) {
            selected = PrefillGemmBackend::MPP;
        }
    }

    backend = selected;
    error.clear();
    return true;
}

}  // namespace strata::metal
