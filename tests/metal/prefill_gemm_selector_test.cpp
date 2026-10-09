#include "strata/metal/prefill_gemm_dispatch.hpp"

#include <array>
#include <cstdio>
#include <cstdint>
#include <string>
#include <string_view>

using namespace strata::metal;

namespace {

struct Case {
    const char* name;
    bool apple10;
    std::uint32_t n;
    std::string_view setting;
    bool succeeds;
    PrefillGemmBackend expected;
    std::string_view error_fragment;
};

bool run_case(const Case& test) {
    PrefillGemmBackend backend = PrefillGemmBackend::MPP;
    std::string error = "stale";
    const bool succeeded = resolve_prefill_gemm_backend(
        test.apple10, test.n, test.setting, backend, error);
    const bool valid =
        succeeded == test.succeeds &&
        (succeeded
             ? backend == test.expected && error.empty()
             : backend == PrefillGemmBackend::MPP &&
                   error.find(test.error_fragment) != std::string::npos);
    std::printf("%s %s%s%s\n", valid ? "ok  " : "FAIL", test.name,
                error.empty() ? "" : ": ",
                error.empty() ? "" : error.c_str());
    return valid;
}

}  // namespace

int main() {
    constexpr std::array<Case, 11u> cases{{
        {"default narrow Apple10 projection", true, 1u, "", true,
         PrefillGemmBackend::SIMD, ""},
        {"default edge-width Apple10 projection", true, 63u, "", true,
         PrefillGemmBackend::SIMD, ""},
        {"default Apple10 wide projection", true, 64u, "", true,
         PrefillGemmBackend::MPP, ""},
        {"default earlier-family wide projection", false, 10240u, "", true,
         PrefillGemmBackend::SIMD, ""},
        {"forced SIMD Apple10 projection", true, 10240u, "simd", true,
         PrefillGemmBackend::SIMD, ""},
        {"forced SIMD earlier-family projection", false, 64u, "simd", true,
         PrefillGemmBackend::SIMD, ""},
        {"forced MPP Apple10 projection", true, 65u, "mpp", true,
         PrefillGemmBackend::MPP, ""},
        {"narrow projection stays SIMD with MPP request", true, 1u, "mpp",
         true, PrefillGemmBackend::SIMD, ""},
        {"narrow projection stays SIMD on earlier family", false, 63u, "mpp",
         true, PrefillGemmBackend::SIMD, ""},
        {"MPP refused on earlier family", false, 64u, "mpp", false,
         PrefillGemmBackend::SIMD, "requires an Apple10 GPU"},
        {"invalid selector rejected", true, 64u, "auto", false,
         PrefillGemmBackend::SIMD, "must be 'mpp' or 'simd'"},
    }};

    bool passed = true;
    for (const auto& test : cases) passed &= run_case(test);
    const Case zero_columns{"zero-width projection rejected", true, 0u, "",
                            false, PrefillGemmBackend::SIMD, "n > 0"};
    passed &= run_case(zero_columns);
    if (!passed) return 1;
    std::puts("all Prefill GEMM selector tests passed");
    return 0;
}
