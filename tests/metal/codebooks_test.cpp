// Compare the Metal codebook header byte-for-byte with the selected ggml-common.h tables.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

// Compile ggml-common.h's GPU tables in a private namespace, leaving its source intact.
#define GGML_COMMON_IMPL_SYCL
#define iq2xxs_grid ggml_iq2xxs_grid
#define iq2xs_grid ggml_iq2xs_grid
#define iq2s_grid ggml_iq2s_grid
#define iq3xxs_grid ggml_iq3xxs_grid
#define iq3s_grid ggml_iq3s_grid
#define iq1s_grid_gpu ggml_iq1s_grid_gpu
#define ksigns_iq2xs ggml_ksigns_iq2xs
#define kmask_iq2xs ggml_kmask_iq2xs
#define kvalues_iq4nl ggml_kvalues_iq4nl
namespace ggml_reference {
#include "ggml-common.h"
}
#undef kvalues_iq4nl
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq1s_grid_gpu
#undef iq3s_grid
#undef iq3xxs_grid
#undef iq2s_grid
#undef iq2xs_grid
#undef iq2xxs_grid
#undef GGML_TABLE_END
#undef GGML_TABLE_BEGIN
#undef GGML_COMMON_IMPL
#undef GGML_COMMON_IMPL_SYCL

#include "codebooks.h"

#define STRATA_GGML_TABLE(name) ggml_reference::ggml_##name
#define STRATA_ASSERT_TABLE(name, count, type) \
    static_assert(std::extent_v<decltype(name)> == count, #name " length"); \
    static_assert(std::is_same_v<std::remove_cv_t<std::remove_extent_t<decltype(name)>>, type>, \
                  #name " element type"); \
    static_assert(std::extent_v<decltype(STRATA_GGML_TABLE(name))> == count, "ggml " #name " length"); \
    static_assert(std::is_same_v<std::remove_cv_t<std::remove_extent_t<decltype(STRATA_GGML_TABLE(name))>>, type>, \
                  "ggml " #name " element type")

STRATA_ASSERT_TABLE(iq2xxs_grid, 256, std::uint64_t);
STRATA_ASSERT_TABLE(iq2xs_grid, 512, std::uint64_t);
STRATA_ASSERT_TABLE(iq2s_grid, 1024, std::uint64_t);
STRATA_ASSERT_TABLE(iq3xxs_grid, 256, std::uint32_t);
STRATA_ASSERT_TABLE(iq3s_grid, 512, std::uint32_t);
STRATA_ASSERT_TABLE(iq1s_grid_gpu, 2048, std::uint32_t);
STRATA_ASSERT_TABLE(ksigns_iq2xs, 128, std::uint8_t);
STRATA_ASSERT_TABLE(kmask_iq2xs, 8, std::uint8_t);
STRATA_ASSERT_TABLE(kvalues_iq4nl, 16, std::int8_t);

#undef STRATA_ASSERT_TABLE

template <typename T, std::size_t N>
bool compare_table(const char* name, const T (&actual)[N], const T (&expected)[N]) {
    const auto* actual_bytes = reinterpret_cast<const unsigned char*>(actual);
    const auto* expected_bytes = reinterpret_cast<const unsigned char*>(expected);
    constexpr std::size_t byte_count = N * sizeof(T);
    for (std::size_t i = 0; i < byte_count; ++i) {
        if (actual_bytes[i] != expected_bytes[i]) {
            std::printf("%s differs at byte %zu: got 0x%02x, expected 0x%02x\n", name, i,
                        static_cast<unsigned int>(actual_bytes[i]),
                        static_cast<unsigned int>(expected_bytes[i]));
            return false;
        }
    }
    return true;
}

int main() {
    bool ok = true;
#define STRATA_COMPARE_TABLE(name) ok = compare_table(#name, name, STRATA_GGML_TABLE(name)) && ok
    STRATA_COMPARE_TABLE(iq2xxs_grid);
    STRATA_COMPARE_TABLE(iq2xs_grid);
    STRATA_COMPARE_TABLE(iq2s_grid);
    STRATA_COMPARE_TABLE(iq3xxs_grid);
    STRATA_COMPARE_TABLE(iq3s_grid);
    STRATA_COMPARE_TABLE(iq1s_grid_gpu);
    STRATA_COMPARE_TABLE(ksigns_iq2xs);
    STRATA_COMPARE_TABLE(kmask_iq2xs);
    STRATA_COMPARE_TABLE(kvalues_iq4nl);
#undef STRATA_COMPARE_TABLE
#undef STRATA_GGML_TABLE

    std::printf("codebooks: %s (9 tables, 25752 bytes)\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
