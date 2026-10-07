#pragma once

#include <cstdint>

namespace twin {

// A float oracle value and its double-precision running magnitude. Constructing
// from one float starts the magnitude at that value's absolute magnitude;
// callers may provide a larger accumulated magnitude explicitly.
struct tf {
    float value = 0.0f;
    double magnitude = 0.0;

    constexpr tf() noexcept = default;
    tf(float value) noexcept;
    tf(float value, double magnitude) noexcept;
};

// Arithmetic is performed on float values. Magnitudes follow the same
// expression over accumulated absolute magnitudes, with subtraction treated
// as addition.
[[nodiscard]] tf operator+(tf lhs, tf rhs) noexcept;
[[nodiscard]] tf operator-(tf lhs, tf rhs) noexcept;
[[nodiscard]] tf operator*(tf lhs, tf rhs) noexcept;
[[nodiscard]] tf operator/(tf lhs, tf rhs) noexcept;
[[nodiscard]] tf operator-(tf operand) noexcept;

// One explicitly fused float operation, matching a kernel's fma call.
[[nodiscard]] tf fma(tf a, tf b, tf c) noexcept;

// Accurate oracle helpers: evaluate the library function in double, round
// once to float, and restart the magnitude at the absolute rounded result.
[[nodiscard]] tf accurate_exp(tf x) noexcept;
[[nodiscard]] tf accurate_log(tf x) noexcept;
[[nodiscard]] tf accurate_sqrt(tf x) noexcept;

// Returns the IEEE-754 bits of the ordinary expression a*b+c. A build can
// call this with a contraction-sensitive triple and check that its result is
// the separately rounded product and sum required by -ffp-contract=off.
[[nodiscard]] std::uint32_t contraction_probe_bits(float a, float b, float c) noexcept;

}  // namespace twin
