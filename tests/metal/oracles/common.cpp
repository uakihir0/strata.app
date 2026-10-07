#include "common.hpp"

#include <bit>
#include <cmath>

namespace twin {
namespace {

double abs_magnitude(double value) noexcept {
    return std::fabs(value);
}

tf from_accurate_result(float value) noexcept {
    return tf{value, abs_magnitude(static_cast<double>(value))};
}

}  // namespace

tf::tf(float initial_value) noexcept
    : value(initial_value),
      magnitude(abs_magnitude(static_cast<double>(initial_value))) {}

tf::tf(float initial_value, double initial_magnitude) noexcept
    : value(initial_value), magnitude(abs_magnitude(initial_magnitude)) {}

tf operator+(tf lhs, tf rhs) noexcept {
    return tf{lhs.value + rhs.value, lhs.magnitude + rhs.magnitude};
}

tf operator-(tf lhs, tf rhs) noexcept {
    return tf{lhs.value - rhs.value, lhs.magnitude + rhs.magnitude};
}

tf operator*(tf lhs, tf rhs) noexcept {
    return tf{lhs.value * rhs.value, lhs.magnitude * rhs.magnitude};
}

tf operator/(tf lhs, tf rhs) noexcept {
    return tf{lhs.value / rhs.value, lhs.magnitude / rhs.magnitude};
}

tf operator-(tf operand) noexcept {
    return tf{-operand.value, operand.magnitude};
}

tf fma(tf a, tf b, tf c) noexcept {
    return tf{std::fma(a.value, b.value, c.value),
              a.magnitude * b.magnitude + c.magnitude};
}

tf accurate_exp(tf x) noexcept {
    return from_accurate_result(static_cast<float>(std::exp(static_cast<double>(x.value))));
}

tf accurate_log(tf x) noexcept {
    return from_accurate_result(static_cast<float>(std::log(static_cast<double>(x.value))));
}

tf accurate_sqrt(tf x) noexcept {
    return from_accurate_result(static_cast<float>(std::sqrt(static_cast<double>(x.value))));
}

std::uint32_t contraction_probe_bits(float a, float b, float c) noexcept {
    return std::bit_cast<std::uint32_t>(a * b + c);
}

}  // namespace twin
