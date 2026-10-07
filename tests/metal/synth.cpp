#include "synth.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace twin {
namespace {

double uniform_open(std::mt19937_64& rng) {
    const std::uint64_t mantissa = rng() >> 11;
    return (static_cast<double>(mantissa) + 0.5) * 0x1.0p-53;
}

float normal(std::mt19937_64& rng) {
    const double radius = std::sqrt(-2.0 * std::log(uniform_open(rng)));
    const double angle = 6.283185307179586476925286766559 * uniform_open(rng);
    return static_cast<float>(radius * std::cos(angle));
}

constexpr std::array<float, 16> kEdge = {
    0.0f, -0.0f, 1.0f, -1.0f, 0x1.0p-24f, -0x1.0p-24f, 65504.0f, -65504.0f,
    0.5f, -0.5f, 1.5f, -1.5f, 127.0f, -127.0f, 1024.0f, -1024.0f,
};

constexpr std::array<float, 16> kFixture = {
    0.0f, 1.0f, -1.0f, 0.125f, -0.25f, 0.5f, -0.75f, 1.25f,
    -1.5f, 2.0f, -2.5f, 3.0f, -4.0f, 8.0f, -16.0f, 32.0f,
};

}  // namespace

std::vector<float> synthesize_floats(InputSet set, std::size_t count,
                                     std::mt19937_64& rng, std::string& err) {
    std::vector<float> values(count);
    switch (set) {
        case InputSet::random:
            for (float& value : values) value = normal(rng);
            break;
        case InputSet::edge:
            for (std::size_t i = 0; i < count; ++i) values[i] = kEdge[i % kEdge.size()];
            break;
        case InputSet::fixture:
            for (std::size_t i = 0; i < count; ++i) values[i] = kFixture[i % kFixture.size()];
            break;
        case InputSet::real:
            err = "real inputs require a model pack and its CUDA dump fixture";
            return {};
    }
    err.clear();
    return values;
}

}  // namespace twin
