// src/metal/engine/expert_profile.cpp - strict, device-free reader for the unchanged STRP ranking format.
#include "strata/metal/expert_profile.hpp"

#include <fstream>
#include <limits>
#include <set>
#include <vector>

namespace strata::metal {
namespace {

std::uint32_t little_u32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint16_t little_u16(const unsigned char* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8);
}

}  // namespace

bool read_expert_profile(const std::string& path, std::uint32_t n_layers, std::uint32_t n_expert,
                         std::vector<std::pair<std::uint32_t, std::uint32_t>>& ranked,
                         std::string& error) {
    error.clear();
    ranked.clear();
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "cannot open expert profile " + path;
        return false;
    }
    const std::streamoff length = input.tellg();
    if (length < 24) {
        error = "expert profile " + path + " is too short to hold a STRP header";
        return false;
    }
    input.seekg(0);
    unsigned char header[24]{};
    if (!input.read(reinterpret_cast<char*>(header), sizeof(header))) {
        error = "failed reading expert profile header " + path;
        return false;
    }
    if (header[0] != 'S' || header[1] != 'T' || header[2] != 'R' || header[3] != 'P') {
        error = "expert profile " + path + " does not start with STRP";
        return false;
    }
    const std::uint32_t version = little_u32(header + 4);
    const std::uint32_t layers = little_u32(header + 8);
    const std::uint32_t experts = little_u32(header + 12);
    const std::uint32_t slots = little_u32(header + 16);
    const std::uint32_t n_ranked = little_u32(header + 20);
    (void)version;  // The STRP version is informational; its binary layout remains stable.
    if (layers != n_layers || experts != n_expert) {
        error = "expert profile " + path + " has dimensions " + std::to_string(layers) + "x" +
                std::to_string(experts) + ", expected " + std::to_string(n_layers) + "x" +
                std::to_string(n_expert);
        return false;
    }
    const std::uint64_t total = static_cast<std::uint64_t>(n_layers) * n_expert;
    if (slots > total || n_ranked > slots) {
        error = "expert profile ranking exceeds its slot count or model dimensions";
        return false;
    }
    const std::uint64_t pair_bytes = static_cast<std::uint64_t>(n_ranked) * 4;
    const std::uint64_t frequency_bytes = total * 4;
    if (pair_bytes > std::numeric_limits<std::uint64_t>::max() - 24 ||
        frequency_bytes > std::numeric_limits<std::uint64_t>::max() - 24 - pair_bytes ||
        static_cast<std::uint64_t>(length) != 24 + pair_bytes + frequency_bytes) {
        error = "expert profile " + path + " has an invalid length";
        return false;
    }

    std::vector<unsigned char> pairs(static_cast<std::size_t>(pair_bytes));
    if (!pairs.empty() &&
        !input.read(reinterpret_cast<char*>(pairs.data()), static_cast<std::streamsize>(pairs.size()))) {
        error = "expert profile " + path + " has a truncated ranking";
        return false;
    }
    std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
    ranked.reserve(n_ranked);
    for (std::uint32_t i = 0; i < n_ranked; ++i) {
        const std::uint32_t layer = little_u16(pairs.data() + static_cast<std::size_t>(i) * 4);
        const std::uint32_t expert = little_u16(pairs.data() + static_cast<std::size_t>(i) * 4 + 2);
        if (layer >= n_layers || expert >= n_expert) {
            error = "expert profile pair " + std::to_string(i) + " is out of range";
            ranked.clear();
            return false;
        }
        if (!seen.emplace(layer, expert).second) {
            error = "expert profile contains a duplicate (layer, expert) pair";
            ranked.clear();
            return false;
        }
        ranked.emplace_back(layer, expert);
    }
    return true;
}

}  // namespace strata::metal
