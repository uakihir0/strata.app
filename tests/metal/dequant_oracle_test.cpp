// tests/metal/dequant_oracle_test.cpp - the artifact's Q4_K dequantizer against ggml-cpu's, bitwise, on random
// blocks (docs/mac/08 section 6.1: the pattern every Metal kernel parity test follows).  CPU only.
#include "strata/artifact/dequant.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const ggml_type_traits* t = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (t == nullptr || t->blck_size != 256 || t->type_size != 144 || t->to_float == nullptr) {
        std::printf("dequant_oracle_test: ggml has no Q4_K traits\n");
        return 1;
    }
    std::mt19937 rng(7);
    std::vector<uint8_t> block(144);
    std::vector<float> a(256), b(256);
    int bad = 0;
    for (int it = 0; it < 4096 && bad == 0; ++it) {
        for (auto& x : block) x = static_cast<uint8_t>(rng());
        const uint16_t d = 0x2000 | (rng() & 0x0fff), dmin = 0x1800 | (rng() & 0x0fff);   // finite, normal fp16
        std::memcpy(block.data(), &d, 2);
        std::memcpy(block.data() + 2, &dmin, 2);
        strata::dequantize_q4_K(block.data(), a.data());
        t->to_float(block.data(), b.data(), 256);
        bad = std::memcmp(a.data(), b.data(), 256 * sizeof(float)) != 0;
        if (bad) std::printf("block %d differs\n", it);
    }
    std::printf("dequant_oracle_test: %s\n", bad ? "FAILED" : "OK (4096 Q4_K blocks bitwise equal to ggml-cpu)");
    return bad;
}
