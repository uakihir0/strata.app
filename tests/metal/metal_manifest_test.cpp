// tests/metal/metal_manifest_test.cpp - the generated tables agree with the built strata.metallib, without a GPU.
#include "strata/metal/context.hpp"

#include "metallib_sha256.inc"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: metal_manifest_test strata.metallib\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string h = strata::metal::sha256_hex(bytes.data(), bytes.size());
    int fail = h != STRATA_METALLIB_SHA256;
    std::size_t n = 0;
    const strata::metal::KernelInfo* k = strata::metal::kernel_manifest(n);
    fail |= n == 0;
    for (std::size_t i = 0; i < n; ++i) { // a kernel name is in the metallib's function table as a plain string
        fail |= bytes.find(std::string(k[i].name) + '\0') == std::string::npos;
        const std::uint64_t group_threads =
            static_cast<std::uint64_t>(k[i].threadgroup_x) * k[i].threadgroup_y * k[i].threadgroup_z;
        fail |= k[i].threadgroup_x == 0 || k[i].threadgroup_y == 0 || k[i].threadgroup_z == 0;
        fail |= group_threads != k[i].threads || group_threads > 1024;
    }
    std::printf("metal_manifest_test: %zu kernels, sha256 %s: %s\n", n, h.c_str(), fail ? "FAILED" : "OK");
    return fail;
}
