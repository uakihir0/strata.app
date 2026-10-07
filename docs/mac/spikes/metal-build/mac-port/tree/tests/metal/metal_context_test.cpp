// tests/metal/metal_context_test.cpp - Context::open on a real GPU: the hash check, the pipeline cache (a miss,
// then a hit, then a damaged archive) and one dispatch.  Needs an Apple GPU.
#include "strata/metal/context.hpp"

#include "metallib_sha256.inc"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::metal;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const char* what, const std::string& detail = {}) {
    std::printf("%s %s %s\n", ok ? "ok  " : "FAIL", what, detail.c_str());
    g_fail |= !ok;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: metal_context_test strata.metallib CACHE_DIR\n"); return 2; }
    const fs::path cache = argv[2];
    fs::remove_all(cache);
    setenv("STRATA_METAL_ARCHIVE", cache.c_str(), 1);
    ContextOptions o;
    o.metallib = argv[1];
    o.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string err;
    std::size_t n = 0;
    kernel_manifest(n);
    {
        auto c = Context::open(o, err);
        check(c != nullptr, "first open", err);
        if (!c) return 1;
        check(c->pipeline_stats().compiled == n && c->pipeline_stats().from_archive == 0, "first open compiles all");
        check(c->run_selftest(err), "selftest dispatch", err);
    }
    {
        auto c = Context::open(o, err);
        check(c != nullptr && c->pipeline_stats().from_archive == n, "second open uses the archive",
              c ? c->pipeline_stats().archive : err);
    }
    for (const auto& e : fs::directory_iterator(cache)) {   // damage the archive: ignored and rebuilt
        std::ofstream(e.path(), std::ios::binary | std::ios::trunc) << std::string(32768, 'x');
    }
    {
        auto c = Context::open(o, err);
        check(c != nullptr && c->pipeline_stats().compiled == n, "a damaged archive is ignored", err);
    }
    {
        auto c = Context::open(o, err);
        check(c != nullptr && c->pipeline_stats().from_archive == n, "and rewritten", err);
    }
    o.metallib_sha256 = std::string(64, '0');
    check(Context::open(o, err) == nullptr && err.find("does not belong to this engine") != std::string::npos,
          "a foreign metallib is refused", err);
    std::printf("metal_context_test: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail;
}
