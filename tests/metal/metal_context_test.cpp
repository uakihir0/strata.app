// tests/metal/metal_context_test.cpp - Context::open on a real GPU: the hash check, the pipeline cache (a miss,
// then a hit, then a damaged archive) and one dispatch.  Needs an Apple GPU.
#include "strata/metal/context.hpp"

#include "metallib_sha256.inc"

#include <algorithm>
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
    std::fflush(stdout);
    g_fail |= !ok;
}

static bool validation_enabled() {
    const char* api = std::getenv("MTL_DEBUG_LAYER");
    const char* shader = std::getenv("MTL_SHADER_VALIDATION");
    return (api != nullptr && std::string(api) == "1") ||
           (shader != nullptr && std::string(shader) == "1");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
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
        const DeviceFacts& facts = c->device_facts();
        check(&facts == &c->device_facts(), "device facts are cached for the Context lifetime");
        check(!c->device_name().empty(), "device name is available");
        check(facts.gpu_core_count > 0, "IORegistry GPU core count is available");
        check(facts.working_set_bytes > 0 && facts.max_buffer_length > 0,
              "Metal memory limits are available");
        check(facts.max_threadgroup_memory_length > 0, "threadgroup memory limit is available");
        check(facts.physical_memory_bytes > 0 && !facts.macos_version.empty(),
              "host memory and macOS version are available");
        check(std::find(facts.families.begin(), facts.families.end(), "apple7") != facts.families.end() &&
                  std::find(facts.families.begin(), facts.families.end(), "metal4") != facts.families.end(),
              "required Metal families are reported");
        check(c->pipeline_stats().compiled == n && c->pipeline_stats().from_archive == 0, "first open compiles all");
        if (validation_enabled()) {
            check(c->pipeline_stats().archive == "disabled while Metal validation is enabled",
                  "pipeline archive is bypassed under validation", c->pipeline_stats().archive);
        }
        check(c->run_selftest(err), "selftest dispatch", err);
    }
    if (validation_enabled()) {
        std::printf("metal_context_test: %s\n", g_fail ? "FAILED" : "OK");
        return g_fail;
    }
    {
        auto c = Context::open(o, err);
        const bool archive_hit =
            c != nullptr && c->pipeline_stats().from_archive == n;
        check(archive_hit, "second open uses the archive",
              c ? c->pipeline_stats().archive : err);
        if (archive_hit) {
            const std::uint32_t buffer_slot = 0;
            std::string binding_error;
            const bool active_slot_accepted = c->validate_pipeline_bindings(
                "strata_selftest_add",
                std::span<const std::uint32_t>(&buffer_slot, 1), {}, binding_error);
            binding_error.clear();
            const bool missing_slot_rejected =
                !c->validate_pipeline_bindings("strata_selftest_add", {}, {},
                                               binding_error) &&
                !binding_error.empty();
            check(active_slot_accepted && missing_slot_rejected,
                  "archive-hit pipeline retains its reflected binding schema",
                  binding_error);
        }
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
