// src/metal/program/main.cpp - skeleton driver: --version, --list-devices, --metallib, --metal-capture.
#include "strata/metal/context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace strata::metal {
const char* engine_version();
int metal_rev();
const char* metallib_sha256();
}  // namespace strata::metal

using namespace strata::metal;

int main(int argc, char** argv) {
    ContextOptions o;
    o.metallib_sha256 = metallib_sha256();
    if (const char* c = std::getenv("STRATA_METAL_CAPTURE"); c != nullptr && *c != '\0') o.capture = c;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--version") {
            std::printf("strata %s metal %d\n", engine_version(), metal_rev());
            return 0;
        } else if (a == "--list-devices") {
            list = true;
        } else if (a == "--metallib" && i + 1 < argc) {
            o.metallib = argv[++i];
            o.metallib_sha256.clear();
            std::fprintf(stderr, "strata: --metallib: %s is not checked against this engine\n", o.metallib.c_str());
        } else if (a == "--metal-capture" && i + 1 < argc) {
            o.capture = argv[++i];
        } else {
            std::fprintf(stderr, "usage: strata [--version | --list-devices] [--metallib PATH] [--metal-capture F]\n");
            return 2;
        }
    }
    if (!o.capture.empty()) {
        if (o.capture.size() < 9 || o.capture.compare(o.capture.size() - 9, 9, ".gputrace") != 0) o.capture += ".gputrace";
        if (access(o.capture.c_str(), F_OK) == 0) {
            std::fprintf(stderr, "strata: %s exists; Metal does not overwrite a trace\n", o.capture.c_str());
            return 2;
        }
    }
    o.pipelines = !list;
    std::string err;
    auto ctx = Context::open(o, err);
    if (!ctx) { std::fprintf(stderr, "strata: %s\n", err.c_str()); return 1; }
    if (list) { std::printf("device %s\n", ctx->device_name().c_str()); return 0; }
    const PipelineStats& s = ctx->pipeline_stats();
    std::fprintf(stderr, "strata: %u pipelines (%u from the cache, %u compiled) in %.1f ms; cache: %s\n",
                 s.from_archive + s.compiled, s.from_archive, s.compiled, s.ms, s.archive.c_str());
    if (!ctx->begin_capture(err)) { std::fprintf(stderr, "strata: %s\n", err.c_str()); return 1; }
    const bool ok = ctx->run_selftest(err);   // the skeleton's stand-in for the first decode window
    ctx->end_capture();
    if (!ok) { std::fprintf(stderr, "strata: %s\n", err.c_str()); return 1; }
    return 0;
}
