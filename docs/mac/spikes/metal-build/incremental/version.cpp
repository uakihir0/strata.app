#include "metallib_sha256.inc"
[[gnu::used]] static const char kTag[] = "metallib_sha256=" STRATA_METALLIB_SHA256 "\n";
const char* metallib_sha256() { return STRATA_METALLIB_SHA256; }
