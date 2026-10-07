// src/metal/program/version.cpp - the engine's identity (docs/mac/08 section 4.5).  The tags are literals in the
// binary: setup.py:engine_version() scans for `engine=`, tools/metal/check_asset.py for both.  STRATA_METAL_REV
// comes from METAL_REV for this file only, and the generated header changes only when strata.metallib does.
#include "metallib_sha256.inc"

namespace strata::metal {

[[gnu::used]] static const char kEngineTag[] = "engine=" STRATA_VERSION "\n";
[[gnu::used]] static const char kMetallibTag[] = "metallib_sha256=" STRATA_METALLIB_SHA256 "\n";

const char* engine_version() { return STRATA_VERSION; }
int metal_rev() { return STRATA_METAL_REV; }
const char* metallib_sha256() { return STRATA_METALLIB_SHA256; }

}  // namespace strata::metal
