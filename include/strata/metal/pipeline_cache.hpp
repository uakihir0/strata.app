// include/strata/metal/pipeline_cache.hpp - the on-disk MTLBinaryArchive that makes the second start fast
// (docs/mac/08 section 10.1).  Every failure here is logged and ignored: without the archive the pipelines compile.
#pragma once

#include <string>

namespace strata::metal {

/// The archive path for this engine, GPU and macOS build, or empty with `why` set when the cache is off
/// (STRATA_METAL_ARCHIVE=0) or no folder can be made.  STRATA_METAL_ARCHIVE unset or 1: ~/Library/Caches/strata;
/// any other value: that folder.
std::string pipeline_archive_path(const std::string& metallib_sha256, const std::string& gpu_name, std::string& why);

/// Remove the other archives in the folder of `keep` and leftover `.tmp.*` files older than ten minutes.
void prune_pipeline_archives(const std::string& keep);

}  // namespace strata::metal
