// src/metal/host/pipeline_cache.mm - see include/strata/metal/pipeline_cache.hpp.
#include "strata/metal/pipeline_cache.hpp"

#import <Foundation/Foundation.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

namespace strata::metal {

static std::string os_build() {
    char b[64] = {};
    size_t n = sizeof b - 1;
    if (sysctlbyname("kern.osversion", b, &n, nullptr, 0) != 0) return "unknown";
    return b;
}

std::string pipeline_archive_path(const std::string& metallib_sha256, const std::string& gpu_name, std::string& why) {
    const char* env = std::getenv("STRATA_METAL_ARCHIVE");
    std::string dir;
    if (env != nullptr && std::string(env) == "0") { why = "off (STRATA_METAL_ARCHIVE=0)"; return {}; }
    if (env != nullptr && *env != '\0' && std::string(env) != "1") {
        dir = env;
    } else {
        NSArray<NSURL*>* u = [[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory
                                                                    inDomains:NSUserDomainMask];
        if (u.count == 0) { why = "no user cache folder"; return {}; }
        dir = std::string(u[0].path.UTF8String) + "/strata";
    }
    if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) { why = "cannot create " + dir; return {}; }
    std::string gpu = gpu_name;
    for (char& c : gpu)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-')) c = '_';
    return dir + "/pipelines-" + metallib_sha256.substr(0, 16) + "-" + gpu + "-" + os_build() + ".binarchive";
}

void prune_pipeline_archives(const std::string& keep) {
    @autoreleasepool {
        NSString* k = @(keep.c_str());
        NSString* dir = [k stringByDeletingLastPathComponent];
        NSFileManager* fm = [NSFileManager defaultManager];
        NSDate* old = [NSDate dateWithTimeIntervalSinceNow:-600];
        for (NSString* f in [fm contentsOfDirectoryAtPath:dir error:nil]) {
            NSString* p = [dir stringByAppendingPathComponent:f];
            if (![f hasPrefix:@"pipelines-"] || [p isEqualToString:k]) continue;
            if ([f containsString:@".tmp."]) {
                NSDate* m = [fm attributesOfItemAtPath:p error:nil].fileModificationDate;
                if (m == nil || [m compare:old] != NSOrderedAscending) continue;   // another engine may be writing it
            } else if (![f hasSuffix:@".binarchive"]) {
                continue;
            }
            [fm removeItemAtPath:p error:nil];
        }
    }
}

}  // namespace strata::metal
