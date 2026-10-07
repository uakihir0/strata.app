// tests/metal/metal_buffers_test.mm - shared buffers, arena alignment and no-copy validation.
#include "strata/metal/context.hpp"

#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

using namespace strata::metal;

static int failures = 0;

static void check(bool ok, const char* name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    failures += !ok;
}

int main() {
    std::uint64_t shifted_gpu_address = 123u;
    check(detail::gpu_address_with_offset(0u, 64u, shifted_gpu_address) &&
              shifted_gpu_address == 0u,
          "offset preserves unavailable GPU address");
    check(detail::gpu_address_with_offset(0x1000u, 64u, shifted_gpu_address) &&
              shifted_gpu_address == 0x1040u,
          "offset advances an available GPU address");
    check(!detail::gpu_address_with_offset(
              std::numeric_limits<std::uint64_t>::max() - 4u, 8u,
              shifted_gpu_address),
          "reject GPU address offset overflow");

    ContextOptions options;
    options.pipelines = false;
    std::string err;
    auto context = Context::open(options, err);
    check(context != nullptr, "open device context", err);
    if (!context) return 1;

    auto empty = context->buffers().allocate("empty", 0, err);
    check(!empty && err.find("empty") != std::string::npos, "reject zero-length allocation", err);

    auto too_large = context->buffers().allocate("too large", std::numeric_limits<std::size_t>::max(), err);
    check(!too_large && err.find("maxBufferLength") != std::string::npos, "reject allocation over maxBufferLength", err);

    auto shared = context->buffers().allocate("strata test shared", 512, err);
    check(static_cast<bool>(shared), "allocate shared buffer", err);
    BufferRef bounded;
    if (shared) {
        id<MTLBuffer> native = (__bridge id<MTLBuffer>)shared.handle;
        check(native.storageMode == MTLStorageModeShared, "buffer uses shared storage");
        check(native.hazardTrackingMode == MTLHazardTrackingModeUntracked, "buffer disables implicit hazard tracking");
        check(shared.gpu_address != 0 && shared.contents != nullptr, "buffer exposes CPU and GPU addresses");
        bounded = shared.slice(128, 256);
        check(bounded && context->buffers().owns(bounded) &&
                  bounded.offset == 128 &&
                  bounded.contents == static_cast<std::uint8_t*>(shared.contents) + 128 &&
                  bounded.gpu_address == shared.gpu_address + 128,
              "slice preserves tracked allocation metadata");
        auto widened = bounded;
        widened.offset = 0;
        widened.length = shared.length;
        widened.contents = shared.contents;
        widened.gpu_address = shared.gpu_address;
        check(!widened && !context->buffers().owns(widened),
              "reject coherently widened BufferRef slices");
        check(!shared.slice(500, 13),
              "reject slice extending beyond the tracked allocation");
        auto forged = shared;
        forged.contents = nullptr;
        check(!forged && !context->buffers().owns(forged),
              "reject a BufferRef with forged CPU contents");
        forged = shared;
        forged.gpu_address = 0;
        check(!forged && !context->buffers().owns(forged),
              "reject a BufferRef with forged GPU address");
        forged = shared;
        forged.offset = 1;
        check(!forged && !context->buffers().owns(forged),
              "reject a BufferRef with forged allocation offset");
        forged = shared;
        forged.length += 1;
        check(!forged && !context->buffers().owns(forged),
              "reject a BufferRef with forged allocation length");
    }

    BufferArena arena(context->buffers(), "strata test arena", 4096, err, true);
    check(arena.capacity() == 4096, "create poisoned arena", err);
    auto first = arena.allocate(13, 256, err);
    check(first && first.length == 13 && first.offset % 256 == 0, "allocate aligned first slice", err);
    if (first) {
        check(reinterpret_cast<std::uintptr_t>(first.contents) % 256 == 0, "CPU slice is aligned");
        check(first.gpu_address % 256 == 0, "GPU slice is aligned");
        check(static_cast<unsigned char*>(first.contents)[0] == 0xff, "poison is initialized once");
    }
    auto second = arena.allocate(64, 256, err);
    check(second && second.offset >= first.offset + first.length, "arena slices do not overlap", err);
    check(!arena.allocate(8, 24, err) && err.find("power of two") != std::string::npos,
          "reject invalid alignment", err);
    check(!arena.allocate(4096, 256, err) && err.find("out of space") != std::string::npos,
          "reject arena overflow", err);
    check(arena.used() <= arena.capacity(), "arena cursor stays within capacity");

    const long page = getpagesize();
    const std::size_t mapping_size = static_cast<std::size_t>(page) * 2;
    void* mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    check(mapping != MAP_FAILED, "create page-aligned test mapping");
    if (mapping != MAP_FAILED) {
        std::memset(mapping, 0x5a, mapping_size);
        auto window = context->buffers().wrap_no_copy("strata test window", mapping, mapping_size, err);
        check(window && window.length == mapping_size, "wrap aligned no-copy window", err);
        if (window) {
            check(static_cast<unsigned char*>(window.contents)[page] == 0x5a, "no-copy window sees mapped bytes");
            check(window.gpu_address % static_cast<std::uint64_t>(page) == 0, "no-copy GPU base is page aligned");
        }
        auto unaligned = context->buffers().wrap_no_copy(
            "unaligned", static_cast<char*>(mapping) + 1, mapping_size, err);
        check(!unaligned && err.find("aligned address and length") != std::string::npos,
              "reject unaligned no-copy base", err);
        auto bad_length = context->buffers().wrap_no_copy("unaligned length", mapping, mapping_size - 1, err);
        check(!bad_length && err.find("aligned address and length") != std::string::npos,
              "reject unaligned no-copy length", err);
    }

    void* owned_mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    bool owned_mapping_released = false;
    std::weak_ptr<void> observed_lifetime;
    check(owned_mapping != MAP_FAILED, "create owner-retained test mapping");
    if (owned_mapping != MAP_FAILED) {
        auto owner = std::shared_ptr<void>(owned_mapping, [&](void* address) {
            owned_mapping_released = munmap(address, mapping_size) == 0;
        });
        observed_lifetime = owner;
        auto owned_window = context->buffers().wrap_no_copy(
            "strata owned test window", owned_mapping, mapping_size, std::move(owner), err);
        check(owned_window && owned_window.length == mapping_size,
              "wrap no-copy window with retained mapping owner", err);
        check(!observed_lifetime.expired(), "Context retains the mapped-file owner");
    }

    const std::size_t partial_file_size = static_cast<std::size_t>(page) - 10;
    char file_name[] = "/tmp/strata-metal-eof.XXXXXX";
    const int file_descriptor = mkstemp(file_name);
    void* file_mapping = MAP_FAILED;
    bool file_mapping_released = false;
    std::weak_ptr<void> observed_file_lifetime;
    if (file_descriptor >= 0) {
        unlink(file_name);
        if (ftruncate(file_descriptor, static_cast<off_t>(partial_file_size)) == 0) {
            const unsigned char marker = 0x7c;
            if (pwrite(file_descriptor, &marker, 1, static_cast<off_t>(partial_file_size - 1)) == 1) {
                file_mapping = mmap(nullptr, partial_file_size, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
            }
        }
        close(file_descriptor);
    }
    check(file_mapping != MAP_FAILED, "map a file whose size ends inside the final VM page");
    if (file_mapping != MAP_FAILED) {
        auto owner = std::shared_ptr<void>(file_mapping, [&](void* address) {
            file_mapping_released = munmap(address, partial_file_size) == 0;
        });
        observed_file_lifetime = owner;
        auto final_page = context->buffers().wrap_no_copy(
            "strata mapped-file final page", file_mapping, static_cast<std::size_t>(page),
            std::move(owner), err);
        check(final_page && final_page.length == static_cast<std::size_t>(page),
              "wrap the final page-rounded range of a short file", err);
        if (final_page) {
            const auto* bytes = static_cast<const unsigned char*>(final_page.contents);
            check(bytes[partial_file_size - 1] == 0x7c && bytes[partial_file_size] == 0,
                  "final page exposes file bytes and zero-filled mmap padding");
        }
        check(!observed_file_lifetime.expired(), "Context retains the short file mapping");
    }

    const std::size_t before_rollback = context->buffers().allocation_count();
    const std::size_t transient_mapping_size = static_cast<std::size_t>(page);
    void* transient_mapping =
        mmap(nullptr, transient_mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    bool transient_mapping_released = false;
    std::weak_ptr<void> transient_lifetime;
    check(transient_mapping != MAP_FAILED, "create transactional no-copy mapping");
    if (transient_mapping != MAP_FAILED) {
        {
            auto transaction = context->buffers().transaction();
            auto transient = context->buffers().allocate("transactional allocation", 256, err);
            check(static_cast<bool>(transient), "allocate inside buffer transaction", err);
            auto owner = std::shared_ptr<void>(transient_mapping, [&](void* address) {
                transient_mapping_released =
                    munmap(address, transient_mapping_size) == 0;
            });
            transient_lifetime = owner;
            auto transient_window = context->buffers().wrap_no_copy(
                "transactional window", transient_mapping, transient_mapping_size,
                std::move(owner), err);
            check(static_cast<bool>(transient_window),
                  "wrap mapping inside buffer transaction", err);
            check(context->buffers().commit_residency(err),
                  "commit provisional residency before rollback", err);
            check(context->buffers().allocation_count() == before_rollback + 2,
                  "transaction sees its provisional allocations");
        }
        check(context->buffers().allocation_count() == before_rollback,
              "failed buffer transaction removes its residency allocations");
        check(transient_lifetime.expired() && transient_mapping_released,
              "rollback releases no-copy Metal handles before their mapping owner");
    }

    const std::size_t expected_minimum = 2 + (mapping == MAP_FAILED ? 0 : 1) +
                                         (owned_mapping == MAP_FAILED ? 0 : 1) +
                                         (file_mapping == MAP_FAILED ? 0 : 1);
    check(context->buffers().allocation_count() >= expected_minimum,
          "buffers are registered for residency");
    check(context->buffers().commit_residency(err), "commit and request residency", err);
    check(context->buffers().allocated_size() > 0, "residency set reports its allocation");
    context.reset();
    check(!shared && !bounded,
          "Context destruction invalidates copied BufferRef views");
    check(observed_lifetime.expired() && owned_mapping_released,
          "release no-copy buffers before their mapped-file owner");
    check(observed_file_lifetime.expired() && file_mapping_released,
          "release a page-rounded file window before its mmap");
    if (mapping != MAP_FAILED) check(munmap(mapping, mapping_size) == 0, "release external mapping");

    std::printf("metal_buffers_test: %s\n", failures ? "FAILED" : "OK");
    return failures != 0;
}
