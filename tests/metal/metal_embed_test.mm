// Metal twin for embedding gather, image rows and stream broadcast (docs/mac/04 §5.2, §6.7).
#include "metallib_sha256.inc"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/embed_dispatch.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/prefill_dispatch.hpp"
#include "twin.hpp"

#include <ggml.h>
#include "../../src/metal/kernels/common/types.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kEmbeddingWidth = 2560u;
constexpr std::uint32_t kStreamCount = 4u;
constexpr std::uint32_t kStreamWidth = kEmbeddingWidth * kStreamCount;
constexpr std::uint32_t kImageToken = 0x80000000u;
constexpr std::uint32_t kPoisonByte = 0xa5u;
constexpr std::uint32_t kMaxShaderSafeImageRows =
    static_cast<std::uint32_t>((std::uint64_t{1u} << 32u) /
                               kEmbeddingWidth);
constexpr std::uint32_t kMaxShaderSafeBroadcastRows =
    static_cast<std::uint32_t>((std::uint64_t{1u} << 32u) /
                               kStreamWidth);

struct Format {
    const char* label;
    ggml_type type;
    std::size_t d_offset;
    EmbedRowsFormat encoder_format;
};

constexpr std::array<Format, 4> kFormats{{
    {"q3_k", GGML_TYPE_Q3_K, offsetof(block_q3_K, d),
     EmbedRowsFormat::q3_k},
    {"iq3_s", GGML_TYPE_IQ3_S, offsetof(block_iq3_s, d),
     EmbedRowsFormat::iq3_s},
    {"iq4_xs", GGML_TYPE_IQ4_XS, offsetof(block_iq4_xs, d),
     EmbedRowsFormat::iq4_xs},
    {"bf16", GGML_TYPE_BF16, 0u, EmbedRowsFormat::bf16},
}};

enum class ImagePattern { row_zero, after_text };

constexpr std::array<std::uint32_t, 7> kTokens{
    5u, 1u, 4u, kImageToken, 2u, 5u, 0u};
constexpr std::uint32_t kTableRows = 6u;

int g_failures = 0;

void check(bool condition, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

bool concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
}

bool wait_for_completion(CommandBuffer& command, const std::string& label,
                         std::string& error) {
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command.unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable after commit";
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = label + " command buffer failed";
        if (native.error.localizedDescription.UTF8String != nullptr) {
            error += ": ";
            error += native.error.localizedDescription.UTF8String;
        }
        return false;
    }
    error.clear();
    return true;
}

bool submit_and_wait(CommandBuffer& command, const std::string& label,
                     std::string& error) {
    return command.commit(error) && wait_for_completion(command, label, error);
}

BufferRef subview(BufferRef buffer, std::size_t offset, std::size_t length) {
    return buffer.slice(offset, length);
}

std::uint8_t next_byte(std::mt19937& rng, twin::InputSet set, std::size_t index,
                       std::size_t row) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<std::uint8_t, 8> edge{
            0x00u, 0xffu, 0x55u, 0xaau, 0x0fu, 0xf0u, 0x01u, 0x80u};
        return edge[(index / 16u + row) % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        return static_cast<std::uint8_t>(index * 37u + row * 53u + (index >> 3u));
    }
    return static_cast<std::uint8_t>(rng() & 0xffu);
}

bool fill_table(const Format& format, const ggml_type_traits& traits,
                twin::InputSet set, std::uint64_t seed,
                std::vector<std::uint8_t>& table, std::size_t& row_bytes,
                std::string& err) {
    if (traits.blck_size <= 0 || traits.type_size <= 0 ||
        kEmbeddingWidth % static_cast<std::uint32_t>(traits.blck_size) != 0u ||
        traits.to_float == nullptr) {
        err = std::string("ggml traits are incomplete for ") + format.label;
        return false;
    }
    const std::size_t blocks_per_row =
        kEmbeddingWidth / static_cast<std::uint32_t>(traits.blck_size);
    row_bytes = blocks_per_row * traits.type_size;
    if (row_bytes > SIZE_MAX / kTableRows) {
        err = "embedding fixture size overflow";
        return false;
    }
    table.resize(row_bytes * kTableRows);

    std::mt19937 rng(static_cast<std::uint32_t>(seed) ^
                     static_cast<std::uint32_t>(format.type * 0x9e3779b9u));
    for (std::size_t row = 0; row < kTableRows; ++row) {
        std::uint8_t* row_data = table.data() + row * row_bytes;
        for (std::size_t byte = 0; byte < row_bytes; ++byte) {
            row_data[byte] = next_byte(rng, set, byte, row);
        }

        if (format.type == GGML_TYPE_BF16) {
            constexpr std::array<std::uint16_t, 8> bf16_edges{
                0x0000u, 0x8000u, 0x3f80u, 0xbf80u,
                0x3f00u, 0xbf00u, 0x7f7fu, 0xff7fu};
            for (std::uint32_t value = 0; value < kEmbeddingWidth; ++value) {
                std::uint16_t bits = 0;
                if (set == twin::InputSet::edge) {
                    bits = bf16_edges[(value + row) % bf16_edges.size()];
                } else if (set == twin::InputSet::fixture) {
                    bits = static_cast<std::uint16_t>(
                        0x3f00u | ((value * 13u + row * 29u) & 0x007fu));
                    if (((value + row) & 1u) != 0u) bits |= 0x8000u;
                } else {
                    bits = static_cast<std::uint16_t>(
                        0x3f00u | (rng() & 0x007fu));
                    if ((rng() & 1u) != 0u) bits |= 0x8000u;
                }
                std::memcpy(row_data + 2u * value, &bits, sizeof(bits));
            }
        } else {
            for (std::size_t block = 0; block < blocks_per_row; ++block) {
                // Keep d finite, normal and modest while exercising varying exponents and signs.
                std::uint16_t d = static_cast<std::uint16_t>(
                    0x2000u | ((row * 97u + block * 71u + seed) & 0x03ffu));
                if (set == twin::InputSet::edge) d = (block & 1u) == 0u ? 0x3800u : 0xb800u;
                std::memcpy(row_data + block * traits.type_size + format.d_offset,
                            &d, sizeof(d));
            }
        }
    }
    err.clear();
    return true;
}

bool test_embed_format(Context& context, const Format& format, std::uint32_t rows,
                       twin::InputSet set, const twin::Options& options) {
    const std::string label = std::string(format.label) + " T=" +
        std::to_string(rows) + " set=" + twin::input_set_name(set);
    const ggml_type_traits* traits = ggml_get_type_traits(format.type);
    if (traits == nullptr) {
        check(false, label + " ggml type traits");
        return false;
    }

    std::vector<std::uint8_t> table;
    std::size_t row_bytes = 0;
    std::string err;
    check(fill_table(format, *traits, set, options.seed, table, row_bytes, err),
          label + " input table", err);
    if (table.empty()) return false;

    const std::size_t table_bytes = table.size();
    const std::size_t token_bytes = rows * sizeof(std::uint32_t);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth * sizeof(float);
    twin::GuardedBuffer table_buffer;
    twin::GuardedBuffer token_buffer;
    twin::GuardedBuffer output_buffer;
    check(table_buffer.allocate(context, label + " table", table_bytes, err),
          label + " table buffer", err);
    check(token_buffer.allocate(context, label + " tokens", token_bytes, err),
          label + " token buffer", err);
    check(output_buffer.allocate(context, label + " output", output_bytes, err),
          label + " output buffer", err);
    if (!table_buffer.view() || !token_buffer.view() || !output_buffer.view()) return false;

    std::memcpy(table_buffer.payload().data(), table.data(), table_bytes);
    std::memcpy(token_buffer.payload().data(), kTokens.data(), token_bytes);
    std::fill(output_buffer.payload().begin(), output_buffer.payload().end(),
              static_cast<std::uint8_t>(kPoisonByte));
    table_buffer.capture_before_dispatch();
    token_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    std::vector<std::uint8_t> expected(output_bytes,
                                       static_cast<std::uint8_t>(kPoisonByte));
    std::vector<twin::ByteRange> written;
    std::vector<float> decoded(kEmbeddingWidth);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t token = kTokens[row];
        if ((token & kImageToken) != 0u) continue;
        traits->to_float(table.data() + static_cast<std::size_t>(token) * row_bytes,
                         decoded.data(), kEmbeddingWidth);
        const std::size_t output_offset =
            static_cast<std::size_t>(row) * kEmbeddingWidth * sizeof(float);
        std::memcpy(expected.data() + output_offset, decoded.data(),
                    kEmbeddingWidth * sizeof(float));
        written.push_back({output_offset, kEmbeddingWidth * sizeof(float)});
    }

    auto command = context.make_command_buffer(label, err);
    check(command != nullptr, label + " command buffer", err);
    if (!command) return false;
    const EmbedRowsBuffers buffers{table_buffer.view(), token_buffer.view(),
                                   output_buffer.view()};
    const bool encoded = encode_embed_rows(
        context, *command, rows, kTableRows, format.encoder_format, buffers,
        err);
    check(encoded, label + " production embedding-row encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(*command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto mismatch = twin::compare_bytes(output_buffer.payload(), expected);
    if (mismatch) {
        twin::report_failure(label.c_str(), options, rows, *mismatch);
    }
    check(!mismatch, label + " exact ggml CPU embedding bytes");
    check(!table_buffer.first_unwritten_change({}),
          label + " table remains read-only");
    check(!token_buffer.first_unwritten_change({}),
          label + " tokens remain read-only");
    const auto changed = output_buffer.first_unwritten_change(written);
    check(!changed, label + " image row and output guards remain unchanged",
          changed ? changed->reason : "");
    return !mismatch && !changed;
}

bool test_embed_rows_encoder_validation(Context& context,
                                        const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t vocabulary_size = kTableRows;
    constexpr std::size_t table_bytes = vocabulary_size * 1100u;
    constexpr std::size_t token_bytes = rows * sizeof(std::uint32_t);
    constexpr std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth * sizeof(float);
    std::string error;

    twin::GuardedBuffer table_storage;
    twin::GuardedBuffer token_storage;
    twin::GuardedBuffer embedding_storage;
    check(table_storage.allocate(context, "embedding rows validation table",
                                 table_bytes + 16u, error),
          "allocate embedding rows table storage", error);
    check(token_storage.allocate(context, "embedding rows validation tokens",
                                 token_bytes + 16u, error),
          "allocate embedding rows token storage", error);
    check(embedding_storage.allocate(context,
                                    "embedding rows validation output",
                                    embedding_bytes + 16u, error),
          "allocate embedding rows output storage", error);
    if (!table_storage.view() || !token_storage.view() ||
        !embedding_storage.view()) {
        return false;
    }

    std::memset(table_storage.payload().data(), 0, table_bytes);
    const std::array<std::uint32_t, rows> valid_tokens{5u, 1u};
    std::memcpy(token_storage.payload().data(), valid_tokens.data(),
                token_bytes);
    const EmbedRowsBuffers valid{
        subview(table_storage.view(), 0u, table_bytes),
        subview(token_storage.view(), 0u, token_bytes),
        subview(embedding_storage.view(), 0u, embedding_bytes),
    };
    auto command =
        context.make_command_buffer("embedding rows encoder validation", error);
    check(command != nullptr, "create embedding rows validation command",
          error);
    if (!command) return false;

    bool passed = true;
    const auto rejects = [&](std::uint32_t row_count,
                             std::uint32_t vocab_size,
                             EmbedRowsFormat format,
                             const EmbedRowsBuffers& buffers,
                             std::string_view expected_error,
                             const std::string& label) {
        std::string detail;
        const bool rejected = !encode_embed_rows(
            context, *command, row_count, vocab_size, format, buffers, detail);
        const bool reason_matches =
            expected_error.empty() ||
            detail.find(expected_error) != std::string::npos;
        check(rejected && reason_matches, label, detail);
        passed &= rejected && reason_matches;
    };

    rejects(0u, vocabulary_size, EmbedRowsFormat::q3_k, valid, "row count",
            "reject zero embedding rows");
    rejects(kMaxShaderSafeImageRows + 1u, vocabulary_size,
            EmbedRowsFormat::q3_k, valid, "shader index",
            "reject embedding rows that overflow the shader index");
    rejects(rows, 0u, EmbedRowsFormat::q3_k, valid, "vocabulary size",
            "reject empty embedding vocabulary");
    rejects(rows, 0x80000001u, EmbedRowsFormat::q3_k, valid,
            "vocabulary size", "reject vocabulary colliding with image flag");
    rejects(rows, vocabulary_size, static_cast<EmbedRowsFormat>(99u), valid,
            "unsupported", "reject unsupported embedding format");

    auto short_table = valid;
    short_table.table.length -= 2u;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, short_table,
            "undersized", "reject undersized embedding table view");
    auto short_tokens = valid;
    short_tokens.tokens.length -= sizeof(std::uint32_t);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, short_tokens,
            "undersized", "reject undersized embedding token view");
    auto short_embedding = valid;
    short_embedding.embedding.length -= kEmbeddingWidth * sizeof(float);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, short_embedding,
            "undersized", "reject undersized embedding output view");
    rejects(rows, vocabulary_size + 1u, EmbedRowsFormat::q3_k, valid,
            "undersized", "reject vocabulary larger than the table view");

    auto misaligned_table = valid;
    misaligned_table.table =
        subview(table_storage.view(), 1u, table_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, misaligned_table,
            "misaligned", "reject misaligned embedding table view");
    auto misaligned_tokens = valid;
    misaligned_tokens.tokens =
        subview(token_storage.view(), 1u, token_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, misaligned_tokens,
            "misaligned", "reject misaligned embedding token view");
    auto misaligned_embedding = valid;
    misaligned_embedding.embedding =
        subview(embedding_storage.view(), 1u, embedding_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            misaligned_embedding, "misaligned",
            "reject misaligned embedding output view");

    BufferRef out_of_bounds_storage = context.buffers().allocate(
        "embedding rows out-of-allocation table", table_bytes + 16u, error);
    check(static_cast<bool>(out_of_bounds_storage),
          "allocate embedding rows out-of-allocation table", error);
    if (!out_of_bounds_storage) return false;
    auto out_of_bounds = valid;
    out_of_bounds.table =
        subview(out_of_bounds_storage, table_bytes + 14u, table_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, out_of_bounds,
            "not owned by this Context",
            "reject out-of-allocation embedding table slice");

    const std::uint32_t invalid_token = vocabulary_size;
    std::memcpy(token_storage.payload().data(), &invalid_token,
                sizeof(invalid_token));
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, valid,
            "outside the vocabulary",
            "reject non-image token outside the vocabulary");
    std::memcpy(token_storage.payload().data(), valid_tokens.data(),
                token_bytes);

    BufferRef shared_output_storage = context.buffers().allocate(
        "embedding rows table/output overlap",
        embedding_bytes + table_bytes + 16u, error);
    check(static_cast<bool>(shared_output_storage),
          "allocate embedding rows table/output overlap", error);
    if (!shared_output_storage) return false;
    auto table_output_overlap = valid;
    table_output_overlap.table =
        subview(shared_output_storage, 0u, table_bytes);
    table_output_overlap.embedding =
        subview(shared_output_storage, 0u, embedding_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            table_output_overlap, "must not overlap",
            "reject embedding output containing the table view");
    auto partial_table_output_overlap = valid;
    partial_table_output_overlap.table =
        subview(shared_output_storage, embedding_bytes - table_bytes / 2u,
                table_bytes);
    partial_table_output_overlap.embedding =
        subview(shared_output_storage, 0u, embedding_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            partial_table_output_overlap, "must not overlap",
            "reject partial embedding table/output overlap");

    BufferRef shared_token_output_storage = context.buffers().allocate(
        "embedding rows token/output overlap", embedding_bytes + 16u, error);
    check(static_cast<bool>(shared_token_output_storage),
          "allocate embedding rows token/output overlap", error);
    if (!shared_token_output_storage) return false;
    auto token_output_overlap = valid;
    token_output_overlap.tokens =
        subview(shared_token_output_storage, 0u, token_bytes);
    token_output_overlap.embedding =
        subview(shared_token_output_storage, 0u, embedding_bytes);
    std::memcpy(token_output_overlap.tokens.contents, valid_tokens.data(),
                token_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            token_output_overlap, "must not overlap",
            "reject embedding output containing the token view");
    auto partial_token_output_overlap = valid;
    partial_token_output_overlap.tokens =
        subview(shared_token_output_storage, 0u, token_bytes);
    partial_token_output_overlap.embedding =
        subview(shared_token_output_storage, sizeof(std::uint32_t),
                embedding_bytes);
    std::memcpy(partial_token_output_overlap.tokens.contents,
                valid_tokens.data(), token_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            partial_token_output_overlap, "must not overlap",
            "reject partial embedding token/output overlap");

    auto same_handle_alias_without_gpu_address = table_output_overlap;
    same_handle_alias_without_gpu_address.table.gpu_address = 0u;
    same_handle_alias_without_gpu_address.embedding.gpu_address = 0u;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            same_handle_alias_without_gpu_address,
            "not owned by this Context",
            "reject forged GPU metadata on embedding table/output views");

    // Both inputs are read-only, so sharing their backing allocation is safe.
    BufferRef read_only_backing = context.buffers().allocate(
        "embedding rows read-only table/token alias", table_bytes, error);
    BufferRef read_only_output = context.buffers().allocate(
        "embedding rows read-only alias output", embedding_bytes, error);
    check(static_cast<bool>(read_only_backing) &&
              static_cast<bool>(read_only_output),
          "allocate embedding rows read-only alias buffers", error);
    if (!read_only_backing || !read_only_output) return false;
    std::memset(read_only_backing.contents, 0, table_bytes);
    std::memcpy(read_only_backing.contents, valid_tokens.data(), token_bytes);
    const EmbedRowsBuffers read_only_alias{
        subview(read_only_backing, 0u, table_bytes),
        subview(read_only_backing, 0u, token_bytes),
        subview(read_only_output, 0u, embedding_bytes),
    };
    auto read_only_command = context.make_command_buffer(
        "embedding rows read-only input alias", error);
    check(read_only_command != nullptr,
          "create embedding rows read-only alias command", error);
    if (!read_only_command) return false;
    const bool accepts_read_only_alias = encode_embed_rows(
        context, *read_only_command, rows, vocabulary_size,
        EmbedRowsFormat::q3_k, read_only_alias, error);
    check(accepts_read_only_alias,
          "allow overlapping read-only embedding table/token inputs", error);
    passed &= accepts_read_only_alias;
    if (accepts_read_only_alias) {
        const bool completes = submit_and_wait(
            *read_only_command, "embedding rows read-only alias", error);
        check(completes,
              "embedding rows read-only alias command completes", error);
        passed &= completes;
    }

    // The high bit marks an image row; its lower bits are not a vocabulary id.
    const std::array<std::uint32_t, rows> image_marked_tokens{
        kImageToken | 0x7fffffffu, 1u};
    std::memcpy(token_storage.payload().data(), image_marked_tokens.data(),
                token_bytes);
    auto image_row_command =
        context.make_command_buffer("embedding rows image-flag validation",
                                    error);
    check(image_row_command != nullptr,
          "create embedding rows image-flag command", error);
    if (!image_row_command) return false;
    const bool accepts_image_flag = encode_embed_rows(
        context, *image_row_command, rows, vocabulary_size,
        EmbedRowsFormat::q3_k, valid, error);
    check(accepts_image_flag,
          "skip vocabulary validation for image-marked embedding rows", error);
    passed &= accepts_image_flag;
    if (accepts_image_flag) {
        const bool completes = submit_and_wait(
            *image_row_command, "embedding rows image-flag validation", error);
        check(completes,
              "embedding rows image-flag command completes", error);
        passed &= completes;
    }
    std::memcpy(token_storage.payload().data(), valid_tokens.data(),
                token_bytes);

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "query page size for embedding cross-handle alias test");
        return false;
    }
    const std::size_t page_size = static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_bytes =
        ((embedding_bytes + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "map anonymous storage for embedding cross-handle alias");
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef shared_table_handle = context.buffers().wrap_no_copy(
        "embedding shared table backing", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_output_handle = context.buffers().wrap_no_copy(
        "embedding shared output backing", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    check(static_cast<bool>(shared_table_handle) &&
              static_cast<bool>(shared_output_handle),
          "wrap embedding backing in distinct Metal handles", error);
    if (!shared_table_handle || !shared_output_handle) return false;
    auto cross_handle_alias = valid;
    cross_handle_alias.table =
        subview(shared_table_handle, 0u, table_bytes);
    cross_handle_alias.embedding =
        subview(shared_output_handle, 0u, embedding_bytes);
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, cross_handle_alias,
            "must not overlap",
            "reject embedding alias across distinct Metal handles");
    auto cross_handle_missing_metadata = cross_handle_alias;
    cross_handle_missing_metadata.table.contents = nullptr;
    cross_handle_missing_metadata.table.gpu_address = 0u;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k,
            cross_handle_missing_metadata, "not owned by this Context",
            "reject forged CPU/GPU metadata on embedding view");

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign embedding Context", error);
    if (!foreign_context) return false;
    BufferRef foreign_table = foreign_context->buffers().allocate(
        "foreign embedding table", table_bytes, error);
    BufferRef foreign_tokens = foreign_context->buffers().allocate(
        "foreign embedding tokens", token_bytes, error);
    BufferRef foreign_output = foreign_context->buffers().allocate(
        "foreign embedding output", embedding_bytes, error);
    check(static_cast<bool>(foreign_table) &&
              static_cast<bool>(foreign_tokens) &&
              static_cast<bool>(foreign_output),
          "allocate foreign embedding buffers", error);
    if (!foreign_table || !foreign_tokens || !foreign_output) return false;
    auto foreign_table_view = valid;
    foreign_table_view.table = foreign_table;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, foreign_table_view,
            "not owned by this Context", "reject foreign embedding table");
    auto foreign_token_view = valid;
    foreign_token_view.tokens = foreign_tokens;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, foreign_token_view,
            "not owned by this Context", "reject foreign embedding tokens");
    auto foreign_output_view = valid;
    foreign_output_view.embedding = foreign_output;
    rejects(rows, vocabulary_size, EmbedRowsFormat::q3_k, foreign_output_view,
            "not owned by this Context", "reject foreign embedding output");

    auto foreign_command = foreign_context->make_command_buffer(
        "foreign embedding rows command", error);
    check(foreign_command != nullptr, "create foreign embedding command",
          error);
    if (!foreign_command) return false;
    std::string foreign_error;
    const bool rejects_foreign_command =
        !encode_embed_rows(context, *foreign_command, rows, vocabulary_size,
                           EmbedRowsFormat::q3_k, valid, foreign_error) &&
        foreign_error.find("different Context") != std::string::npos;
    check(rejects_foreign_command,
          "reject foreign embedding command buffer", foreign_error);
    passed &= rejects_foreign_command;
    foreign_command->discard();

    auto enqueued = context.make_command_buffer(
        "enqueued embedding rows command", error);
    check(enqueued != nullptr, "create enqueued embedding rows command", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "enqueue embedding rows validation command", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    std::string enqueued_error;
    const bool rejects_enqueued =
        !encode_embed_rows(context, *enqueued, rows, vocabulary_size,
                           EmbedRowsFormat::q3_k, valid, enqueued_error) &&
        enqueued_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_enqueued, "reject enqueued embedding command buffer",
          enqueued_error);
    passed &= rejects_enqueued;
    const bool enqueued_completes =
        submit_and_wait(*enqueued, "enqueued embedding rows", error);
    check(enqueued_completes, "enqueued embedding rows command completes",
          error);
    passed &= enqueued_completes;

    auto committed = context.make_command_buffer(
        "committed embedding rows command", error);
    check(committed != nullptr, "create committed embedding rows command",
          error);
    if (!committed) return false;
    const bool commit_ok = committed->commit(error);
    check(commit_ok, "commit embedding rows validation command", error);
    if (!commit_ok) return false;
    std::string committed_error;
    const bool rejects_committed =
        !encode_embed_rows(context, *committed, rows, vocabulary_size,
                           EmbedRowsFormat::q3_k, valid, committed_error) &&
        committed_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_committed, "reject committed embedding command buffer",
          committed_error);
    passed &= rejects_committed;
    const bool committed_completes =
        wait_for_completion(*committed, "committed embedding rows", error);
    check(committed_completes,
          "committed embedding rows command completes", error);
    passed &= committed_completes;
    return passed;
}

bool test_gr_broadcast(Context& context, std::uint32_t rows, twin::InputSet set,
                       const twin::Options& options) {
    const std::string label = "gr_broadcast T=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t input_count = static_cast<std::size_t>(rows) * kEmbeddingWidth;
    const std::size_t output_count = static_cast<std::size_t>(rows) * kStreamWidth;
    const std::size_t input_bytes = input_count * sizeof(std::uint32_t);
    const std::size_t output_bytes = output_count * sizeof(std::uint32_t);

    std::vector<std::uint32_t> input(input_count);
    std::vector<std::uint32_t> expected(output_count);
    std::mt19937 rng(static_cast<std::uint32_t>(options.seed) ^ (rows * 0x85ebca6bu));
    constexpr std::array<std::uint32_t, 8> edge{
        0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x7fc12345u, 0x7f800000u, 0x00000001u, 0xffffffffu};
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t i = 0; i < kEmbeddingWidth; ++i) {
            const std::size_t index =
                static_cast<std::size_t>(row) * kEmbeddingWidth + i;
            if (set == twin::InputSet::edge) {
                input[index] = edge[(i + 3u * row) % edge.size()];
            } else if (set == twin::InputSet::fixture) {
                input[index] = 0x9e3779b9u * (i + 1u) ^ (row * 0x7f4a7c15u);
            } else {
                input[index] = rng();
            }
        }
        for (std::uint32_t i = 0; i < kStreamWidth; ++i) {
            expected[static_cast<std::size_t>(row) * kStreamWidth + i] =
                input[static_cast<std::size_t>(row) * kEmbeddingWidth +
                      (i % kEmbeddingWidth)];
        }
    }

    twin::GuardedBuffer input_buffer;
    twin::GuardedBuffer output_buffer;
    std::string err;
    check(input_buffer.allocate(context, label + " input", input_bytes, err),
          label + " input buffer", err);
    check(output_buffer.allocate(context, label + " output", output_bytes, err),
          label + " output buffer", err);
    if (!input_buffer.view() || !output_buffer.view()) return false;
    std::memcpy(input_buffer.payload().data(), input.data(), input_bytes);
    std::fill(output_buffer.payload().begin(), output_buffer.payload().end(),
              static_cast<std::uint8_t>(kPoisonByte));
    input_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    check(command != nullptr, label + " command buffer", err);
    if (!command) return false;
    const GRBroadcastBuffers buffers{input_buffer.view(),
                                     output_buffer.view()};
    const bool encoded =
        encode_gr_broadcast(context, *command, rows, buffers, err);
    check(encoded, label + " production GR broadcast encode", err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(*command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), output_bytes);
    const auto mismatch = twin::compare_bytes(output_buffer.payload(), expected_bytes);
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    check(!mismatch, label + " exact 4-stream broadcast bits");
    check(!input_buffer.first_unwritten_change({}),
          label + " embedding input and guards remain unchanged");
    const std::array<twin::ByteRange, 1> written{{{0u, output_bytes}}};
    const auto changed = output_buffer.first_unwritten_change(written);
    check(!changed, label + " output guards remain unchanged",
          changed ? changed->reason : "");
    return !mismatch && !changed;
}

bool test_gr_broadcast_encoder_validation(
    Context& context, const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth *
        sizeof(std::uint32_t);
    constexpr std::size_t streams_bytes =
        static_cast<std::size_t>(rows) * kStreamWidth *
        sizeof(std::uint32_t);
    std::string error;

    twin::GuardedBuffer embedding_storage;
    twin::GuardedBuffer streams_storage;
    check(embedding_storage.allocate(context, "GR broadcast validation input",
                                     embedding_bytes + 16u, error),
          "allocate GR broadcast input storage", error);
    check(streams_storage.allocate(context, "GR broadcast validation output",
                                   streams_bytes + 16u, error),
          "allocate GR broadcast output storage", error);
    if (!embedding_storage.view() || !streams_storage.view()) return false;

    const GRBroadcastBuffers valid{
        subview(embedding_storage.view(), 0u, embedding_bytes),
        subview(streams_storage.view(), 0u, streams_bytes),
    };
    auto command =
        context.make_command_buffer("GR broadcast encoder validation", error);
    check(command != nullptr, "create GR broadcast validation command", error);
    if (!command) return false;

    bool passed = true;
    const auto rejects = [&](std::uint32_t row_count,
                             const GRBroadcastBuffers& buffers,
                             std::string_view expected_error,
                             const std::string& label) {
        std::string detail;
        const bool rejected = !encode_gr_broadcast(
            context, *command, row_count, buffers, detail);
        const bool reason_matches =
            expected_error.empty() ||
            detail.find(expected_error) != std::string::npos;
        check(rejected && reason_matches, label, detail);
        passed &= rejected && reason_matches;
    };

    rejects(0u, valid, "shader index", "reject zero GR broadcast rows");
    rejects(kMaxShaderSafeBroadcastRows + 1u, valid, "shader index",
            "reject GR broadcast rows that overflow the shader index");

    auto short_embedding = valid;
    short_embedding.embedding.length -= sizeof(std::uint32_t);
    rejects(rows, short_embedding, "undersized",
            "reject undersized GR broadcast embedding view");
    auto short_streams = valid;
    short_streams.streams.length -= sizeof(std::uint32_t);
    rejects(rows, short_streams, "undersized",
            "reject undersized GR broadcast output view");

    auto misaligned_embedding = valid;
    misaligned_embedding.embedding =
        subview(embedding_storage.view(), 1u, embedding_bytes);
    rejects(rows, misaligned_embedding, "misaligned",
            "reject misaligned GR broadcast embedding view");
    auto misaligned_streams = valid;
    misaligned_streams.streams =
        subview(streams_storage.view(), 1u, streams_bytes);
    rejects(rows, misaligned_streams, "misaligned",
            "reject misaligned GR broadcast output view");

    BufferRef out_of_bounds_storage = context.buffers().allocate(
        "GR broadcast out-of-allocation view", embedding_bytes + 16u, error);
    check(static_cast<bool>(out_of_bounds_storage),
          "allocate GR broadcast out-of-allocation storage", error);
    if (!out_of_bounds_storage) return false;
    auto out_of_bounds = valid;
    out_of_bounds.embedding = subview(
        out_of_bounds_storage, embedding_bytes + 12u, embedding_bytes);
    rejects(rows, out_of_bounds, "not owned by this Context",
            "reject out-of-allocation GR broadcast slice");

    BufferRef shared_storage = context.buffers().allocate(
        "GR broadcast overlapping input/output", streams_bytes + 16u, error);
    check(static_cast<bool>(shared_storage),
          "allocate GR broadcast overlap storage", error);
    if (!shared_storage) return false;
    auto contained_alias = valid;
    contained_alias.embedding =
        subview(shared_storage, 0u, embedding_bytes);
    contained_alias.streams = subview(shared_storage, 0u, streams_bytes);
    rejects(rows, contained_alias, "must not overlap",
            "reject GR broadcast output containing its input");
    auto partial_alias = valid;
    partial_alias.embedding =
        subview(shared_storage, sizeof(std::uint32_t), embedding_bytes);
    partial_alias.streams = subview(shared_storage, 0u, streams_bytes);
    rejects(rows, partial_alias, "must not overlap",
            "reject partial GR broadcast input/output overlap");

    auto address_less_alias = contained_alias;
    address_less_alias.embedding.gpu_address = 0u;
    address_less_alias.streams.gpu_address = 0u;
    rejects(rows, address_less_alias, "not owned by this Context",
            "reject forged GPU metadata on GR broadcast views");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "query page size for GR broadcast shared-backing test");
        return false;
    }
    const std::size_t page_size = static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_bytes =
        ((streams_bytes + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "map anonymous storage for GR broadcast shared backing");
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef shared_input_handle = context.buffers().wrap_no_copy(
        "GR broadcast shared input", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef shared_output_handle = context.buffers().wrap_no_copy(
        "GR broadcast shared output", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    check(static_cast<bool>(shared_input_handle) &&
              static_cast<bool>(shared_output_handle),
          "wrap GR broadcast backing in distinct Metal handles", error);
    if (!shared_input_handle || !shared_output_handle) return false;
    check(shared_input_handle.handle != shared_output_handle.handle,
          "GR broadcast shared backing uses distinct MTLBuffer handles");
    check(shared_input_handle.contents == shared_output_handle.contents,
          "GR broadcast handles share CPU backing address");
    auto cross_handle_alias = valid;
    cross_handle_alias.embedding =
        subview(shared_input_handle, 0u, embedding_bytes);
    cross_handle_alias.streams =
        subview(shared_output_handle, 0u, streams_bytes);
    rejects(rows, cross_handle_alias, "must not overlap",
            "reject GR broadcast overlap across distinct Metal handles");
    auto cross_handle_missing_metadata = cross_handle_alias;
    cross_handle_missing_metadata.embedding.contents = nullptr;
    cross_handle_missing_metadata.embedding.gpu_address = 0u;
    rejects(rows, cross_handle_missing_metadata, "not owned by this Context",
            "reject forged CPU/GPU metadata on GR broadcast view");

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign GR broadcast Context",
          error);
    if (!foreign_context) return false;
    BufferRef foreign_embedding = foreign_context->buffers().allocate(
        "foreign GR broadcast embedding", embedding_bytes, error);
    BufferRef foreign_streams = foreign_context->buffers().allocate(
        "foreign GR broadcast streams", streams_bytes, error);
    check(static_cast<bool>(foreign_embedding) &&
              static_cast<bool>(foreign_streams),
          "allocate foreign GR broadcast views", error);
    if (!foreign_embedding || !foreign_streams) return false;
    auto foreign_input = valid;
    foreign_input.embedding = foreign_embedding;
    rejects(rows, foreign_input, "not owned by this Context",
            "reject foreign GR broadcast input");
    auto foreign_output = valid;
    foreign_output.streams = foreign_streams;
    rejects(rows, foreign_output, "not owned by this Context",
            "reject foreign GR broadcast output");

    auto foreign_command = foreign_context->make_command_buffer(
        "foreign GR broadcast validation command", error);
    check(foreign_command != nullptr,
          "create foreign GR broadcast command", error);
    if (!foreign_command) return false;
    std::string foreign_error;
    const bool rejects_foreign_command =
        !encode_gr_broadcast(context, *foreign_command, rows, valid,
                             foreign_error) &&
        foreign_error.find("different Context") != std::string::npos;
    check(rejects_foreign_command,
          "reject foreign GR broadcast command buffer", foreign_error);
    passed &= rejects_foreign_command;
    foreign_command->discard();

    auto enqueued =
        context.make_command_buffer("enqueued GR broadcast validation", error);
    check(enqueued != nullptr, "create enqueued GR broadcast command", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "enqueue GR broadcast validation command", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    std::string enqueued_error;
    const bool rejects_enqueued =
        !encode_gr_broadcast(context, *enqueued, rows, valid, enqueued_error) &&
        enqueued_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_enqueued,
          "reject enqueued GR broadcast command buffer", enqueued_error);
    passed &= rejects_enqueued;
    const bool enqueued_completes =
        submit_and_wait(*enqueued, "enqueued GR broadcast", error);
    check(enqueued_completes,
          "enqueued GR broadcast command completes", error);
    passed &= enqueued_completes;

    auto committed = context.make_command_buffer(
        "committed GR broadcast validation", error);
    check(committed != nullptr, "create committed GR broadcast command", error);
    if (!committed) return false;
    const bool commit_ok = committed->commit(error);
    check(commit_ok, "commit GR broadcast validation command", error);
    if (!commit_ok) return false;
    std::string committed_error;
    const bool rejects_committed =
        !encode_gr_broadcast(context, *committed, rows, valid,
                             committed_error) &&
        committed_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_committed,
          "reject committed GR broadcast command buffer", committed_error);
    passed &= rejects_committed;
    const bool committed_completes =
        wait_for_completion(*committed, "committed GR broadcast", error);
    check(committed_completes,
          "committed GR broadcast command completes", error);
    passed &= committed_completes;
    return passed;
}

std::uint32_t image_word(twin::InputSet set, std::size_t index,
                         std::mt19937& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<std::uint32_t, 12> edge{
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
            0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u,
            0x7fc12345u, 0xffc54321u, 0x7fa00001u, 0xffffffffu};
        return edge[index % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        return 0x9e3779b9u * static_cast<std::uint32_t>(index + 1u) ^
               0x7f4a7c15u;
    }
    return rng();
}

bool test_embed_image(Context& context, std::uint32_t rows, ImagePattern pattern,
                      twin::InputSet set, const twin::Options& options) {
    const std::string label = "embed_image T=" + std::to_string(rows) +
        (pattern == ImagePattern::row_zero ? " row-zero" : " after-text") +
        " set=" + twin::input_set_name(set);
    const std::size_t values =
        static_cast<std::size_t>(rows) * kEmbeddingWidth;
    const std::size_t value_bytes = values * sizeof(std::uint32_t);
    const std::size_t token_bytes =
        static_cast<std::size_t>(rows) * sizeof(std::uint32_t);

    std::vector<std::uint32_t> tokens(rows);
    std::vector<std::uint32_t> image(values);
    std::vector<std::uint32_t> initial(values);
    std::vector<std::uint32_t> expected(values);
    std::vector<twin::ByteRange> written;
    std::mt19937 rng(static_cast<std::uint32_t>(options.seed) ^
                     (rows * 0x85ebca6bu) ^ 0x494d4755u);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const bool is_image = pattern == ImagePattern::row_zero
            ? row % 3u == 0u
            : row % 3u == 1u;
        tokens[row] = is_image
            ? (kImageToken | (0x12345u + row))
            : (0x12345u + row);
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * kEmbeddingWidth;
        for (std::uint32_t column = 0; column < kEmbeddingWidth; ++column) {
            const std::size_t index = row_offset + column;
            image[index] = image_word(set, index, rng);
            if (set == twin::InputSet::edge) {
                initial[index] = 0xd00d0000u ^ static_cast<std::uint32_t>(
                    row * kEmbeddingWidth + column);
            } else if (set == twin::InputSet::fixture) {
                initial[index] = 0xa5a50000u |
                    static_cast<std::uint32_t>((row * 257u + column) & 0xffffu);
            } else {
                initial[index] = rng();
            }
        }
    }
    expected = initial;
    for (std::uint32_t row = 0; row < rows; ++row) {
        if ((tokens[row] & kImageToken) == 0u) continue;
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * kEmbeddingWidth;
        const std::size_t row_bytes =
            static_cast<std::size_t>(kEmbeddingWidth) * sizeof(std::uint32_t);
        written.push_back({row_offset * sizeof(std::uint32_t), row_bytes});
        std::copy_n(image.begin() + static_cast<std::ptrdiff_t>(row_offset),
                    kEmbeddingWidth,
                    expected.begin() + static_cast<std::ptrdiff_t>(row_offset));
    }

    twin::GuardedBuffer token_buffer;
    twin::GuardedBuffer image_buffer;
    twin::GuardedBuffer embedding_buffer;
    std::string err;
    check(token_buffer.allocate(context, label + " tokens", token_bytes, err),
          label + " token buffer", err);
    check(image_buffer.allocate(context, label + " image input", value_bytes, err),
          label + " image input buffer", err);
    check(embedding_buffer.allocate(context, label + " embedding", value_bytes, err),
          label + " embedding buffer", err);
    if (!token_buffer.view() || !image_buffer.view() || !embedding_buffer.view()) {
        return false;
    }
    std::memcpy(token_buffer.payload().data(), tokens.data(), token_bytes);
    std::memcpy(image_buffer.payload().data(), image.data(), value_bytes);
    std::memcpy(embedding_buffer.payload().data(), initial.data(), value_bytes);
    token_buffer.capture_before_dispatch();
    image_buffer.capture_before_dispatch();
    embedding_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    check(command != nullptr, label + " command buffer", err);
    if (!command) return false;
    const EmbedImageBuffers buffers{
        token_buffer.view(), image_buffer.view(), embedding_buffer.view()};
    const bool encoded =
        encode_embed_image(context, *command, rows, buffers, err);
    check(encoded, label + " validated Metal encoder", err);
    if (!encoded) return false;
    const bool dispatched = submit_and_wait(*command, label, err);
    check(dispatched, label + " command completes", err);
    if (!dispatched) return false;

    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), value_bytes);
    const auto mismatch =
        twin::compare_bytes(embedding_buffer.payload(), expected_bytes);
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    check(!mismatch, label + " exact image copy and non-image preservation");
    check(!token_buffer.first_unwritten_change({}),
          label + " tokens and guards remain unchanged");
    check(!image_buffer.first_unwritten_change({}),
          label + " image input and guards remain unchanged");
    const auto changed = embedding_buffer.first_unwritten_change(written);
    check(!changed, label + " non-image rows and embedding guards remain unchanged",
          changed ? changed->reason : "");
    return !mismatch && !changed;
}

bool test_prefill_embedding_stage(Context& context, bool with_image) {
    const std::string label =
        with_image ? "Prefill embedding stage with image rows"
                   : "Prefill embedding stage without image rows";
    const Format& format = kFormats.back();
    const ggml_type_traits* traits = ggml_get_type_traits(format.type);
    if (traits == nullptr) {
        check(false, label + " ggml type traits");
        return false;
    }

    const std::uint32_t rows = with_image ? 3u : 2u;
    std::vector<std::uint32_t> tokens{5u, 1u};
    if (with_image) tokens.push_back(kImageToken | 0x12345u);
    std::vector<std::uint8_t> table;
    std::size_t row_bytes = 0u;
    std::string error;
    check(fill_table(format, *traits, twin::InputSet::fixture, 0x6d2b79f5u,
                     table, row_bytes, error),
          label + " fixture table", error);
    if (table.empty()) return false;

    const std::size_t token_bytes =
        static_cast<std::size_t>(rows) * sizeof(std::uint32_t);
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth * sizeof(float);
    const std::size_t stream_bytes =
        static_cast<std::size_t>(rows) * kStreamWidth * sizeof(float);
    twin::GuardedBuffer table_buffer;
    twin::GuardedBuffer token_buffer;
    twin::GuardedBuffer image_buffer;
    twin::GuardedBuffer embedding_buffer;
    twin::GuardedBuffer stream_buffer;
    check(table_buffer.allocate(context, label + " table", table.size(), error),
          label + " table buffer", error);
    check(token_buffer.allocate(context, label + " tokens", token_bytes, error),
          label + " token buffer", error);
    if (with_image) {
        check(image_buffer.allocate(context, label + " image",
                                    embedding_bytes, error),
              label + " image buffer", error);
    }
    check(embedding_buffer.allocate(context, label + " embedding",
                                   embedding_bytes, error),
          label + " embedding buffer", error);
    check(stream_buffer.allocate(context, label + " streams", stream_bytes,
                                 error),
          label + " stream buffer", error);
    if (!table_buffer.view() || !token_buffer.view() ||
        (with_image && !image_buffer.view()) || !embedding_buffer.view() ||
        !stream_buffer.view()) {
        return false;
    }

    std::memcpy(table_buffer.payload().data(), table.data(), table.size());
    std::memcpy(token_buffer.payload().data(), tokens.data(), token_bytes);
    std::vector<std::uint32_t> image_words(
        static_cast<std::size_t>(rows) * kEmbeddingWidth);
    std::mt19937 rng(0x4f1bbcdcu);
    if (with_image) {
        for (std::size_t index = 0u; index < image_words.size(); ++index) {
            image_words[index] =
                image_word(twin::InputSet::fixture, index, rng);
        }
        std::memcpy(image_buffer.payload().data(), image_words.data(),
                    embedding_bytes);
    }

    constexpr std::uint8_t poison = 0xa5u;
    std::memset(embedding_buffer.payload().data(), poison, embedding_bytes);
    std::memset(stream_buffer.payload().data(), poison, stream_bytes);
    table_buffer.capture_before_dispatch();
    token_buffer.capture_before_dispatch();
    if (with_image) image_buffer.capture_before_dispatch();
    embedding_buffer.capture_before_dispatch();
    stream_buffer.capture_before_dispatch();

    PrefillChunkMetadata chunk;
    const bool built =
        build_prefill_chunk_metadata(64u, tokens, chunk, error);
    check(built, label + " metadata", error);
    if (!built) return false;

    std::vector<std::uint32_t> expected_embedding(
        static_cast<std::size_t>(rows) * kEmbeddingWidth);
    std::vector<float> decoded(kEmbeddingWidth);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * kEmbeddingWidth;
        if ((tokens[row] & kImageToken) != 0u) {
            std::copy_n(image_words.begin() +
                            static_cast<std::ptrdiff_t>(row_offset),
                        kEmbeddingWidth,
                        expected_embedding.begin() +
                            static_cast<std::ptrdiff_t>(row_offset));
            continue;
        }
        traits->to_float(table.data() +
                             static_cast<std::size_t>(tokens[row]) * row_bytes,
                         decoded.data(), kEmbeddingWidth);
        std::memcpy(expected_embedding.data() + row_offset, decoded.data(),
                    kEmbeddingWidth * sizeof(float));
    }

    std::vector<std::uint32_t> expected_streams(
        static_cast<std::size_t>(rows) * kStreamWidth);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const auto source = expected_embedding.begin() +
            static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(row) * kEmbeddingWidth);
        for (std::uint32_t stream = 0u; stream < kStreamCount; ++stream) {
            std::copy_n(source, kEmbeddingWidth,
                        expected_streams.begin() +
                            static_cast<std::ptrdiff_t>(
                                static_cast<std::size_t>(row) * kStreamWidth +
                                stream * kEmbeddingWidth));
        }
    }

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const PrefillEmbeddingBuffers buffers{
        {table_buffer.view(), token_buffer.view(), embedding_buffer.view()},
        with_image ? image_buffer.view() : BufferRef{},
        stream_buffer.view(),
    };
    const bool encoded = encode_prefill_embedding(
        context, *command, chunk, kTableRows, format.encoder_format, buffers,
        error);
    check(encoded, label + " single-encoder production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool completed = submit_and_wait(*command, label, error);
    check(completed, label + " command completes", error);
    if (!completed) return false;

    const auto expected_embedding_bytes =
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(expected_embedding.data()),
            embedding_bytes);
    const auto expected_stream_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected_streams.data()),
        stream_bytes);
    const auto embedding_mismatch = twin::compare_bytes(
        embedding_buffer.payload(), expected_embedding_bytes);
    const auto stream_mismatch =
        twin::compare_bytes(stream_buffer.payload(), expected_stream_bytes);
    check(!embedding_mismatch,
          label + " gathers text rows and copies image rows bit-exactly");
    check(!stream_mismatch,
          label + " broadcasts each embedding into all four streams");
    check(!table_buffer.first_unwritten_change({}),
          label + " table remains unchanged");
    check(!token_buffer.first_unwritten_change({}),
          label + " tokens remain unchanged");
    if (with_image) {
        check(!image_buffer.first_unwritten_change({}),
              label + " image rows remain unchanged");
    }
    const std::array<twin::ByteRange, 1> embedding_written{{
        {0u, embedding_bytes},
    }};
    const std::array<twin::ByteRange, 1> streams_written{{
        {0u, stream_bytes},
    }};
    check(!embedding_buffer.first_unwritten_change(embedding_written),
          label + " embedding guards remain unchanged");
    check(!stream_buffer.first_unwritten_change(streams_written),
          label + " stream guards remain unchanged");
    if (embedding_mismatch || stream_mismatch) return false;

    std::memset(embedding_buffer.payload().data(), poison, embedding_bytes);
    std::memset(stream_buffer.payload().data(), poison, stream_bytes);
    embedding_buffer.capture_before_dispatch();
    stream_buffer.capture_before_dispatch();
    auto shared_command =
        context.make_command_buffer(label + " shared Encoder segment", error);
    check(shared_command != nullptr,
          label + " shared Encoder command buffer", error);
    if (!shared_command) return false;
    const bool shared_encoded = Encoder::encode(
        context, *shared_command, "shared embedding and broadcast",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_embed_rows(context, encoder, rows, kTableRows,
                                   format.encoder_format, buffers.rows,
                                   encode_error)) {
                return false;
            }
            if (with_image) {
                const EmbedImageBuffers image_buffers{
                    buffers.rows.tokens, buffers.image,
                    buffers.rows.embedding,
                };
                if (!encode_embed_image(context, encoder, rows, image_buffers,
                                        encode_error)) {
                    return false;
                }
            }
            return encode_gr_broadcast(
                context, encoder, rows,
                {buffers.rows.embedding, buffers.streams}, encode_error);
        },
        error);
    check(shared_encoded,
          label + " shared Encoder composes gather, image and broadcast",
          error);
    if (!shared_encoded) return false;
    const bool shared_completed =
        submit_and_wait(*shared_command, label + " shared Encoder", error);
    check(shared_completed,
          label + " shared Encoder command completes", error);
    if (!shared_completed) return false;
    const auto shared_embedding_mismatch = twin::compare_bytes(
        embedding_buffer.payload(), expected_embedding_bytes);
    const auto shared_stream_mismatch =
        twin::compare_bytes(stream_buffer.payload(), expected_stream_bytes);
    check(!shared_embedding_mismatch,
          label + " shared Encoder embedding matches bitwise");
    check(!shared_stream_mismatch,
          label + " shared Encoder broadcast matches bitwise");
    check(!embedding_buffer.first_unwritten_change(embedding_written),
          label + " shared Encoder embedding guards remain unchanged");
    check(!stream_buffer.first_unwritten_change(streams_written),
          label + " shared Encoder stream guards remain unchanged");
    if (shared_embedding_mismatch || shared_stream_mismatch) return false;

    std::memset(embedding_buffer.payload().data(), poison, embedding_bytes);
    std::memset(stream_buffer.payload().data(), poison, stream_bytes);
    embedding_buffer.capture_before_dispatch();
    stream_buffer.capture_before_dispatch();
    auto coordinator_command = context.make_command_buffer(
        label + " shared Prefill coordinator", error);
    check(coordinator_command != nullptr,
          label + " shared Prefill coordinator command buffer", error);
    if (!coordinator_command) return false;
    const bool coordinator_encoded = Encoder::encode(
        context, *coordinator_command, "shared Prefill embedding coordinator",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_embedding(
                context, encoder, chunk, kTableRows, format.encoder_format,
                buffers, encode_error);
        },
        error);
    check(coordinator_encoded,
          label + " Prefill coordinator composes shared Encoder stages", error);
    if (!coordinator_encoded) return false;
    const bool coordinator_completed = submit_and_wait(
        *coordinator_command, label + " shared Prefill coordinator", error);
    check(coordinator_completed,
          label + " shared Prefill coordinator command completes", error);
    if (!coordinator_completed) return false;

    const auto coordinator_embedding_mismatch = twin::compare_bytes(
        embedding_buffer.payload(), expected_embedding_bytes);
    const auto coordinator_stream_mismatch = twin::compare_bytes(
        stream_buffer.payload(), expected_stream_bytes);
    check(!coordinator_embedding_mismatch,
          label + " shared Prefill coordinator embedding matches bitwise");
    check(!coordinator_stream_mismatch,
          label + " shared Prefill coordinator streams match bitwise");
    check(!embedding_buffer.first_unwritten_change(embedding_written),
          label + " shared Prefill coordinator embedding guards remain unchanged");
    check(!stream_buffer.first_unwritten_change(streams_written),
          label + " shared Prefill coordinator stream guards remain unchanged");
    if (coordinator_embedding_mismatch || coordinator_stream_mismatch) return false;

    if (concurrent_dispatch_mode() && !with_image) {
        auto group_command = context.make_command_buffer(
            "Prefill coordinator open concurrent-group rejection", error);
        check(group_command != nullptr,
              "Prefill open-group command buffer", error);
        if (!group_command) return false;
        bool group_open = false;
        bool coordinator_accepted = true;
        bool group_closed = false;
        std::string coordinator_error;
        const bool segment_encoded = Encoder::encode(
            context, *group_command, "Prefill coordinator rejects open group",
            [&](Encoder& encoder, std::string& body_error) {
                if (!encoder.group_begin(body_error)) return false;
                group_open = encoder.concurrent_group_open();
                coordinator_accepted = encode_prefill_embedding(
                    context, encoder, chunk, kTableRows,
                    format.encoder_format, buffers, coordinator_error);
                std::string group_error;
                group_closed = encoder.group_end(group_error);
                body_error = !coordinator_accepted
                    ? coordinator_error
                    : !group_closed
                        ? group_error
                        : "Prefill coordinator unexpectedly accepted an open group";
                return false;
            },
            error);
        const bool group_discarded =
            group_command->unsafe_native_handle() == nullptr;
        const bool rejected_group = group_open && !coordinator_accepted &&
            group_closed && !segment_encoded && group_discarded &&
            error.find("ordered concurrent-group boundary") != std::string::npos;
        check(rejected_group,
              "Prefill coordinator rejects an open concurrent group", error);
        group_command->discard();
        if (!rejected_group) return false;

        const std::uint32_t changed_token = tokens.front() - 1u;
        std::memcpy(token_buffer.payload().data(), &changed_token,
                    sizeof(changed_token));
        std::memset(stream_buffer.payload().data(), poison, stream_bytes);
        stream_buffer.capture_before_dispatch();
        std::vector<std::uint8_t> streams_before(
            stream_buffer.payload().begin(), stream_buffer.payload().end());
        auto discard_command = context.make_command_buffer(
            "Prefill shared later-stage rejection", error);
        check(discard_command != nullptr,
              "Prefill shared discard command buffer", error);
        if (!discard_command) return false;
        bool prior_broadcast_encoded = false;
        const bool discard_segment_encoded = Encoder::encode(
            context, *discard_command,
            "Prefill later-stage failure discards prior dispatch",
            [&](Encoder& encoder, std::string& body_error) {
                if (!encode_gr_broadcast(
                        context, encoder, rows,
                        {buffers.rows.embedding, buffers.streams}, body_error)) {
                    return false;
                }
                prior_broadcast_encoded = true;
                return encode_prefill_embedding(
                    context, encoder, chunk, kTableRows,
                    format.encoder_format, buffers, body_error);
            },
            error);
        const bool segment_discarded =
            discard_command->unsafe_native_handle() == nullptr;
        const bool streams_unchanged = std::equal(
            streams_before.begin(), streams_before.end(),
            stream_buffer.payload().begin(), stream_buffer.payload().end());
        const bool rejected_late = prior_broadcast_encoded &&
            !discard_segment_encoded && segment_discarded && streams_unchanged &&
            error.find("does not match chunk metadata") != std::string::npos;
        check(rejected_late,
              "Prefill shared failure discards earlier broadcast work", error);
        discard_command->discard();
        std::memcpy(token_buffer.payload().data(), tokens.data(), token_bytes);
        if (!rejected_late) return false;
    }

    return true;
}

bool test_prefill_embedding_stage_discards_partial_work(Context& context) {
    enum class FailureStage { token_mismatch, missing_image, broadcast_alias };
    constexpr std::array cases{
        FailureStage::token_mismatch,
        FailureStage::missing_image,
        FailureStage::broadcast_alias,
    };
    bool passed = true;
    const std::size_t table_bytes =
        static_cast<std::size_t>(kTableRows) * 5120u;
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(kEmbeddingWidth) * sizeof(float);
    const std::size_t stream_bytes =
        static_cast<std::size_t>(kStreamWidth) * sizeof(float);

    for (const FailureStage failure : cases) {
        const std::string label =
            failure == FailureStage::token_mismatch
                ? "Prefill embedding rejects token-buffer mismatch"
                : failure == FailureStage::missing_image
                      ? "Prefill embedding discards failed image stage"
                      : "Prefill embedding discards failed broadcast stage";
        std::uint32_t metadata_token = 1u;
        std::uint32_t gpu_token = 1u;
        if (failure == FailureStage::token_mismatch) gpu_token = 2u;
        if (failure == FailureStage::missing_image) {
            metadata_token = kImageToken | 1u;
            gpu_token = metadata_token;
        }
        std::string error;
        const std::array<std::uint32_t, 1> token_ids{metadata_token};
        BufferRef table =
            context.buffers().allocate(label + " table", table_bytes, error);
        BufferRef tokens =
            context.buffers().allocate(label + " tokens", sizeof(gpu_token),
                                       error);
        check(static_cast<bool>(table) && static_cast<bool>(tokens),
              label + " input allocations", error);
        if (!table || !tokens) return false;
        std::memset(table.contents, 0, table_bytes);
        std::memcpy(tokens.contents, &gpu_token, sizeof(gpu_token));

        BufferRef embedding;
        BufferRef streams;
        if (failure == FailureStage::broadcast_alias) {
            streams = context.buffers().allocate(label + " shared output",
                                                 stream_bytes, error);
            if (!streams) return false;
            embedding = subview(streams, 0u, embedding_bytes);
        } else {
            embedding = context.buffers().allocate(label + " embedding",
                                                   embedding_bytes, error);
            streams = context.buffers().allocate(label + " streams",
                                                 stream_bytes, error);
            if (!embedding || !streams) return false;
        }
        std::memset(embedding.contents, 0x5au, embedding_bytes);
        if (failure != FailureStage::broadcast_alias) {
            std::memset(streams.contents, 0x5au, stream_bytes);
        }
        std::vector<std::uint8_t> before(embedding_bytes, 0x5au);

        PrefillChunkMetadata chunk;
        const bool built =
            build_prefill_chunk_metadata(0u, token_ids, chunk, error);
        check(built, label + " metadata", error);
        if (!built) {
            passed = false;
            continue;
        }
        auto command = context.make_command_buffer(label, error);
        check(command != nullptr, label + " command buffer", error);
        if (!command) {
            passed = false;
            continue;
        }
        const PrefillEmbeddingBuffers buffers{
            {table, tokens, embedding},
            {},
            streams,
        };
        const bool rejected = !encode_prefill_embedding(
            context, *command, chunk, kTableRows, EmbedRowsFormat::bf16,
            buffers, error);
        const std::string_view expected_error =
            failure == FailureStage::token_mismatch
                ? "does not match chunk metadata"
                : failure == FailureStage::missing_image
                      ? "not owned by this Context"
                      : "overlap";
        const bool reason_matches =
            error.find(expected_error) != std::string::npos;
        const bool discarded = command->unsafe_native_handle() == nullptr;
        const bool output_unchanged =
            std::memcmp(embedding.contents, before.data(),
                        embedding_bytes) == 0;
        check(rejected && reason_matches, label + " fails at the intended stage",
              error);
        check(discarded, label + " discards its unsubmitted partial command");
        check(output_unchanged, label + " leaves output bytes untouched");
        passed &= rejected && reason_matches && discarded && output_unchanged;
    }
    return passed;
}

bool test_prefill_rejects_enqueued_command_without_committing(
    Context& context) {
    constexpr std::uint32_t expected_token = 1u;
    constexpr std::uint32_t changed_token = 2u;
    constexpr std::size_t table_bytes =
        static_cast<std::size_t>(kTableRows) * 5120u;
    constexpr std::size_t embedding_bytes =
        static_cast<std::size_t>(kEmbeddingWidth) * sizeof(float);
    constexpr std::size_t stream_bytes =
        static_cast<std::size_t>(kStreamWidth) * sizeof(float);
    std::string error;
    BufferRef table =
        context.buffers().allocate("Prefill enqueued table", table_bytes,
                                   error);
    BufferRef tokens =
        context.buffers().allocate("Prefill enqueued tokens",
                                   sizeof(changed_token), error);
    BufferRef embedding =
        context.buffers().allocate("Prefill enqueued embedding",
                                   embedding_bytes, error);
    BufferRef streams =
        context.buffers().allocate("Prefill enqueued streams", stream_bytes,
                                   error);
    check(static_cast<bool>(table) && static_cast<bool>(tokens) &&
              static_cast<bool>(embedding) && static_cast<bool>(streams),
          "allocate Prefill enqueued-command buffers", error);
    if (!table || !tokens || !embedding || !streams) return false;
    std::memset(table.contents, 0, table_bytes);
    std::memcpy(tokens.contents, &changed_token, sizeof(changed_token));

    const std::array<std::uint32_t, 1> token_ids{expected_token};
    PrefillChunkMetadata chunk;
    const bool built =
        build_prefill_chunk_metadata(0u, token_ids, chunk, error);
    check(built, "build Prefill enqueued-command metadata", error);
    if (!built) return false;

    auto command = context.make_command_buffer(
        "Prefill must not commit enqueued command", error);
    check(command != nullptr,
          "create Prefill enqueued-command buffer", error);
    if (!command) return false;
    const bool enqueued = command->enqueue(error);
    check(enqueued, "enqueue Prefill validation command", error);
    if (!enqueued) {
        command->discard();
        return false;
    }
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();

    const PrefillEmbeddingBuffers buffers{
        {table, tokens, embedding},
        {},
        streams,
    };
    const bool rejected = !encode_prefill_embedding(
        context, *command, chunk, kTableRows, EmbedRowsFormat::bf16, buffers,
        error);
    const bool correct_reason =
        error.find("before command-buffer submission") != std::string::npos;
    const bool remains_enqueued =
        native != nil && native.status == MTLCommandBufferStatusEnqueued;
    check(rejected && correct_reason,
          "reject enqueued Prefill command before stage validation", error);
    check(remains_enqueued,
          "leave rejected enqueued Prefill command uncommitted");
    if (!rejected || !correct_reason || !remains_enqueued) return false;

    const bool committed = command->commit(error);
    check(committed, "commit enqueued Prefill validation command", error);
    if (!committed) return false;
    const bool completed = wait_for_completion(
        *command, "enqueued Prefill validation", error);
    check(completed, "complete enqueued Prefill validation command", error);
    return completed;
}

bool test_command_buffer_encode_serializes_submission(Context& context) {
    std::string error;
    auto command =
        context.make_command_buffer("atomic encode and enqueue test", error);
    check(command != nullptr, "create atomic encode command buffer", error);
    if (!command) return false;

    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool encoder_entered = false;
    bool release_encoder = false;
    bool enqueue_started = false;
    bool enqueue_returned = false;
    bool enqueue_succeeded = false;
    std::string enqueue_error;
    std::string encode_error;
    bool encode_succeeded = true;

    std::thread encoder([&] {
        encode_succeeded = command->encode_unsubmitted(
            [&](void*, std::string& callback_error) {
                std::unique_lock<std::mutex> lock(state_mutex);
                encoder_entered = true;
                state_changed.notify_all();
                state_changed.wait(lock, [&] { return release_encoder; });
                callback_error = "intentional atomic-discard test failure";
                return false;
            },
            encode_error, true);
    });

    bool entered_in_time = false;
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        entered_in_time = state_changed.wait_for(
            lock, std::chrono::seconds(5), [&] { return encoder_entered; });
    }
    check(entered_in_time, "enter atomic command-buffer encoder callback");
    if (!entered_in_time) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            release_encoder = true;
        }
        state_changed.notify_all();
        encoder.join();
        return false;
    }

    std::thread submitter([&] {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            enqueue_started = true;
        }
        state_changed.notify_all();
        std::string submit_error;
        const bool submitted = command->enqueue(submit_error);
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            enqueue_succeeded = submitted;
            enqueue_error = std::move(submit_error);
            enqueue_returned = true;
        }
        state_changed.notify_all();
    });

    bool submit_attempted = false;
    bool enqueue_was_blocked = false;
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        submit_attempted = state_changed.wait_for(
            lock, std::chrono::seconds(5), [&] { return enqueue_started; });
        if (submit_attempted) {
            enqueue_was_blocked = !state_changed.wait_for(
                lock, std::chrono::milliseconds(100),
                [&] { return enqueue_returned; });
        }
        release_encoder = true;
    }
    state_changed.notify_all();
    encoder.join();
    submitter.join();

    const bool enqueue_rejected =
        !enqueue_succeeded &&
        enqueue_error.find("discarded") != std::string::npos;
    check(submit_attempted, "start enqueue during encoding");
    check(enqueue_was_blocked,
          "hold enqueue until atomic encoding decides command-buffer state");
    check(!encode_succeeded &&
              encode_error.find("intentional atomic-discard") !=
                  std::string::npos,
          "discard command buffer before releasing encoding lock", encode_error);
    check(enqueue_rejected,
          "reject enqueue after atomic encoder failure", enqueue_error);
    return submit_attempted && enqueue_was_blocked && !encode_succeeded &&
           enqueue_rejected;
}

bool test_command_buffer_dependency_orders_submission(Context& context) {
    std::string error;
    auto event = context.make_events("dependency completion gate", error);
    check(event != nullptr, "create dependency completion event", error);
    if (!event) return false;
    const std::uint64_t completion_value = event->reserve_value();
    auto producer =
        context.make_command_buffer("dependency producer", error);
    auto consumer =
        context.make_command_buffer("dependency consumer", error);
    check(producer != nullptr && consumer != nullptr,
          "create dependent command buffers", error);
    if (!producer || !consumer) return false;
    const bool wait_encoded =
        event->encode_wait(*producer, completion_value, error);
    check(wait_encoded, "encode producer completion gate", error);
    if (!wait_encoded) return false;
    const bool dependency_added = consumer->depends_on(*producer, error);
    check(dependency_added, "register command-buffer dependency", error);
    if (!dependency_added) return false;

    std::string early_submit_error;
    const bool early_submit = consumer->enqueue(early_submit_error);
    check(!early_submit &&
              early_submit_error.find("dependency") != std::string::npos,
          "reject consumer before producer enters queue", early_submit_error);
    if (early_submit) return false;

    const bool producer_committed = producer->commit(error);
    check(producer_committed, "submit dependency producer first", error);
    if (!producer_committed) return false;

    std::string incomplete_submit_error;
    const bool incomplete_submit = consumer->enqueue(incomplete_submit_error);
    check(!incomplete_submit &&
              incomplete_submit_error.find("complete successfully") !=
                  std::string::npos,
          "reject consumer until producer completes successfully",
          incomplete_submit_error);
    if (incomplete_submit) return false;

    const bool signal_ok = event->signal_host(completion_value, error);
    check(signal_ok, "release producer completion gate", error);
    if (!signal_ok) return false;
    const bool producer_completed =
        producer->wait_until_completed(error);
    check(producer_completed, "wait for successful dependency producer",
          error);
    if (!producer_completed) return false;

    const bool consumer_committed = consumer->commit(error);
    check(consumer_committed, "submit dependency consumer after producer",
          error);
    if (!consumer_committed) return false;
    const bool completed =
        wait_for_completion(*consumer, "dependent command buffers", error);
    check(completed, "complete ordered command-buffer dependency", error);
    return completed;
}

bool test_embedding_cross_command_buffer_fence(Context& context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth * sizeof(float);
    constexpr std::size_t streams_bytes =
        static_cast<std::size_t>(rows) * kStreamWidth * sizeof(float);
    std::string error;
    twin::GuardedBuffer token_buffer;
    twin::GuardedBuffer image_buffer;
    twin::GuardedBuffer embedding_buffer;
    twin::GuardedBuffer streams_buffer;
    check(token_buffer.allocate(context, "queue-fence image tokens",
                                rows * sizeof(std::uint32_t), error),
          "allocate queue-fence token buffer", error);
    check(image_buffer.allocate(context, "queue-fence image rows",
                                embedding_bytes, error),
          "allocate queue-fence image buffer", error);
    check(embedding_buffer.allocate(context, "queue-fence embeddings",
                                    embedding_bytes, error),
          "allocate queue-fence embedding buffer", error);
    check(streams_buffer.allocate(context, "queue-fence streams",
                                  streams_bytes, error),
          "allocate queue-fence stream buffer", error);
    if (!token_buffer.view() || !image_buffer.view() ||
        !embedding_buffer.view() || !streams_buffer.view()) {
        return false;
    }

    const std::array<std::uint32_t, rows> tokens{
        kImageToken, kImageToken | 1u};
    std::vector<std::uint32_t> images(
        static_cast<std::size_t>(rows) * kEmbeddingWidth);
    std::vector<std::uint32_t> expected_streams(
        static_cast<std::size_t>(rows) * kStreamWidth);
    std::mt19937 rng(0x14f2a7c9u);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t value = 0u; value < kEmbeddingWidth; ++value) {
            const std::size_t source =
                static_cast<std::size_t>(row) * kEmbeddingWidth + value;
            images[source] =
                image_word(twin::InputSet::fixture, source, rng);
            for (std::uint32_t stream = 0u; stream < kStreamCount; ++stream) {
                expected_streams[static_cast<std::size_t>(row) *
                                     kStreamWidth +
                                 stream * kEmbeddingWidth + value] =
                    images[source];
            }
        }
    }
    std::memcpy(token_buffer.payload().data(), tokens.data(),
                rows * sizeof(std::uint32_t));
    std::memcpy(image_buffer.payload().data(), images.data(), embedding_bytes);
    std::memset(embedding_buffer.payload().data(), 0xa5, embedding_bytes);
    std::memset(streams_buffer.payload().data(), 0xa5, streams_bytes);
    token_buffer.capture_before_dispatch();
    image_buffer.capture_before_dispatch();
    embedding_buffer.capture_before_dispatch();
    streams_buffer.capture_before_dispatch();

    auto image_command =
        context.make_command_buffer("Prefill image producer", error);
    check(image_command != nullptr,
          "create queue-fence image producer command", error);
    if (!image_command) return false;
    const EmbedImageBuffers image_buffers{
        token_buffer.view(), image_buffer.view(), embedding_buffer.view()};
    const bool image_encoded = encode_embed_image(
        context, *image_command, rows, image_buffers, error);
    check(image_encoded, "encode queue-fence image producer", error);
    if (!image_encoded) return false;
    const bool producer_committed = image_command->commit(error);
    check(producer_committed, "commit queue-fence image producer", error);
    if (!producer_committed) return false;

    auto broadcast_command =
        context.make_command_buffer("Prefill broadcast consumer", error);
    check(broadcast_command != nullptr,
          "create queue-fence broadcast consumer command", error);
    if (!broadcast_command) return false;
    const bool broadcast_encoded = encode_gr_broadcast(
        context, *broadcast_command, rows,
        {embedding_buffer.view(), streams_buffer.view()}, error);
    check(broadcast_encoded, "encode queue-fence broadcast consumer", error);
    if (!broadcast_encoded) {
        broadcast_command->discard();
        return false;
    }
    const bool consumer_committed = broadcast_command->commit(error);
    check(consumer_committed, "commit queue-fence broadcast consumer", error);
    if (!consumer_committed) return false;
    const bool completed = wait_for_completion(
        *broadcast_command, "queue-fence broadcast consumer", error);
    check(completed, "complete queue-fence producer/consumer chain", error);
    if (!completed) return false;

    const auto expected_embedding = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(images.data()),
        embedding_bytes);
    const auto expected_stream_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected_streams.data()),
        streams_bytes);
    const bool embedding_matches =
        !twin::compare_bytes(embedding_buffer.payload(), expected_embedding);
    const bool streams_match =
        !twin::compare_bytes(streams_buffer.payload(), expected_stream_bytes);
    check(embedding_matches,
          "cross-command-buffer image rows match their raw bits");
    check(streams_match,
          "cross-command-buffer consumer sees producer writes");
    return embedding_matches && streams_match;
}

bool test_embed_image_encoder_validation(Context& context,
                                         const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t token_bytes = rows * sizeof(std::uint32_t);
    constexpr std::size_t image_bytes =
        static_cast<std::size_t>(rows) * kEmbeddingWidth * sizeof(float);
    std::string error;

    twin::GuardedBuffer token_storage;
    twin::GuardedBuffer image_storage;
    twin::GuardedBuffer embedding_storage;
    check(token_storage.allocate(context, "embed encoder tokens",
                                 token_bytes + 16u, error),
          "allocate encoder token storage", error);
    check(image_storage.allocate(context, "embed encoder image",
                                 image_bytes + 16u, error),
          "allocate encoder image storage", error);
    check(embedding_storage.allocate(context, "embed encoder output",
                                     image_bytes + 16u, error),
          "allocate encoder output storage", error);
    if (!token_storage.view() || !image_storage.view() ||
        !embedding_storage.view()) {
        return false;
    }

    const EmbedImageBuffers valid{
        subview(token_storage.view(), 0u, token_bytes),
        subview(image_storage.view(), 0u, image_bytes),
        subview(embedding_storage.view(), 0u, image_bytes),
    };
    auto command = context.make_command_buffer("embed image validation", error);
    check(command != nullptr, "create encoder validation command", error);
    if (!command) return false;
    bool passed = true;
    const auto rejects = [&](std::uint32_t row_count,
                             const EmbedImageBuffers& buffers,
                             std::string_view expected_error,
                             const std::string& label) {
        std::string detail;
        const bool rejected =
            !encode_embed_image(context, *command, row_count, buffers, detail);
        const bool reason_matches =
            expected_error.empty() ||
            detail.find(expected_error) != std::string::npos;
        check(rejected && reason_matches, label, detail);
        passed &= rejected && reason_matches;
    };

    rejects(0u, valid, "must be positive", "reject zero image rows");
    rejects(kMaxShaderSafeImageRows + 1u, valid, "32-bit shader index",
            "reject rows that overflow the shader's uint index");

    auto short_tokens = valid;
    short_tokens.tokens.length = token_bytes - sizeof(std::uint32_t);
    rejects(rows, short_tokens, "undersized",
            "reject one-row token view at T=2");
    auto short_image = valid;
    short_image.image.length =
        image_bytes - kEmbeddingWidth * sizeof(float);
    rejects(rows, short_image, "undersized",
            "reject one-row image view at T=2");
    auto short_embedding = valid;
    short_embedding.embedding.length =
        image_bytes - kEmbeddingWidth * sizeof(float);
    rejects(rows, short_embedding, "undersized",
            "reject one-row embedding view at T=2");

    auto misaligned_tokens = valid;
    misaligned_tokens.tokens =
        subview(token_storage.view(), 1u, token_bytes);
    rejects(rows, misaligned_tokens, "misaligned",
            "reject misaligned token view");
    auto misaligned_image = valid;
    misaligned_image.image = subview(image_storage.view(), 1u, image_bytes);
    rejects(rows, misaligned_image, "misaligned",
            "reject misaligned image view");
    auto misaligned_embedding = valid;
    misaligned_embedding.embedding =
        subview(embedding_storage.view(), 1u, image_bytes);
    rejects(rows, misaligned_embedding, "misaligned",
            "reject misaligned embedding view");

    auto out_of_allocation = valid;
    out_of_allocation.image.offset =
        image_storage.view().offset + image_storage.view().length - 4u;
    out_of_allocation.image.contents = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(image_storage.view().contents) +
        image_storage.view().length - 4u);
    out_of_allocation.image.gpu_address =
        image_storage.view().gpu_address + image_storage.view().length - 4u;
    rejects(rows, out_of_allocation, "not owned by this Context",
            "reject forged image view beyond allocation");

    BufferRef shared_token_output = context.buffers().allocate(
        "embed token-output overlap", image_bytes + 16u, error);
    check(static_cast<bool>(shared_token_output),
          "allocate token-output overlap storage", error);
    if (!shared_token_output) return false;
    auto token_exact_overlap = valid;
    token_exact_overlap.tokens = subview(shared_token_output, 0u, token_bytes);
    token_exact_overlap.embedding =
        subview(shared_token_output, 0u, image_bytes);
    rejects(rows, token_exact_overlap, "must not overlap",
            "reject embedding alias with all token bytes");
    auto token_partial_overlap = valid;
    token_partial_overlap.tokens = subview(shared_token_output, 0u, token_bytes);
    token_partial_overlap.embedding =
        subview(shared_token_output, token_bytes - sizeof(std::uint32_t),
                image_bytes);
    rejects(rows, token_partial_overlap, "must not overlap",
            "reject partial embedding/token overlap");

    BufferRef shared_image_output = context.buffers().allocate(
        "embed image-output overlap", image_bytes + 16u, error);
    check(static_cast<bool>(shared_image_output),
          "allocate image-output overlap storage", error);
    if (!shared_image_output) return false;
    auto image_exact_overlap = valid;
    image_exact_overlap.image =
        subview(shared_image_output, 0u, image_bytes);
    image_exact_overlap.embedding =
        subview(shared_image_output, 0u, image_bytes);
    rejects(rows, image_exact_overlap, "must not overlap",
            "reject exact image/embedding overlap");
    auto image_partial_overlap = valid;
    image_partial_overlap.image =
        subview(shared_image_output, 0u, image_bytes);
    image_partial_overlap.embedding =
        subview(shared_image_output, sizeof(float), image_bytes);
    rejects(rows, image_partial_overlap, "must not overlap",
            "reject partial image/embedding overlap");

    auto metadata_less_alias = valid;
    metadata_less_alias.tokens = shared_token_output.slice(0u, token_bytes);
    metadata_less_alias.tokens.contents = nullptr;
    metadata_less_alias.embedding = shared_token_output.slice(0u, image_bytes);
    metadata_less_alias.embedding.contents = nullptr;
    rejects(rows, metadata_less_alias, "not owned by this Context",
            "reject alias views with forged CPU backing metadata");

    const long page_size_raw = getpagesize();
    if (page_size_raw <= 0) {
        check(false, "query page size for shared-backing alias test");
        return false;
    }
    const std::size_t page_size = static_cast<std::size_t>(page_size_raw);
    const std::size_t no_copy_bytes =
        ((image_bytes + page_size - 1u) / page_size) * page_size;
    void* const mapped_backing = mmap(
        nullptr, no_copy_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
        -1, 0);
    if (mapped_backing == MAP_FAILED) {
        check(false, "map anonymous storage for cross-handle alias test");
        return false;
    }
    auto backing_lifetime = std::shared_ptr<void>(
        mapped_backing, [no_copy_bytes](void* address) {
            if (address != nullptr) munmap(address, no_copy_bytes);
        });
    BufferRef aliased_image_handle = context.buffers().wrap_no_copy(
        "embed shared backing input", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    BufferRef aliased_output_handle = context.buffers().wrap_no_copy(
        "embed shared backing output", mapped_backing, no_copy_bytes,
        backing_lifetime, error);
    check(static_cast<bool>(aliased_image_handle) &&
              static_cast<bool>(aliased_output_handle),
          "wrap one backing allocation in distinct Metal handles", error);
    if (!aliased_image_handle || !aliased_output_handle) return false;
    check(aliased_image_handle.handle != aliased_output_handle.handle,
          "cross-handle alias uses distinct MTLBuffer handles");
    check(aliased_image_handle.contents == aliased_output_handle.contents,
          "cross-handle alias shares CPU backing address");
    auto cross_handle_alias = valid;
    cross_handle_alias.image =
        subview(aliased_image_handle, 0u, image_bytes);
    cross_handle_alias.embedding =
        subview(aliased_output_handle, 0u, image_bytes);
    rejects(rows, cross_handle_alias, "must not overlap",
            "reject exact overlap across distinct Metal handles");
    auto cross_handle_missing_metadata = cross_handle_alias;
    cross_handle_missing_metadata.image.contents = nullptr;
    cross_handle_missing_metadata.image.gpu_address = 0u;
    rejects(rows, cross_handle_missing_metadata, "not owned by this Context",
            "reject forged CPU/GPU metadata on cross-handle alias");

    // Token and image inputs are both read-only, so an overlapping backing
    // allocation is safe as long as the output stays separate.
    BufferRef shared_read_only = context.buffers().allocate(
        "embed read-only overlap", image_bytes, error);
    BufferRef separate_output = context.buffers().allocate(
        "embed read-only overlap output", image_bytes, error);
    check(static_cast<bool>(shared_read_only) &&
              static_cast<bool>(separate_output),
          "allocate read-only overlap buffers", error);
    if (!shared_read_only || !separate_output) return false;
    const std::array<std::uint32_t, rows> overlap_tokens{
        kImageToken | 17u, 23u};
    std::memset(shared_read_only.contents, 0, image_bytes);
    std::memcpy(shared_read_only.contents, overlap_tokens.data(), token_bytes);
    const EmbedImageBuffers read_only_alias{
        subview(shared_read_only, 0u, token_bytes),
        subview(shared_read_only, 0u, image_bytes),
        subview(separate_output, 0u, image_bytes),
    };
    auto read_only_command =
        context.make_command_buffer("embed read-only overlap", error);
    check(read_only_command != nullptr,
          "create read-only overlap command", error);
    if (!read_only_command) return false;
    const bool accepts_read_only_overlap =
        encode_embed_image(context, *read_only_command, rows,
                           read_only_alias, error);
    check(accepts_read_only_overlap,
          "allow overlapping read-only token/image inputs", error);
    passed &= accepts_read_only_overlap;
    if (accepts_read_only_overlap) {
        const bool completes = submit_and_wait(
            *read_only_command, "embed read-only overlap", error);
        check(completes, "read-only overlap dispatch completes", error);
        passed &= completes;
    }

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign embed Context", error);
    if (!foreign_context) return false;
    BufferRef foreign_token = foreign_context->buffers().allocate(
        "foreign embed tokens", token_bytes, error);
    BufferRef foreign_image = foreign_context->buffers().allocate(
        "foreign embed image", image_bytes, error);
    BufferRef foreign_output = foreign_context->buffers().allocate(
        "foreign embed output", image_bytes, error);
    check(static_cast<bool>(foreign_token) && static_cast<bool>(foreign_image) &&
              static_cast<bool>(foreign_output),
          "allocate foreign embed buffers", error);
    if (!foreign_token || !foreign_image || !foreign_output) return false;
    rejects(rows, {foreign_token, valid.image, valid.embedding},
            "not owned by this Context", "reject foreign token buffer");
    rejects(rows, {valid.tokens, foreign_image, valid.embedding},
            "not owned by this Context", "reject foreign image buffer");
    rejects(rows, {valid.tokens, valid.image, foreign_output},
            "not owned by this Context", "reject foreign output buffer");

    auto foreign_command =
        foreign_context->make_command_buffer("foreign embed command", error);
    check(foreign_command != nullptr, "create foreign embed command", error);
    if (!foreign_command) return false;
    std::string foreign_error;
    const bool rejects_foreign_command =
        !encode_embed_image(context, *foreign_command, rows, valid,
                            foreign_error) &&
        foreign_error.find("different Context") != std::string::npos;
    check(rejects_foreign_command, "reject foreign embed command buffer",
          foreign_error);
    passed &= rejects_foreign_command;

    auto enqueued = context.make_command_buffer("enqueued embed command", error);
    check(enqueued != nullptr, "create enqueued embed command", error);
    if (!enqueued) return false;
    const bool enqueue_ok = enqueued->enqueue(error);
    check(enqueue_ok, "enqueue embed validation command", error);
    if (!enqueue_ok) return false;
    std::string enqueued_error;
    const bool rejects_enqueued =
        !encode_embed_image(context, *enqueued, rows, valid, enqueued_error) &&
        enqueued_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_enqueued, "reject enqueued embed command buffer",
          enqueued_error);
    passed &= rejects_enqueued;
    const bool enqueued_completes =
        submit_and_wait(*enqueued, "enqueued embed command", error);
    check(enqueued_completes, "enqueued embed command completes", error);
    passed &= enqueued_completes;

    auto committed =
        context.make_command_buffer("committed embed command", error);
    check(committed != nullptr, "create committed embed command", error);
    if (!committed) return false;
    const bool commit_ok = committed->commit(error);
    check(commit_ok, "commit embed validation command", error);
    if (!commit_ok) return false;
    std::string committed_error;
    const bool rejects_committed =
        !encode_embed_image(context, *committed, rows, valid, committed_error) &&
        committed_error.find("before command-buffer submission") !=
            std::string::npos;
    check(rejects_committed, "reject committed embed command buffer",
          committed_error);
    passed &= rejects_committed;
    const bool committed_completes =
        wait_for_completion(*committed, "committed embed command", error);
    check(committed_completes, "committed embed command completes", error);
    passed &= committed_completes;

    return passed;
}

bool run_set(Context& context, twin::InputSet set, const twin::Options& options) {
    bool passed = true;
    for (const Format& format : kFormats) {
        for (const std::uint32_t rows : {1u, 7u}) {
            passed &= test_embed_format(context, format, rows, set, options);
        }
    }
    constexpr std::array image_patterns{
        ImagePattern::row_zero, ImagePattern::after_text};
    for (const std::uint32_t rows : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        for (const ImagePattern pattern : image_patterns) {
            passed &= test_embed_image(context, rows, pattern, set, options);
        }
    }
    for (const std::uint32_t rows : {1u, 3u, 8u}) {
        passed &= test_gr_broadcast(context, rows, set, options);
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_embed_test strata.metallib "
                     "[--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }

    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr, "metal_embed_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    bool passed = true;
    if (options.input_set_explicit) {
        passed = run_set(*context, options.input_set, options);
    } else {
        constexpr std::array input_sets{
            twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
        for (const twin::InputSet set : input_sets) {
            twin::Options case_options = options;
            case_options.input_set = set;
            case_options.input_set_explicit = true;
            std::printf("embedding input set: %s\n", twin::input_set_name(set));
            passed &= run_set(*context, set, case_options);
        }
    }
    passed &= test_embed_rows_encoder_validation(*context, argv[1]);
    passed &= test_embed_image_encoder_validation(*context, argv[1]);
    passed &= test_gr_broadcast_encoder_validation(*context, argv[1]);
    passed &= test_prefill_embedding_stage(*context, false);
    passed &= test_prefill_embedding_stage(*context, true);
    passed &= test_prefill_embedding_stage_discards_partial_work(*context);
    passed &=
        test_prefill_rejects_enqueued_command_without_committing(*context);
    passed &=
        test_command_buffer_encode_serializes_submission(*context);
    passed &=
        test_command_buffer_dependency_orders_submission(*context);
    passed &= test_embedding_cross_command_buffer_fence(*context);
    check(passed && g_failures == 0,
          "Metal image/token embedding and stream broadcast parity");
    std::printf("metal_embed_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
