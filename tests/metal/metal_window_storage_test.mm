// tests/metal/metal_window_storage_test.mm - live GGUF mmap-to-Metal full-mode mapping.
#include "strata/metal/window.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/model_storage.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/experts.hpp"
#include "strata/metal/window_ple_stage.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "window_source_lock_test_access.hpp"

#include <Metal/Metal.h>
#include <libproc.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace strata::metal;

namespace {

template <typename T>
concept HasPleLifecycle = requires(T& reader, std::string& error) {
    reader.open(std::string{}, error);
    reader.close();
};

static_assert(!HasPleLifecycle<strata::kernels::PleTableReader>,
              "ModelStorage must expose PLE row access without table lifecycle controls");

using PublicLeaseSlot =
    std::remove_cvref_t<decltype(WindowSourceFileLock{}.lock_lease())>;
static_assert(!std::is_convertible_v<std::shared_ptr<void>, PublicLeaseSlot>,
              "storage callers must not be able to forge a descriptor-owning lease");

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class CommandGatesCleanup {
public:
    CommandGatesCleanup(Events* layer_gate, std::uint64_t layer_value,
                        Events* completion_gate, std::uint64_t completion_value,
                        id<MTLCommandBuffer> command_buffer)
        : layer_gate_(layer_gate), layer_value_(layer_value),
          completion_gate_(completion_gate), completion_value_(completion_value),
          command_buffer_(command_buffer) {}

    ~CommandGatesCleanup() {
        if (pending_) {
            std::string ignored;
            (void)finish_and_drain(ignored);
        }
    }

    bool release_layer(std::string& error) {
        if (!layer_pending_) return true;
        if (!layer_gate_->signal_host(layer_value_, error)) return false;
        layer_pending_ = false;
        return true;
    }

    bool finish_and_drain(std::string& error) {
        bool succeeded = true;
        if (layer_pending_ && !release_layer(error)) succeeded = false;
        if (completion_pending_) {
            std::string completion_error;
            if (!completion_gate_->signal_host(completion_value_, completion_error)) {
                if (error.empty()) error = completion_error;
                succeeded = false;
            } else {
                completion_pending_ = false;
            }
        }
        if (succeeded) {
            [command_buffer_ waitUntilCompleted];
            pending_ = false;
        }
        return succeeded;
    }

private:
    Events* layer_gate_ = nullptr;
    std::uint64_t layer_value_ = 0;
    Events* completion_gate_ = nullptr;
    std::uint64_t completion_value_ = 0;
    id<MTLCommandBuffer> command_buffer_;
    bool layer_pending_ = true;
    bool completion_pending_ = true;
    bool pending_ = true;
};

void test_warmup_wave_barrier() {
    std::array<std::atomic<std::size_t>, 9> touched{};
    std::array<std::size_t, 2> completed_at_wave{};
    std::string error;
    const bool ran = run_full_mode_warmup_waves(
        touched.size(),
        [&](std::size_t, std::size_t item) {
            touched[item].fetch_add(1, std::memory_order_relaxed);
        },
        [&](std::size_t wave, std::size_t completed) {
            if (wave >= completed_at_wave.size()) return false;
            completed_at_wave[wave] = completed;
            return completed == wave * kFullModeWarmWorkers;
        },
        error);
    require(ran, "nine-item warm-up scheduler failed: " + error);
    require(completed_at_wave[0] == 0 && completed_at_wave[1] == kFullModeWarmWorkers,
            "the next warm-up wave must begin after all eight prior items complete");
    for (const auto& count : touched) {
        require(count.load(std::memory_order_relaxed) == 1,
                "each warm-up scheduler item must run exactly once");
    }
}

void put_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

void put_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

void put_string(std::vector<std::uint8_t>& bytes, const std::string& value) {
    put_u64(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void write_all_at(int descriptor, const void* data, std::size_t size,
                  std::uint64_t offset) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0u;
    while (written < size) {
        require(offset <= static_cast<std::uint64_t>(
                              std::numeric_limits<off_t>::max()) -
                              written,
                "sparse PLE fixture offset exceeds off_t");
        const ssize_t result = ::pwrite(
            descriptor, bytes + written, size - written,
            static_cast<off_t>(offset + written));
        if (result < 0 && errno == EINTR) continue;
        require(result > 0, "write sparse PLE fixture bytes");
        written += static_cast<std::size_t>(result);
    }
}

std::uint8_t sparse_fixture_iq4_code(std::uint16_t row_ordinal,
                                     std::uint32_t column) {
    return static_cast<std::uint8_t>(
        (row_ordinal * 7u + column * 3u + (column >> 4u)) & 0x0fu);
}

constexpr std::array<std::int8_t, 16> kSparseFixtureIq4Codebook{
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

void write_sparse_ple_fixture(
    const fs::path& path, const WindowPleRows& rows,
    std::unordered_map<std::uint32_t, std::uint16_t>& row_ordinals) {
    require(rows.token_count() == kWindowPleMaxTokens,
            "sparse PLE fixture needs a full T=8 row set");

    std::vector<std::uint8_t> header;
    put_u32(header, 0x46554747u);
    put_u32(header, 3u);
    put_u64(header, 1u);
    put_u64(header, 1u);
    put_string(header, "general.architecture");
    put_u32(header, 8u);
    put_string(header, "qwen4exp");
    put_string(header, "per_layer_token_embd.weight");
    put_u32(header, 2u);
    put_u64(header, strata::kernels::PLE_HEAD_DIM);
    put_u64(header, strata::kernels::PLE_TABLE_ROWS);
    put_u32(header, 20u);
    put_u64(header, 0u);

    const std::uint64_t data_start =
        (static_cast<std::uint64_t>(header.size()) + 31u) & ~std::uint64_t{31u};
    header.resize(static_cast<std::size_t>(data_start), 0u);
    const std::uint64_t table_bytes =
        strata::kernels::PLE_TABLE_ROWS *
        static_cast<std::uint64_t>(strata::kernels::PLE_ROW_BYTES);
    require(table_bytes <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<off_t>::max()) -
                    data_start,
            "sparse PLE fixture size exceeds off_t");

    row_ordinals.clear();
    for (const std::uint32_t row_id : rows.all_rows()) {
        if (row_ordinals.find(row_id) == row_ordinals.end()) {
            require(row_ordinals.size() < 1024u,
                    "sparse PLE fixture supports at most 1024 unique row markers");
            row_ordinals.emplace(
                row_id, static_cast<std::uint16_t>(row_ordinals.size()));
        }
    }
    require(!row_ordinals.empty(),
            "sparse PLE fixture needs at least one unique row marker");

    const int descriptor = ::open(
        path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    require(descriptor >= 0, "create sparse PLE GGUF fixture");
    try {
        write_all_at(descriptor, header.data(), header.size(), 0u);
        require(::ftruncate(
                    descriptor,
                    static_cast<off_t>(data_start + table_bytes)) == 0,
                "size sparse PLE GGUF fixture");
        std::array<std::uint8_t, strata::kernels::PLE_ROW_BYTES> encoded_row{};
        for (const auto& [row_id, row_ordinal] : row_ordinals) {
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(0x3c00u + row_ordinal);
            std::array<bool, 16> seen_codes{};
            for (std::size_t block = 0;
                 block < strata::kernels::PLE_HEAD_DIM / 32u; ++block) {
                const std::size_t block_offset = block * 18u;
                encoded_row[block_offset] =
                    static_cast<std::uint8_t>(scale_bits & 0xffu);
                encoded_row[block_offset + 1u] =
                    static_cast<std::uint8_t>(scale_bits >> 8u);
                for (std::size_t nibble = 0; nibble < 16u; ++nibble) {
                    const std::uint32_t low_column =
                        static_cast<std::uint32_t>(block * 32u + nibble);
                    const std::uint32_t high_column = low_column + 16u;
                    const std::uint8_t low_code =
                        sparse_fixture_iq4_code(row_ordinal, low_column);
                    const std::uint8_t high_code =
                        sparse_fixture_iq4_code(row_ordinal, high_column);
                    seen_codes[low_code] = true;
                    seen_codes[high_code] = true;
                    encoded_row[block_offset + 2u + nibble] =
                        static_cast<std::uint8_t>(low_code | (high_code << 4u));
                }
            }
            require(std::all_of(seen_codes.begin(), seen_codes.end(),
                                [](bool seen) { return seen; }),
                    "each sparse PLE row must exercise every IQ4_NL code");
            const std::uint64_t row_offset =
                data_start + static_cast<std::uint64_t>(row_id) *
                                 strata::kernels::PLE_ROW_BYTES;
            write_all_at(descriptor, encoded_row.data(), encoded_row.size(),
                         row_offset);
        }
    } catch (...) {
        (void)::close(descriptor);
        std::error_code ignored;
        fs::remove(path, ignored);
        throw;
    }
    require(::close(descriptor) == 0, "close sparse PLE GGUF fixture");
}

struct Fixture {
    fs::path path;
    fs::path dense_pack_path;
    std::uint64_t file_size = 0;
    std::uint64_t data_start = 0;
    std::uint64_t expert_slab_bytes = 0;
    std::uint64_t expert_count = 9;
    std::uint64_t router_bytes = 0;
    std::uint64_t router_row_bytes = 0;
    std::uint64_t dense_pack_bytes = 0;
    std::string router_tensor_name;
    std::string second_router_tensor_name;
    std::uint64_t second_router_source_bytes = 0;
    std::uint64_t second_router_row_bytes = 0;
    std::array<std::uint64_t, 3> offsets{};
    std::uint64_t router_offset = 0;
    std::uint64_t second_router_offset = 0;
    std::uint64_t dense_offset = 0;
    std::uint64_t ple_offset = 0;
    std::uint64_t post_ple_offset = 0;
};

Fixture write_fixture(const fs::path& directory, bool overlap_ple = false,
                      bool bridge_ple = false, bool include_router = true,
                      std::uint32_t router_layer = 0,
                      bool include_second_router = false) {
    constexpr std::uint64_t kExpertSlabBytes = 18ull * 3'641;
    constexpr std::uint64_t kExperts = 9;
    constexpr std::uint64_t kTensorBytes = kExperts * kExpertSlabBytes;
    constexpr std::uint64_t kRouterBytes = 2'560 * kExperts * sizeof(float);
    constexpr std::uint64_t kDenseBytes = 64 * 64 * sizeof(float);
    constexpr std::uint64_t kPleRows = 728;
    constexpr std::uint64_t kPleBytes = kPleRows * strata::kernels::PLE_ROW_BYTES;
    constexpr std::uint64_t kPostPleBytes = 1'024 * sizeof(float);
    Fixture fixture;
    fixture.dense_pack_path = directory / "dense.bin";
    fixture.expert_slab_bytes = kExpertSlabBytes;
    fixture.router_bytes = include_router ? kRouterBytes : 0;
    fixture.router_row_bytes = include_router ? kRouterBytes / 2 : 0;
    fixture.second_router_source_bytes =
        include_router && include_second_router ? 2'560 * sizeof(float) : 0;
    fixture.second_router_row_bytes =
        include_router && include_second_router ? 2'560 * sizeof(std::uint16_t) : 0;
    fixture.dense_pack_bytes = fixture.router_row_bytes +
                               fixture.second_router_row_bytes;
    const std::array<std::string, 6> names = {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "token_embd.weight",
        "blk." + std::to_string(router_layer) + ".ffn_gate_inp.weight",
        "per_layer_token_embd.weight",
    };
    fixture.path = directory / (include_second_router ? "multi-router.gguf" :
                                overlap_ple ? "overlap.gguf" :
                                bridge_ple ? "bridge.gguf" :
                                include_router ? "tiny.gguf" : "plain.gguf");
    fixture.router_tensor_name = names[4];
    fixture.second_router_tensor_name =
        include_second_router ? "blk." + std::to_string(router_layer) +
                                    ".ffn_gate_inp_shexp.weight"
                              : "";
    fixture.offsets[0] = (fixture.router_bytes + 31) & ~std::uint64_t{31};
    fixture.offsets[1] = (fixture.offsets[0] + kTensorBytes + 31) & ~std::uint64_t{31};
    fixture.offsets[2] = (fixture.offsets[1] + kTensorBytes + 31) & ~std::uint64_t{31};
    fixture.dense_offset = (fixture.offsets[2] + kTensorBytes + 31) & ~std::uint64_t{31};
    fixture.ple_offset = fixture.dense_offset + kDenseBytes;
    if (overlap_ple) fixture.ple_offset -= 32;
    fixture.post_ple_offset = (fixture.ple_offset + kPleBytes + 31) & ~std::uint64_t{31};
    if (include_router && include_second_router) {
        const std::uint64_t prior_data_end = bridge_ple
            ? fixture.post_ple_offset + kPostPleBytes
            : fixture.ple_offset + kPleBytes;
        fixture.second_router_offset =
            (prior_data_end + 31) & ~std::uint64_t{31};
    }

    std::vector<std::uint8_t> bytes;
    put_u32(bytes, 0x46554747u);
    put_u32(bytes, 3);
    put_u64(bytes, 5 + static_cast<std::uint64_t>(include_router) +
                       static_cast<std::uint64_t>(include_router &&
                                                  include_second_router) +
                       static_cast<std::uint64_t>(bridge_ple));
    put_u64(bytes, 0);
    for (std::size_t i = 0; i < 3; ++i) {
        put_string(bytes, names[i]);
        put_u32(bytes, 3);
        put_u64(bytes, 64);
        put_u64(bytes, 3'641);
        put_u64(bytes, kExperts);
        put_u32(bytes, 42);
        put_u64(bytes, fixture.offsets[i]);
    }
    put_string(bytes, names[3]);
    put_u32(bytes, 2);
    put_u64(bytes, 64);
    put_u64(bytes, 64);
    put_u32(bytes, 0);
    put_u64(bytes, fixture.dense_offset);
    if (include_router) {
        put_string(bytes, names[4]);
        put_u32(bytes, 2);
        put_u64(bytes, 2'560);
        put_u64(bytes, kExperts);
        put_u32(bytes, 0);
        put_u64(bytes, fixture.router_offset);
    }
    put_string(bytes, names[5]);
    put_u32(bytes, 2);
    put_u64(bytes, strata::kernels::PLE_HEAD_DIM);
    put_u64(bytes, kPleRows);
    put_u32(bytes, 20);
    put_u64(bytes, fixture.ple_offset);
    if (include_router && include_second_router) {
        put_string(bytes, fixture.second_router_tensor_name);
        put_u32(bytes, 1);
        put_u64(bytes, 2'560);
        put_u32(bytes, 0);
        put_u64(bytes, fixture.second_router_offset);
    }
    if (bridge_ple) {
        put_string(bytes, "post_ple.weight");
        put_u32(bytes, 2);
        put_u64(bytes, 1);
        put_u64(bytes, 1'024);
        put_u32(bytes, 0);
        put_u64(bytes, fixture.post_ple_offset);
    }
    fixture.data_start = (bytes.size() + 31) & ~std::uint64_t{31};
    std::uint64_t data_end = bridge_ple
        ? fixture.post_ple_offset + kPostPleBytes
        : fixture.ple_offset + kPleBytes;
    if (include_router && include_second_router) {
        data_end = fixture.second_router_offset + fixture.second_router_source_bytes;
    }
    bytes.resize(static_cast<std::size_t>(fixture.data_start + data_end), 0);
    fixture.file_size = bytes.size();
    for (std::size_t tensor = 0; tensor < 3; ++tensor) {
        for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
            const std::size_t begin = static_cast<std::size_t>(
                fixture.data_start + fixture.offsets[tensor] + expert * kExpertSlabBytes);
            std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                      bytes.begin() + static_cast<std::ptrdiff_t>(begin + kExpertSlabBytes),
                      static_cast<std::uint8_t>(0x30 + tensor * kExperts + expert));
        }
    }
    const std::size_t dense_begin =
        static_cast<std::size_t>(fixture.data_start + fixture.dense_offset);
    std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(dense_begin),
              bytes.begin() + static_cast<std::ptrdiff_t>(dense_begin + kDenseBytes), 0x4d);
    if (include_router) {
        const std::size_t router_begin =
            static_cast<std::size_t>(fixture.data_start + fixture.router_offset);
        std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(router_begin),
                  bytes.begin() + static_cast<std::ptrdiff_t>(router_begin + kRouterBytes), 0x5c);
    }
    if (include_router && include_second_router) {
        const std::size_t router_begin = static_cast<std::size_t>(
            fixture.data_start + fixture.second_router_offset);
        std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(router_begin),
                  bytes.begin() + static_cast<std::ptrdiff_t>(
                      router_begin + fixture.second_router_source_bytes),
                  0x6b);
    }
    const std::size_t ple_begin =
        static_cast<std::size_t>(fixture.data_start + fixture.ple_offset);
    std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(ple_begin),
              bytes.begin() + static_cast<std::ptrdiff_t>(ple_begin + kPleBytes), 0x7e);
    if (bridge_ple) {
        const std::size_t post_ple_begin =
            static_cast<std::size_t>(fixture.data_start + fixture.post_ple_offset);
        std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(post_ple_begin), bytes.end(), 0x6d);
    }
    std::ofstream output(fixture.path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("failed writing synthetic GGUF fixture");
    std::vector<std::uint8_t> packed_router(
        static_cast<std::size_t>(fixture.dense_pack_bytes));
    for (std::size_t i = 0; i < fixture.router_row_bytes; ++i) {
        packed_router[i] = static_cast<std::uint8_t>(0x80 + (i % 0x70));
    }
    for (std::size_t i = 0; i < fixture.second_router_row_bytes; ++i) {
        packed_router[fixture.router_row_bytes + i] =
            static_cast<std::uint8_t>(0x20 + (i % 0x50));
    }
    std::ofstream dense(fixture.dense_pack_path, std::ios::binary);
    dense.write(reinterpret_cast<const char*>(packed_router.data()),
                static_cast<std::streamsize>(packed_router.size()));
    if (!dense) throw std::runtime_error("failed writing synthetic dense pack fixture");
    return fixture;
}

WindowSourceFileLock lock_fixture_file(const fs::path& path) {
    std::string error;
    WindowSourceFileLock result =
        WindowSourceLockTestAccess::acquire(path.string(), false, error);
    if (!result.lock_lease()) {
        throw std::runtime_error("lock synthetic source for inspection: " + error);
    }
    return result;
}

void rebind_fixture_dense_pack_catalog(WindowSources& sources) {
    const auto found = std::find_if(
        sources.inspected_file_locks.begin(), sources.inspected_file_locks.end(),
        [&](const WindowSourceFileLock& lock) {
            return lock.path == sources.dense_pack_path;
        });
    require(found != sources.inspected_file_locks.end(),
            "find dense-pack lease for the synthetic row catalog");
    WindowSourceFileLock replacement = lock_fixture_file(sources.dense_pack_path);
    require(WindowSourceLockTestAccess::bind_dense_pack_catalog(
                replacement, sources.dense_pack_tensors),
            "bind the synthetic dense-pack row catalog");
    *found = std::move(replacement);
}

void set_dense_pack_file(WindowSources& sources, const fs::path& path) {
    WindowSourceFileLock dense_lock = lock_fixture_file(path);
    sources.dense_pack_path = path.string();
    sources.dense_pack_file_size = dense_lock.file_size;
    sources.dense_pack_device_id = dense_lock.file_device_id;
    sources.dense_pack_object_id = dense_lock.file_object_id;
    sources.inspected_file_locks.push_back(std::move(dense_lock));
}

fs::path fixture_artifact_lock_path(const fs::path& directory) {
    const fs::path resolved = fs::weakly_canonical(directory);
    return resolved.parent_path() /
           ("." + resolved.filename().string() + ".strata-artifact-set.lock");
}

bool can_take_exclusive_lock(const fs::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) throw std::runtime_error("open artifact-set lock probe");
    const int result = ::flock(descriptor, LOCK_EX | LOCK_NB);
    if (result == 0) (void)::flock(descriptor, LOCK_UN);
    (void)::close(descriptor);
    return result == 0;
}

WindowSources inspect_fixture(const Fixture& fixture) {
    const fs::path artifact_lock_path =
        fixture_artifact_lock_path(fixture.path.parent_path());
    const int create_lock = ::open(
        artifact_lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (create_lock < 0) throw std::runtime_error("create synthetic artifact-set lock");
    (void)::close(create_lock);
    WindowSourceFileLock artifact_lock = lock_fixture_file(artifact_lock_path);
    WindowSourceFileLock shard_lock = lock_fixture_file(fixture.path);
    strata::GgufFile file(fixture.path.string());
    WindowFileSource shard;
    shard.path = file.path();
    shard.file_size = file.file_size();
    shard.file_device_id = shard_lock.file_device_id;
    shard.file_object_id = shard_lock.file_object_id;
    const std::array<std::string, 4> names = {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "token_embd.weight",
    };
    for (const std::string& name : names) {
        const strata::TensorInfo* tensor = file.find(name);
        if (tensor == nullptr) {
            throw std::runtime_error("synthetic GGUF tensor is missing");
        }
        if (name == "token_embd.weight") {
            shard.tensors.push_back(
                {name, file.data_start() + tensor->offset, strata::tensor_payload_bytes(*tensor), 0});
            continue;
        }
        const std::uint64_t payload = strata::tensor_payload_bytes(*tensor);
        shard.tensors.push_back(
            {name, file.data_start() + tensor->offset, payload,
             payload / fixture.expert_count});
    }
    const strata::TensorInfo* ple = file.find("per_layer_token_embd.weight");
    if (ple == nullptr) throw std::runtime_error("synthetic GGUF PLE tensor is missing");
    shard.excluded_tensors.push_back(
        {ple->name, file.data_start() + ple->offset, strata::tensor_payload_bytes(*ple),
         WindowTensorExclusionReason::ple});
    const strata::TensorInfo* router = file.find(fixture.router_tensor_name);
    if (router != nullptr) {
        shard.excluded_tensors.push_back(
            {router->name, file.data_start() + router->offset, strata::tensor_payload_bytes(*router),
             WindowTensorExclusionReason::converted});
    }
    if (const strata::TensorInfo* post_ple = file.find("post_ple.weight")) {
        shard.tensors.push_back(
            {post_ple->name, file.data_start() + post_ple->offset,
             strata::tensor_payload_bytes(*post_ple), 0});
    }
    const strata::TensorInfo* second_router = nullptr;
    if (!fixture.second_router_tensor_name.empty()) {
        second_router = file.find(fixture.second_router_tensor_name);
        if (second_router == nullptr) {
            throw std::runtime_error("second synthetic router tensor is missing");
        }
        shard.excluded_tensors.push_back(
            {second_router->name, file.data_start() + second_router->offset,
             strata::tensor_payload_bytes(*second_router),
             WindowTensorExclusionReason::converted});
    }
    WindowSources sources;
    sources.files.push_back(std::move(shard));
    sources.inspected_file_locks.push_back(std::move(artifact_lock));
    sources.inspected_file_locks.push_back(std::move(shard_lock));
    sources.dense_pack_path = fixture.dense_pack_path.string();
    WindowSourceFileLock dense_lock = lock_fixture_file(fixture.dense_pack_path);
    if (dense_lock.file_size != fixture.dense_pack_bytes) {
        throw std::runtime_error("stat synthetic dense-pack fixture");
    }
    sources.dense_pack_file_size = dense_lock.file_size;
    sources.dense_pack_device_id = dense_lock.file_device_id;
    sources.dense_pack_object_id = dense_lock.file_object_id;
    sources.inspected_file_locks.push_back(std::move(dense_lock));
    if (router != nullptr) {
        sources.dense_pack_tensors.push_back(
            {fixture.router_tensor_name, 0, fixture.router_row_bytes,
             fixture.router_row_bytes, 4});
    }
    if (!fixture.second_router_tensor_name.empty()) {
        sources.dense_pack_tensors.push_back(
            {fixture.second_router_tensor_name, fixture.router_row_bytes,
             fixture.second_router_row_bytes,
             fixture.second_router_row_bytes, 4});
    }
    rebind_fixture_dense_pack_catalog(sources);
    return sources;
}

WindowSources inspect_fixture_with_packed_row_padding(const Fixture& fixture,
                                                       const fs::path& padded_path,
                                                       std::uint64_t prefix_bytes,
                                                       std::uint64_t tail_bytes) {
    WindowSources sources = inspect_fixture(fixture);
    std::ifstream input(fixture.dense_pack_path, std::ios::binary);
    std::vector<std::uint8_t> row(
        static_cast<std::size_t>(fixture.dense_pack_bytes));
    input.read(reinterpret_cast<char*>(row.data()),
               static_cast<std::streamsize>(row.size()));
    require(input.gcount() == static_cast<std::streamsize>(row.size()),
            "read packed router row for the padded-source fixture");
    std::vector<std::uint8_t> padded(
        static_cast<std::size_t>(prefix_bytes + row.size() + tail_bytes), 0xE7);
    std::copy(row.begin(), row.end(),
              padded.begin() + static_cast<std::ptrdiff_t>(prefix_bytes));
    std::ofstream output(padded_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(padded.data()),
                 static_cast<std::streamsize>(padded.size()));
    require(static_cast<bool>(output), "write padded dense-pack source fixture");
    output.close();
    set_dense_pack_file(sources, padded_path);
    require(sources.dense_pack_tensors.size() == 1,
            "padded-source fixture must contain one converted router row");
    sources.dense_pack_tensors[0].file_offset = prefix_bytes;
    rebind_fixture_dense_pack_catalog(sources);
    return sources;
}

void install_same_size_replacement(const fs::path& original, const fs::path& replacement,
                                   const fs::path& backup, std::uint64_t changed_offset) {
    fs::copy_file(original, replacement, fs::copy_options::overwrite_existing);
    {
        std::fstream file(replacement, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) throw std::runtime_error("open same-size replacement fixture");
        file.seekg(static_cast<std::streamoff>(changed_offset));
        char byte = 0;
        file.read(&byte, 1);
        if (file.gcount() != 1) throw std::runtime_error("read replacement fixture byte");
        byte ^= 0x5a;
        file.seekp(static_cast<std::streamoff>(changed_offset));
        file.write(&byte, 1);
        if (!file) throw std::runtime_error("change replacement fixture byte");
    }
    fs::rename(original, backup);
    fs::rename(replacement, original);
}

void restore_replaced_file(const fs::path& original, const fs::path& backup) {
    fs::remove(original);
    fs::rename(backup, original);
}

fs::path mapped_file_path(const void* address) {
    std::array<char, PROC_PIDPATHINFO_MAXSIZE> region_path{};
    const int path_length = proc_regionfilename(
        getpid(), reinterpret_cast<std::uint64_t>(address),
        region_path.data(), static_cast<std::uint32_t>(region_path.size()));
    if (path_length <= 0) return {};
    region_path.back() = '\0';
    return region_path.data();
}

bool region_path_is_not_gguf(const fs::path& region_path, const fs::path& gguf_path,
                             std::string& error) {
    std::error_code region_error;
    const bool same_file = fs::equivalent(region_path, gguf_path, region_error);
    if (region_error) {
        error = "backing path cannot be compared: " + region_error.message();
        return false;
    }
    if (same_file) {
        error = "buffer is backed by the GGUF shard";
        return false;
    }
    error.clear();
    return true;
}

void require_not_overlapping_gguf_mapping(BufferRef buffer, const strata::GgufFile& gguf,
                                const std::string& label) {
    require(buffer && buffer.contents != nullptr, label + " must expose its allocation");
    const std::uintptr_t buffer_start =
        reinterpret_cast<std::uintptr_t>(buffer.contents);
    const std::uintptr_t source_start =
        reinterpret_cast<std::uintptr_t>(gguf.mapping_data());
    const std::uintptr_t address_limit = std::numeric_limits<std::uintptr_t>::max();
    require(buffer.length <= address_limit - buffer_start &&
                gguf.file_size() <= address_limit - source_start,
            label + " or GGUF mapping range overflows the process address space");
    const std::uintptr_t buffer_end =
        buffer_start + static_cast<std::uintptr_t>(buffer.length);
    const std::uintptr_t source_end =
        source_start + static_cast<std::uintptr_t>(gguf.file_size());
    require(buffer_end <= source_start || source_end <= buffer_start,
            label + " overlaps the GGUF mapping address range");
}

void run_tiered_copy_test(Context& context, const Fixture& fixture,
                          WindowSources sources, const Geometry& base_geometry) {
    std::string error;
    require(!region_path_is_not_gguf(fixture.path / "missing-region-path", fixture.path, error) &&
                error.find("cannot be compared") != std::string::npos,
            "the tiered backing-path check must fail closed when path equivalence errors");

    Geometry geometry = base_geometry;
    geometry.family = "synthetic";
    geometry.model = "tiered-copy";
    geometry.nonexpert_bytes = 64 * 64 * sizeof(float);
    geometry.dense_pack_bytes = fixture.dense_pack_bytes;
    geometry.n_qsa_layers = 1;
    geometry.n_kv_head = 1;
    geometry.head_dim = 1;
    geometry.n_embd = 1;
    geometry.vocab = 1;
    geometry.types = {"f32", "iq2_xs"};
    geometry.layers = {{
        3 * fixture.expert_slab_bytes,
        "iq2_xs",
        "iq2_xs",
    }};

    PlanResult memory_plan;
    memory_plan.fits = true;
    memory_plan.mode = "tiered";
    memory_plan.resident_order = {{0, 7}, {0, 2}};
    memory_plan.experts_resident = memory_plan.resident_order.size();
    memory_plan.experts_resident_bytes =
        memory_plan.resident_order.size() * geometry.layers[0].blob_bytes;
    const std::uint64_t max_blob = geometry.layers[0].blob_bytes;
    const std::uint64_t slot_stride =
        (max_blob + kWindowPageBytes - 1) / kWindowPageBytes * kWindowPageBytes;
    memory_plan.slots =
        2 * (geometry.n_expert - memory_plan.resident_order.size());
    memory_plan.slot_bytes = memory_plan.slots * slot_stride;
    memory_plan.free_bytes = 10'000;

    {
        WindowSources remapped_pack = inspect_fixture_with_packed_row_padding(
            fixture, fixture.dense_pack_path.parent_path() / "tiered-remapped.bin",
            0, fixture.dense_pack_bytes);
        remapped_pack.dense_pack_tensors[0].file_offset = fixture.dense_pack_bytes;
        require(TieredStorage::open(context, fixture.path.string(), geometry,
                                    remapped_pack, memory_plan, error) == nullptr &&
                    error.find("differs from the WeightStore index inspection") !=
                        std::string::npos &&
                    context.buffers().allocation_count() == 0,
                "tiered storage must reject a row remapped to another valid file range");
    }

    const fs::path pending_marker =
        fixture.dense_pack_path.parent_path() / ".strata-generation-pending";
    {
        std::ofstream marker(pending_marker);
        require(static_cast<bool>(marker), "create interrupted-generation marker fixture");
    }
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                sources, memory_plan, error) == nullptr &&
                error.find("artifact generation is incomplete") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered startup must refuse a pack marked as an interrupted generation: " + error);
    fs::remove(pending_marker);

    WindowSources bad_pack = sources;
    bad_pack.dense_pack_tensors[0].file_offset = bad_pack.dense_pack_file_size;
    rebind_fixture_dense_pack_catalog(bad_pack);
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                bad_pack, memory_plan, error) == nullptr &&
                error.find("dense-pack inventory") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered storage must reject an out-of-file dense-pack row before allocating");

    WindowSources bad_pack_source_length = sources;
    --bad_pack_source_length.dense_pack_tensors[0].source_byte_count;
    rebind_fixture_dense_pack_catalog(bad_pack_source_length);
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                bad_pack_source_length, memory_plan, error) == nullptr &&
                error.find("invalid converted router source") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered storage must reject a converted source-length mismatch before allocating");

    WindowSources bad_pack_destination_length = sources;
    --bad_pack_destination_length.dense_pack_tensors[0].source_byte_count;
    --bad_pack_destination_length.dense_pack_tensors[0].byte_count;
    rebind_fixture_dense_pack_catalog(bad_pack_destination_length);
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                bad_pack_destination_length, memory_plan, error) == nullptr &&
                error.find("invalid converted router source") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered storage must reject a converted destination-length mismatch before allocating");

    WindowSources bad_pack_size = sources;
    ++bad_pack_size.dense_pack_file_size;
    for (WindowSourceFileLock& lock : bad_pack_size.inspected_file_locks) {
        if (lock.path == bad_pack_size.dense_pack_path) ++lock.file_size;
    }
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                bad_pack_size, memory_plan, error) == nullptr &&
                error.find("file size changed after WeightStore inspection") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered storage must reject a dense-pack size mismatch before allocating");
    PlanResult bad_slot_bytes = memory_plan;
    --bad_slot_bytes.slot_bytes;
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                sources, bad_slot_bytes, error) == nullptr &&
                error.find("slot pool does not match its planned size") != std::string::npos &&
                context.buffers().allocation_count() == 0,
            "tiered storage must reject a slot arena size mismatch before allocating");
    PlanResult undersized_slot_pool = memory_plan;
    undersized_slot_pool.slots -= 1;
    undersized_slot_pool.slot_bytes -= slot_stride;
    require(TieredStorage::open(context, fixture.path.string(), geometry,
                                sources, undersized_slot_pool, error) == nullptr &&
                error.find("slot pool does not match its planned size or per-layer miss capacity") !=
                    std::string::npos &&
                context.buffers().allocation_count() == 0,
            "each ping-pong half must hold a layer's maximum misses before allocating");

    bad_pack.inspected_file_locks.clear();
    bad_pack_source_length.inspected_file_locks.clear();
    bad_pack_destination_length.inspected_file_locks.clear();
    bad_pack_size.inspected_file_locks.clear();

    const int artifact_writer = ::open(
        fixture_artifact_lock_path(fixture.path.parent_path()).c_str(),
        O_RDONLY | O_CLOEXEC);
    require(artifact_writer >= 0, "open synthetic artifact-set lock for writer probe");
    errno = 0;
    const int artifact_exclusive = ::flock(artifact_writer, LOCK_EX | LOCK_NB);
    const int artifact_lock_error = errno;
    if (artifact_exclusive == 0) (void)::flock(artifact_writer, LOCK_UN);
    (void)::close(artifact_writer);
    require(artifact_exclusive != 0 &&
                (artifact_lock_error == EWOULDBLOCK || artifact_lock_error == EAGAIN),
            "WeightStore artifact-set lease must remain held through tiered startup");

    const int writer_lock = ::open(fixture.path.c_str(), O_RDONLY | O_CLOEXEC);
    require(writer_lock >= 0, "open locked synthetic GGUF for writer probe");
    errno = 0;
    const int exclusive_lock = ::flock(writer_lock, LOCK_EX | LOCK_NB);
    const int exclusive_lock_error = errno;
    if (exclusive_lock == 0) (void)::flock(writer_lock, LOCK_UN);
    (void)::close(writer_lock);
    require(exclusive_lock != 0 &&
                (exclusive_lock_error == EWOULDBLOCK || exclusive_lock_error == EAGAIN),
            "the WeightStore inspection lock must remain held through tiered startup");

    const fs::path shard_replacement = fixture.path.string() + ".replacement";
    const fs::path shard_backup = fixture.path.string() + ".backup";
    install_same_size_replacement(
        fixture.path, shard_replacement, shard_backup,
        fixture.data_start + fixture.dense_offset);
    auto replaced_shard = TieredStorage::open(context, fixture.path.string(), geometry,
                                              sources, memory_plan, error);
    const bool rejected_replaced_shard =
        replaced_shard == nullptr &&
        error.find("inspected file identity") != std::string::npos &&
        context.buffers().allocation_count() == 0;
    restore_replaced_file(fixture.path, shard_backup);
    require(rejected_replaced_shard,
            "tiered storage must reject a same-size GGUF path replacement before allocating");

    const fs::path pack_replacement = fixture.dense_pack_path.string() + ".replacement";
    const fs::path pack_backup = fixture.dense_pack_path.string() + ".backup";
    install_same_size_replacement(fixture.dense_pack_path, pack_replacement,
                                  pack_backup, 0);
    auto replaced_pack = TieredStorage::open(context, fixture.path.string(), geometry,
                                             sources, memory_plan, error);
    const bool rejected_replaced_pack =
        replaced_pack == nullptr &&
        error.find("file identity changed after WeightStore inspection") != std::string::npos &&
        context.buffers().allocation_count() == 0;
    restore_replaced_file(fixture.dense_pack_path, pack_backup);
    require(rejected_replaced_pack,
            "tiered storage must reject a same-size dense.bin path replacement before allocating");

    auto storage = TieredStorage::open(context, fixture.path.string(), geometry,
                                       sources, memory_plan, error);
    require(storage != nullptr, "open tiered storage and copy startup ranges: " + error);
    sources.inspected_file_locks.clear();
    const fs::path source_set_lock =
        fixture_artifact_lock_path(fixture.path.parent_path());
    const fs::path pack_set_lock =
        fixture_artifact_lock_path(fixture.dense_pack_path.parent_path());
    require(!can_take_exclusive_lock(source_set_lock) &&
                !can_take_exclusive_lock(pack_set_lock),
            "TieredStorage must retain both artifact-set locks after inspection leases are released");
    const TieredLayout& layout = storage->layout();
    const BufferRef dense_arena = storage->dense_arena();
    const BufferRef resident_arena = storage->resident_arena();
    const BufferRef slot_arena = storage->slot_arena();
    require(dense_arena && dense_arena.length == layout.dense_arena_bytes &&
                dense_arena.length % kWindowPageBytes == 0 &&
                dense_arena.gpu_address % kWindowPageBytes == 0 &&
                resident_arena && resident_arena.length == layout.resident_arena_bytes &&
                slot_arena && slot_arena.length == memory_plan.slot_bytes &&
                storage->slot_pool() != nullptr &&
                storage->slot_pool()->half_capacity(0) ==
                    geometry.n_expert - memory_plan.resident_order.size() &&
                storage->slot_pool()->half_capacity(1) ==
                    geometry.n_expert - memory_plan.resident_order.size(),
            "tiered arenas and ping-pong slot pool must match their planned sizes and alignments");

    strata::GgufFile source_file(fixture.path.string());
    const fs::path source_mapping_path = mapped_file_path(source_file.mapping_data());
    std::error_code source_mapping_error;
    require(!source_mapping_path.empty() &&
                fs::equivalent(source_mapping_path, fixture.path, source_mapping_error) &&
                !source_mapping_error,
            "the region-name probe must identify the source GGUF mapping before checking Metal buffers");
    const strata::TensorInfo* dense_tensor = source_file.find("token_embd.weight");
    require(dense_tensor != nullptr, "synthetic tiered fixture is missing its dense tensor");
    const std::uint8_t* expected_dense =
        source_file.mapping_data() + source_file.data_start() + dense_tensor->offset;
    const auto tensor_binding = [&](const std::string& name) -> const TieredTensorBinding& {
        const auto found = std::find_if(layout.tensors.begin(), layout.tensors.end(),
                                        [&](const TieredTensorBinding& binding) {
                                            return binding.tensor_name == name;
                                        });
        if (found == layout.tensors.end()) {
            throw std::runtime_error("tiered layout is missing tensor binding " + name);
        }
        return *found;
    };
    const TieredTensorBinding dense_binding = tensor_binding("token_embd.weight");
    require(dense_binding.arena_offset + dense_binding.byte_count <= dense_arena.length &&
                std::memcmp(static_cast<const std::uint8_t*>(dense_arena.contents) +
                                dense_binding.arena_offset,
                            expected_dense, dense_binding.byte_count) == 0,
            "tiered dense arena must contain the exact non-expert GGUF bytes");

    std::vector<std::uint8_t> packed_router(
        static_cast<std::size_t>(fixture.dense_pack_bytes));
    {
        std::ifstream input(fixture.dense_pack_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(packed_router.data()),
                   static_cast<std::streamsize>(packed_router.size()));
        require(input.gcount() == static_cast<std::streamsize>(packed_router.size()),
                "read synthetic converted-router bytes");
    }
    const TieredTensorBinding& router_binding =
        tensor_binding("blk.0.ffn_gate_inp.weight");
    require(router_binding.arena_offset + router_binding.byte_count <= dense_arena.length &&
                router_binding.byte_count == packed_router.size() &&
                std::memcmp(static_cast<const std::uint8_t*>(dense_arena.contents) +
                                router_binding.arena_offset,
                            packed_router.data(), packed_router.size()) == 0,
            "tiered dense arena must contain the exact converted router bytes from dense.bin");

    const std::uint64_t blob = geometry.layers[0].blob_bytes;
    const std::uint64_t second_offset = (blob + 255) & ~std::uint64_t{255};
    const std::array<std::uint32_t, 2> resident_order = {7, 2};
    const auto* resident_bytes = static_cast<const std::uint8_t*>(resident_arena.contents);
    for (std::size_t slot = 0; slot < resident_order.size(); ++slot) {
        const std::uint32_t expert_index = resident_order[slot];
        const TieredExpert& expert = layout.experts[expert_index];
        const std::uint64_t expected_offset = slot == 0 ? 0 : second_offset;
        require(expert.resident_slot == static_cast<std::int32_t>(slot) &&
                    expert.resident_arena_offset == expected_offset,
                "resident arena slots must follow the memory plan's profile order");
        std::uint64_t role_offset = 0;
        for (std::size_t role = 0; role < expert.roles.size(); ++role) {
            const TieredSourceRange& source = expert.roles[role];
            const std::uint8_t* expected =
                source_file.mapping_data() + source.file_offset;
            const std::uint8_t* actual =
                resident_bytes + expert.resident_arena_offset + role_offset;
            require(std::memcmp(actual, expected,
                                static_cast<std::size_t>(source.byte_count)) == 0,
                    "resident arena role bytes must exactly match the selected GGUF expert slab");
            role_offset += source.byte_count;
        }
    }

    const BufferRef address_table = storage->expert_address_table();
    const BufferRef resident_table = storage->resident_table();
    const BufferRef plan_buffer = storage->plan_buffer();
    require(address_table && resident_table && plan_buffer,
            "tiered addr, res and plan tables must be shared Metal buffers");
    const auto* addresses = static_cast<const std::uint64_t*>(address_table.contents);
    const auto* resident = static_cast<const std::int32_t*>(resident_table.contents);
    const auto* plans = static_cast<const TieredMoePlan*>(plan_buffer.contents);
    require(resident[7] == 0 && resident[2] == 1 && resident[0] == -1 &&
                resident[geometry.n_expert] == 0 &&
                addresses[7 * 3] == layout.experts[7].resident_addresses[0] &&
                addresses[7 * 3 + 1] == layout.experts[7].resident_addresses[1] &&
                addresses[7 * 3 + 2] == layout.experts[7].resident_addresses[2] &&
                addresses[0] == 0 && addresses[1] == 0 && addresses[2] == 0 &&
                plans[0].n_groups == 0 && plans[0].n_miss == 0,
            "tiered shared tables must match the host layout and start with zeroed plans");

    auto expert_store = ExpertStore::create(*storage, geometry, error);
    require(expert_store != nullptr, "create tiered ExpertStore: " + error);
    TieredMoePlan bad_plan{};
    bad_plan.n_groups = 2;
    bad_plan.n_entries = 2;
    bad_plan.n_miss = 1;
    bad_plan.skip = 0;
    bad_plan.grp_e[0] = 3;
    bad_plan.grp_e[1] = 3;
    bad_plan.grp_res[0] = 0;
    bad_plan.grp_res[1] = 0;
    bad_plan.grp_start[0] = 0;
    bad_plan.grp_start[1] = 1;
    bad_plan.grp_start[2] = 2;
    auto invalid_plan_command =
        context.make_command_buffer("invalid tiered plan validation", error);
    require(invalid_plan_command != nullptr,
            "create invalid-plan command buffer: " + error);
    auto invalid_plan_leases =
        expert_store->prepare_command_buffer(*invalid_plan_command, error);
    require(invalid_plan_leases != nullptr,
            "prepare invalid-plan completion callback: " + error);
    ExpertLayerStats invalid_plan_stats;
    require(!expert_store->resolve_layer(invalid_plan_leases, 0, bad_plan,
                                         invalid_plan_stats, error) &&
                error.find("invalid tiered plan") != std::string::npos,
            "ExpertStore must reject duplicate experts before reading or publishing addresses");

    Geometry transposed_geometry = geometry;
    transposed_geometry.n_layers = geometry.n_expert;
    transposed_geometry.n_expert = geometry.n_layers;
    auto transposed_store =
        ExpertStore::create(*storage, transposed_geometry, error);
    require(transposed_store == nullptr &&
                error.find("layer/expert coordinates") != std::string::npos,
            "ExpertStore must reject equal-product but transposed model dimensions");

    TieredMoePlan resolve_plan{};
    resolve_plan.n_groups = 4;
    resolve_plan.n_entries = 10;
    resolve_plan.n_miss = 2;
    resolve_plan.skip = 0;
    resolve_plan.grp_e[0] = 7;
    resolve_plan.grp_e[1] = 3;
    resolve_plan.grp_e[2] = 2;
    resolve_plan.grp_e[3] = 8;
    resolve_plan.grp_res[0] = 1;
    resolve_plan.grp_res[1] = 0;
    resolve_plan.grp_res[2] = 1;
    resolve_plan.grp_res[3] = 0;
    resolve_plan.grp_start[0] = 0;
    resolve_plan.grp_start[1] = 4;
    resolve_plan.grp_start[2] = 6;
    resolve_plan.grp_start[3] = 7;
    resolve_plan.grp_start[4] = 10;
    for (std::int32_t entry = 0; entry < 10; ++entry) {
        resolve_plan.ent_tok[entry] = entry / 10;
        resolve_plan.ent_dst[entry] = entry;
    }

    for (const auto& malformed : {
             std::pair<const char*, TieredMoePlan>{
                 "out-of-range destination", [&] {
                     TieredMoePlan plan = resolve_plan;
                     plan.ent_dst[0] = 10;
                     return plan;
                 }()},
             std::pair<const char*, TieredMoePlan>{
                 "duplicate destination", [&] {
                     TieredMoePlan plan = resolve_plan;
                     plan.ent_dst[1] = 0;
                     return plan;
                 }()},
             std::pair<const char*, TieredMoePlan>{
                 "out-of-range token row", [&] {
                     TieredMoePlan plan = resolve_plan;
                     plan.ent_tok[0] = 1;
                     return plan;
                 }()},
         }) {
        auto malformed_command =
            context.make_command_buffer("malformed tiered entry indices", error);
        require(malformed_command != nullptr,
                "create malformed-index command buffer: " + error);
        auto malformed_leases =
            expert_store->prepare_command_buffer(*malformed_command, error);
        require(malformed_leases != nullptr,
                "prepare malformed-index completion callback: " + error);
        ExpertLayerStats malformed_stats;
        require(!expert_store->resolve_layer(malformed_leases, 0, malformed.second,
                                             malformed_stats, error) &&
                    error.find("invalid tiered plan entry index") != std::string::npos,
                std::string("reject ") + malformed.first +
                    " before reading or publishing slots");
    }

    auto arrival_events = context.make_events("tiered resolver GPU arrival", error);
    auto release_events = context.make_events("tiered resolver host release", error);
    auto layer_done_events = context.make_events("tiered resolver layer completion", error);
    auto completion_gate_events = context.make_events("tiered resolver command completion", error);
    require(arrival_events != nullptr && release_events != nullptr &&
                layer_done_events != nullptr && completion_gate_events != nullptr,
            "create layer-handshake test events: " + error);
    const std::uint64_t arrival_value = arrival_events->reserve_value();
    const std::uint64_t release_value = release_events->reserve_value();
    const std::uint64_t layer_done_value = layer_done_events->reserve_value();
    const std::uint64_t completion_gate_value = completion_gate_events->reserve_value();
    auto resolve_command =
        context.make_command_buffer("tiered expert miss resolution", error);
    require(resolve_command != nullptr,
            "create tiered expert resolver command buffer: " + error);
    require(arrival_events->encode_signal(*resolve_command, arrival_value, error) &&
                release_events->encode_wait(*resolve_command, release_value, error) &&
                layer_done_events->encode_signal(*resolve_command, layer_done_value, error) &&
                completion_gate_events->encode_wait(*resolve_command,
                                                   completion_gate_value, error),
            "encode plan-ready and host-release events: " + error);
    auto resolve_leases =
        expert_store->prepare_command_buffer(*resolve_command, error);
    require(resolve_leases != nullptr,
            "register slot release before command-buffer submission: " + error);
    id<MTLCommandBuffer> native_resolve =
        (__bridge id<MTLCommandBuffer>)resolve_command->unsafe_native_handle();
    require(resolve_command->commit(error),
            "submit plan-producing command buffer: " + error);
    CommandGatesCleanup gate_cleanup(
        release_events.get(), release_value, completion_gate_events.get(),
        completion_gate_value, native_resolve);
    require(arrival_events->wait(arrival_value, 5000, error),
            "wait until the simulated GPU plan is ready: " + error);
    ExpertLayerStats first_resolve;
    require(expert_store->resolve_layer(resolve_leases, 0, resolve_plan,
                                        first_resolve, error),
            "resolve a mixed plan after its command buffer is submitted: " + error);
    require(first_resolve.hits == 5 && first_resolve.lookups == 10 &&
                first_resolve.ram_blobs + first_resolve.file_blobs == 2 &&
                first_resolve.file_bytes == first_resolve.file_blobs * blob,
            "DONE counters must count routed hits and classify each cold expert once");

    for (const std::uint32_t expert_id : {3u, 8u}) {
        auto lease = storage->slot_pool()->lookup_ready(
            0, expert_id, layout.experts[expert_id].blob_bytes, error);
        require(lease && error.empty(),
                "resolved cold expert must be published in the slot cache: " + error);
        const TieredExpert& expert = layout.experts[expert_id];
        std::uint64_t role_offset = 0;
        for (std::size_t role = 0; role < expert.roles.size(); ++role) {
            const TieredSourceRange& source = expert.roles[role];
            require(addresses[expert_id * 3 + role] ==
                        lease.view().gpu_address + role_offset &&
                        std::memcmp(static_cast<const std::uint8_t*>(lease.view().contents) +
                                        role_offset,
                                    source_file.mapping_data() + source.file_offset,
                                    static_cast<std::size_t>(source.byte_count)) == 0,
                    "miss resolution must read exact GGUF slab bytes and publish their GPU addresses");
            role_offset += source.byte_count;
        }
    }

    require(storage->slot_pool()->half_capacity(0) == 7,
            "synthetic tiered plan must expose seven slots for its even-layer half");
    std::vector<SlotPool::Lease> other_layer_pins;
    other_layer_pins.reserve(5);
    for (const std::uint32_t expert_id : {0u, 1u, 4u, 5u, 6u}) {
        auto lease = storage->slot_pool()->acquire(
            0, expert_id, layout.experts[expert_id].blob_bytes, error);
        require(lease && !lease.view().cache_hit,
                "claim another cold expert slot while the first layer is in flight: " +
                    error);
        require(lease.publish(error),
                "publish synthetic fill slot for the retirement test: " + error);
        other_layer_pins.emplace_back(std::move(lease));
    }
    auto exhausted_half = storage->slot_pool()->acquire(
        0, 2, layout.experts[2].blob_bytes, error);
    require(!exhausted_half && error.find("no unpinned slot") != std::string::npos,
            "all same-parity slots stay pinned before the layer-complete handshake (lease=" +
                std::to_string(static_cast<bool>(exhausted_half)) + ", error=" + error + ")");
    require(gate_cleanup.release_layer(error),
            "release the submitted layer after slot reads: " + error);
    require(layer_done_events->wait(layer_done_value, 5000, error),
            "wait until the simulated GPU layer has consumed its expert addresses: " + error);
    resolve_leases->retire_layer(0);
    auto reused_half = storage->slot_pool()->acquire(
        0, 2, layout.experts[2].blob_bytes, error);
    require(reused_half && !reused_half.view().cache_hit,
            "retiring a completed layer must free its parity half before command completion: " +
                error);
    require(reused_half.publish(error),
            "publish a slot reused after its layer-complete handshake: " + error);
    require(native_resolve.status != MTLCommandBufferStatusCompleted,
            "the command buffer stays in flight after one layer's leases are retired");
    require(gate_cleanup.finish_and_drain(error),
            "release the command-completion gate and drain the GPU command: " + error);
    other_layer_pins.clear();
    reused_half.reset();

    require(expert_store->prefetch_layer(0, error),
            "prefetch every cold expert of a layer: " + error);
    for (std::uint32_t expert_id = 0; expert_id < geometry.n_expert; ++expert_id) {
        if (resident[expert_id] >= 0) continue;
        auto ready = storage->slot_pool()->lookup_ready(
            0, expert_id, layout.experts[expert_id].blob_bytes, error);
        require(ready && error.empty(),
                "prefill prefetch must populate a slot even when the source pages are cached");
    }

    auto cache_command =
        context.make_command_buffer("tiered expert cache-hit resolution", error);
    require(cache_command != nullptr, "create cache-hit resolver command: " + error);
    auto cache_leases =
        expert_store->prepare_command_buffer(*cache_command, error);
    require(cache_leases != nullptr,
            "prepare cache-hit command-buffer completion callback: " + error);
    ExpertLayerStats cached_resolve;
    require(expert_store->resolve_layer(cache_leases, 0, resolve_plan,
                                        cached_resolve, error) &&
                cached_resolve.hits == 5 && cached_resolve.lookups == 10 &&
                cached_resolve.ram_blobs == 2 && cached_resolve.file_blobs == 0 &&
                cached_resolve.file_bytes == 0,
            "repeated misses must use ready slot-cache entries and report RAM blobs");
    require(cache_command->commit(error),
            "commit tiered cache-hit command buffer: " + error);
    id<MTLCommandBuffer> native_cache =
        (__bridge id<MTLCommandBuffer>)cache_command->unsafe_native_handle();
    [native_cache waitUntilCompleted];
    require(context.buffers().allocation_count() == 6,
            "tiered mode must allocate its three arenas and three tables, without GGUF windows");
    require_not_overlapping_gguf_mapping(dense_arena, source_file, "tiered dense arena");
    require_not_overlapping_gguf_mapping(resident_arena, source_file, "tiered resident arena");
    require_not_overlapping_gguf_mapping(slot_arena, source_file, "tiered slot arena");
    require_not_overlapping_gguf_mapping(address_table, source_file, "tiered address table");
    require_not_overlapping_gguf_mapping(resident_table, source_file, "tiered resident table");
    require_not_overlapping_gguf_mapping(plan_buffer, source_file, "tiered plan table");

    storage.reset();
    require(can_take_exclusive_lock(source_set_lock) &&
                can_take_exclusive_lock(pack_set_lock),
            "destroying TieredStorage must release its artifact-set locks");
    require(std::memcmp(static_cast<const std::uint8_t*>(dense_arena.contents) +
                            dense_binding.arena_offset,
                        expected_dense, dense_binding.byte_count) == 0,
            "Context must retain the anonymous dense arena backing after storage destruction");
}

Geometry owner_test_geometry(const Fixture& fixture, const WindowSources& sources) {
    Geometry geometry;
    geometry.family = "synthetic";
    geometry.model = "storage-owner";
    geometry.shards.push_back({fixture.path.filename().string(), fixture.file_size});
    geometry.n_layers = 1;
    geometry.n_expert = static_cast<std::uint32_t>(fixture.expert_count);
    for (const WindowFileSource& file : sources.files) {
        for (const WindowTensorSource& tensor : file.tensors) {
            if (tensor.expert_slab_bytes == 0) geometry.nonexpert_bytes += tensor.byte_count;
        }
    }
    geometry.dense_pack_bytes = fixture.dense_pack_bytes;
    geometry.n_qsa_layers = 1;
    geometry.n_kv_head = 1;
    geometry.head_dim = 1;
    geometry.n_embd = 1;
    geometry.vocab = 1;
    geometry.types = {"f32", "iq2_xs"};
    geometry.layers = {{
        3 * fixture.expert_slab_bytes,
        "iq2_xs",
        "iq2_xs",
    }};
    std::string error;
    require(compute_wrapped_unread(sources, geometry.wrapped_unread_bytes, error),
            "compute owner-test wrapped bytes: " + error);
    return geometry;
}

PlanResult owner_test_full_plan(const Geometry& geometry, const DeviceFacts& facts) {
    PlanResult plan;
    plan.fits = true;
    plan.mode = "full";
    plan.w_bytes = facts.working_set_bytes;
    plan.ram_bytes = facts.physical_memory_bytes;
    plan.available_bytes = -1;
    plan.context = 13;
    plan.kv = KvFormat::Int8;
    QsaKvLayout kv_layout;
    std::string kv_error;
    require(make_qsa_kv_layout(plan.context, plan.kv, kv_layout, kv_error),
            "compute owner-test full QSA KV layout: " + kv_error);
    plan.kv_bytes = kv_layout.allocation_bytes;
    plan.dense_bytes = geometry.nonexpert_bytes;
    plan.unread_bytes = geometry.wrapped_unread_bytes;
    plan.pack_bytes = geometry.dense_pack_bytes;
    plan.experts_total = static_cast<std::uint64_t>(geometry.n_layers) * geometry.n_expert;
    for (const LayerGeometry& layer : geometry.layers) {
        plan.experts_total_bytes += layer.blob_bytes * geometry.n_expert;
    }
    plan.experts_resident = plan.experts_total;
    plan.experts_resident_bytes = plan.experts_total_bytes;
    for (std::uint32_t layer = 0; layer < geometry.n_layers; ++layer) {
        for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
            plan.resident_order.emplace_back(layer, expert);
        }
    }
    plan.total_bytes = plan.dense_bytes + plan.unread_bytes + plan.pack_bytes +
                       plan.kv_bytes + plan.experts_resident_bytes;
    plan.ram_bytes_used = plan.total_bytes - plan.unread_bytes;
    plan.anonymous_bytes_used = plan.ram_bytes_used;
    plan.free_bytes = static_cast<std::int64_t>(plan.w_bytes - plan.total_bytes);
    return plan;
}

PlanResult owner_test_tiered_plan(const Geometry& geometry, const DeviceFacts& facts) {
    PlanResult plan;
    plan.fits = true;
    plan.mode = "tiered";
    plan.w_bytes = facts.working_set_bytes;
    plan.ram_bytes = facts.physical_memory_bytes;
    plan.available_bytes = -1;
    plan.context = 13;
    plan.kv = KvFormat::Int8;
    QsaKvLayout kv_layout;
    std::string kv_error;
    require(make_qsa_kv_layout(plan.context, plan.kv, kv_layout, kv_error),
            "compute owner-test tiered QSA KV layout: " + kv_error);
    plan.kv_bytes = kv_layout.allocation_bytes;
    plan.dense_bytes = geometry.nonexpert_bytes;
    plan.pack_bytes = geometry.dense_pack_bytes;
    plan.experts_total = static_cast<std::uint64_t>(geometry.n_layers) * geometry.n_expert;
    for (const LayerGeometry& layer : geometry.layers) {
        plan.experts_total_bytes += layer.blob_bytes * geometry.n_expert;
    }
    plan.resident_order = {{0, 7}, {0, 2}};
    plan.experts_resident = plan.resident_order.size();
    plan.experts_resident_bytes =
        plan.experts_resident * geometry.layers[0].blob_bytes;
    const std::uint64_t max_blob = geometry.layers[0].blob_bytes;
    const std::uint64_t slot_stride =
        (max_blob + kWindowPageBytes - 1) / kWindowPageBytes * kWindowPageBytes;
    plan.slots = 2 * (geometry.n_expert - plan.experts_resident);
    plan.slot_bytes = plan.slots * slot_stride;
    plan.total_bytes = plan.dense_bytes + plan.pack_bytes + plan.kv_bytes +
                       plan.slot_bytes + plan.experts_resident_bytes;
    plan.ram_bytes_used = plan.total_bytes;
    plan.anonymous_bytes_used = plan.ram_bytes_used;
    plan.free_bytes = static_cast<std::int64_t>(plan.w_bytes - plan.total_bytes);
    return plan;
}

void check_window_ple_staging(Context& context,
                              strata::kernels::PleTableReader& undersized_reader,
                              const fs::path& directory) {
    Geometry geometry;
    geometry.family = "qwen4exp";
    geometry.n_layers = kSessionModelLayers;
    geometry.n_expert = 512u;
    geometry.n_qsa_layers = kSessionQsaLayers;
    geometry.n_kv_head = kSessionKvHeads;
    geometry.head_dim = kSessionHeadWidth;
    geometry.n_embd = kSessionEmbeddingWidth;
    geometry.vocab = kSessionVocabularySize;
    const std::array<std::uint32_t, kWindowPleMaxTokens> tokens{
        0u, 7u, 42u, 1'001u, 248'319u, 9u, 23u, 17u};
    const std::array<std::uint32_t, 2> history{5u, 0u};
    WindowPleRows all_rows;
    std::string error;
    require(build_window_ple_rows(tokens, history, geometry, all_rows, error),
            "build full Window PLE row set for staging: " + error);

    std::unordered_map<std::uint32_t, std::uint16_t> row_ordinals;
    const fs::path ple_path = directory / "sparse-full-ple.gguf";
    write_sparse_ple_fixture(ple_path, all_rows, row_ordinals);
    auto model = std::make_shared<strata::GgufModel>(
        std::vector<std::string>{ple_path.string()});
    std::size_t ple_shard = 0u;
    require(model->find("per_layer_token_embd.weight", &ple_shard) != nullptr,
            "sparse PLE GGUF exposes its full-size tensor");

    strata::kernels::PleTable table;
    require(table.open_mapped(model, ple_shard, error) &&
                table.rows() == strata::kernels::PLE_TABLE_ROWS &&
                table.mode() == strata::kernels::PleIo::Mmap,
            "open sparse full-size PLE table for Window staging: " + error);
    strata::kernels::PleTableReader reader;
    table.reader_view(reader);

    auto scratch = WindowScratchStorage::open(context, geometry, 4u, error);
    require(scratch != nullptr && scratch->is_valid(),
            "open independent T=8 Window scratch for PLE staging: " + error);
    BufferRef capacity_view;
    require(scratch->view(scratch->layout().ple_input, capacity_view, error) &&
                capacity_view.length ==
                    static_cast<std::size_t>(kWindowScratchTokenRows) *
                        strata::kernels::NG_N_EMBD * sizeof(float),
            "get the full-capacity named PLE scratch region: " + error);

    for (std::uint32_t token_count = 1u;
         token_count <= kWindowPleMaxTokens; ++token_count) {
        WindowPleRows window_rows;
        require(build_window_ple_rows(
                    std::span<const std::uint32_t>(tokens.data(), token_count),
                    history, geometry, window_rows, error),
                "build prefix Window PLE rows for staging: " + error);
        const std::span<const std::uint32_t> prefix =
            window_rows.all_rows();
        const std::span<const std::uint32_t> full_prefix =
            all_rows.all_rows().first(prefix.size());
        require(std::equal(prefix.begin(), prefix.end(), full_prefix.begin()),
                "each T-sized Window PLE row set must match the T=8 prefix");

        auto* capacity = static_cast<float*>(capacity_view.contents);
        const std::size_t capacity_floats =
            capacity_view.length / sizeof(float);
        std::fill_n(capacity, capacity_floats, 123.25f);
        BufferRef staged;
        require(stage_window_ple_input(reader, window_rows, *scratch, staged,
                                       error),
                "gather Window PLE rows into named scratch: " + error);
        const std::size_t active_floats =
            static_cast<std::size_t>(token_count) *
            strata::kernels::NG_N_EMBD;
        require(staged && staged.contents == capacity_view.contents &&
                    staged.length == active_floats * sizeof(float) &&
                    staged.gpu_address == capacity_view.gpu_address,
                "staged PLE input must expose exactly the active bounded prefix");

        const auto* values = static_cast<const float*>(staged.contents);
        for (std::uint32_t token_row = 0; token_row < token_count;
             ++token_row) {
            const std::span<const std::uint32_t> ids =
                window_rows.rows_for(token_row);
            for (std::uint32_t head = 0;
                 head < kWindowPleRowsPerToken; ++head) {
                const std::uint32_t row_id = ids[head];
                const std::uint16_t row_ordinal = row_ordinals.at(row_id);
                const float scale =
                    1.0f + static_cast<float>(row_ordinal) / 1024.0f;
                const std::size_t row_offset =
                    static_cast<std::size_t>(token_row) *
                        strata::kernels::NG_N_EMBD +
                    static_cast<std::size_t>(head) *
                        strata::kernels::PLE_HEAD_DIM;
                for (std::uint32_t column = 0;
                     column < strata::kernels::PLE_HEAD_DIM; ++column) {
                    const std::uint8_t expected_code =
                        static_cast<std::uint8_t>(
                            (row_ordinal * 7u + column * 3u +
                             (column >> 4u)) &
                            0x0fu);
                    const float expected =
                        scale * static_cast<float>(
                                    kSparseFixtureIq4Codebook[expected_code]);
                    require(values[row_offset + column] == expected,
                            "staged PLE input must preserve row, column, IQ4 split-half, and scale decoding");
                }
            }
        }
        require(std::all_of(values + active_floats,
                            values + capacity_floats,
                            [](float value) { return value == 123.25f; }),
                "staging must leave inactive T=8 scratch rows untouched");
    }

    BufferRef rejected;
    require(!stage_window_ple_input(undersized_reader, all_rows, *scratch,
                                    rejected, error) &&
                !rejected &&
                error.find("row count") != std::string::npos,
            "Window staging must reject a truncated PLE table before publishing scratch");
    WindowPleRows empty_rows;
    rejected = capacity_view;
    require(!stage_window_ple_input(reader, empty_rows, *scratch, rejected,
                                    error) &&
                !rejected &&
                error.find("1..8") != std::string::npos,
            "Window staging must reject an uninitialized row set transactionally");
}

void run_model_storage_owner_test(Context& context, const fs::path& directory,
                                  const Fixture& packed_fixture) {
    std::string error;
    const fs::path plain_directory = directory / "plain";
    fs::create_directory(plain_directory);
    const Fixture plain_fixture = write_fixture(plain_directory, false, false, false);
    WindowSources plain_sources = inspect_fixture(plain_fixture);
    const Geometry plain_geometry = owner_test_geometry(plain_fixture, plain_sources);
    const DeviceFacts facts = context.device_facts();
    const PlanResult full_plan = owner_test_full_plan(plain_geometry, facts);

    Geometry mismatched_geometry = plain_geometry;
    ++mismatched_geometry.nonexpert_bytes;
    const std::size_t allocations_before_full = context.buffers().allocation_count();
    const fs::path invalid_router_directory = directory / "invalid-router-layer";
    fs::create_directory(invalid_router_directory);
    const Fixture invalid_router_fixture =
        write_fixture(invalid_router_directory, false, false, true, 1);
    WindowSources invalid_router_sources = inspect_fixture(invalid_router_fixture);
    const Geometry invalid_router_geometry =
        owner_test_geometry(invalid_router_fixture, invalid_router_sources);
    const PlanResult invalid_router_plan =
        owner_test_full_plan(invalid_router_geometry, facts);
    require(ModelStorage::open(context, invalid_router_fixture.path.string(),
                               invalid_router_geometry, invalid_router_sources,
                               invalid_router_plan, error) == nullptr &&
                error.find("supported F32 router") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must reject a converted router whose block exceeds geometry");
    const PlanResult invalid_router_tiered_plan =
        owner_test_tiered_plan(invalid_router_geometry, facts);
    require(TieredStorage::open(context, invalid_router_fixture.path.string(),
                                invalid_router_geometry, invalid_router_sources,
                                invalid_router_tiered_plan, error) == nullptr &&
                error.find("invalid converted router source") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "tiered owner must reject a converted router whose block exceeds geometry");

    require(ModelStorage::open(context, plain_fixture.path.string(), mismatched_geometry,
                               plain_sources, full_plan, error) == nullptr &&
                error.find("do not match the inspected model geometry") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must reject a plan derived from different model bytes before allocating");

    PlanResult mismatched_kv_plan = full_plan;
    mismatched_kv_plan.total_bytes -= mismatched_kv_plan.kv_bytes;
    mismatched_kv_plan.kv_bytes = 0;
    require(ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                               plain_sources, mismatched_kv_plan, error) == nullptr &&
                error.find("main QSA KV bytes do not match the memory plan") !=
                    std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "model-storage owner must reject forged main KV bytes before allocating");

    Geometry understated_dense_geometry = plain_geometry;
    --understated_dense_geometry.nonexpert_bytes;
    const PlanResult understated_dense_plan =
        owner_test_full_plan(understated_dense_geometry, facts);
    require(ModelStorage::open(context, plain_fixture.path.string(), understated_dense_geometry,
                               plain_sources, understated_dense_plan, error) == nullptr &&
                error.find("non-expert or PLE payload bytes") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must reject geometry and plan that jointly understate dense GGUF bytes");

    Geometry understated_expert_geometry = plain_geometry;
    --understated_expert_geometry.layers[0].blob_bytes;
    const PlanResult understated_expert_plan =
        owner_test_full_plan(understated_expert_geometry, facts);
    require(ModelStorage::open(context, plain_fixture.path.string(), understated_expert_geometry,
                               plain_sources, understated_expert_plan, error) == nullptr &&
                error.find("expert role payload bytes do not match geometry") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must reject geometry and plan that jointly understate expert bytes");

    WindowSources unlocked_sources = plain_sources;
    unlocked_sources.inspected_file_locks.clear();
    require(ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                               unlocked_sources, full_plan, error) == nullptr &&
                error.find("inspection lock") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must require the inspection locks that protect its no-copy mappings");

    WindowSources forged_lock_sources = plain_sources;
    bool replaced_source_lock = false;
    for (WindowSourceFileLock& lock : forged_lock_sources.inspected_file_locks) {
        if (lock.path != plain_fixture.path.string()) continue;
        WindowSourceLockTestAccess::clear(lock);
        replaced_source_lock = true;
        break;
    }
    require(replaced_source_lock &&
                ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                                   forged_lock_sources, full_plan, error) == nullptr &&
                error.find("inspection lock") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_full,
            "full-mode owner must reject forged lock metadata without an owning descriptor lease");

    WindowPlan full_window_plan;
    require(plan_windows(plain_sources, facts.max_buffer_length, full_window_plan, error),
            "plan windows for post-open rollback regression: " + error);
    std::unique_ptr<FullModeStorage> rolled_back_full_storage;
    {
        auto outer_transaction = context.buffers().transaction();
        rolled_back_full_storage = FullModeStorage::open(
            context, plain_fixture.path.string(), plain_geometry, plain_sources,
            full_window_plan, error);
        require(rolled_back_full_storage != nullptr,
                "open inner full-mode storage before warm-up failure: " + error);
        require(context.buffers().allocation_count() > allocations_before_full,
                "inner full-mode storage must commit provisional allocations");
        // Model the warm-up failure path: the enclosing ModelStorage transaction
        // must roll back the already-opened FullModeStorage allocations.
    }
    require(context.buffers().allocation_count() == allocations_before_full,
            "outer storage transaction must roll back inner allocations after warm-up failure");
    FullModeTensorView rolled_back_view;
    require(!rolled_back_full_storage->dense_tensor_view(
                "token_embd.weight", rolled_back_view, error) &&
                !rolled_back_view.buffer &&
                error.find("live GPU binding") != std::string::npos,
            "tensor lookup must reject stale views after enclosing transaction rollback");

    auto full_storage = ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                                           plain_sources, full_plan, error);
    require(full_storage != nullptr, "open and warm full-mode model storage owner: " + error);
    require(std::string(full_storage->mode()) == "full" &&
                full_storage->resident_experts() == plain_geometry.n_expert &&
                full_storage->window_count() != 0 && full_storage->full_mode_warmed(),
            "full-mode owner must retain mapped windows after a successful warm-up");
    strata::kernels::PleTableReader* full_ple = full_storage->ple_reader();
    require(full_ple != nullptr && full_ple->is_open() &&
                full_ple->mode() == strata::kernels::PleIo::Mmap &&
                full_ple->rows() == 728,
            "full-mode ModelStorage must publish its mapped PLE reader");
    const KvStore* full_kv = full_storage->kv_store();
    const BufferRef full_rope = full_kv != nullptr ? full_kv->rope() : BufferRef{};
    require(full_kv != nullptr &&
                full_kv->allocation().length == full_plan.kv_bytes &&
                full_kv->layout().requested_cells == full_plan.context &&
                full_kv->layout().format == full_plan.kv &&
                full_rope &&
                full_rope.length == full_kv->layout().rope.bytes &&
                full_rope.contents ==
                    static_cast<std::byte*>(full_kv->allocation().contents) +
                        full_kv->layout().rope.offset &&
                context.buffers().owns(full_kv->allocation()),
            "model-storage owner must expose its exact main KV and shared RoPE allocation");

    std::unique_ptr<ModelStorage> provisional_storage;
    BufferRef provisional_kv_allocation;
    BufferRef provisional_rope;
    const std::size_t allocations_before_owner_rollback =
        context.buffers().allocation_count();
    const std::uint64_t bytes_before_owner_rollback =
        context.buffers().allocated_size();
    {
        auto owner_transaction = context.buffers().transaction();
        provisional_storage = ModelStorage::open(
            context, plain_fixture.path.string(), plain_geometry, plain_sources,
            full_plan, error);
        require(provisional_storage != nullptr &&
                    provisional_storage->kv_store() != nullptr,
                "open model storage for enclosing-transaction rollback: " + error);
        provisional_kv_allocation =
            provisional_storage->kv_store()->allocation();
        provisional_rope = provisional_storage->kv_store()->rope();
        require(provisional_kv_allocation && provisional_rope &&
                    context.buffers().allocation_count() >
                        allocations_before_owner_rollback,
                "provisional main KV and RoPE views must be live before rollback");
    }
    require(context.buffers().allocation_count() ==
                    allocations_before_owner_rollback &&
                context.buffers().allocated_size() == bytes_before_owner_rollback &&
                provisional_storage != nullptr &&
                provisional_storage->kv_store() != nullptr &&
                !provisional_storage->kv_store()->allocation() &&
                !provisional_storage->kv_store()->rope() &&
                !provisional_kv_allocation && !provisional_rope,
            "outer rollback must release main KV and invalidate its shared RoPE views");
    provisional_storage.reset();

    const std::size_t allocations_after_full = context.buffers().allocation_count();
    PlanResult duplicate_resident = full_plan;
    duplicate_resident.resident_order[1] = duplicate_resident.resident_order[0];
    require(ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                               plain_sources, duplicate_resident, error) == nullptr &&
                error.find("duplicate expert") != std::string::npos &&
                context.buffers().allocation_count() == allocations_after_full,
            "full-mode owner must reject duplicate resident IDs before allocating");

    WindowSources packed_sources = inspect_fixture(packed_fixture);
    const Geometry packed_geometry = owner_test_geometry(packed_fixture, packed_sources);
    const PlanResult packed_full_plan = owner_test_full_plan(packed_geometry, facts);
    const auto expect_invalid_pack = [&](WindowSources invalid, const std::string& reason) {
        rebind_fixture_dense_pack_catalog(invalid);
        require(ModelStorage::open(context, packed_fixture.path.string(), packed_geometry,
                                   invalid, packed_full_plan, error) == nullptr &&
                    error.find(reason) != std::string::npos &&
                    context.buffers().allocation_count() == allocations_after_full,
                "full-mode owner must reject malformed converted rows before allocating: " +
                    error);
    };
    WindowSources remapped_pack_row = inspect_fixture_with_packed_row_padding(
        packed_fixture, directory / "remapped-dense-row.bin", 0,
        packed_fixture.dense_pack_bytes);
    remapped_pack_row.dense_pack_tensors[0].file_offset =
        packed_fixture.dense_pack_bytes;
    require(ModelStorage::open(context, packed_fixture.path.string(), packed_geometry,
                               remapped_pack_row, packed_full_plan, error) == nullptr &&
                error.find("differs from the WeightStore index inspection") !=
                    std::string::npos &&
                context.buffers().allocation_count() == allocations_after_full,
            "full-mode owner must reject a same-size row remapped to another valid file range");

    WindowSources bad_pack_kind = packed_sources;
    bad_pack_kind.dense_pack_tensors[0].kind = 3;
    expect_invalid_pack(std::move(bad_pack_kind), "converted dense-pack inventory");
    WindowSources missing_pack_row = packed_sources;
    missing_pack_row.dense_pack_tensors.clear();
    expect_invalid_pack(std::move(missing_pack_row), "inventory does not match inspected geometry");
    WindowSources duplicate_pack_row = packed_sources;
    duplicate_pack_row.dense_pack_tensors.push_back(
        duplicate_pack_row.dense_pack_tensors.front());
    expect_invalid_pack(std::move(duplicate_pack_row), "converted dense-pack inventory");
    WindowSources overlapping_pack_rows = packed_sources;
    overlapping_pack_rows.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight", 0, 256, 256, 4});
    expect_invalid_pack(std::move(overlapping_pack_rows), "file ranges overlap");
    WindowSources out_of_range_pack = packed_sources;
    out_of_range_pack.dense_pack_tensors[0].file_offset =
        out_of_range_pack.dense_pack_file_size;
    expect_invalid_pack(std::move(out_of_range_pack), "converted dense-pack inventory");
    WindowSources extra_pack_row = inspect_fixture_with_packed_row_padding(
        packed_fixture, directory / "extra-dense-row.bin", 11, 256);
    extra_pack_row.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight",
         11 + packed_fixture.dense_pack_bytes, 256, 256, 4});
    expect_invalid_pack(std::move(extra_pack_row), "inventory does not match inspected geometry");
    WindowSources wrong_pack_lock = packed_sources;
    ++wrong_pack_lock.dense_pack_object_id;
    expect_invalid_pack(std::move(wrong_pack_lock), "inspection lock");

    PlanResult insufficient_pack_padding = packed_full_plan;
    const std::uint64_t pack_remainder =
        packed_geometry.dense_pack_bytes % kWindowPageBytes;
    const std::uint64_t pack_padding =
        pack_remainder == 0 ? 0 : kWindowPageBytes - pack_remainder;
    require(pack_padding != 0, "packed fixture must need arena page padding");
    insufficient_pack_padding.w_bytes =
        insufficient_pack_padding.reserve_bytes +
        insufficient_pack_padding.total_bytes + pack_padding - 1;
    require(ModelStorage::open(context, packed_fixture.path.string(), packed_geometry,
                               packed_sources, insufficient_pack_padding, error) == nullptr &&
                error.find("page padding exceeds") != std::string::npos &&
                context.buffers().allocation_count() == allocations_after_full,
            "full-mode owner must charge dense-pack arena page padding against recomputed free budget");

    WindowPlan packed_window_plan;
    require(plan_windows(packed_sources, facts.max_buffer_length, packed_window_plan, error),
            "plan packed full-mode windows: " + error);
    {
        auto outer_transaction = context.buffers().transaction();
        auto opened = FullModeStorage::open(context, packed_fixture.path.string(),
                                            packed_geometry, packed_sources,
                                            packed_window_plan, error);
        require(opened != nullptr, "open packed full-mode storage for rollback: " + error);
        const BufferRef arena = opened->dense_pack_arena();
        require(arena && arena.length == 3 * kWindowPageBytes &&
                    opened->layout().dense_pack_bytes == packed_fixture.dense_pack_bytes &&
                    opened->layout().dense_pack_arena_bytes == arena.length,
                "full-mode router arena must be padded to its 16 KiB allocation boundary");
        const auto row = std::find_if(
            opened->layout().tensors.begin(), opened->layout().tensors.end(),
            [](const TensorGpuBinding& binding) {
                return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
            });
        require(row != opened->layout().tensors.end() &&
                    row->gpu_address >= arena.gpu_address &&
                    row->gpu_address + row->byte_count <= arena.gpu_address + arena.length,
                "converted router binding must point inside the dense-pack arena");
        FullModeTensorView router_view;
        require(opened->dense_tensor_view("blk.0.ffn_gate_inp.weight",
                                          router_view, error),
                "resolve converted router as a full-mode tensor view: " + error);
        require(router_view.ggml_type == 30 &&
                    router_view.shape == std::vector<std::uint64_t>{2'560, 9} &&
                    router_view.buffer.length == packed_fixture.dense_pack_bytes &&
                    router_view.buffer.gpu_address == row->gpu_address,
                "converted router view must expose its runtime BF16 type, GGUF shape and arena slice");
        std::vector<std::uint8_t> expected(
            static_cast<std::size_t>(packed_fixture.dense_pack_bytes));
        std::ifstream input(packed_fixture.dense_pack_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(expected.data()),
                   static_cast<std::streamsize>(expected.size()));
        require(input.gcount() == static_cast<std::streamsize>(expected.size()) &&
                    std::memcmp(arena.contents, expected.data(), expected.size()) == 0,
                "full-mode dense-pack arena must contain the exact converted router row");
        require(std::memcmp(router_view.buffer.contents, expected.data(),
                            expected.size()) == 0,
                "converted router view must expose the exact dense-pack bytes");
        const auto* arena_bytes = static_cast<const std::uint8_t*>(arena.contents);
        require(std::all_of(arena_bytes + expected.size(),
                            arena_bytes + arena.length,
                            [](std::uint8_t value) { return value == 0; }),
                "full-mode dense-pack arena padding must be initialized");
        require(context.buffers().allocation_count() > allocations_after_full,
                "packed full-mode storage must allocate provisional arena buffers");

        const fs::path padded_pack = directory / "padded-dense.bin";
        WindowSources padded_sources = inspect_fixture_with_packed_row_padding(
            packed_fixture, padded_pack, 37, 53);
        WindowPlan padded_plan;
        require(plan_windows(padded_sources, facts.max_buffer_length, padded_plan, error),
                "plan padded dense-pack source: " + error);
        auto padded_storage = FullModeStorage::open(
            context, packed_fixture.path.string(), packed_geometry, padded_sources,
            padded_plan, error);
        require(padded_storage != nullptr,
                "open a dense pack with unlisted prefix and tail bytes: " + error);
        const BufferRef padded_arena = padded_storage->dense_pack_arena();
        const auto padded_row = std::find_if(
            padded_storage->layout().tensors.begin(),
            padded_storage->layout().tensors.end(),
            [](const TensorGpuBinding& binding) {
                return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
            });
        require(padded_arena && padded_row != padded_storage->layout().tensors.end() &&
                    padded_row->gpu_address == padded_arena.gpu_address &&
                    std::memcmp(padded_arena.contents, expected.data(), expected.size()) == 0 &&
                    std::all_of(
                        static_cast<const std::uint8_t*>(padded_arena.contents) +
                            expected.size(),
                        static_cast<const std::uint8_t*>(padded_arena.contents) +
                            padded_arena.length,
                        [](std::uint8_t value) { return value == 0; }),
                "only the indexed dense-pack row may be copied into the GPU arena");
    }
    require(context.buffers().allocation_count() == allocations_after_full,
            "outer storage transaction must roll back the converted-router arena too");

    auto packed_storage = ModelStorage::open(context, packed_fixture.path.string(),
                                             packed_geometry, packed_sources,
                                             packed_full_plan, error);
    require(packed_storage != nullptr &&
                std::string(packed_storage->mode()) == "full" &&
                packed_storage->full_mode_warmed(),
            "full-mode owner must load and warm validated converted router rows: " + error);

    const fs::path stray_pack = plain_directory / "unlisted-dense.bin";
    {
        std::ofstream output(stray_pack, std::ios::binary);
        output.put('x');
        require(static_cast<bool>(output), "write an unlisted dense-pack byte");
    }
    WindowSources unlisted_pack_sources = plain_sources;
    set_dense_pack_file(unlisted_pack_sources, stray_pack);
    rebind_fixture_dense_pack_catalog(unlisted_pack_sources);
    auto unlisted_pack_storage = ModelStorage::open(
        context, plain_fixture.path.string(), plain_geometry, unlisted_pack_sources,
        full_plan, error);
    require(unlisted_pack_storage != nullptr &&
                std::string(unlisted_pack_storage->mode()) == "full",
            "full-mode owner may ignore unlisted dense.bin bytes when no rows are converted: " +
                error);
    WindowPlan unlisted_pack_plan;
    require(plan_windows(unlisted_pack_sources, facts.max_buffer_length,
                         unlisted_pack_plan, error),
            "plan a non-empty unlisted dense-pack source: " + error);
    auto unlisted_pack_full = FullModeStorage::open(
        context, plain_fixture.path.string(), plain_geometry, unlisted_pack_sources,
        unlisted_pack_plan, error);
    require(unlisted_pack_full != nullptr &&
                !unlisted_pack_full->dense_pack_arena() &&
                unlisted_pack_full->layout().dense_pack_bytes == 0 &&
                std::none_of(unlisted_pack_full->layout().tensors.begin(),
                             unlisted_pack_full->layout().tensors.end(),
                             [](const TensorGpuBinding& binding) {
                                 return binding.tensor_name.starts_with("blk.") &&
                                        binding.tensor_name.find("ffn_gate_inp") !=
                                            std::string::npos;
                             }),
            "full mode must not allocate or bind bytes from an unlisted dense pack");
    const std::size_t allocations_before_mtp = context.buffers().allocation_count();

    PlanResult mtp_plan = full_plan;
    mtp_plan.mtp_bytes = 1;
    require(ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                               plain_sources, mtp_plan, error) == nullptr &&
                error.find("inspected MTP runtime sources") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_mtp,
            "storage owner must require an inspected MTP catalog before allocating");
    PlanResult mtp_kv_plan = full_plan;
    mtp_kv_plan.mtp_kv_bytes = 1;
    require(ModelStorage::open(context, plain_fixture.path.string(), plain_geometry,
                               plain_sources, mtp_kv_plan, error) == nullptr &&
                error.find("inspected MTP runtime sources") != std::string::npos &&
                context.buffers().allocation_count() == allocations_before_mtp,
            "storage owner must require an inspected MTP catalog for drafter KV");

    const PlanResult tiered_plan = owner_test_tiered_plan(packed_geometry, facts);
    const std::size_t allocations_before_tiered_failure =
        context.buffers().allocation_count();
    const std::uint64_t bytes_before_tiered_failure =
        context.buffers().allocated_size();
    WindowSources unavailable_tiered_sources;
    require(ModelStorage::open(context, packed_fixture.path.string(),
                               packed_geometry, unavailable_tiered_sources,
                               tiered_plan, error) == nullptr &&
                !error.empty() &&
                context.buffers().allocation_count() ==
                    allocations_before_tiered_failure &&
                context.buffers().allocated_size() == bytes_before_tiered_failure,
            "tiered model-storage failure must roll back its provisional main KV allocation");

    auto tiered_storage = ModelStorage::open(context, packed_fixture.path.string(),
                                             packed_geometry, packed_sources,
                                             tiered_plan, error);
    require(tiered_storage != nullptr, "open tiered model storage owner: " + error);
    require(std::string(tiered_storage->mode()) == "tiered" &&
                tiered_storage->resident_experts() == tiered_plan.experts_resident &&
                tiered_storage->window_count() == 0 && !tiered_storage->full_mode_warmed(),
            "tiered owner must report copied resident storage without full-mode warm-up");
    require(tiered_storage->ple_reader() == nullptr,
            "tiered ModelStorage must not publish a reader for its closed startup mapping");
    require(tiered_storage->kv_store() != nullptr &&
                tiered_storage->kv_store()->allocation().length ==
                    tiered_plan.kv_bytes &&
                tiered_storage->kv_store()->rope(),
            "tiered owner must publish its planned main KV and shared RoPE allocation");

    check_window_ple_staging(context, *full_ple, directory);
}

void run_multi_converted_view_offset_test(Context& context,
                                          const fs::path& parent_directory) {
    const fs::path directory = parent_directory / "multi-converted-view";
    fs::create_directory(directory);
    const Fixture fixture = write_fixture(directory, false, false, true, 0, true);
    WindowSources sources = inspect_fixture(fixture);
    Geometry geometry = owner_test_geometry(fixture, sources);
    WindowPlan plan;
    std::string error;
    require(plan_windows(sources, context.device_facts().max_buffer_length,
                         plan, error),
            "plan two converted router rows: " + error);
    auto storage = FullModeStorage::open(context, fixture.path.string(),
                                         geometry, sources, plan, error);
    require(storage != nullptr,
            "open two converted router rows for tensor-view offset check: " + error);

    FullModeTensorView first;
    FullModeTensorView second;
    require(storage->dense_tensor_view(fixture.router_tensor_name, first, error) &&
                storage->dense_tensor_view(fixture.second_router_tensor_name, second, error),
            "resolve both converted router rows: " + error);
    require(first.ggml_type == 30 && second.ggml_type == 30 &&
                first.shape == std::vector<std::uint64_t>{2'560, 9} &&
                second.shape == std::vector<std::uint64_t>{2'560} &&
                first.buffer.length == fixture.router_row_bytes &&
                second.buffer.length == fixture.second_router_row_bytes &&
                second.buffer.gpu_address - first.buffer.gpu_address ==
                    fixture.router_row_bytes &&
                static_cast<const std::uint8_t*>(second.buffer.contents) ==
                    static_cast<const std::uint8_t*>(first.buffer.contents) +
                        fixture.router_row_bytes,
            "second converted router view must preserve its nonzero arena offset and metadata");

    const auto* bytes =
        static_cast<const std::uint8_t*>(second.buffer.contents);
    for (std::size_t i = 0; i < fixture.second_router_row_bytes; ++i) {
        const auto expected = static_cast<std::uint8_t>(0x20 + (i % 0x50));
        require(bytes[i] == expected,
                "second converted router view must begin at its own packed row data");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool never_wrap_only = argc == 3 && std::string(argv[2]) == "--never-wrap";
        const bool tiered_copy_only = argc == 3 && std::string(argv[2]) == "--tiered-copy";
        const bool model_owner_only = argc == 3 && std::string(argv[2]) == "--model-owner";
        require(argc == 2 || never_wrap_only || tiered_copy_only || model_owner_only,
                "usage: metal_window_storage_test strata.metallib "
                "[--never-wrap|--tiered-copy|--model-owner]");
        if (argc == 2) test_warmup_wave_barrier();
        char directory_template[] = "/tmp/strata-window-storage.XXXXXX";
        char* directory_name = mkdtemp(directory_template);
        require(directory_name != nullptr, "create temporary fixture directory");
        const fs::path directory(directory_name);
        const Fixture fixture = write_fixture(directory);
        WindowSources sources = inspect_fixture(fixture);
        std::string error;

        ContextOptions options;
        options.metallib = argv[1];
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal context: " + error);
        WindowPlan plan;
        require(plan_windows(sources, context->device_facts().max_buffer_length, plan, error),
                "plan synthetic GGUF with the live Metal limit: " + error);
        Geometry geometry = owner_test_geometry(fixture, sources);
        if (model_owner_only) {
            run_model_storage_owner_test(*context, directory, fixture);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_model_storage_owner_test: OK\n";
            return 0;
        }
        require(compute_wrapped_unread(sources, geometry.wrapped_unread_bytes, error),
                "compute planned unread bytes: " + error);
        if (tiered_copy_only) {
            run_tiered_copy_test(*context, fixture, std::move(sources), geometry);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_tiered_copy_test: OK\n";
            return 0;
        }
        WindowSources wrong_file_identity = sources;
        ++wrong_file_identity.files[0].file_object_id;
        require(FullModeStorage::open(*context, fixture.path.string(), geometry,
                                      wrong_file_identity, plan, error) == nullptr &&
                    (error.find("inspected file identity") != std::string::npos ||
                     error.find("missing or mismatched WeightStore inspection lock") !=
                         std::string::npos) &&
                    context->buffers().allocation_count() == 0,
                "full-mode storage must reject a GGUF identity that differs from inspection: " + error);
        WindowSources mislabeled_sources = sources;
        auto& mislabeled_tensors = mislabeled_sources.files[0].tensors;
        const auto dense_source = std::find_if(
            mislabeled_tensors.begin(), mislabeled_tensors.end(), [](const WindowTensorSource& tensor) {
                return tensor.tensor_name == "token_embd.weight";
            });
        require(dense_source != mislabeled_tensors.end(),
                "synthetic dense tensor must appear in the included source inventory");
        const WindowTensorSource dense_range = *dense_source;
        mislabeled_tensors.erase(dense_source);
        mislabeled_sources.files[0].excluded_tensors.push_back(
            {dense_range.tensor_name, dense_range.file_offset, dense_range.byte_count,
             WindowTensorExclusionReason::converted});
        require(FullModeStorage::open(*context, fixture.path.string(), geometry,
                                      mislabeled_sources, plan, error) == nullptr &&
                    error.find("only a supported F32 router") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a non-router falsely classified as converted");

        WindowSources incomplete_sources = sources;
        incomplete_sources.files[0].tensors.pop_back();
        require(FullModeStorage::open(*context, fixture.path.string(), geometry,
                                      incomplete_sources, plan, error) == nullptr &&
                    error.find("accounted for exactly once") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a source inventory that omits a GGUF tensor");
        Geometry wrong_geometry = geometry;
        ++wrong_geometry.wrapped_unread_bytes;
        require(FullModeStorage::open(*context, fixture.path.string(), wrong_geometry,
                                      sources, plan, error) == nullptr &&
                    error.find("wrapped_unread_bytes") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a GGUF window union absent from the memory plan");
        Geometry wrong_shard = geometry;
        wrong_shard.shards[0].bytes += kWindowPageBytes;
        require(FullModeStorage::open(*context, fixture.path.string(), wrong_shard,
                                      sources, plan, error) == nullptr &&
                    error.find("shard name or size") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject shard sizes that differ from the memory plan");
        wrong_shard = geometry;
        wrong_shard.shards[0].name = "different.gguf";
        require(FullModeStorage::open(*context, fixture.path.string(), wrong_shard,
                                      sources, plan, error) == nullptr &&
                    error.find("shard name or size") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject shard names that differ from the memory plan");
        WindowPlan incomplete_plan = plan;
        incomplete_plan.files[0].windows[0].end -= kWindowPageBytes;
        require(FullModeStorage::open(*context, fixture.path.string(), geometry,
                                      sources, incomplete_plan, error) == nullptr &&
                    error.find("exact planned tensor union") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a window plan that omits part of the planned union before allocation");
        WindowPlan ple_overwrap_plan = plan;
        ple_overwrap_plan.files[0].windows[0].end += kWindowPageBytes;
        require(FullModeStorage::open(*context, fixture.path.string(), geometry,
                                      sources, ple_overwrap_plan, error) == nullptr &&
                    error.find("exact planned tensor union") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a window extended into unplanned PLE payload pages");

        const Fixture overlapping_fixture = write_fixture(directory, true);
        const WindowSources overlapping_sources = inspect_fixture(overlapping_fixture);
        Geometry overlapping_geometry =
            owner_test_geometry(overlapping_fixture, overlapping_sources);
        WindowPlan overlapping_plan;
        require(plan_windows(overlapping_sources, context->device_facts().max_buffer_length,
                             overlapping_plan, error) &&
                    compute_wrapped_unread(overlapping_sources,
                                           overlapping_geometry.wrapped_unread_bytes, error),
                "plan overlapping-range fixture: " + error);
        require(FullModeStorage::open(*context, overlapping_fixture.path.string(),
                                      overlapping_geometry, overlapping_sources,
                                      overlapping_plan, error) == nullptr &&
                    error.find("tensor payload ranges overlap") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject overlapping source and excluded tensor payloads");

        const Fixture bridge_fixture = write_fixture(directory, false, true);
        const WindowSources bridge_sources = inspect_fixture(bridge_fixture);
        Geometry bridge_geometry = owner_test_geometry(bridge_fixture, bridge_sources);
        WindowPlan bridge_plan;
        require(plan_windows(bridge_sources, context->device_facts().max_buffer_length,
                             bridge_plan, error) &&
                    compute_wrapped_unread(bridge_sources,
                                           bridge_geometry.wrapped_unread_bytes, error),
                "plan PLE-bridging fixture: " + error);
        require(FullModeStorage::open(*context, bridge_fixture.path.string(),
                                      bridge_geometry, bridge_sources, bridge_plan, error) == nullptr &&
                    error.find("more than one PLE boundary page") != std::string::npos &&
                    context->buffers().allocation_count() == 0,
                "storage must reject a valid tensor plan whose window would wrap the PLE payload");

        auto storage = FullModeStorage::open(
            *context, fixture.path.string(), geometry, sources, plan, error);
        require(storage != nullptr, "map full-mode GGUF windows: " + error);
        require(storage->windows().size() == 1 && storage->windows()[0].size() == 1 &&
                    storage->windows()[0][0].length > 8 * kWindowPageBytes &&
                    storage->windows()[0][0].length % kWindowPageBytes == 0,
                "one multi-page, page-rounded GGUF window should be wrapped");

        const FullModeLayout& layout = storage->layout();
        FullModeTensorView token_view;
        require(storage->dense_tensor_view("token_embd.weight", token_view, error),
                "resolve a whole GGUF tensor view: " + error);
        require(token_view.ggml_type == 0 &&
                    token_view.shape == std::vector<std::uint64_t>{64, 64} &&
                    token_view.buffer &&
                    token_view.buffer.length == 64 * 64 * sizeof(float) &&
                    *static_cast<const std::uint8_t*>(token_view.buffer.contents) == 0x4d,
                "dense tensor view must retain GGML type, shape, exact extent and mapped bytes");
        FullModeTensorView rejected_view = token_view;
        require(!storage->dense_tensor_view("blk.0.ffn_gate_exps.weight",
                                            rejected_view, error) &&
                    !rejected_view.buffer && rejected_view.shape.empty() &&
                    error.find("per-expert slabs") != std::string::npos,
                "dense tensor lookup must reject per-expert slab collections");
        require(!storage->dense_tensor_view("per_layer_token_embd.weight",
                                            rejected_view, error) &&
                    !rejected_view.buffer &&
                    error.find("excluded from the dense view catalog") !=
                        std::string::npos,
                "dense tensor lookup must reject the separately owned PLE payload");
        require(!storage->dense_tensor_view("missing.weight", rejected_view, error) &&
                    !rejected_view.buffer &&
                    error.find("no tensor named") != std::string::npos,
                "dense tensor lookup must reject unknown GGUF names");

        strata::GgufFile fixture_model(fixture.path.string());
        const strata::TensorInfo* ple = fixture_model.find("per_layer_token_embd.weight");
        require(ple != nullptr, "synthetic GGUF must contain a PLE tensor");
        const std::uint64_t ple_start = fixture_model.data_start() + ple->offset;
        const std::uint64_t ple_end = ple_start + strata::tensor_payload_bytes(*ple);
        require(std::none_of(sources.files[0].tensors.begin(), sources.files[0].tensors.end(),
                             [](const WindowTensorSource& tensor) {
                                 return tensor.tensor_name == "per_layer_token_embd.weight";
                             }) &&
                    std::none_of(plan.files[0].slices.begin(), plan.files[0].slices.end(),
                                 [](const WindowSlice& slice) {
                                     return slice.tensor_name == "per_layer_token_embd.weight";
                                 }) &&
                    std::none_of(layout.tensors.begin(), layout.tensors.end(),
                                 [](const TensorGpuBinding& binding) {
                                     return binding.tensor_name == "per_layer_token_embd.weight";
                                 }),
                "the PLE tensor must not enter full-mode GPU address bindings");
        const auto router_binding = std::find_if(
            layout.tensors.begin(), layout.tensors.end(),
            [](const TensorGpuBinding& binding) {
                return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
            });
        const BufferRef dense_pack_arena = storage->dense_pack_arena();
        require(router_binding != layout.tensors.end() && dense_pack_arena &&
                    dense_pack_arena.length == 3 * kWindowPageBytes &&
                    router_binding->gpu_address == dense_pack_arena.gpu_address &&
                    router_binding->byte_count == fixture.dense_pack_bytes,
                "converted router address binding must point to the aligned BF16 arena");
        std::vector<std::uint8_t> packed_router(
            static_cast<std::size_t>(fixture.dense_pack_bytes));
        {
            std::ifstream input(fixture.dense_pack_path, std::ios::binary);
            input.read(reinterpret_cast<char*>(packed_router.data()),
                       static_cast<std::streamsize>(packed_router.size()));
            require(input.gcount() == static_cast<std::streamsize>(packed_router.size()) &&
                        std::memcmp(dense_pack_arena.contents, packed_router.data(),
                                    packed_router.size()) == 0,
                    "full-mode dense-pack arena must contain the converted BF16 router bytes");
        }
        bool saw_ple_padding = false;
        for (std::size_t index = 0; index < plan.files[0].windows.size(); ++index) {
            const FileWindow& file_window = plan.files[0].windows[index];
            const std::uint64_t overlap_start = std::max(file_window.start, ple_start);
            const std::uint64_t overlap_end = std::min(file_window.end, ple_end);
            if (overlap_end <= overlap_start) continue;
            const std::uint64_t overlap = overlap_end - overlap_start;
            require(overlap <= kWindowPageBytes,
                    "a full-mode no-copy window may overlap only PLE boundary padding");
            saw_ple_padding = true;
            const BufferRef& window = storage->windows()[0][index];
            const std::uint64_t mapped_ple_start =
                window.gpu_address + (overlap_start - file_window.start);
            const std::uint64_t mapped_ple_end = mapped_ple_start + overlap;
            for (std::uint64_t address : layout.expert_addresses) {
                require(address == 0 || address < mapped_ple_start || address >= mapped_ple_end,
                        "an expert address-table entry must not point into the PLE payload");
            }
        }
        require(saw_ple_padding,
                "fixture must exercise the page-rounded overlap at the excluded PLE boundary");
        if (never_wrap_only) {
            storage.reset();
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_never_wrap_test: OK\n";
            return 0;
        }
        require(layout.tensors.size() == 29 &&
                    layout.expert_addresses.size() ==
                        static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts *
                            kFullModeExpertRoles &&
                    layout.resident.size() ==
                        static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts,
                "full-mode tensor, address and resident tables have unexpected sizes");
        std::vector<std::uint64_t> expected;
        expected.reserve(9 * kFullModeExpertRoles);
        for (std::size_t expert = 0; expert < 9; ++expert) {
            for (std::size_t role = 0; role < kFullModeExpertRoles; ++role) {
                expected.push_back(layout.tensors[role * 9 + expert].gpu_address);
            }
        }
        require(std::equal(expected.begin(), expected.end(), layout.expert_addresses.begin()),
                "expert GPU addresses must follow layer, expert, gate/up/down order");

        const BufferRef address_table = storage->expert_address_table();
        const BufferRef resident_table = storage->resident_table();
        require(address_table && resident_table, "full-mode tables must be shared Metal buffers");
        const auto* mapped_addresses = static_cast<const std::uint64_t*>(address_table.contents);
        const auto* mapped_resident = static_cast<const std::int32_t*>(resident_table.contents);
        require(std::equal(expected.begin(), expected.end(), mapped_addresses),
                "shared expert address table must match its checked host layout");
        require(std::all_of(mapped_resident, mapped_resident + 9,
                            [](std::int32_t value) { return value == 1; }) &&
                    mapped_resident[9] == 0,
                "the full-mode resident table must mark active experts and clear unused capacity");
        std::vector<std::pair<std::uint32_t, std::uint32_t>> expert_order;
        for (std::uint32_t expert = 9; expert != 0; --expert) {
            expert_order.emplace_back(0, expert - 1);
        }
        require(storage->warm_up(expert_order, error),
                "CPU and GPU page warm-up: " + error);

        const BufferRef live_window = storage->windows()[0][0];
        const std::uint64_t live_window_start = plan.files[0].windows[0].start;
        storage.reset();
        const auto* payload = static_cast<const std::uint8_t*>(live_window.contents);
        require(payload[fixture.data_start + fixture.offsets[0] - live_window_start] == 0x30 &&
                    payload[fixture.data_start + fixture.offsets[2] - live_window_start] == 0x42 &&
                    payload[fixture.data_start + fixture.offsets[2] +
                            9 * fixture.expert_slab_bytes - 1 - live_window_start] == 0x4a,
                "Context must keep the GGUF mmap alive after the storage handle is destroyed");
        require(context->buffers().commit_residency(error), "commit full-mode residency: " + error);
        run_multi_converted_view_offset_test(*context, directory);
        require(context->buffers().commit_residency(error),
                "commit multi-row converted tensor-view residency: " + error);
        context.reset();

        std::error_code ignored;
        fs::remove_all(directory, ignored);
        std::cout << "metal_window_storage_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_window_storage_test: " << exception.what() << '\n';
        return 1;
    }
}
