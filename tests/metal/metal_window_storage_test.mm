// tests/metal/metal_window_storage_test.mm - live GGUF mmap-to-Metal full-mode mapping.
#include "strata/metal/window.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/model_storage.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/experts.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window_ple_dispatch.hpp"
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"
#include "strata/metal/window_ple_stage.hpp"
#include "strata/metal/window_scratch.hpp"
#include "strata/metal/session_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "window_source_lock_test_access.hpp"
#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <Metal/Metal.h>
#include <libproc.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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

constexpr std::uint32_t kGgmlTypeQ3K = 11u;

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
static_assert(std::is_same_v<decltype(std::declval<const ReadOnlyBufferRef&>().contents()),
                             const void*>,
              "full-mode model views must expose only const CPU mappings");
static_assert(!std::is_convertible_v<ReadOnlyBufferRef, BufferRef>,
              "read-only model views must not convert to mutable buffer references");
static_assert(std::is_same_v<decltype(std::declval<const ModelStorage&>().full_mode_storage()),
                             const FullModeStorage*>,
              "ModelStorage must expose only a borrowed const full-mode resolver");
static_assert(std::is_same_v<decltype(std::declval<ModelStorage&>().tiered_storage()),
                             TieredStorage*>,
              "ModelStorage must expose only a borrowed tiered resolver source");

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool event_has_value(const Events& events, std::uint64_t expected) {
    const auto value = events.signaled_value();
    return value && *value == expected;
}

bool event_has_reached_value(const Events& events, std::uint64_t expected) {
    const auto value = events.signaled_value();
    return value && *value >= expected;
}

struct TestQ8_1 {
    std::uint16_t ds[2]{};
    std::int8_t qs[32]{};
};
static_assert(sizeof(TestQ8_1) == 36);

struct TestMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(TestMoeExpertAddress) == 24);

struct TestMoeGroupedParams {
    std::uint32_t n_expert = 0;
    std::uint32_t layer = 0;
    std::uint32_t n_rows = 0;
};
static_assert(sizeof(TestMoeGroupedParams) == 12);

class CommandGatesCleanup {
public:
    CommandGatesCleanup(Events* layer_gate, std::uint64_t layer_arrival_value,
                        std::uint64_t layer_release_value,
                        Events* completion_gate,
                        std::uint64_t completion_arrival_value,
                        std::uint64_t completion_value,
                        id<MTLCommandBuffer> command_buffer,
                        std::function<void()> prepare_failed_release = {})
        : events_(layer_gate), completion_gate_(completion_gate),
          completion_arrival_value_(completion_arrival_value),
          completion_value_(completion_value), command_buffer_(command_buffer),
          prepare_failed_release_(std::move(prepare_failed_release)) {
        layer_gates_.push_back(
            {layer_gate, layer_arrival_value, layer_release_value});
    }

    ~CommandGatesCleanup() {
        if (pending_) {
            std::string cleanup_error;
            if (!finish_and_drain_impl(cleanup_error, true)) {
                std::cerr << "Metal test command-buffer cleanup failed: "
                          << cleanup_error << '\n';
            }
        }
    }

    void add_layer_gate(Events* layer_gate, std::uint64_t arrival_value,
                        std::uint64_t release_value) {
        layer_gates_.push_back({layer_gate, arrival_value, release_value});
    }

    bool release_layer(std::string& error) {
        return release_layer_impl(error, false);
    }

    void mark_layer_released() noexcept {
        if (!layer_gates_.empty()) layer_gates_.erase(layer_gates_.begin());
    }

    bool finish_and_drain(std::string& error) {
        return finish_and_drain_impl(error, false);
    }

private:
    struct LayerGate {
        Events* events = nullptr;
        std::uint64_t arrival_value = 0;
        std::uint64_t release_value = 0;
    };

    bool release_layer_impl(std::string& error, bool failed_request) {
        if (layer_gates_.empty()) return true;
        const LayerGate gate = layer_gates_.front();
        if (gate.events == nullptr ||
            !gate.events->wait(gate.arrival_value, 5000, error)) {
            if (failed_request) {
                if (failure_guard_prepared_) {
                    (void)release_after_safe_failure(error);
                } else {
                    (void)release_after_terminal_command_buffer(error);
                }
            }
            return false;
        }
        if (failed_request && !failure_guard_prepared_) {
            if (prepare_failed_release_) prepare_failed_release_();
            failure_guard_prepared_ = true;
        }
        if (!gate.events->signal_host(gate.release_value, error)) {
            if (failed_request) (void)release_after_safe_failure(error);
            return false;
        }
        layer_gates_.erase(layer_gates_.begin());
        return true;
    }

    bool finish_and_drain_impl(std::string& error, bool failed_request) {
        bool succeeded = true;
        while (!layer_gates_.empty()) {
            if (!release_layer_impl(error, failed_request)) {
                succeeded = false;
                break;
            }
        }
        if (completion_pending_) {
            std::string completion_error;
            const bool completion_arrived =
                completion_gate_->wait(completion_arrival_value_, 5000,
                                       completion_error);
            if (!completion_arrived ||
                !completion_gate_->signal_host(completion_value_, completion_error)) {
                if (error.empty()) error = completion_error;
                if (failed_request) {
                    if (failure_guard_prepared_) {
                        (void)release_after_safe_failure(error);
                    } else {
                        (void)release_after_terminal_command_buffer(error);
                    }
                }
                succeeded = false;
            } else {
                completion_pending_ = false;
            }
        }
        if (succeeded) {
            [command_buffer_ waitUntilCompleted];
            pending_ = false;
            if (command_buffer_.status == MTLCommandBufferStatusError) {
                error = "Metal test command buffer failed while draining";
                return false;
            }
        }
        return succeeded;
    }

    bool release_after_terminal_command_buffer(std::string& error) {
        const MTLCommandBufferStatus status = command_buffer_.status;
        if (status != MTLCommandBufferStatusCompleted &&
            status != MTLCommandBufferStatusError) {
            return false;
        }
        std::string drain_error;
        if (!events_->release_gpu_waits(5000, drain_error)) {
            if (!error.empty()) error += "; ";
            error += "terminal command-buffer wait release failed: " + drain_error;
            return false;
        }
        pending_ = false;
        return true;
    }

    bool release_after_safe_failure(std::string& error) {
        if (events_ == nullptr) return false;
        std::string drain_error;
        if (!events_->release_gpu_waits(5000, drain_error)) {
            if (!error.empty()) error += "; ";
            error += "terminal guarded-wait release failed: " + drain_error;
            return false;
        }
        pending_ = false;
        return true;
    }

    std::vector<LayerGate> layer_gates_;
    Events* events_ = nullptr;
    Events* completion_gate_ = nullptr;
    std::uint64_t completion_arrival_value_ = 0;
    std::uint64_t completion_value_ = 0;
    id<MTLCommandBuffer> command_buffer_;
    std::function<void()> prepare_failed_release_;
    bool completion_pending_ = true;
    bool failure_guard_prepared_ = false;
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
    std::uint32_t expert_layers = 1;
    std::array<std::uint64_t, 3> offsets{};
    std::vector<std::array<std::uint64_t, 3>> layer_offsets;
    std::uint64_t router_offset = 0;
    std::uint64_t second_router_offset = 0;
    std::uint64_t dense_offset = 0;
    std::uint64_t ple_offset = 0;
    std::uint64_t post_ple_offset = 0;
};

Fixture write_fixture(const fs::path& directory, bool overlap_ple = false,
                      bool bridge_ple = false, bool include_router = true,
                      std::uint32_t router_layer = 0,
                      bool include_second_router = false,
                      std::uint32_t expert_layers = 1) {
    constexpr std::uint64_t kExpertSlabBytes = 18ull * 3'641;
    constexpr std::uint64_t kExperts = 9;
    constexpr std::uint64_t kTensorBytes = kExperts * kExpertSlabBytes;
    constexpr std::uint64_t kRouterBytes = 2'560 * kExperts * sizeof(float);
    constexpr std::uint64_t kDenseBytes = 64 * 64 * sizeof(float);
    constexpr std::uint64_t kPleRows = 728;
    constexpr std::uint64_t kPleBytes = kPleRows * strata::kernels::PLE_ROW_BYTES;
    constexpr std::uint64_t kPostPleBytes = 1'024 * sizeof(float);
    Fixture fixture;
    require(expert_layers != 0, "synthetic fixture must contain an expert layer");
    fixture.expert_layers = expert_layers;
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
    fixture.path = directory / (expert_layers > 1 ? "multi-layer.gguf" :
                                include_second_router ? "multi-router.gguf" :
                                overlap_ple ? "overlap.gguf" :
                                bridge_ple ? "bridge.gguf" :
                                include_router ? "tiny.gguf" : "plain.gguf");
    fixture.router_tensor_name =
        "blk." + std::to_string(router_layer) + ".ffn_gate_inp.weight";
    fixture.second_router_tensor_name =
        include_second_router ? "blk." + std::to_string(router_layer) +
                                    ".ffn_gate_inp_shexp.weight"
                              : "";
    fixture.layer_offsets.resize(expert_layers);
    std::uint64_t next_expert_offset =
        (fixture.router_bytes + 31) & ~std::uint64_t{31};
    for (std::uint32_t layer = 0; layer < expert_layers; ++layer) {
        for (std::size_t role = 0; role < 3; ++role) {
            fixture.layer_offsets[layer][role] = next_expert_offset;
            next_expert_offset =
                (next_expert_offset + kTensorBytes + 31) & ~std::uint64_t{31};
        }
    }
    fixture.offsets = fixture.layer_offsets.front();
    fixture.dense_offset = next_expert_offset;
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
    put_u64(bytes, 2 + static_cast<std::uint64_t>(expert_layers) * 3 +
                       static_cast<std::uint64_t>(include_router) +
                       static_cast<std::uint64_t>(include_router &&
                                                  include_second_router) +
                       static_cast<std::uint64_t>(bridge_ple));
    put_u64(bytes, 0);
    const std::array<std::string, 3> role_names = {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    };
    for (std::uint32_t layer = 0; layer < expert_layers; ++layer) {
        for (std::size_t role = 0; role < role_names.size(); ++role) {
            put_string(bytes, "blk." + std::to_string(layer) + "." +
                                  role_names[role]);
            put_u32(bytes, 3);
            put_u64(bytes, 64);
            put_u64(bytes, 3'641);
            put_u64(bytes, kExperts);
            put_u32(bytes, 42);
            put_u64(bytes, fixture.layer_offsets[layer][role]);
        }
    }
    put_string(bytes, "token_embd.weight");
    put_u32(bytes, 2);
    put_u64(bytes, 64);
    put_u64(bytes, 64);
    put_u32(bytes, 0);
    put_u64(bytes, fixture.dense_offset);
    if (include_router) {
        put_string(bytes, fixture.router_tensor_name);
        put_u32(bytes, 2);
        put_u64(bytes, 2'560);
        put_u64(bytes, kExperts);
        put_u32(bytes, 0);
        put_u64(bytes, fixture.router_offset);
    }
    put_string(bytes, "per_layer_token_embd.weight");
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
    for (std::uint32_t layer = 0; layer < expert_layers; ++layer) {
        for (std::size_t tensor = 0; tensor < 3; ++tensor) {
            for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
                const std::size_t begin = static_cast<std::size_t>(
                    fixture.data_start + fixture.layer_offsets[layer][tensor] +
                    expert * kExpertSlabBytes);
                const auto value = static_cast<std::uint8_t>(
                    0x30 + layer * 27 + tensor * kExperts + expert);
                std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                          bytes.begin() + static_cast<std::ptrdiff_t>(begin + kExpertSlabBytes),
                          value);
            }
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

Fixture write_fullmode_handoff_fixture(const fs::path& directory) {
    constexpr std::uint64_t kExperts = 256u;
    constexpr std::uint64_t kGateUpElements = 2'560u * 640u;
    constexpr std::uint64_t kDownElements = 640u * 2'560u;
    constexpr std::uint64_t kExpertSlabBytes =
        (kGateUpElements / 64u) * sizeof(block_q2_0);
    constexpr std::uint64_t kRouterSourceBytes =
        2'560u * kExperts * sizeof(float);
    constexpr std::uint64_t kRouterPackedBytes =
        2'560u * kExperts * sizeof(std::uint16_t);
    constexpr std::uint64_t kPleRows = strata::kernels::PLE_TABLE_ROWS;
    constexpr std::uint64_t kPleBytes =
        kPleRows * strata::kernels::PLE_ROW_BYTES;
    static_assert(kGateUpElements == kDownElements);
    static_assert(kExpertSlabBytes == 460'800u);

    struct Tensor {
        std::string name;
        std::vector<std::uint64_t> shape;
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
    };

    Fixture fixture;
    fixture.path = directory / "fullmode-handoff.gguf";
    fixture.dense_pack_path = directory / "fullmode-handoff-dense.bin";
    fixture.expert_count = kExperts;
    fixture.expert_slab_bytes = kExpertSlabBytes;
    fixture.expert_layers = 1u;
    fixture.router_tensor_name = "blk.0.ffn_gate_inp.weight";
    fixture.router_bytes = kRouterSourceBytes;
    fixture.router_row_bytes = kRouterPackedBytes;
    fixture.dense_pack_bytes = kRouterPackedBytes;
    fixture.layer_offsets.resize(1u);

    const auto align32 = [](std::uint64_t value) {
        return (value + 31u) & ~std::uint64_t{31u};
    };
    std::uint64_t cursor = 0;
    std::vector<Tensor> tensors{
        {"blk.0.ffn_gate_exps.weight", {2'560u, 640u, kExperts}, 42u, 0u},
        {"blk.0.ffn_up_exps.weight", {2'560u, 640u, kExperts}, 42u, 0u},
        {"blk.0.ffn_down_exps.weight", {640u, 2'560u, kExperts}, 42u, 0u},
        {"token_embd.weight", {2'560u, 1u}, kGgmlTypeQ3K, 0u},
        {fixture.router_tensor_name, {2'560u, kExperts}, 0u, 0u},
        {"per_layer_token_embd.weight",
         {strata::kernels::PLE_HEAD_DIM, kPleRows}, 20u, 0u},
    };
    for (Tensor& tensor : tensors) {
        cursor = align32(cursor);
        tensor.offset = cursor;
        if (tensor.name.ends_with("ffn_gate_exps.weight") ||
            tensor.name.ends_with("ffn_up_exps.weight") ||
            tensor.name.ends_with("ffn_down_exps.weight")) {
            cursor += kExperts * kExpertSlabBytes;
        } else if (tensor.name == "token_embd.weight") {
            fixture.dense_offset = cursor;
            cursor += (2'560u / 256u) * sizeof(block_q3_K);
        } else if (tensor.name == fixture.router_tensor_name) {
            fixture.router_offset = cursor;
            cursor += kRouterSourceBytes;
        } else {
            fixture.ple_offset = cursor;
            cursor += kPleBytes;
        }
    }
    fixture.offsets = {tensors[0].offset, tensors[1].offset, tensors[2].offset};
    fixture.layer_offsets[0] = fixture.offsets;
    fixture.data_start = 0;
    std::vector<std::uint8_t> header;
    put_u32(header, 0x46554747u);
    put_u32(header, 3u);
    put_u64(header, tensors.size());
    put_u64(header, 0u);
    for (const Tensor& tensor : tensors) {
        put_string(header, tensor.name);
        put_u32(header, static_cast<std::uint32_t>(tensor.shape.size()));
        for (const std::uint64_t dimension : tensor.shape) {
            put_u64(header, dimension);
        }
        put_u32(header, tensor.type);
        put_u64(header, tensor.offset);
    }
    fixture.data_start = align32(header.size());
    fixture.file_size = fixture.data_start + cursor;

    const int descriptor =
        ::open(fixture.path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
               0600);
    require(descriptor >= 0, "create sparse full-mode handoff GGUF fixture");
    try {
        write_all_at(descriptor, header.data(), header.size(), 0u);
        require(fixture.file_size <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<off_t>::max()) &&
                    ::ftruncate(descriptor,
                                static_cast<off_t>(fixture.file_size)) == 0,
                "size sparse full-mode handoff GGUF fixture");

        std::vector<block_q3_K> embedding_row(2'560u / 256u);
        for (block_q3_K& block : embedding_row) {
            std::fill_n(block.hmask, std::size(block.hmask), 0u);
            std::fill_n(block.qs, std::size(block.qs), 0x55u);
            std::fill_n(block.scales, std::size(block.scales), 0u);
            block.d = strata::kernels::f16_from_f32(0.001f);
        }
        write_all_at(descriptor, embedding_row.data(),
                     embedding_row.size() * sizeof(block_q3_K),
                     fixture.data_start + fixture.dense_offset);

        // Equal zero router logits select expert IDs 0..9 in stable tie order.
        // Give those ten expert slabs finite positive Q2_0 values so the test
        // exercises nonzero routed as well as shared FFN output.
        std::vector<block_q2_0> slab(
            static_cast<std::size_t>(kExpertSlabBytes / sizeof(block_q2_0)));
        for (block_q2_0& block : slab) {
            block.d = 0x1000u;
            std::fill_n(block.qs, 16u, 0xaau);
        }
        for (std::size_t role = 0; role < fixture.offsets.size(); ++role) {
            for (std::uint64_t expert = 0; expert < 10u; ++expert) {
                write_all_at(
                    descriptor, slab.data(),
                    slab.size() * sizeof(block_q2_0),
                    fixture.data_start + fixture.offsets[role] +
                        expert * kExpertSlabBytes);
            }
        }
        Geometry ple_geometry;
        ple_geometry.family = "qwen4exp";
        ple_geometry.model = "synthetic-fullmode-handoff";
        ple_geometry.n_layers = kSessionModelLayers;
        ple_geometry.n_expert = kExperts;
        ple_geometry.n_qsa_layers = kSessionQsaLayers;
        ple_geometry.n_kv_head = kSessionKvHeads;
        ple_geometry.head_dim = kSessionHeadWidth;
        ple_geometry.n_embd = kSessionEmbeddingWidth;
        ple_geometry.vocab = kSessionVocabularySize;
        const std::array<std::uint32_t, 1u> ple_tokens{0u};
        const std::array<std::uint32_t, 2u> ple_history{5u, 0u};
        WindowPleRows ple_rows;
        std::string ple_error;
        require(build_window_ple_rows(ple_tokens, ple_history, ple_geometry,
                                      ple_rows, ple_error),
                "build synthetic FullMode PLE table rows: " + ple_error);
        std::array<block_iq4_nl,
                   strata::kernels::PLE_HEAD_DIM / 32u> ple_row{};
        for (block_iq4_nl& block : ple_row) {
            block.d = 0x3c00u;
            std::fill_n(block.qs, 16u, 0x88u);
        }
        for (const std::uint32_t row : ple_rows.all_rows()) {
            require(row < kPleRows,
                    "synthetic FullMode PLE row is inside the sparse table");
            write_all_at(
                descriptor, ple_row.data(), sizeof(ple_row),
                fixture.data_start + fixture.ple_offset +
                    static_cast<std::uint64_t>(row) *
                        strata::kernels::PLE_ROW_BYTES);
        }
    } catch (...) {
        (void)::close(descriptor);
        std::error_code ignored;
        fs::remove(fixture.path, ignored);
        throw;
    }
    require(::close(descriptor) == 0,
            "close sparse full-mode handoff GGUF fixture");

    const int pack_descriptor =
        ::open(fixture.dense_pack_path.c_str(),
               O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    require(pack_descriptor >= 0,
            "create sparse full-mode handoff dense pack");
    const bool pack_sized =
        ::ftruncate(pack_descriptor,
                    static_cast<off_t>(fixture.dense_pack_bytes)) == 0;
    (void)::close(pack_descriptor);
    require(pack_sized, "size sparse full-mode handoff dense pack");
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
    const std::array<std::string, 3> expert_roles = {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    };
    for (std::uint32_t layer = 0; layer < fixture.expert_layers; ++layer) {
        for (const std::string& role : expert_roles) {
            const std::string name =
                "blk." + std::to_string(layer) + "." + role;
            const strata::TensorInfo* tensor = file.find(name);
            if (tensor == nullptr) {
                throw std::runtime_error("synthetic GGUF tensor is missing: " + name);
            }
            const std::uint64_t payload = strata::tensor_payload_bytes(*tensor);
            shard.tensors.push_back(
                {name, file.data_start() + tensor->offset, payload,
                 payload / fixture.expert_count});
        }
    }
    for (const std::string& name : {std::string("token_embd.weight")}) {
        const strata::TensorInfo* tensor = file.find(name);
        if (tensor == nullptr) {
            throw std::runtime_error("synthetic GGUF tensor is missing");
        }
        shard.tensors.push_back(
            {name, file.data_start() + tensor->offset,
             strata::tensor_payload_bytes(*tensor), 0});
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
    geometry.n_layers = 3;
    geometry.types = {"f32", "iq2_xs"};
    const LayerGeometry tiered_layer{
        3 * fixture.expert_slab_bytes,
        "iq2_xs",
        "iq2_xs",
    };
    geometry.layers = {tiered_layer, tiered_layer, tiered_layer};

    PlanResult memory_plan;
    memory_plan.fits = true;
    memory_plan.mode = "tiered";
    memory_plan.resident_order = {
        {0, 7}, {0, 2}, {1, 7}, {1, 2}, {2, 7}, {2, 2},
    };
    memory_plan.experts_resident = memory_plan.resident_order.size();
    memory_plan.experts_resident_bytes =
        memory_plan.resident_order.size() * geometry.layers[0].blob_bytes;
    const std::uint64_t max_blob = geometry.layers[0].blob_bytes;
    const std::uint64_t slot_stride =
        (max_blob + kWindowPageBytes - 1) / kWindowPageBytes * kWindowPageBytes;
    const std::uint64_t max_layer_misses = geometry.n_expert - 2;
    memory_plan.slots = 2 * max_layer_misses;
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
                storage->slot_pool()->half_capacity(0) == max_layer_misses &&
                storage->slot_pool()->half_capacity(1) == max_layer_misses,
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
    const BufferRef usage_table = storage->usage_table();
    require(address_table && resident_table && plan_buffer && usage_table &&
                usage_table.length == static_cast<std::size_t>(kTieredMaxLayers) *
                                          geometry.n_expert * sizeof(float),
            "tiered addr, res, plan and usage tables must be shared Metal buffers");
    const auto* addresses = static_cast<const std::uint64_t*>(address_table.contents);
    const auto* resident = static_cast<const std::int32_t*>(resident_table.contents);
    const auto* plans = static_cast<const TieredMoePlan*>(plan_buffer.contents);
    require(resident[7] == 0 && resident[2] == 1 && resident[0] == -1 &&
                resident[geometry.n_expert + 7] == 2 &&
                resident[geometry.n_expert + 2] == 3 &&
                resident[2 * geometry.n_expert + 7] == 4 &&
                resident[2 * geometry.n_expert + 2] == 5 &&
                addresses[7 * 3] == layout.experts[7].resident_addresses[0] &&
                addresses[7 * 3 + 1] == layout.experts[7].resident_addresses[1] &&
                addresses[7 * 3 + 2] == layout.experts[7].resident_addresses[2] &&
                addresses[0] == 0 && addresses[1] == 0 && addresses[2] == 0 &&
                plans[0].n_groups == 0 && plans[0].n_miss == 0 &&
                std::all_of(static_cast<const float*>(usage_table.contents),
                            static_cast<const float*>(usage_table.contents) +
                                usage_table.length / sizeof(float),
                            [](float value) { return value == 0.0f; }),
            "tiered shared tables must match the host layout and start with zeroed plans and usage");

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

    auto* writable_plans = static_cast<TieredMoePlan*>(plan_buffer.contents);
    BufferRef plan_ids_buffer = context.buffers().allocate(
        "tiered resolver plan IDs", 10 * sizeof(std::int32_t), error);
    require(static_cast<bool>(plan_ids_buffer),
            "allocate IDs for the GPU plan producer: " + error);
    const ReadOnlyBufferRef plan_ids =
        context.buffers().read_only(plan_ids_buffer);
    const ReadOnlyBufferRef plan_resident =
        context.buffers().read_only(resident_table);
    const MoePlanBuffers gpu_plan_buffers{
        plan_ids, plan_resident, usage_table, plan_buffer};
    require(plan_ids && plan_resident,
            "create read-only views for the GPU plan producer: " + error);
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

    auto handshake_events = context.make_events("tiered resolver layer handshake", error);
    require(handshake_events != nullptr, "create layer-handshake test event: " + error);
    const std::uint64_t arrival_value = handshake_events->reserve_value();
    const std::uint64_t release_value = handshake_events->reserve_value();
    const std::uint64_t consumer_arrival_value = handshake_events->reserve_value();
    const std::uint64_t consumer_release_value = handshake_events->reserve_value();
    const std::uint64_t layer_done_value = handshake_events->reserve_value();
    const std::uint64_t later_arrival_value = handshake_events->reserve_value();
    const std::uint64_t later_release_value = handshake_events->reserve_value();
    const std::uint64_t completion_gate_value = handshake_events->reserve_value();
    const std::array<std::int32_t, 10> resolve_ids{
        7, 7, 7, 7, 3, 3, 2, 8, 8, 8};
    std::memcpy(plan_ids_buffer.contents, resolve_ids.data(), sizeof(resolve_ids));
    auto resolve_command =
        context.make_command_buffer("tiered expert miss resolution", error);
    require(resolve_command != nullptr,
            "create tiered expert resolver command buffer: " + error);
    require(encode_moe_plan(context, *resolve_command, 1, geometry.n_expert,
                            true, 0, gpu_plan_buffers, error),
            "encode GPU plan before signaling plan-ready: " + error);
    require(handshake_events->encode_signal(*resolve_command, arrival_value, error) &&
                handshake_events->encode_wait(*resolve_command, release_value, error) &&
                handshake_events->encode_signal(*resolve_command,
                                                consumer_arrival_value, error) &&
                handshake_events->encode_wait(*resolve_command,
                                              consumer_release_value, error),
            "encode the first plan-ready and host-release events: " + error);

    constexpr std::size_t kInputBlocks = 80;
    constexpr std::size_t kOutputEntries = 10;
    constexpr std::size_t kOutputWidth = 640;
    constexpr std::size_t kIQ2XSBlockBytes = 74;
    constexpr std::size_t kGateRowsInDispatch = 8;
    constexpr std::size_t kBytesReadPerExpertRole =
        kGateRowsInDispatch * 10 * kIQ2XSBlockBytes;
    constexpr std::size_t kOutputBytes =
        kOutputEntries * kOutputWidth * sizeof(float);
    require(fixture.expert_slab_bytes >= kBytesReadPerExpertRole,
            "the bounded GPU gate dispatch must fit inside each synthetic IQ2_XS role slab");
    std::array<TestQ8_1, kInputBlocks> activations{};
    for (TestQ8_1& activation : activations) {
        activation.ds[0] = 0x3c00;  // IEEE-754 half 1.0.
        std::fill(activation.qs, activation.qs + 32, 1);
    }
    BufferRef activation_buffer = context.buffers().allocate(
        "tiered plan consumer Q8_1 activations",
        sizeof(activations), error);
    BufferRef gate_output = context.buffers().allocate(
        "tiered plan consumer gate output", kOutputBytes, error);
    BufferRef up_output = context.buffers().allocate(
        "tiered plan consumer up output", kOutputBytes, error);
    require(activation_buffer && gate_output && up_output,
            "allocate buffers for the GPU expert-address consumer: " + error);
    std::memcpy(activation_buffer.contents, activations.data(),
                sizeof(activations));
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::fill_n(static_cast<float*>(gate_output.contents),
                kOutputEntries * kOutputWidth, poison);
    std::fill_n(static_cast<float*>(up_output.contents),
                kOutputEntries * kOutputWidth, poison);
    const ReadOnlyBufferRef address_input =
        context.buffers().read_only(storage->expert_address_table());
    const ReadOnlyBufferRef activation_input =
        context.buffers().read_only(activation_buffer);
    const ReadOnlyBufferRef plan_input =
        context.buffers().read_only(storage->plan_buffer());
    require(address_input && activation_input && plan_input,
            "create read-only GPU expert consumer inputs: " + error);
    const std::array<EncoderReadOnlyBufferBinding, 3> consumer_inputs = {{
        {address_input, 0,
         static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert *
             sizeof(TestMoeExpertAddress),
         alignof(TestMoeExpertAddress)},
        {activation_input, 1, sizeof(activations), alignof(TestQ8_1)},
        {plan_input, 2, sizeof(TieredMoePlan), alignof(std::uint32_t)},
    }};
    const std::array<EncoderBufferBinding, 2> consumer_outputs = {{
        {gate_output, 3, kOutputBytes, alignof(float)},
        {up_output, 4, kOutputBytes, alignof(float)},
    }};
    const TestMoeGroupedParams consumer_params{geometry.n_expert, 0, 1};
    const EncoderBytesBinding params_binding{
        &consumer_params, sizeof(consumer_params), 5};
    // One x group reads gate rows 0..7 only; the up projection starts at x=80.
    require(Encoder::encode(
                context, *resolve_command, "tiered expert address consumer",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encoder.dispatch(
                        "strata_moe_gu_iq2_xs", {1, resolve_plan.n_groups, 1},
                        {256, 1, 1}, consumer_outputs, consumer_inputs,
                        std::span<const EncoderBytesBinding>(&params_binding, 1),
                        {}, encode_error);
                },
                error),
            "encode the GPU IQ2_XS consumer of resolved expert addresses: " +
                error);
    require(handshake_events->encode_signal(*resolve_command, layer_done_value, error) &&
                encode_moe_plan(context, *resolve_command, 1, geometry.n_expert,
                                true, 2, gpu_plan_buffers, error) &&
                handshake_events->encode_signal(*resolve_command, later_arrival_value, error) &&
                handshake_events->encode_wait(*resolve_command, later_release_value, error) &&
                handshake_events->encode_wait(*resolve_command,
                                              completion_gate_value, error),
            "encode two ordered GPU plans and their host-release events: " + error);
    auto resolve_leases =
        expert_store->prepare_command_buffer(*resolve_command, error);
    require(resolve_leases != nullptr,
            "register slot release before command-buffer submission: " + error);
    ExpertLayerStats invalid_event_stats;
    require(!expert_store->resolve_layer_after_plan(
                resolve_leases, 0, *handshake_events, arrival_value,
                layer_done_value, 0, invalid_event_stats, error) &&
                error.find("consecutive positive event values") != std::string::npos &&
                event_has_value(*handshake_events, 0),
            "reject a layer-ready value that could release a later GPU wait");
    id<MTLCommandBuffer> native_resolve =
        (__bridge id<MTLCommandBuffer>)resolve_command->unsafe_native_handle();
    require(resolve_command->commit(error),
            "submit plan-producing command buffer: " + error);
    bool consumer_gate_released = false;
    CommandGatesCleanup gate_cleanup(
        handshake_events.get(), arrival_value, release_value,
        handshake_events.get(), later_arrival_value, completion_gate_value,
        native_resolve, [&] {
            if (!consumer_gate_released) {
                // Prevent test cleanup from running the queued expert kernel
                // with unresolved addresses after releasing an earlier wait.
                writable_plans[0].n_groups = 0;
            }
        });
    gate_cleanup.add_layer_gate(handshake_events.get(), consumer_arrival_value,
                                consumer_release_value);
    gate_cleanup.add_layer_gate(handshake_events.get(), later_arrival_value,
                                later_release_value);
    require(handshake_events->wait(arrival_value, 5000, error),
            "wait for first GPU-produced plan before replacing its consumed ids: " + error);
    std::uint32_t later_resident_expert = geometry.n_expert;
    const std::size_t later_layer_resident_offset =
        static_cast<std::size_t>(2) * geometry.n_expert;
    for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
        if (layout.resident[later_layer_resident_offset + expert] >= 0) {
            later_resident_expert = expert;
            break;
        }
    }
    require(later_resident_expert < geometry.n_expert,
            "find a layer-2 resident expert for the later GPU plan");
    const std::array<std::int32_t, 10> later_ids = [] (std::uint32_t expert) {
        std::array<std::int32_t, 10> ids{};
        ids.fill(static_cast<std::int32_t>(expert));
        return ids;
    }(later_resident_expert);
    std::memcpy(plan_ids_buffer.contents, later_ids.data(), sizeof(later_ids));
    ExpertLayerStats first_resolve;
    require(expert_store->resolve_layer_after_plan(
                resolve_leases, 0, *handshake_events, arrival_value, release_value,
                5000, first_resolve, error),
            "resolve the shared GPU plan and release the waiting layer: " + error);
    gate_cleanup.mark_layer_released();
    require(handshake_events->wait(consumer_arrival_value, 5000, error),
            "hold the GPU expert consumer at its test gate before reading slots: " + error);
    require(event_has_value(*handshake_events, consumer_arrival_value) &&
                std::memcmp(writable_plans, &resolve_plan, sizeof(resolve_plan)) == 0 &&
                first_resolve.hits == 5 && first_resolve.lookups == 10 &&
                first_resolve.ram_blobs + first_resolve.file_blobs == 2 &&
                first_resolve.file_bytes == first_resolve.file_blobs * blob,
            "handshake must publish misses before the blocked GPU consumer and count routed hits");

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
            "all same-parity slots stay pinned while the GPU consumer is gated before its reads (lease=" +
                std::to_string(static_cast<bool>(exhausted_half)) + ", error=" + error + ")");
    require(gate_cleanup.release_layer(error),
            "release the gated GPU expert consumer after proving its slots stay pinned: " +
                error);
    gate_cleanup.mark_layer_released();
    consumer_gate_released = true;
    require(handshake_events->wait(layer_done_value, 5000, error),
            "wait until the GPU IQ2_XS layer has consumed its expert addresses: " + error);
    const auto* gate_values = static_cast<const float*>(gate_output.contents);
    const auto* up_values = static_cast<const float*>(up_output.contents);
    std::size_t finite_gate_outputs = 0;
    for (std::size_t entry = 0; entry < kOutputEntries; ++entry) {
        for (std::size_t row = 0; row < 8; ++row) {
            const std::size_t index = entry * kOutputWidth + row;
            finite_gate_outputs += std::isfinite(gate_values[index]);
        }
    }
    require(finite_gate_outputs == kOutputEntries * 8 &&
                std::isnan(gate_values[kOutputWidth + 8]) &&
                std::isnan(up_values[0]),
            "the GPU expert consumer must overwrite its routed gate outputs before layer retirement (finite outputs=" +
                std::to_string(finite_gate_outputs) + ", first gate=" +
                std::to_string(gate_values[0]) + ", first up=" +
                std::to_string(up_values[0]) + ")");
    require(handshake_events->wait(later_arrival_value, 5000, error),
            "wait for the later GPU plan after the earlier layer's expert reads: " + error);
    require(writable_plans[2].n_groups == 1 &&
                writable_plans[2].n_entries == 10 &&
                writable_plans[2].n_miss == 0 && writable_plans[2].skip == 1,
            "later GPU plan must confirm that its repeated layer resident needs no slot");
    ExpertLayerStats later_stats;
    require(expert_store->resolve_layer_after_plan(
                resolve_leases, 2, *handshake_events, later_arrival_value,
                later_release_value, 5000, later_stats, error),
            "resolve a later plan and automatically retire completed prior layers: " + error);
    gate_cleanup.mark_layer_released();
    require(later_stats.hits == 10 && later_stats.lookups == 10 &&
                event_has_value(*handshake_events, later_release_value),
            "later plan handshake must retire earlier leases before signaling its layer-ready value");
    auto reused_half = storage->slot_pool()->acquire(
        0, 2, layout.experts[2].blob_bytes, error);
    require(reused_half && !reused_half.view().cache_hit,
            "a later plan event must automatically free completed prior-layer slots before command completion: " +
                error);
    require(reused_half.publish(error),
            "publish a slot reused after its layer-complete handshake: " + error);
    require(native_resolve.status != MTLCommandBufferStatusCompleted,
            "the command buffer stays in flight after one layer's leases are retired");
    require(gate_cleanup.finish_and_drain(error),
            "release the command-completion gate and drain the GPU command: " + error);
    other_layer_pins.clear();
    reused_half.reset();

    const std::array<std::int32_t, 10> cleanup_ids{
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    std::memcpy(plan_ids_buffer.contents, cleanup_ids.data(), sizeof(cleanup_ids));
    std::fill_n(static_cast<float*>(gate_output.contents),
                kOutputEntries * kOutputWidth, poison);
    std::fill_n(static_cast<float*>(up_output.contents),
                kOutputEntries * kOutputWidth, poison);
    auto cleanup_events =
        context.make_events("tiered failed-request cleanup gates", error);
    require(cleanup_events != nullptr,
            "create failed-request cleanup test events: " + error);
    const std::uint64_t cleanup_plan_arrival = cleanup_events->reserve_value();
    const std::uint64_t cleanup_plan_release = cleanup_events->reserve_value();
    const std::uint64_t cleanup_consumer_arrival = cleanup_events->reserve_value();
    const std::uint64_t cleanup_consumer_release = cleanup_events->reserve_value();
    const std::uint64_t cleanup_layer_done = cleanup_events->reserve_value();
    const std::uint64_t cleanup_completion = cleanup_events->reserve_value();
    auto cleanup_command =
        context.make_command_buffer("tiered failed-request cleanup", error);
    require(cleanup_command != nullptr,
            "create failed-request cleanup command buffer: " + error);
    require(encode_moe_plan(context, *cleanup_command, 1, geometry.n_expert,
                            true, 0, gpu_plan_buffers, error) &&
                cleanup_events->encode_signal(*cleanup_command,
                                              cleanup_plan_arrival, error) &&
                cleanup_events->encode_wait(*cleanup_command,
                                            cleanup_plan_release, error),
            "encode cleanup plan and its host gate: " + error);
    require(Encoder::encode(
                context, *cleanup_command, "failed-request guarded consumer",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encoder.dispatch(
                        "strata_moe_gu_iq2_xs", {1, 1, 1}, {256, 1, 1},
                        consumer_outputs, consumer_inputs,
                        std::span<const EncoderBytesBinding>(&params_binding, 1),
                        {}, encode_error);
                },
                error) &&
                cleanup_events->encode_signal(*cleanup_command,
                                              cleanup_consumer_arrival, error) &&
                cleanup_events->encode_wait(*cleanup_command,
                                            cleanup_consumer_release, error) &&
                cleanup_events->encode_signal(*cleanup_command,
                                              cleanup_layer_done, error) &&
                cleanup_events->encode_wait(*cleanup_command,
                                            cleanup_completion, error),
            "encode guarded consumer and staged cleanup gates: " + error);
    auto cleanup_leases =
        expert_store->prepare_command_buffer(*cleanup_command, error);
    require(cleanup_leases != nullptr,
            "prepare failed-request cleanup completion callback: " + error);
    id<MTLCommandBuffer> native_cleanup =
        (__bridge id<MTLCommandBuffer>)cleanup_command->unsafe_native_handle();
    require(cleanup_command->commit(error),
            "submit failed-request cleanup command buffer: " + error);
    bool cleanup_guard_applied = false;
    {
        CommandGatesCleanup failed_request_cleanup(
            cleanup_events.get(), cleanup_plan_arrival, cleanup_plan_release,
            cleanup_events.get(), cleanup_layer_done, cleanup_completion,
            native_cleanup, [&] {
                writable_plans[0].n_groups = 0;
                cleanup_guard_applied = true;
            });
        failed_request_cleanup.add_layer_gate(
            cleanup_events.get(), cleanup_consumer_arrival,
            cleanup_consumer_release);
        require(cleanup_events->wait(cleanup_plan_arrival, 5000, error),
                "wait for the failed request's GPU plan before cleanup: " + error);
    }
    require(cleanup_guard_applied &&
                cleanup_events->wait(cleanup_layer_done, 5000, error) &&
                native_cleanup.status == MTLCommandBufferStatusCompleted &&
                std::all_of(static_cast<const float*>(gate_output.contents),
                            static_cast<const float*>(gate_output.contents) +
                                kOutputEntries * kOutputWidth,
                            [](float value) { return std::isnan(value); }),
            "failed-request cleanup must guard the consumer after plan arrival, drain every staged wait, and leave outputs untouched");

    const std::array<std::int32_t, 10> skip_ids{
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    std::memcpy(plan_ids_buffer.contents, skip_ids.data(), sizeof(skip_ids));
    const std::uint64_t skip_arrival = handshake_events->reserve_value();
    const std::uint64_t skip_release = handshake_events->reserve_value();
    const std::uint64_t skip_done = handshake_events->reserve_value();
    const std::uint64_t skip_completion = handshake_events->reserve_value();
    auto skip_command = context.make_command_buffer("tiered skipped plan", error);
    require(skip_command != nullptr, "create skipped-plan command buffer: " + error);
    require(encode_moe_plan(context, *skip_command, 1, geometry.n_expert,
                            true, 0, gpu_plan_buffers, error),
            "encode GPU all-resident plan before skip signal: " + error);
    require(handshake_events->encode_signal(*skip_command, skip_arrival, error) &&
                handshake_events->encode_wait(*skip_command, skip_release, error) &&
                handshake_events->encode_signal(*skip_command, skip_done, error) &&
                handshake_events->encode_wait(*skip_command, skip_completion, error),
            "encode skipped-plan handshake events: " + error);
    auto skip_leases = expert_store->prepare_command_buffer(*skip_command, error);
    require(skip_leases != nullptr,
            "prepare skipped-plan completion callback: " + error);
    id<MTLCommandBuffer> native_skip =
        (__bridge id<MTLCommandBuffer>)skip_command->unsafe_native_handle();
    require(skip_command->commit(error), "submit skipped-plan command buffer: " + error);
    CommandGatesCleanup skip_cleanup(handshake_events.get(), skip_arrival,
                                     skip_release,
                                     handshake_events.get(), skip_done,
                                     skip_completion, native_skip);
    require(handshake_events->wait(skip_arrival, 5000, error),
            "wait for GPU all-resident plan before poisoning ignored arrays: " + error);
    require(writable_plans[0].n_groups == 1 &&
                writable_plans[0].n_entries == 10 &&
                writable_plans[0].n_miss == 0 && writable_plans[0].skip == 1,
            "GPU plan kernel must publish the valid skip header before its event");
    writable_plans[0].grp_e[0] = -1;
    writable_plans[0].grp_res[0] = 7;
    writable_plans[0].grp_start[0] = -1;
    writable_plans[0].ent_tok[0] = -1;
    writable_plans[0].ent_dst[0] = -1;
    ExpertLayerStats skipped_stats;
    require(expert_store->resolve_layer_after_plan(
                skip_leases, 0, *handshake_events, skip_arrival, skip_release,
                5000, skipped_stats, error),
            "resolve a skipped plan without inspecting its arrays: " + error);
    skip_cleanup.mark_layer_released();
    require(skipped_stats.hits == 10 && skipped_stats.lookups == 10 &&
                skipped_stats.ram_blobs == 0 && skipped_stats.file_blobs == 0 &&
                event_has_reached_value(*handshake_events, skip_release),
            "a skipped plan must report resident hits and signal the host release (hits=" +
                std::to_string(skipped_stats.hits) + ", lookups=" +
                std::to_string(skipped_stats.lookups) + ", ram=" +
                std::to_string(skipped_stats.ram_blobs) + ", file=" +
                std::to_string(skipped_stats.file_blobs) + ", released=" +
                std::to_string(event_has_reached_value(*handshake_events, skip_release)) +
                ", error=" + error + ")");
    require(handshake_events->wait(skip_done, 5000, error),
            "wait for skipped layer completion: " + error);
    skip_leases->retire_layer(0);
    require(skip_cleanup.finish_and_drain(error),
            "drain skipped-plan command buffer: " + error);

    TieredMoePlan invalid_skipped_plan{};
    invalid_skipped_plan.n_groups = 0;
    invalid_skipped_plan.n_entries = 10;
    invalid_skipped_plan.skip = 1;
    writable_plans[0] = invalid_skipped_plan;
    const std::uint64_t invalid_arrival = handshake_events->reserve_value();
    const std::uint64_t invalid_release = handshake_events->reserve_value();
    const std::uint64_t invalid_done = handshake_events->reserve_value();
    const std::uint64_t invalid_completion = handshake_events->reserve_value();
    auto invalid_handshake_command =
        context.make_command_buffer("tiered invalid skipped plan", error);
    require(invalid_handshake_command != nullptr,
            "create invalid skipped-plan command buffer: " + error);
    require(handshake_events->encode_signal(
                *invalid_handshake_command, invalid_arrival, error) &&
                handshake_events->encode_wait(
                    *invalid_handshake_command, invalid_release, error) &&
                handshake_events->encode_signal(
                    *invalid_handshake_command, invalid_done, error) &&
                handshake_events->encode_wait(
                    *invalid_handshake_command, invalid_completion, error),
            "encode invalid-plan handshake events: " + error);
    auto invalid_handshake_leases =
        expert_store->prepare_command_buffer(*invalid_handshake_command, error);
    require(invalid_handshake_leases != nullptr,
            "prepare invalid-plan completion callback: " + error);
    id<MTLCommandBuffer> native_invalid =
        (__bridge id<MTLCommandBuffer>)invalid_handshake_command->unsafe_native_handle();
    require(invalid_handshake_command->commit(error),
            "submit invalid skipped-plan command buffer: " + error);
    CommandGatesCleanup invalid_cleanup(handshake_events.get(), invalid_arrival,
                                       invalid_release,
                                       handshake_events.get(), invalid_done,
                                       invalid_completion, native_invalid);
    ExpertLayerStats invalid_handshake_stats;
    require(!expert_store->resolve_layer_after_plan(
                invalid_handshake_leases, 0, *handshake_events,
                invalid_arrival, invalid_release, 5000,
                invalid_handshake_stats, error) &&
                error.find("counts, row width, or skip flag") != std::string::npos &&
                event_has_value(*handshake_events, invalid_arrival),
            "a skipped plan with no groups must fail without signaling layer-ready");
    invalid_skipped_plan.n_groups = 1;
    invalid_skipped_plan.n_miss = 1;
    writable_plans[0] = invalid_skipped_plan;
    require(!expert_store->resolve_layer_after_plan(
                invalid_handshake_leases, 0, *handshake_events,
                invalid_arrival, invalid_release, 5000,
                invalid_handshake_stats, error) &&
                error.find("skipped plan reports misses") != std::string::npos &&
                event_has_value(*handshake_events, invalid_arrival),
            "a skipped plan reporting misses must fail without signaling layer-ready");
    require(invalid_cleanup.release_layer(error) &&
                handshake_events->wait(invalid_done, 5000, error) &&
                invalid_cleanup.finish_and_drain(error),
            "caller must release and drain GPU waits after plan-resolution failure: " + error);

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
    require(context.buffers().allocation_count() == 11,
            "tiered mode must allocate its three arenas, four tables, GPU-plan input and GPU-reader fixtures, without GGUF windows");
    require_not_overlapping_gguf_mapping(dense_arena, source_file, "tiered dense arena");
    require_not_overlapping_gguf_mapping(resident_arena, source_file, "tiered resident arena");
    require_not_overlapping_gguf_mapping(slot_arena, source_file, "tiered slot arena");
    require_not_overlapping_gguf_mapping(address_table, source_file, "tiered address table");
    require_not_overlapping_gguf_mapping(resident_table, source_file, "tiered resident table");
    require_not_overlapping_gguf_mapping(plan_buffer, source_file, "tiered plan table");
    require_not_overlapping_gguf_mapping(usage_table, source_file, "tiered usage table");

    const std::uint64_t abandoned_plan_ready = handshake_events->reserve_value();
    const std::uint64_t abandoned_layer_ready = handshake_events->reserve_value();
    require(abandoned_plan_ready != 0 &&
                abandoned_layer_ready == abandoned_plan_ready + 1 &&
                handshake_events->release_gpu_waits(5000, error),
            "retire the synthetic handshake event after its final GPU use: " + error);
    ExpertLayerStats released_event_stats;
    require(!expert_store->resolve_layer_after_plan(
                cache_leases, 0, *handshake_events, abandoned_plan_ready,
                abandoned_layer_ready, 0, released_event_stats, error) &&
                error.find("event is unavailable or was released") != std::string::npos,
            "ExpertStore must reject a released handshake event instead of reading its sentinel");

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
    constexpr std::uint64_t kFixedAllowanceBytes = 256ull * 1024 * 1024;
    PlanResult plan;
    plan.fits = true;
    plan.mode = "full";
    plan.w_bytes = facts.working_set_bytes;
    plan.ram_bytes = facts.physical_memory_bytes;
    plan.available_bytes = -1;
    plan.context = 13;
    plan.kv = KvFormat::Int8;
    plan.fixed_bytes = kFixedAllowanceBytes;
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
                       plan.fixed_bytes + plan.kv_bytes +
                       plan.experts_resident_bytes;
    plan.ram_bytes_used = plan.total_bytes - plan.unread_bytes;
    plan.anonymous_bytes_used = plan.ram_bytes_used;
    plan.free_bytes = static_cast<std::int64_t>(plan.w_bytes - plan.total_bytes);
    return plan;
}

PlanResult owner_test_tiered_plan(const Geometry& geometry, const DeviceFacts& facts) {
    constexpr std::uint64_t kFixedAllowanceBytes = 256ull * 1024 * 1024;
    PlanResult plan;
    plan.fits = true;
    plan.mode = "tiered";
    plan.w_bytes = facts.working_set_bytes;
    plan.ram_bytes = facts.physical_memory_bytes;
    plan.available_bytes = -1;
    plan.context = 13;
    plan.kv = KvFormat::Int8;
    plan.fixed_bytes = kFixedAllowanceBytes;
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
    plan.total_bytes = plan.dense_bytes + plan.pack_bytes + plan.fixed_bytes +
                       plan.kv_bytes + plan.slot_bytes +
                       plan.experts_resident_bytes;
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
            "full-mode owner must reject a converted router whose block exceeds geometry: " +
                error);
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
    require(full_storage->tiered_storage() == nullptr,
            "full-mode ModelStorage must not publish a tiered resolver source");
    require(std::string(full_storage->mode()) == "full" &&
                full_storage->resident_experts() == plain_geometry.n_expert &&
                full_storage->window_count() != 0 && full_storage->full_mode_warmed(),
            "full-mode owner must retain mapped windows after a successful warm-up");
    strata::kernels::PleTableReader* full_ple = full_storage->ple_reader();
    require(full_ple != nullptr && full_ple->is_open() &&
                full_ple->mode() == strata::kernels::PleIo::Mmap &&
                full_ple->rows() == 728,
            "full-mode ModelStorage must publish its mapped PLE reader");
    const FullModeStorage* full_mode = full_storage->full_mode_storage();
    const BufferRef full_mode_usage =
        full_mode != nullptr ? full_mode->usage_table() : BufferRef{};
    const BufferRef model_usage = full_storage->moe_usage_table();
    FullModeTensorView owner_token_view;
    require(full_mode != nullptr &&
                full_mode->dense_tensor_view("token_embd.weight",
                                             owner_token_view, error) &&
                owner_token_view.buffer &&
                context.buffers().owns(owner_token_view.buffer) &&
                full_mode->expert_address_table() &&
                full_mode->resident_table() &&
                context.buffers().owns(full_mode->expert_address_table()) &&
                context.buffers().owns(full_mode->resident_table()) &&
                full_mode_usage && model_usage &&
                full_mode_usage.gpu_address == model_usage.gpu_address &&
                full_mode_usage.length == static_cast<std::size_t>(kFullModeMaxLayers) *
                                              plain_geometry.n_expert * sizeof(float) &&
                context.buffers().owns(model_usage) &&
                std::all_of(static_cast<const float*>(model_usage.contents),
                            static_cast<const float*>(model_usage.contents) +
                                model_usage.length / sizeof(float),
                            [](float value) { return value == 0.0f; }),
            "full-mode ModelStorage must expose borrowed dense weights, expert tables and zeroed usage: " +
                error);
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
    ReadOnlyBufferRef provisional_weight;
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
        FullModeTensorView provisional_token_view;
        require(provisional_storage->full_mode_storage() != nullptr &&
                    provisional_storage->full_mode_storage()->dense_tensor_view(
                        "token_embd.weight", provisional_token_view, error),
                "resolve a provisional read-only full-mode weight view: " + error);
        provisional_weight = provisional_token_view.buffer;
        require(provisional_kv_allocation && provisional_rope && provisional_weight &&
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
                !provisional_kv_allocation && !provisional_rope &&
                !provisional_weight && !context.buffers().owns(provisional_weight),
            "outer rollback must release provisional buffers and invalidate read-only weight views");
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
        const ReadOnlyBufferRef arena = opened->dense_pack_arena();
        require(arena && arena.length() == 3 * kWindowPageBytes &&
                    opened->layout().dense_pack_bytes == packed_fixture.dense_pack_bytes &&
                    opened->layout().dense_pack_arena_bytes == arena.length(),
                "full-mode router arena must be padded to its 16 KiB allocation boundary");
        const auto row = std::find_if(
            opened->layout().tensors.begin(), opened->layout().tensors.end(),
            [](const TensorGpuBinding& binding) {
                return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
            });
        require(row != opened->layout().tensors.end() &&
                    row->gpu_address >= arena.gpu_address() &&
                    row->gpu_address + row->byte_count <= arena.gpu_address() + arena.length(),
                "converted router binding must point inside the dense-pack arena");
        FullModeTensorView router_view;
        require(opened->dense_tensor_view("blk.0.ffn_gate_inp.weight",
                                          router_view, error),
                "resolve converted router as a full-mode tensor view: " + error);
        require(router_view.ggml_type == 30 &&
                    router_view.shape == std::vector<std::uint64_t>{2'560, 9} &&
                    router_view.buffer.length() == packed_fixture.dense_pack_bytes &&
                    router_view.buffer.gpu_address() == row->gpu_address,
                "converted router view must expose its runtime BF16 type, GGUF shape and arena slice");
        std::vector<std::uint8_t> expected(
            static_cast<std::size_t>(packed_fixture.dense_pack_bytes));
        std::ifstream input(packed_fixture.dense_pack_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(expected.data()),
                   static_cast<std::streamsize>(expected.size()));
        require(input.gcount() == static_cast<std::streamsize>(expected.size()) &&
                    std::memcmp(arena.contents(), expected.data(), expected.size()) == 0,
                "full-mode dense-pack arena must contain the exact converted router row");
        require(std::memcmp(router_view.buffer.contents(), expected.data(),
                            expected.size()) == 0,
                "converted router view must expose the exact dense-pack bytes");
        const auto* arena_bytes = static_cast<const std::uint8_t*>(arena.contents());
        require(std::all_of(arena_bytes + expected.size(),
                            arena_bytes + arena.length(),
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
        const ReadOnlyBufferRef padded_arena = padded_storage->dense_pack_arena();
        const auto padded_row = std::find_if(
            padded_storage->layout().tensors.begin(),
            padded_storage->layout().tensors.end(),
            [](const TensorGpuBinding& binding) {
                return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
            });
        require(padded_arena && padded_row != padded_storage->layout().tensors.end() &&
                    padded_row->gpu_address == padded_arena.gpu_address() &&
                    std::memcmp(padded_arena.contents(), expected.data(), expected.size()) == 0 &&
                    std::all_of(
                        static_cast<const std::uint8_t*>(padded_arena.contents()) +
                            expected.size(),
                        static_cast<const std::uint8_t*>(padded_arena.contents()) +
                            padded_arena.length(),
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
    require(tiered_storage->full_mode_storage() == nullptr,
            "tiered ModelStorage must not publish full-mode weight views or tables");
    require(tiered_storage->tiered_storage() != nullptr,
            "tiered ModelStorage must expose the borrowed ExpertStore construction source");
    WindowT1MoeLayer0FullBuffers wrong_mode_moe_buffers{};
    wrong_mode_moe_buffers.ffn.routed.addresses =
        context.buffers().read_only(
            tiered_storage->tiered_storage()->expert_address_table());
    auto wrong_mode_moe_command = context.make_command_buffer(
        "reject tiered table in full-mode Window MoE", error);
    require(wrong_mode_moe_command != nullptr,
            "create full-mode MoE table-source regression command: " + error);
    require(!encode_window_t1_moe_layer0_full_mode(
                context, *full_mode, *wrong_mode_moe_command,
                packed_geometry.vocab, packed_geometry.n_expert,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                wrong_mode_moe_buffers, error) &&
                wrong_mode_moe_command->unsafe_native_handle() == nullptr &&
                error.find("FullModeStorage") != std::string::npos,
            "full-mode Window MoE must reject the TieredStorage address table before dispatch");
    const BufferRef tiered_model_usage = tiered_storage->moe_usage_table();
    require(tiered_model_usage && context.buffers().owns(tiered_model_usage) &&
                tiered_model_usage.length == static_cast<std::size_t>(kTieredMaxLayers) *
                                                packed_geometry.n_expert * sizeof(float) &&
                std::all_of(static_cast<const float*>(tiered_model_usage.contents),
                            static_cast<const float*>(tiered_model_usage.contents) +
                                tiered_model_usage.length / sizeof(float),
                            [](float value) { return value == 0.0f; }),
            "tiered ModelStorage must expose its zeroed persistent MoE usage counters");
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
                first.buffer.length() == fixture.router_row_bytes &&
                second.buffer.length() == fixture.second_router_row_bytes &&
                second.buffer.gpu_address() - first.buffer.gpu_address() ==
                    fixture.router_row_bytes &&
                static_cast<const std::uint8_t*>(second.buffer.contents()) ==
                    static_cast<const std::uint8_t*>(first.buffer.contents()) +
                        fixture.router_row_bytes,
            "second converted router view must preserve its nonzero arena offset and metadata");

    const auto* bytes =
        static_cast<const std::uint8_t*>(second.buffer.contents());
    for (std::size_t i = 0; i < fixture.second_router_row_bytes; ++i) {
        const auto expected = static_cast<std::uint8_t>(0x20 + (i % 0x50));
        require(bytes[i] == expected,
                "second converted router view must begin at its own packed row data");
    }
}

BufferRef allocate_zeroed_buffer(Context& context, const std::string& label,
                                 std::size_t bytes) {
    std::string error;
    BufferRef buffer = context.buffers().allocate(label, bytes, error);
    require(static_cast<bool>(buffer), "allocate " + label + ": " + error);
    std::memset(buffer.contents, 0, buffer.length);
    return buffer;
}

BufferRef require_scratch_view(Context& context,
                               const WindowScratchStorage& scratch,
                               WindowScratchRegion region,
                               const std::string& label) {
    (void)context;
    BufferRef view;
    std::string error;
    require(scratch.view(region, view, error),
            "resolve Window scratch " + label + ": " + error);
    return view;
}

void run_fullmode_handoff_test(Context& context,
                               const fs::path& parent_directory,
                               bool use_ple_entry = false) {
    constexpr std::uint32_t kExperts = 256u;
    constexpr std::uint32_t kVocabulary = 1u;
    constexpr std::uint32_t kRows = 1u;
    constexpr std::size_t kGrRows = HC_DIM;
    constexpr std::size_t kGrLowRank = HC_LOW_RANK;
    constexpr std::size_t kGdnQ8Blocks = N_EMBD / 32u;
    constexpr std::size_t kSharedQ8Blocks = 2'560u / 32u;
    constexpr std::size_t kMoeQ8Blocks = 640u / 32u;
    static_assert(sizeof(block_q8_0) == 34u);

    const fs::path directory = parent_directory /
        (use_ple_entry ? "fullmode-ple-handoff" : "fullmode-handoff");
    fs::create_directory(directory);
    const Fixture fixture = write_fullmode_handoff_fixture(directory);
    WindowSources sources = inspect_fixture(fixture);
    Geometry full_geometry = owner_test_geometry(fixture, sources);
    full_geometry.family = "qwen4exp";
    full_geometry.model = "synthetic-fullmode-handoff";
    full_geometry.n_qsa_layers = 1u;
    full_geometry.n_kv_head = 2u;
    full_geometry.head_dim = 256u;
    full_geometry.n_embd = N_EMBD;
    full_geometry.vocab = kVocabulary;
    full_geometry.types = {"f32", "iq4_nl", "q2_0", "q3_k"};
    full_geometry.layers[0].gate_up_type = "q2_0";
    full_geometry.layers[0].down_type = "q2_0";

    std::string error;
    WindowPlan plan;
    require(plan_windows(sources, context.device_facts().max_buffer_length,
                         plan, error),
            "plan sparse 256-expert full-mode fixture: " + error);
    auto full_mode = FullModeStorage::open(
        context, fixture.path.string(), full_geometry, sources, plan, error);
    require(full_mode != nullptr,
            "open sparse production-shaped FullModeStorage: " + error);

    Geometry runtime_geometry;
    runtime_geometry.family = "qwen4exp";
    runtime_geometry.model = "synthetic-window-runtime";
    runtime_geometry.n_layers = 48u;
    runtime_geometry.n_expert = kExperts;
    runtime_geometry.n_qsa_layers = 12u;
    runtime_geometry.n_kv_head = 2u;
    runtime_geometry.head_dim = 256u;
    runtime_geometry.n_embd = N_EMBD;
    runtime_geometry.vocab = kSessionVocabularySize;
    auto scratch =
        WindowScratchStorage::open(context, runtime_geometry, 4u, error);
    require(scratch != nullptr,
            "allocate production-shaped Window scratch: " + error);
    auto session =
        SessionStateStorage::open(context, runtime_geometry, error);
    require(session != nullptr,
            "allocate production-shaped session state: " + error);
    GdnSessionStateViews gdn_state;
    require(session->gdn_layer(0u, gdn_state, error),
            "resolve layer-0 GDN session views: " + error);

    auto params_ring = WindowParamsRing::open(context, error);
    require(params_ring != nullptr,
            "open full-mode handoff WindowParamsRing: " + error);
    const std::array<std::uint32_t, kRows> tokens{0u};
    const std::array<WindowSamplerSettings, kRows> samplers{};
    const BufferRef params = params_ring->build_slot(
        0u, 1u, tokens, samplers, 0u, 0x11223344u, 0x55667788u, error);
    require(static_cast<bool>(params),
            "build ordinary T=1 handoff parameters: " + error);

    const WindowScratchLayout& layout = scratch->layout();
    const BufferRef residual =
        require_scratch_view(context, *scratch, layout.residual, "residual");
    const BufferRef mixed =
        require_scratch_view(context, *scratch, layout.mixed, "mixed");
    const BufferRef block_output =
        require_scratch_view(context, *scratch, layout.block_output,
                              "block output");
    const BufferRef inject =
        require_scratch_view(context, *scratch, layout.inject, "injection");
    const BufferRef inject2 =
        require_scratch_view(context, *scratch, layout.inject2,
                              "second injection");
    const BufferRef gr_lo =
        require_scratch_view(context, *scratch, layout.gr_lo, "GR low rank");
    const BufferRef gr_rs =
        require_scratch_view(context, *scratch, layout.gr_rs, "GR rs");
    const BufferRef gr_xn =
        require_scratch_view(context, *scratch, layout.gr_xn, "GR normalized");
    const BufferRef xq =
        require_scratch_view(context, *scratch, layout.xq, "GDN Q8_1");
    const BufferRef gdn_qkv =
        require_scratch_view(context, *scratch, layout.gdn_qkv_replay,
                              "GDN QKV");
    const BufferRef gdn_h =
        require_scratch_view(context, *scratch, layout.gdn_h_replay,
                              "GDN H");
    const BufferRef gdn_gate =
        require_scratch_view(context, *scratch, layout.gdn_gate_replay,
                              "GDN gate");
    const BufferRef gdn_beta =
        require_scratch_view(context, *scratch, layout.gdn_beta_replay,
                              "GDN beta");
    const BufferRef gdn_z =
        require_scratch_view(context, *scratch, layout.gdn_z, "GDN Z");
    const BufferRef gdn_y =
        require_scratch_view(context, *scratch, layout.gdn_y, "GDN Y");
    const BufferRef gdn_yq = allocate_zeroed_buffer(
        context, "full-mode handoff GDN Y Q8_1",
        (static_cast<std::size_t>(GDN_VALUE_DIM) / 32u) * sizeof(Q81));
    const BufferRef router_logits =
        require_scratch_view(context, *scratch, layout.router_logits,
                              "router logits");
    const BufferRef router_ids =
        require_scratch_view(context, *scratch, layout.router_ids,
                              "router IDs");
    const BufferRef router_weights =
        require_scratch_view(context, *scratch, layout.router_weights,
                              "router weights");
    const BufferRef moe_plans =
        require_scratch_view(context, *scratch, layout.moe_plans, "MoE plans");
    const BufferRef moe_gate =
        require_scratch_view(context, *scratch, layout.moe_gate, "MoE gate");
    const BufferRef moe_up =
        require_scratch_view(context, *scratch, layout.moe_up, "MoE up");
    const BufferRef moe_hidden =
        require_scratch_view(context, *scratch, layout.moe_hidden,
                              "MoE hidden");
    const BufferRef moe_hidden_q =
        require_scratch_view(context, *scratch, layout.moe_hidden_q,
                              "MoE hidden Q8_1");
    const BufferRef moe_parts =
        require_scratch_view(context, *scratch, layout.moe_parts, "MoE parts");
    const BufferRef shared_gate =
        require_scratch_view(context, *scratch, layout.shared_gate,
                              "shared gate");
    const BufferRef shared_up =
        require_scratch_view(context, *scratch, layout.shared_up, "shared up");
    const BufferRef shared_hidden =
        require_scratch_view(context, *scratch, layout.shared_hidden,
                              "shared hidden");
    const BufferRef shared_hidden_q =
        require_scratch_view(context, *scratch, layout.shared_hidden_q,
                              "shared hidden Q8_1");
    const BufferRef shared_output =
        require_scratch_view(context, *scratch, layout.shared_output,
                              "shared down output");
    const BufferRef shared_gate_logit =
        require_scratch_view(context, *scratch, layout.shared_gate_logit,
                              "shared gate logit");
    const BufferRef pending_output = use_ple_entry
        ? require_scratch_view(context, *scratch, layout.embedding,
                               "pending PLE-branch MoE output")
        : require_scratch_view(context, *scratch, layout.ple_input,
                               "pending MoE output");
    const BufferRef attention_entry_mixed =
        require_scratch_view(context, *scratch, layout.ple_value,
                              "next attention mixed");
    const BufferRef next_attention_block_decoy =
        require_scratch_view(context, *scratch, layout.ple_query,
                              "attention block-output decoy");
    const BufferRef next_attention_inject_decoy =
        require_scratch_view(context, *scratch, layout.ple_gate,
                              "attention injection decoy");
    const BufferRef gdn_sink =
        allocate_zeroed_buffer(context, "full-mode handoff GDN sink",
                               sizeof(float));
    const BufferRef gdn_active_rows = allocate_zeroed_buffer(
        context, "full-mode handoff GDN active rows", sizeof(std::uint32_t));
    const BufferRef router_sink =
        allocate_zeroed_buffer(context, "full-mode handoff router sink",
                               sizeof(float));
    const ReadOnlyBufferRef active_rows =
        context.buffers().read_only(params).slice(
            offsetof(strata::metal::WindowParams, nc),
            sizeof(std::uint32_t));
    require(static_cast<bool>(active_rows),
            "form the WindowParams.nc live-row input");
    std::memcpy(gdn_active_rows.contents, &kRows, sizeof(kRows));

    auto allocate = [&](const std::string& label, std::size_t bytes) {
        return allocate_zeroed_buffer(context, label, bytes);
    };
    auto attention_norm =
        allocate("full-mode handoff attention norm", kGrRows * sizeof(float));
    auto attention_down = allocate(
        "full-mode handoff attention down weights",
        kGrLowRank * kGrRows * sizeof(std::uint16_t));
    auto attention_inject_weights = allocate(
        "full-mode handoff attention injection weights",
        4u * kGrRows * sizeof(std::uint16_t));
    auto attention_up = allocate(
        "full-mode handoff attention up weights",
        kGrRows * kGrLowRank * sizeof(std::uint16_t));
    auto ffn_norm =
        allocate("full-mode handoff FFN norm", kGrRows * sizeof(float));
    auto ffn_down = allocate("full-mode handoff FFN down weights",
                             kGrLowRank * kGrRows * sizeof(std::uint16_t));
    auto ffn_inject_weights = allocate(
        "full-mode handoff FFN injection weights",
        4u * kGrRows * sizeof(std::uint16_t));
    auto ffn_up = allocate("full-mode handoff FFN up weights",
                           kGrRows * kGrLowRank * sizeof(std::uint16_t));
    auto next_attention_norm = allocate(
        "full-mode handoff next attention norm",
        kGrRows * sizeof(float));
    auto next_attention_down = allocate(
        "full-mode handoff next attention down weights",
        kGrLowRank * kGrRows * sizeof(std::uint16_t));
    auto next_attention_inject_weights = allocate(
        "full-mode handoff next attention injection weights",
        4u * kGrRows * sizeof(std::uint16_t));
    auto next_attention_up = allocate(
        "full-mode handoff next attention up weights",
        kGrRows * kGrLowRank * sizeof(std::uint16_t));

    auto qkv_weights = allocate(
        "full-mode handoff GDN QKV weights",
        static_cast<std::size_t>(GDN_CONV_CHANNELS) * kGdnQ8Blocks *
            sizeof(block_q8_0));
    auto conv_weights = allocate(
        "full-mode handoff GDN convolution weights",
        static_cast<std::size_t>(GDN_CONV_CHANNELS) * GDN_CONV_KERNEL *
            sizeof(float));
    auto alpha = allocate(
        "full-mode handoff GDN alpha",
        static_cast<std::size_t>(N_EMBD) * GDN_V_HEADS *
            sizeof(std::uint16_t));
    auto beta_weights = allocate(
        "full-mode handoff GDN beta weights",
        static_cast<std::size_t>(N_EMBD) * GDN_V_HEADS *
            sizeof(std::uint16_t));
    auto dt_bias = allocate("full-mode handoff GDN dt bias",
                            GDN_V_HEADS * sizeof(float));
    auto gdn_a = allocate("full-mode handoff GDN A",
                           GDN_V_HEADS * sizeof(float));
    auto gate_weights = allocate(
        "full-mode handoff GDN gate weights",
        static_cast<std::size_t>(GDN_VALUE_DIM) * kGdnQ8Blocks *
            sizeof(block_q8_0));
    auto gamma = allocate("full-mode handoff GDN gamma",
                          GDN_HEAD_DIM * sizeof(float));
    auto output_weights = allocate(
        "full-mode handoff GDN output weights",
        static_cast<std::size_t>(N_EMBD) *
            (GDN_VALUE_DIM / 32u) * sizeof(block_q8_0));

    auto shared_gate_weights = allocate(
        "full-mode handoff shared gate weights",
        static_cast<std::size_t>(640u) * kSharedQ8Blocks *
            sizeof(block_q8_0));
    auto shared_up_weights = allocate(
        "full-mode handoff shared up weights",
        static_cast<std::size_t>(640u) * kSharedQ8Blocks *
            sizeof(block_q8_0));
    auto shared_down_weights = allocate(
        "full-mode handoff shared down weights",
        static_cast<std::size_t>(2'560u) * kMoeQ8Blocks *
            sizeof(block_q8_0));
    auto shared_gate_input_weights = allocate(
        "full-mode handoff shared gate-input weights",
        2'560u * sizeof(std::uint16_t));
    auto shared_gate_sink = allocate(
        "full-mode handoff shared gate sink", sizeof(float));
    auto shared_gate_projection_sink =
        allocate("full-mode handoff shared gate projection sink",
                 sizeof(float));
    auto shared_up_sink =
        allocate("full-mode handoff shared up sink", sizeof(float));
    auto shared_down_sink =
        allocate("full-mode handoff shared down sink", sizeof(float));

    std::fill_n(static_cast<float*>(attention_norm.contents),
                attention_norm.length / sizeof(float), 1.0f);
    std::fill_n(static_cast<float*>(ffn_norm.contents),
                ffn_norm.length / sizeof(float), 1.0f);
    std::fill_n(static_cast<float*>(next_attention_norm.contents),
                next_attention_norm.length / sizeof(float), 1.0f);
    std::fill_n(static_cast<float*>(gamma.contents),
                gamma.length / sizeof(float), 1.0f);
    auto fill_q8_0 = [](BufferRef buffer, std::uint16_t scale_bits,
                        std::int8_t value) {
        auto* blocks = static_cast<block_q8_0*>(buffer.contents);
        const std::size_t count = buffer.length / sizeof(block_q8_0);
        for (std::size_t block = 0; block < count; ++block) {
            blocks[block].d = scale_bits;
            std::fill_n(blocks[block].qs, 32u, value);
        }
    };
    // Positive Q8_0 shared weights make the pending combined output observably
    // nonzero alongside the selected nonzero Q2_0 routed experts.
    fill_q8_0(shared_gate_weights, 0x1c00u, 1);
    fill_q8_0(shared_up_weights, 0x1c00u, 1);
    fill_q8_0(shared_down_weights, 0x1c00u, 1);
    std::fill_n(
        static_cast<std::uint16_t*>(ffn_inject_weights.contents),
        ffn_inject_weights.length / sizeof(std::uint16_t),
        strata::kernels::bf16_from_f32(0.125f));
    require(next_attention_inject_decoy.length >= 4u * sizeof(float),
            "next attention injection decoy covers the T=1 stream values");
    std::fill_n(static_cast<float*>(next_attention_inject_decoy.contents),
                4u, 4.0f);
    std::fill_n(static_cast<std::uint16_t*>(shared_gate_input_weights.contents),
                shared_gate_input_weights.length / sizeof(std::uint16_t),
                strata::kernels::bf16_from_f32(0.01f));
    std::memset(pending_output.contents, 0xa5, pending_output.length);

    FullModeTensorView token_embedding_view;
    require(full_mode->dense_tensor_view("token_embd.weight",
                                         token_embedding_view, error),
            "resolve mapped FullMode token embedding: " + error);
    require(token_embedding_view.ggml_type == kGgmlTypeQ3K &&
                token_embedding_view.shape ==
                    std::vector<std::uint64_t>{N_EMBD, kVocabulary},
            "FullMode token embedding has the expected Q3_K [2560, 1] shape");
    auto embedding_seed_command = context.make_command_buffer(
        "Seed FullMode input stem for handoff reference", error);
    require(embedding_seed_command != nullptr,
            "create FullMode input-stem reference command: " + error);
    require(encode_window_embedding(
        context, *embedding_seed_command, kVocabulary,
                EmbedRowsFormat::q3_k,
                {params, token_embedding_view.buffer, pending_output, residual},
                error),
            "encode mapped FullMode token embedding and GR broadcast: " +
                error);
    require(embedding_seed_command->commit(error),
            "commit FullMode input-stem reference: " + error);
    id<MTLCommandBuffer> native_embedding_seed =
        (__bridge id<MTLCommandBuffer>)embedding_seed_command->unsafe_native_handle();
    [native_embedding_seed waitUntilCompleted];
    require(native_embedding_seed.status == MTLCommandBufferStatusCompleted,
            "FullMode input-stem reference completed");
    const std::vector<std::uint8_t> initial_residual(
        static_cast<const std::uint8_t*>(residual.contents),
        static_cast<const std::uint8_t*>(residual.contents) +
            residual.length);

    FullModeTensorView router_view;
    require(full_mode->dense_tensor_view(fixture.router_tensor_name,
                                         router_view, error),
            "resolve converted BF16 layer-0 router weights: " + error);
    const ReadOnlyBufferRef router_tensor = router_view.buffer;
    require(router_tensor.length() ==
                static_cast<std::size_t>(kExperts) * N_EMBD *
                    sizeof(std::uint16_t),
            "converted router has the production BF16 matrix extent");

    const GRReadBuffers attention{
        residual, next_attention_block_decoy, inject2, attention_norm,
        gr_xn, gr_rs,
        attention_down, attention_inject_weights, gr_lo, inject, attention_up,
        mixed};
    const WindowGDNBlockBuffers gdn{
        mixed, gdn_state.conv_history, gdn_state.recurrent,
        MatVecWeightFormat::q8_0, qkv_weights, conv_weights, alpha,
        beta_weights, dt_bias, gdn_a, MatVecWeightFormat::q8_0,
        gate_weights, gamma, MatVecWeightFormat::q8_0, output_weights, xq,
        gdn_qkv, gdn_h, gdn_gate, gdn_beta, gdn_z, gdn_y, gdn_yq,
        block_output, gdn_sink, gdn_active_rows};
    const GRReadBuffers ffn{
        residual, block_output, inject, ffn_norm,
        gr_xn, gr_rs, ffn_down, ffn_inject_weights, gr_lo, inject2, ffn_up,
        mixed};
    const WindowT1AttentionGDNFFNBuffers layer0{
        params, attention, gdn, ffn};
    const WindowT1MoeRouterPrefixBuffers producer{
        layer0, router_tensor, router_logits, router_ids, router_weights,
        router_sink};
    const MoeRoutedFFNBuffers routed{
        full_mode->expert_address_table(),
        context.buffers().read_only(xq),
        context.buffers().read_only(moe_plans),
        moe_gate, moe_up, moe_hidden, moe_hidden_q, moe_parts};
    const MoeSharedFFNBuffers shared{
        context.buffers().read_only(xq),
        context.buffers().read_only(mixed),
        context.buffers().read_only(shared_gate_weights),
        context.buffers().read_only(shared_up_weights),
        context.buffers().read_only(shared_down_weights),
        context.buffers().read_only(shared_gate_input_weights),
        active_rows,
        context.buffers().read_only(moe_parts),
        context.buffers().read_only(router_weights),
        shared_gate, shared_up, shared_hidden, shared_hidden_q,
        shared_output, shared_gate_logit, pending_output,
        shared_gate_projection_sink, shared_up_sink, shared_down_sink,
        shared_gate_sink};
    const WindowT1MoeLayer0FullBuffers full_buffers{
        {producer, full_mode->resident_table(), full_mode->usage_table(),
         moe_plans},
        {routed, shared}};

    PLEBlockBuffers ple_block{};
    WindowMoePLEEntryBuffers ple_entry{};
    std::vector<std::uint8_t> ple_history_initial;
    if (use_ple_entry) {
        const std::array<std::uint32_t, 2u> committed_tokens{5u, 0u};
        WindowPleRows ple_rows;
        require(build_window_ple_rows(tokens, committed_tokens,
                                      runtime_geometry, ple_rows, error),
                "build layer-1 PLE row IDs: " + error);
        strata::kernels::PleTableReader* ple_reader =
            full_mode->ple_reader();
        require(ple_reader != nullptr && ple_reader->is_open(),
                "resolve FullMode's mapped PLE reader");
        BufferRef ple_input;
        require(stage_window_ple_input(*ple_reader, ple_rows, *scratch,
                                       ple_input, error),
                "stage mapped FullMode PLE rows into Window scratch: " + error);
        require(ple_input.length ==
                    static_cast<std::size_t>(kRows) * N_EMBD * sizeof(float),
                "staged PLE input exposes exactly the active T=1 rows");
        const BufferRef ple_key =
            require_scratch_view(context, *scratch, layout.ple_key,
                                 "PLE key");
        const BufferRef ple_value =
            require_scratch_view(context, *scratch, layout.ple_value,
                                 "PLE value");
        const BufferRef ple_query =
            require_scratch_view(context, *scratch, layout.ple_query,
                                 "PLE query");
        const BufferRef ple_gate =
            require_scratch_view(context, *scratch, layout.ple_gate,
                                 "PLE gate");
        const BufferRef ple_gated =
            require_scratch_view(context, *scratch, layout.ple_gated,
                                 "PLE gated");
        const BufferRef ple_norm =
            require_scratch_view(context, *scratch, layout.ple_norm,
                                 "PLE convolution norm");
        const BufferRef ple_history = session->ple_history();
        require(ple_history && context.buffers().owns(ple_history),
                "resolve Context-owned PLE history for the Window entry");
        ple_history_initial.assign(
            static_cast<const std::uint8_t*>(ple_history.contents),
            static_cast<const std::uint8_t*>(ple_history.contents) +
                ple_history.length);
        const BufferRef ple_key_weights = allocate(
            "FullMode handoff PLE Q2_0 key weights",
            static_cast<std::size_t>(HC_DIM) * (N_EMBD / 64u) *
                sizeof(block_q2_0));
        const BufferRef ple_value_weights = allocate(
            "FullMode handoff PLE BF16 value weights",
            static_cast<std::size_t>(N_EMBD) * N_EMBD *
                sizeof(std::uint16_t));
        const BufferRef ple_key_q8_1 = allocate(
            "FullMode handoff PLE Q8_1 key activations",
            static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81));
        const BufferRef ple_sink =
            require_scratch_view(context, *scratch, layout.matvec_sink,
                                 "PLE mat-vec sink");
        const BufferRef ple_key_gamma =
            allocate("FullMode handoff PLE key gamma",
                     static_cast<std::size_t>(HC_DIM) * sizeof(float));
        const BufferRef ple_query_gamma =
            allocate("FullMode handoff PLE query gamma",
                     static_cast<std::size_t>(HC_DIM) * sizeof(float));
        const BufferRef ple_conv_gamma =
            allocate("FullMode handoff PLE convolution gamma",
                     static_cast<std::size_t>(HC_DIM) * sizeof(float));
        const BufferRef ple_conv1d = allocate(
            "FullMode handoff PLE convolution weights",
            static_cast<std::size_t>(HC_DIM) * 4u * sizeof(std::uint16_t));
        auto* q2_blocks =
            static_cast<block_q2_0*>(ple_key_weights.contents);
        const std::size_t q2_block_count =
            ple_key_weights.length / sizeof(block_q2_0);
        for (std::size_t block = 0; block < q2_block_count; ++block) {
            q2_blocks[block].d = 0x1000u;
            std::fill_n(q2_blocks[block].qs, 16u, 0xaau);
        }
        std::fill_n(
            static_cast<std::uint16_t*>(ple_value_weights.contents),
            ple_value_weights.length / sizeof(std::uint16_t),
            strata::kernels::bf16_from_f32(0.0001f));
        std::fill_n(static_cast<float*>(ple_key_gamma.contents),
                    HC_DIM, 1.0f);
        std::fill_n(static_cast<float*>(ple_query_gamma.contents),
                    HC_DIM, 1.0f);
        std::fill_n(static_cast<float*>(ple_conv_gamma.contents),
                    HC_DIM, 1.0f);
        std::fill_n(static_cast<std::uint16_t*>(ple_conv1d.contents),
                    ple_conv1d.length / sizeof(std::uint16_t),
                    strata::kernels::f16_from_f32(0.125f));
        require(std::all_of(static_cast<const float*>(ple_input.contents),
                            static_cast<const float*>(ple_input.contents) +
                                N_EMBD,
                            [](float value) { return std::isfinite(value); }) &&
                    std::any_of(static_cast<const float*>(ple_input.contents),
                                static_cast<const float*>(ple_input.contents) +
                                    N_EMBD,
                                [](float value) {
                                    return std::abs(value) > 1.0e-7f;
                                }),
                "mapped PLE rows stage finite nonzero table values");
        ple_block = {
            ple_input,
            ple_key_q8_1,
            ple_key_weights,
            ple_value_weights,
            ple_sink,
            gdn_active_rows,
            {ple_key, residual, ple_query, ple_norm, ple_value, ple_gate,
             ple_gated, ple_history, ple_conv1d, ple_key_gamma,
             ple_query_gamma, ple_conv_gamma},
        };
        ple_entry = {full_buffers.ffn, ffn, ple_block};
    }

    const GRReadBuffers next_attention{
        residual, next_attention_block_decoy, next_attention_inject_decoy,
        next_attention_norm, gr_xn, gr_rs, next_attention_down,
        next_attention_inject_weights, gr_lo, inject, next_attention_up,
        attention_entry_mixed};
    const WindowMoeAttentionEntryBuffers attention_entry{
        full_buffers.ffn, ffn, next_attention};
    WindowMoePendingConsumer pending_consumer = attention_entry;
    if (use_ple_entry) {
        pending_consumer = WindowMoePendingPLEConsumer{
            ple_entry, MatVecWeightFormat::q2_0, 1u,
            PLEHistoryUpdate::defer};
    }
    const WindowT1FullModeStepBuffers full_step_buffers{
        pending_output, residual, full_buffers, pending_consumer};

    WindowT1FullModeStepBuffers mismatched_branch = full_step_buffers;
    mismatched_branch.consumer =
        use_ple_entry
            ? WindowMoePendingConsumer{attention_entry}
            : WindowMoePendingConsumer{WindowMoePendingPLEConsumer{}};
    auto branch_mismatch_command = context.make_command_buffer(
        "Reject mismatched FullMode consumer branch", error);
    require(branch_mismatch_command != nullptr,
            "create mismatched FullMode branch command: " + error);
    const bool branch_mismatch_encoded = Encoder::encode(
        context, *branch_mismatch_command, "mismatched FullMode branch",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_full_mode_step(
                context, *full_mode, encoder, kVocabulary, kExperts,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                use_ple_entry, mismatched_branch, encode_error);
        },
        error);
    require(!branch_mismatch_encoded &&
                error.find("does not match the selected PLE branch") !=
                    std::string::npos,
            "T=1 FullMode step must reject a branch/payload mismatch before dispatch");

    WindowMoePendingConsumer mismatched_consumer =
        use_ple_entry
            ? WindowMoePendingConsumer{attention_entry}
            : WindowMoePendingConsumer{WindowMoePendingPLEConsumer{
                  ple_entry, MatVecWeightFormat::q2_0, 1u,
                  PLEHistoryUpdate::defer}};
    auto mismatch_command = context.make_command_buffer(
        "Reject mismatched Window pending consumer", error);
    require(mismatch_command != nullptr,
            "create mismatched Window branch command: " + error);
    const bool mismatch_encoded = Encoder::encode(
        context, *mismatch_command, "mismatched Window pending consumer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_pending_consumer(
                context, encoder, kRows, use_ple_entry,
                mismatched_consumer, encode_error);
        },
        error);
    require(!mismatch_encoded &&
                error.find("does not match the selected PLE branch") !=
                    std::string::npos,
            "Window pending consumer must reject a branch/tag mismatch before dispatch");

    WindowT1FullModeStepBuffers mismatched_streams = full_step_buffers;
    mismatched_streams.streams = attention_entry_mixed;
    auto stream_mismatch_command = context.make_command_buffer(
        "Reject mismatched FullMode input streams", error);
    require(stream_mismatch_command != nullptr,
            "create mismatched FullMode input-stream command: " + error);
    const bool stream_mismatch_encoded = Encoder::encode(
        context, *stream_mismatch_command, "mismatched FullMode input streams",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_full_mode_step(
                context, *full_mode, encoder, kVocabulary, kExperts,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                use_ple_entry, mismatched_streams, encode_error);
        },
        error);
    require(!stream_mismatch_encoded &&
                error.find("must be the layer-0 attention residual view") !=
                    std::string::npos,
            "T=1 FullMode step must reject streams that do not feed layer-0 attention");

    WindowT1FullModeStepBuffers later_weight_alias = full_step_buffers;
    later_weight_alias.embedding = attention_norm;
    auto later_weight_alias_command = context.make_command_buffer(
        "Reject embedding overlap with later attention weights", error);
    require(later_weight_alias_command != nullptr,
            "create later-weight alias command: " + error);
    const bool later_weight_alias_encoded = Encoder::encode(
        context, *later_weight_alias_command, "later attention weight alias",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_full_mode_step(
                context, *full_mode, encoder, kVocabulary, kExperts,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                use_ple_entry, later_weight_alias, encode_error);
        },
        error);
    require(!later_weight_alias_encoded &&
                error.find("embedding scratch overlaps") != std::string::npos,
            "T=1 FullMode step must reject embedding scratch reused by a later attention read");

    WindowT1FullModeStepBuffers partial_pending_alias = full_step_buffers;
    partial_pending_alias.embedding = pending_output.slice(
        sizeof(float), pending_output.length - sizeof(float));
    require(static_cast<bool>(partial_pending_alias.embedding),
            "create partial view into the pending MoE output");
    auto partial_pending_alias_command = context.make_command_buffer(
        "Reject partial embedding overlap with pending output", error);
    require(partial_pending_alias_command != nullptr,
            "create partial pending-output alias command: " + error);
    const bool partial_pending_alias_encoded = Encoder::encode(
        context, *partial_pending_alias_command,
        "partial pending-output alias",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_t1_full_mode_step(
                context, *full_mode, encoder, kVocabulary, kExperts,
                MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                use_ple_entry, partial_pending_alias, encode_error);
        },
        error);
    require(!partial_pending_alias_encoded &&
                error.find("embedding scratch overlaps") != std::string::npos,
            "embedding scratch may reuse shared.output only when the views are exactly equal");

    WindowT1FullModeStepBuffers late_failure = full_step_buffers;
    if (use_ple_entry) {
        WindowMoePendingPLEConsumer invalid_ple =
            std::get<WindowMoePendingPLEConsumer>(late_failure.consumer);
        invalid_ple.buffers.ple.post.residual = attention_entry_mixed;
        late_failure.consumer = std::move(invalid_ple);
    } else {
        WindowMoeAttentionEntryBuffers invalid_attention =
            std::get<WindowMoeAttentionEntryBuffers>(late_failure.consumer);
        invalid_attention.attention.residual = attention_entry_mixed;
        late_failure.consumer = std::move(invalid_attention);
    }
    const std::vector<std::uint8_t> residual_before_late_failure(
        static_cast<const std::uint8_t*>(residual.contents),
        static_cast<const std::uint8_t*>(residual.contents) + residual.length);
    const std::vector<std::uint8_t> pending_before_late_failure(
        static_cast<const std::uint8_t*>(pending_output.contents),
        static_cast<const std::uint8_t*>(pending_output.contents) +
            pending_output.length);
    auto late_failure_command = context.make_command_buffer(
        "Discard failed FullMode T=1 step", error);
    require(late_failure_command != nullptr,
            "create late-failure FullMode step command: " + error);
    const bool late_failure_encoded = encode_window_t1_full_mode_step(
        context, *full_mode, *late_failure_command, kVocabulary, kExperts,
        MoeExpertFormat::q2_0, MoeDownFormat::q2_0, use_ple_entry,
        late_failure, error);
    require(!late_failure_encoded,
            "T=1 FullMode step must propagate a late consumer validation error");
    require(late_failure_command->unsafe_native_handle() == nullptr,
            "standalone T=1 FullMode step must discard earlier dispatches after late failure");
    require(std::equal(
                residual_before_late_failure.begin(),
                residual_before_late_failure.end(),
                static_cast<const std::uint8_t*>(residual.contents)) &&
                std::equal(
                    pending_before_late_failure.begin(),
                    pending_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(pending_output.contents)),
            "late consumer failure must discard the input stem and producer segment");

    auto row_count_command = context.make_command_buffer(
        "Reject multi-row Window pending consumer", error);
    require(row_count_command != nullptr,
            "create multi-row Window consumer command: " + error);
    const bool row_count_encoded = Encoder::encode(
        context, *row_count_command, "multi-row Window pending consumer",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_pending_consumer(
                context, encoder, 2u, use_ple_entry, pending_consumer,
                encode_error);
        },
        error);
    require(!row_count_encoded &&
                error.find("requires exactly one row") != std::string::npos,
            "T=1 pending consumer must reject rows that its FullMode producer did not write");

    auto command = context.make_command_buffer(
        "FullMode layer-0 MoE to no-PLE attention handoff", error);
    require(command != nullptr,
            "create same-segment FullMode handoff command: " + error);
    require(session->track_use(*command, error),
            "track GDN state through FullMode handoff: " + error);
    constexpr std::size_t kActiveEmbeddingBytes =
        N_EMBD * sizeof(float);
    constexpr std::size_t kActiveResidualBytes =
        4u * N_EMBD * sizeof(float);
    std::vector<std::uint8_t> embedding_sentinel(
        kActiveEmbeddingBytes, 0x5au);
    std::memcpy(pending_output.contents, embedding_sentinel.data(),
                embedding_sentinel.size());
    require(!std::equal(initial_residual.begin(),
                        initial_residual.begin() + kActiveEmbeddingBytes,
                        embedding_sentinel.begin()),
            "the reference token embedding differs from the integrated embedding sentinel");
    std::vector<std::uint8_t> residual_sentinel(
        kActiveResidualBytes, 0x3cu);
    std::memcpy(residual.contents, residual_sentinel.data(),
                residual_sentinel.size());
    require(!std::equal(initial_residual.begin(),
                        initial_residual.begin() + kActiveResidualBytes,
                        residual_sentinel.begin()),
            "the reference input stem differs from the integrated residual sentinel");
    const bool encoded = encode_window_t1_full_mode_step(
        context, *full_mode, *command, kVocabulary, kExperts,
        MoeExpertFormat::q2_0, MoeDownFormat::q2_0, use_ple_entry,
        full_step_buffers, error);
    require(encoded,
            "encode FullMode token embedding, producer and selected consumer in one segment: " +
                error);
    require(command->commit(error),
            "commit FullMode input stem, producer and pending consumer: " +
                error);
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    [native waitUntilCompleted];
    require(native.status == MTLCommandBufferStatusCompleted,
            "same-segment FullMode input stem, producer and pending consumer completed: " +
                (native.error ? std::string(native.error.localizedDescription.UTF8String)
                              : std::string{}));

    const auto* route_id_values =
        static_cast<const std::int32_t*>(router_ids.contents);
    std::array<bool, kExperts> selected{};
    for (std::size_t route = 0; route < 10u; ++route) {
        const std::int32_t expert = route_id_values[route];
        require(expert >= 0 && static_cast<std::uint32_t>(expert) < kExperts &&
                    !selected[static_cast<std::size_t>(expert)] &&
                    static_cast<std::size_t>(expert) == route,
                "FullMode top-10 route stably selects the populated expert slabs");
        selected[static_cast<std::size_t>(expert)] = true;
    }
    const auto* route_weight_values =
        static_cast<const float*>(router_weights.contents);
    float route_weight_sum = 0.0f;
    for (std::size_t route = 0; route < 10u; ++route) {
        require(std::isfinite(route_weight_values[route]) &&
                    route_weight_values[route] > 0.0f,
                "FullMode top-10 route weights are positive and finite");
        route_weight_sum += route_weight_values[route];
    }
    require(std::abs(route_weight_sum - 1.0f) < 1.0e-5f,
            "FullMode decode route weights sum to one");
    const auto* routed_part_values =
        static_cast<const float*>(moe_parts.contents);
    bool routed_parts_finite = true;
    bool routed_parts_nonzero = false;
    for (std::size_t index = 0; index < 10u * 2'560u; ++index) {
        routed_parts_finite &= std::isfinite(routed_part_values[index]);
        routed_parts_nonzero |=
            std::abs(routed_part_values[index]) > 1.0e-9f;
    }
    require(routed_parts_finite && routed_parts_nonzero,
            "FullMode Q2_0 routed experts produce finite nonzero down parts");
    const auto* pending_values =
        static_cast<const float*>(pending_output.contents);
    bool pending_is_finite = true;
    bool pending_is_nonzero = false;
    for (std::size_t index = 0; index < N_EMBD; ++index) {
        pending_is_finite &= std::isfinite(pending_values[index]);
        pending_is_nonzero |= std::abs(pending_values[index]) > 1.0e-7f;
    }
    require(pending_is_finite && pending_is_nonzero,
            "FullMode routed/shared combine leaves a finite nonzero pending output");
    const auto* producer_injection =
        static_cast<const float*>(inject2.contents);
    const auto* injection_decoy =
        static_cast<const float*>(next_attention_inject_decoy.contents);
    require(std::all_of(producer_injection, producer_injection + 4u,
                        [](float value) { return std::isfinite(value); }) &&
                std::any_of(producer_injection, producer_injection + 4u,
                            [](float value) {
                                return std::abs(value) > 1.0e-7f;
                            }),
            "FullMode producer's FFN read emits a finite nonzero injection");
    if (!use_ple_entry) {
        require(!std::equal(
                    reinterpret_cast<const std::uint8_t*>(producer_injection),
                    reinterpret_cast<const std::uint8_t*>(producer_injection) +
                        4u * sizeof(float),
                    reinterpret_cast<const std::uint8_t*>(injection_decoy)),
                "next attention injection decoy differs from the producer injection");
    }

    if (use_ple_entry) {
        const std::array<BufferRef, 6u> outputs{
            ple_block.post.key, ple_block.post.value, ple_block.post.query,
            ple_block.post.gate, ple_block.post.gated,
            ple_block.post.ple_norm};
        const std::array<std::size_t, 6u> extents{
            kRows * HC_DIM * sizeof(float),
            kRows * N_EMBD * sizeof(float),
            kRows * HC_DIM * sizeof(float), kRows * HC * sizeof(float),
            kRows * HC_DIM * sizeof(float), kRows * HC_DIM * sizeof(float)};
        std::array<std::vector<std::uint8_t>, 6u> actual_ple_outputs;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            require(outputs[index].length >= extents[index],
                    "PLE handoff output view covers its active T=1 extent");
            const auto* begin =
                static_cast<const std::uint8_t*>(outputs[index].contents);
            actual_ple_outputs[index].assign(begin, begin + extents[index]);
        }
        const std::vector<std::uint8_t> actual_residual(
            static_cast<const std::uint8_t*>(residual.contents),
            static_cast<const std::uint8_t*>(residual.contents) +
                residual.length);
        const std::vector<std::uint8_t> pending_snapshot(
            static_cast<const std::uint8_t*>(pending_output.contents),
            static_cast<const std::uint8_t*>(pending_output.contents) +
                pending_output.length);
        const auto finite_nonzero = [](BufferRef view, std::size_t count) {
            const auto* values = static_cast<const float*>(view.contents);
            return std::all_of(values, values + count,
                               [](float value) {
                                   return std::isfinite(value);
                               }) &&
                   std::any_of(values, values + count, [](float value) {
                       return std::abs(value) > 1.0e-7f;
                   });
        };
        require(finite_nonzero(ple_block.post.value, N_EMBD) &&
                    finite_nonzero(ple_block.post.gated, HC_DIM),
                "same-segment FullMode-to-PLE consumer produces finite nonzero value and gated rows");

        BufferRef reference_residual = allocate_zeroed_buffer(
            context, "full-mode PLE handoff reference residual",
            residual.length);
        std::memcpy(reference_residual.contents, initial_residual.data(),
                    initial_residual.size());
        PLEBlockBuffers reference_ple = ple_block;
        reference_ple.post.residual = reference_residual;
        const WindowPLEEntryBuffers reference_entry{
            pending_output, inject2, reference_ple};
        auto reference_command = context.make_command_buffer(
            "Reference FullMode-to-PLE entry", error);
        require(reference_command != nullptr,
                "create direct PLE-entry reference command: " + error);
        require(session->track_use(*reference_command, error),
                "track PLE history through the direct reference: " + error);
        require(encode_window_ple_entry(
                    context, *reference_command, kRows,
                    MatVecWeightFormat::q2_0, 1u,
                    PLEHistoryUpdate::defer, reference_entry, error),
                "encode direct GR-post plus PLE-entry reference: " + error);
        require(reference_command->commit(error),
                "commit direct PLE-entry reference: " + error);
        id<MTLCommandBuffer> native_reference =
            (__bridge id<MTLCommandBuffer>)reference_command->unsafe_native_handle();
        [native_reference waitUntilCompleted];
        require(native_reference.status == MTLCommandBufferStatusCompleted,
                "direct PLE-entry reference completed");

        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const auto* expected =
                static_cast<const std::uint8_t*>(outputs[index].contents);
            require(std::equal(actual_ple_outputs[index].begin(),
                               actual_ple_outputs[index].end(), expected),
                    "typed FullMode PLE handoff matches the direct PLE entry for output " +
                        std::to_string(index));
        }
        require(actual_residual.size() == reference_residual.length &&
                    std::equal(actual_residual.begin(), actual_residual.end(),
                               static_cast<const std::uint8_t*>(
                                   reference_residual.contents)),
                "typed PLE handoff matches direct GR-post plus PLE residual and applies the pending write once");
        require(std::equal(
                    pending_snapshot.begin(), pending_snapshot.end(),
                    static_cast<const std::uint8_t*>(pending_output.contents)),
                "PLE consumer preserves the FullMode pending output");
        require(std::equal(
                    ple_history_initial.begin(), ple_history_initial.end(),
                    static_cast<const std::uint8_t*>(
                        ple_block.post.history.contents)),
                "deferred Window PLE entry preserves committed history");
        return;
    }

    const auto output_snapshots = [&] {
        const std::array<BufferRef, 5u> outputs{
            gr_xn, gr_rs, gr_lo, inject, attention_entry_mixed};
        const std::array<std::size_t, 5u> extents{
            4u * N_EMBD * sizeof(float), 4u * sizeof(float),
            kRows * HC_LOW_RANK * sizeof(float), 4u * sizeof(float),
            N_EMBD * sizeof(float)};
        std::array<std::vector<std::uint8_t>, 5u> snapshots;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const auto* begin =
                static_cast<const std::uint8_t*>(outputs[index].contents);
            require(outputs[index].length >= extents[index],
                    "GR handoff output view covers its T=1 write extent");
            snapshots[index].assign(begin, begin + extents[index]);
        }
        return snapshots;
    };
    const auto actual_outputs = output_snapshots();
    const std::vector<std::uint8_t> pending_snapshot(
        static_cast<const std::uint8_t*>(pending_output.contents),
        static_cast<const std::uint8_t*>(pending_output.contents) +
            pending_output.length);
    BufferRef reference_residual = allocate_zeroed_buffer(
        context, "full-mode handoff reference residual", residual.length);
    std::memcpy(reference_residual.contents, initial_residual.data(),
                initial_residual.size());

    const BufferRef reference_normalized =
        require_scratch_view(context, *scratch, layout.gdn_qkv_replay,
                             "reference normalized");
    const BufferRef reference_lo =
        require_scratch_view(context, *scratch, layout.gdn_h_replay,
                             "reference low rank");
    const BufferRef reference_inject_out =
        require_scratch_view(context, *scratch, layout.gdn_gate_replay,
                             "reference injection output");
    const BufferRef reference_mixed =
        require_scratch_view(context, *scratch, layout.gdn_beta_replay,
                             "reference mixed");
    const BufferRef reference_rs =
        require_scratch_view(context, *scratch, layout.ple_gate,
                             "reference rs");
    const GRReadBuffers reference_attention{
        reference_residual, pending_output, inject2,
        next_attention_norm,
        reference_normalized, reference_rs, next_attention_down,
        next_attention_inject_weights, reference_lo, reference_inject_out,
        next_attention_up, reference_mixed};
    auto reference_command = context.make_command_buffer(
        "Reference pending MoE attention read", error);
    require(reference_command != nullptr,
            "create reference pending attention command: " + error);
    require(encode_gr_read(context, *reference_command, kRows, true,
                           reference_attention, error),
            "encode direct GR read using actual pending producer views: " +
                error);
    require(reference_command->commit(error),
            "commit reference pending attention read: " + error);
    id<MTLCommandBuffer> native_reference =
        (__bridge id<MTLCommandBuffer>)reference_command->unsafe_native_handle();
    [native_reference waitUntilCompleted];
    require(native_reference.status == MTLCommandBufferStatusCompleted,
            "reference pending attention read completed");

    const auto exact_output_match = [&](std::size_t actual_index,
                                        BufferRef expected,
                                        const char* label) {
        const auto* expected_bytes =
            static_cast<const std::uint8_t*>(expected.contents);
        const bool matches = actual_outputs[actual_index].size() <= expected.length &&
            std::equal(actual_outputs[actual_index].begin(),
                       actual_outputs[actual_index].end(), expected_bytes);
        if (!matches) {
            const float actual_first =
                *reinterpret_cast<const float*>(
                    actual_outputs[actual_index].data());
            const float expected_first =
                *static_cast<const float*>(expected.contents);
            require(false, std::string("typed attention handoff matches direct ") +
                               label + " read (first values " +
                               std::to_string(actual_first) + " vs " +
                               std::to_string(expected_first) + ")");
        }
    };
    exact_output_match(0u, reference_normalized, "normalized");
    exact_output_match(1u, reference_rs, "rs");
    exact_output_match(2u, reference_lo, "low-rank");
    exact_output_match(3u, reference_inject_out, "injection");
    exact_output_match(4u, reference_mixed, "mixed");
    require(std::equal(
                static_cast<const std::uint8_t*>(residual.contents),
                static_cast<const std::uint8_t*>(residual.contents) +
                    residual.length,
                static_cast<const std::uint8_t*>(reference_residual.contents)),
            "typed no-PLE consumer applies the pending write once to GR residual");
    require(std::equal(pending_snapshot.begin(), pending_snapshot.end(),
                       static_cast<const std::uint8_t*>(pending_output.contents)),
            "no-PLE attention consumer preserves its pending producer output");
    const auto* consumed_mixed =
        static_cast<const float*>(attention_entry_mixed.contents);
    require(std::all_of(consumed_mixed,
                        consumed_mixed + N_EMBD,
                        [](float value) { return std::isfinite(value); }) &&
                std::any_of(consumed_mixed, consumed_mixed + N_EMBD,
                            [](float value) {
                                return std::abs(value) > 1.0e-7f;
                            }),
            "no-PLE attention read consumes the pending output into finite mixed activations");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool never_wrap_only = argc == 3 && std::string(argv[2]) == "--never-wrap";
        const bool tiered_copy_only = argc == 3 && std::string(argv[2]) == "--tiered-copy";
        const bool model_owner_only = argc == 3 && std::string(argv[2]) == "--model-owner";
        const bool fullmode_handoff_only =
            argc == 3 && std::string(argv[2]) == "--fullmode-handoff";
        const bool fullmode_ple_handoff_only =
            argc == 3 && std::string(argv[2]) == "--fullmode-ple-handoff";
        require(argc == 2 || never_wrap_only || tiered_copy_only ||
                    model_owner_only || fullmode_handoff_only ||
                    fullmode_ple_handoff_only,
                "usage: metal_window_storage_test strata.metallib "
                "[--never-wrap|--tiered-copy|--model-owner|--fullmode-handoff|"
                "--fullmode-ple-handoff]");
        if (argc == 2) test_warmup_wave_barrier();
        char directory_template[] = "/tmp/strata-window-storage.XXXXXX";
        char* directory_name = mkdtemp(directory_template);
        require(directory_name != nullptr, "create temporary fixture directory");
        const fs::path directory(directory_name);
        const Fixture fixture = write_fixture(
            directory, false, false, true, 0, false,
            tiered_copy_only ? 3u : 1u);
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
        if (fullmode_handoff_only) {
            run_fullmode_handoff_test(*context, directory);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_handoff_test: OK\n";
            return 0;
        }
        if (fullmode_ple_handoff_only) {
            run_fullmode_handoff_test(*context, directory, true);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_ple_handoff_test: OK\n";
            return 0;
        }
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
                    storage->windows()[0][0].length() > 8 * kWindowPageBytes &&
                    storage->windows()[0][0].length() % kWindowPageBytes == 0,
                "one multi-page, page-rounded GGUF window should be wrapped");

        const FullModeLayout& layout = storage->layout();
        FullModeTensorView token_view;
        require(storage->dense_tensor_view("token_embd.weight", token_view, error),
                "resolve a whole GGUF tensor view: " + error);
        require(token_view.ggml_type == 0 &&
                    token_view.shape == std::vector<std::uint64_t>{64, 64} &&
                    token_view.buffer &&
                    token_view.buffer.length() == 64 * 64 * sizeof(float) &&
                    *static_cast<const std::uint8_t*>(token_view.buffer.contents()) == 0x4d,
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
        const ReadOnlyBufferRef dense_pack_arena = storage->dense_pack_arena();
        require(router_binding != layout.tensors.end() && dense_pack_arena &&
                    dense_pack_arena.length() == 3 * kWindowPageBytes &&
                    router_binding->gpu_address == dense_pack_arena.gpu_address() &&
                    router_binding->byte_count == fixture.dense_pack_bytes,
                "converted router address binding must point to the aligned BF16 arena");
        std::vector<std::uint8_t> packed_router(
            static_cast<std::size_t>(fixture.dense_pack_bytes));
        {
            std::ifstream input(fixture.dense_pack_path, std::ios::binary);
            input.read(reinterpret_cast<char*>(packed_router.data()),
                       static_cast<std::streamsize>(packed_router.size()));
            require(input.gcount() == static_cast<std::streamsize>(packed_router.size()) &&
                        std::memcmp(dense_pack_arena.contents(), packed_router.data(),
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
            const ReadOnlyBufferRef& window = storage->windows()[0][index];
            const std::uint64_t mapped_ple_start =
                window.gpu_address() + (overlap_start - file_window.start);
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

        const ReadOnlyBufferRef address_table = storage->expert_address_table();
        const ReadOnlyBufferRef resident_table = storage->resident_table();
        require(address_table && resident_table, "full-mode tables must be shared Metal buffers");
        const auto* mapped_addresses = static_cast<const std::uint64_t*>(address_table.contents());
        const auto* mapped_resident = static_cast<const std::int32_t*>(resident_table.contents());
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

        const ReadOnlyBufferRef live_window = storage->windows()[0][0];
        const std::uint64_t live_window_start = plan.files[0].windows[0].start;
        storage.reset();
        const auto* payload = static_cast<const std::uint8_t*>(live_window.contents());
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
