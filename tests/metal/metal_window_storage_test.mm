// tests/metal/metal_window_storage_test.mm - live GGUF mmap-to-Metal full-mode mapping.
#include "strata/metal/window.hpp"
#include "strata/metal/tiered.hpp"
#include "strata/metal/model_storage.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/experts.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_moe_expert_dispatch.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"
#include "strata/metal/moe_dispatch.hpp"
#include "strata/metal/window_dispatch.hpp"
#include "strata/metal/window_ple_dispatch.hpp"
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"
#include "strata/metal/window_ple_stage.hpp"
#include "strata/metal/window_t1_input.hpp"
#include "strata/metal/window_speculative_input.hpp"
#include "strata/metal/window_scratch.hpp"
#include "strata/metal/window_output_projection.hpp"
#include "strata/metal/window_output_head.hpp"
#include "strata/metal/window_output_sampling.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/metal/session_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "window_source_lock_test_access.hpp"
#include "oracles/iq4_xs_mv.hpp"
#include "oracles/bf16_mv.hpp"
#include "oracles/moe_plan.hpp"
#include "oracles/moe_route.hpp"
#include "oracles/moe_gu_q2_0.hpp"
#include "oracles/moe_down_q2_0.hpp"
#include "oracles/moe_swiglu.hpp"
#include "oracles/moe_combine.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"
#include "twin.hpp"
#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <Metal/Metal.h>
#include <libproc.h>

#include <algorithm>
#include <array>
#include <bit>
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
#include <span>
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
constexpr std::uint32_t kGgmlTypeQ8_0 = 8u;
constexpr std::uint32_t kGgmlTypeBF16 = 30u;
constexpr std::uint32_t kGgmlTypeQ2_0 = 42u;
constexpr std::uint32_t kGgmlTypeIQ4XS = 23u;

MatVecWeightFormat fullmode_ple_key_format(std::uint32_t ggml_type) {
    switch (ggml_type) {
        case kGgmlTypeQ2_0:
            return MatVecWeightFormat::q2_0;
        case kGgmlTypeQ8_0:
            return MatVecWeightFormat::q8_0;
        case kGgmlTypeBF16:
            return MatVecWeightFormat::bf16;
        default:
            throw std::runtime_error(
                "unsupported synthetic FullMode PLE key type");
    }
}

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

void set_iq4_xs_fixture_code(block_iq4_xs& block, std::uint32_t index,
                             std::uint32_t code) {
    const std::uint32_t subblock = index / 32u;
    const std::uint32_t local = index % 32u;
    const std::uint32_t byte_index =
        16u * subblock + (local < 16u ? local : local - 16u);
    if (local < 16u) {
        block.qs[byte_index] = static_cast<std::uint8_t>(
            (block.qs[byte_index] & 0xf0u) | (code & 0x0fu));
    } else {
        block.qs[byte_index] = static_cast<std::uint8_t>(
            (block.qs[byte_index] & 0x0fu) | ((code & 0x0fu) << 4u));
    }
}

void set_iq4_xs_fixture_scale(block_iq4_xs& block, std::uint32_t subblock,
                              std::uint32_t code) {
    const std::uint32_t low_shift = 4u * (subblock % 2u);
    const std::uint8_t low_mask =
        static_cast<std::uint8_t>(0x0fu << low_shift);
    block.scales_l[subblock / 2u] = static_cast<std::uint8_t>(
        (block.scales_l[subblock / 2u] &
         static_cast<std::uint8_t>(~low_mask)) |
        static_cast<std::uint8_t>((code & 0x0fu) << low_shift));
    const std::uint32_t high_shift = 2u * subblock;
    const std::uint16_t high_mask =
        static_cast<std::uint16_t>(0x03u << high_shift);
    block.scales_h = static_cast<std::uint16_t>(
        (block.scales_h & static_cast<std::uint16_t>(~high_mask)) |
        static_cast<std::uint16_t>(((code >> 4u) & 0x03u) << high_shift));
}

struct TestMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(TestMoeExpertAddress) == 24);

struct TestPrefillMoePlan {
    std::uint32_t n_groups = 0u;
    std::uint32_t n_entries = 0u;
    std::uint32_t n_miss = 0u;
    std::uint32_t skip = 0u;
    std::uint32_t n_tiles = 0u;
    std::uint32_t bad = 0u;
    std::uint32_t pad[2]{};
    std::int32_t grp_e[512]{};
    std::int32_t grp_res[512]{};
    std::int32_t grp_start[513]{};
};
static_assert(sizeof(TestPrefillMoePlan) == kPrefillMoePlanBytes);

struct TestPrefillMoeTile {
    std::uint32_t group = 0u;
    std::uint32_t first_entry = 0u;
};
static_assert(sizeof(TestPrefillMoeTile) == 8u);

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
    std::uint64_t output_weight_offset = 0;
    std::uint64_t output_weight_bytes = 0;
    std::array<block_iq4_xs, 2'560u / 256u> output_weight_first_row{};
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

Fixture write_fullmode_handoff_fixture(
    const fs::path& directory,
    std::uint32_t ple_key_type = kGgmlTypeQ2_0,
    bool include_output_weight = false,
    bool include_output_head = false) {
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
    const std::string fixture_suffix =
        std::string("fullmode-handoff-") + std::to_string(ple_key_type) +
        (include_output_head ? "-head" :
         include_output_weight ? "-output" : "");
    fixture.path = directory / (fixture_suffix + ".gguf");
    fixture.dense_pack_path = directory /
        (fixture_suffix + "-dense.bin");
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
        {"blk.1.ple_key.weight", {2'560u, 10'240u}, ple_key_type, 0u},
        {"blk.1.ple_value.weight", {2'560u, 2'560u}, kGgmlTypeBF16, 0u},
        {"blk.1.ple_norm_key.weight", {10'240u}, 0u, 0u},
        {"blk.1.ple_norm_query.weight", {10'240u}, 0u, 0u},
        {"blk.1.ple_norm_conv.weight", {10'240u}, 0u, 0u},
        {"blk.1.ple_conv1d.weight", {4u, 10'240u}, 1u, 0u},
        {"per_layer_token_embd.weight",
         {strata::kernels::PLE_HEAD_DIM, kPleRows}, 20u, 0u},
    };
    if (include_output_head) {
        tensors.push_back(
            {"output_hc_norm.weight", {10'240u}, 0u, 0u});
        tensors.push_back(
            {"output_hc_down.weight", {10'240u, 320u}, kGgmlTypeBF16, 0u});
        tensors.push_back(
            {"output_hc_up.weight", {320u, 10'240u}, kGgmlTypeBF16, 0u});
    }
    if (include_output_weight) {
        tensors.push_back(
            {"output.weight", {2'560u, 248'320u}, kGgmlTypeIQ4XS, 0u});
    }
    std::unordered_map<std::string, std::uint64_t> dense_tensor_offsets;
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
        } else if (tensor.name == "per_layer_token_embd.weight") {
            fixture.ple_offset = cursor;
            cursor += kPleBytes;
        } else if (tensor.name == "output.weight") {
            fixture.output_weight_offset = cursor;
            fixture.output_weight_bytes =
                (2'560u * 248'320u / 256u) * sizeof(block_iq4_xs);
            cursor += fixture.output_weight_bytes;
        } else {
            std::uint64_t elements = 1u;
            for (const std::uint64_t dimension : tensor.shape) {
                require(dimension != 0u &&
                            elements <=
                                std::numeric_limits<std::uint64_t>::max() /
                                    dimension,
                        "FullMode PLE tensor fixture shape product");
                elements *= dimension;
            }
            std::uint64_t tensor_bytes = 0u;
            if (tensor.type == kGgmlTypeQ2_0) {
                require(elements % 64u == 0u,
                        "FullMode PLE Q2_0 fixture block alignment");
                tensor_bytes = (elements / 64u) * sizeof(block_q2_0);
            } else if (tensor.type == kGgmlTypeQ8_0) {
                require(elements % 32u == 0u,
                        "FullMode PLE Q8_0 fixture block alignment");
                tensor_bytes = (elements / 32u) * sizeof(block_q8_0);
            } else if (tensor.type == kGgmlTypeBF16 || tensor.type == 1u) {
                tensor_bytes = elements * sizeof(std::uint16_t);
            } else {
                require(tensor.type == 0u,
                        "FullMode PLE fixture only uses supported scalar formats");
                tensor_bytes = elements * sizeof(float);
            }
            dense_tensor_offsets.emplace(tensor.name, cursor);
            cursor += tensor_bytes;
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

        if (include_output_weight) {
            block_iq4_xs& first_block = fixture.output_weight_first_row.front();
            first_block.d = strata::kernels::f16_from_f32(0.25f);
            for (std::uint32_t subblock = 0u; subblock < 8u; ++subblock) {
                set_iq4_xs_fixture_scale(first_block, subblock, 40u);
            }
            for (std::uint32_t index = 0u; index < 256u; ++index) {
                set_iq4_xs_fixture_code(first_block, index, 15u);
            }
            write_all_at(
                descriptor, fixture.output_weight_first_row.data(),
                sizeof(fixture.output_weight_first_row),
                fixture.data_start + fixture.output_weight_offset);
        }

        if (include_output_head) {
            const std::vector<float> norm_gamma(10'240u, 1.0f);
            write_all_at(
                descriptor, norm_gamma.data(),
                norm_gamma.size() * sizeof(float),
                fixture.data_start +
                    dense_tensor_offsets.at("output_hc_norm.weight"));

            std::vector<std::uint16_t> down_weights(
                10'240u * 320u, strata::kernels::bf16_from_f32(0.0f));
            down_weights[0] = strata::kernels::bf16_from_f32(1.0f);
            write_all_at(
                descriptor, down_weights.data(),
                down_weights.size() * sizeof(std::uint16_t),
                fixture.data_start +
                    dense_tensor_offsets.at("output_hc_down.weight"));

            std::vector<std::uint16_t> up_weights(
                320u * 10'240u, strata::kernels::bf16_from_f32(0.0f));
            for (std::size_t output_row = 0u; output_row < 10'240u;
                 ++output_row) {
                up_weights[output_row * 320u] =
                    strata::kernels::bf16_from_f32(0.5f);
            }
            write_all_at(
                descriptor, up_weights.data(),
                up_weights.size() * sizeof(std::uint16_t),
                fixture.data_start +
                    dense_tensor_offsets.at("output_hc_up.weight"));
        }

        constexpr std::uint64_t kPleKeyElements = 2'560u * 10'240u;
        const std::uint64_t ple_key_offset =
            fixture.data_start +
            dense_tensor_offsets.at("blk.1.ple_key.weight");
        if (ple_key_type == kGgmlTypeQ2_0) {
            std::vector<block_q2_0> ple_key_blocks(
                static_cast<std::size_t>(kPleKeyElements / 64u));
            for (block_q2_0& block : ple_key_blocks) {
                block.d = 0x1000u;
                std::fill_n(block.qs, std::size(block.qs), 0xaau);
            }
            write_all_at(descriptor, ple_key_blocks.data(),
                         ple_key_blocks.size() * sizeof(block_q2_0),
                         ple_key_offset);
        } else if (ple_key_type == kGgmlTypeQ8_0) {
            std::vector<block_q8_0> ple_key_blocks(
                static_cast<std::size_t>(kPleKeyElements / 32u));
            for (block_q8_0& block : ple_key_blocks) {
                block.d = 0x1000u;
                std::fill_n(block.qs, std::size(block.qs), 1);
            }
            write_all_at(descriptor, ple_key_blocks.data(),
                         ple_key_blocks.size() * sizeof(block_q8_0),
                         ple_key_offset);
        } else if (ple_key_type == kGgmlTypeBF16) {
            std::vector<std::uint16_t> ple_key_weights(
                static_cast<std::size_t>(kPleKeyElements),
                strata::kernels::bf16_from_f32(0.001f));
            write_all_at(descriptor, ple_key_weights.data(),
                         ple_key_weights.size() * sizeof(std::uint16_t),
                         ple_key_offset);
        } else {
            throw std::runtime_error(
                "unsupported synthetic FullMode PLE key type");
        }
        std::vector<std::uint16_t> ple_value(
            static_cast<std::size_t>(2'560u) * 2'560u,
            strata::kernels::bf16_from_f32(0.0001f));
        write_all_at(descriptor, ple_value.data(),
                     ple_value.size() * sizeof(std::uint16_t),
                     fixture.data_start + dense_tensor_offsets.at(
                                               "blk.1.ple_value.weight"));
        const std::vector<float> ple_gamma(
            10'240u, 1.0f);
        for (const char* tensor_name : {
                 "blk.1.ple_norm_key.weight",
                 "blk.1.ple_norm_query.weight",
                 "blk.1.ple_norm_conv.weight"}) {
            write_all_at(descriptor, ple_gamma.data(),
                         ple_gamma.size() * sizeof(float),
                         fixture.data_start +
                             dense_tensor_offsets.at(tensor_name));
        }
        std::vector<std::uint16_t> ple_conv(
            4u * 10'240u,
            strata::kernels::f16_from_f32(0.125f));
        write_all_at(descriptor, ple_conv.data(),
                     ple_conv.size() * sizeof(std::uint16_t),
                     fixture.data_start + dense_tensor_offsets.at(
                                               "blk.1.ple_conv1d.weight"));

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
    for (const std::string& name : {
             std::string("blk.1.ple_key.weight"),
             std::string("blk.1.ple_value.weight"),
             std::string("blk.1.ple_norm_key.weight"),
             std::string("blk.1.ple_norm_query.weight"),
             std::string("blk.1.ple_norm_conv.weight"),
             std::string("blk.1.ple_conv1d.weight")}) {
        if (const strata::TensorInfo* tensor = file.find(name)) {
            shard.tensors.push_back(
                {tensor->name, file.data_start() + tensor->offset,
                 strata::tensor_payload_bytes(*tensor), 0});
        }
    }
    if (const strata::TensorInfo* tensor = file.find("output.weight")) {
        shard.tensors.push_back(
            {tensor->name, file.data_start() + tensor->offset,
             strata::tensor_payload_bytes(*tensor), 0});
    }
    for (const std::string& name : {
             std::string("output_hc_norm.weight"),
             std::string("output_hc_down.weight"),
             std::string("output_hc_up.weight")}) {
        if (const strata::TensorInfo* tensor = file.find(name)) {
            shard.tensors.push_back(
                {tensor->name, file.data_start() + tensor->offset,
                 strata::tensor_payload_bytes(*tensor), 0});
        }
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

void verify_unpinned_tiered_q2_is_not_dereferenced(
    Context& context, ExpertStore& expert_store) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t n_experts = 256u;
    constexpr std::uint32_t layer = 0u;
    constexpr std::size_t expert_bytes = 640u * 40u * 18u;
    constexpr std::size_t gathered_bytes =
        rows * 10u * 2560u * sizeof(std::uint16_t);
    constexpr std::size_t hidden_bytes =
        rows * 10u * 640u * sizeof(std::uint16_t);
    constexpr std::size_t tile_capacity = 11u;
    std::string error;

    BufferRef addresses = context.buffers().allocate(
        "unpinned Q2_0 address table", n_experts * sizeof(TestMoeExpertAddress),
        error);
    BufferRef address_snapshot = context.buffers().allocate(
        "unpinned Q2_0 address snapshot",
        n_experts * sizeof(TestMoeExpertAddress), error);
    BufferRef weights = context.buffers().allocate(
        "unpinned Q2_0 expert storage", 2u * expert_bytes, error);
    BufferRef gathered =
        context.buffers().allocate("unpinned Q2_0 gathered rows",
                                   gathered_bytes, error);
    BufferRef plan = context.buffers().allocate(
        "unpinned Q2_0 plan", kPrefillMoePlanStrideBytes, error);
    BufferRef tiles = context.buffers().allocate(
        "unpinned Q2_0 tiles",
        tile_capacity * sizeof(TestPrefillMoeTile), error);
    BufferRef hidden = context.buffers().allocate(
        "unpinned Q2_0 hidden output", hidden_bytes + 64u, error);
    BufferRef status =
        context.buffers().allocate("unpinned Q2_0 status", 64u, error);
    require(addresses && address_snapshot && weights && gathered && plan &&
                tiles && hidden && status,
            "allocate buffers for an unpinned tiered Q2_0 address: " + error);

    std::memset(addresses.contents, 0, addresses.length);
    std::memset(weights.contents, 0, weights.length);
    auto* const weight_bytes = static_cast<std::uint8_t*>(weights.contents);
    for (std::size_t role = 0u; role < 2u; ++role) {
        for (std::size_t block = 0u; block < 640u * 40u; ++block) {
            const std::size_t offset = role * expert_bytes + block * 18u;
            weight_bytes[offset] = 0x00u;
            weight_bytes[offset + 1u] = 0x3cu;  // F16 scale 1.0
            std::fill(weight_bytes + offset + 2u,
                      weight_bytes + offset + 18u, 0xaau);
        }
    }
    auto* const gathered_values =
        static_cast<std::uint16_t*>(gathered.contents);
    std::fill(gathered_values, gathered_values + gathered_bytes /
                                          sizeof(std::uint16_t),
              0x3c00u);
    std::memset(plan.contents, 0, plan.length);
    std::memset(tiles.contents, 0, tiles.length);
    std::memset(hidden.contents, 0xa5, hidden.length);
    std::memset(status.contents, 0xa5, status.length);
    auto* const address_table =
        static_cast<TestMoeExpertAddress*>(addresses.contents);
    address_table[0].role[0] = weights.gpu_address;
    address_table[0].role[1] = weights.gpu_address + expert_bytes;
    auto* const pf_plan = static_cast<TestPrefillMoePlan*>(plan.contents);
    pf_plan->n_groups = 1u;
    pf_plan->n_entries = rows * 10u;
    pf_plan->n_tiles = 1u;
    pf_plan->grp_e[0] = 0;
    pf_plan->grp_start[0] = 0;
    pf_plan->grp_start[1] = static_cast<std::int32_t>(rows * 10u);
    static_cast<TestPrefillMoeTile*>(tiles.contents)[0] = {0u, 0u};

    auto command =
        context.make_command_buffer("unpinned tiered Q2_0 protection", error);
    require(command != nullptr,
            "create unpinned tiered Q2_0 command buffer: " + error);
    auto leases = expert_store.prepare_command_buffer(*command, error);
    require(leases != nullptr,
            "prepare empty tiered Q2_0 lease set: " + error);
    require(context.buffers().mark_tiered_slot_arena(weights, error),
            "mark Q2_0 safety-probe storage as a tiered slot arena: " + error);
    const std::array<PrefillMoeWeightStorageView, 1u> weight_views{{
        {context.buffers().read_only(weights)},
    }};
    const PrefillMoeExpertTileBuffers buffers{
        context.buffers().read_only(addresses),
        address_snapshot,
        weight_views,
        leases,
        context.buffers().read_only(gathered),
        context.buffers().read_only(plan),
        context.buffers().read_only(tiles),
        hidden,
        status,
    };
    require(!leases->pins(layer, address_table[0].role[0], expert_bytes),
            "the empty lease set must not pin the active Q2_0 weight range");
    require(encode_prefill_moe_gu_q2_0(
                context, *command, rows, n_experts, layer, buffers, error),
            "encode the unpinned-address safety probe: " + error);
    require(command->commit(error) && command->wait_until_completed(error),
            "complete the unpinned-address safety probe: " + error);
    const auto* const status_word =
        static_cast<const std::uint32_t*>(status.contents);
    const auto* const hidden_bytes_view =
        static_cast<const std::uint8_t*>(hidden.contents);
    require(status_word[0] == 4u &&
                std::all_of(hidden_bytes_view,
                            hidden_bytes_view + hidden_bytes,
                            [](std::uint8_t byte) { return byte == 0u; }) &&
                std::all_of(hidden_bytes_view + hidden_bytes,
                            hidden_bytes_view + hidden.length,
                            [](std::uint8_t byte) { return byte == 0xa5u; }),
            "an unpinned active tiered address must set status and zero its tile without reading it");
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
    require(slot_arena && slot_arena.is_tiered_slot_arena(),
            "TieredStorage must owner-mark its reusable slot arena");
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

    auto lease_probe_command =
        context.make_command_buffer("tiered lease command affinity", error);
    require(lease_probe_command != nullptr,
            "create command-buffer lease affinity probe: " + error);
    auto lease_probe =
        expert_store->prepare_command_buffer(*lease_probe_command, error);
    require(lease_probe != nullptr,
            "prepare command-buffer lease affinity probe: " + error);
    bool matches_prepared_command = false;
    require(Encoder::encode(
                context, *lease_probe_command, "tiered lease affinity match",
                [&](Encoder& encoder, std::string& encode_error) {
                    matches_prepared_command =
                        lease_probe->prepared_for(encoder);
                    if (!matches_prepared_command) {
                        encode_error =
                            "lease set did not match its prepared command buffer";
                    }
                    return matches_prepared_command;
                },
                error) &&
                matches_prepared_command,
            "accept the lease set on its prepared command buffer: " + error);

    auto unrelated_command =
        context.make_command_buffer("unrelated lease command", error);
    require(unrelated_command != nullptr,
            "create unrelated lease command buffer: " + error);
    bool rejects_unrelated_command = false;
    require(Encoder::encode(
                context, *unrelated_command, "tiered lease affinity mismatch",
                [&](Encoder& encoder, std::string&) {
                    rejects_unrelated_command =
                        !lease_probe->prepared_for(encoder);
                    return rejects_unrelated_command;
                },
                error) &&
                rejects_unrelated_command,
            "reject the lease set on another command buffer: " + error);
    require(!lease_probe->pins(0u, 0x1000u, sizeof(std::uint16_t)),
            "an empty command-buffer lease set does not pin an expert address");
    lease_probe->retire_layer(0u);
    require(!lease_probe->pins(0u, 0x1000u, sizeof(std::uint16_t)),
            "retired layer leases cannot validate a GPU address");
    unrelated_command->discard();
    lease_probe_command->discard();

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
    const auto pinned_role = [&](std::uint32_t expert, std::size_t role) {
        const std::uint64_t address =
            addresses[static_cast<std::size_t>(expert) * 3u + role];
        return resolve_leases->pins(
            0u, address,
            layout.experts[expert].roles[role].byte_count);
    };
    require(pinned_role(3u, 0u) && pinned_role(3u, 1u) &&
                pinned_role(8u, 0u) && pinned_role(8u, 1u) &&
                !pinned_role(7u, 0u),
            "ExpertLeaseSet pins both active tiered experts and excludes resident addresses");
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
    verify_unpinned_tiered_q2_is_not_dereferenced(context, *expert_store);
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
                               bool use_ple_entry = false,
                               std::uint32_t ple_key_type = kGgmlTypeQ2_0) {
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
    const Fixture fixture =
        write_fullmode_handoff_fixture(directory, ple_key_type);
    WindowSources sources = inspect_fixture(fixture);
    Geometry full_geometry = owner_test_geometry(fixture, sources);
    full_geometry.family = "qwen4exp";
    full_geometry.model = "synthetic-fullmode-handoff";
    full_geometry.n_qsa_layers = 1u;
    full_geometry.n_kv_head = 2u;
    full_geometry.head_dim = 256u;
    full_geometry.n_embd = N_EMBD;
    full_geometry.vocab = kVocabulary;
    full_geometry.types = {
        "bf16", "f16", "f32", "iq4_nl", "q2_0", "q3_k", "q8_0"};
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
    WindowSamplerSettings sampler_settings{};
    sampler_settings.temperature = 0.7f;
    sampler_settings.top_p = 0.85f;
    sampler_settings.min_p = 0.03f;
    sampler_settings.penalty_repeat = 1.2f;
    sampler_settings.penalty_freq = 0.25f;
    sampler_settings.penalty_present = 0.35f;
    sampler_settings.top_k = 23;
    sampler_settings.min_keep = 4;
    sampler_settings.greedy = 0;
    sampler_settings.hist_off = 17u;
    sampler_settings.hist_len = 29u;
    const std::array<WindowSamplerSettings, kRows> samplers{
        sampler_settings};
    const WindowScratchLayout& layout = scratch->layout();
    const std::array<std::uint32_t, 2u> committed_tokens{5u, 0u};
    WindowT1PreparedInput prepared_input;
    if (use_ple_entry) {
        const BufferRef ple_input_storage =
            require_scratch_view(context, *scratch, layout.ple_input,
                                 "uninitialized PLE input sentinel");
        std::memset(ple_input_storage.contents, 0x5bu,
                    ple_input_storage.length);
        std::vector<std::uint8_t> ple_input_before(
            static_cast<const std::uint8_t*>(ple_input_storage.contents),
            static_cast<const std::uint8_t*>(ple_input_storage.contents) +
                ple_input_storage.length);
        WindowT1PreparedInput invalid_token_input;
        require(!prepare_window_t1_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 0u, 1u, runtime_geometry.vocab, samplers[0],
                    0x11223344u, 0x55667788u, true, invalid_token_input,
                    error) &&
                    error.find("outside the model vocabulary") !=
                        std::string::npos &&
                    !invalid_token_input.params &&
                    !invalid_token_input.ple_input &&
                    !params_ring->view(0u) &&
                    std::equal(ple_input_before.begin(),
                               ple_input_before.end(),
                               static_cast<const std::uint8_t*>(
                                   ple_input_storage.contents)),
                "invalid T=1 token must not mutate prepared input storage");
        WindowT1PreparedInput invalid_position_input;
        require(!prepare_window_t1_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 0u,
                    static_cast<std::uint32_t>(std::numeric_limits<
                        std::int32_t>::max()),
                    tokens[0], samplers[0], 0x11223344u, 0x55667788u,
                    true, invalid_position_input, error) &&
                    error.find("positions exceed") != std::string::npos &&
                    !invalid_position_input.params &&
                    !invalid_position_input.ple_input &&
                    !params_ring->view(0u) &&
                    std::equal(ple_input_before.begin(),
                               ple_input_before.end(),
                               static_cast<const std::uint8_t*>(
                                   ple_input_storage.contents)),
                "invalid T=1 position must not mutate prepared input storage");
        WindowT1PreparedInput rejected_input;
        require(!prepare_window_t1_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 0u, 1u, tokens[0], samplers[0],
                    0x11223344u, 0x55667788u, true, rejected_input, error) &&
                    error.find("must be initialized") != std::string::npos &&
                    !rejected_input.params && !rejected_input.ple_input &&
                    !params_ring->view(0u) &&
                    std::equal(ple_input_before.begin(),
                               ple_input_before.end(),
                               static_cast<const std::uint8_t*>(
                                   ple_input_storage.contents)),
                "uninitialized PLE preparation must preserve scratch and the parameter slot");
        require(session->initialize_ple_token_ids(
                    committed_tokens, runtime_geometry, error),
                "initialize PLE token history for T=1 input preparation: " +
                    error);
    }
    require(prepare_window_t1_input(
                *session, runtime_geometry, *full_mode, *params_ring,
                *scratch, 0u, 1u, tokens[0], samplers[0], 0x11223344u,
                0x55667788u, use_ple_entry, prepared_input, error),
            "prepare T=1 Window inputs: " + error);
    const BufferRef params = prepared_input.params;
    require(static_cast<bool>(params),
            "prepared T=1 input publishes its parameter block");
    const auto* prepared_params =
        static_cast<const strata::metal::WindowParams*>(params.contents);
    require(prepared_params->T == 1u && prepared_params->nc == 1u &&
                prepared_params->pos0 == 1u &&
                prepared_params->n_keep == 0u &&
                prepared_params->tokens[0] == tokens[0] &&
                prepared_params->step[0].pos == 1 &&
                prepared_params->step[0].n_kv == 2 &&
                prepared_params->step[0].n_bid == 0 &&
                prepared_params->step[0].width == 2 &&
                prepared_params->samp[0].temperature ==
                    sampler_settings.temperature &&
                prepared_params->samp[0].top_p == sampler_settings.top_p &&
                prepared_params->samp[0].min_p == sampler_settings.min_p &&
                prepared_params->samp[0].penalty_repeat ==
                    sampler_settings.penalty_repeat &&
                prepared_params->samp[0].penalty_freq ==
                    sampler_settings.penalty_freq &&
                prepared_params->samp[0].penalty_present ==
                    sampler_settings.penalty_present &&
                prepared_params->samp[0].top_k == sampler_settings.top_k &&
                prepared_params->samp[0].min_keep == sampler_settings.min_keep &&
                prepared_params->samp[0].greedy == sampler_settings.greedy &&
                prepared_params->samp[0].counter == 1u &&
                prepared_params->samp[0].hist_off ==
                    sampler_settings.hist_off &&
                prepared_params->samp[0].hist_len ==
                    sampler_settings.hist_len &&
                prepared_params->seed_lo == 0x11223344u &&
                prepared_params->seed_hi == 0x55667788u,
            "prepared T=1 parameters must match the token, position and seed");
    if (!use_ple_entry) {
        require(!prepared_input.ple_input &&
                    session->ple_token_id_suffix().empty(),
                "no-PLE T=1 preparation must not need or mutate token history");
    }

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
        8u * (static_cast<std::size_t>(GDN_VALUE_DIM) / 32u) *
            sizeof(Q81));
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
                               3u * static_cast<std::size_t>(
                                        GDN_CONV_CHANNELS) *
                                   sizeof(float));
    const BufferRef gdn_active_rows = allocate_zeroed_buffer(
        context, "full-mode handoff GDN active rows", sizeof(std::uint32_t));
    const BufferRef router_sink =
        allocate_zeroed_buffer(context, "full-mode handoff router sink",
                               static_cast<std::size_t>(kExperts) * 3u *
                                   sizeof(float));
    const ReadOnlyBufferRef active_rows =
        context.buffers().read_only(params).slice(
            offsetof(strata::metal::WindowParams, nc),
            sizeof(std::uint32_t));
    require(static_cast<bool>(active_rows),
            "form the legacy WindowParams.nc live-row input");
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
        "full-mode handoff shared gate sink", 3u * sizeof(float));
    auto shared_gate_projection_sink =
        allocate("full-mode handoff shared gate projection sink",
                 3u * 640u * sizeof(float));
    auto shared_up_sink =
        allocate("full-mode handoff shared up sink",
                 3u * 640u * sizeof(float));
    auto shared_down_sink =
        allocate("full-mode handoff shared down sink",
                 3u * N_EMBD * sizeof(float));

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

    if (!use_ple_entry) {
        const auto exercise_generic_fullmode_producer =
            [&](std::uint32_t rows, bool caller_owned,
                bool induce_late_failure, bool consume_pending,
                bool induce_consumer_failure = false) {
                std::vector<std::uint32_t> generic_tokens(rows, 0u);
                std::vector<WindowSamplerSettings> generic_samplers(
                    rows, sampler_settings);
                const BufferRef generic_params = params_ring->build_slot(
                    1u, 1u, generic_tokens, generic_samplers, 0u,
                    0x11223344u, 0x55667788u, error);
                require(static_cast<bool>(generic_params),
                        "build generic FullMode producer parameters: " +
                            error);
                strata::metal::WindowParams generic_parameter_values{};
                std::memcpy(&generic_parameter_values,
                            generic_params.contents,
                            sizeof(generic_parameter_values));
                std::uint32_t expected_compiled_columns = 1u;
                while (expected_compiled_columns < rows) {
                    expected_compiled_columns <<= 1u;
                }
                require(generic_parameter_values.T == rows &&
                            generic_parameter_values.nc ==
                                expected_compiled_columns,
                        "generic FullMode parameters separate active rows from compiled columns");
                std::memcpy(gdn_active_rows.contents, &rows, sizeof(rows));

                WindowMoeLayer0FullBuffers generic_buffers = full_buffers;
                generic_buffers.producer.router.layer0.params =
                    generic_params;
                const ReadOnlyBufferRef generic_params_read_only =
                    context.buffers().read_only(generic_params);
                generic_buffers.ffn.shared.active_rows =
                    generic_params_read_only.slice(
                        offsetof(strata::metal::WindowParams, T),
                        sizeof(std::uint32_t));
                require(static_cast<bool>(
                            generic_buffers.ffn.shared.active_rows),
                        "bind shared expert active rows to WindowParams.T");
                const GRReadBuffers generic_attention{
                    residual, next_attention_block_decoy,
                    next_attention_inject_decoy, next_attention_norm,
                    gr_xn, gr_rs, next_attention_down,
                    next_attention_inject_weights, gr_lo, inject,
                    next_attention_up, attention_entry_mixed};
                GRReadBuffers generic_previous_gr{};
                generic_previous_gr.residual = residual;
                generic_previous_gr.inject_out = inject2;
                const WindowMoeAttentionEntryBuffers generic_attention_entry{
                    generic_buffers.ffn, generic_previous_gr,
                    generic_attention};
                const WindowMoePendingConsumer generic_pending_consumer{
                    generic_attention_entry};
                auto* const generic_residual =
                    static_cast<float*>(residual.contents);
                for (std::uint32_t row = 0u; row < rows; ++row) {
                    for (std::size_t index = 0u;
                         index < 4u * N_EMBD; ++index) {
                        generic_residual[
                            static_cast<std::size_t>(row) * 4u * N_EMBD +
                            index] =
                            static_cast<float>((row + 1u) *
                                               (index % 13u + 1u)) *
                            0.001f;
                    }
                }
                const std::size_t generic_residual_bytes =
                    static_cast<std::size_t>(rows) * 4u * N_EMBD *
                    sizeof(float);
                std::vector<std::uint8_t> generic_residual_before(
                    static_cast<const std::uint8_t*>(residual.contents),
                    static_cast<const std::uint8_t*>(residual.contents) +
                        generic_residual_bytes);
                std::vector<std::uint8_t> params_before(
                    static_cast<const std::uint8_t*>(generic_params.contents),
                    static_cast<const std::uint8_t*>(generic_params.contents) +
                        sizeof(strata::metal::WindowParams));
                std::vector<std::int32_t> resident(kExperts);
                std::vector<float> usage(kExperts);
                const ReadOnlyBufferRef resident_view =
                    full_mode->resident_table();
                const BufferRef usage_view = full_mode->usage_table();
                std::memcpy(resident.data(), resident_view.contents(),
                            resident.size() * sizeof(std::int32_t));
                std::memcpy(usage.data(), usage_view.contents,
                            usage.size() * sizeof(float));

                std::vector<BufferRef> mutation_views;
                const auto append_gr_mutations =
                    [&](const GRReadBuffers& gr) {
                        mutation_views.insert(
                            mutation_views.end(),
                            {gr.residual, gr.norm, gr.normalized, gr.rs,
                             gr.hc_down, gr.hc_inject, gr.lo, gr.inject_out,
                             gr.hc_up, gr.mixed});
                    };
                append_gr_mutations(generic_buffers.producer.router.layer0
                                        .attention);
                const WindowGDNBlockBuffers& generic_gdn =
                    generic_buffers.producer.router.layer0.gdn;
                mutation_views.insert(
                    mutation_views.end(),
                    {generic_gdn.history, generic_gdn.state, generic_gdn.xq,
                     generic_gdn.qkv, generic_gdn.h, generic_gdn.gate,
                     generic_gdn.beta, generic_gdn.z, generic_gdn.y,
                     generic_gdn.yq, generic_gdn.block_out,
                     generic_gdn.sink, generic_gdn.active_rows});
                append_gr_mutations(
                    generic_buffers.producer.router.layer0.ffn);
                mutation_views.insert(
                    mutation_views.end(),
                    {generic_buffers.producer.router.router_logits,
                     generic_buffers.producer.router.route_ids,
                     generic_buffers.producer.router.route_weights,
                     generic_buffers.producer.router.router_matvec_sink,
                     generic_buffers.producer.usage,
                     generic_buffers.producer.plans,
                     generic_buffers.ffn.routed.gate,
                     generic_buffers.ffn.routed.up,
                     generic_buffers.ffn.routed.hidden,
                     generic_buffers.ffn.routed.down_activations,
                     generic_buffers.ffn.routed.parts,
                     generic_buffers.ffn.shared.gate,
                     generic_buffers.ffn.shared.up,
                     generic_buffers.ffn.shared.hidden,
                     generic_buffers.ffn.shared.down_activations,
                     generic_buffers.ffn.shared.shared_output,
                     generic_buffers.ffn.shared.shared_gate_logits,
                     generic_buffers.ffn.shared.output,
                     generic_buffers.ffn.shared.gate_sink,
                     generic_buffers.ffn.shared.up_sink,
                     generic_buffers.ffn.shared.down_sink,
                     generic_buffers.ffn.shared.shared_gate_sink});
                if (consume_pending) {
                    append_gr_mutations(generic_attention_entry.attention);
                }

                struct BufferSnapshot {
                    BufferRef view;
                    std::vector<std::uint8_t> bytes;
                };
                std::vector<BufferSnapshot> before;
                before.reserve(mutation_views.size());
                if (induce_late_failure || induce_consumer_failure) {
                    for (BufferRef view : mutation_views) {
                        require(view && view.contents != nullptr,
                                "snapshot generic producer mutation view");
                        const auto* const begin =
                            static_cast<const std::uint8_t*>(view.contents);
                        before.push_back(
                            {view, std::vector<std::uint8_t>(
                                       begin, begin + view.length)});
                    }
                } else {
                    std::memset(generic_buffers.producer.router.router_logits
                                    .contents,
                                0xa5,
                                generic_buffers.producer.router.router_logits
                                    .length);
                    std::memset(generic_buffers.producer.router.route_ids
                                    .contents,
                                0xa5,
                                generic_buffers.producer.router.route_ids
                                    .length);
                    std::memset(generic_buffers.producer.router.route_weights
                                    .contents,
                                0xa5,
                                generic_buffers.producer.router.route_weights
                                    .length);
                    std::memset(generic_buffers.producer.plans.contents, 0xa5,
                                generic_buffers.producer.plans.length);
                    std::memset(generic_buffers.ffn.routed.parts.contents,
                                0xa5,
                                generic_buffers.ffn.routed.parts.length);
                    std::memset(generic_buffers.ffn.shared.output.contents,
                                0xa5,
                                generic_buffers.ffn.shared.output.length);
                    std::memset(pending_output.contents, 0xa5,
                                pending_output.length);
                    if (consume_pending) {
                        std::memset(attention_entry_mixed.contents, 0xa5,
                                    attention_entry_mixed.length);
                    }
                }

                WindowMoeLayer0FullBuffers dispatched_buffers =
                    generic_buffers;
                if (induce_late_failure) {
                    dispatched_buffers.ffn.shared.down_sink =
                        dispatched_buffers.ffn.shared.down_sink.slice(0u, 0u);
                    require(static_cast<bool>(
                                dispatched_buffers.ffn.shared.down_sink),
                            "form an owned empty late-failure sink view");
                }
                WindowMoePendingConsumer dispatched_consumer =
                    generic_pending_consumer;
                if (induce_consumer_failure) {
                    auto& invalid_attention =
                        std::get<WindowMoeAttentionEntryBuffers>(
                            dispatched_consumer);
                    invalid_attention.attention.mixed =
                        invalid_attention.attention.mixed.slice(0u, 0u);
                    require(static_cast<bool>(
                                invalid_attention.attention.mixed),
                            "form an owned empty late-consumer output view");
                }
                if (rows == 2u && !caller_owned && consume_pending &&
                    !induce_consumer_failure) {
                    WindowMoePendingConsumer aliased_consumer =
                        generic_pending_consumer;
                    auto& aliased_attention =
                        std::get<WindowMoeAttentionEntryBuffers>(
                            aliased_consumer);
                    aliased_attention.attention.inject_out =
                        full_mode->usage_table();
                    const std::vector<std::uint8_t> usage_before_alias(
                        static_cast<const std::uint8_t*>(usage_view.contents),
                        static_cast<const std::uint8_t*>(usage_view.contents) +
                            usage_view.length);
                    const WindowMoeLayer0FullModeStepBuffers aliased_step{
                        dispatched_buffers, aliased_consumer};
                    auto alias_command = context.make_command_buffer(
                        "Reject generic consumer output alias to MoE usage",
                        error);
                    require(alias_command != nullptr,
                            "create consumer/producers alias check: " +
                                error);
                    const bool alias_encoded = Encoder::encode(
                        context, *alias_command,
                        "generic consumer/producers alias check",
                        [&](Encoder& encoder,
                            std::string& encode_error) {
                            return encode_window_moe_layer0_full_mode_step(
                                context, *full_mode, encoder, kVocabulary,
                                kExperts, MoeExpertFormat::q2_0,
                                MoeDownFormat::q2_0, false, aliased_step,
                                encode_error);
                        },
                        error);
                    require(
                        !alias_encoded &&
                            alias_command->unsafe_native_handle() == nullptr &&
                            error.find("overlaps producer storage") !=
                                std::string::npos &&
                            std::equal(
                                usage_before_alias.begin(),
                                usage_before_alias.end(),
                                static_cast<const std::uint8_t*>(
                                    usage_view.contents)),
                        "generic step rejects consumer writes into producer usage before any dispatch");
                }
                auto command = context.make_command_buffer(
                    "generic FullMode layer-0 producer T=" +
                        std::to_string(rows),
                    error);
                require(command != nullptr,
                        "create generic FullMode producer command: " + error);
                require(session->track_use(*command, error),
                        "track GDN state through generic FullMode producer: " +
                            error);
                const WindowMoeLayer0FullModeStepBuffers step_buffers{
                    dispatched_buffers, dispatched_consumer};
                const bool encoded = consume_pending
                    ? (caller_owned
                           ? Encoder::encode(
                                 context, *command,
                                 "caller-owned generic FullMode producer/consumer",
                                 [&](Encoder& encoder,
                                     std::string& encode_error) {
                                     return encode_window_moe_layer0_full_mode_step(
                                         context, *full_mode, encoder,
                                         kVocabulary, kExperts,
                                         MoeExpertFormat::q2_0,
                                         MoeDownFormat::q2_0, false,
                                         step_buffers, encode_error);
                                 },
                                 error)
                           : encode_window_moe_layer0_full_mode_step(
                                 context, *full_mode, *command, kVocabulary,
                                 kExperts, MoeExpertFormat::q2_0,
                                 MoeDownFormat::q2_0, false, step_buffers,
                                 error))
                    : (caller_owned
                           ? Encoder::encode(
                                 context, *command,
                                 "caller-owned generic FullMode producer",
                                 [&](Encoder& encoder,
                                     std::string& encode_error) {
                                     return encode_window_moe_layer0_full_mode(
                                         context, *full_mode, encoder,
                                         kVocabulary, kExperts,
                                         MoeExpertFormat::q2_0,
                                         MoeDownFormat::q2_0,
                                         dispatched_buffers, encode_error);
                                 },
                                 error)
                           : encode_window_moe_layer0_full_mode(
                                 context, *full_mode, *command, kVocabulary,
                                 kExperts, MoeExpertFormat::q2_0,
                                 MoeDownFormat::q2_0, dispatched_buffers,
                                 error));
                if (induce_late_failure || induce_consumer_failure) {
                    bool unchanged = before.size() == mutation_views.size();
                    for (const BufferSnapshot& snapshot : before) {
                        const auto* const current =
                            static_cast<const std::uint8_t*>(
                                snapshot.view.contents);
                        unchanged &=
                            std::equal(snapshot.bytes.begin(),
                                       snapshot.bytes.end(), current);
                    }
                    require(!encoded &&
                                command->unsafe_native_handle() == nullptr &&
                                error.find("undersized") != std::string::npos &&
                                unchanged,
                            (induce_consumer_failure
                                 ? "late generic consumer validation failure"
                                 : "late shared-MoE validation failure") +
                                std::string(
                                    " discards the FullMode transaction and preserves all views"));
                    std::memcpy(gdn_active_rows.contents, &kRows,
                                sizeof(kRows));
                    return;
                }
                require(encoded,
                        "encode generic FullMode layer-0 producer: " + error);
                require(command->commit(error),
                        "commit generic FullMode layer-0 producer: " + error);
                require(command->wait_until_completed(error),
                        "wait for generic FullMode layer-0 producer: " +
                            error);

                std::uint32_t columns = 1u;
                while (columns < rows) columns <<= 1u;
                const std::span<const float> mixed_rows(
                    static_cast<const float*>(mixed.contents),
                    static_cast<std::size_t>(rows) * N_EMBD);
                std::vector<float> compiled_mixed(
                    static_cast<std::size_t>(columns) * N_EMBD, 0.0f);
                std::copy(mixed_rows.begin(), mixed_rows.end(),
                          compiled_mixed.begin());
                for (std::uint32_t row = rows; row < columns; ++row) {
                    std::copy_n(mixed_rows.begin(), N_EMBD,
                                compiled_mixed.begin() +
                                    static_cast<std::size_t>(row) * N_EMBD);
                }
                const auto* const router_weight_data =
                    static_cast<const std::uint16_t*>(
                        router_tensor.contents());
                std::vector<float> expected_logits;
                require(twin::bf16_matvec_reference(
                            std::span<const std::uint16_t>(
                                router_weight_data,
                                static_cast<std::size_t>(kExperts) * N_EMBD),
                            compiled_mixed, N_EMBD, kExperts, columns, rows,
                            256u, expected_logits, error),
                        "compute generic FullMode router oracle: " + error);
                const std::size_t active_logits_count =
                    static_cast<std::size_t>(rows) * kExperts;
                const auto* const actual_logits =
                    static_cast<const float*>(
                        generic_buffers.producer.router.router_logits
                            .contents);
                require(std::memcmp(actual_logits, expected_logits.data(),
                                    active_logits_count * sizeof(float)) == 0,
                        "generic FullMode router logits match the BF16 scalar oracle");

                std::vector<std::int32_t> expected_ids;
                std::vector<twin::tf> expected_route_weights;
                require(twin::moe_route_reference(
                            std::span<const float>(expected_logits.data(),
                                                   active_logits_count),
                            rows, kExperts, false, expected_ids,
                            expected_route_weights, error),
                        "compute generic FullMode route oracle: " + error);
                const std::size_t active_routes =
                    static_cast<std::size_t>(rows) * 10u;
                const std::size_t active_route_id_bytes =
                    active_routes * sizeof(std::int32_t);
                const auto* const actual_ids =
                    static_cast<const std::int32_t*>(
                        generic_buffers.producer.router.route_ids.contents);
                require(std::memcmp(actual_ids, expected_ids.data(),
                                    active_route_id_bytes) == 0,
                        "generic FullMode route IDs match the stable top-10 oracle");
                const auto* const actual_weights =
                    static_cast<const float*>(
                        generic_buffers.producer.router.route_weights
                            .contents);
                bool route_weights_match = true;
                for (std::size_t route = 0u; route < active_routes; ++route) {
                    route_weights_match &=
                        std::isfinite(actual_weights[route]) &&
                        std::abs(actual_weights[route] -
                                 expected_route_weights[route].value) <=
                            1.0e-6f;
                }
                require(route_weights_match,
                        "generic FullMode route weights match the reciprocal oracle");

                TieredMoePlan expected_plan{};
                std::vector<float> expected_usage;
                require(twin::moe_plan_reference(
                            expected_ids, rows, kExperts, false, 0u,
                            resident, usage, expected_plan, expected_usage,
                            error),
                        "compute generic FullMode plan oracle: " + error);
                require(std::memcmp(
                            generic_buffers.producer.plans.contents,
                            &expected_plan, sizeof(expected_plan)) == 0,
                        "generic FullMode plan consumes every active route row");
                require(std::memcmp(usage_view.contents, usage.data(),
                                    usage.size() * sizeof(float)) == 0,
                        "full-mode speculative planning leaves usage unchanged");

                for (std::uint32_t row = 0u; row < rows; ++row) {
                    for (std::uint32_t rank = 0u; rank < 10u; ++rank) {
                        require(expected_ids[static_cast<std::size_t>(row) *
                                                 10u +
                                             rank] ==
                                    static_cast<std::int32_t>(rank),
                                "synthetic FullMode route selects seeded experts 0..9");
                    }
                }
                constexpr std::size_t kGateUpBlocksPerExpert =
                    640u * (N_EMBD / 64u);
                constexpr std::size_t kDownBlocksPerExpert =
                    N_EMBD * (640u / 64u);
                std::vector<block_q2_0> oracle_gate_weights(
                    10u * kGateUpBlocksPerExpert);
                std::vector<block_q2_0> oracle_up_weights(
                    10u * kGateUpBlocksPerExpert);
                std::vector<block_q2_0> oracle_down_weights(
                    10u * kDownBlocksPerExpert);
                const auto fill_q2_0_expert_weights =
                    [](std::vector<block_q2_0>& weights) {
                        for (block_q2_0& block : weights) {
                            block.d = 0x1000u;
                            std::fill_n(block.qs, std::size(block.qs),
                                        0xaau);
                        }
                    };
                fill_q2_0_expert_weights(oracle_gate_weights);
                fill_q2_0_expert_weights(oracle_up_weights);
                fill_q2_0_expert_weights(oracle_down_weights);
                std::vector<std::int32_t> oracle_expert_slots(kExperts, -1);
                for (std::uint32_t expert = 0u; expert < 10u; ++expert) {
                    oracle_expert_slots[expert] =
                        static_cast<std::int32_t>(expert);
                }
                const std::span<const Q81> input_activations(
                    static_cast<const Q81*>(xq.contents),
                    static_cast<std::size_t>(rows) * (N_EMBD / 32u));
                std::vector<twin::tf> expected_routed_gate;
                std::vector<twin::tf> expected_routed_up;
                require(twin::moe_gu_q2_0_reference(
                            oracle_gate_weights, oracle_up_weights,
                            input_activations, oracle_expert_slots,
                            expected_plan, rows, expected_routed_gate,
                            expected_routed_up, error),
                        "compute generic FullMode routed gate/up scalar oracle: " +
                            error);
                const std::size_t routed_hidden_count =
                    static_cast<std::size_t>(rows) * 10u * 640u;
                std::vector<float> routed_gate_values(routed_hidden_count);
                std::vector<float> routed_up_values(routed_hidden_count);
                std::transform(expected_routed_gate.begin(),
                               expected_routed_gate.end(),
                               routed_gate_values.begin(),
                               [](const twin::tf& value) {
                                   return value.value;
                               });
                std::transform(expected_routed_up.begin(),
                               expected_routed_up.end(),
                               routed_up_values.begin(),
                               [](const twin::tf& value) {
                                   return value.value;
                               });
                std::vector<twin::tf> expected_routed_hidden;
                require(twin::moe_swiglu_reference(
                            routed_gate_values, routed_up_values,
                            expected_routed_hidden, error),
                        "compute generic FullMode routed SwiGLU oracle: " +
                            error);
                std::vector<float> routed_hidden_values(routed_hidden_count);
                std::transform(expected_routed_hidden.begin(),
                               expected_routed_hidden.end(),
                               routed_hidden_values.begin(),
                               [](const twin::tf& value) {
                                   return value.value;
                               });
                std::vector<Q81> routed_hidden_q8_1;
                require(twin::quantize_q8_1_reference(
                            routed_hidden_values, 640u, rows * 10u,
                            routed_hidden_q8_1, error),
                        "quantize generic FullMode routed SwiGLU oracle: " +
                            error);
                std::vector<twin::tf> expected_routed_parts;
                require(twin::moe_down_q2_0_reference(
                            oracle_down_weights, routed_hidden_q8_1,
                            oracle_expert_slots, expected_plan, rows,
                            expected_routed_parts, error),
                        "compute generic FullMode routed down scalar oracle: " +
                            error);

                const std::size_t gate_weight_count =
                    640u * (N_EMBD / 32u);
                const std::size_t down_weight_count =
                    N_EMBD * (640u / 32u);
                std::vector<block_q8_0> oracle_shared_gate_weights(
                    gate_weight_count);
                std::vector<block_q8_0> oracle_shared_up_weights(
                    gate_weight_count);
                std::vector<block_q8_0> oracle_shared_down_weights(
                    down_weight_count);
                const auto fill_q8_0_shared_weights =
                    [](std::vector<block_q8_0>& weights) {
                        for (block_q8_0& block : weights) {
                            block.d = 0x1c00u;
                            std::fill_n(block.qs, std::size(block.qs),
                                        1);
                        }
                    };
                fill_q8_0_shared_weights(oracle_shared_gate_weights);
                fill_q8_0_shared_weights(oracle_shared_up_weights);
                fill_q8_0_shared_weights(oracle_shared_down_weights);
                const std::size_t compiled_input_count =
                    static_cast<std::size_t>(columns) * (N_EMBD / 32u);
                std::vector<Q81> compiled_input_activations(
                    compiled_input_count);
                std::copy(input_activations.begin(),
                          input_activations.end(),
                          compiled_input_activations.begin());
                const std::size_t compiled_mixed_count =
                    static_cast<std::size_t>(columns) * N_EMBD;
                std::vector<float> compiled_mixed_for_oracle(
                    compiled_mixed_count, 0.0f);
                std::copy(mixed_rows.begin(), mixed_rows.end(),
                          compiled_mixed_for_oracle.begin());
                for (std::uint32_t row = rows; row < columns; ++row) {
                    std::copy_n(
                        input_activations.begin(), N_EMBD / 32u,
                        compiled_input_activations.begin() +
                            static_cast<std::size_t>(row) *
                                (N_EMBD / 32u));
                    std::copy_n(
                        mixed_rows.begin(), N_EMBD,
                        compiled_mixed_for_oracle.begin() +
                            static_cast<std::size_t>(row) * N_EMBD);
                }
                std::vector<float> expected_shared_gate_full;
                std::vector<float> expected_shared_up_full;
                require(twin::q8_0_matvec_reference(
                            oracle_shared_gate_weights,
                            compiled_input_activations, N_EMBD, 640u,
                            columns, rows, expected_shared_gate_full, error) &&
                            twin::q8_0_matvec_reference(
                                oracle_shared_up_weights,
                                compiled_input_activations, N_EMBD, 640u,
                                columns, rows, expected_shared_up_full,
                                error),
                        "compute generic FullMode shared gate/up scalar oracle: " +
                            error);
                std::vector<twin::tf> expected_shared_hidden;
                require(twin::moe_swiglu_reference(
                            std::span<const float>(
                                expected_shared_gate_full.data(),
                                static_cast<std::size_t>(rows) * 640u),
                            std::span<const float>(
                                expected_shared_up_full.data(),
                                static_cast<std::size_t>(rows) * 640u),
                            expected_shared_hidden, error),
                        "compute generic FullMode shared SwiGLU oracle: " +
                            error);
                std::vector<float> shared_hidden_padded(
                    static_cast<std::size_t>(columns) * 640u, 0.0f);
                for (std::size_t row = 0u; row < rows; ++row) {
                    shared_hidden_padded[row * 640u] =
                        expected_shared_hidden[row * 640u].value;
                    for (std::size_t column = 1u; column < 640u; ++column) {
                        shared_hidden_padded[row * 640u + column] =
                            expected_shared_hidden[row * 640u + column].value;
                    }
                }
                for (std::uint32_t row = rows; row < columns; ++row) {
                    std::copy_n(shared_hidden_padded.begin(), 640u,
                                shared_hidden_padded.begin() +
                                    static_cast<std::size_t>(row) * 640u);
                }
                std::vector<Q81> shared_hidden_q8_1;
                require(twin::quantize_q8_1_reference(
                            shared_hidden_padded, 640u, columns,
                            shared_hidden_q8_1, error),
                        "quantize generic FullMode shared SwiGLU oracle: " +
                            error);
                std::vector<float> expected_shared_output_full;
                require(twin::q8_0_matvec_reference(
                            oracle_shared_down_weights, shared_hidden_q8_1,
                            640u, N_EMBD, columns, rows,
                            expected_shared_output_full, error),
                        "compute generic FullMode shared down scalar oracle: " +
                            error);
                std::vector<std::uint16_t> oracle_shared_gate_input_weights(
                    N_EMBD, strata::kernels::bf16_from_f32(0.01f));
                std::vector<float> expected_shared_gate_logits_full;
                require(twin::bf16_matvec_reference(
                            oracle_shared_gate_input_weights,
                            compiled_mixed_for_oracle, N_EMBD, 1u, columns,
                            rows, 256u, expected_shared_gate_logits_full,
                            error),
                        "compute generic FullMode shared gate scalar oracle: " +
                            error);

                std::vector<float> expected_routed_parts_values(
                    expected_routed_parts.size());
                std::transform(expected_routed_parts.begin(),
                               expected_routed_parts.end(),
                               expected_routed_parts_values.begin(),
                               [](const twin::tf& value) {
                                   return value.value;
                               });
                std::vector<float> route_weight_values(active_routes);
                std::transform(expected_route_weights.begin(),
                               expected_route_weights.end(),
                               route_weight_values.begin(),
                               [](const twin::tf& value) {
                                   return value.value;
                               });
                std::vector<float> expected_shared_output_values(
                    expected_shared_output_full.begin(),
                    expected_shared_output_full.begin() +
                        static_cast<std::size_t>(rows) * N_EMBD);
                std::vector<float> expected_shared_gate_logits_values(
                    expected_shared_gate_logits_full.begin(),
                    expected_shared_gate_logits_full.begin() + rows);
                std::vector<twin::tf> expected_pending_values;
                require(twin::moe_combine_reference(
                            expected_routed_parts_values, route_weight_values,
                            expected_shared_output_values,
                            expected_shared_gate_logits_values, rows,
                            expected_pending_values, error),
                        "compute generic FullMode routed/shared combine oracle: " +
                            error);
                const auto compare_expected = [&](BufferRef actual_view,
                                                  std::size_t count,
                                                  std::span<const twin::tf> expected,
                                                  const std::string& label) {
                    if (count != expected.size() ||
                        actual_view.length < count * sizeof(float)) {
                        require(false, label + " oracle extent matches output");
                    }
                    const std::span<const float> actual(
                        static_cast<const float*>(actual_view.contents),
                        count);
                    bool matches = true;
                    for (std::size_t index = 0u; index < count; ++index) {
                        const double tolerance =
                            64.0 * std::numeric_limits<float>::epsilon() *
                            std::max(1.0, expected[index].magnitude);
                        matches &= std::isfinite(actual[index]) &&
                                   std::abs(
                                       static_cast<double>(actual[index]) -
                                       expected[index].value) <= tolerance;
                    }
                    require(matches,
                            label +
                                " matches the independent scalar oracle within scaled float tolerance");
                };
                compare_expected(
                    generic_buffers.ffn.routed.parts,
                    active_routes * N_EMBD,
                    expected_routed_parts,
                    "generic FullMode routed down outputs");
                std::vector<twin::tf> expected_shared_outputs(
                    expected_shared_output_values.size());
                std::transform(
                    expected_shared_output_values.begin(),
                    expected_shared_output_values.end(),
                    expected_shared_outputs.begin(),
                    [](float value) { return twin::tf{value}; });
                compare_expected(
                    generic_buffers.ffn.shared.shared_output,
                    static_cast<std::size_t>(rows) * N_EMBD,
                    expected_shared_outputs,
                    "generic FullMode shared expert outputs");
                std::vector<twin::tf> expected_pending_logits(
                    expected_shared_gate_logits_values.size());
                std::transform(
                    expected_shared_gate_logits_values.begin(),
                    expected_shared_gate_logits_values.end(),
                    expected_pending_logits.begin(),
                    [](float value) { return twin::tf{value}; });
                compare_expected(
                    generic_buffers.ffn.shared.shared_gate_logits, rows,
                    expected_pending_logits,
                    "generic FullMode shared gate logits");
                compare_expected(
                    generic_buffers.ffn.shared.output,
                    static_cast<std::size_t>(rows) * N_EMBD,
                    expected_pending_values,
                    "generic FullMode combined pending output");

                const auto check_active_output =
                    [&](BufferRef view, std::size_t active_bytes,
                        const std::string& label) {
                        const auto* const bytes =
                            static_cast<const std::uint8_t*>(view.contents);
                        const bool tail_unchanged =
                            active_bytes <= view.length &&
                            std::all_of(bytes + active_bytes,
                                        bytes + view.length,
                                        [](std::uint8_t byte) {
                                            return byte == 0xa5u;
                                        });
                        require(tail_unchanged,
                                label +
                                    " writes only its active multi-row extent");
                    };
                check_active_output(
                    generic_buffers.producer.router.router_logits,
                    active_logits_count * sizeof(float),
                    "generic FullMode router logits");
                check_active_output(
                    generic_buffers.producer.router.route_ids,
                    active_route_id_bytes,
                    "generic FullMode route IDs");
                check_active_output(
                    generic_buffers.producer.router.route_weights,
                    active_routes * sizeof(float),
                    "generic FullMode route weights");
                check_active_output(
                    generic_buffers.producer.plans,
                    sizeof(TieredMoePlan),
                    "generic FullMode plan");
                check_active_output(
                    generic_buffers.ffn.routed.parts,
                    active_routes * N_EMBD * sizeof(float),
                    "generic FullMode routed parts");
                check_active_output(
                    generic_buffers.ffn.shared.output,
                    static_cast<std::size_t>(rows) * N_EMBD *
                        sizeof(float),
                    "generic FullMode pending output");
                const auto* const pending_values =
                    static_cast<const float*>(
                        generic_buffers.ffn.shared.output.contents);
                for (std::uint32_t row = 0u; row < rows; ++row) {
                    const auto begin =
                        pending_values + static_cast<std::size_t>(row) *
                                             N_EMBD;
                    require(std::all_of(begin, begin + N_EMBD,
                                        [](float value) {
                                            return std::isfinite(value);
                                        }) &&
                                std::any_of(begin, begin + N_EMBD,
                                            [](float value) {
                                                return std::abs(value) >
                                                       1.0e-7f;
                                            }),
                            "generic FullMode consumer output is finite and nonzero for every row");
                }
                if (consume_pending) {
                    const BufferRef oracle_pending = allocate(
                        "generic FullMode expected pending output",
                        expected_pending_values.size() * sizeof(float));
                    std::vector<float> expected_pending_floats(
                        expected_pending_values.size());
                    std::transform(
                        expected_pending_values.begin(),
                        expected_pending_values.end(),
                        expected_pending_floats.begin(),
                        [](const twin::tf& value) { return value.value; });
                    std::memcpy(oracle_pending.contents,
                                expected_pending_floats.data(),
                                expected_pending_floats.size() *
                                    sizeof(float));
                    const BufferRef oracle_residual = allocate(
                        "generic FullMode expected consumer residual",
                        generic_residual_bytes);
                    std::memcpy(oracle_residual.contents,
                                generic_residual_before.data(),
                                generic_residual_bytes);
                    const BufferRef oracle_inject_out = allocate(
                        "generic FullMode expected consumer injection",
                        static_cast<std::size_t>(rows) * 4u *
                            sizeof(float));
                    const BufferRef oracle_mixed_out = allocate(
                        "generic FullMode expected consumer mixed output",
                        static_cast<std::size_t>(rows) * N_EMBD *
                            sizeof(float));
                    GRReadBuffers direct_attention = generic_attention;
                    direct_attention.residual = oracle_residual;
                    direct_attention.block_out = oracle_pending;
                    direct_attention.inject_prev = inject2;
                    direct_attention.inject_out = oracle_inject_out;
                    direct_attention.mixed = oracle_mixed_out;
                    auto direct_attention_command =
                        context.make_command_buffer(
                            "generic FullMode scalar-result GR consumer",
                            error);
                    require(direct_attention_command != nullptr &&
                                encode_gr_read(
                                    context, *direct_attention_command, rows,
                                    true, direct_attention, error) &&
                                direct_attention_command->commit(error) &&
                                direct_attention_command->wait_until_completed(
                                    error),
                            "encode and complete direct GR consumer over the scalar MoE output: " +
                                error);
                    require(std::memcmp(
                                attention_entry_mixed.contents,
                                oracle_mixed_out.contents,
                                static_cast<std::size_t>(rows) * N_EMBD *
                                    sizeof(float)) == 0,
                            "generic pending consumer matches direct apply=true GR output for the independently computed MoE result");
                    require(std::memcmp(
                                residual.contents, oracle_residual.contents,
                                generic_residual_bytes) == 0,
                            "generic pending consumer residual matches the direct GR reference");
                    const auto* const consumed_values =
                        static_cast<const float*>(
                            attention_entry_mixed.contents);
                    for (std::uint32_t row = 0u; row < rows; ++row) {
                        const auto begin =
                            consumed_values +
                            static_cast<std::size_t>(row) * N_EMBD;
                        require(std::all_of(begin, begin + N_EMBD,
                                            [](float value) {
                                                return std::isfinite(value);
                                            }) &&
                                    std::any_of(begin, begin + N_EMBD,
                                                [](float value) {
                                                    return std::abs(value) >
                                                           1.0e-7f;
                                                }),
                                "generic pending selector computes nonzero next-attention rows");
                    }
                    check_active_output(
                        attention_entry_mixed,
                        static_cast<std::size_t>(rows) * N_EMBD *
                            sizeof(float),
                        "generic pending consumer mixed output");
                }
                require(std::equal(
                            params_before.begin(), params_before.end(),
                            static_cast<const std::uint8_t*>(
                                generic_params.contents)),
                        "generic FullMode producer preserves its parameters");
                std::memcpy(gdn_active_rows.contents, &kRows,
                            sizeof(kRows));
            };
        exercise_generic_fullmode_producer(1u, false, false, false);
        exercise_generic_fullmode_producer(2u, false, false, true);
        exercise_generic_fullmode_producer(3u, false, false, true);
        exercise_generic_fullmode_producer(8u, true, false, true);
        exercise_generic_fullmode_producer(2u, true, true, false);
        exercise_generic_fullmode_producer(3u, false, true, false);
        exercise_generic_fullmode_producer(2u, false, false, true, true);
        exercise_generic_fullmode_producer(3u, true, false, true, true);
    }

    PLEBlockBuffers ple_block{};
    WindowMoePLEEntryBuffers ple_entry{};
    std::vector<std::uint8_t> ple_history_initial;
    if (use_ple_entry) {
        WindowPleRows ple_rows;
        require(session->build_ple_rows(tokens, runtime_geometry, ple_rows,
                                        error),
                "build session-owned layer-1 PLE row IDs: " + error);
        strata::kernels::PleTableReader* ple_reader =
            full_mode->ple_reader();
        require(ple_reader != nullptr && ple_reader->is_open(),
                "resolve FullMode's mapped PLE reader");
        const BufferRef ple_input = prepared_input.ple_input;
        require(ple_input.length ==
                    static_cast<std::size_t>(kRows) * N_EMBD * sizeof(float),
                "prepared PLE input exposes exactly the active T=1 rows");
        std::vector<float> expected_ple_input(
            static_cast<std::size_t>(kRows) * N_EMBD);
        require(ple_reader->gather_batch(ple_rows.all_rows().data(), kRows,
                                         expected_ple_input.data(), error),
                "gather direct reference rows for prepared PLE input: " +
                    error);
        require(std::memcmp(ple_input.contents, expected_ple_input.data(),
                            expected_ple_input.size() * sizeof(float)) == 0,
                "prepared PLE input matches the session-owned committed suffix rows");
        const auto prepared_suffix = session->ple_token_id_suffix();
        require(std::equal(prepared_suffix.begin(), prepared_suffix.end(),
                           committed_tokens.begin(), committed_tokens.end()),
                "preparing PLE rows must not advance committed token history");
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
        FullModeTensorView ple_key_weights;
        require(full_mode->dense_tensor_view("blk.1.ple_key.weight",
                                             ple_key_weights, error),
                "resolve FullMode PLE native key tensor: " + error);
        require(ple_key_weights.ggml_type == ple_key_type,
                "FullMode PLE key tensor preserves its inspected native type");
        FullModeTensorView ple_value_weights;
        require(full_mode->dense_tensor_view("blk.1.ple_value.weight",
                                             ple_value_weights, error),
                "resolve FullMode PLE BF16 value tensor: " + error);
        FullModeTensorView ple_key_gamma;
        require(full_mode->dense_tensor_view("blk.1.ple_norm_key.weight",
                                             ple_key_gamma, error),
                "resolve FullMode PLE key gamma: " + error);
        FullModeTensorView ple_query_gamma;
        require(full_mode->dense_tensor_view("blk.1.ple_norm_query.weight",
                                             ple_query_gamma, error),
                "resolve FullMode PLE query gamma: " + error);
        FullModeTensorView ple_conv_gamma;
        require(full_mode->dense_tensor_view("blk.1.ple_norm_conv.weight",
                                             ple_conv_gamma, error),
                "resolve FullMode PLE convolution gamma: " + error);
        FullModeTensorView ple_conv1d;
        require(full_mode->dense_tensor_view("blk.1.ple_conv1d.weight",
                                             ple_conv1d, error),
                "resolve FullMode PLE convolution weights: " + error);
        const BufferRef ple_key_q8_1 = allocate(
            "FullMode handoff PLE Q8_1 key activations",
            static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81));
        const BufferRef ple_sink =
            require_scratch_view(context, *scratch, layout.matvec_sink,
                                 "PLE mat-vec sink");
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
                "mapped PLE rows prepare finite nonzero table values");
        ple_block = {
            ple_input,
            ple_key_q8_1,
            ple_key_weights.buffer,
            ple_value_weights.buffer,
            ple_sink,
            context.buffers().read_only(gdn_active_rows),
            {ple_key, residual, ple_query, ple_norm, ple_value, ple_gate,
             ple_gated, ple_history, ple_conv1d.buffer,
             ple_key_gamma.buffer, ple_query_gamma.buffer,
             ple_conv_gamma.buffer},
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
        PLEBlockBuffers step_ple_block = ple_block;
        step_ple_block.key_weights = {};
        step_ple_block.value_weights = {};
        step_ple_block.post.conv1d = {};
        step_ple_block.post.key_gamma = {};
        step_ple_block.post.query_gamma = {};
        step_ple_block.post.conv_gamma = {};
        const WindowMoePLEEntryBuffers step_ple_entry{
            full_buffers.ffn, ffn, step_ple_block};
        const MatVecWeightFormat resolved_key_format =
            fullmode_ple_key_format(ple_key_type);
        const MatVecWeightFormat decoy_key_format =
            resolved_key_format == MatVecWeightFormat::bf16
                ? MatVecWeightFormat::q2_0
                : MatVecWeightFormat::bf16;
        // The FullMode step must replace this format decoy from the mapped
        // model metadata for every supported native key type.
        pending_consumer = WindowMoePendingPLEConsumer{
            step_ple_entry, decoy_key_format, 1u,
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
        "Reject undersized multi-row Window pending consumer views", error);
    require(row_count_command != nullptr,
            "create multi-row Window consumer command: " + error);
    WindowMoePendingConsumer undersized_consumer = pending_consumer;
    std::string expected_short_view_diagnostic = "undersized";
    if (use_ple_entry) {
        auto& ple_consumer =
            std::get<WindowMoePendingPLEConsumer>(undersized_consumer);
        ple_consumer.compiled_columns = 2u;
        BufferRef active_rows_t2 = allocate(
            "multi-row PLE active columns", sizeof(std::uint32_t));
        require(static_cast<bool>(active_rows_t2),
                "allocate multi-row PLE active-column metadata");
        *static_cast<std::uint32_t*>(active_rows_t2.contents) = 2u;
        ple_consumer.buffers.ple.active_rows =
            context.buffers().read_only(active_rows_t2);
        require(static_cast<bool>(ple_consumer.buffers.ple.active_rows),
                "bind matching T=2 PLE active-column metadata");
        ple_consumer.buffers.ple.input_f32 =
            ple_consumer.buffers.ple.input_f32.slice(
                0u, static_cast<std::size_t>(N_EMBD) * sizeof(float));
        expected_short_view_diagnostic =
            ple_consumer.key_format == MatVecWeightFormat::bf16
                ? "activations buffer is undersized"
                : "Q8_1 input buffer is undersized";
    } else {
        auto& attention_consumer =
            std::get<WindowMoeAttentionEntryBuffers>(undersized_consumer);
        const std::size_t one_row_bytes =
            static_cast<std::size_t>(4u) * N_EMBD * sizeof(float);
        const BufferRef one_row_residual =
            attention_consumer.attention.residual.slice(0u, one_row_bytes);
        attention_consumer.attention.residual = one_row_residual;
        attention_consumer.previous_gr_read.residual = one_row_residual;
    }
    const bool row_count_encoded = Encoder::encode(
        context, *row_count_command,
        "undersized multi-row Window pending consumer views",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_window_moe_pending_consumer(
                context, encoder, 2u, use_ple_entry, undersized_consumer,
                encode_error);
        },
        error);
    require(!row_count_encoded &&
                row_count_command->unsafe_native_handle() == nullptr &&
                error.find(expected_short_view_diagnostic) !=
                    std::string::npos,
            "multi-row pending consumer must reject views sized only for its T=1 producer");

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
    std::memcpy(residual.contents, initial_residual.data(),
                initial_residual.size());
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
                    fullmode_ple_key_format(ple_key_type), 1u,
                    PLEHistoryUpdate::defer, reference_entry, error),
                "encode direct GR-post plus native-format PLE-entry reference: " +
                    error);
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

        WindowMoeLayer0FullBuffers generic_ple_producer = full_buffers;
        generic_ple_producer.producer.router.layer0.params = params;
        const ReadOnlyBufferRef generic_ple_params =
            context.buffers().read_only(params);
        generic_ple_producer.ffn.shared.active_rows =
            generic_ple_params.slice(
                offsetof(strata::metal::WindowParams, T),
                sizeof(std::uint32_t));
        require(static_cast<bool>(
                    generic_ple_producer.ffn.shared.active_rows),
                "bind generic PLE producer active rows to WindowParams.T");
        const WindowMoeLayer0FullModeStepBuffers generic_ple_step{
            generic_ple_producer, pending_consumer};
        WindowMoeLayer0FullModeStepBuffers aliased_ple_input_step =
            generic_ple_step;
        std::get<WindowMoePendingPLEConsumer>(
            aliased_ple_input_step.consumer)
            .buffers.ple.input_f32 = pending_output;
        auto aliased_ple_input_command = context.make_command_buffer(
            "Reject PLE table input alias to pending MoE output", error);
        require(aliased_ple_input_command != nullptr,
                "create PLE-input alias rejection command: " + error);
        const bool aliased_ple_input_encoded =
            encode_window_moe_layer0_full_mode_step(
                context, *full_mode, *aliased_ple_input_command, kVocabulary,
                kExperts, MoeExpertFormat::q2_0, MoeDownFormat::q2_0, true,
                aliased_ple_input_step, error);
        require(!aliased_ple_input_encoded &&
                    aliased_ple_input_command->unsafe_native_handle() ==
                        nullptr &&
                    error.find("outside declared handoffs") !=
                        std::string::npos &&
                    std::equal(
                        pending_snapshot.begin(), pending_snapshot.end(),
                        static_cast<const std::uint8_t*>(
                            pending_output.contents)),
                "generic step rejects PLE table input alias to pending output before any dispatch");
        std::memcpy(residual.contents, initial_residual.data(),
                    initial_residual.size());
        auto generic_ple_command = context.make_command_buffer(
            "Generic FullMode producer and PLE consumer", error);
        require(generic_ple_command != nullptr,
                "create generic FullMode-to-PLE command: " + error);
        require(session->track_use(*generic_ple_command, error),
                "track GDN state through generic FullMode-to-PLE step: " +
                    error);
        require(encode_window_moe_layer0_full_mode_step(
                    context, *full_mode, *generic_ple_command, kVocabulary,
                    kExperts, MoeExpertFormat::q2_0, MoeDownFormat::q2_0,
                    true, generic_ple_step, error),
                "encode generic FullMode producer and resolved PLE consumer: " +
                    error);
        require(generic_ple_command->commit(error),
                "commit generic FullMode-to-PLE step: " + error);
        require(generic_ple_command->wait_until_completed(error),
                "wait for generic FullMode-to-PLE step: " + error);
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const auto* actual =
                static_cast<const std::uint8_t*>(outputs[index].contents);
            require(std::equal(actual_ple_outputs[index].begin(),
                               actual_ple_outputs[index].end(), actual),
                    "generic FullMode-to-PLE step matches T=1 token-step output " +
                        std::to_string(index));
        }
        require(std::equal(
                    actual_residual.begin(), actual_residual.end(),
                    static_cast<const std::uint8_t*>(residual.contents)),
                "generic FullMode-to-PLE step matches T=1 residual");
        require(std::equal(
                    pending_snapshot.begin(), pending_snapshot.end(),
                    static_cast<const std::uint8_t*>(pending_output.contents)),
                "generic PLE composition preserves the producer pending output");
        require(std::equal(
                    ple_history_initial.begin(), ple_history_initial.end(),
                    static_cast<const std::uint8_t*>(
                        ple_block.post.history.contents)),
                "generic deferred PLE composition preserves committed history");
        WindowMoeLayer0FullModeStepBuffers advancing_ple_step =
            generic_ple_step;
        std::get<WindowMoePendingPLEConsumer>(
            advancing_ple_step.consumer)
            .history_update = PLEHistoryUpdate::advance_rows;
        auto advancing_ple_command = context.make_command_buffer(
            "Reject speculative PLE history advancement", error);
        require(advancing_ple_command != nullptr,
                "create speculative PLE-history rejection command: " + error);
        const bool advancing_ple_encoded =
            encode_window_moe_layer0_full_mode_step(
                context, *full_mode, *advancing_ple_command, kVocabulary,
                kExperts, MoeExpertFormat::q2_0, MoeDownFormat::q2_0, true,
                advancing_ple_step, error);
        require(!advancing_ple_encoded &&
                    advancing_ple_command->unsafe_native_handle() == nullptr &&
                    error.find("must defer committed history advancement") !=
                        std::string::npos &&
                    std::equal(
                        ple_history_initial.begin(),
                        ple_history_initial.end(),
                        static_cast<const std::uint8_t*>(
                            ple_block.post.history.contents)),
                "speculative FullMode PLE step rejects history advancement before encoding");

        const std::vector<std::uint32_t> generic_ple_tokens(3u, tokens[0]);
        std::vector<WindowSamplerSettings> generic_ple_samplers(
            generic_ple_tokens.size(), sampler_settings);
        const BufferRef speculative_ple_scratch = require_scratch_view(
            context, *scratch, layout.ple_input,
            "T=3 speculative PLE preparation scratch");
        std::memset(speculative_ple_scratch.contents, 0x6du,
                    speculative_ple_scratch.length);
        const std::vector<std::uint8_t> speculative_ple_before_rejection(
            static_cast<const std::uint8_t*>(
                speculative_ple_scratch.contents),
            static_cast<const std::uint8_t*>(speculative_ple_scratch.contents) +
                speculative_ple_scratch.length);
        std::vector<std::uint32_t> invalid_later_ple_tokens =
            generic_ple_tokens;
        invalid_later_ple_tokens[2] = runtime_geometry.vocab;
        WindowSpeculativePreparedInput rejected_speculative_input;
        require(!prepare_window_speculative_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 1u, 1u, invalid_later_ple_tokens,
                    generic_ple_samplers, 0x11223344u, 0x55667788u, true,
                    rejected_speculative_input, error) &&
                    error.find("outside the model vocabulary") !=
                        std::string::npos &&
                    !rejected_speculative_input.params &&
                    !rejected_speculative_input.ple_input &&
                    !params_ring->view(1u) &&
                    std::equal(
                        speculative_ple_before_rejection.begin(),
                        speculative_ple_before_rejection.end(),
                        static_cast<const std::uint8_t*>(
                            speculative_ple_scratch.contents)),
                "invalid final speculative token must preserve PLE scratch and the parameter slot");
        auto* const aliased_tokens =
            static_cast<std::uint32_t*>(speculative_ple_scratch.contents);
        auto* const aliased_samplers =
            reinterpret_cast<WindowSamplerSettings*>(
                static_cast<std::uint8_t*>(
                    speculative_ple_scratch.contents) +
                256u);
        std::copy(generic_ple_tokens.begin(), generic_ple_tokens.end(),
                  aliased_tokens);
        std::copy(generic_ple_samplers.begin(),
                  generic_ple_samplers.end(), aliased_samplers);
        WindowSpeculativePreparedInput generic_prepared_ple_input;
        require(prepare_window_speculative_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 1u, 1u,
                    std::span<const std::uint32_t>(aliased_tokens, 3u),
                    std::span<const WindowSamplerSettings>(aliased_samplers,
                                                           3u),
                    0x11223344u, 0x55667788u, true,
                    generic_prepared_ple_input, error),
                "prepare T=3 speculative Window parameters from spans backed by PLE scratch: " +
                    error);
        const BufferRef generic_ple_t3_params =
            generic_prepared_ple_input.params;
        const BufferRef generic_ple_input =
            generic_prepared_ple_input.ple_input;
        require(generic_ple_t3_params && generic_ple_input &&
                    generic_ple_input.length ==
                        3u * N_EMBD * sizeof(float),
                "T=3 speculative preparation publishes active PLE rows");
        require(session->ple_token_id_suffix().size() ==
                    committed_tokens.size() &&
                    std::equal(committed_tokens.begin(),
                               committed_tokens.end(),
                               session->ple_token_id_suffix().begin()),
                "T=3 speculative input preparation leaves committed PLE token history unchanged");
        strata::metal::WindowParams generic_ple_parameter_values{};
        std::memcpy(&generic_ple_parameter_values,
                    generic_ple_t3_params.contents,
                    sizeof(generic_ple_parameter_values));
        require(generic_ple_parameter_values.T == 3u &&
                    generic_ple_parameter_values.nc == 4u &&
                    generic_ple_parameter_values.pos0 == 1u &&
                    generic_ple_parameter_values.n_keep == 0u &&
                    std::equal(generic_ple_tokens.begin(),
                               generic_ple_tokens.end(),
                               generic_ple_parameter_values.tokens) &&
                    generic_ple_parameter_values.samp[0].temperature ==
                        sampler_settings.temperature &&
                    generic_ple_parameter_values.samp[0].top_k ==
                        sampler_settings.top_k &&
                    generic_ple_parameter_values.samp[0].hist_len ==
                        sampler_settings.hist_len,
                "generic PLE preparation snapshots token and sampler spans before staging into aliased scratch");
        WindowPleRows generic_ple_rows;
        require(session->build_ple_rows(generic_ple_tokens,
                                         runtime_geometry, generic_ple_rows,
                                         error),
                "build T=3 generic PLE row IDs: " + error);
        strata::kernels::PleTableReader* const generic_ple_reader =
            full_mode->ple_reader();
        require(generic_ple_reader != nullptr &&
                    generic_ple_reader->is_open(),
                "resolve open FullMode PLE reader for T=3 rows");
        std::vector<float> generic_ple_direct_rows(
            3u * static_cast<std::size_t>(N_EMBD));
        require(generic_ple_reader->gather_batch(
                    generic_ple_rows.all_rows().data(), 3u,
                    generic_ple_direct_rows.data(), error) &&
                    std::memcmp(generic_ple_input.contents,
                                generic_ple_direct_rows.data(),
                                generic_ple_input.length) == 0,
                "T=3 speculative PLE preparation matches a direct table gather");
        // This compact fixture has a one-token vocabulary. Keep its valid
        // staged allocation, then use distinct synthetic row values to expose
        // row broadcast and row ordering mistakes in the T=3 composition.
        auto* const generic_ple_input_values =
            static_cast<float*>(generic_ple_input.contents);
        for (std::uint32_t row = 0u; row < 3u; ++row) {
            for (std::size_t index = 0u; index < N_EMBD; ++index) {
                generic_ple_input_values[
                    static_cast<std::size_t>(row) * N_EMBD + index] =
                    static_cast<float>((row + 1u) * (index % 29u + 1u)) *
                    0.0002f;
            }
        }
        for (std::uint32_t row = 1u; row < 3u; ++row) {
            require(std::memcmp(
                        generic_ple_input_values,
                        generic_ple_input_values +
                            static_cast<std::size_t>(row) * N_EMBD,
                        N_EMBD * sizeof(float)) != 0,
                    "T=3 PLE table inputs are distinct for every token row");
        }

        WindowMoeLayer0FullBuffers generic_ple_multi_producer = full_buffers;
        generic_ple_multi_producer.producer.router.layer0.params =
            generic_ple_t3_params;
        const ReadOnlyBufferRef generic_ple_t3_params_read_only =
            context.buffers().read_only(generic_ple_t3_params);
        generic_ple_multi_producer.ffn.shared.active_rows =
            generic_ple_t3_params_read_only.slice(
                offsetof(strata::metal::WindowParams, T),
                sizeof(std::uint32_t));
        require(static_cast<bool>(
                    generic_ple_multi_producer.ffn.shared.active_rows),
                "bind T=3 generic FullMode active rows");
        constexpr std::uint32_t generic_ple_rows_count = 3u;
        std::memcpy(gdn_active_rows.contents, &generic_ple_rows_count,
                    sizeof(generic_ple_rows_count));
        const std::size_t generic_ple_embedding_bytes =
            generic_ple_rows_count * N_EMBD * sizeof(float);
        const std::size_t generic_ple_stream_bytes =
            generic_ple_rows_count * HC_DIM * sizeof(float);
        const BufferRef generic_ple_embedding = allocate(
            "T=3 speculative FullMode embedding",
            generic_ple_embedding_bytes);
        const BufferRef direct_generic_ple_embedding = allocate(
            "T=3 direct embedding reference", generic_ple_embedding_bytes);
        const BufferRef direct_generic_ple_streams = allocate(
            "T=3 direct GR broadcast reference", generic_ple_stream_bytes);
        const ReadOnlyBufferRef generic_ple_token_rows =
            generic_ple_t3_params_read_only.slice(
                offsetof(strata::metal::WindowParams, tokens),
                generic_ple_rows_count * sizeof(std::uint32_t));
        require(static_cast<bool>(generic_ple_token_rows),
                "bind T=3 speculative token rows");
        auto direct_input_command = context.make_command_buffer(
            "Direct T=3 embedding and GR broadcast reference", error);
        require(direct_input_command != nullptr,
                "create direct T=3 embedding reference command: " + error);
        require(Encoder::encode(
                    context, *direct_input_command,
                    "direct T=3 embedding and GR broadcast",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encode_embed_rows(
                                context, encoder, generic_ple_rows_count,
                                kVocabulary, EmbedRowsFormat::q3_k,
                                EmbedRowsReadOnlyBuffers{
                                    token_embedding_view.buffer,
                                    generic_ple_token_rows,
                                    direct_generic_ple_embedding},
                                encode_error)) {
                            return false;
                        }
                        return encode_gr_broadcast(
                            context, encoder, generic_ple_rows_count,
                            {context.buffers().read_only(
                                 direct_generic_ple_embedding),
                             direct_generic_ple_streams},
                            encode_error);
                    },
                    error),
                "encode direct T=3 embedding and GR broadcast: " + error);
        require(direct_input_command->commit(error),
                "commit direct T=3 embedding reference: " + error);
        require(direct_input_command->wait_until_completed(error),
                "wait for direct T=3 embedding reference: " + error);
        std::memset(residual.contents, 0xa5u, residual.length);
        const WindowEmbeddingBuffers generic_ple_embedding_buffers{
            generic_ple_t3_params, token_embedding_view.buffer,
            generic_ple_embedding, residual};
        auto speculative_input_command = context.make_command_buffer(
            "T=3 speculative embedding and GR broadcast", error);
        require(speculative_input_command != nullptr,
                "create T=3 speculative embedding command: " + error);
        require(encode_window_speculative_embedding(
                    context, *speculative_input_command, kVocabulary,
                    EmbedRowsFormat::q3_k, generic_ple_embedding_buffers,
                    error),
                "encode T=3 speculative embedding and GR broadcast: " +
                    error);
        require(speculative_input_command->commit(error),
                "commit T=3 speculative embedding: " + error);
        require(speculative_input_command->wait_until_completed(error),
                "wait for T=3 speculative embedding: " + error);
        require(std::memcmp(generic_ple_embedding.contents,
                            direct_generic_ple_embedding.contents,
                            generic_ple_embedding_bytes) == 0 &&
                    std::memcmp(residual.contents,
                                direct_generic_ple_streams.contents,
                                generic_ple_stream_bytes) == 0,
                "T=3 speculative embedding and stream broadcast match direct row-wise references");
        const auto* const residual_tail =
            static_cast<const std::uint8_t*>(residual.contents) +
            generic_ple_stream_bytes;
        require(std::all_of(
                    residual_tail,
                    static_cast<const std::uint8_t*>(residual.contents) +
                        residual.length,
                    [](std::uint8_t value) { return value == 0xa5u; }),
                "T=3 speculative stream broadcast preserves inactive tails");
        const std::vector<std::uint8_t> generic_ple_reference_residual_input(
            static_cast<const std::uint8_t*>(residual.contents),
            static_cast<const std::uint8_t*>(residual.contents) +
                residual.length);
        const std::vector<std::uint8_t> generic_embedding_before_rejection(
            static_cast<const std::uint8_t*>(generic_ple_embedding.contents),
            static_cast<const std::uint8_t*>(generic_ple_embedding.contents) +
                generic_ple_embedding.length);
        const std::vector<std::uint8_t> generic_streams_before_rejection(
            static_cast<const std::uint8_t*>(residual.contents),
            static_cast<const std::uint8_t*>(residual.contents) +
                residual.length);
        strata::metal::WindowParams invalid_token_values{};
        std::memcpy(&invalid_token_values, generic_ple_t3_params.contents,
                    sizeof(invalid_token_values));
        invalid_token_values.tokens[2] = kVocabulary;
        const BufferRef invalid_token_params = allocate(
            "T=3 invalid token parameter block",
            sizeof(invalid_token_values));
        std::memcpy(invalid_token_params.contents, &invalid_token_values,
                    sizeof(invalid_token_values));
        WindowEmbeddingBuffers invalid_token_input{
            invalid_token_params, token_embedding_view.buffer,
            generic_ple_embedding, residual};
        auto invalid_token_command = context.make_command_buffer(
            "Reject invalid final speculative token", error);
        require(invalid_token_command != nullptr,
                "create invalid speculative token command: " + error);
        const bool invalid_token_encoded =
            encode_window_speculative_embedding(
                context, *invalid_token_command, kVocabulary,
                EmbedRowsFormat::q3_k, invalid_token_input, error);
        require(!invalid_token_encoded &&
                    invalid_token_command->unsafe_native_handle() == nullptr &&
                    error.find("token id is outside") != std::string::npos &&
                    std::equal(
                        generic_embedding_before_rejection.begin(),
                        generic_embedding_before_rejection.end(),
                        static_cast<const std::uint8_t*>(
                            generic_ple_embedding.contents)) &&
                    std::equal(
                        generic_streams_before_rejection.begin(),
                        generic_streams_before_rejection.end(),
                        static_cast<const std::uint8_t*>(residual.contents)),
                "T=3 speculative embedding rejects an invalid later token before dispatch");
        std::memcpy(&invalid_token_values, generic_ple_t3_params.contents,
                    sizeof(invalid_token_values));
        invalid_token_values.n_keep = 1u;
        std::memcpy(invalid_token_params.contents, &invalid_token_values,
                    sizeof(invalid_token_values));
        auto non_speculative_command = context.make_command_buffer(
            "Reject committed rows in speculative input stem", error);
        require(non_speculative_command != nullptr,
                "create non-speculative input rejection command: " + error);
        const bool non_speculative_encoded =
            encode_window_speculative_embedding(
                context, *non_speculative_command, kVocabulary,
                EmbedRowsFormat::q3_k, invalid_token_input, error);
        require(!non_speculative_encoded &&
                    non_speculative_command->unsafe_native_handle() ==
                        nullptr &&
                    error.find("n_keep == 0") != std::string::npos,
                "generic input stem rejects committed rows before dispatch");

        const std::vector<std::uint32_t> t8_tokens(8u, 0u);
        const std::vector<WindowSamplerSettings> t8_samplers(
            t8_tokens.size(), sampler_settings);
        WindowSpeculativePreparedInput t8_prepared_input;
        require(prepare_window_speculative_input(
                    *session, runtime_geometry, *full_mode, *params_ring,
                    *scratch, 0u, 1u, t8_tokens, t8_samplers,
                    0x10203040u, 0x50607080u, false, t8_prepared_input,
                    error),
                "prepare T=8 speculative Window parameters: " + error);
        const BufferRef t8_params = t8_prepared_input.params;
        strata::metal::WindowParams t8_parameter_values{};
        std::memcpy(&t8_parameter_values, t8_params.contents,
                    sizeof(t8_parameter_values));
        require(t8_params && !t8_prepared_input.ple_input &&
                    t8_parameter_values.T == 8u &&
                    t8_parameter_values.nc == 8u &&
                    t8_parameter_values.pos0 == 1u &&
                    t8_parameter_values.n_keep == 0u &&
                    std::equal(t8_tokens.begin(), t8_tokens.end(),
                               t8_parameter_values.tokens),
                "T=8 speculative preparation builds parameters without touching PLE history");
        require(session->ple_token_id_suffix().size() ==
                    committed_tokens.size() &&
                    std::equal(committed_tokens.begin(),
                               committed_tokens.end(),
                               session->ple_token_id_suffix().begin()),
                "T=8 no-PLE preparation does not advance committed PLE token history");
        constexpr std::size_t t8_tail_bytes = 64u;
        const std::size_t t8_embedding_active_bytes =
            8u * N_EMBD * sizeof(float);
        const std::size_t t8_streams_active_bytes =
            8u * HC_DIM * sizeof(float);
        const BufferRef t8_embedding = allocate(
            "T=8 speculative embedding",
            t8_embedding_active_bytes + t8_tail_bytes);
        const BufferRef t8_streams = allocate(
            "T=8 speculative GR streams",
            t8_streams_active_bytes + t8_tail_bytes);
        const BufferRef t8_direct_embedding = allocate(
            "T=8 direct embedding reference", t8_embedding_active_bytes);
        const BufferRef t8_direct_streams = allocate(
            "T=8 direct GR broadcast reference", t8_streams_active_bytes);
        std::memset(t8_embedding.contents, 0xa5u, t8_embedding.length);
        std::memset(t8_streams.contents, 0xa5u, t8_streams.length);
        const ReadOnlyBufferRef t8_params_read_only =
            context.buffers().read_only(t8_params);
        const ReadOnlyBufferRef t8_token_rows = t8_params_read_only.slice(
            offsetof(strata::metal::WindowParams, tokens),
            t8_tokens.size() * sizeof(std::uint32_t));
        require(static_cast<bool>(t8_token_rows),
                "bind T=8 speculative token rows");
        auto t8_reference_command = context.make_command_buffer(
            "Direct T=8 embedding and GR broadcast reference", error);
        require(t8_reference_command != nullptr,
                "create direct T=8 embedding reference command: " + error);
        require(Encoder::encode(
                    context, *t8_reference_command,
                    "direct T=8 embedding and GR broadcast",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encode_embed_rows(
                                context, encoder, 8u, kVocabulary,
                                EmbedRowsFormat::q3_k,
                                EmbedRowsReadOnlyBuffers{
                                    token_embedding_view.buffer, t8_token_rows,
                                    t8_direct_embedding},
                                encode_error)) {
                            return false;
                        }
                        return encode_gr_broadcast(
                            context, encoder, 8u,
                            {context.buffers().read_only(t8_direct_embedding),
                             t8_direct_streams},
                            encode_error);
                    },
                    error),
                "encode direct T=8 embedding and GR broadcast: " + error);
        require(t8_reference_command->commit(error),
                "commit direct T=8 embedding reference: " + error);
        require(t8_reference_command->wait_until_completed(error),
                "wait for direct T=8 embedding reference: " + error);
        require(std::any_of(
                    static_cast<const float*>(t8_direct_embedding.contents),
                    static_cast<const float*>(t8_direct_embedding.contents) +
                        t8_embedding_active_bytes / sizeof(float),
                    [](float value) {
                        return std::isfinite(value) &&
                               std::abs(value) > 1.0e-7f;
                    }),
                "T=8 direct embedding reference contains nonzero data");
        const WindowEmbeddingBuffers t8_embedding_buffers{
            t8_params, token_embedding_view.buffer, t8_embedding,
            t8_streams};
        auto t8_embedding_command = context.make_command_buffer(
            "T=8 speculative embedding and broadcast", error);
        require(t8_embedding_command != nullptr,
                "create T=8 speculative embedding command: " + error);
        require(encode_window_speculative_embedding(
                    context, *t8_embedding_command, kVocabulary,
                    EmbedRowsFormat::q3_k, t8_embedding_buffers, error),
                "encode T=8 speculative embedding and broadcast: " + error);
        require(t8_embedding_command->commit(error),
                "commit T=8 speculative embedding: " + error);
        require(t8_embedding_command->wait_until_completed(error),
                "wait for T=8 speculative embedding: " + error);
        require(std::memcmp(t8_embedding.contents,
                            t8_direct_embedding.contents,
                            t8_embedding_active_bytes) == 0 &&
                    std::memcmp(t8_streams.contents, t8_direct_streams.contents,
                                t8_streams_active_bytes) == 0,
                "T=8 speculative embedding and broadcast match direct row-wise references");
        const auto* const t8_embedding_tail =
            static_cast<const std::uint8_t*>(t8_embedding.contents) +
            t8_embedding_active_bytes;
        const auto* const t8_stream_tail =
            static_cast<const std::uint8_t*>(t8_streams.contents) +
            t8_streams_active_bytes;
        require(std::all_of(
                    t8_embedding_tail,
                    static_cast<const std::uint8_t*>(t8_embedding.contents) +
                        t8_embedding.length,
                    [](std::uint8_t value) { return value == 0xa5u; }) &&
                    std::all_of(
                        t8_stream_tail,
                        static_cast<const std::uint8_t*>(t8_streams.contents) +
                            t8_streams.length,
                        [](std::uint8_t value) { return value == 0xa5u; }),
                "T=8 speculative embedding and broadcast preserve inactive tails");
        for (std::uint32_t row = 1u; row < 8u; ++row) {
            require(std::memcmp(
                        t8_embedding.contents,
                        static_cast<const std::uint8_t*>(
                            t8_embedding.contents) +
                            static_cast<std::size_t>(row) * N_EMBD *
                                sizeof(float),
                        N_EMBD * sizeof(float)) == 0,
                    "T=8 embedding gather fills each repeated-token row");
            require(std::memcmp(
                        t8_streams.contents,
                        static_cast<const std::uint8_t*>(
                            t8_streams.contents) +
                            static_cast<std::size_t>(row) * HC_DIM *
                                sizeof(float),
                        HC_DIM * sizeof(float)) == 0,
                    "T=8 GR broadcast initializes each row");
        }
        WindowMoePendingConsumer generic_ple_multi_consumer =
            pending_consumer;
        auto& generic_ple_payload =
            std::get<WindowMoePendingPLEConsumer>(
                generic_ple_multi_consumer);
        generic_ple_payload.buffers.ple.input_f32 = generic_ple_input;
        generic_ple_payload.buffers.ple.key_q8_1 = allocate(
            "T=3 generic PLE Q8_1 key activations",
            3u * static_cast<std::size_t>(N_EMBD / 32u) * sizeof(Q81));
        generic_ple_payload.buffers.ple.active_rows =
            generic_ple_t3_params_read_only.slice(
                offsetof(strata::metal::WindowParams, T),
                sizeof(std::uint32_t));
        const WindowMoeFullModeInputStepBuffers generic_ple_multi_step{
            pending_output, residual, generic_ple_multi_producer,
            generic_ple_multi_consumer};

        const std::array<std::size_t, 6u> generic_ple_active_bytes{
            3u * HC_DIM * sizeof(float), 3u * N_EMBD * sizeof(float),
            3u * HC_DIM * sizeof(float), 3u * HC * sizeof(float),
            3u * HC_DIM * sizeof(float), 3u * HC_DIM * sizeof(float)};
        for (BufferRef output : outputs) {
            std::memset(output.contents, 0xa5u, output.length);
        }
        std::memcpy(ple_block.post.history.contents,
                    ple_history_initial.data(), ple_history_initial.size());
        WindowMoeFullModeInputStepBuffers late_input_failure =
            generic_ple_multi_step;
        auto& late_ple_consumer =
            std::get<WindowMoePendingPLEConsumer>(
                late_input_failure.consumer);
        late_ple_consumer.buffers.ple.post.gated =
            late_ple_consumer.buffers.ple.post.gated.slice(0u, 0u);
        require(static_cast<bool>(
                    late_ple_consumer.buffers.ple.post.gated),
                "form an owned empty late PLE output view");
        const std::vector<std::uint8_t> pending_before_late_failure(
            static_cast<const std::uint8_t*>(pending_output.contents),
            static_cast<const std::uint8_t*>(pending_output.contents) +
                pending_output.length);
        const std::vector<std::uint8_t> residual_before_late_failure(
            static_cast<const std::uint8_t*>(residual.contents),
            static_cast<const std::uint8_t*>(residual.contents) +
                residual.length);
        const BufferRef usage_view = full_mode->usage_table();
        const std::vector<std::uint8_t> usage_before_late_failure(
            static_cast<const std::uint8_t*>(usage_view.contents),
            static_cast<const std::uint8_t*>(usage_view.contents) +
                usage_view.length);
        const std::vector<std::uint8_t> embedding_before_late_failure(
            static_cast<const std::uint8_t*>(generic_ple_embedding.contents),
            static_cast<const std::uint8_t*>(generic_ple_embedding.contents) +
                generic_ple_embedding.length);
        std::vector<std::vector<std::uint8_t>>
            ple_outputs_before_late_failure;
        ple_outputs_before_late_failure.reserve(outputs.size());
        for (BufferRef output : outputs) {
            ple_outputs_before_late_failure.emplace_back(
                static_cast<const std::uint8_t*>(output.contents),
                static_cast<const std::uint8_t*>(output.contents) +
                    output.length);
        }
        const std::vector<std::uint8_t> ple_history_before_late_failure(
            static_cast<const std::uint8_t*>(ple_block.post.history.contents),
            static_cast<const std::uint8_t*>(ple_block.post.history.contents) +
                ple_block.post.history.length);
        const auto late_writes_unchanged = [&]() {
            if (!std::equal(
                    pending_before_late_failure.begin(),
                    pending_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(
                        pending_output.contents)) ||
                !std::equal(
                    residual_before_late_failure.begin(),
                    residual_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(residual.contents)) ||
                !std::equal(
                    usage_before_late_failure.begin(),
                    usage_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(usage_view.contents)) ||
                !std::equal(
                    embedding_before_late_failure.begin(),
                    embedding_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(
                        generic_ple_embedding.contents)) ||
                !std::equal(
                    ple_history_before_late_failure.begin(),
                    ple_history_before_late_failure.end(),
                    static_cast<const std::uint8_t*>(
                        ple_block.post.history.contents))) {
                return false;
            }
            for (std::size_t index = 0u; index < outputs.size(); ++index) {
                const BufferRef output = outputs[index];
                if (!std::equal(
                        ple_outputs_before_late_failure[index].begin(),
                        ple_outputs_before_late_failure[index].end(),
                        static_cast<const std::uint8_t*>(output.contents))) {
                    return false;
                }
            }
            return true;
        };
        auto late_input_command = context.make_command_buffer(
            "Rollback T=3 embedding and FullMode on late PLE failure", error);
        require(late_input_command != nullptr,
                "create T=3 late FullMode input failure command: " + error);
        const bool late_input_encoded =
            encode_window_moe_full_mode_input_step(
                context, *full_mode, *late_input_command, kVocabulary,
                kExperts, MoeExpertFormat::q2_0, MoeDownFormat::q2_0, true,
                late_input_failure, error);
        require(!late_input_encoded &&
                    late_input_command->unsafe_native_handle() == nullptr &&
                    late_writes_unchanged(),
                "late T=3 PLE failure discards input, FullMode producer and consumer writes");
        auto caller_owned_late_input_command = context.make_command_buffer(
            "Rollback caller-owned T=3 FullMode input on late PLE failure",
            error);
        require(caller_owned_late_input_command != nullptr,
                "create caller-owned late FullMode input failure command: " +
                    error);
        const bool caller_owned_late_input_encoded = Encoder::encode(
            context, *caller_owned_late_input_command,
            "caller-owned T=3 FullMode input failure",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_window_moe_full_mode_input_step(
                    context, *full_mode, encoder, kVocabulary, kExperts,
                    MoeExpertFormat::q2_0, MoeDownFormat::q2_0, true,
                    late_input_failure, encode_error);
            },
            error);
        require(!caller_owned_late_input_encoded &&
                    caller_owned_late_input_command->unsafe_native_handle() ==
                        nullptr &&
                    late_writes_unchanged(),
                "caller-owned encoder body propagates late PLE failure and discards the entire input/producer/consumer segment");
        auto generic_ple_multi_command = context.make_command_buffer(
            "T=3 generic FullMode input and PLE consumer", error);
        require(generic_ple_multi_command != nullptr,
                "create T=3 generic FullMode-to-PLE command: " + error);
        require(session->track_use(*generic_ple_multi_command, error),
                "track GDN state through T=3 generic FullMode-to-PLE step: " +
                    error);
        require(encode_window_moe_full_mode_input_step(
                    context, *full_mode, *generic_ple_multi_command,
                    kVocabulary, kExperts, MoeExpertFormat::q2_0,
                    MoeDownFormat::q2_0, true, generic_ple_multi_step,
                    error),
                "encode T=3 generic FullMode input-to-PLE step: " + error);
        require(generic_ple_multi_command->commit(error),
                "commit T=3 generic FullMode-to-PLE step: " + error);
        require(generic_ple_multi_command->wait_until_completed(error),
                "wait for T=3 generic FullMode-to-PLE step: " + error);
        for (std::size_t index = 0u; index < outputs.size(); ++index) {
            BufferRef output = outputs[index];
            const std::size_t active_bytes =
                generic_ple_active_bytes[index];
            require(output.length >= active_bytes,
                    "T=3 generic PLE output covers its active extent");
            const auto* const bytes =
                static_cast<const std::uint8_t*>(output.contents);
            require(std::all_of(bytes + active_bytes,
                                bytes + output.length,
                                [](std::uint8_t value) {
                                    return value == 0xa5u;
                                }),
                    "T=3 generic PLE consumer preserves padded output tails");
        }
        for (const auto& [view, row_width] :
             {std::pair{ple_block.post.value, N_EMBD},
              std::pair{ple_block.post.gated, HC_DIM}}) {
            const auto* const values =
                static_cast<const float*>(view.contents);
            for (std::uint32_t row = 0u; row < 3u; ++row) {
                const float* const begin =
                    values + static_cast<std::size_t>(row) * row_width;
                require(std::all_of(begin, begin + row_width,
                                    [](float value) {
                                        return std::isfinite(value);
                                    }) &&
                            std::any_of(begin, begin + row_width,
                                        [](float value) {
                                            return std::abs(value) > 1.0e-7f;
                                        }),
                        "T=3 generic PLE step computes finite nonzero values for every row");
            }
        }
        BufferRef reference_ple_residual = allocate_zeroed_buffer(
            context, "T=3 direct PLE reference residual", residual.length);
        std::memcpy(reference_ple_residual.contents,
                    generic_ple_reference_residual_input.data(),
                    generic_ple_reference_residual_input.size());
        PLEBlockBuffers reference_ple_t3 =
            generic_ple_payload.buffers.ple;
        const std::array<BufferRef, 6u> reference_outputs{
            allocate("T=3 direct PLE reference key", outputs[0].length),
            allocate("T=3 direct PLE reference value", outputs[1].length),
            allocate("T=3 direct PLE reference query", outputs[2].length),
            allocate("T=3 direct PLE reference gate", outputs[3].length),
            allocate("T=3 direct PLE reference gated", outputs[4].length),
            allocate("T=3 direct PLE reference norm", outputs[5].length)};
        for (std::size_t index = 0u; index < reference_outputs.size(); ++index) {
            std::memset(reference_outputs[index].contents, 0xa5u,
                        reference_outputs[index].length);
        }
        reference_ple_t3.key_q8_1 = allocate(
            "T=3 direct PLE reference Q8_1 key activations",
            generic_ple_payload.buffers.ple.key_q8_1.length);
        reference_ple_t3.matvec_sink = allocate(
            "T=3 direct PLE reference mat-vec sink",
            generic_ple_payload.buffers.ple.matvec_sink.length);
        reference_ple_t3.key_weights = ple_block.key_weights;
        reference_ple_t3.value_weights = ple_block.value_weights;
        reference_ple_t3.post.conv1d = ple_block.post.conv1d;
        reference_ple_t3.post.key_gamma = ple_block.post.key_gamma;
        reference_ple_t3.post.query_gamma =
            ple_block.post.query_gamma;
        reference_ple_t3.post.conv_gamma = ple_block.post.conv_gamma;
        reference_ple_t3.post.key = reference_outputs[0];
        reference_ple_t3.post.value = reference_outputs[1];
        reference_ple_t3.post.query = reference_outputs[2];
        reference_ple_t3.post.gate = reference_outputs[3];
        reference_ple_t3.post.gated = reference_outputs[4];
        reference_ple_t3.post.ple_norm = reference_outputs[5];
        reference_ple_t3.post.residual = reference_ple_residual;
        const WindowPLEEntryBuffers reference_ple_entry_t3{
            pending_output, inject2, reference_ple_t3};
        auto reference_ple_command = context.make_command_buffer(
            "T=3 direct PLE entry reference", error);
        require(reference_ple_command != nullptr,
                "create T=3 direct PLE reference command: " + error);
        require(session->track_use(*reference_ple_command, error),
                "track PLE history through T=3 direct reference: " + error);
        require(encode_window_ple_entry(
                    context, *reference_ple_command,
                    generic_ple_rows_count,
                    fullmode_ple_key_format(ple_key_type),
                    generic_ple_parameter_values.nc,
                    PLEHistoryUpdate::defer, reference_ple_entry_t3, error),
                "encode direct T=3 PLE-entry reference: " + error);
        require(reference_ple_command->commit(error),
                "commit direct T=3 PLE-entry reference: " + error);
        require(reference_ple_command->wait_until_completed(error),
                "wait for direct T=3 PLE-entry reference: " + error);
        for (std::size_t index = 0u; index < outputs.size(); ++index) {
            require(std::memcmp(outputs[index].contents,
                                reference_outputs[index].contents,
                                outputs[index].length) == 0,
                    "T=3 composed PLE output matches direct entry for output " +
                        std::to_string(index));
        }
        require(std::memcmp(residual.contents,
                            reference_ple_residual.contents,
                            residual.length) == 0,
                "T=3 composed PLE residual matches the direct entry");
        require(std::equal(
                    ple_history_initial.begin(), ple_history_initial.end(),
                    static_cast<const std::uint8_t*>(
                        ple_block.post.history.contents)),
                "T=3 generic speculative PLE step defers committed history");
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

void run_fullmode_output_projection_test(Context& context,
                                         const fs::path& parent_directory) {
    constexpr std::uint32_t kRows = 1u;
    constexpr std::size_t kOutputBytes =
        static_cast<std::size_t>(kWindowOutputVocabularySize) * sizeof(float);
    constexpr std::size_t kQ8Bytes =
        (kWindowOutputInputWidth / 32u) * 36u;

    const fs::path directory = parent_directory / "fullmode-output-projection";
    fs::create_directory(directory);
    const Fixture fixture =
        write_fullmode_handoff_fixture(directory, kGgmlTypeQ2_0, true);
    WindowSources sources = inspect_fixture(fixture);
    Geometry full_geometry = owner_test_geometry(fixture, sources);
    full_geometry.family = "qwen4exp";
    full_geometry.model = "synthetic-fullmode-output-projection";
    full_geometry.n_expert = 256u;
    full_geometry.n_qsa_layers = 1u;
    full_geometry.n_kv_head = 2u;
    full_geometry.head_dim = 256u;
    full_geometry.n_embd = kWindowOutputInputWidth;
    full_geometry.vocab = kWindowOutputVocabularySize;
    full_geometry.types = {"q2_0", "iq4_xs"};
    full_geometry.layers[0].gate_up_type = "q2_0";
    full_geometry.layers[0].down_type = "q2_0";

    std::string error;
    WindowPlan plan;
    require(plan_windows(sources, context.device_facts().max_buffer_length,
                         plan, error),
            "plan sparse FullMode output.weight fixture: " + error);
    auto full_mode = FullModeStorage::open(
        context, fixture.path.string(), full_geometry, sources, plan, error);
    require(full_mode != nullptr,
            "open FullMode output.weight fixture: " + error);
    FullModeTensorView output_weight;
    require(full_mode->dense_tensor_view("output.weight", output_weight,
                                         error) &&
                output_weight.ggml_type == kGgmlTypeIQ4XS &&
                output_weight.shape ==
                    std::vector<std::uint64_t>{kWindowOutputInputWidth,
                                               kWindowOutputVocabularySize} &&
                output_weight.buffer.length() == fixture.output_weight_bytes,
            "FullMode resolves the production-shaped IQ4_XS output.weight: " +
                error);

    Geometry runtime_geometry;
    runtime_geometry.family = "qwen4exp";
    runtime_geometry.model = "synthetic-window-runtime";
    runtime_geometry.n_layers = kSessionModelLayers;
    runtime_geometry.n_expert = 256u;
    runtime_geometry.n_qsa_layers = kSessionQsaLayers;
    runtime_geometry.n_kv_head = kSessionKvHeads;
    runtime_geometry.head_dim = kSessionHeadWidth;
    runtime_geometry.n_embd = kWindowOutputInputWidth;
    runtime_geometry.vocab = kWindowOutputVocabularySize;
    auto scratch =
        WindowScratchStorage::open(context, runtime_geometry, 4u, error);
    require(scratch != nullptr,
            "allocate FullMode output projection scratch: " + error);

    const WindowScratchLayout& layout = scratch->layout();
    const BufferRef mixed = require_scratch_view(
        context, *scratch, layout.head_mixed, "head mixed input");
    const BufferRef q8_1 = require_scratch_view(
        context, *scratch, layout.head_q8_1, "head Q8_1 activations");
    const BufferRef logits = require_scratch_view(
        context, *scratch, layout.head_logits, "head logits");
    const BufferRef sink = require_scratch_view(
        context, *scratch, layout.matvec_sink, "head mat-vec sink");
    std::fill_n(static_cast<float*>(mixed.contents),
                mixed.length / sizeof(float), 1.0f);
    std::memset(q8_1.contents, 0xa5, q8_1.length);
    std::memset(logits.contents, 0xa5, logits.length);
    std::memset(sink.contents, 0xa5, sink.length);
    BufferRef active_rows =
        allocate_zeroed_buffer(context, "output projection active rows",
                               sizeof(kRows));
    std::memcpy(active_rows.contents, &kRows, sizeof(kRows));
    const WindowOutputProjectionBuffers projection{
        mixed, q8_1, context.buffers().read_only(active_rows), logits, sink};

    auto success = context.make_command_buffer(
        "FullMode output projection success", error);
    require(success != nullptr,
            "create FullMode output projection command: " + error);
    require(Encoder::encode(
                context, *success, "FullMode output projection",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_window_output_projection(
                        context, encoder, *full_mode, kRows, projection,
                        encode_error);
                },
                error),
            "encode FullMode Q8_1-to-logits projection: " + error);
    require(success->commit(error),
            "commit FullMode output projection: " + error);
    id<MTLCommandBuffer> native_success =
        (__bridge id<MTLCommandBuffer>)success->unsafe_native_handle();
    [native_success waitUntilCompleted];
    require(native_success.status == MTLCommandBufferStatusCompleted,
            "FullMode output projection command completes");

    std::string oracle_error;
    std::vector<Q81> oracle_activations;
    require(twin::quantize_q8_1_reference(
                std::span<const float>(
                    static_cast<const float*>(mixed.contents),
                    kWindowOutputInputWidth),
                kWindowOutputInputWidth, 1u, oracle_activations,
                oracle_error),
            "quantize output projection oracle activations: " + oracle_error);
    const auto* const mapped_weights =
        static_cast<const block_iq4_xs*>(output_weight.buffer.contents());
    require(std::memcmp(mapped_weights, fixture.output_weight_first_row.data(),
                        sizeof(fixture.output_weight_first_row)) == 0,
            "FullMode maps the independently retained synthetic output row");
    std::vector<float> expected_first_logit;
    require(twin::iq4_xs_matvec_reference(
                std::span<const block_iq4_xs>(
                    fixture.output_weight_first_row),
                oracle_activations, kWindowOutputInputWidth, 1u, 1u, 1u,
                expected_first_logit, oracle_error),
            "evaluate the mapped first vocabulary row with the IQ4_XS oracle: " +
                oracle_error);
    const auto* const output = static_cast<const float*>(logits.contents);
    require(expected_first_logit.size() == 1u && output[0] != 0.0f &&
                output[0] == expected_first_logit[0] &&
                output[0] > 0.0f &&
                std::all_of(output + 1u,
                            output + kWindowOutputVocabularySize,
                            [](float value) { return value == 0.0f; }),
            "mapped nonzero IQ4_XS row matches the independent Q8_1 projection oracle");
    const auto* const logits_tail =
        static_cast<const std::uint8_t*>(logits.contents) + kOutputBytes;
    require(std::all_of(logits_tail,
                        static_cast<const std::uint8_t*>(logits.contents) +
                            logits.length,
                        [](std::uint8_t byte) { return byte == 0xa5u; }),
            "projection writes only the active logits row");
    require(std::any_of(static_cast<const std::uint8_t*>(q8_1.contents),
                        static_cast<const std::uint8_t*>(q8_1.contents) +
                            kQ8Bytes,
                        [](std::uint8_t byte) { return byte != 0xa5u; }),
            "projection quantizes mixed rows into dedicated Q8_1 scratch");
    require(*static_cast<const std::uint32_t*>(active_rows.contents) == kRows,
            "projection preserves active-row metadata");
    require(std::all_of(static_cast<const float*>(mixed.contents),
                        static_cast<const float*>(mixed.contents) +
                            mixed.length / sizeof(float),
                        [](float value) { return value == 1.0f; }),
            "projection preserves its F32 head input");

    std::memset(q8_1.contents, 0xa5, q8_1.length);
    std::memset(logits.contents, 0x5a, logits.length);
    WindowOutputProjectionBuffers misaligned_output = projection;
    misaligned_output.logits = logits.slice(2u, logits.length - 2u);
    auto rollback = context.make_command_buffer(
        "FullMode output projection late failure", error);
    require(rollback != nullptr,
            "create FullMode output projection rollback command: " + error);
    const bool rejected_late =
        !encode_window_output_projection(context, *rollback, *full_mode,
                                         kRows, misaligned_output, error) &&
        error.find("mat-vec output buffer is undersized or misaligned") !=
            std::string::npos &&
        rollback->unsafe_native_handle() == nullptr;
    require(rejected_late,
            "late output alignment failure must discard prior Q8_1 encoding: " +
                error);
    require(std::all_of(static_cast<const std::uint8_t*>(q8_1.contents),
                        static_cast<const std::uint8_t*>(q8_1.contents) +
                            q8_1.length,
                        [](std::uint8_t byte) { return byte == 0xa5u; }) &&
                std::all_of(
                    static_cast<const std::uint8_t*>(logits.contents),
                    static_cast<const std::uint8_t*>(logits.contents) +
                        logits.length,
                    [](std::uint8_t byte) { return byte == 0x5au; }),
            "discarded output projection leaves quantized and logits scratch unchanged");

    std::memset(q8_1.contents, 0xa5, q8_1.length);
    std::memset(logits.contents, 0x5a, logits.length);
    auto caller_rollback = context.make_command_buffer(
        "FullMode output projection caller-owned late failure", error);
    require(caller_rollback != nullptr,
            "create caller-owned output projection rollback command: " + error);
    const bool caller_rejected_late =
        !Encoder::encode(
            context, *caller_rollback,
            "FullMode output projection caller-owned rollback",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_window_output_projection(
                    context, encoder, *full_mode, kRows, misaligned_output,
                    encode_error);
            },
            error) &&
        error.find("mat-vec output buffer is undersized or misaligned") !=
            std::string::npos &&
        caller_rollback->unsafe_native_handle() == nullptr;
    require(caller_rejected_late,
            "caller-owned late mat-vec failure must discard its Q8_1 dispatch: " +
                error);
    require(std::all_of(static_cast<const std::uint8_t*>(q8_1.contents),
                        static_cast<const std::uint8_t*>(q8_1.contents) +
                            q8_1.length,
                        [](std::uint8_t byte) { return byte == 0xa5u; }) &&
                std::all_of(
                    static_cast<const std::uint8_t*>(logits.contents),
                    static_cast<const std::uint8_t*>(logits.contents) +
                        logits.length,
                    [](std::uint8_t byte) { return byte == 0x5au; }),
            "discarded caller-owned projection segment leaves scratch unchanged");

    WindowOutputProjectionBuffers aliased_q8 = projection;
    aliased_q8.q8_1 = mixed.slice(0u, kQ8Bytes);
    auto alias_command = context.make_command_buffer(
        "FullMode output projection alias rejection", error);
    require(alias_command != nullptr,
            "create FullMode output projection alias command: " + error);
    const bool rejected_alias =
        !encode_window_output_projection(context, *alias_command, *full_mode,
                                         kRows, aliased_q8, error) &&
        error.find("Q8_1 scratch overlaps mixed input") !=
            std::string::npos &&
        alias_command->unsafe_native_handle() == nullptr;
    require(rejected_alias,
            "output projection must reject Q8_1/mixed alias before dispatch: " +
                error);

    BufferRef row_alias_backing = allocate_zeroed_buffer(
        context, "output projection aliased active-row backing", kQ8Bytes);
    require(static_cast<bool>(row_alias_backing),
            "allocate shared backing for Q8_1 and active-row alias");
    std::memset(row_alias_backing.contents, 0x6bu, row_alias_backing.length);
    std::memcpy(row_alias_backing.contents, &kRows, sizeof(kRows));
    std::vector<std::uint8_t> row_alias_before(kQ8Bytes);
    std::memcpy(row_alias_before.data(), row_alias_backing.contents, kQ8Bytes);
    WindowOutputProjectionBuffers aliased_row_count = projection;
    aliased_row_count.q8_1 = row_alias_backing;
    aliased_row_count.active_rows = context.buffers().read_only(
        row_alias_backing.slice(0u, sizeof(kRows)));
    auto row_alias_command = context.make_command_buffer(
        "FullMode output projection read-only input alias rejection", error);
    require(row_alias_command != nullptr,
            "create output projection read-only input alias command: " + error);
    const bool rejected_row_alias =
        !encode_window_output_projection(
            context, *row_alias_command, *full_mode, kRows,
            aliased_row_count, error) &&
        error.find("Q8_1 output overlaps a later read-only input") !=
            std::string::npos &&
        row_alias_command->unsafe_native_handle() == nullptr;
    require(rejected_row_alias,
            "output projection must reject Q8_1 aliasing active-row metadata: " +
                error);
    require(std::equal(row_alias_before.begin(), row_alias_before.end(),
                       static_cast<const std::uint8_t*>(
                           row_alias_backing.contents)),
            "rejected read-only input alias leaves shared backing unchanged");
}

void run_fullmode_output_head_test(Context& context,
                                   const fs::path& parent_directory) {
    constexpr std::uint32_t kRows = 1u;
    constexpr std::size_t kActiveLogitsBytes =
        static_cast<std::size_t>(kWindowOutputVocabularySize) * sizeof(float);

    const fs::path directory = parent_directory / "fullmode-output-head";
    const fs::path valid_directory = directory / "valid";
    const fs::path missing_output_directory = directory / "missing-output";
    fs::create_directories(valid_directory);
    fs::create_directories(missing_output_directory);
    const Fixture fixture = write_fullmode_handoff_fixture(
        valid_directory, kGgmlTypeQ2_0, true, true);
    WindowSources sources = inspect_fixture(fixture);
    Geometry full_geometry = owner_test_geometry(fixture, sources);
    full_geometry.family = "qwen4exp";
    full_geometry.model = "synthetic-fullmode-output-head";
    full_geometry.n_expert = 256u;
    full_geometry.n_qsa_layers = 1u;
    full_geometry.n_kv_head = 2u;
    full_geometry.head_dim = 256u;
    full_geometry.n_embd = kWindowOutputInputWidth;
    full_geometry.vocab = kWindowOutputVocabularySize;
    full_geometry.types = {"q2_0", "bf16", "f32", "iq4_xs"};
    full_geometry.layers[0].gate_up_type = "q2_0";
    full_geometry.layers[0].down_type = "q2_0";

    std::string error;
    WindowPlan plan;
    require(plan_windows(sources, context.device_facts().max_buffer_length,
                         plan, error),
            "plan FullMode output-head fixture: " + error);
    auto full_mode = FullModeStorage::open(
        context, fixture.path.string(), full_geometry, sources, plan, error);
    require(full_mode != nullptr,
            "open FullMode output-head fixture: " + error);

    Geometry runtime_geometry;
    runtime_geometry.family = "qwen4exp";
    runtime_geometry.model = "synthetic-window-runtime";
    runtime_geometry.n_layers = kSessionModelLayers;
    runtime_geometry.n_expert = 256u;
    runtime_geometry.n_qsa_layers = kSessionQsaLayers;
    runtime_geometry.n_kv_head = kSessionKvHeads;
    runtime_geometry.head_dim = kSessionHeadWidth;
    runtime_geometry.n_embd = kWindowOutputInputWidth;
    runtime_geometry.vocab = kWindowOutputVocabularySize;
    auto scratch =
        WindowScratchStorage::open(context, runtime_geometry, 4u, error);
    require(scratch != nullptr,
            "allocate FullMode output-head scratch: " + error);

    const WindowScratchLayout& layout = scratch->layout();
    BufferRef residual = require_scratch_view(
        context, *scratch, layout.residual, "output-head residual");
    BufferRef normalized = require_scratch_view(
        context, *scratch, layout.head_normalized, "head normalized");
    BufferRef low_rank = require_scratch_view(
        context, *scratch, layout.head_low_rank, "head low rank");
    BufferRef gate = require_scratch_view(
        context, *scratch, layout.head_gate, "head gate");
    BufferRef mixed = require_scratch_view(
        context, *scratch, layout.head_mixed, "head mixed");
    BufferRef q8_1 = require_scratch_view(
        context, *scratch, layout.head_q8_1, "head Q8_1");
    BufferRef logits = require_scratch_view(
        context, *scratch, layout.head_logits, "head logits");
    BufferRef matvec_sink = require_scratch_view(
        context, *scratch, layout.matvec_sink, "head mat-vec sink");
    std::fill_n(static_cast<float*>(residual.contents),
                residual.length / sizeof(float), 1.0f);
    std::memset(normalized.contents, 0xa5, normalized.length);
    std::memset(low_rank.contents, 0xa5, low_rank.length);
    std::memset(gate.contents, 0xa5, gate.length);
    std::memset(mixed.contents, 0xa5, mixed.length);
    std::memset(q8_1.contents, 0xa5, q8_1.length);
    std::memset(logits.contents, 0xa5, logits.length);
    std::memset(matvec_sink.contents, 0xa5, matvec_sink.length);
    BufferRef active_rows = allocate_zeroed_buffer(
        context, "FullMode output-head active rows", sizeof(kRows));
    std::memcpy(active_rows.contents, &kRows, sizeof(kRows));
    const WindowOutputHeadBuffers head_buffers{
        residual, context.buffers().read_only(active_rows)};

    const WindowOutputHeadBuffers aliased_residual_buffers{
        logits.slice(0u, residual.length), context.buffers().read_only(active_rows)};
    require(static_cast<bool>(aliased_residual_buffers.residual),
            "create an output-logits alias for output-head residual");
    auto alias_rejected = context.make_command_buffer(
        "FullMode output-head cross-stage alias rejection", error);
    require(alias_rejected != nullptr,
            "create FullMode output-head alias command: " + error);
    WindowOutputHeadResult alias_result;
    require(!encode_window_output_head(
                context, *alias_rejected, *full_mode, *scratch, kRows,
                aliased_residual_buffers, alias_result, error) &&
                error.find("overlaps later projection output scratch") !=
                    std::string::npos &&
                alias_rejected->unsafe_native_handle() == nullptr &&
                !alias_result.logits,
            "residual/logits cross-stage alias must be rejected transactionally");
    const auto filled_with = [](BufferRef view, std::uint8_t value) {
        const auto* const bytes =
            static_cast<const std::uint8_t*>(view.contents);
        return std::all_of(bytes, bytes + view.length,
                           [value](std::uint8_t byte) {
                               return byte == value;
                           });
    };
    require(filled_with(normalized, 0xa5u) &&
                filled_with(low_rank, 0xa5u) &&
                filled_with(gate, 0xa5u) &&
                filled_with(mixed, 0xa5u) &&
                filled_with(q8_1, 0xa5u) &&
                filled_with(logits, 0xa5u) &&
                filled_with(matvec_sink, 0xa5u),
            "cross-stage alias rejection preserves all head scratch bytes");

    WindowOutputHeadResult head_result;

    auto success = context.make_command_buffer(
        "FullMode output-head composition", error);
    require(success != nullptr,
            "create FullMode output-head command: " + error);
    require(Encoder::encode(
                context, *success, "FullMode output-head composition",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_window_output_head(
                        context, encoder, *full_mode, *scratch, kRows,
                        head_buffers, head_result, encode_error);
                },
                error),
            "encode FullMode mixer and vocabulary projection: " + error);
    require(success->commit(error),
            "commit FullMode output-head composition: " + error);
    id<MTLCommandBuffer> native_success =
        (__bridge id<MTLCommandBuffer>)success->unsafe_native_handle();
    [native_success waitUntilCompleted];
    require(native_success.status == MTLCommandBufferStatusCompleted,
            "FullMode output-head command completes");
    require(head_result.logits &&
                head_result.logits.length == kActiveLogitsBytes,
            "output-head result publishes only active logits rows");

    FullModeTensorView output_weight;
    require(full_mode->dense_tensor_view("output.weight", output_weight,
                                         error),
            "resolve output-head oracle weight: " + error);
    const auto* const mapped_weights =
        static_cast<const block_iq4_xs*>(output_weight.buffer.contents());
    require(std::memcmp(mapped_weights, fixture.output_weight_first_row.data(),
                        sizeof(fixture.output_weight_first_row)) == 0,
            "composed output head resolves the retained synthetic output row");
    std::string oracle_error;
    std::vector<Q81> oracle_activations;
    require(twin::quantize_q8_1_reference(
                std::span<const float>(
                    static_cast<const float*>(mixed.contents),
                    kWindowOutputInputWidth),
                kWindowOutputInputWidth, 1u, oracle_activations,
                oracle_error),
            "quantize composed head oracle activation: " + oracle_error);
    std::vector<float> expected_first_logit;
    require(twin::iq4_xs_matvec_reference(
                std::span<const block_iq4_xs>(
                    fixture.output_weight_first_row),
                oracle_activations, kWindowOutputInputWidth, 1u, 1u, 1u,
                expected_first_logit, oracle_error),
            "compute composed output-head scalar oracle: " + oracle_error);
    const auto* const actual_logits =
        static_cast<const float*>(head_result.logits.contents);
    require(expected_first_logit.size() == 1u &&
                std::isfinite(actual_logits[0]) &&
                actual_logits[0] > 0.0f &&
                actual_logits[0] == expected_first_logit[0],
            "FullMode output-head logit matches the independent scalar oracle");
    const float expected_norm =
        1.0f / std::sqrt(1.0f + 1.0e-6f);
    const float expected_down_scaled = 0.25f * expected_norm;
    const float expected_down_silu =
        expected_down_scaled /
        (1.0f + std::exp(-expected_down_scaled));
    const float expected_up = 0.5f * expected_down_silu;
    const float expected_mixed =
        expected_norm / (1.0f + std::exp(-expected_up));
    require(std::all_of(
                static_cast<const float*>(mixed.contents),
                static_cast<const float*>(mixed.contents) +
                    kWindowOutputInputWidth,
                [expected_mixed](float value) {
                    return std::isfinite(value) &&
                           std::abs(value - expected_mixed) < 2.0e-5f;
                }),
            "head mixer output matches the sparse FullMode scalar oracle");
    require(std::all_of(
                static_cast<const float*>(residual.contents),
                static_cast<const float*>(residual.contents) +
                    residual.length / sizeof(float),
                [](float value) { return value == 1.0f; }),
            "composed output head preserves the final GR residual");
    require(std::all_of(
                static_cast<const std::uint8_t*>(logits.contents) +
                    kActiveLogitsBytes,
                static_cast<const std::uint8_t*>(logits.contents) +
                    logits.length,
                [](std::uint8_t byte) { return byte == 0xa5u; }),
            "composed output head writes only the active logits row");

    const Fixture missing_output_fixture =
        write_fullmode_handoff_fixture(missing_output_directory,
                                       kGgmlTypeQ2_0, false, true);
    WindowSources missing_sources = inspect_fixture(missing_output_fixture);
    Geometry missing_geometry = owner_test_geometry(
        missing_output_fixture, missing_sources);
    missing_geometry.family = "qwen4exp";
    missing_geometry.model = "synthetic-fullmode-output-head-missing-weight";
    missing_geometry.n_expert = 256u;
    missing_geometry.n_qsa_layers = 1u;
    missing_geometry.n_kv_head = 2u;
    missing_geometry.head_dim = 256u;
    missing_geometry.n_embd = kWindowOutputInputWidth;
    missing_geometry.vocab = kWindowOutputVocabularySize;
    missing_geometry.types = {"q2_0", "bf16", "f32"};
    missing_geometry.layers[0].gate_up_type = "q2_0";
    missing_geometry.layers[0].down_type = "q2_0";
    WindowPlan missing_plan;
    require(plan_windows(missing_sources,
                         context.device_facts().max_buffer_length,
                         missing_plan, error),
            "plan FullMode output-head missing-weight fixture: " + error);
    auto missing_weight = FullModeStorage::open(
        context, missing_output_fixture.path.string(), missing_geometry,
        missing_sources, missing_plan, error);
    require(missing_weight != nullptr,
            "open FullMode output-head missing-weight fixture: " + error);

    const auto snapshot_bytes = [](BufferRef view) {
        std::vector<std::uint8_t> bytes(view.length);
        std::memcpy(bytes.data(), view.contents, view.length);
        return bytes;
    };
    std::memset(normalized.contents, 0x5au, normalized.length);
    std::memset(low_rank.contents, 0x5au, low_rank.length);
    std::memset(gate.contents, 0x5au, gate.length);
    std::memset(mixed.contents, 0x5au, mixed.length);
    std::memset(q8_1.contents, 0x5au, q8_1.length);
    std::memset(logits.contents, 0x5au, logits.length);
    std::memset(matvec_sink.contents, 0x5au, matvec_sink.length);
    const auto normalized_before = snapshot_bytes(normalized);
    const auto low_rank_before = snapshot_bytes(low_rank);
    const auto gate_before = snapshot_bytes(gate);
    const auto mixed_before = snapshot_bytes(mixed);
    const auto q8_1_before = snapshot_bytes(q8_1);
    const auto logits_before = snapshot_bytes(logits);
    const auto sink_before = snapshot_bytes(matvec_sink);
    WindowOutputHeadResult failed_result;
    auto rollback = context.make_command_buffer(
        "FullMode output-head missing-weight rollback", error);
    require(rollback != nullptr,
            "create FullMode output-head rollback command: " + error);
    const bool rejected_late =
        !encode_window_output_head(
            context, *rollback, *missing_weight, *scratch, kRows,
            head_buffers, failed_result, error) &&
        error.find("cannot resolve output.weight") != std::string::npos &&
        rollback->unsafe_native_handle() == nullptr && !failed_result.logits;
    require(rejected_late,
            "missing output.weight must discard the previously encoded head mixer: " +
                error);
    require(std::equal(normalized_before.begin(), normalized_before.end(),
                       static_cast<const std::uint8_t*>(normalized.contents)) &&
                std::equal(low_rank_before.begin(), low_rank_before.end(),
                           static_cast<const std::uint8_t*>(low_rank.contents)) &&
                std::equal(gate_before.begin(), gate_before.end(),
                           static_cast<const std::uint8_t*>(gate.contents)) &&
                std::equal(mixed_before.begin(), mixed_before.end(),
                           static_cast<const std::uint8_t*>(mixed.contents)) &&
                std::equal(q8_1_before.begin(), q8_1_before.end(),
                           static_cast<const std::uint8_t*>(q8_1.contents)) &&
                std::equal(logits_before.begin(), logits_before.end(),
                           static_cast<const std::uint8_t*>(logits.contents)) &&
                std::equal(sink_before.begin(), sink_before.end(),
                           static_cast<const std::uint8_t*>(
                               matvec_sink.contents)),
            "late FullMode output-head failure preserves all scratch bytes");

    auto caller_owned_rollback = context.make_command_buffer(
        "FullMode output-head caller-owned rollback", error);
    require(caller_owned_rollback != nullptr,
            "create caller-owned FullMode output-head rollback command: " +
                error);
    WindowOutputHeadResult caller_owned_result;
    const bool caller_owned_rejected =
        !Encoder::encode(
            context, *caller_owned_rollback,
            "FullMode output-head caller-owned rollback",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_window_output_head(
                    context, encoder, *missing_weight, *scratch, kRows,
                    head_buffers, caller_owned_result, encode_error);
            },
            error) &&
        error.find("cannot resolve output.weight") != std::string::npos &&
        caller_owned_rollback->unsafe_native_handle() == nullptr &&
        !caller_owned_result.logits;
    require(caller_owned_rejected,
            "caller-owned late output-weight failure must discard the whole segment: " +
                error);
    require(std::equal(normalized_before.begin(), normalized_before.end(),
                       static_cast<const std::uint8_t*>(normalized.contents)) &&
                std::equal(low_rank_before.begin(), low_rank_before.end(),
                           static_cast<const std::uint8_t*>(low_rank.contents)) &&
                std::equal(gate_before.begin(), gate_before.end(),
                           static_cast<const std::uint8_t*>(gate.contents)) &&
                std::equal(mixed_before.begin(), mixed_before.end(),
                           static_cast<const std::uint8_t*>(mixed.contents)) &&
                std::equal(q8_1_before.begin(), q8_1_before.end(),
                           static_cast<const std::uint8_t*>(q8_1.contents)) &&
                std::equal(logits_before.begin(), logits_before.end(),
                           static_cast<const std::uint8_t*>(logits.contents)) &&
                std::equal(sink_before.begin(), sink_before.end(),
                           static_cast<const std::uint8_t*>(
                               matvec_sink.contents)),
            "caller-owned rollback preserves all head scratch bytes");
}

float reference_window_sampler_uniform(std::uint64_t seed,
                                       std::uint64_t counter) {
    std::array<std::uint32_t, 4> words{
        static_cast<std::uint32_t>(counter),
        static_cast<std::uint32_t>(counter >> 32u),
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32u)};
    for (std::uint32_t round = 0u; round < 10u; ++round) {
        const std::uint64_t product0 =
            std::uint64_t{0x9e3779b9u} * words[0];
        const std::uint64_t product1 =
            std::uint64_t{0xbb67ae85u} * words[2];
        words = {
            static_cast<std::uint32_t>(product1 >> 32u) ^ words[1] ^ round,
            static_cast<std::uint32_t>(product1),
            static_cast<std::uint32_t>(product0 >> 32u) ^ words[3],
            static_cast<std::uint32_t>(product0)};
    }
    return static_cast<float>(words[0] >> 8u) * (1.0f / 16777216.0f);
}

TailPick reference_window_sampler_tail(
    std::span<const SampCandidate> candidates,
    const WindowSamplerSettings& settings, std::uint64_t seed,
    std::uint64_t counter) {
    const std::size_t k = candidates.size();
    std::array<float, 64> logits{};
    std::array<double, 64> probabilities{};
    for (std::size_t index = 0u; index < k; ++index) {
        logits[index] = std::bit_cast<float>(candidates[index].logit_bits);
    }

    std::size_t keep = k;
    if (settings.top_p < 1.0f) {
        float maximum = logits[0];
        for (std::size_t index = 1u; index < k; ++index) {
            maximum = std::fmax(maximum, logits[index]);
        }
        double sum = 0.0;
        for (std::size_t index = 0u; index < k; ++index) {
            probabilities[index] = std::exp(
                static_cast<double>(logits[index]) -
                static_cast<double>(maximum));
            sum += probabilities[index];
        }
        for (std::size_t index = 0u; index < k; ++index) {
            probabilities[index] /= sum;
        }
        double cumulative = 0.0;
        std::size_t cut = k;
        for (std::size_t index = 0u; index < k; ++index) {
            cumulative += probabilities[index];
            if (cumulative >= static_cast<double>(settings.top_p)) {
                cut = index + 1u;
                break;
            }
        }
        if (cut < static_cast<std::size_t>(settings.min_keep)) {
            cut = std::min(k, static_cast<std::size_t>(settings.min_keep));
        }
        keep = cut;
    }
    if (settings.min_p > 0.0f) {
        const float threshold =
            logits[0] + std::log(settings.min_p);
        for (std::size_t index = 0u; index < keep; ++index) {
            if (logits[index] < threshold) {
                keep = index;
                break;
            }
        }
    }

    const float inverse_temperature =
        1.0f / settings.temperature;
    float scaled_maximum = logits[0] * inverse_temperature;
    for (std::size_t index = 1u; index < keep; ++index) {
        scaled_maximum =
            std::fmax(scaled_maximum,
                      logits[index] * inverse_temperature);
    }
    double sum = 0.0;
    for (std::size_t index = 0u; index < keep; ++index) {
        probabilities[index] = std::exp(
            static_cast<double>(logits[index] * inverse_temperature) -
            static_cast<double>(scaled_maximum));
        sum += probabilities[index];
    }
    for (std::size_t index = 0u; index < keep; ++index) {
        probabilities[index] /= sum;
    }

    const float uniform = reference_window_sampler_uniform(seed, counter);
    std::size_t picked = keep - 1u;
    double cumulative = 0.0;
    for (std::size_t index = 0u; index < keep; ++index) {
        cumulative += probabilities[index];
        if (static_cast<double>(uniform) < cumulative) {
            picked = index;
            break;
        }
    }
    return {candidates[picked].id,
            static_cast<float>(probabilities[picked])};
}

void run_fullmode_output_sampling_test(Context& context,
                                       const fs::path& parent_directory) {
    constexpr std::uint32_t kRows = 2u;
    constexpr std::size_t kActiveLogitsBytes =
        static_cast<std::size_t>(kRows) * kWindowOutputVocabularySize *
        sizeof(float);
    constexpr std::size_t kActiveCandidatesBytes =
        static_cast<std::size_t>(kRows) * 64u * sizeof(SampCandidate);

    const fs::path directory = parent_directory / "fullmode-output-sampling";
    fs::create_directory(directory);
    const Fixture fixture = write_fullmode_handoff_fixture(
        directory, kGgmlTypeQ2_0, true, true);
    WindowSources sources = inspect_fixture(fixture);
    Geometry full_geometry = owner_test_geometry(fixture, sources);
    full_geometry.family = "qwen4exp";
    full_geometry.model = "synthetic-fullmode-output-sampling";
    full_geometry.n_expert = 256u;
    full_geometry.n_qsa_layers = 1u;
    full_geometry.n_kv_head = 2u;
    full_geometry.head_dim = 256u;
    full_geometry.n_embd = kWindowOutputInputWidth;
    full_geometry.vocab = kWindowOutputVocabularySize;
    full_geometry.types = {"q2_0", "bf16", "f32", "iq4_xs"};
    full_geometry.layers[0].gate_up_type = "q2_0";
    full_geometry.layers[0].down_type = "q2_0";

    std::string error;
    WindowPlan plan;
    require(plan_windows(sources, context.device_facts().max_buffer_length,
                         plan, error),
            "plan FullMode output-sampling fixture: " + error);
    auto full_mode = FullModeStorage::open(
        context, fixture.path.string(), full_geometry, sources, plan, error);
    require(full_mode != nullptr,
            "open FullMode output-sampling fixture: " + error);

    Geometry runtime_geometry;
    runtime_geometry.family = "qwen4exp";
    runtime_geometry.model = "synthetic-window-runtime";
    runtime_geometry.n_layers = kSessionModelLayers;
    runtime_geometry.n_expert = 256u;
    runtime_geometry.n_qsa_layers = kSessionQsaLayers;
    runtime_geometry.n_kv_head = kSessionKvHeads;
    runtime_geometry.head_dim = kSessionHeadWidth;
    runtime_geometry.n_embd = kWindowOutputInputWidth;
    runtime_geometry.vocab = kWindowOutputVocabularySize;
    auto scratch =
        WindowScratchStorage::open(context, runtime_geometry, 4u, error);
    require(scratch != nullptr,
            "allocate FullMode output-sampling scratch: " + error);
    auto params_ring = WindowParamsRing::open(context, error);
    require(params_ring != nullptr,
            "open FullMode output-sampling WindowParamsRing: " + error);

    const std::array<std::uint32_t, kRows> tokens{0u};
    WindowSamplerSettings settings{};
    settings.temperature = 0.8f;
    settings.top_p = 1.0f;
    settings.min_p = 0.0f;
    settings.penalty_repeat = 1.0f;
    settings.penalty_freq = 0.0f;
    settings.penalty_present = 0.0f;
    settings.top_k = 4;
    settings.min_keep = 1;
    settings.greedy = 0;
    WindowSamplerSettings second_settings{};
    second_settings.temperature = 0.37f;
    second_settings.top_p = 0.6f;
    second_settings.min_p = 0.15f;
    second_settings.penalty_repeat = 1.25f;
    second_settings.penalty_freq = 0.1f;
    second_settings.penalty_present = 0.2f;
    second_settings.top_k = 2;
    second_settings.min_keep = 2;
    second_settings.greedy = 0;
    second_settings.hist_off = 5u;
    second_settings.hist_len = 1u;
    const std::array<WindowSamplerSettings, kRows> samplers{
        settings, second_settings};
    BufferRef params = params_ring->build_slot(
        0u, 10u, tokens, samplers, 0u, 0x12345678u, 0x9abcdef0u, error);
    require(static_cast<bool>(params),
            "build sampled WindowParams slot: " + error);

    const WindowScratchLayout& layout = scratch->layout();
    BufferRef residual = require_scratch_view(
        context, *scratch, layout.residual, "output-sampling residual");
    const std::array scratch_regions{
        layout.head_normalized, layout.head_low_rank, layout.head_gate,
        layout.head_mixed, layout.head_q8_1, layout.head_logits,
        layout.sampler_candidates, layout.sampler_selection,
        layout.sampler_output, layout.matvec_sink,
    };
    std::vector<BufferRef> writable_scratch;
    writable_scratch.reserve(scratch_regions.size());
    for (std::size_t index = 0u; index < scratch_regions.size(); ++index) {
        BufferRef view = require_scratch_view(
            context, *scratch, scratch_regions[index],
            "output-sampling writable region " + std::to_string(index));
        std::memset(view.contents, 0xa5, view.length);
        writable_scratch.push_back(view);
    }
    BufferRef history = require_scratch_view(
        context, *scratch, layout.sampler_history,
        "output-sampling history");
    std::memset(history.contents, 0, history.length);
    static_cast<std::int32_t*>(history.contents)[5u] = 1;
    std::fill_n(static_cast<float*>(residual.contents),
                residual.length / sizeof(float), 1.0f);

    auto command = context.make_command_buffer(
        "FullMode output sampling success", error);
    require(command != nullptr,
            "create FullMode output-sampling command: " + error);
    WindowOutputSamplingResult encoded;
    require(encode_window_output_sampling(
                context, *command, *full_mode, *scratch, residual, params,
                encoded, error),
            "encode FullMode head and sampled candidates: " + error);
    require(encoded.rows == kRows &&
                encoded.logits.length == kActiveLogitsBytes &&
                encoded.selected_candidates.length ==
                    kActiveCandidatesBytes,
            "publish only active logits and candidate rows");
    require(command->commit(error),
            "commit FullMode output-sampling command: " + error);

    std::vector<TailPick> picks;
    require(sample_window_output_tail(context, *command, params, encoded,
                                      picks, error),
            "run CPU sampling tail after GPU candidate completion: " + error);
    require(picks.size() == kRows,
            "CPU tail returns one sampled result per active row");
    auto* const selected =
        static_cast<SampCandidate*>(
            encoded.selected_candidates.contents);
    const std::uint64_t seed =
        (static_cast<std::uint64_t>(0x9abcdef0u) << 32u) | 0x12345678u;
    const auto* const logits =
        static_cast<const float*>(encoded.logits.contents);
    const std::array<SampCandidate, 4u> expected_row0{{
        {0, std::bit_cast<std::uint32_t>(logits[0u])},
        {1, std::bit_cast<std::uint32_t>(0.0f)},
        {2, std::bit_cast<std::uint32_t>(0.0f)},
        {3, std::bit_cast<std::uint32_t>(0.0f)},
    }};
    const std::array<SampCandidate, 2u> expected_row1{{
        {0, std::bit_cast<std::uint32_t>(
                logits[kWindowOutputVocabularySize])},
        {2, std::bit_cast<std::uint32_t>(0.0f)},
    }};
    const std::array<std::span<const SampCandidate>, kRows>
        expected_candidates{
            std::span<const SampCandidate>(expected_row0),
            std::span<const SampCandidate>(expected_row1)};
    for (std::uint32_t row = 0u; row < kRows; ++row) {
        const std::size_t candidate_offset =
            static_cast<std::size_t>(row) * 64u;
        const std::uint32_t requested_k =
            static_cast<std::uint32_t>(samplers[row].top_k);
        const auto fixture_candidates = expected_candidates[row];
        require(fixture_candidates.size() == requested_k &&
                    logits[static_cast<std::size_t>(row) *
                               kWindowOutputVocabularySize] > 0.0f &&
                    std::all_of(
                        logits + static_cast<std::size_t>(row) *
                                     kWindowOutputVocabularySize +
                            1u,
                        logits + (static_cast<std::size_t>(row) + 1u) *
                                     kWindowOutputVocabularySize,
                        [](float value) { return value == 0.0f; }),
                "synthetic fixture logits match row " + std::to_string(row));
        for (std::size_t candidate_index = 0u;
             candidate_index < fixture_candidates.size(); ++candidate_index) {
            require(selected[candidate_offset + candidate_index].id ==
                        fixture_candidates[candidate_index].id &&
                        selected[candidate_offset + candidate_index]
                                .logit_bits ==
                            fixture_candidates[candidate_index].logit_bits,
                    "GPU top-k candidate matches independent fixture row " +
                        std::to_string(row) + " index " +
                        std::to_string(candidate_index));
        }
        const TailPick expected = reference_window_sampler_tail(
            fixture_candidates, samplers[row], seed, 10u + row);
        require(picks[row].id == expected.id &&
                    picks[row].probability == expected.probability &&
                    std::isfinite(picks[row].probability) &&
                    picks[row].probability > 0.0f &&
                    picks[row].probability <= 1.0f,
                "CPU tail matches the independent row-specific sampler oracle for row " +
                    std::to_string(row));
    }
    std::vector<TailPick> guarded_picks{{-7, 0.25f}};
    const SampCandidate original_first_candidate = selected[0u];
    selected[0u].logit_bits =
        std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::quiet_NaN());
    const bool rejected_nonfinite_candidate =
        !sample_window_output_tail(context, *command, params, encoded,
                                   guarded_picks, error) &&
        error.find("non-finite candidate logit") != std::string::npos &&
        guarded_picks.size() == 1u && guarded_picks[0].id == -7 &&
        guarded_picks[0].probability == 0.25f;
    require(rejected_nonfinite_candidate,
            "CPU tail rejects non-finite GPU candidates without publishing picks: " +
                error);
    selected[0u] = original_first_candidate;

    selected[0u].logit_bits =
        std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::max());
    const bool rejected_nonfinite_probability =
        !sample_window_output_tail(context, *command, params, encoded,
                                   guarded_picks, error) &&
        error.find("produced an invalid probability") != std::string::npos &&
        guarded_picks.size() == 1u && guarded_picks[0].id == -7 &&
        guarded_picks[0].probability == 0.25f;
    require(rejected_nonfinite_probability,
            "CPU tail rejects overflowed probability without publishing picks: " +
                error);
    selected[0u] = original_first_candidate;

    auto* const mutable_window_params =
        static_cast<strata::metal::WindowParams*>(params.contents);
    const std::uint32_t original_rows = mutable_window_params->T;
    mutable_window_params->T = std::numeric_limits<std::uint32_t>::max();
    const bool rejected_wrapped_row_count =
        !sample_window_output_tail(context, *command, params, encoded,
                                   guarded_picks, error) &&
        error.find("row count must be in 1..8") != std::string::npos &&
        guarded_picks.size() == 1u && guarded_picks[0].id == -7 &&
        guarded_picks[0].probability == 0.25f;
    require(rejected_wrapped_row_count,
            "CPU tail rejects oversized T without looping or publishing picks: " +
                error);
    mutable_window_params->T = original_rows;

    const auto snapshot_bytes = [](BufferRef view) {
        std::vector<std::uint8_t> bytes(view.length);
        std::memcpy(bytes.data(), view.contents, view.length);
        return bytes;
    };
    const auto snapshot_writable_scratch = [&] {
        std::vector<std::vector<std::uint8_t>> before;
        before.reserve(writable_scratch.size());
        for (BufferRef view : writable_scratch) {
            std::memset(view.contents, 0x5a, view.length);
            before.push_back(snapshot_bytes(view));
        }
        return before;
    };
    const auto require_scratch_unchanged =
        [&](const std::vector<std::vector<std::uint8_t>>& before,
            const std::string& label) {
            for (std::size_t index = 0u;
                 index < writable_scratch.size(); ++index) {
                const BufferRef view = writable_scratch[index];
                require(std::equal(
                            before[index].begin(), before[index].end(),
                            static_cast<const std::uint8_t*>(view.contents)),
                        label + " preserves writable scratch " +
                            std::to_string(index));
            }
        };
    const auto before_alias = snapshot_writable_scratch();
    BufferRef aliased_residual = require_scratch_view(
        context, *scratch, layout.sampler_candidates,
        "residual/sampler-candidate alias");
    auto alias_command = context.make_command_buffer(
        "FullMode output sampling residual alias rejection", error);
    require(alias_command != nullptr,
            "create output-sampling residual alias command: " + error);
    WindowOutputSamplingResult alias_result;
    const bool rejected_alias =
        !encode_window_output_sampling(
            context, *alias_command, *full_mode, *scratch, aliased_residual,
            params, alias_result, error) &&
        error.find("residual overlaps sampler output scratch") !=
            std::string::npos &&
        alias_command->unsafe_native_handle() == nullptr &&
        alias_result.rows == 0u && !alias_result.logits &&
        !alias_result.selected_candidates;
    require(rejected_alias,
            "residual/candidate alias must be rejected before head dispatch: " +
                error);
    require_scratch_unchanged(before_alias,
                              "residual/candidate alias rejection");

    std::memset(history.contents, 0, history.length);
    BufferRef invalid_params = params_ring->build_slot(
        1u, 10u, tokens, samplers, 0u, 0x12345678u, 0x9abcdef0u, error);
    require(static_cast<bool>(invalid_params),
            "build rollback WindowParams slot: " + error);
    auto* const invalid_window_params =
        static_cast<strata::metal::WindowParams*>(
            invalid_params.contents);
    invalid_window_params->samp[0].penalty_repeat = 0.0f;
    const auto before_invalid_setting = snapshot_writable_scratch();
    auto invalid_setting_command = context.make_command_buffer(
        "FullMode invalid sampler settings preflight", error);
    require(invalid_setting_command != nullptr,
            "create invalid sampler settings command: " + error);
    WindowOutputSamplingResult invalid_setting_result;
    const bool rejected_invalid_setting =
        !encode_window_output_sampling(
            context, *invalid_setting_command, *full_mode, *scratch, residual,
            invalid_params, invalid_setting_result, error) &&
        error.find("repeat penalty must be greater than zero") !=
            std::string::npos &&
        invalid_setting_command->unsafe_native_handle() == nullptr &&
        invalid_setting_result.rows == 0u &&
        !invalid_setting_result.logits &&
        !invalid_setting_result.selected_candidates;
    require(rejected_invalid_setting,
            "invalid sampler settings fail before head dispatch: " + error);
    require_scratch_unchanged(before_invalid_setting,
                              "invalid sampler setting rejection");

    invalid_window_params->samp[0].penalty_repeat = 1.0f;
    invalid_window_params->samp[0].temperature =
        std::numeric_limits<float>::denorm_min();
    auto tiny_temperature_command = context.make_command_buffer(
        "FullMode unscalable sampler temperature preflight", error);
    require(tiny_temperature_command != nullptr,
            "create unscalable temperature command: " + error);
    WindowOutputSamplingResult tiny_temperature_result;
    const bool rejected_tiny_temperature =
        !encode_window_output_sampling(
            context, *tiny_temperature_command, *full_mode, *scratch,
            residual, invalid_params, tiny_temperature_result, error) &&
        error.find("temperature is too small") != std::string::npos &&
        tiny_temperature_command->unsafe_native_handle() == nullptr &&
        tiny_temperature_result.rows == 0u &&
        !tiny_temperature_result.logits &&
        !tiny_temperature_result.selected_candidates;
    require(rejected_tiny_temperature,
            "temperature without a finite reciprocal fails before dispatch: " +
                error);
    require_scratch_unchanged(before_invalid_setting,
                              "unscalable temperature rejection");

    invalid_params = params_ring->build_slot(
        1u, 10u, tokens, samplers, 0u, 0x12345678u, 0x9abcdef0u, error);
    require(static_cast<bool>(invalid_params),
            "rebuild rollback WindowParams slot: " + error);
    auto* const late_invalid_window_params =
        static_cast<strata::metal::WindowParams*>(
            invalid_params.contents);
    late_invalid_window_params->samp[0].hist_off =
        std::numeric_limits<std::uint32_t>::max();
    late_invalid_window_params->samp[0].hist_len = 1u;
    const auto before_late_failure = snapshot_writable_scratch();
    auto rollback = context.make_command_buffer(
        "FullMode output sampling late sampler failure", error);
    require(rollback != nullptr,
            "create FullMode output-sampling rollback command: " + error);
    WindowOutputSamplingResult rejected_result;
    const bool rejected_late =
        !encode_window_output_sampling(
            context, *rollback, *full_mode, *scratch, residual,
            invalid_params, rejected_result, error) &&
        error.find("sampler row history exceeds its buffer") !=
            std::string::npos &&
        rollback->unsafe_native_handle() == nullptr &&
        rejected_result.rows == 0u && !rejected_result.logits &&
        !rejected_result.selected_candidates;
    require(rejected_late,
            "late sampler validation must discard the encoded output head: " +
                error);
    require_scratch_unchanged(before_late_failure,
                              "discarded head/sampler transaction");

    auto caller_owned_rollback = context.make_command_buffer(
        "FullMode caller-owned output sampling late failure", error);
    require(caller_owned_rollback != nullptr,
            "create caller-owned output-sampling rollback command: " + error);
    WindowOutputSamplingResult caller_owned_result;
    const bool caller_owned_rejected =
        !Encoder::encode(
            context, *caller_owned_rollback,
            "FullMode caller-owned output-sampling rollback",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_window_output_sampling(
                    context, encoder, *full_mode, *scratch, residual,
                    invalid_params, caller_owned_result, encode_error);
            },
            error) &&
        error.find("sampler row history exceeds its buffer") !=
            std::string::npos &&
        caller_owned_rollback->unsafe_native_handle() == nullptr &&
        caller_owned_result.rows == 0u && !caller_owned_result.logits &&
        !caller_owned_result.selected_candidates;
    require(caller_owned_rejected,
            "caller-owned late sampler failure must discard the whole segment: " +
                error);
    require_scratch_unchanged(before_late_failure,
                              "discarded caller-owned transaction");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool never_wrap_only = argc == 3 && std::string(argv[2]) == "--never-wrap";
        const bool tiered_copy_only = argc == 3 && std::string(argv[2]) == "--tiered-copy";
        const bool model_owner_only = argc == 3 && std::string(argv[2]) == "--model-owner";
        const bool fullmode_handoff_only =
            argc == 3 && std::string(argv[2]) == "--fullmode-handoff";
        const std::string mode = argc == 3 ? argv[2] : "";
        const bool fullmode_output_projection_only =
            mode == "--fullmode-output-projection";
        const bool fullmode_output_head_only =
            mode == "--fullmode-output-head";
        const bool fullmode_output_sampling_only =
            mode == "--fullmode-output-sampling";
        const bool fullmode_ple_handoff_only =
            mode == "--fullmode-ple-handoff" ||
            mode == "--fullmode-ple-handoff-q8" ||
            mode == "--fullmode-ple-handoff-bf16";
        require(argc == 2 || never_wrap_only || tiered_copy_only ||
            model_owner_only || fullmode_handoff_only ||
                    fullmode_output_projection_only ||
                    fullmode_output_head_only ||
                    fullmode_output_sampling_only ||
                    fullmode_ple_handoff_only,
                "usage: metal_window_storage_test strata.metallib "
                "[--never-wrap|--tiered-copy|--model-owner|--fullmode-handoff|"
                "--fullmode-output-projection|--fullmode-output-head|"
                "--fullmode-output-sampling|"
                "--fullmode-ple-handoff|--fullmode-ple-handoff-q8|"
                "--fullmode-ple-handoff-bf16]");
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
        if (fullmode_output_projection_only) {
            run_fullmode_output_projection_test(*context, directory);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_output_projection_test: OK\n";
            return 0;
        }
        if (fullmode_output_head_only) {
            run_fullmode_output_head_test(*context, directory);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_output_head_test: OK\n";
            return 0;
        }
        if (fullmode_output_sampling_only) {
            run_fullmode_output_sampling_test(*context, directory);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_output_sampling_test: OK\n";
            return 0;
        }
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
            const std::uint32_t ple_key_type =
                mode == "--fullmode-ple-handoff-q8"
                    ? kGgmlTypeQ8_0
                    : mode == "--fullmode-ple-handoff-bf16"
                          ? kGgmlTypeBF16
                          : kGgmlTypeQ2_0;
            run_fullmode_handoff_test(*context, directory, true,
                                      ple_key_type);
            context.reset();
            std::error_code ignored;
            fs::remove_all(directory, ignored);
            std::cout << "metal_fullmode_ple_handoff_test (GGML type "
                      << ple_key_type << "): OK\n";
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
