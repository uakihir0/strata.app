// Device-free tests for geometry v1 and the U1 planner.
#include "strata/metal/geometry.hpp"
#include "strata/metal/expert_profile.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/mtp_kv.hpp"
#include "strata/metal/mtp_params_layout.hpp"
#include "strata/metal/plan.hpp"
#include "strata/metal/prefill.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::metal::kMtpParamsPrefillGroupRows;
using strata::metal::kMtpParamsSlotBytes;

constexpr std::uint64_t kFixedAllowanceBytes = 256ull * 1024 * 1024;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

strata::metal::Geometry q2_geometry() {
    strata::metal::Geometry geometry;
    geometry.family = "qwen";
    geometry.model = "Q2_0";
    geometry.shards = {{"q2-00001.gguf", 37'623'740'192ull},
                       {"q2-00002.gguf", 28'800'138'432ull}};
    geometry.n_layers = 48;
    geometry.n_expert = 512;
    geometry.nonexpert_bytes = 3'638'853'120ull;
    geometry.wrapped_unread_bytes = 1'024;
    geometry.mtp_bytes = 823'887'872ull;
    geometry.mtp_bind_bytes = 16'300'000;
    geometry.n_qsa_layers = 12;
    geometry.n_kv_head = 2;
    geometry.head_dim = 256;
    geometry.n_embd = 2'560;
    geometry.vocab = 248'320;
    geometry.types = {"q2_0", "bf16", "f16", "f32"};
    geometry.layers.assign(48, {1'382'400, "q2_0", "q2_0"});
    return geometry;
}

strata::metal::PlanOptions base_options() {
    strata::metal::PlanOptions options;
    options.working_set_bytes = 64ull * 1024 * 1024 * 1024;
    options.ram_bytes = 64ull * 1024 * 1024 * 1024;
    options.available_bytes = 40ull * 1024 * 1024 * 1024;
    options.gpu = "Apple_M5_Pro";
    options.max_context = 65'536;
    options.spec_enabled = true;
    options.prefill_mode = strata::metal::PrefillMode::Auto;
    options.prefill_auto_max = 8'192;
    return options;
}

bool parse_geometry(const std::string& text, strata::metal::Geometry& geometry, std::string& error) {
    std::istringstream input(text);
    return strata::metal::read_geometry(input, geometry, error);
}

void geometry_round_trip() {
    auto expected = q2_geometry();
    check(strata::metal::is_qwen4exp_catalog_family(expected.family) &&
              strata::metal::is_qwen4exp_runtime_geometry(expected),
          "Qwen package family and architecture dimensions identify the runtime geometry");
    auto coder = expected;
    coder.family = "coder";
    check(strata::metal::is_qwen4exp_catalog_family(coder.family) &&
              strata::metal::is_qwen4exp_runtime_geometry(coder),
          "Coder package family shares the Qwen4exp runtime geometry");
    auto inspected_default = expected;
    inspected_default.family = "qwen4exp";
    check(strata::metal::is_qwen4exp_catalog_family(inspected_default.family) &&
              strata::metal::is_qwen4exp_runtime_geometry(inspected_default),
          "directly inspected geometry keeps the Qwen4exp default family label");
    std::stringstream encoded;
    std::string error;
    check(strata::metal::write_geometry(encoded, expected, error), "write geometry v1");
    strata::metal::Geometry actual;
    const bool read_ok = strata::metal::read_geometry(encoded, actual, error);
    if (!read_ok) std::cerr << "geometry read error: " << error << '\n';
    check(read_ok, "read geometry v1");
    check(actual.family == expected.family && actual.model == expected.model, "geometry identity round trips");
    check(actual.shards.size() == 2 && actual.shards[1].bytes == expected.shards[1].bytes,
          "geometry shard sizes round trip");
    check(actual.layers.size() == 48 && actual.layers[47].blob_bytes == 1'382'400,
          "geometry layer records round trip");
    check(actual.types == expected.types, "geometry tensor types round trip");
    std::stringstream encoded_again;
    check(strata::metal::write_geometry(encoded_again, actual, error) &&
              encoded_again.str() == encoded.str(),
          "geometry writer emits a stable canonical serialization");
    check(strata::metal::geometry_type_name(29) == "iq1_m",
          "known IQ1_M GGML type keeps its supported canonical name");
    check(strata::metal::geometry_type_name(987'654) == "ggml_type_987654",
          "unknown GGML type IDs remain identifiable in diagnostics");
}

void geometry_rejects_ambiguous_inputs() {
    std::string error;
    strata::metal::Geometry parsed;
    check(!parse_geometry("strata-geometry 2\n", parsed, error) &&
              error == "unsupported geometry file version 2",
          "unknown geometry version is rejected precisely");

    auto geometry = q2_geometry();
    std::stringstream encoded;
    check(strata::metal::write_geometry(encoded, geometry, error), "prepare malformed geometry fixture");
    std::string text = encoded.str();
    const std::size_t types_at = text.find("types=q2_0,bf16,f16,f32\n");
    text.replace(types_at, std::string("types=q2_0,bf16,f16,f32\n").size(), "types=q2_0,\n");
    check(!parse_geometry(text, parsed, error),
          "empty trailing tensor type is rejected");

    text = encoded.str();
    const std::size_t layer_at = text.find("layer 0 ");
    text.replace(layer_at, std::string("layer 0 ").size(), "layer 48 ");
    check(!parse_geometry(text, parsed, error),
          "out-of-range and missing layer is rejected");

    text = encoded.str() + "shardx_path=value\n";
    check(!parse_geometry(text, parsed, error),
          "unknown shard record is rejected");

    text = encoded.str() + "shard1_name=duplicate.gguf\n";
    check(!parse_geometry(text, parsed, error) &&
              error.find("duplicate geometry key shard1_name at line ") == 0,
          "duplicate shard records report their key and line");

    text = encoded.str() + "shard01_name=alias.gguf\nshard01_bytes=1\n";
    check(!parse_geometry(text, parsed, error) &&
              error.find("non-canonical geometry shard name key at line ") == 0,
          "zero-padded shard IDs cannot alias and hide duplicate records");
    text = encoded.str() + "shard01_bytes=1\n";
    check(!parse_geometry(text, parsed, error) &&
              error.find("non-canonical geometry shard size key at line ") == 0,
          "zero-padded shard size keys are rejected independently");

    text = encoded.str() + "types=q2_0,bf16,f16,f32\n";
    check(!parse_geometry(text, parsed, error) &&
              error.find("duplicate geometry key types at line ") == 0,
          "duplicate ordinary keys report their key and line");

    text = encoded.str();
    text.replace(text.find("shard2_name="), std::string("shard2_name=").size(), "shard3_name=");
    check(!parse_geometry(text, parsed, error),
          "non-consecutive shard records are rejected");

    text = encoded.str();
    text.append("invalid=");
    text.push_back(static_cast<char>(0xc0));
    text.push_back(static_cast<char>(0xaf));
    text.push_back('\n');
    check(!parse_geometry(text, parsed, error) && error.find("is not valid UTF-8") != std::string::npos,
          "malformed UTF-8 geometry input is rejected");

    text = encoded.str();
    const std::size_t type_line = text.find("types=q2_0,bf16,f16,f32\n");
    text.replace(type_line, std::string("types=q2_0,bf16,f16,f32\n").size(), "types=bf16,f16,f32\n");
    check(!parse_geometry(text, parsed, error),
          "layer tensor types absent from the type inventory are rejected");

    auto invalid = q2_geometry();
    invalid.types.erase(invalid.types.begin());
    check(!strata::metal::validate_geometry(invalid, error) &&
              error == "geometry layer type is missing from the types list",
          "direct geometry inputs receive the same full validation as parsed files");
    invalid = q2_geometry();
    invalid.model = std::string("bad-") + static_cast<char>(0xed) + static_cast<char>(0xa0) +
                    static_cast<char>(0x80);
    check(!strata::metal::validate_geometry(invalid, error),
          "direct geometry values reject malformed UTF-8");
    check(strata::metal::validate_geometry(q2_geometry(), error) && error.empty(),
          "successful geometry validation clears a previous error");
    strata::metal::PlanResult plan;
    auto options = base_options();
    check(!strata::metal::make_plan(invalid, options, plan, error),
          "planner rejects directly constructed geometry that violates the file contract");
}

void prefill_formula() {
    constexpr std::uint32_t chunks[] = {1'024, 2'048, 4'096, 6'144, 8'192, 16'384, 32'768};
    constexpr std::uint32_t contexts[] = {65'536, 262'144};
    for (const std::uint32_t chunk : chunks) {
        for (const std::uint32_t context : contexts) {
            const std::uint64_t fixed = 64ull * 1024 * 1024 + 32ull * 817'600 +
                1'024ull * (context / 4 + 2) + 48ull * 6'400 + 8ull * 512 + 65'536 + 4'096;
            const std::uint64_t tiles = (10ull * chunk + 63) / 64;
            const std::uint64_t regular = chunk * 267'304ull + 8ull * tiles + fixed;
            const std::uint64_t bf16x2 = chunk * 272'424ull + 8ull * tiles + fixed;
            check(strata::metal::prefill_bytes(chunk, context, false) == regular,
                  "prefill bytes match the documented formula for every candidate");
            check(strata::metal::prefill_bytes(chunk, context, true) == bf16x2,
                  "BF16X2 prefill bytes match every candidate and context");
        }
    }
}

void mtp_params_layout() {
    using namespace strata::metal;
    std::string error;
    MtpParamsLayout layout;
    check(make_mtp_params_layout(0, layout, error) &&
              layout.prefill_group_slots == 1 && layout.slot_count == 9 &&
              layout.allocation_bytes == 9 * kMtpParamsSlotBytes,
          "MTP parameter layout reserves a short-read group without GPU prefill");
    std::uint64_t offset = 0;
    check(layout.slot_offset(0, offset) && offset == 0 &&
              layout.slot_offset(7, offset) &&
              offset == 7 * kMtpParamsSlotBytes &&
              layout.slot_offset(8, offset) &&
              offset == 8 * kMtpParamsSlotBytes &&
              !layout.slot_offset(9, offset),
          "fixed and short-read MTP parameter slots have bounded 256-byte offsets");

    check(make_mtp_params_layout(1, layout, error) &&
              layout.prefill_group_slots == 1 && layout.slot_count == 9 &&
              layout.allocation_bytes == 9 * kMtpParamsSlotBytes,
          "one-row MTP prefill reserves one parameter group");
    check(make_mtp_params_layout(8, layout, error) &&
              layout.prefill_group_slots == 1 && layout.slot_count == 9,
          "an eight-row MTP prefill uses one parameter group");
    check(make_mtp_params_layout(9, layout, error) &&
              layout.prefill_group_slots == 2 && layout.slot_count == 10 &&
              layout.slot_offset(9, offset) &&
              offset == 9 * kMtpParamsSlotBytes,
          "MTP prefill groups round up at the eight-row boundary");
    check(make_mtp_params_layout(32'768, layout, error) &&
              layout.prefill_group_slots == 4'096 &&
              layout.slot_count == 4'104 &&
              layout.allocation_bytes == 4'104ull * kMtpParamsSlotBytes,
          "maximum automatic prefill reserves one slot for every group");
    check(!layout.slot_offset(layout.slot_count, offset),
          "MTP parameter layout rejects the first slot beyond its extent");
}

void full_plan() {
    auto geometry = q2_geometry();
    auto options = base_options();
    strata::metal::PlanResult plan;
    std::string error;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute full plan");
    check(error.empty() && plan.fits && plan.mode == "full", "large budget selects full mode");
    check(plan.experts_total == 24'576 && plan.experts_resident == 24'576,
          "full mode marks every expert resident");
    check(plan.prefill_chunk == 8'192, "full mode selects the largest fitting auto chunk");
    check(plan.kv_bytes == 26'368ull * 65'536 + 36'912,
          "main KV includes exact indexer overhead in its single allocation");
    check(plan.mtp_kv_bytes == 2'048ull * 65'536,
          "MTP KV reserves the drafter's fp16 K/V without unused indexer state");
    check(plan.mtp_params_bytes == 1'032ull * kMtpParamsSlotBytes,
          "MTP parameter memory covers fixed passes and every selected prefill group");
    check(plan.session_state_bytes == 118'038'528u &&
              plan.window_scratch_bytes == 47'911'680u,
          "planner reports exact Qwen4exp session and 65,536-cell Window scratch sizes");
    check(plan.session_state_bytes <= plan.fixed_bytes &&
              plan.window_scratch_bytes <=
                  plan.fixed_bytes - plan.session_state_bytes,
          "Qwen4exp fixed-state subtotal fits the unchanged fixed allowance");
    auto coder_geometry = q2_geometry();
    coder_geometry.family = "coder";
    coder_geometry.n_expert = 256;
    auto coder_options = options;
    coder_options.max_context = 32'768;
    strata::metal::PlanResult coder_plan;
    check(strata::metal::make_plan(coder_geometry, coder_options,
                                   coder_plan, error) &&
              coder_plan.session_state_bytes == 118'038'528u &&
              coder_plan.window_scratch_bytes == 47'641'344u &&
              coder_plan.fixed_bytes == plan.fixed_bytes,
          "Coder planner uses the exact 256-expert scratch while retaining the fixed allowance");
    check(coder_plan.session_state_bytes <= coder_plan.fixed_bytes &&
              coder_plan.window_scratch_bytes <=
                  coder_plan.fixed_bytes - coder_plan.session_state_bytes,
          "Coder fixed-state subtotal fits the unchanged fixed allowance");
    check(plan.fixed_bytes == kFixedAllowanceBytes &&
              coder_plan.fixed_bytes == kFixedAllowanceBytes,
          "Qwen and Coder plans retain the 256 MiB fixed admission charge");
    const std::uint64_t full_total = plan.dense_bytes + plan.unread_bytes +
        plan.pack_bytes + plan.mtp_bytes + plan.mtp_bind_bytes +
        plan.mtp_params_bytes + plan.fixed_bytes + plan.kv_bytes +
        plan.mtp_kv_bytes + plan.prefill_bytes + plan.experts_resident_bytes;
    check(plan.total_bytes == full_total,
          "full-mode total charges MTP parameter bytes exactly once");
    const std::string line = strata::metal::plan_line(plan);
    check(line.starts_with("PLAN fits=1 mode=full gpu=Apple_M5_Pro"), "PLAN prefix is stable");
    check(line.find("kv=fp16 context=65536") != std::string::npos, "PLAN reports KV mode and context");
    check(line.find("experts_total=24576 experts_total_mib=") != std::string::npos,
          "PLAN reports expert totals");
    check(line.find("mtp_params_mib=1") != std::string::npos,
          "PLAN reports the dedicated MTP parameter allocation");
    const std::size_t bind_key = line.find("mtp_bind_mib=");
    const std::size_t fixed_key = line.find("fixed_mib=");
    const std::size_t full_key = line.find("w_full_mib=");
    const std::size_t params_key = line.find("mtp_params_mib=");
    const std::size_t session_key = line.find("session_state_mib=");
    const std::size_t scratch_key = line.find("window_scratch_mib=");
    check(bind_key < fixed_key && full_key < params_key &&
              params_key < session_key && session_key < scratch_key &&
              line.ends_with("window_scratch_mib=46"),
          "PLAN preserves existing key order and appends the fixed-state breakdown");
    check(line.find("w_full_mib=") != std::string::npos && line.find("reason=") == std::string::npos,
          "fitting PLAN reports full threshold without a reason");
    check(plan.mtp_bind_bytes == 16'300'000,
          "default MTP window preserves the catalogue geometry bind baseline");
    check(plan.mtp_bind_capacity == 32'768 &&
              strata::metal::validate_mtp_memory_plan(geometry, plan, error),
          "planned MTP bind capacity and bytes validate against geometry");
    auto mismatched_mtp_params = plan;
    ++mismatched_mtp_params.mtp_params_bytes;
    check(!strata::metal::validate_mtp_memory_plan(
              geometry, mismatched_mtp_params, error) &&
              error.find("MTP parameter bytes do not match") != std::string::npos,
          "MTP runtime validation rejects a parameter-array size mismatch");

    auto no_prefill_options = options;
    no_prefill_options.prefill_mode = strata::metal::PrefillMode::None;
    check(strata::metal::make_plan(geometry, no_prefill_options, plan, error) &&
              plan.prefill_chunk == 0 &&
              plan.mtp_params_bytes ==
                  (strata::metal::kMtpParamsFixedSlots + 1ull) *
                      strata::metal::kMtpParamsSlotBytes,
          "speculation without GPU prefill reserves one short-read group");

    options.mtp_window = 65'536;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_bytes == 29'112'288 &&
              plan.mtp_bind_capacity == 65'536,
          "MTP bind scratch grows with a 65,536-cell window");
    options.max_context = 262'144;
    options.mtp_window = 262'144;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_bytes == 105'986'016,
          "a full-context MTP window accounts for the 262,144-cell scratch");
    options.mtp_window = 0;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_bytes == 105'986'016 &&
              plan.mtp_bind_capacity == 262'144,
          "zero MTP window uses the full planned context");
    options.mtp_window = 32'768;
    options.max_context = 8'192;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_bytes < 16'300'000,
          "the MTP scratch window is capped by a smaller planned context");
    options.mtp_window = 8;
    options.max_context = 13;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_capacity == 64 &&
              strata::metal::validate_mtp_memory_plan(geometry, plan, error),
          "rounded MTP bind capacity fits a non-aligned context");
    auto oversized_mtp_capacity = plan;
    oversized_mtp_capacity.mtp_bind_capacity = 128;
    check(!strata::metal::validate_mtp_memory_plan(
              geometry, oversized_mtp_capacity, error) &&
              error.find("exceeds the rounded planned context") != std::string::npos,
          "MTP plan rejects bind capacity beyond rounded context");
    options.mtp_window = 32'768;
    options.max_context = 65'536;
    options.mtp_window = 262'145;
    options.max_context = 262'144;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.mtp_bind_bytes == 105'986'016,
          "an MTP window beyond context uses the full-context allocation");
    options.mtp_window = 32'768;
    options.max_context = 65'536;

    options.spec_enabled = false;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute non-speculative plan");
    check(plan.fits && plan.mtp_bytes == 0 && plan.mtp_bind_bytes == 0 &&
              plan.mtp_params_bytes == 0 && plan.mtp_kv_bytes == 0,
          "MTP memory is omitted when speculation is disabled");
    check(plan.mtp_bind_capacity == 0,
          "MTP bind capacity is omitted when speculation is disabled");

    geometry.wrapped_unread_bytes = 512ull * 1024 * 1024;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute full plan with wrapped unread ranges");
    options.ram_bytes = plan.total_bytes - plan.unread_bytes + 6ull * 1024 * 1024 * 1024 + 1;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "recompute physical RAM threshold without wrapped unread bytes");
    check(plan.ram_bytes_used == plan.total_bytes - plan.unread_bytes && !plan.ram_warning,
          "full-mode RAM accounting excludes unread mapped ranges");
}

void auto_prefill_candidate_caps() {
    auto geometry = q2_geometry();
    auto options = base_options();
    options.max_context = 65'536;
    options.prefill_mode = strata::metal::PrefillMode::Auto;
    strata::metal::PlanResult plan;
    std::string error;
    const std::pair<std::uint32_t, std::uint32_t> caps_and_chunks[] = {
        {1'024, 1'024}, {2'048, 2'048}, {4'096, 4'096}, {6'144, 6'144},
        {8'192, 8'192}, {16'384, 16'384}, {32'768, 32'768}};
    for (const auto& [cap, expected] : caps_and_chunks) {
        options.prefill_auto_max = cap;
        check(strata::metal::make_plan(geometry, options, plan, error),
              "compute automatic prefill candidate");
        check(plan.fits && plan.mode == "full" && plan.prefill_chunk == expected,
              "auto:N selects the largest permitted candidate that fits");
    }
    options.expert_cache = 1;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute explicit-cache plan above its memory-only full threshold");
    const std::uint64_t always = plan.dense_bytes + plan.pack_bytes + plan.mtp_bytes +
        plan.mtp_bind_bytes + 9ull * kMtpParamsSlotBytes + plan.fixed_bytes +
        plan.kv_bytes + plan.mtp_kv_bytes;
    const std::uint64_t memory_full = plan.reserve_bytes + always + geometry.wrapped_unread_bytes +
        strata::metal::prefill_bytes(1'024, options.max_context, options.prefill_bf16x2) +
        (1'024 / kMtpParamsPrefillGroupRows - 1) * kMtpParamsSlotBytes +
        plan.experts_total_bytes;
    check(plan.fits && plan.mode == "tiered" && plan.w_full_bytes == memory_full &&
              plan.w_full_bytes < plan.w_bytes,
          "w_full reports the all-experts memory threshold even when an explicit count cap forces tiered mode");
}

void kv_plan_matches_storage_layout() {
    using namespace strata::metal;
    const auto geometry = q2_geometry();
    auto options = base_options();
    options.max_context = 65;
    options.prefill_mode = PrefillMode::None;
    constexpr std::array formats{
        KvFormat::Fp16, KvFormat::Int8, KvFormat::Q4_0, KvFormat::K8v4};
    for (const KvFormat format : formats) {
        options.kv = format;
        PlanResult plan;
        QsaKvLayout layout;
        MtpKvLayout mtp_layout;
        std::string error;
        const bool layout_ok = make_qsa_kv_layout(options.max_context, format, layout, error);
        check(layout_ok, "compute exact QSA cache layout for planner comparison");
        const bool mtp_layout_ok = make_mtp_kv_layout(options.max_context, format, mtp_layout, error);
        check(mtp_layout_ok, "compute exact MTP drafter KV layout for planner comparison");
        const bool plan_ok = make_plan(geometry, options, plan, error);
        check(plan_ok, "compute plan for non-four-aligned context");
        if (layout_ok && mtp_layout_ok && plan_ok) {
            check(plan.kv_bytes == layout.allocation_bytes &&
                      layout.cell_capacity == 68,
                  "planner reserves the rounded, format-specific QSA allocation size");
            check(plan.mtp_kv_bytes == mtp_layout.allocation_bytes &&
                      mtp_layout.cell_capacity == layout.cell_capacity,
                  "planner reserves the rounded, format-specific MTP drafter K/V layout");
        }
    }
}

void tiered_plan_and_residency_controls() {
    auto geometry = q2_geometry();
    auto options = base_options();
    options.working_set_bytes = 36ull * 1024 * 1024 * 1024;
    options.max_context = 32'768;
    options.kv = strata::metal::KvFormat::K8v4;
    options.ram_bytes = 48ull * 1024 * 1024 * 1024;
    options.available_bytes = 40ull * 1024 * 1024 * 1024;
    options.prefill_auto_max = 8'192;
    strata::metal::PlanResult plan;
    std::string error;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute tiered plan");
    check(error.empty() && plan.fits && plan.mode == "tiered", "reduced budget selects tiered mode");
    check(plan.experts_resident > 0 && plan.experts_resident < plan.experts_total,
          "tiered plan admits only the fitting experts");
    check(plan.prefill_chunk == 2'048, "tiered auto mode uses the 24 GiB threshold");
    check(plan.slots > 0 && plan.slot_bytes % (16 * 1024) == 0, "slot pool uses page-rounded slots");
    check(plan.experts_total - plan.experts_resident > 0, "tiered PLAN reports cold experts");
    check(plan.anonymous_bytes_used == plan.ram_bytes_used - plan.mtp_bytes,
          "tiered available-memory estimate excludes file-backed MTP weights");
    strata::metal::MtpParamsLayout tiered_params;
    check(make_mtp_params_layout(plan.prefill_chunk, tiered_params, error) &&
              plan.mtp_params_bytes == tiered_params.allocation_bytes,
          "tiered plans size MTP parameter groups to the selected prefill chunk");
    const std::uint64_t tiered_total = plan.dense_bytes + plan.pack_bytes +
        plan.mtp_bytes + plan.mtp_bind_bytes + plan.mtp_params_bytes +
        plan.fixed_bytes + plan.kv_bytes + plan.mtp_kv_bytes +
        plan.prefill_bytes + plan.slot_bytes + plan.experts_resident_bytes;
    check(plan.total_bytes == tiered_total,
          "tiered total charges MTP parameter bytes exactly once");
    check(!plan.warnings.empty() && plan.warnings.front().starts_with("no expert profile;"),
          "tiered admission without a ranking emits the documented warning");

    options.prefill_auto_max = 1'500;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute tiered plan with a non-candidate prefill cap");
    check(plan.prefill_chunk == 1'024,
          "tiered auto mode chooses a documented candidate below a non-candidate cap");
    options.prefill_auto_max = 8'192;

    options.available_bytes = 1;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute tiered plan under available-memory pressure");
    check(plan.ram_warning && std::any_of(plan.warnings.begin(), plan.warnings.end(),
              [](const std::string& warning) {
                  return warning.find("MiB planned, ") != std::string::npos &&
                         warning.find("MiB available") != std::string::npos;
              }),
          "available-memory warning reports planned and available MiB");
    options.available_bytes = 40ull * 1024 * 1024 * 1024;

    options.expert_cache = 0;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute zero-resident plan");
    check(plan.fits && plan.experts_resident == 0, "zero expert cache keeps all experts cold");
    check(plan.slots == 1'024, "empty resident set still reserves two complete layer halves");

    options.expert_cache = -1;
    options.resident_budget_is_set = true;
    options.resident_budget_bytes = 0;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute explicit zero-byte resident budget");
    check(plan.fits && plan.experts_resident == 0,
          "an explicit zero resident budget differs from an omitted cap");
    options.resident_budget_bytes = 10 * geometry.layers[0].blob_bytes;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute positive byte-capped resident set");
    check(plan.experts_resident_bytes <= options.resident_budget_bytes,
          "resident budget caps admitted bytes");
    options.resident_budget_is_set = false;

    options.expert_cache = 1'000;
    options.expert_cache_per_layer = true;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute per-layer quota plan");
    check(plan.experts_resident <= 960, "per-layer quota uses floor of the first pass count");
    std::vector<std::uint64_t> per_layer(geometry.n_layers, 0);
    for (const auto& [layer, expert] : plan.resident_order) {
        (void)expert;
        ++per_layer[layer];
    }
    check(*std::max_element(per_layer.begin(), per_layer.end()) <= 20,
          "per-layer plan enforces its resident quota");
}

strata::metal::Geometry compact_geometry(std::uint64_t default_blob = 8'192) {
    auto geometry = q2_geometry();
    geometry.family = "synthetic";
    geometry.shards = {{"compact.gguf", 1}};
    geometry.n_expert = 2;
    geometry.nonexpert_bytes = 1ull * 1024 * 1024;
    geometry.wrapped_unread_bytes = 0;
    geometry.mtp_bytes = 0;
    geometry.mtp_bind_bytes = 0;
    geometry.n_qsa_layers = 1;
    geometry.n_kv_head = 1;
    geometry.head_dim = 64;
    geometry.n_embd = 64;
    geometry.vocab = 128;
    geometry.types = {"q2_0"};
    geometry.layers.assign(48, {default_blob, "q2_0", "q2_0"});
    return geometry;
}

void byte_fit_and_display_rounding() {
    auto geometry = q2_geometry();
    auto options = base_options();
    options.prefill_mode = strata::metal::PrefillMode::None;
    options.expert_cache = 0;
    strata::metal::PlanResult plan;
    std::string error;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute exact tiered empty-resident requirement");
    options.working_set_bytes = options.reserve_mib * 1024ull * 1024 + plan.total_bytes;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "plan exactly at the byte-accurate fit boundary");
    check(plan.fits && plan.free_bytes == 0,
          "fit at exact byte boundary succeeds with zero byte-level free space");
    check(strata::metal::plan_line(plan).find("free_mib=-") != std::string::npos,
          "component-wise MiB rounding may show negative free_mib on a byte-exact fit");
}

void count_cap_and_zero_per_layer_quota() {
    std::string error;
    strata::metal::PlanResult empty;
    auto geometry = compact_geometry();
    auto options = base_options();
    options.spec_enabled = false;
    options.prefill_mode = strata::metal::PrefillMode::None;
    geometry.layers[0].blob_bytes = 64 * 1024;
    options.expert_cache = 0;
    check(strata::metal::make_plan(geometry, options, empty, error),
          "compute compact empty-resident baseline");

    options.expert_order = {{0, 0}};
    options.expert_cache = 1;
    options.working_set_bytes = options.reserve_mib * 1024ull * 1024 +
                                empty.total_bytes + 16 * 1024;
    strata::metal::PlanResult limited;
    check(strata::metal::make_plan(geometry, options, limited, error),
          "compute expert-count-capped plan");
    check(limited.fits && limited.experts_resident == 1 &&
              limited.resident_order == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{1, 0}},
          "expert-cache N caps count even when later small experts fit under its byte cap");

    geometry = compact_geometry();
    options.expert_order.clear();
    options.expert_cache = 0;
    check(strata::metal::make_plan(geometry, options, empty, error),
          "recompute baseline for per-layer quota");
    options.expert_cache = 47;
    options.expert_cache_per_layer = true;
    options.working_set_bytes = options.reserve_mib * 1024ull * 1024 +
                                empty.total_bytes + 47 * 8'192ull;
    check(strata::metal::make_plan(geometry, options, limited, error),
          "compute sub-48 per-layer quota plan");
    check(limited.experts_resident == 0,
          "literal floor(first-pass-count / 48) quota of zero admits no experts");
}

void prefill_validation_and_bf16x2() {
    auto geometry = q2_geometry();
    auto options = base_options();
    options.prefill_auto_max = 512;
    strata::metal::PlanResult plan;
    std::string error;
    check(!strata::metal::make_plan(geometry, options, plan, error) &&
              error == "automatic prefill maximum must be at least 1024",
          "automatic prefill cannot select a chunk below 1024");

    options = base_options();
    options.prefill_bf16x2 = true;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "compute BF16X2 prefill plan");
    check(plan.prefill_bytes == strata::metal::prefill_bytes(
                                    plan.prefill_chunk, options.max_context, true),
          "planner uses the BF16X2 prefill allocation formula");

    options = base_options();
    options.max_context = 262'144;
    options.prefill_mode = strata::metal::PrefillMode::Fixed;
    options.prefill_chunk = strata::metal::kPrefillMaxShaderRows;
    check(strata::metal::make_plan(geometry, options, plan, error) &&
              plan.prefill_chunk == strata::metal::kPrefillMaxShaderRows &&
              plan.prefill_bytes == strata::metal::prefill_bytes(
                                        strata::metal::kPrefillMaxShaderRows,
                                        options.max_context, options.prefill_bf16x2),
          "fixed prefill accepts the exact inclusive shader row limit");

    options.prefill_chunk = strata::metal::kPrefillMaxShaderRows + 1u;
    check(!strata::metal::make_plan(geometry, options, plan, error) &&
              error == "fixed prefill chunk exceeds the Metal shader row limit",
          "fixed prefill rejects a chunk above the embedding/broadcast shader limit");
}

void expert_profile_reader() {
    const auto path = std::filesystem::temp_directory_path() / "strata-metal-profile-test.bin";
    auto append_u32 = [](std::vector<unsigned char>& bytes, std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xff));
        }
    };
    auto append_u16 = [](std::vector<unsigned char>& bytes, std::uint16_t value) {
        bytes.push_back(static_cast<unsigned char>(value & 0xff));
        bytes.push_back(static_cast<unsigned char>((value >> 8) & 0xff));
    };
    auto make_profile = [&](const std::vector<std::pair<std::uint16_t, std::uint16_t>>& pairs) {
        std::vector<unsigned char> bytes = {'S', 'T', 'R', 'P'};
        append_u32(bytes, 1);  // version
        append_u32(bytes, 2);  // layers
        append_u32(bytes, 3);  // experts per layer
        append_u32(bytes, 2);  // slots
        append_u32(bytes, static_cast<std::uint32_t>(pairs.size()));
        for (const auto& [layer, expert] : pairs) {
            append_u16(bytes, layer);
            append_u16(bytes, expert);
        }
        for (int i = 0; i < 6; ++i) append_u32(bytes, 0);  // frequency table
        return bytes;
    };
    auto write_profile = [&](const std::vector<unsigned char>& bytes) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(output);
    };

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranked;
    std::string error;
    const auto valid = make_profile({{1, 2}});
    check(write_profile(valid), "write a compact STRP profile fixture");
    check(strata::metal::read_expert_profile(path.string(), 2, 3, ranked, error) &&
              ranked == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{1, 2}},
          "read ranked (layer, expert) pairs from STRP");
    check(!strata::metal::read_expert_profile(path.string(), 48, 512, ranked, error),
          "reject an expert profile for different model dimensions");

    const auto duplicate = make_profile({{1, 2}, {1, 2}});
    check(write_profile(duplicate), "write duplicate STRP fixture");
    check(!strata::metal::read_expert_profile(path.string(), 2, 3, ranked, error) &&
              error.find("duplicate") != std::string::npos,
          "reject duplicate expert-profile pairs");

    auto truncated = valid;
    truncated.pop_back();
    check(write_profile(truncated), "write truncated STRP fixture");
    check(!strata::metal::read_expert_profile(path.string(), 2, 3, ranked, error) &&
              error.find("invalid length") != std::string::npos,
          "reject a truncated STRP frequency table");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

void no_fit_and_unsupported_types() {
    auto geometry = q2_geometry();
    auto options = base_options();
    options.working_set_bytes = 8ull * 1024 * 1024 * 1024;
    options.max_context = 262'144;
    options.prefill_mode = strata::metal::PrefillMode::None;
    strata::metal::PlanResult plan;
    std::string error;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute no-fit plan");
    check(!plan.fits && plan.reason.starts_with("the_model_does_not_fit:_needs_"),
          "no-fit plan explains the minimum tiered requirement");
    check(plan.kv_warning, "large KV usage produces the documented no-fit advisory");
    const std::string line = strata::metal::plan_line(plan);
    check(line.find("fits=0") != std::string::npos,
          "no-fit plan is still emitted as a PLAN line");
    check(line.find("reason=") < line.find("mtp_params_mib=") &&
              line.ends_with("window_scratch_mib=48"),
          "PLAN appends fixed-state breakdown keys after the optional no-fit reason");

    geometry.types.push_back("mx_unknown");
    options.working_set_bytes = 64ull * 1024 * 1024 * 1024;
    check(strata::metal::make_plan(geometry, options, plan, error), "compute unsupported-type plan");
    check(!plan.fits && plan.reason == "unsupported_type_mx_unknown",
          "unsupported tensor type takes precedence over memory fit");

    geometry.layers[0].blob_bytes = UINT64_MAX;
    check(strata::metal::make_plan(geometry, options, plan, error),
          "unsupported types are checked before expert byte arithmetic");
    check(!plan.fits && plan.reason == "unsupported_type_mx_unknown",
          "unsupported type diagnostic survives otherwise overflowing expert sizes");
}

void qwen_plan_requires_runtime_geometry() {
    auto geometry = q2_geometry();
    geometry.n_embd -= 1;
    auto options = base_options();
    strata::metal::PlanResult plan;
    std::string error;
    check(!strata::metal::make_plan(geometry, options, plan, error) &&
              error.find("supported 48-layer runtime geometry") != std::string::npos,
          "Qwen catalogue plans must reject geometry outside the fixed runtime ABI");
}

}  // namespace

int main() {
    geometry_round_trip();
    geometry_rejects_ambiguous_inputs();
    prefill_formula();
    mtp_params_layout();
    full_plan();
    kv_plan_matches_storage_layout();
    tiered_plan_and_residency_controls();
    auto_prefill_candidate_caps();
    byte_fit_and_display_rounding();
    count_cap_and_zero_per_layer_quota();
    prefill_validation_and_bf16x2();
    expert_profile_reader();
    no_fit_and_unsupported_types();
    qwen_plan_requires_runtime_geometry();
    if (failures != 0) {
        std::cerr << failures << " plan/geometry checks failed\n";
        return 1;
    }
    std::cout << "geometry v1 and memory planner checks passed\n";
    return 0;
}
