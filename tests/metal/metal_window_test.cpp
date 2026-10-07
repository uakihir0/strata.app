// tests/metal/metal_window_test.cpp - device-free window planning and model-header fixtures.
#include "strata/metal/window.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using strata::metal::FileWindowPlan;
using strata::metal::FullModeLayout;
using strata::metal::WindowFileSource;
using strata::metal::WindowPlan;
using strata::metal::WindowSources;
using strata::metal::WindowTensorSource;
using strata::metal::kFullModeExpertRoles;
using strata::metal::kFullModeMaxExperts;
using strata::metal::kFullModeMaxLayers;
using strata::metal::kNoExpertIndex;
using strata::metal::kWindowMergeGapBytes;
using strata::metal::kWindowPageBytes;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void skip_space(std::string_view json, std::size_t& position) {
    while (position < json.size() &&
           (json[position] == ' ' || json[position] == '\t' ||
            json[position] == '\r' || json[position] == '\n')) {
        ++position;
    }
}

bool parse_string(std::string_view json, std::size_t& position, std::string& result) {
    if (position >= json.size() || json[position] != '"') return false;
    ++position;
    result.clear();
    while (position < json.size()) {
        const char c = json[position++];
        if (c == '"') return true;
        if (c != '\\') {
            result.push_back(c);
            continue;
        }
        if (position >= json.size()) return false;
        const char escaped = json[position++];
        switch (escaped) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u':
                // Fixture tensor names are ASCII. Reject escaped non-ASCII rather than misdecode it.
                if (json.size() - position < 4) return false;
                for (std::size_t i = 0; i < 4; ++i) {
                    const char digit = json[position + i];
                    if (!((digit >= '0' && digit <= '9') ||
                          (digit >= 'a' && digit <= 'f') || (digit >= 'A' && digit <= 'F'))) {
                        return false;
                    }
                }
                if (json.substr(position, 4) != "0000") return false;
                position += 4;
                break;
            default: return false;
        }
    }
    return false;
}

bool skip_value(std::string_view json, std::size_t& position) {
    skip_space(json, position);
    if (position >= json.size()) return false;
    if (json[position] == '"') {
        std::string ignored;
        return parse_string(json, position, ignored);
    }
    if (json[position] == '[' || json[position] == '{') {
        const char open = json[position++];
        const char close = open == '[' ? ']' : '}';
        skip_space(json, position);
        if (position < json.size() && json[position] == close) {
            ++position;
            return true;
        }
        while (position < json.size()) {
            if (open == '{') {
                std::string ignored;
                if (!parse_string(json, position, ignored)) return false;
                skip_space(json, position);
                if (position >= json.size() || json[position++] != ':') return false;
            }
            if (!skip_value(json, position)) return false;
            skip_space(json, position);
            if (position >= json.size()) return false;
            if (json[position] == close) {
                ++position;
                return true;
            }
            if (json[position++] != ',') return false;
            skip_space(json, position);
        }
        return false;
    }
    const std::size_t start = position;
    while (position < json.size() && json[position] != ',' && json[position] != ']' &&
           json[position] != '}' && json[position] != ' ' && json[position] != '\t' &&
           json[position] != '\r' && json[position] != '\n') {
        ++position;
    }
    return position > start;
}

bool parse_u64(std::string_view json, std::size_t& position, std::uint64_t& result) {
    const std::size_t start = position;
    while (position < json.size() && json[position] >= '0' && json[position] <= '9') ++position;
    if (position == start) return false;
    const auto parsed = std::from_chars(json.data() + start, json.data() + position, result);
    return parsed.ec == std::errc{} && parsed.ptr == json.data() + position;
}

bool parse_tensor_object(std::string_view json, std::size_t& position,
                         std::string& name, std::uint64_t& offset, std::uint64_t& bytes) {
    if (position >= json.size() || json[position++] != '{') return false;
    bool have_name = false;
    bool have_offset = false;
    bool have_bytes = false;
    while (position < json.size()) {
        skip_space(json, position);
        if (position < json.size() && json[position] == '}') {
            ++position;
            return have_name && have_offset && have_bytes;
        }
        std::string key;
        if (!parse_string(json, position, key)) return false;
        skip_space(json, position);
        if (position >= json.size() || json[position++] != ':') return false;
        skip_space(json, position);
        if (key == "name") {
            if (have_name || !parse_string(json, position, name)) return false;
            have_name = true;
        } else if (key == "offset") {
            if (have_offset || !parse_u64(json, position, offset)) return false;
            have_offset = true;
        } else if (key == "bytes") {
            if (have_bytes || !parse_u64(json, position, bytes)) return false;
            have_bytes = true;
        } else if (!skip_value(json, position)) {
            return false;
        }
        skip_space(json, position);
        if (position >= json.size()) return false;
        if (json[position] == '}') {
            ++position;
            return have_name && have_offset && have_bytes;
        }
        if (json[position++] != ',') return false;
    }
    return false;
}

std::vector<WindowTensorSource> read_fixture_tensors(const fs::path& path,
                                                     std::uint64_t data_start,
                                                     bool is_swift,
                                                     std::uint64_t& file_size,
                                                     std::uint64_t* ple_offset,
                                                     std::uint64_t* ple_bytes) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "could not open header fixture " + path.string());
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::size_t table_key = json.find("\"tensors\"");
    require(table_key != std::string::npos, "fixture lacks a tensor table: " + path.string());
    const std::size_t table_start = json.find('[', table_key);
    require(table_start != std::string::npos, "fixture tensor table is malformed: " + path.string());

    std::vector<WindowTensorSource> tensors;
    file_size = data_start;
    std::size_t position = table_start + 1;
    while (position < json.size()) {
        skip_space(json, position);
        if (position < json.size() && json[position] == ']') break;
        std::string name;
        std::uint64_t offset = 0;
        std::uint64_t bytes = 0;
        require(parse_tensor_object(json, position, name, offset, bytes),
                "malformed tensor object in " + path.string());
        require(bytes > 0 && offset <= std::numeric_limits<std::uint64_t>::max() - data_start &&
                    bytes <= std::numeric_limits<std::uint64_t>::max() - data_start - offset,
                "fixture tensor range overflows");
        const std::uint64_t file_offset = data_start + offset;
        file_size = std::max(file_size, file_offset + bytes);
        const bool ple = name == "per_layer_token_embd.weight";
        if (ple) {
            if (ple_offset != nullptr) *ple_offset = file_offset;
            if (ple_bytes != nullptr) *ple_bytes = bytes;
        }
        const bool converted_router = is_swift &&
            (name.ends_with(".ffn_gate_inp.weight") ||
             name.ends_with(".ffn_gate_inp_shexp.weight"));
        if (!ple && !converted_router) {
            const bool expert = name.find(".ffn_gate_exps.weight") != std::string::npos ||
                                name.find(".ffn_up_exps.weight") != std::string::npos ||
                                name.find(".ffn_down_exps.weight") != std::string::npos;
            require(!expert || bytes % 512 == 0, "fixture expert payload is not 512-way divisible");
            tensors.push_back({name, file_offset, bytes, expert ? bytes / 512 : 0});
        }
        skip_space(json, position);
        if (position < json.size() && json[position] == ',') ++position;
    }
    return tensors;
}

WindowFileSource make_fixture(const std::string& filename, std::uint64_t data_start, bool swift,
                              std::uint64_t* ple_offset = nullptr,
                              std::uint64_t* ple_bytes = nullptr) {
    WindowFileSource file;
    file.path = filename;
    file.tensors = read_fixture_tensors(fs::path(STRATA_MODEL_HEADER_DIR) / filename,
                                        data_start, swift, file.file_size, ple_offset, ple_bytes);
    return file;
}

void test_rounding_and_merge_boundaries() {
    WindowSources sources;
    WindowFileSource file{"synthetic.gguf", 3 * kWindowMergeGapBytes, {}, {}};
    file.tensors = {
        {"first", 7, 100, 0},
        {"joined", 107 + kWindowMergeGapBytes - 1, 100, 0},
        {"separate", 206 + 2 * kWindowMergeGapBytes, 100, 0},
    };
    sources.files.push_back(std::move(file));

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 4 * kWindowMergeGapBytes, plan, error),
            "merge plan failed: " + error);
    require(plan.files.size() == 1 && plan.files[0].windows.size() == 2,
            "a gap below 256 MiB must merge and a gap of exactly 256 MiB must split");
    require(plan.files[0].windows[0].start == 0 &&
                plan.files[0].windows[0].start % kWindowPageBytes == 0 &&
                plan.files[0].windows[0].end % kWindowPageBytes == 0,
            "window bounds must round outward to 16 KiB");
    require(plan.files[0].slices.size() == 3 &&
                plan.files[0].slices[0].window_index == plan.files[0].slices[1].window_index &&
                plan.files[0].slices[1].window_index != plan.files[0].slices[2].window_index,
            "tensor slices do not follow the strict merge rule");

    std::uint64_t unread = 0;
    require(strata::metal::compute_wrapped_unread(sources, unread, error),
            "wrapped-unread calculation failed: " + error);
    require(unread == plan.wrapped_unread_bytes,
            "the plan and standalone wrapped-unread total disagree");
}

void test_expert_boundary_splitting() {
    WindowSources sources;
    WindowFileSource file{"experts.gguf", 32 * kWindowPageBytes, {}, {}};
    file.tensors.push_back({"blk.0.ffn_gate_exps.weight", 0, 6 * 2 * kWindowPageBytes,
                            2 * kWindowPageBytes});
    sources.files.push_back(std::move(file));

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 4 * kWindowPageBytes, plan, error),
            "expert split plan failed: " + error);
    const FileWindowPlan& result = plan.files.front();
    require(result.windows.size() == 3 && result.slices.size() == 6,
            "six slabs should split into three two-slab windows");
    for (std::size_t i = 0; i < result.slices.size(); ++i) {
        require(result.slices[i].expert_index == i &&
                    result.slices[i].window_index == i / 2 &&
                    result.slices[i].byte_count == 2 * kWindowPageBytes,
                "expert index, slab size or split boundary is incorrect");
    }
    for (const auto& window : result.windows) {
        require(window.end - window.start == 4 * kWindowPageBytes,
                "expert window must not exceed the forced maxBufferLength");
    }
    const auto original_unread = plan.wrapped_unread_bytes;
    require(strata::metal::plan_windows(sources, 12 * kWindowPageBytes, plan, error),
            "large-buffer expert plan failed: " + error);
    require(plan.wrapped_unread_bytes == original_unread && plan.files.front().windows.size() == 1,
            "wrapped_unread must not depend on maxBufferLength");
}

void test_unaligned_expert_splits() {
    WindowSources sources;
    WindowFileSource file{"unaligned-experts.gguf", 4 * kWindowPageBytes, {}, {}};
    const std::uint64_t slab_bytes = kWindowPageBytes + 1;
    file.tensors.push_back({"blk.0.ffn_up_exps.weight", 7, 3 * slab_bytes, slab_bytes});
    sources.files.push_back(std::move(file));

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 2 * kWindowPageBytes, plan, error),
            "unaligned expert split plan failed: " + error);
    const FileWindowPlan& result = plan.files.front();
    require(result.windows.size() == 3 && result.slices.size() == 3,
            "unaligned expert slabs should split independently under the forced limit");
    const std::uint64_t expected_offsets[] = {7, 8, 9};
    for (std::size_t i = 0; i < result.slices.size(); ++i) {
        const auto& slice = result.slices[i];
        const auto& window = result.windows[i];
        require(slice.expert_index == i && slice.window_index == i &&
                    slice.byte_count == slab_bytes &&
                    slice.window_offset == expected_offsets[i] &&
                    slice.file_offset == window.start + slice.window_offset &&
                    window.start % kWindowPageBytes == 0 &&
                    window.end % kWindowPageBytes == 0 &&
                    window.end - window.start == 2 * kWindowPageBytes,
                "unaligned expert split lost its slab boundary or page-rounded mapping");
    }
}

void test_merged_gap_coverage_when_forced_to_split() {
    WindowSources sources;
    WindowFileSource file{"merged-gap.gguf", 8 * kWindowPageBytes, {}, {}};
    file.tensors = {{"before-gap", 0, kWindowPageBytes, 0},
                    {"after-gap", 6 * kWindowPageBytes, kWindowPageBytes, 0}};
    sources.files.push_back(std::move(file));

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 2 * kWindowPageBytes, plan, error),
            "forced split across a merged gap failed: " + error);
    const auto& result = plan.files.front();
    require(result.windows.size() == 5 && result.slices.size() == 2,
            "the merged range should be tiled by bounded windows across its unread gap");
    std::uint64_t covered_end = result.windows.front().start;
    for (const auto& window : result.windows) {
        require(window.start <= covered_end && window.end - window.start <= 2 * kWindowPageBytes,
                "forced-split windows leave a gap or exceed maxBufferLength");
        covered_end = std::max(covered_end, window.end);
    }
    require(result.windows.front().start == 0 && covered_end == 7 * kWindowPageBytes &&
                result.slices[0].window_index == 0 &&
                result.slices[1].window_index == 4 &&
                plan.wrapped_unread_bytes == 5 * kWindowPageBytes,
            "forced split must preserve the complete pre-split merged range and unread total");
}

void test_invalid_sources() {
    WindowPlan plan;
    std::string error;
    WindowSources sources{};
    sources.files.push_back({"small.gguf", 4 * kWindowPageBytes,
                             {{"tensor", 0, 3 * kWindowPageBytes, 0}}, {}});
    require(!strata::metal::plan_windows(sources, 2 * kWindowPageBytes, plan, error) &&
                error.find("exceeds maxBufferLength") != std::string::npos,
            "an indivisible tensor larger than maxBufferLength must fail");
    sources.files[0].tensors[0].expert_slab_bytes = 3 * kWindowPageBytes;
    require(!strata::metal::plan_windows(sources, 2 * kWindowPageBytes, plan, error) &&
                error.find("exceeds maxBufferLength") != std::string::npos,
            "a single expert slab larger than maxBufferLength must fail");
    sources.files[0].tensors[0] = {"out-of-bounds", 3 * kWindowPageBytes,
                                   2 * kWindowPageBytes, 0};
    require(!strata::metal::plan_windows(sources, 4 * kWindowPageBytes, plan, error),
            "out-of-file tensor ranges must fail");
    sources.files[0].file_size = 8 * kWindowPageBytes;
    sources.files[0].tensors = {{"overlap-a", 0, 2 * kWindowPageBytes, 0},
                                {"overlap-b", kWindowPageBytes, 2 * kWindowPageBytes, 0}};
    require(!strata::metal::plan_windows(sources, 4 * kWindowPageBytes, plan, error),
            "overlapping tensor ranges must fail");
    require(!strata::metal::plan_windows(sources, kWindowPageBytes - 1, plan, error),
            "a maxBufferLength below one page must fail");
}

void test_full_mode_address_layout() {
    constexpr std::uint64_t page = kWindowPageBytes;
    WindowSources sources{};
    sources.files.push_back({"full.gguf", 8 * page,
                             {{"token_embd.weight", 0, page, 0},
                              {"blk.0.ffn_gate_exps.weight", page, 2 * page, page},
                              {"blk.0.ffn_up_exps.weight", 3 * page, 2 * page, page},
                              {"blk.0.ffn_down_exps.weight", 5 * page, 2 * page, page}},
                             {}});
    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 8 * page, plan, error),
            "full-mode layout fixture plan failed: " + error);
    const std::vector<std::vector<std::uint64_t>> gpu_bases{{0x100000}};
    FullModeLayout layout;
    require(strata::metal::build_full_mode_layout(sources, plan, gpu_bases, 8 * page,
                                                   1, 2, layout, error),
            "full-mode address layout failed: " + error);
    std::vector<std::vector<std::uint64_t>> unavailable_gpu_bases{{0}};
    FullModeLayout unavailable_address_layout;
    require(!strata::metal::build_full_mode_layout(
                sources, plan, unavailable_gpu_bases, 8 * page, 1, 2,
                unavailable_address_layout, error) &&
                error.find("GPU address is unavailable") != std::string::npos,
            "full-mode layout must reject unavailable GPU window addresses");
    require(layout.tensors.size() == 7 &&
                layout.expert_addresses.size() ==
                    static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts *
                        kFullModeExpertRoles &&
                layout.resident.size() ==
                    static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts &&
                layout.resident[0] == 1 && layout.resident[1] == 1 &&
                layout.resident[2] == 0,
            "full-mode tensor bindings or all-resident table has the wrong shape");
    const auto address = [&](std::uint64_t expert, std::uint64_t role) {
        return layout.expert_addresses[static_cast<std::size_t>(
            expert * kFullModeExpertRoles + role)];
    };
    require(address(0, 0) == 0x104000 && address(1, 0) == 0x108000 &&
                address(0, 1) == 0x10c000 && address(1, 1) == 0x110000 &&
                address(0, 2) == 0x114000 && address(1, 2) == 0x118000,
            "expert address table must use gate, up, down order and each slab's exact offset");
    require(layout.tensors.front().tensor_name == "token_embd.weight" &&
                layout.tensors.front().expert_index == kNoExpertIndex &&
                layout.tensors.front().gpu_address == 0x100000 &&
                layout.tensors.front().byte_count == page,
            "whole dense tensor must retain its window address and full byte count");

    WindowSources packed_sources = sources;
    constexpr std::uint64_t packed_row_bytes = 512;
    packed_sources.files[0].excluded_tensors.push_back(
        {"blk.0.ffn_gate_inp.weight", 7 * page, 2 * packed_row_bytes,
         strata::metal::WindowTensorExclusionReason::converted});
    packed_sources.files[0].excluded_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight", 7 * page, 512,
         strata::metal::WindowTensorExclusionReason::converted});
    packed_sources.dense_pack_path = "dense.bin";
    packed_sources.dense_pack_file_size = packed_row_bytes + 256;
    packed_sources.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp.weight", 0, packed_row_bytes, packed_row_bytes, 4});
    packed_sources.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight", packed_row_bytes, 256, 256, 4});
    require(strata::metal::build_full_mode_layout(
                packed_sources, plan, gpu_bases, 8 * page, 1, 2, layout, error,
                0x300000000ull),
            "full-mode layout must bind converted routers in the separate arena: " + error);
    const auto packed_binding = std::find_if(
        layout.tensors.begin(), layout.tensors.end(),
        [](const strata::metal::TensorGpuBinding& binding) {
            return binding.tensor_name == "blk.0.ffn_gate_inp.weight";
        });
    const auto shared_router_binding = std::find_if(
        layout.tensors.begin(), layout.tensors.end(),
        [](const strata::metal::TensorGpuBinding& binding) {
            return binding.tensor_name == "blk.0.ffn_gate_inp_shexp.weight";
        });
    require(layout.dense_pack_bytes == packed_row_bytes + 256 &&
                layout.dense_pack_arena_bytes == page &&
                packed_binding != layout.tensors.end() &&
                packed_binding->gpu_address == 0x300000000ull &&
                packed_binding->byte_count == packed_row_bytes &&
                shared_router_binding != layout.tensors.end() &&
                shared_router_binding->gpu_address == 0x300000000ull + packed_row_bytes &&
                shared_router_binding->byte_count == 256,
            "converted-router layout must align each row and expose both exact bindings");

    WindowSources overlapping_pack = packed_sources;
    overlapping_pack.dense_pack_tensors[1].file_offset = 256;
    require(!strata::metal::build_full_mode_layout(
                overlapping_pack, plan, gpu_bases, 8 * page, 1, 2, layout, error,
                0x300000000ull) &&
                error.find("dense-pack ranges overlap") != std::string::npos,
            "layout builder must reject overlapping converted-router source rows");

    WindowSources overlarge_pack = packed_sources;
    overlarge_pack.files[0].excluded_tensors.pop_back();
    overlarge_pack.dense_pack_tensors.pop_back();
    overlarge_pack.files[0].excluded_tensors.back().byte_count = 18 * page;
    overlarge_pack.dense_pack_file_size = 9 * page;
    overlarge_pack.dense_pack_tensors[0].source_byte_count = 9 * page;
    overlarge_pack.dense_pack_tensors[0].byte_count = 9 * page;
    require(!strata::metal::build_full_mode_layout(
                overlarge_pack, plan, gpu_bases, 8 * page, 1, 2, layout, error,
                0x300000000ull) &&
                error.find("arena exceeds maxBufferLength") != std::string::npos,
            "layout builder must reject an arena larger than Metal's single-buffer limit");

    WindowSources ple_source = sources;
    ple_source.files[0].tensors.push_back({"per_layer_token_embd.weight", 7 * page, page, 0});
    WindowPlan ple_plan;
    require(strata::metal::plan_windows(ple_source, 8 * page, ple_plan, error),
            "PLE rejection fixture plan failed: " + error);
    require(!strata::metal::build_full_mode_layout(
                ple_source, ple_plan, gpu_bases, 8 * page, 1, 2, layout, error) &&
                error.find("PLE tensor") != std::string::npos,
            "full-mode layout must reject a PLE tensor source");

    WindowPlan missing_role = plan;
    missing_role.files[0].slices.pop_back();
    require(!strata::metal::build_full_mode_layout(sources, missing_role, gpu_bases, 8 * page,
                                                    1, 2, layout, error) &&
                error.find("does not cover every byte") != std::string::npos,
            "the address layout must reject a missing tensor slab");

    WindowPlan wrong_expert = plan;
    wrong_expert.files[0].slices[1].expert_index = 2;
    require(!strata::metal::build_full_mode_layout(sources, wrong_expert, gpu_bases, 8 * page,
                                                    1, 2, layout, error),
            "the address layout must reject an out-of-range expert index");

    WindowPlan wrong_window = plan;
    wrong_window.files[0].slices[0].window_offset += page;
    require(!strata::metal::build_full_mode_layout(sources, wrong_window, gpu_bases, 8 * page,
                                                    1, 2, layout, error),
            "the address layout must reject a mismatched window offset");

    WindowPlan missing_expert_role = plan;
    auto& missing_role_slices = missing_expert_role.files[0].slices;
    missing_role_slices.erase(
        std::remove_if(missing_role_slices.begin(), missing_role_slices.end(), [](const auto& slice) {
            return slice.tensor_name == "blk.0.ffn_down_exps.weight";
        }),
        missing_role_slices.end());
    sources.files[0].tensors.pop_back();
    require(!strata::metal::build_full_mode_layout(sources, missing_expert_role, gpu_bases,
                                                    8 * page, 1, 2, layout, error) &&
                error.find("missing one or more") != std::string::npos,
            "the address table must require all three roles for every expert");
}

void test_full_mode_warm_order() {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> profile_order;
    for (std::uint32_t expert = 8; expert != 0; --expert) {
        profile_order.emplace_back(1, expert - 1);
    }
    profile_order.emplace_back(0, 7);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ordered;
    std::string error;
    require(strata::metal::order_full_mode_warm_experts(2, 8, profile_order, ordered, error),
            "expert warm-order construction failed: " + error);
    require(ordered.size() == 16 &&
                std::equal(profile_order.begin(), profile_order.end(), ordered.begin()) &&
                ordered[strata::metal::kFullModeWarmWorkers] == profile_order.back() &&
                ordered[strata::metal::kFullModeWarmWorkers + 1] == std::pair{0u, 0u},
            "profile order must span the eight-expert wave boundary before row-major fallback");
    require(!strata::metal::order_full_mode_warm_experts(
                2, 8, {{0, 0}, {0, 0}}, ordered, error) &&
                error.find("duplicate") != std::string::npos,
            "expert warm-order construction must reject duplicate entries");
}

void test_coder_expert_table_stride() {
    constexpr std::uint64_t page = kWindowPageBytes;
    constexpr std::uint32_t layers = 2;
    constexpr std::uint32_t experts = 256;
    WindowFileSource file{"coder-experts.gguf",
                          static_cast<std::uint64_t>(layers) * experts *
                              kFullModeExpertRoles * page,
                          {}, {}};
    const char* roles[] = {"gate", "up", "down"};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint32_t role = 0; role < kFullModeExpertRoles; ++role) {
            const std::uint64_t offset =
                (static_cast<std::uint64_t>(layer) * kFullModeExpertRoles + role) * experts * page;
            file.tensors.push_back({"blk." + std::to_string(layer) + ".ffn_" + roles[role] +
                                        "_exps.weight",
                                    offset, static_cast<std::uint64_t>(experts) * page, page});
        }
    }
    WindowSources sources{};
    sources.files.push_back(std::move(file));
    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, sources.files[0].file_size, plan, error),
            "Coder address table fixture plan failed: " + error);

    constexpr std::uint64_t gpu_base = 0x100000000;
    FullModeLayout layout;
    require(strata::metal::build_full_mode_layout(
                sources, plan, {{gpu_base}}, sources.files[0].file_size,
                layers, experts, layout, error),
            "Coder address table layout failed: " + error);
    const std::size_t layer_one = experts;
    const std::size_t gate = (layer_one * kFullModeExpertRoles);
    require(layout.expert_addresses.size() ==
                    static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts *
                        kFullModeExpertRoles &&
                layout.resident.size() ==
                    static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts,
            "Coder tables must reserve the documented maximum capacity");
    require(layout.expert_addresses[gate] == gpu_base + 3ull * experts * page &&
                layout.expert_addresses[gate + 1] == gpu_base + 4ull * experts * page &&
                layout.expert_addresses[gate + 2] == gpu_base + 5ull * experts * page &&
                layout.resident[layer_one] == 1 &&
                layout.resident[static_cast<std::size_t>(layers) * experts] == 0,
            "Coder's second layer must use the active expert count as its row stride");
}

void test_page_rounded_eof_window() {
    constexpr std::uint64_t page = kWindowPageBytes;
    WindowSources sources{};
    sources.files.push_back({"short-final-page.gguf", 4 * page - 10,
                             {{"blk.0.ffn_gate_exps.weight", 0, page, page},
                              {"blk.0.ffn_up_exps.weight", page, page, page},
                              {"blk.0.ffn_down_exps.weight", 3 * page, page - 10, page - 10}},
                             {}});
    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, 8 * page, plan, error),
            "page-rounded EOF plan failed: " + error);
    require(plan.files[0].windows.size() == 1 &&
                plan.files[0].windows[0].end == 4 * page &&
                plan.files[0].windows[0].end > sources.files[0].file_size,
            "planner must round the final window to the mapped page extent");
    FullModeLayout layout;
    require(strata::metal::build_full_mode_layout(
                sources, plan, {{0x200000}}, 8 * page, 1, 1, layout, error),
            "layout must accept only the final mapped page padding: " + error);

    WindowPlan invalid_plan = plan;
    invalid_plan.files[0].windows[0].end += page;
    require(!strata::metal::build_full_mode_layout(
                sources, invalid_plan, {{0x200000}}, 8 * page, 1, 1, layout, error),
            "layout must reject windows that extend beyond the mapped final page");
}

void test_model_fixtures() {
    WindowSources q2;
    q2.files.push_back(make_fixture("q2_0-s1.json", 11'024'672, false));
    q2.files.push_back(make_fixture("q2_0-s2.json", 11'024'672, false));
    WindowPlan plan;
    std::string error;
    constexpr std::uint64_t kM5ProMaxBuffer =
        3'888ull * (1024ull * 1024 * 1024) / 100;
    require(strata::metal::plan_windows(q2, kM5ProMaxBuffer, plan, error),
            "Q2_0 fixture plan failed: " + error);
    require(plan.files.size() == 2 && plan.files[0].windows.size() == 1 &&
                plan.files[1].windows.empty(),
            "Q2_0 must plan one shard-1 window and no PLE shard-2 window");

    WindowSources swift;
    std::uint64_t ple_offset = 0;
    std::uint64_t ple_bytes = 0;
    swift.files.push_back(make_fixture("swift-IQ2_XS-s1.json", 10'967'808, true,
                                       &ple_offset, &ple_bytes));
    // Swift shard 2's data_start is recorded alongside shard 1 in 03-model-reference.md, section 3.8.
    swift.files.push_back(make_fixture("swift-IQ2_XS-s2.json", 56'320, true));
    require(ple_bytes != 0, "Swift shard-1 fixture must contain the PLE tensor range");
    require(strata::metal::plan_windows(swift, kM5ProMaxBuffer, plan, error),
            "Swift IQ2_XS fixture plan failed: " + error);
    require(plan.files.size() == 2 && plan.files[0].windows.size() == 2 &&
                plan.files[1].windows.size() == 1,
            "Swift IQ2_XS shard 1 must use two windows separated by the excluded PLE tensor");
    require(plan.wrapped_unread_bytes >= 252'149'760 &&
                plan.wrapped_unread_bytes - 252'149'760 < 4 * kWindowPageBytes,
            "Swift wrapped_unread differs from router bytes by " +
                std::to_string(plan.wrapped_unread_bytes >= 252'149'760
                                   ? plan.wrapped_unread_bytes - 252'149'760
                                   : 0) +
                " bytes (total " + std::to_string(plan.wrapped_unread_bytes) + ")");
    for (const auto& window : plan.files[0].windows) {
        require(window.end - window.start <= kM5ProMaxBuffer,
                "fixture window exceeds maxBufferLength");
        const std::uint64_t overlap_start = std::max(window.start, ple_offset);
        const std::uint64_t overlap_end = std::min(window.end, ple_offset + ple_bytes);
        const std::uint64_t overlap =
            overlap_end > overlap_start ? overlap_end - overlap_start : 0;
        require(overlap < kWindowPageBytes,
                "Swift window wraps more than page-rounding padding of the PLE table");
    }
    for (const auto& file : plan.files) {
        require(std::none_of(file.slices.begin(), file.slices.end(), [](const auto& slice) {
                    return slice.tensor_name == "per_layer_token_embd.weight";
                }),
                "the PLE tensor must not have a WindowSlice mapping");
        for (const auto& window : file.windows) {
            require(window.start % kWindowPageBytes == 0 &&
                        window.end % kWindowPageBytes == 0 &&
                        window.end - window.start <= kM5ProMaxBuffer,
                    "fixture window has invalid page alignment or length");
        }
        for (const auto& slice : file.slices) {
            require(slice.window_index < file.windows.size(),
                    "fixture tensor slice references a missing window");
            const auto& window = file.windows[static_cast<std::size_t>(slice.window_index)];
            require(slice.file_offset == window.start + slice.window_offset &&
                        slice.window_offset <= window.end - window.start &&
                        slice.byte_count <= window.end - window.start - slice.window_offset,
                    "fixture tensor slice does not map fully into its file window");
        }
    }
}

}  // namespace

int main() {
    try {
        test_rounding_and_merge_boundaries();
        test_expert_boundary_splitting();
        test_unaligned_expert_splits();
        test_merged_gap_coverage_when_forced_to_split();
        test_invalid_sources();
        test_full_mode_address_layout();
        test_full_mode_warm_order();
        test_coder_expert_table_stride();
        test_page_rounded_eof_window();
        test_model_fixtures();
        std::cout << "metal_window_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_window_test: " << exception.what() << '\n';
        return 1;
    }
}
