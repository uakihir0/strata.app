// Host-only tests for tiered dense/resident layout and table construction.
#include "strata/metal/tiered.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace strata::metal;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Geometry make_geometry() {
    Geometry geometry;
    geometry.family = "synthetic";
    geometry.model = "tiered";
    geometry.shards.push_back({"model.gguf", 20'000});
    geometry.n_layers = 2;
    geometry.n_expert = 3;
    geometry.nonexpert_bytes = 300;
    geometry.dense_pack_bytes = 16;
    geometry.n_qsa_layers = 1;
    geometry.n_kv_head = 1;
    geometry.head_dim = 1;
    geometry.n_embd = 1;
    geometry.vocab = 1;
    geometry.types = {"iq2_xs"};
    geometry.layers = {{768, "iq2_xs", "iq2_xs"}, {768, "iq2_xs", "iq2_xs"}};
    return geometry;
}

WindowSources make_sources() {
    WindowFileSource file;
    file.path = "/tmp/model.gguf";
    file.file_size = 20'000;
    const char* roles[] = {"gate", "up", "down"};
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        for (std::uint32_t role = 0; role < 3; ++role) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                                     roles[role] + "_exps.weight";
            const std::uint64_t offset = 1'024 + (layer * 3 + role) * 768;
            file.tensors.push_back({name, offset, 768, 256});
        }
    }
    file.tensors.push_back({"token_embd.weight", 10'000, 300, 0});
    file.excluded_tensors.push_back(
        {"per_layer_token_embd.weight", 12'000, 128, WindowTensorExclusionReason::ple});
    file.excluded_tensors.push_back(
        {"blk.0.ffn_gate_inp.weight", 13'000, 32, WindowTensorExclusionReason::converted});

    WindowSources sources;
    sources.files.push_back(std::move(file));
    sources.dense_pack_path = "/tmp/dense.bin";
    sources.dense_pack_file_size = 64;
    sources.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp.weight", 16, 16, 16, 4});
    return sources;
}

PlanResult make_plan() {
    PlanResult plan;
    plan.fits = true;
    plan.mode = "tiered";
    plan.resident_order = {{1, 2}, {0, 0}};
    plan.experts_resident = plan.resident_order.size();
    plan.experts_resident_bytes = 1'536;
    plan.free_bytes = 20'000;
    return plan;
}

void test_layout_tables_and_copy_ranges() {
    const Geometry geometry = make_geometry();
    const WindowSources sources = make_sources();
    const PlanResult memory_plan = make_plan();
    TieredLayout layout;
    std::string error;
    constexpr std::uint64_t dense_base = 0x1'0000'0000;
    constexpr std::uint64_t resident_base = 0x2'0000'0000;
    require(build_tiered_layout(geometry, sources, memory_plan, dense_base,
                                resident_base, layout, error),
            "build tiered layout: " + error);

    require(layout.dense_copies.size() == 2 &&
                layout.dense_copies[0].tensor_name == "token_embd.weight" &&
                layout.dense_copies[0].arena_offset == 0 &&
                !layout.dense_copies[0].from_dense_pack &&
                layout.dense_copies[1].tensor_name == "blk.0.ffn_gate_inp.weight" &&
                layout.dense_copies[1].from_dense_pack &&
                layout.dense_copies[1].source_offset == 16 &&
                layout.dense_copies[1].arena_offset == 512 &&
                layout.dense_arena_bytes == 16'384,
            "dense arena must align tensors to 256 bytes and its wrapped length to 16 KiB");
    require(layout.tensors.size() == 2 &&
                layout.tensors[0].gpu_address == dense_base &&
                layout.tensors[1].gpu_address == dense_base + 512 &&
                layout.tensors[1].byte_count == 16,
            "dense tensor bindings must resolve to their checked arena addresses");

    const std::size_t layer1_expert2 = 1 * geometry.n_expert + 2;
    const std::size_t layer0_expert0 = 0;
    const TieredExpert& first = layout.experts[layer1_expert2];
    const TieredExpert& second = layout.experts[layer0_expert0];
    require(first.resident_slot == 0 && second.resident_slot == 1 &&
                first.resident_arena_offset == 0 &&
                second.resident_arena_offset == 768 &&
                first.roles[0].file_offset == 1'024 + (3 + 0) * 768 + 2 * 256 &&
                first.roles[1].file_offset == 1'024 + (3 + 1) * 768 + 2 * 256 &&
                first.roles[2].file_offset == 1'024 + (3 + 2) * 768 + 2 * 256,
            "resident slots and gate/up/down source offsets must follow profile order");
    require(first.resident_addresses[0] == resident_base &&
                first.resident_addresses[1] == resident_base + 256 &&
                first.resident_addresses[2] == resident_base + 512 &&
                second.resident_addresses[0] == resident_base + 768 &&
                layout.resident_arena_bytes == 1'536,
            "resident expert addresses must point into consecutive role slabs");

    const std::size_t active = geometry.n_layers * geometry.n_expert;
    require(layout.resident[layer1_expert2] == 0 &&
                layout.resident[layer0_expert0] == 1 &&
                layout.resident[1] == -1 &&
                layout.resident[active] == 0,
            "res must contain slot indices, -1 for cold experts and zeros in unused capacity");
    require(layout.expert_addresses[layer1_expert2 * 3] == resident_base &&
                layout.expert_addresses[layer1_expert2 * 3 + 1] == resident_base + 256 &&
                layout.expert_addresses[layer1_expert2 * 3 + 2] == resident_base + 512 &&
                layout.expert_addresses[1 * 3] == 0,
            "addr must contain resident role pointers and leave cold entries empty");
    require(layout.plans[0].n_groups == 0 && layout.plans[47].skip == 0,
            "the shared plan buffer must start fully zeroed");
}

void test_invalid_layouts_fail_closed() {
    const Geometry geometry = make_geometry();
    const WindowSources sources = make_sources();
    const PlanResult plan = make_plan();
    TieredLayout ignored;
    std::string error;
    require(!build_tiered_layout(geometry, sources, plan, 0,
                                 0x2'0000'0000, ignored, error) &&
                error.find("GPU address base is unavailable") != std::string::npos,
            "tiered dense table must reject an unavailable GPU address base");
    require(!build_tiered_layout(geometry, sources, plan, 0x1'0000'0000,
                                 0, ignored, error) &&
                error.find("GPU address base is unavailable") != std::string::npos,
            "tiered resident table must reject an unavailable GPU address base");

    PlanResult full = plan;
    full.mode = "full";
    require(!build_tiered_layout(geometry, sources, full, 0, 0, ignored, error),
            "tiered layout must reject a full-mode plan");

    PlanResult duplicate = plan;
    duplicate.resident_order.push_back({1, 2});
    ++duplicate.experts_resident;
    duplicate.experts_resident_bytes += 768;
    require(!build_tiered_layout(geometry, sources, duplicate, 0, 0, ignored, error),
            "tiered layout must reject duplicate resident experts");

    PlanResult no_padding_budget = plan;
    no_padding_budget.free_bytes = 16'067;
    require(!build_tiered_layout(geometry, sources, no_padding_budget, 0, 0, ignored, error),
            "tiered layout must account for tensor and page alignment padding in the remaining budget");

    WindowSources missing_role = sources;
    missing_role.files[0].tensors.erase(missing_role.files[0].tensors.begin());
    require(!build_tiered_layout(geometry, missing_role, plan, 0, 0, ignored, error),
            "tiered layout must reject a missing gate/up/down expert role");

    WindowSources overlapping_gguf = sources;
    overlapping_gguf.files[0].tensors[1].file_offset =
        overlapping_gguf.files[0].tensors[0].file_offset;
    require(!build_tiered_layout(geometry, overlapping_gguf, plan, 0, 0, ignored, error),
            "tiered layout must reject overlapping GGUF tensor ranges");

    WindowSources bad_pack = sources;
    bad_pack.dense_pack_tensors[0].file_offset = 60;
    require(!build_tiered_layout(geometry, bad_pack, plan, 0, 0, ignored, error),
            "tiered layout must reject a converted row that extends past dense.bin");

    WindowSources overlapping_pack = sources;
    overlapping_pack.files[0].excluded_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight", 14'000, 32,
         WindowTensorExclusionReason::converted});
    overlapping_pack.dense_pack_tensors.push_back(
        {"blk.0.ffn_gate_inp_shexp.weight", 24, 16, 16, 4});
    Geometry two_conversions = geometry;
    two_conversions.dense_pack_bytes = 32;
    require(!build_tiered_layout(two_conversions, overlapping_pack, plan, 0, 0,
                                 ignored, error),
            "tiered layout must reject overlapping dense.bin tensor ranges");

    require(!build_tiered_layout(geometry, sources, plan, 1, 0, ignored, error),
            "dense arena GPU base must satisfy its documented 16 KiB alignment");
    require(!build_tiered_layout(geometry, sources, plan,
                                 0xffff'ffff'ffff'c000ull, 0, ignored, error),
            "dense arena GPU base must leave room for its complete page-rounded address range");
}

}  // namespace

int main() {
    try {
        test_layout_tables_and_copy_ranges();
        test_invalid_layouts_fail_closed();
        std::cout << "metal_tiered_layout_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_tiered_layout_test: " << exception.what() << '\n';
        return 1;
    }
}
