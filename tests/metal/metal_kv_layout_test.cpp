// tests/metal/metal_kv_layout_test.cpp - device-free main and MTP KV layout checks.
#include "strata/metal/kv.hpp"
#include "strata/metal/mtp_kv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t kv_bytes_per_cell(strata::metal::KvFormat format) {
    switch (format) {
        case strata::metal::KvFormat::Fp16: return 2'048;
        case strata::metal::KvFormat::Int8: return 1'056;
        case strata::metal::KvFormat::K8v4: return 816;
        case strata::metal::KvFormat::Q4_0: return 576;
    }
    throw std::runtime_error("unknown fixture format");
}

std::size_t mtp_kv_bytes_per_cell(strata::metal::KvFormat format) {
    switch (format) {
        case strata::metal::KvFormat::Fp16: return 2'048;
        case strata::metal::KvFormat::Int8:
        case strata::metal::KvFormat::K8v4: return 1'056;
        case strata::metal::KvFormat::Q4_0: return 576;
    }
    throw std::runtime_error("unknown MTP fixture format");
}

void check_layout(strata::metal::KvFormat format, std::uint32_t requested) {
    using namespace strata::metal;
    QsaKvLayout layout;
    std::string error;
    require(make_qsa_kv_layout(requested, format, layout, error),
            "layout construction failed: " + error);
    const std::uint32_t capacity = (requested + 3u) / 4u * 4u;
    require(error.empty(), "successful construction left an error");
    require(layout.requested_cells == requested && layout.cell_capacity == capacity,
            "requested and rounded cache capacities differ");
    require(layout.kv_rows == capacity * 2u, "K/V row count is not two rows per cell");
    require(layout.pooled_rows_per_layer == capacity / 4u + 2u,
            "indexer pool does not include its two spare rows");
    require(layout.layers.size() == kQsaKvLayerCount, "QSA layer count is not 12");

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    const auto add_interval = [&](KvRegion region) {
        if (region.bytes == 0) {
            require(region.offset == 0, "an absent region has a nonzero offset");
            return;
        }
        require(region.offset <= layout.allocation_bytes &&
                    region.bytes <= layout.allocation_bytes - region.offset,
                "region extends beyond the backing allocation");
        intervals.emplace_back(region.offset, region.offset + region.bytes);
    };

    for (const QsaKvLayerLayout& layer : layout.layers) {
        add_interval(layer.k);
        add_interval(layer.k_scale);
        add_interval(layer.v);
        add_interval(layer.v_scale);
        add_interval(layer.indexer_tail);
        add_interval(layer.indexer_dead);
        add_interval(layer.indexer_pooled);
        add_interval(layer.indexer_block_pos);
        require(layer.k.offset % 2u == 0 && layer.v.offset % 2u == 0 &&
                    layer.k_scale.offset % 2u == 0 && layer.v_scale.offset % 2u == 0,
                "K/V region is not half- or block-aligned");
        require(layer.indexer_tail.offset % alignof(float) == 0 &&
                    layer.indexer_dead.offset % alignof(float) == 0 &&
                    layer.indexer_pooled.offset % alignof(float) == 0 &&
                    layer.indexer_block_pos.offset % alignof(std::int32_t) == 0,
                "indexer state is not naturally aligned");
        require(layer.indexer_tail.bytes == 3u * 128u * sizeof(float),
                "indexer tail extent is wrong");
        require(layer.indexer_dead.bytes == 128u * sizeof(float),
                "indexer dead-row extent is wrong");
        require(layer.indexer_pooled.bytes ==
                    static_cast<std::size_t>(capacity / 4u + 2u) * 128u * sizeof(float),
                "indexer pooled extent is wrong");
        require(layer.indexer_block_pos.bytes == sizeof(std::int32_t),
                "indexer block position extent is wrong");

        const std::size_t rows = static_cast<std::size_t>(capacity) * 2u;
        switch (format) {
            case KvFormat::Fp16:
                require(layer.k.bytes == rows * 256u * sizeof(std::uint16_t) &&
                            layer.v.bytes == layer.k.bytes &&
                            layer.k_scale.bytes == 0 && layer.v_scale.bytes == 0,
                        "fp16 pool extents are wrong");
                break;
            case KvFormat::Int8:
                require(layer.k.bytes == rows * 256u && layer.v.bytes == layer.k.bytes &&
                            layer.k_scale.bytes == rows * 4u * sizeof(std::uint16_t) &&
                            layer.v_scale.bytes == layer.k_scale.bytes,
                        "int8 pool extents are wrong");
                break;
            case KvFormat::K8v4:
                require(layer.k.bytes == rows * 256u &&
                            layer.k_scale.bytes == rows * 4u * sizeof(std::uint16_t) &&
                            layer.v.bytes == rows * 8u * 18u && layer.v_scale.bytes == 0,
                        "k8v4 pool extents are wrong");
                break;
            case KvFormat::Q4_0:
                require(layer.k.bytes == rows * 8u * 18u &&
                            layer.v.bytes == layer.k.bytes &&
                            layer.k_scale.bytes == 0 && layer.v_scale.bytes == 0,
                        "q4_0 pool extents are wrong");
                break;
        }
    }

    add_interval(layout.rope);
    require(layout.rope.offset % (2u * alignof(float)) == 0,
            "shared RoPE table is not float2-aligned");
    require(layout.rope.bytes == static_cast<std::size_t>(capacity) * 256u,
            "shared RoPE extent is wrong");
    std::sort(intervals.begin(), intervals.end());
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        require(intervals[i - 1].second <= intervals[i].first,
                "QSA state regions overlap");
    }

    const std::size_t expected =
        static_cast<std::size_t>(kQsaKvLayerCount) *
            (kv_bytes_per_cell(format) * capacity +
             (static_cast<std::size_t>(capacity / 4u) + 2u) * 128u * sizeof(float) +
             3u * 128u * sizeof(float) + 128u * sizeof(float) + sizeof(std::int32_t)) +
        static_cast<std::size_t>(capacity) * 256u;
    require(layout.allocation_bytes == expected,
            "allocation size does not match the documented per-format regions");
}

void test_capacity_and_formats() {
    using strata::metal::KvFormat;
    constexpr std::array formats{KvFormat::Fp16, KvFormat::Int8,
                                 KvFormat::K8v4, KvFormat::Q4_0};
    constexpr std::array<std::uint32_t, 9> capacities{
        1, 2, 3, 4, 5, 7, 8, 65, strata::metal::kKvMaximumCells};
    for (const auto format : formats) {
        for (const auto capacity : capacities) check_layout(format, capacity);
    }
}

void test_rejections_are_transactional() {
    using namespace strata::metal;
    QsaKvLayout layout;
    MtpKvLayout mtp_layout;
    std::string error;
    require(!make_qsa_kv_layout(0, KvFormat::Fp16, layout, error) &&
                layout.layers.empty() && !error.empty(),
            "zero capacity was not rejected with a cleared result");
    require(!make_qsa_kv_layout(kKvMaximumCells + 1u, KvFormat::Fp16, layout, error) &&
                layout.layers.empty() && !error.empty(),
            "capacity above the Metal limit was not rejected");
    require(!make_qsa_kv_layout(1, static_cast<KvFormat>(255), layout, error) &&
                layout.layers.empty() && !error.empty(),
            "unknown format was not rejected");
    require(make_mtp_kv_layout(65, KvFormat::Fp16, mtp_layout, error),
            "seed a populated MTP layout before failure checks");
    require(!make_mtp_kv_layout(0, KvFormat::Fp16, mtp_layout, error) &&
                mtp_layout.requested_cells == 0 && mtp_layout.cell_capacity == 0 &&
                mtp_layout.allocation_bytes == 0 && mtp_layout.k.bytes == 0 &&
                mtp_layout.k_scale.bytes == 0 && mtp_layout.v.bytes == 0 &&
                mtp_layout.v_scale.bytes == 0 && !error.empty(),
            "zero MTP capacity was not rejected with a cleared result");
    require(!make_mtp_kv_layout(kKvMaximumCells + 1u, KvFormat::Fp16, mtp_layout, error) &&
                mtp_layout.allocation_bytes == 0 && !error.empty(),
            "MTP capacity above the Metal limit was not rejected");
    require(!make_mtp_kv_layout(1, static_cast<KvFormat>(255), mtp_layout, error) &&
                mtp_layout.allocation_bytes == 0 && !error.empty(),
            "unknown MTP format was not rejected");
}

void check_mtp_layout(strata::metal::KvFormat format, std::uint32_t requested) {
    using namespace strata::metal;
    MtpKvLayout layout;
    std::string error;
    require(make_mtp_kv_layout(requested, format, layout, error),
            "MTP layout construction failed: " + error);
    const std::uint32_t capacity = (requested + 3u) / 4u * 4u;
    const KvFormat storage_format = format == KvFormat::K8v4 ? KvFormat::Int8 : format;
    require(error.empty(), "successful MTP layout construction left an error");
    require(layout.requested_format == format && layout.storage_format == storage_format,
            "MTP requested and effective storage formats differ");
    require(layout.requested_cells == requested && layout.cell_capacity == capacity,
            "MTP requested and rounded capacities differ");
    require(layout.kv_rows == capacity * kQsaKvHeadCount,
            "MTP K/V row count is not two rows per cell");

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    const auto add_interval = [&](KvRegion region) {
        if (region.bytes == 0) {
            require(region.offset == 0, "an absent MTP region has a nonzero offset");
            return;
        }
        require(region.offset <= layout.allocation_bytes &&
                    region.bytes <= layout.allocation_bytes - region.offset,
                "MTP region extends beyond the backing allocation");
        intervals.emplace_back(region.offset, region.offset + region.bytes);
    };
    add_interval(layout.k);
    add_interval(layout.k_scale);
    add_interval(layout.v);
    add_interval(layout.v_scale);

    const std::size_t rows = static_cast<std::size_t>(layout.kv_rows);
    switch (storage_format) {
        case KvFormat::Fp16:
            require(layout.k.bytes == rows * 256u * sizeof(std::uint16_t) &&
                        layout.v.bytes == layout.k.bytes &&
                        layout.k_scale.bytes == 0 && layout.v_scale.bytes == 0,
                    "MTP fp16 K/V extents are wrong");
            break;
        case KvFormat::Int8:
            require(layout.k.bytes == rows * 256u && layout.v.bytes == layout.k.bytes &&
                        layout.k_scale.bytes == rows * 4u * sizeof(std::uint16_t) &&
                        layout.v_scale.bytes == layout.k_scale.bytes,
                    "MTP int8 K/V or scale extents are wrong");
            break;
        case KvFormat::K8v4:
            require(false, "MTP K8v4 must resolve to plain int8");
            break;
        case KvFormat::Q4_0:
            require(layout.k.bytes == rows * 8u * 18u &&
                        layout.v.bytes == layout.k.bytes &&
                        layout.k_scale.bytes == 0 && layout.v_scale.bytes == 0,
                    "MTP q4_0 K/V extents are wrong");
            break;
    }
    require(layout.k.offset % (storage_format == KvFormat::Int8 ? 1u : 2u) == 0 &&
                layout.v.offset % (storage_format == KvFormat::Int8 ? 1u : 2u) == 0 &&
                layout.k_scale.offset % 2u == 0 && layout.v_scale.offset % 2u == 0,
            "MTP K/V regions are not correctly aligned");

    std::sort(intervals.begin(), intervals.end());
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        require(intervals[i - 1].second <= intervals[i].first,
                "MTP KV regions overlap");
    }
    require(layout.allocation_bytes ==
                mtp_kv_bytes_per_cell(format) * static_cast<std::size_t>(capacity),
            "MTP allocation size does not match its format-specific K/V regions");
}

void test_mtp_capacity_and_formats() {
    using strata::metal::KvFormat;
    constexpr std::array formats{KvFormat::Fp16, KvFormat::Int8,
                                 KvFormat::K8v4, KvFormat::Q4_0};
    constexpr std::array<std::uint32_t, 9> capacities{
        1, 2, 3, 4, 5, 7, 8, 65, strata::metal::kKvMaximumCells};
    for (const auto format : formats) {
        for (const auto capacity : capacities) check_mtp_layout(format, capacity);
    }
}

}  // namespace

int main() {
    try {
        test_capacity_and_formats();
        test_rejections_are_transactional();
        test_mtp_capacity_and_formats();
        std::cout << "metal_kv_layout_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_kv_layout_test: " << exception.what() << '\n';
        return 1;
    }
}
