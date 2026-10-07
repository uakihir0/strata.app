// tests/metal/metal_mtp_kv_store_test.mm - one-allocation MTP drafter KV views on an Apple GPU.
#include "strata/metal/mtp_kv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace strata::metal;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_view(BufferRef view, BufferRef allocation, KvRegion region,
                const std::string& name) {
    if (region.bytes == 0) {
        require(!view, name + " should be absent");
        return;
    }
    require(view && view.handle == allocation.handle && view.length == region.bytes &&
                view.offset == allocation.offset + region.offset &&
                view.gpu_address == allocation.gpu_address + region.offset &&
                view.contents == static_cast<std::byte*>(allocation.contents) + region.offset,
            name + " does not map the expected subregion of the single allocation");
}

void check_format(Context& context, KvFormat format, std::size_t allocations_before) {
    std::string error;
    auto store = MtpKvStore::open(context, 13, format, error);
    require(store != nullptr, "open MTP KV store: " + error);
    const BufferRef allocation = store->allocation();
    require(context.buffers().allocation_count() == allocations_before + 1,
            "MtpKvStore did not create exactly one Metal allocation");
    require(context.buffers().owns(allocation) &&
                allocation.length == store->layout().allocation_bytes,
            "MTP backing allocation is not Context-owned or has the wrong extent");
    require(store->layout().cell_capacity == 16 &&
                store->layout().kv_rows == 16 * kQsaKvHeadCount,
            "MTP KV store did not round capacity to four-cell groups");

    MtpKvViews views;
    require(store->views(views, error), "get MTP KV views: " + error);
    const MtpKvLayout& layout = store->layout();
    check_view(views.k, allocation, layout.k, "K");
    check_view(views.k_scale, allocation, layout.k_scale, "K scales");
    check_view(views.v, allocation, layout.v, "V");
    check_view(views.v_scale, allocation, layout.v_scale, "V scales");

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    for (const KvRegion region : {layout.k, layout.k_scale, layout.v, layout.v_scale}) {
        if (region.bytes == 0) continue;
        intervals.emplace_back(region.offset, region.offset + region.bytes);
    }
    std::sort(intervals.begin(), intervals.end());
    for (std::size_t index = 1; index < intervals.size(); ++index) {
        require(intervals[index - 1].second <= intervals[index].first,
                "MTP Metal views overlap");
    }

    const std::array<std::pair<BufferRef, std::uint8_t>, 4> markers{{
        {views.k, 0x16}, {views.k_scale, 0x2d}, {views.v, 0x43}, {views.v_scale, 0x5a}}};
    for (const auto& [view, marker] : markers) {
        if (!view) continue;
        auto* bytes = static_cast<std::uint8_t*>(view.contents);
        bytes[0] = marker;
        bytes[view.length - 1] = static_cast<std::uint8_t>(marker ^ 0xff);
        require(bytes[0] == marker &&
                    bytes[view.length - 1] == static_cast<std::uint8_t>(marker ^ 0xff),
                "MTP KV view did not expose writable allocation bytes");
    }
}

void test_outer_transaction_rollback(Context& context) {
    const std::size_t allocations_before = context.buffers().allocation_count();
    std::unique_ptr<MtpKvStore> provisional;
    BufferRef original_allocation;
    MtpKvViews original_views;
    std::uint64_t original_identity = 0;
    {
        auto outer = context.buffers().transaction();
        std::string error;
        provisional = MtpKvStore::open(context, 9, KvFormat::Int8, error);
        require(provisional != nullptr,
                "open inner MTP KV store before outer rollback: " + error);
        original_allocation = provisional->allocation();
        original_identity = context.buffers().allocation_identity(original_allocation);
        require(static_cast<bool>(original_allocation),
                "MTP KV allocation was not observable before outer rollback");
        require(original_identity != 0 &&
                    context.buffers().owns(original_allocation, original_identity),
                "MTP KV allocation has no live identity");
        require(provisional->views(original_views, error),
                "capture MTP KV views before outer rollback: " + error);
        require(context.buffers().allocation_count() == allocations_before + 1,
                "inner MTP KV open did not register its provisional allocation");
    }
    require(context.buffers().allocation_count() == allocations_before,
            "outer transaction did not roll back a committed inner MTP KV allocation");
    require(context.buffers().allocation_identity(original_allocation) == 0 &&
                !context.buffers().owns(original_allocation, original_identity) &&
                !original_allocation && !original_views.k &&
                !original_views.k_scale && !original_views.v &&
                !original_views.v_scale,
            "rolled-back MTP KV allocation retained a live view or identity");
    std::string error;
    BufferRef replacement = context.buffers().allocate(
        "post-rollback MTP KV identity check", original_allocation.length, error);
    require(static_cast<bool>(replacement),
            "allocate after MTP KV rollback: " + error);
    const std::uint64_t replacement_identity =
        context.buffers().allocation_identity(replacement);
    require(replacement_identity != 0 && replacement_identity != original_identity &&
                context.buffers().owns(replacement, replacement_identity) &&
                !context.buffers().owns(original_allocation, original_identity),
            "a later allocation revived the rolled-back MTP KV identity");
    require(context.buffers().commit_residency(error),
            "commit post-rollback identity allocation: " + error);
    require(!provisional->allocation(),
            "MTP KV store exposed its allocation after outer rollback");
    MtpKvViews invalidated_views;
    invalidated_views.k = original_allocation;
    require(!provisional->views(invalidated_views, error) &&
                error.find("no longer owned by its Context") != std::string::npos,
            "MTP KV store returned views after outer rollback");
    require(!invalidated_views.k && !invalidated_views.k_scale && !invalidated_views.v &&
                !invalidated_views.v_scale,
            "failed MTP KV view lookup did not clear every output view");
    provisional.reset();
}

void test_invalid_inputs_do_not_allocate(Context& context) {
    const std::size_t allocations_before = context.buffers().allocation_count();
    std::string error;
    require(MtpKvStore::open(context, 0, KvFormat::Fp16, error) == nullptr &&
                error.find("between 1 and") != std::string::npos,
            "MtpKvStore accepted zero capacity");
    require(MtpKvStore::open(context, 1, static_cast<KvFormat>(255), error) == nullptr &&
                error.find("unknown MTP KV format") != std::string::npos,
            "MtpKvStore accepted an unknown KV format");
    require(context.buffers().allocation_count() == allocations_before,
            "invalid MTP KV inputs registered a Metal allocation");
}

}  // namespace

int main() {
    try {
        ContextOptions options;
        options.pipelines = false;
        std::string error;
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal context: " + error);

        constexpr std::array formats{
            KvFormat::Fp16, KvFormat::Int8, KvFormat::Q4_0, KvFormat::K8v4};
        for (const KvFormat format : formats) {
            const std::size_t allocations_before = context->buffers().allocation_count();
            check_format(*context, format, allocations_before);
        }
        test_outer_transaction_rollback(*context);
        test_invalid_inputs_do_not_allocate(*context);
        std::cout << "metal_mtp_kv_store_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_mtp_kv_store_test: " << exception.what() << '\n';
        return 1;
    }
}
