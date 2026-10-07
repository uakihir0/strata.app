// tests/metal/metal_kv_store_test.mm - one-allocation QSA store views on an Apple GPU.
#include "strata/metal/kv.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

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

void check_format(Context& context, KvFormat format, std::size_t expected_new_allocations) {
    std::string error;
    const std::size_t allocations_before = context.buffers().allocation_count();
    auto store = KvStore::open(context, 13, format, error);
    require(store != nullptr, "open QSA KV store: " + error);
    const BufferRef allocation = store->allocation();
    require(context.buffers().allocation_count() == allocations_before + expected_new_allocations,
            "KvStore did not create exactly one Metal allocation");
    require(context.buffers().owns(allocation) &&
                allocation.length == store->layout().allocation_bytes,
            "backing allocation is not Context-owned or has the wrong extent");
    require(store->layout().cell_capacity == 16,
            "KvStore did not round the requested capacity to four cells");
    check_view(store->rope(), allocation, store->layout().rope, "shared RoPE table");

    for (const std::uint32_t index : {0u, 5u, 11u}) {
        QsaKvLayerViews views;
        require(store->layer(index, views, error), "get QSA layer views: " + error);
        const QsaKvLayerLayout& regions = store->layout().layers[index];
        check_view(views.k, allocation, regions.k, "K");
        check_view(views.k_scale, allocation, regions.k_scale, "K scales");
        check_view(views.v, allocation, regions.v, "V");
        check_view(views.v_scale, allocation, regions.v_scale, "V scales");
        check_view(views.indexer_tail, allocation, regions.indexer_tail, "indexer tail");
        check_view(views.indexer_dead, allocation, regions.indexer_dead, "indexer dead row");
        check_view(views.indexer_pooled, allocation, regions.indexer_pooled, "indexer pool");
        check_view(views.indexer_block_pos, allocation, regions.indexer_block_pos,
                   "indexer block position");
        require(views.indexer_tail.gpu_address % alignof(float) == 0 &&
                    views.indexer_dead.gpu_address % alignof(float) == 0 &&
                    views.indexer_pooled.gpu_address % alignof(float) == 0 &&
                    views.indexer_block_pos.gpu_address % alignof(std::int32_t) == 0,
                "Metal indexer subview address is misaligned");
        for (std::size_t byte = 0; byte < views.indexer_tail.length; ++byte) {
            require(static_cast<const std::uint8_t*>(views.indexer_tail.contents)[byte] == 0,
                    "indexer tail was not initialized");
        }
        for (std::size_t byte = 0; byte < views.indexer_dead.length; ++byte) {
            require(static_cast<const std::uint8_t*>(views.indexer_dead.contents)[byte] == 0,
                    "indexer dead row was not initialized");
        }
        std::int32_t block_pos = -1;
        std::memcpy(&block_pos, views.indexer_block_pos.contents, sizeof(block_pos));
        require(block_pos == 0, "indexer block position was not initialized");
    }

    QsaKvLayerViews invalid;
    require(!store->layer(kQsaKvLayerCount, invalid, error) &&
                !invalid.k && error.find("out of range") != std::string::npos,
            "out-of-range QSA layer lookup did not fail closed");
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
        for (const KvFormat format : formats) check_format(*context, format, 1);

        auto invalid = KvStore::open(*context, 0, KvFormat::Fp16, error);
        require(invalid == nullptr && error.find("between 1 and") != std::string::npos,
                "KvStore accepted zero capacity");
        std::cout << "metal_kv_store_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_kv_store_test: " << exception.what() << '\n';
        return 1;
    }
}
