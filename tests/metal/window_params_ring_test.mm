// Fixed-address WindowParams ring storage and transaction rollback.
#include "strata/metal/context.hpp"
#include "strata/metal/window_params.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace strata::metal;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool params_equal(const WindowParams& left, const WindowParams& right) {
    return std::memcmp(&left, &right, sizeof(WindowParams)) == 0;
}

}  // namespace

int main() {
    try {
        std::string error;
        ContextOptions options;
        options.pipelines = false;
        auto context = Context::open(options, error);
        require(context != nullptr, "open Metal Context: " + error);

        const std::size_t allocations_before_ring =
            context->buffers().allocation_count();
        const std::uint64_t bytes_before_ring =
            context->buffers().allocated_size();
        auto ring = WindowParamsRing::open(*context, error);
        require(ring != nullptr && ring->is_valid(),
                "open fixed WindowParams ring: " + error);
        const std::size_t allocations_after_ring =
            context->buffers().allocation_count();
        const std::uint64_t bytes_after_ring =
            context->buffers().allocated_size();
        require(allocations_after_ring == allocations_before_ring + 1u &&
                    bytes_after_ring > bytes_before_ring,
                "WindowParams ring allocation accounting mismatch: count " +
                    std::to_string(allocations_before_ring) + " -> " +
                    std::to_string(allocations_after_ring) + ", bytes " +
                    std::to_string(bytes_before_ring) + " -> " +
                    std::to_string(bytes_after_ring));
        require(!ring->view(0u) && !ring->view(1u) &&
                    !ring->view(kWindowParamsRingSlotCount),
                "WindowParams ring exposed an uninitialized or invalid slot");

        const std::array<std::uint32_t, 1> decode_tokens{17u};
        const std::array<WindowSamplerSettings, 1> decode_sampler{{
            {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
             0, 1, 1, 3u, 0u},
        }};
        const BufferRef decode_view = ring->build_slot(
            0u, 2048u, decode_tokens, decode_sampler, 1u,
            0x12345678u, 0x9abcdef0u, error);
        require(decode_view && decode_view.length == sizeof(WindowParams) &&
                    decode_view.offset % kWindowParamsRingSlotAlignment == 0u &&
                    decode_view.offset == 0u,
                "build the T=1 write-through slot: " + error);
        WindowParams decode_expected{};
        require(build_window_params(
                    2048u, decode_tokens, decode_sampler, 1u,
                    0x12345678u, 0x9abcdef0u, decode_expected, error) &&
                    params_equal(*static_cast<const WindowParams*>(
                                     decode_view.contents),
                                 decode_expected),
                "T=1 ring contents differ from the canonical WindowParams builder: " +
                    error);

        std::array<std::uint32_t, kWindowMaxRows> verify_tokens{};
        std::array<WindowSamplerSettings, kWindowMaxRows> verify_samplers{};
        for (std::uint32_t row = 0u; row < kWindowMaxRows; ++row) {
            verify_tokens[row] = 100u + row;
            verify_samplers[row] = {
                0.8f, 0.92f, 0.03f, 1.1f, 0.05f, 0.07f,
                40, 2, 0, 7u + row, 3u + row,
            };
        }
        const BufferRef verify_view = ring->build_slot(
            1u, 4096u, verify_tokens, verify_samplers, 0u,
            11u, 29u, error);
        std::uint64_t expected_verify_gpu_address = 0u;
        require(verify_view &&
                    verify_view.length == sizeof(WindowParams) &&
                    verify_view.handle == decode_view.handle &&
                    verify_view.offset - decode_view.offset ==
                        kWindowParamsRingSlotStride &&
                    detail::gpu_address_with_offset(
                        decode_view.gpu_address,
                        kWindowParamsRingSlotStride,
                        expected_verify_gpu_address) &&
                    verify_view.gpu_address ==
                        expected_verify_gpu_address,
                "build the T=8 verify slot at its fixed aligned address: " + error);
        WindowParams verify_expected{};
        require(build_window_params(
                    4096u, verify_tokens, verify_samplers, 0u,
                    11u, 29u, verify_expected, error) &&
                    params_equal(*static_cast<const WindowParams*>(
                                     verify_view.contents),
                                 verify_expected),
                "T=8 ring contents differ from the canonical WindowParams builder: " +
                    error);
        require(ring->view(0u).handle == decode_view.handle &&
                    ring->view(0u).offset == decode_view.offset &&
                    ring->view(1u).offset == verify_view.offset,
                "WindowParams ring views changed their fixed slot addresses");

        std::array<std::byte, sizeof(WindowParams)> decode_before_failure{};
        std::memcpy(decode_before_failure.data(), decode_view.contents,
                    decode_before_failure.size());
        const std::array<std::uint32_t, 0> empty_tokens{};
        const std::array<WindowSamplerSettings, 0> empty_samplers{};
        require(!ring->build_slot(0u, 1u, empty_tokens, empty_samplers,
                                  0u, 0u, 0u, error) &&
                    !error.empty() &&
                    std::memcmp(decode_view.contents,
                                decode_before_failure.data(),
                                decode_before_failure.size()) == 0 &&
                    ring->view(0u),
                "failed slot construction changed a previously published slot");
        require(!ring->build_slot(kWindowParamsRingSlotCount, 1u,
                                  decode_tokens, decode_sampler, 0u,
                                  0u, 0u, error) &&
                    error.find("out of range") != std::string::npos,
                "WindowParams ring accepted an out-of-range slot index");

        const std::size_t allocations_before_rollback =
            context->buffers().allocation_count();
        const std::uint64_t bytes_before_rollback =
            context->buffers().allocated_size();
        std::unique_ptr<WindowParamsRing> provisional_ring;
        BufferRef provisional_view;
        {
            auto outer = context->buffers().transaction();
            provisional_ring = WindowParamsRing::open(*context, error);
            require(provisional_ring != nullptr,
                    "open provisional WindowParams ring: " + error);
            provisional_view = provisional_ring->build_slot(
                0u, 8u, decode_tokens, decode_sampler, 1u, 1u, 2u, error);
            require(static_cast<bool>(provisional_view),
                    "build provisional WindowParams slot: " + error);
        }
        require(provisional_ring != nullptr &&
                    !provisional_ring->is_valid() &&
                    !provisional_ring->view(0u) &&
                    !provisional_view &&
                    context->buffers().allocation_count() ==
                        allocations_before_rollback &&
                    context->buffers().allocated_size() ==
                        bytes_before_rollback,
                "outer rollback left a live WindowParams allocation or view");

        ContextOptions limited_options;
        limited_options.pipelines = false;
        limited_options.max_buffer_length_cap = kWindowParamsRingBytes - 1u;
        auto limited_context = Context::open(limited_options, error);
        require(limited_context != nullptr,
                "open limited-buffer Metal Context: " + error);
        require(WindowParamsRing::open(*limited_context, error) == nullptr &&
                    error.find("maxBufferLength") != std::string::npos &&
                    limited_context->buffers().allocation_count() == 0u &&
                    limited_context->buffers().allocated_size() == 0u,
                "WindowParams ring ignored the live maxBufferLength cap");

        std::cout << "window_params_ring_test: all checks passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "window_params_ring_test: " << exception.what() << '\n';
        return 1;
    }
}
