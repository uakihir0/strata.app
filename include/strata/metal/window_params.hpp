// Host construction of one Metal decode or verify window parameter block.
#pragma once

#include "strata/metal/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace strata::metal {

inline constexpr std::uint32_t kWindowMaxRows = 8u;
inline constexpr std::uint32_t kWindowMaxSelectedCells = 2051u;

/// Mirrors StepRow in src/metal/kernels/common/types.h.
struct WindowStepRow {
    std::int32_t pos = 0;
    std::int32_t n_kv = 0;
    std::int32_t n_bid = 0;
    std::int32_t width = 0;
};

/// Sampler values resolved by the session/options layer before building a window.
/// The per-row Philox counter is derived from pos0 by build_window_params().
struct WindowSamplerSettings {
    float temperature = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    std::int32_t top_k = 0;
    std::int32_t min_keep = 1;
    std::int32_t greedy = 1;
    std::uint32_t hist_off = 0;
    std::uint32_t hist_len = 0;
};

/// Mirrors SampRow in src/metal/kernels/common/types.h.
struct WindowSampRow {
    float temperature;
    float top_p;
    float min_p;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    std::int32_t top_k;
    std::int32_t min_keep;
    std::int32_t greedy;
    std::uint32_t counter;
    std::uint32_t hist_off;
    std::uint32_t hist_len;
};

/// Mirrors WindowParams in src/metal/kernels/common/types.h.
struct WindowParams {
    std::uint32_t T = 0;
    std::uint32_t nc = 0;
    std::uint32_t pos0 = 0;
    std::uint32_t n_keep = 0;
    std::uint32_t seed_lo = 0;
    std::uint32_t seed_hi = 0;
    std::uint32_t tokens[kWindowMaxRows]{};
    WindowStepRow step[kWindowMaxRows]{};
    WindowSampRow samp[kWindowMaxRows]{};
};

inline constexpr std::uint32_t kWindowParamsRingSlotCount = 2u;
inline constexpr std::size_t kWindowParamsRingSlotAlignment = 256u;
inline constexpr std::size_t kWindowParamsRingSlotStride =
    (sizeof(WindowParams) + kWindowParamsRingSlotAlignment - 1u) /
    kWindowParamsRingSlotAlignment * kWindowParamsRingSlotAlignment;
inline constexpr std::size_t kWindowParamsRingBytes =
    kWindowParamsRingSlotCount * kWindowParamsRingSlotStride;

/// Build the shared parameter block for one decode/verify window.
///
/// `samplers` has one resolved row per token. The builder derives nc, every
/// StepRow and each Philox counter, zeros unused rows, and leaves `output`
/// unchanged on failure. `n_keep` is zero for an ordinary speculative window
/// and 1..T for a write-through window or commit block.
bool build_window_params(std::uint32_t pos0,
                         std::span<const std::uint32_t> tokens,
                         std::span<const WindowSamplerSettings> samplers,
                         std::uint32_t n_keep,
                         std::uint32_t seed_lo,
                         std::uint32_t seed_hi,
                         WindowParams& output,
                         std::string& error);

/// Own the fixed-address, two-slot shared WindowParams ring used by windows.
///
/// Context must outlive this ring. The caller selects a slot whose previous
/// command-buffer use has completed before rebuilding it; this class does not
/// track GPU submissions. A slot view is available only after build_slot()
/// has initialized it.
class WindowParamsRing {
public:
    static std::unique_ptr<WindowParamsRing> open(Context& context,
                                                  std::string& error);
    ~WindowParamsRing();
    WindowParamsRing(const WindowParamsRing&) = delete;
    WindowParamsRing& operator=(const WindowParamsRing&) = delete;

    bool is_valid() const noexcept;
    BufferRef view(std::uint32_t slot) const noexcept;
    BufferRef build_slot(std::uint32_t slot, std::uint32_t pos0,
                         std::span<const std::uint32_t> tokens,
                         std::span<const WindowSamplerSettings> samplers,
                         std::uint32_t n_keep, std::uint32_t seed_lo,
                         std::uint32_t seed_hi, std::string& error);

private:
    struct Impl;
    explicit WindowParamsRing(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal
