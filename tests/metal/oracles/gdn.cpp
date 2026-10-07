#include "oracles/gdn.hpp"

#include "../../../src/metal/kernels/common/params.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oracle {
namespace {

using twin::tf;

constexpr std::uint32_t kChannels = GDN_CONV_CHANNELS;
constexpr std::uint32_t kHeadDim = GDN_HEAD_DIM;
constexpr std::uint32_t kQKHeads = GDN_QK_HEADS;
constexpr std::uint32_t kVHeads = GDN_V_HEADS;
constexpr std::uint32_t kValueDim = GDN_VALUE_DIM;
constexpr float kEpsilon = RMS_EPSILON;

tf rsqrt_accurate(tf value) {
    const float result = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(value.value)));
    return tf{result};
}

std::array<tf, 32> butterfly(std::array<tf, 32> values) {
    for (std::uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values;
}

tf sigmoid(tf value) {
    return tf{1.0f} / (tf{1.0f} + twin::accurate_exp(-value));
}

tf log1p(tf value) {
    const tf one{1.0f};
    const tf u = one + value;
    if (u.value == 1.0f) return value;
    return twin::accurate_log(u) * (value / (u - one));
}

tf softplus(tf value) {
    if (value.value > 20.0f) return tf{value.value};
    const tf result = log1p(twin::accurate_exp(value));
    // A library function's U(n) magnitude is |f(value)|. Do not carry the
    // cancellation in log1p's internal (1 + exp(x)) - 1 into the bound.
    return tf{result.value};
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(std::uint32_t{value} << 16u);
}

}  // namespace

void gdn_conv_l2(std::span<const float> history,
                 std::span<const float> qkv,
                 std::span<const float> conv_weights,
                 std::uint32_t rows, std::vector<tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * kChannels);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t channel = 0u; channel < kChannels; ++channel) {
            const std::size_t hist_base =
                static_cast<std::size_t>(channel) * 3u;
            const std::size_t weight_base =
                static_cast<std::size_t>(channel) * GDN_CONV_KERNEL;
            std::array<tf, 3> taps{};
            for (std::uint32_t j = 0u; j < taps.size(); ++j) {
                const std::uint32_t source = row + j;
                taps[j] = source < 3u
                    ? tf{history[hist_base + source]}
                    : tf{qkv[static_cast<std::size_t>(source - 3u) *
                                  kChannels +
                              channel]};
            }
            const tf current{
                qkv[static_cast<std::size_t>(row) * kChannels + channel]};
            tf value = twin::fma(
                current, tf{conv_weights[weight_base + 3u]},
                twin::fma(
                    taps[2], tf{conv_weights[weight_base + 2u]},
                    twin::fma(taps[1],
                              tf{conv_weights[weight_base + 1u]},
                              taps[0] * tf{conv_weights[weight_base]})));
            value = value /
                    (tf{1.0f} +
                     twin::accurate_exp(-value));
            output[static_cast<std::size_t>(row) * kChannels + channel] =
                value;
        }

        for (std::uint32_t block = 0u;
             block < 2u * kQKHeads; ++block) {
            std::array<tf, 4> warp_sums{};
            for (std::uint32_t warp = 0u; warp < warp_sums.size(); ++warp) {
                std::array<tf, 32> lanes{};
                for (std::uint32_t lane = 0u; lane < lanes.size(); ++lane) {
                    const std::uint32_t channel =
                        block * kHeadDim + warp * 32u + lane;
                    const tf value =
                        output[static_cast<std::size_t>(row) * kChannels +
                               channel];
                    lanes[lane] = value * value;
                }
                warp_sums[warp] = butterfly(lanes)[0];
            }
            const tf sum =
                ((warp_sums[0] + warp_sums[1]) + warp_sums[2]) +
                warp_sums[3];
            const tf scale = rsqrt_accurate(sum + tf{kEpsilon});
            for (std::uint32_t channel = 0u; channel < kHeadDim; ++channel) {
                const std::size_t base =
                    static_cast<std::size_t>(row) * kChannels +
                    block * kHeadDim + channel;
                output[base] = output[base] * scale;
            }
        }
    }
}

void gdn_ab(std::span<const float> input,
            std::span<const std::uint16_t> alpha,
            std::span<const std::uint16_t> beta_weights,
            std::span<const float> dt_bias,
            std::span<const float> a_values,
            std::uint32_t rows, std::vector<tf>& gate,
            std::vector<tf>& beta) {
    gate.resize(static_cast<std::size_t>(rows) * kVHeads);
    beta.resize(static_cast<std::size_t>(rows) * kVHeads);
    for (std::uint32_t head = 0u; head < kVHeads; ++head) {
        std::array<std::array<tf, VERIFY_MAX_T>, 32> alpha_acc{};
        std::array<std::array<tf, VERIFY_MAX_T>, 32> beta_acc{};
        for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
            for (std::uint32_t j = lane; j < 320u; j += 32u) {
                const std::size_t weight_base =
                    static_cast<std::size_t>(head) * N_EMBD + 8u * j;
                for (std::uint32_t t = 0u; t < rows; ++t) {
                    const std::size_t input_base =
                        static_cast<std::size_t>(t) * N_EMBD + 8u * j;
                    tf av = alpha_acc[lane][t];
                    tf bv = beta_acc[lane][t];
                    for (std::uint32_t k = 0u; k < 8u; ++k) {
                        av = twin::fma(
                            tf{bf16_to_float(alpha[weight_base + k])},
                            tf{input[input_base + k]}, av);
                        bv = twin::fma(
                            tf{bf16_to_float(
                                beta_weights[weight_base + k])},
                            tf{input[input_base + k]}, bv);
                    }
                    alpha_acc[lane][t] = av;
                    beta_acc[lane][t] = bv;
                }
            }
        }

        for (std::uint32_t t = 0u; t < rows; ++t) {
            std::array<tf, 32> alpha_lanes{};
            std::array<tf, 32> beta_lanes{};
            for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                alpha_lanes[lane] = alpha_acc[lane][t];
                beta_lanes[lane] = beta_acc[lane][t];
            }
            const tf alpha_dot = butterfly(alpha_lanes)[0];
            const tf beta_dot = butterfly(beta_lanes)[0];
            const std::size_t index =
                static_cast<std::size_t>(t) * kVHeads + head;
            gate[index] =
                softplus(alpha_dot + tf{dt_bias[head]}) *
                tf{a_values[head]};
            beta[index] = sigmoid(beta_dot);
        }
    }
}

void gdn_step_norm(std::vector<float>& state, std::span<const float> h,
                   std::span<const float> gate, std::span<const float> beta,
                   std::span<const float> z, std::span<const float> gamma,
                   std::uint32_t rows, strata::metal::GDNStateMode mode,
                   std::vector<tf>& output, std::vector<tf>& final_state) {
    if (mode != strata::metal::GDNStateMode::commit) {
        output.resize(static_cast<std::size_t>(rows) * kValueDim);
    } else {
        output.clear();
    }
    final_state.resize(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        final_state[i] = tf{state[i]};
    }

    const bool write_state =
        mode != strata::metal::GDNStateMode::window;
    for (std::uint32_t head = 0u; head < kVHeads; ++head) {
        const std::uint32_t qk_head = head % kQKHeads;
        std::array<std::array<std::array<tf, 32>, 4>, kHeadDim> s{};
        for (std::uint32_t column = 0u; column < kHeadDim; ++column) {
            for (std::uint32_t row_group = 0u; row_group < 4u; ++row_group) {
                for (std::uint32_t r = 0u; r < 32u; ++r) {
                    const std::uint32_t state_row = 32u * row_group + r;
                    const std::size_t index =
                        (static_cast<std::size_t>(state_row) * kVHeads +
                         head) *
                            kHeadDim +
                        column;
                    s[column][row_group][r] = tf{state[index]};
                }
            }
        }

        for (std::uint32_t t = 0u; t < rows; ++t) {
            const std::size_t hbase =
                static_cast<std::size_t>(t) * kChannels;
            const std::size_t head_index =
                static_cast<std::size_t>(t) * kVHeads + head;
            const tf decay = twin::accurate_exp(tf{gate[head_index]});
            std::array<tf, kHeadDim> oc{};
            for (std::uint32_t column = 0u; column < kHeadDim; ++column) {
                std::array<tf, 4> red_k{};
                for (std::uint32_t row_group = 0u; row_group < 4u;
                     ++row_group) {
                    tf kv{};
                    for (std::uint32_t r = 0u; r < 32u; ++r) {
                        const std::uint32_t state_row = 32u * row_group + r;
                        const tf k{h[hbase + kQKHeads * kHeadDim +
                                     qk_head * kHeadDim + state_row]};
                        kv = twin::fma(s[column][row_group][r], k, kv);
                    }
                    red_k[row_group] = kv;
                }
                const tf kv_column =
                    ((red_k[0] + red_k[1]) + red_k[2]) + red_k[3];
                const tf v{h[hbase + 2u * kQKHeads * kHeadDim +
                             head * kHeadDim + column]};
                const tf delta = twin::fma(-decay, kv_column, v) *
                                 tf{beta[head_index]};

                std::array<tf, 4> red_o{};
                for (std::uint32_t row_group = 0u; row_group < 4u;
                     ++row_group) {
                    tf accum{};
                    for (std::uint32_t r = 0u; r < 32u; ++r) {
                        const std::uint32_t state_row = 32u * row_group + r;
                        const tf k{h[hbase + kQKHeads * kHeadDim +
                                     qk_head * kHeadDim + state_row]};
                        const tf q{h[hbase + qk_head * kHeadDim + state_row]};
                        s[column][row_group][r] =
                            twin::fma(decay,
                                      s[column][row_group][r], k * delta);
                        accum = twin::fma(
                            s[column][row_group][r], q, accum);
                    }
                    red_o[row_group] = accum;
                }
                oc[column] =
                    (((red_o[0] + red_o[1]) + red_o[2]) + red_o[3]) *
                    rsqrt_accurate(tf{static_cast<float>(kHeadDim)});
            }

            std::array<tf, 4> warp_sums{};
            for (std::uint32_t warp = 0u; warp < 4u; ++warp) {
                std::array<tf, 32> squares{};
                for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                    const tf value = oc[warp * 32u + lane];
                    squares[lane] = value * value;
                }
                warp_sums[warp] = butterfly(squares)[0];
            }
            const tf norm_sum =
                ((warp_sums[0] + warp_sums[1]) + warp_sums[2]) +
                warp_sums[3];
            const tf scale =
                rsqrt_accurate(norm_sum /
                                   tf{static_cast<float>(kHeadDim)} +
                               tf{kEpsilon});
            if (mode != strata::metal::GDNStateMode::commit) {
                for (std::uint32_t column = 0u; column < kHeadDim; ++column) {
                    const std::size_t index =
                        static_cast<std::size_t>(t) * kValueDim +
                        static_cast<std::size_t>(head) * kHeadDim + column;
                    const tf normalized =
                        (oc[column] * scale) * tf{gamma[column]};
                    output[index] =
                        normalized * sigmoid(tf{z[index]});
                }
            }
        }

        if (write_state) {
            for (std::uint32_t column = 0u; column < kHeadDim; ++column) {
                for (std::uint32_t row_group = 0u; row_group < 4u;
                     ++row_group) {
                    for (std::uint32_t r = 0u; r < 32u; ++r) {
                        const std::uint32_t state_row = 32u * row_group + r;
                        const std::size_t index =
                            (static_cast<std::size_t>(state_row) * kVHeads +
                             head) *
                                kHeadDim +
                            column;
                        state[index] = s[column][row_group][r].value;
                        final_state[index] = s[column][row_group][r];
                    }
                }
            }
        }
    }
}

void gdn_conv_commit(std::vector<float>& history,
                     std::span<const float> qkv, std::uint32_t rows) {
    if (rows == 0u) return;
    for (std::uint32_t channel = 0u; channel < kChannels; ++channel) {
        const std::size_t base = static_cast<std::size_t>(channel) * 3u;
        std::array<float, 3> next{};
        for (std::uint32_t j = 0u; j < next.size(); ++j) {
            const std::uint32_t source = rows + j;
            next[j] = source < 3u
                ? history[base + source]
                : qkv[static_cast<std::size_t>(source - 3u) * kChannels +
                      channel];
        }
        for (std::uint32_t j = 0u; j < next.size(); ++j) {
            history[base + j] = next[j];
        }
    }
}

}  // namespace oracle
