// src/metal/engine/plan.cpp - U1's byte-exact planner and stable PLAN line.
#include "strata/metal/plan.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/mtp_kv.hpp"
#include "strata/metal/mtp_params_layout.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/session_state.hpp"
#include "strata/metal/window_scratch.hpp"
#include "../kernels/common/params.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>

namespace strata::metal {
namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024;
constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;
constexpr std::uint64_t kPage = 16ull * 1024;
constexpr std::uint64_t kFixedBytes = 256ull * 1024 * 1024;
constexpr std::uint64_t kDefaultMtpWindow = 32'768;

static_assert(kPrefillMaxShaderRows ==
              (std::uint64_t{1u} << 32u) / HC_DIM);

bool add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
    return value / divisor + (value % divisor != 0);
}

bool mtp_prefill_params_bytes(std::uint32_t prefill_chunk,
                              std::uint64_t& bytes, std::string& error) {
    MtpParamsLayout baseline;
    MtpParamsLayout layout;
    if (!make_mtp_params_layout(0, baseline, error) ||
        !make_mtp_params_layout(prefill_chunk, layout, error)) {
        return false;
    }
    if (layout.allocation_bytes < baseline.allocation_bytes) {
        error = "MTP prefill parameter bytes are smaller than the short-read baseline";
        return false;
    }
    bytes = layout.allocation_bytes - baseline.allocation_bytes;
    return true;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) return value;
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - remainder)) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return value + alignment - remainder;
}

bool mtp_bind_variable_bytes(std::uint64_t cap, std::uint64_t& bytes) {
    const std::uint64_t chunks = cap / 64;
    std::uint64_t scratch_floats = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t ids_bytes = 0;
    return cap != 0 && cap % 64 == 0 && multiply(chunks, 24 * 258, scratch_floats) &&
           add(scratch_floats, 64, scratch_floats) &&
           multiply(scratch_floats, sizeof(float), scratch_bytes) &&
           multiply(cap, sizeof(std::uint32_t), ids_bytes) &&
           add(scratch_bytes, ids_bytes, bytes);
}

bool mtp_bind_bytes_for_capacity(const Geometry& geometry, std::uint64_t cap,
                                 std::uint64_t& bytes, std::string& error) {
    bytes = 0;
    if (geometry.mtp_bytes == 0 || geometry.mtp_bind_bytes == 0) {
        error = "speculative planning requires MTP geometry";
        return false;
    }
    if (cap == 0 || cap > 262'144 || cap % 64 != 0) {
        error = "MTP bind capacity must be a multiple of 64 within the context limit";
        return false;
    }
    std::uint64_t default_variable = 0;
    std::uint64_t requested_variable = 0;
    if (!mtp_bind_variable_bytes(kDefaultMtpWindow, default_variable) ||
        !mtp_bind_variable_bytes(cap, requested_variable) ||
        geometry.mtp_bind_bytes < default_variable) {
        error = "MTP bind geometry is smaller than the supported default-window scratch";
        return false;
    }
    const std::uint64_t fixed = geometry.mtp_bind_bytes - default_variable;
    if (!add(fixed, requested_variable, bytes)) {
        error = "MTP bind byte count overflows";
        return false;
    }
    return true;
}

bool planned_mtp_bind_bytes(const Geometry& geometry, const PlanOptions& options,
                            std::uint64_t& bytes, std::uint32_t& capacity,
                            std::string& error) {
    bytes = 0;
    capacity = 0;
    if (!options.spec_enabled) return true;
    const std::uint64_t requested_cap =
        options.mtp_window > 0 && options.mtp_window < options.max_context ?
            options.mtp_window : options.max_context;
    const std::uint64_t cap = align_up(requested_cap, 64);
    if (cap > std::numeric_limits<std::uint32_t>::max() ||
        !mtp_bind_bytes_for_capacity(geometry, cap, bytes, error)) {
        if (error.empty()) error = "MTP bind capacity cannot be represented";
        return false;
    }
    capacity = static_cast<std::uint32_t>(cap);
    return true;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> fallback_order(const Geometry& geometry) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;
    order.reserve(static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert);
    for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
        for (std::uint32_t layer = 0; layer < geometry.n_layers; ++layer) {
            order.emplace_back(layer, expert);
        }
    }
    return order;
}

bool make_order(const Geometry& geometry, const PlanOptions& options,
                std::vector<std::pair<std::uint32_t, std::uint32_t>>& order,
                std::string& error) {
    const auto fallback = fallback_order(geometry);
    std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
    order.clear();
    const auto& candidate = options.expert_order.empty() ? fallback : options.expert_order;
    for (const auto& expert : candidate) {
        if (expert.first >= geometry.n_layers || expert.second >= geometry.n_expert) {
            error = "expert profile contains an out-of-range (layer, expert) pair";
            return false;
        }
        if (!seen.insert(expert).second) {
            error = "expert profile contains a duplicate (layer, expert) pair";
            return false;
        }
        order.push_back(expert);
    }
    for (const auto& expert : fallback) {
        if (seen.insert(expert).second) order.push_back(expert);
    }
    return true;
}

bool sum_first(const Geometry& geometry, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& order,
               std::uint64_t count, std::uint64_t& bytes) {
    bytes = 0;
    count = std::min<std::uint64_t>(count, order.size());
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto layer = order[static_cast<std::size_t>(i)].first;
        if (!add(bytes, geometry.layers[layer].blob_bytes, bytes)) return false;
    }
    return true;
}

std::uint64_t slot_pool_bytes(const std::vector<std::uint64_t>& resident_by_layer,
                              std::uint32_t n_expert, std::uint64_t slot_bytes,
                              std::uint64_t explicit_minimum, std::uint64_t* slots_out) {
    std::uint64_t max_misses = 0;
    for (const auto count : resident_by_layer) {
        const std::uint64_t misses = n_expert - count;
        max_misses = std::max(max_misses, misses);
    }
    std::uint64_t slots = std::max(2 * max_misses, explicit_minimum);
    std::uint64_t bytes = 0;
    if (!multiply(slots, slot_bytes, bytes)) bytes = std::numeric_limits<std::uint64_t>::max();
    if (slots_out != nullptr) *slots_out = slots;
    return bytes;
}

struct Admission {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> residents;
    std::vector<std::uint64_t> per_layer;
    std::uint64_t bytes = 0;
};

bool admit_experts(const Geometry& geometry, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& order,
                   std::uint64_t cap_bytes, std::uint64_t A, std::uint64_t P, std::uint64_t budget,
                   std::uint64_t slot_bytes, std::uint64_t explicit_slots,
                   std::uint64_t count_cap, std::uint64_t quota,
                   Admission& admission) {
    admission = {};
    admission.per_layer.assign(geometry.n_layers, 0);
    for (const auto& expert : order) {
        if (admission.residents.size() >= count_cap) break;
        const std::uint32_t layer = expert.first;
        const std::uint64_t blob = geometry.layers[layer].blob_bytes;
        if (quota != std::numeric_limits<std::uint64_t>::max() &&
            admission.per_layer[layer] >= quota) {
            continue;
        }
        std::uint64_t next_expert_bytes = 0;
        if (!add(admission.bytes, blob, next_expert_bytes) || next_expert_bytes > cap_bytes) continue;
        auto next_per_layer = admission.per_layer;
        ++next_per_layer[layer];
        std::uint64_t next_slots = 0;
        const std::uint64_t pool = slot_pool_bytes(next_per_layer, geometry.n_expert, slot_bytes,
                                                   explicit_slots, &next_slots);
        std::uint64_t need = A;
        if (!add(need, P, need) || !add(need, next_expert_bytes, need) ||
            !add(need, pool, need) || need > budget) {
            continue;
        }
        admission.bytes = next_expert_bytes;
        admission.per_layer[layer] = next_per_layer[layer];
        admission.residents.push_back(expert);
    }
    return true;
}

std::uint64_t auto_minimum_chunk(std::uint32_t context, std::uint32_t cap) {
    const std::uint64_t rounded_context = align_up(context, 1024);
    return std::min<std::uint64_t>({1024, cap, rounded_context});
}

std::string type_reason(std::string type) {
    if (type.empty()) return "unsupported_type_unknown";
    if (type[0] >= 'A' && type[0] <= 'Z') type[0] = static_cast<char>(type[0] - 'A' + 'a');
    for (char& c : type) if (c == ' ') c = '_';
    return "unsupported_type_" + type;
}

bool is_supported_type(const std::string& type) {
    static const std::set<std::string> supported = {
        "q2_0", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs",
        "iq4_nl", "q3_k", "q4_k", "q5_k", "q6_k", "q4_0", "q5_0", "q8_0",
        "bf16", "f16", "f32"};
    return supported.contains(type);
}

}  // namespace

const char* kv_format_name(KvFormat format) {
    switch (format) {
        case KvFormat::Fp16: return "fp16";
        case KvFormat::Int8: return "int8";
        case KvFormat::K8v4: return "k8v4";
        case KvFormat::Q4_0: return "q4_0";
    }
    return "fp16";
}

bool validate_mtp_memory_plan(const Geometry& geometry, const PlanResult& plan,
                              std::string& error) {
    error.clear();
    if (!plan.fits || plan.mtp_bytes == 0 || plan.mtp_bind_bytes == 0 ||
        plan.mtp_bind_capacity == 0 || plan.mtp_kv_bytes == 0 ||
        plan.mtp_params_bytes == 0) {
        error = "MTP runtime storage needs a fitting speculative memory plan";
        return false;
    }
    if (geometry.mtp_bytes != plan.mtp_bytes) {
        error = "MTP runtime file bytes do not match the inspected geometry";
        return false;
    }
    if (plan.context == 0 ||
        plan.mtp_bind_capacity > align_up(plan.context, 64)) {
        error = "MTP bind capacity exceeds the rounded planned context";
        return false;
    }
    std::uint64_t expected_bind_bytes = 0;
    if (!mtp_bind_bytes_for_capacity(geometry, plan.mtp_bind_capacity,
                                     expected_bind_bytes, error)) {
        return false;
    }
    if (expected_bind_bytes != plan.mtp_bind_bytes) {
        error = "MTP bind scratch bytes do not match the inspected geometry and capacity";
        return false;
    }
    MtpParamsLayout expected_params;
    if (!make_mtp_params_layout(plan.prefill_chunk, expected_params, error)) {
        return false;
    }
    if (expected_params.allocation_bytes != plan.mtp_params_bytes) {
        error = "MTP parameter bytes do not match the fixed passes and prefill groups";
        return false;
    }
    MtpKvLayout expected_kv;
    if (!make_mtp_kv_layout(plan.context, plan.kv, expected_kv, error)) return false;
    if (expected_kv.allocation_bytes != plan.mtp_kv_bytes) {
        error = "MTP K/V bytes do not match the memory plan";
        return false;
    }
    error.clear();
    return true;
}

std::uint64_t prefill_bytes(std::uint32_t chunk, std::uint32_t max_context, bool bf16x2) {
    if (chunk == 0 || max_context == 0) return 0;
    const std::uint64_t P = bf16x2 ? 272'424 : 267'304;
    const std::uint64_t max_blocks = static_cast<std::uint64_t>(max_context) / 4 + 2;
    std::uint64_t fixed = 64ull * kMiB;
    fixed += 32ull * 817'600;                   // one attention partial buffer for a 32-row batch
    fixed += 1'024ull * max_blocks;              // block scores
    fixed += 48ull * 6'400;                      // PfPlan
    fixed += 8ull * 512;                         // fixed tile-list part
    fixed += 65'536;                             // alignment and final partial-tile slack
    fixed += 4'096;                              // PrefillParams and batch slots
    const std::uint64_t tiles = ceil_div(10ull * chunk, 64);
    if (chunk > (std::numeric_limits<std::uint64_t>::max() - fixed) / P) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(chunk) * P + 8ull * tiles + fixed;
}

bool make_plan(const Geometry& geometry, const PlanOptions& options, PlanResult& result,
               std::string& error) {
    result = {};
    error.clear();
    if (!validate_geometry(geometry, error)) return false;
    if (options.working_set_bytes == 0 || options.ram_bytes == 0 ||
        options.max_context == 0 || options.max_context > 262'144 ||
        options.reserve_mib > std::numeric_limits<std::uint64_t>::max() / kMiB ||
        options.expert_cache < -1) {
        error = "plan inputs contain a zero or out-of-range budget or context";
        return false;
    }
    if (options.prefill_mode != PrefillMode::None &&
        options.prefill_mode != PrefillMode::Auto &&
        options.prefill_mode != PrefillMode::Fixed) {
        error = "unknown prefill mode";
        return false;
    }
    if (static_cast<unsigned>(options.kv) > static_cast<unsigned>(KvFormat::Q4_0)) {
        error = "unknown KV format";
        return false;
    }
    if ((options.prefill_mode == PrefillMode::Fixed && options.prefill_chunk == 0) ||
        (options.prefill_mode == PrefillMode::Auto && options.prefill_auto_max < 1024)) {
        error = options.prefill_mode == PrefillMode::Fixed ?
            "fixed prefill chunk must be positive" :
            "automatic prefill maximum must be at least 1024";
        return false;
    }
    if (options.prefill_mode == PrefillMode::Fixed &&
        options.prefill_chunk > kPrefillMaxShaderRows) {
        error = "fixed prefill chunk exceeds the Metal shader row limit";
        return false;
    }

    result.w_bytes = options.working_set_bytes;
    result.gpu = options.gpu;
    result.kv = options.kv;
    result.context = options.max_context;
    result.reserve_bytes = options.reserve_mib * kMiB;
    result.ram_bytes = options.ram_bytes;
    result.available_bytes = options.available_is_override ? -1 :
                             static_cast<std::int64_t>(std::min<std::uint64_t>(
                                 options.available_bytes,
                                 static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
    result.dense_bytes = geometry.nonexpert_bytes;
    result.pack_bytes = geometry.dense_pack_bytes;
    result.mtp_bytes = options.spec_enabled ? geometry.mtp_bytes : 0;
    if (!planned_mtp_bind_bytes(geometry, options, result.mtp_bind_bytes,
                                result.mtp_bind_capacity, error)) return false;
    std::uint64_t fixed_mtp_params_bytes = 0;
    if (options.spec_enabled) {
        MtpParamsLayout fixed_params;
        if (!make_mtp_params_layout(0, fixed_params, error)) return false;
        fixed_mtp_params_bytes = fixed_params.allocation_bytes;
        result.mtp_params_bytes = fixed_mtp_params_bytes;
    }
    result.fixed_bytes = kFixedBytes;
    result.experts_total = static_cast<std::uint64_t>(geometry.n_layers) * geometry.n_expert;
    QsaKvLayout qsa_kv_layout;
    if (!make_qsa_kv_layout(options.max_context, options.kv, qsa_kv_layout, error)) return false;
    result.kv_bytes = qsa_kv_layout.allocation_bytes;
    if (is_qwen4exp_catalog_family(geometry.family)) {
        if (!is_qwen4exp_runtime_geometry(geometry)) {
            error = "Qwen4exp planning requires the supported 48-layer runtime geometry";
            return false;
        }
        SessionStateLayout session_layout;
        WindowScratchLayout scratch_layout;
        if (!make_session_state_layout(geometry, session_layout, error) ||
            !make_window_scratch_layout(geometry, qsa_kv_layout.cell_capacity,
                                        scratch_layout, error)) {
            return false;
        }
        result.session_state_bytes = session_layout.allocation_bytes;
        result.window_scratch_bytes = scratch_layout.allocation_bytes;
        std::uint64_t exact_fixed_subtotal = 0;
        if (!add(result.session_state_bytes, result.window_scratch_bytes,
                 exact_fixed_subtotal)) {
            error = "exact fixed-state byte count overflows";
            return false;
        }
        if (exact_fixed_subtotal > result.fixed_bytes) {
            error = "exact session and Window scratch bytes exceed the fixed-memory allowance";
            return false;
        }
    }
    if (options.spec_enabled) {
        MtpKvLayout mtp_kv_layout;
        if (!make_mtp_kv_layout(options.max_context, options.kv, mtp_kv_layout, error)) return false;
        result.mtp_kv_bytes = mtp_kv_layout.allocation_bytes;
    }
    for (const auto& type : geometry.types) {
        if (!is_supported_type(type)) {
            result.reason = type_reason(type);
            return true;
        }
    }
    for (const auto& layer : geometry.layers) {
        std::uint64_t layer_experts = 0;
        if (!multiply(layer.blob_bytes, geometry.n_expert, layer_experts) ||
            !add(result.experts_total_bytes, layer_experts, result.experts_total_bytes)) {
            error = "expert byte count overflows";
            return false;
        }
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;
    if (!make_order(geometry, options, order, error)) return false;

    const std::uint64_t reserve = result.reserve_bytes;
    const std::uint64_t budget = result.w_bytes > reserve ? result.w_bytes - reserve : 0;
    std::uint64_t A = 0;
    for (const std::uint64_t component : {result.dense_bytes, result.pack_bytes, result.mtp_bytes,
                                          result.mtp_bind_bytes, result.mtp_params_bytes,
                                          result.fixed_bytes, result.kv_bytes, result.mtp_kv_bytes}) {
        if (!add(A, component, A)) {
            error = "always-resident byte count overflows";
            return false;
        }
    }

    std::uint64_t cap_bytes = result.experts_total_bytes;
    const std::uint64_t count_cap = options.expert_cache < 0 ? result.experts_total :
        std::min<std::uint64_t>(static_cast<std::uint64_t>(options.expert_cache),
                                result.experts_total);
    if (options.expert_cache >= 0) {
        std::uint64_t first_n_bytes = 0;
        if (!sum_first(geometry, order,
                       count_cap, first_n_bytes)) {
            error = "expert count cap overflows";
            return false;
        }
        cap_bytes = std::min(cap_bytes, first_n_bytes);
    }
    if (options.resident_budget_is_set) {
        cap_bytes = std::min(cap_bytes, options.resident_budget_bytes);
    }

    std::uint64_t prefill_minimum = 0;
    std::uint32_t prefill_minimum_chunk = 0;
    if (options.prefill_mode == PrefillMode::Fixed) {
        prefill_minimum_chunk = options.prefill_chunk;
        prefill_minimum = prefill_bytes(options.prefill_chunk, options.max_context,
                                        options.prefill_bf16x2);
    } else if (options.prefill_mode == PrefillMode::Auto) {
        prefill_minimum_chunk = static_cast<std::uint32_t>(
            auto_minimum_chunk(options.max_context, options.prefill_auto_max));
        prefill_minimum = prefill_bytes(prefill_minimum_chunk,
                                        options.max_context, options.prefill_bf16x2);
    }

    std::uint64_t minimum_prefill_params_bytes = 0;
    if (options.spec_enabled && prefill_minimum_chunk != 0 &&
        !mtp_prefill_params_bytes(prefill_minimum_chunk,
                                  minimum_prefill_params_bytes, error)) {
        return false;
    }
    std::uint64_t full_need = A;
    if (!add(full_need, geometry.wrapped_unread_bytes, full_need) ||
        !add(full_need, prefill_minimum, full_need) ||
        !add(full_need, minimum_prefill_params_bytes, full_need) ||
        !add(full_need, result.experts_total_bytes, full_need)) {
        error = "full-residency byte count overflows";
        return false;
    }
    std::uint64_t w_full = 0;
    if (!add(reserve, full_need, w_full)) {
        error = "full-residency working-set threshold overflows";
        return false;
    }
    result.w_full_bytes = w_full;
    const bool full_mode = cap_bytes == result.experts_total_bytes && full_need <= budget;
    result.mode = full_mode ? "full" : "tiered";

    if (options.prefill_mode == PrefillMode::Fixed) {
        result.prefill_chunk = options.prefill_chunk;
    } else if (options.prefill_mode == PrefillMode::Auto) {
        const std::uint64_t context_cap = align_up(options.max_context, 1024);
        std::vector<std::uint32_t> candidates = {8192, 6144, 4096, 2048, 1024};
        if (options.prefill_auto_max >= 16'384) candidates.insert(candidates.begin(), 16'384);
        if (options.prefill_auto_max >= 32'768) candidates.insert(candidates.begin(), 32'768);
        if (full_mode) {
            for (const std::uint32_t candidate : candidates) {
                if (candidate > options.prefill_auto_max || candidate > context_cap) continue;
                const std::uint64_t bytes = prefill_bytes(candidate, options.max_context,
                                                          options.prefill_bf16x2);
                std::uint64_t params_bytes = 0;
                if (options.spec_enabled &&
                    !mtp_prefill_params_bytes(candidate, params_bytes, error)) {
                    return false;
                }
                std::uint64_t need = A;
                if (add(need, geometry.wrapped_unread_bytes, need) &&
                    add(need, bytes, need) && add(need, params_bytes, need) &&
                    add(need, result.experts_total_bytes, need) &&
                    need <= budget) {
                    result.prefill_chunk = candidate;
                    break;
                }
            }
            if (result.prefill_chunk == 0) {
                error = "minimum automatic prefill chunk is not representable";
                return false;
            }
        } else {
            const std::uint32_t preferred = budget >= A && budget - A >= 24ull * kGiB ? 2048 : 1024;
            result.prefill_chunk = preferred <= options.prefill_auto_max && preferred <= context_cap ?
                preferred : 1024;
            if (result.prefill_chunk > options.prefill_auto_max ||
                result.prefill_chunk > context_cap) {
                error = "no automatic prefill candidate is allowed";
                return false;
            }
        }
    }
    if (result.prefill_chunk != 0) {
        result.prefill_bytes = prefill_bytes(result.prefill_chunk, options.max_context,
                                             options.prefill_bf16x2);
    }
    std::uint64_t prefill_params_bytes = 0;
    if (options.spec_enabled && result.prefill_chunk != 0 &&
        !mtp_prefill_params_bytes(result.prefill_chunk,
                                  prefill_params_bytes, error)) {
        return false;
    }
    if (options.spec_enabled &&
        !add(fixed_mtp_params_bytes, prefill_params_bytes,
             result.mtp_params_bytes)) {
        error = "MTP parameter byte count overflows";
        return false;
    }

    if (result.mode == "full") {
        result.unread_bytes = geometry.wrapped_unread_bytes;
        result.experts_resident_bytes = result.experts_total_bytes;
        result.experts_resident = result.experts_total;
        result.resident_order = order;
    } else {
        result.unread_bytes = 0;
        std::uint64_t max_blob = 0;
        for (const auto& layer : geometry.layers) max_blob = std::max(max_blob, layer.blob_bytes);
        const std::uint64_t one_slot_bytes = align_up(max_blob, kPage);

        std::uint64_t P = result.prefill_bytes;
        if (options.prefill_mode == PrefillMode::None) P = 0;
        if (options.spec_enabled &&
            !add(P, prefill_params_bytes, P)) {
            error = "tiered prefill and MTP parameter byte count overflows";
            return false;
        }
        if (options.expert_cache_per_layer) {
            Admission first;
            admit_experts(geometry, order, cap_bytes, A, P, budget, one_slot_bytes,
                          options.expert_slots, count_cap,
                          std::numeric_limits<std::uint64_t>::max(), first);
            const std::uint64_t quota = first.residents.size() / geometry.n_layers;
            Admission second;
            admit_experts(geometry, order, cap_bytes, A, P, budget, one_slot_bytes,
                          options.expert_slots, count_cap, quota, second);
            result.resident_order = std::move(second.residents);
            result.experts_resident_bytes = second.bytes;
            result.experts_resident = result.resident_order.size();
        } else {
            Admission admitted;
            admit_experts(geometry, order, cap_bytes, A, P, budget, one_slot_bytes,
                          options.expert_slots, count_cap,
                          std::numeric_limits<std::uint64_t>::max(), admitted);
            result.resident_order = std::move(admitted.residents);
            result.experts_resident_bytes = admitted.bytes;
            result.experts_resident = result.resident_order.size();
        }
        const std::uint64_t explicit_min = options.expert_slots;
        std::vector<std::uint64_t> resident_by_layer(geometry.n_layers, 0);
        for (const auto& expert : result.resident_order) ++resident_by_layer[expert.first];
        result.slot_bytes = slot_pool_bytes(resident_by_layer, geometry.n_expert,
                                            one_slot_bytes, explicit_min, &result.slots);
    }

    result.fits = true;

    const std::uint64_t pool_bytes = result.mode == "tiered" ? result.slot_bytes : 0;
    std::uint64_t total = 0;
    for (const std::uint64_t component : {result.dense_bytes, result.unread_bytes, result.pack_bytes,
                                          result.mtp_bytes, result.mtp_bind_bytes,
                                          result.mtp_params_bytes, result.fixed_bytes, result.kv_bytes,
                                          result.mtp_kv_bytes, result.prefill_bytes, pool_bytes,
                                          result.experts_resident_bytes}) {
        if (!add(total, component, total)) {
            error = "planned total byte count overflows";
            return false;
        }
    }
    result.total_bytes = total;
    result.ram_bytes_used = total - result.unread_bytes;
    result.anonymous_bytes_used = result.ram_bytes_used - result.mtp_bytes;
    result.free_bytes = budget >= total ?
        static_cast<std::int64_t>(std::min<std::uint64_t>(budget - total,
                                   static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) :
        -static_cast<std::int64_t>(std::min<std::uint64_t>(total - budget,
                                    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));

    if (result.fits && total > budget) {
        result.fits = false;
        std::uint64_t minimum = A;
        if (!add(minimum, result.prefill_bytes, minimum) ||
            !add(minimum, prefill_params_bytes, minimum)) {
            error = "minimum plan byte count overflows";
            return false;
        }
        if (result.mode == "tiered") {
            std::uint64_t max_blob = 0;
            for (const auto& layer : geometry.layers) max_blob = std::max(max_blob, layer.blob_bytes);
            const std::uint64_t minimum_slots = std::max<std::uint64_t>(
                2ull * geometry.n_expert, options.expert_slots);
            std::uint64_t S0 = 0;
            if (!multiply(minimum_slots, align_up(max_blob, kPage), S0) || !add(minimum, S0, minimum)) {
                error = "minimum slot-pool byte count overflows";
                return false;
            }
        }
        const std::uint64_t needs_mib = ceil_div(minimum, kMiB);
        const std::uint64_t available_mib = ceil_div(budget, kMiB);
        result.reason = "the_model_does_not_fit:_needs_" + std::to_string(needs_mib) +
                        "_MiB_of_GPU_memory_with_no_expert_resident,_" +
                        std::to_string(available_mib) + "_MiB_available";
        std::uint64_t kv_total = result.kv_bytes;
        if (!add(kv_total, result.mtp_kv_bytes, kv_total)) {
            error = "KV byte count overflows";
            return false;
        }
        result.kv_warning = A > budget && kv_total > A / 4;
    }

    if (result.mode == "tiered" && options.expert_order.empty()) {
        result.warnings.emplace_back(
            "no expert profile; the resident set is arbitrary and the hit rate will be low");
    }
    const std::uint64_t ram_limit = result.ram_bytes > 6ull * kGiB ? result.ram_bytes - 6ull * kGiB : 0;
    const bool physical_ram_warning = result.ram_bytes_used > ram_limit;
    const bool available_ram_warning = !options.available_is_override && result.mode == "tiered" &&
        static_cast<std::uint64_t>(std::max<std::int64_t>(0, result.available_bytes)) <
            result.anonymous_bytes_used;
    if (physical_ram_warning || available_ram_warning) {
        result.ram_warning = true;
        std::string message =
            "the plan leaves less than 6 GiB of RAM for macOS and other apps; expect memory pressure";
        if (available_ram_warning) {
            message += " (" + std::to_string(ceil_div(result.anonymous_bytes_used, kMiB)) +
                       " MiB planned, " +
                       std::to_string(ceil_div(static_cast<std::uint64_t>(result.available_bytes), kMiB)) +
                       " MiB available)";
        }
        result.warnings.push_back(std::move(message));
    }

    error.clear();
    return true;
}

std::string plan_line(const PlanResult& plan) {
    auto ceil_mib = [](std::uint64_t bytes) { return ceil_div(bytes, kMiB); };
    std::uint64_t total_mib = ceil_mib(plan.dense_bytes) + ceil_mib(plan.unread_bytes) +
        ceil_mib(plan.pack_bytes) + ceil_mib(plan.mtp_bytes) + ceil_mib(plan.mtp_bind_bytes) +
        ceil_mib(plan.mtp_params_bytes) + ceil_mib(plan.fixed_bytes) + ceil_mib(plan.kv_bytes) +
        ceil_mib(plan.mtp_kv_bytes) + ceil_mib(plan.prefill_bytes) +
        ceil_mib(plan.mode == "tiered" ? plan.slot_bytes : 0) +
        ceil_mib(plan.experts_resident_bytes);
    const std::uint64_t w_mib = ceil_mib(plan.w_bytes);
    const std::uint64_t reserve_mib = ceil_mib(plan.reserve_bytes);
    const std::int64_t free_mib = static_cast<std::int64_t>(w_mib) -
                                  static_cast<std::int64_t>(reserve_mib) -
                                  static_cast<std::int64_t>(total_mib);
    std::ostringstream out;
    out << "PLAN fits=" << (plan.fits ? 1 : 0)
        << " mode=" << plan.mode
        << " gpu=" << plan.gpu
        << " w_mib=" << w_mib
        << " reserve_mib=" << reserve_mib
        << " ram_mib=" << ceil_mib(plan.ram_bytes)
        << " avail_mib=" << (plan.available_bytes < 0 ? -1 : static_cast<std::int64_t>(
             ceil_mib(static_cast<std::uint64_t>(plan.available_bytes))))
        << " dense_mib=" << ceil_mib(plan.dense_bytes)
        << " unread_mib=" << ceil_mib(plan.unread_bytes)
        << " pack_mib=" << ceil_mib(plan.pack_bytes)
        << " mtp_mib=" << ceil_mib(plan.mtp_bytes)
        << " mtp_bind_mib=" << ceil_mib(plan.mtp_bind_bytes)
        << " fixed_mib=" << ceil_mib(plan.fixed_bytes)
        << " kv_mib=" << ceil_mib(plan.kv_bytes)
        << " mtp_kv_mib=" << ceil_mib(plan.mtp_kv_bytes)
        << " kv=" << kv_format_name(plan.kv)
        << " context=" << plan.context
        << " prefill_chunk=" << plan.prefill_chunk
        << " prefill_mib=" << ceil_mib(plan.prefill_bytes)
        << " experts_total=" << plan.experts_total
        << " experts_total_mib=" << ceil_mib(plan.experts_total_bytes)
        << " experts_resident=" << plan.experts_resident
        << " experts_resident_mib=" << ceil_mib(plan.experts_resident_bytes)
        << " slots=" << plan.slots
        << " slot_mib=" << ceil_mib(plan.mode == "tiered" ? plan.slot_bytes : 0)
        << " ssd_experts=" << (plan.experts_total - plan.experts_resident)
        << " total_mib=" << total_mib
        << " free_mib=" << free_mib
        << " ram_warning=" << (plan.ram_warning ? 1 : 0)
        << " w_full_mib=" << ceil_mib(plan.w_full_bytes);
    if (!plan.fits && !plan.reason.empty()) out << " reason=" << plan.reason;
    out << " mtp_params_mib=" << ceil_mib(plan.mtp_params_bytes);
    out << " session_state_mib=" << ceil_mib(plan.session_state_bytes);
    out << " window_scratch_mib=" << ceil_mib(plan.window_scratch_bytes);
    return out.str();
}

}  // namespace strata::metal
