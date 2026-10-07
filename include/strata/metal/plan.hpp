// include/strata/metal/plan.hpp - memory plan computation with no Metal dependency.
#pragma once

#include "strata/metal/geometry.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {

enum class KvFormat : std::uint8_t { Fp16, Int8, K8v4, Q4_0 };
enum class PrefillMode : std::uint8_t { None, Auto, Fixed };

struct PlanOptions {
    std::uint64_t working_set_bytes = 0;
    std::uint64_t ram_bytes = 0;
    std::uint64_t available_bytes = 0;
    bool available_is_override = false;
    std::string gpu = "simulated";
    std::uint64_t reserve_mib = 700;
    std::uint32_t max_context = 4096;
    std::uint32_t mtp_window = 32'768;
    KvFormat kv = KvFormat::Fp16;
    bool spec_enabled = false;
    PrefillMode prefill_mode = PrefillMode::None;
    std::uint32_t prefill_chunk = 0;
    std::uint32_t prefill_auto_max = 8192;
    bool prefill_bf16x2 = false;
    std::int64_t expert_cache = -1;  // -1 means auto/all; 0 deliberately forces no residency.
    bool resident_budget_is_set = false;
    std::uint64_t resident_budget_bytes = 0;
    std::uint64_t expert_slots = 0;
    bool expert_cache_per_layer = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> expert_order;
};

struct PlanResult {
    bool fits = false;
    std::string mode = "tiered";
    std::string gpu = "simulated";
    KvFormat kv = KvFormat::Fp16;
    std::uint32_t context = 0;
    std::string reason;
    std::vector<std::string> warnings;
    std::uint64_t w_bytes = 0;
    std::uint64_t reserve_bytes = 0;
    std::uint64_t ram_bytes = 0;
    std::int64_t available_bytes = -1;
    std::uint64_t dense_bytes = 0;
    std::uint64_t unread_bytes = 0;
    std::uint64_t pack_bytes = 0;
    std::uint64_t mtp_bytes = 0;
    std::uint64_t mtp_bind_bytes = 0;
    std::uint32_t mtp_bind_capacity = 0;
    /// Fixed and selected-chunk MtpParams slot-array bytes when spec is enabled.
    std::uint64_t mtp_params_bytes = 0;
    /// Exact subcomponents of fixed_bytes; informational and not added again.
    std::uint64_t session_state_bytes = 0;
    std::uint64_t window_scratch_bytes = 0;
    std::uint64_t fixed_bytes = 0;
    std::uint64_t kv_bytes = 0;
    std::uint64_t mtp_kv_bytes = 0;
    std::uint64_t prefill_bytes = 0;
    std::uint32_t prefill_chunk = 0;
    std::uint64_t experts_total_bytes = 0;
    std::uint64_t experts_resident_bytes = 0;
    std::uint64_t experts_total = 0;
    std::uint64_t experts_resident = 0;
    std::uint64_t slots = 0;
    std::uint64_t slot_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::int64_t free_bytes = 0;
    std::uint64_t w_full_bytes = 0;
    std::uint64_t ram_bytes_used = 0;
    std::uint64_t anonymous_bytes_used = 0;
    bool ram_warning = false;
    bool kv_warning = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> resident_order;
};

bool make_plan(const Geometry& geometry, const PlanOptions& options, PlanResult& result,
               std::string& error);
/// Validate the MTP byte fields in a plan against its inspected geometry and
/// effective rounded bind-window capacity.
bool validate_mtp_memory_plan(const Geometry& geometry, const PlanResult& plan,
                              std::string& error);
std::string plan_line(const PlanResult& plan);
std::uint64_t prefill_bytes(std::uint32_t chunk, std::uint32_t max_context, bool bf16x2);
const char* kv_format_name(KvFormat format);

}  // namespace strata::metal
