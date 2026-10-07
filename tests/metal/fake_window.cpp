// tests/metal/fake_window.cpp - deterministic logits and counters for H6.
#include "fake_window.hpp"

#include <thread>

namespace strata::metal::test {

FakeWindow::FakeWindow(bool fail_gpu, std::chrono::milliseconds delay, bool report_length_after_stop)
    : fail_gpu_(fail_gpu), delay_(delay), report_length_after_stop_(report_length_after_stop) {
    info_.context = 32;
    info_.kv = "fp16";
    info_.expert_slots = 5;
    info_.expert_cache_mib = 123;
    info_.spec = 0;
    info_.mtp_max = 0;
    info_.lookup = 3;
    info_.vram_free_mib = 512;
    info_.cvec = "none";
    info_.arena_mib = 2048;
    info_.spec_min_p = 0.25;
    info_.conversation_cache_mib = 64;
    info_.conversation_cache_slots = 4;
    info_.conversation_cache_min_free_mib = 512;
    info_.engine = "0.1.34";
    info_.metal_rev = 7;
    info_.stop_eos = true;
    info_.gpu = "Apple M5 Pro";
    info_.gpu_budget_mib = 4096;
    info_.residency = "full";
    info_.ssd_experts = 0;
    info_.plan_mib = 12345;
    info_.vram_reserve_mib = 700;
}

bool FakeWindow::run_request(const ServeRequest& request, const ServeCallbacks& callbacks,
                             ServeOutcome& outcome, ServeFailure& failure) {
    if (request.image &&
        (request.images.size() != 2 || request.images[0].n_tokens != 4 ||
         request.images[0].nx != 2 || request.images[0].ny != 2 ||
         request.images[1].n_tokens != 2 || request.images[1].nx != 2 || request.images[1].ny != 1 ||
         request.mrope_positions != std::vector<std::array<std::int32_t, 3>>{
                                       {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                       {2, 2, 2}, {3, 3, 3}, {3, 3, 4}, {5, 5, 5}})) {
        failure.kind = ServeFailureKind::Request;
        failure.message = "fake window received invalid SVE1 rows or M-RoPE positions";
        return false;
    }
    if (fail_gpu_) {
        failure.kind = ServeFailureKind::Gpu;
        failure.label = "fake_window";
        failure.message = "synthetic command-buffer failure";
        return false;
    }

    if (!callbacks.resume(0)) return false;
    const auto prompt_tokens = static_cast<std::int64_t>(request.token_ids.size());
    const double prompt_rate = static_cast<double>(prompt_tokens) * 250.0;
    if (!callbacks.prompt_progress(prompt_tokens, prompt_tokens, 4.0, prompt_rate) ||
        !callbacks.prompt_reused(0)) {
        return false;
    }

    outcome.prompt_ms = 4.0;
    outcome.decode_ms = 6.0;
    outcome.finish = "length";
    outcome.drafts_accepted = 0;
    outcome.drafts_offered = 0;
    outcome.checkpoints = 1;
    outcome.hits = 3;
    outcome.lookups = 4;
    outcome.ram_blobs = 1;
    outcome.file_blobs = 2;
    outcome.file_mb = 0.5;
    outcome.report_resident_ram = true;
    outcome.resident_ram_gib = 2.5;
    outcome.exchanges = 7;
    outcome.cumulative_file_reads = 8;
    outcome.report_expert_tiers = true;
    outcome.cumulative_ram_blobs = 9;
    outcome.cumulative_expert_file_blobs = 10;
    outcome.cumulative_expert_file_mb = 1.5;

    const bool eos_prompt = !request.token_ids.empty() && request.token_ids.front() == 248044;
    for (std::int64_t position = 0; position < request.max_new; ++position) {
        if (callbacks.should_stop()) {
            if (!report_length_after_stop_) outcome.finish = "cancel";
            break;
        }
        if (delay_.count() > 0) std::this_thread::sleep_for(delay_);
        if (callbacks.should_stop()) {
            if (!report_length_after_stop_) outcome.finish = "cancel";
            break;
        }
        const std::int32_t token =
            eos_prompt && position == 0
                ? 248044
                : static_cast<std::int32_t>(1'000 +
                                            (request.sampling.seed +
                                             static_cast<std::uint64_t>(request.token_ids[
                                                 static_cast<std::size_t>(position) % request.token_ids.size()]) +
                                             static_cast<std::uint64_t>(position)) %
                                                200'000);
        if (!callbacks.token(token)) return false;
        if (token == 248044 || token == 248046) {
            outcome.finish = "stop";
            break;
        }
    }
    return true;
}

}  // namespace strata::metal::test
