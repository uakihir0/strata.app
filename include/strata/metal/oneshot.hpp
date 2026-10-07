// include/strata/metal/oneshot.hpp - one-shot output shared by the Metal driver and host tests.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::metal {

struct OneShotSpeculationStats {
    bool enabled = false;
    std::int32_t window = 0;
    double min_draft_probability = 0.0;
    std::int64_t rounds = 0;
    std::int64_t drafts_accepted = 0;
    std::int64_t drafts_offered = 0;
    std::vector<std::int64_t> window_counts;  // Index zero is unused; index T counts windows of size T.
    std::int32_t suffix_draft = 0;
    std::int64_t suffix_windows = 0;
    std::int64_t suffix_drafts_accepted = 0;
    std::int64_t suffix_drafts_offered = 0;
    std::vector<std::int64_t> accepted_per_round;
    bool mtp_used = false;
    std::int64_t mtp_rounds = 0;
    double mtp_draft_total_ms = 0.0;
    double mtp_prompt_ms = 0.0;
    std::uint64_t mtp_vram_bytes = 0;
};

struct OneShotMetalStats {
    bool gpu_window_available = false;
    double encode_total_ms = 0.0;
    double gpu_total_ms = 0.0;
    double host_wait_total_ms = 0.0;
    double commit_draft_total_ms = 0.0;
    std::int32_t dispatches_at_t1 = 0;
    std::int32_t dispatches_at_t4 = 0;
    std::uint64_t gpu_bytes_read = 0;

    bool expert_tiers_available = false;
    std::int64_t resident_experts = 0;
    std::int64_t slot_hits = 0;
    std::int64_t page_cache_hits = 0;
    std::int64_t ssd_blobs = 0;
    double ssd_megabytes_read = 0.0;

    bool miss_handshake_available = false;
    double miss_handshake_total_ms = 0.0;
    double miss_handshake_max_hop_ms = 0.0;
    std::int64_t miss_handshake_layers = 0;

    bool adaptive_tier_available = false;
    std::int64_t adaptive_experts_swapped = 0;
    std::int32_t adaptive_every_rounds = 0;
    double adaptive_total_ms = 0.0;

    bool routing_prefetch_available = false;
    std::int64_t experts_warmed = 0;
    std::int64_t warmed_file_reads = 0;
    std::int64_t file_reads = 0;
    double routing_predictor_total_ms = 0.0;
    std::int64_t routing_layers_skipped = 0;

    bool stage_timing_available = false;
    double milliseconds_per_layer = 0.0;
    double qsa_milliseconds_per_layer = 0.0;
    double gdn_milliseconds_per_layer = 0.0;
    double experts_milliseconds_per_layer = 0.0;
};

struct OneShotDumpSummary {
    std::string label;
    std::string path;
    std::string details;  // Optional suffix, such as "(n_embd 2560, rms 0.1, mean|.| 0.2)".
};

struct OneShotReport {
    std::vector<std::int64_t> prompt_ids;
    std::vector<std::int64_t> output_ids;
    double decode_total_ms = 0.0;
    double prefill_ms = 0.0;
    double time_to_first_token_ms = 0.0;
    std::int64_t rounds = 0;  // Decode windows, including ordinary T=1 windows without speculation.

    OneShotSpeculationStats speculation;
    std::vector<OneShotDumpSummary> early_dumps;
    std::vector<OneShotDumpSummary> post_summary_dumps;
    std::vector<OneShotDumpSummary> post_stats_dumps;

    bool include_stats = false;
    OneShotMetalStats stats;
};

/// Format the upstream-compatible one-shot summary and the documented Metal stats lines.
std::string format_one_shot_report(const OneShotReport& report);
/// Format the `--stats` lines. `decode_ms_per_token` is the one-shot wall-clock average.
std::string format_one_shot_stats(const OneShotMetalStats& stats, std::int64_t rounds,
                                  double decode_ms_per_token);

}  // namespace strata::metal
