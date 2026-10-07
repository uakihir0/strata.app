// src/metal/program/stats.cpp - documented Metal one-shot telemetry formatting.
#include "strata/metal/oneshot.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace strata::metal {

std::string format_one_shot_stats(const OneShotMetalStats& stats, std::int64_t rounds,
                                  double decode_ms_per_token) {
    std::string output;

#define STRATA_APPEND_FORMAT(output, ...)                                                                    \
    do {                                                                                                     \
        const int strata_required = std::snprintf(nullptr, 0, __VA_ARGS__);                                  \
        if (strata_required < 0) throw std::runtime_error("cannot format one-shot stats");                  \
        std::vector<char> strata_buffer(static_cast<std::size_t>(strata_required) + 1);                       \
        const int strata_written = std::snprintf(strata_buffer.data(), strata_buffer.size(), __VA_ARGS__);   \
        if (strata_written != strata_required) throw std::runtime_error("cannot format one-shot stats");    \
        (output).append(strata_buffer.data(), static_cast<std::size_t>(strata_written));                     \
    } while (false)

    STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token",
                         decode_ms_per_token);
    if (rounds > 0 && stats.gpu_window_available) {
        const double divisor = static_cast<double>(rounds);
        STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/round (encode %.3f, gpu %.3f, host wait %.3f)\n",
                             "gpu window",
                             // GPU time is contained in the commit-to-wake wait; do not count it twice.
                             (stats.encode_total_ms + stats.host_wait_total_ms) / divisor,
                             stats.encode_total_ms / divisor, stats.gpu_total_ms / divisor,
                             stats.host_wait_total_ms / divisor);
        STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/round\n", "commit+draft",
                             stats.commit_draft_total_ms / divisor);
        STRATA_APPEND_FORMAT(output, "%-24s %d per window at T=1, %d at T=4\n", "dispatches",
                             stats.dispatches_at_t1, stats.dispatches_at_t4);
        if (stats.gpu_total_ms > 0.0) {
            const double bandwidth = static_cast<double>(stats.gpu_bytes_read) /
                                     (stats.gpu_total_ms * 1'000'000.0);
            STRATA_APPEND_FORMAT(output, "%-24s %.1f GB/s over the window\n", "gpu bandwidth", bandwidth);
        }
    }
    if (stats.expert_tiers_available) {
        STRATA_APPEND_FORMAT(output, "%-24s resident %lld, slot hits %lld, page cache %lld, ssd %lld blobs, "
                                    "%.1f MB read\n",
                             "expert tiers", static_cast<long long>(stats.resident_experts),
                             static_cast<long long>(stats.slot_hits),
                             static_cast<long long>(stats.page_cache_hits),
                             static_cast<long long>(stats.ssd_blobs), stats.ssd_megabytes_read);
    }
    if (rounds > 0 && stats.miss_handshake_available) {
        STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/round over %lld layers (max hop %.3f ms)\n",
                             "miss handshake",
                             stats.miss_handshake_total_ms / static_cast<double>(rounds),
                             static_cast<long long>(stats.miss_handshake_layers),
                             stats.miss_handshake_max_hop_ms);
    }
    if (rounds > 0 && stats.adaptive_tier_available) {
        STRATA_APPEND_FORMAT(output, "%-24s %lld experts swapped into the resident set (every %d rounds, "
                                    "%.3f ms/round)\n",
                             "adaptive tier", static_cast<long long>(stats.adaptive_experts_swapped),
                             stats.adaptive_every_rounds, stats.adaptive_total_ms / static_cast<double>(rounds));
    }
    if (stats.routing_prefetch_available) {
        const double warmed_percent =
            stats.file_reads > 0
                ? 100.0 * static_cast<double>(stats.warmed_file_reads) / static_cast<double>(stats.file_reads)
                : 0.0;
        const double predictor_ms_per_round =
            rounds > 0 ? stats.routing_predictor_total_ms / static_cast<double>(rounds) : 0.0;
        STRATA_APPEND_FORMAT(output, "%-24s %lld experts warmed, %lld of the file tier's %lld blob reads had "
                                    "been warmed (%.1f%%); predictor %.1f ms/round on its thread, %lld layers "
                                    "skipped (still busy)\n",
                             "routing prefetch", static_cast<long long>(stats.experts_warmed),
                             static_cast<long long>(stats.warmed_file_reads), static_cast<long long>(stats.file_reads),
                             warmed_percent, predictor_ms_per_round,
                             static_cast<long long>(stats.routing_layers_skipped));
    }
    if (stats.stage_timing_available) {
        STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/layer (QSA %.3f, GDN %.3f, experts %.3f)\n", "layers",
                             stats.milliseconds_per_layer, stats.qsa_milliseconds_per_layer,
                             stats.gdn_milliseconds_per_layer, stats.experts_milliseconds_per_layer);
    }

#undef STRATA_APPEND_FORMAT
    return output;
}

}  // namespace strata::metal
