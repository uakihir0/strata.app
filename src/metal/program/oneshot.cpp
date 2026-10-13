// src/metal/program/oneshot.cpp - upstream-compatible one-shot output formatting.
#include "strata/metal/oneshot.hpp"
#include "strata/metal/serve.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::metal {
namespace {

#define STRATA_APPEND_FORMAT(output, ...)                                                                    \
    do {                                                                                                     \
        const int strata_required = std::snprintf(nullptr, 0, __VA_ARGS__);                                  \
        if (strata_required < 0) throw std::runtime_error("cannot format one-shot output");                 \
        std::vector<char> strata_buffer(static_cast<std::size_t>(strata_required) + 1);                       \
        const int strata_written = std::snprintf(strata_buffer.data(), strata_buffer.size(), __VA_ARGS__);   \
        if (strata_written != strata_required) throw std::runtime_error("cannot format one-shot output");   \
        (output).append(strata_buffer.data(), static_cast<std::size_t>(strata_written));                     \
    } while (false)

void append_dump_summaries(std::string& output, const std::vector<OneShotDumpSummary>& dumps) {
    for (const OneShotDumpSummary& dump : dumps) {
        STRATA_APPEND_FORMAT(output, "%-24s %s%s%s\n", dump.label.c_str(), dump.path.c_str(),
                             dump.details.empty() ? "" : " ", dump.details.c_str());
    }
}

}  // namespace

std::string format_one_shot_report(const OneShotReport& report) {
    std::string output;
    const OneShotSpeculationStats& speculation = report.speculation;
    if (speculation.enabled) {
        STRATA_APPEND_FORMAT(output,
                             "%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                             "speculation", static_cast<long long>(speculation.rounds), speculation.window,
                             static_cast<long long>(speculation.drafts_accepted),
                             static_cast<long long>(speculation.drafts_offered),
                             speculation.drafts_offered > 0
                                 ? static_cast<double>(speculation.drafts_accepted) /
                                       static_cast<double>(speculation.drafts_offered)
                                 : 0.0,
                             speculation.rounds > 0
                                 ? static_cast<double>(speculation.drafts_accepted + speculation.rounds) /
                                       static_cast<double>(speculation.rounds)
                                 : 0.0);
        if (speculation.min_draft_probability > 0.0) {
            STRATA_APPEND_FORMAT(output, "%-24s", "window sizes");
            for (std::size_t i = 1; i < speculation.window_counts.size(); ++i) {
                STRATA_APPEND_FORMAT(output, " T%zu:%lld", i,
                                     static_cast<long long>(speculation.window_counts[i]));
            }
            STRATA_APPEND_FORMAT(output, "  (min draft probability %.2f)\n",
                                 speculation.min_draft_probability);
        }
        if (speculation.suffix_draft > 0) {
            STRATA_APPEND_FORMAT(output, "%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts",
                                 static_cast<long long>(speculation.suffix_windows),
                                 static_cast<long long>(speculation.suffix_drafts_accepted),
                                 static_cast<long long>(speculation.suffix_drafts_offered));
        }
        STRATA_APPEND_FORMAT(output, "%-24s", "accepted per round");
        for (std::size_t i = 0; i < speculation.accepted_per_round.size(); ++i) {
            STRATA_APPEND_FORMAT(output, " %zu:%lld", i,
                                 static_cast<long long>(speculation.accepted_per_round[i]));
        }
        output.push_back('\n');
        if (speculation.mtp_used && report.rounds > 0) {
            STRATA_APPEND_FORMAT(output, "%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, "
                                         "%.0f MiB of VRAM\n",
                                 "mtp", speculation.mtp_rounds > 0
                                            ? speculation.mtp_draft_total_ms /
                                                  static_cast<double>(speculation.mtp_rounds)
                                            : 0.0,
                                 static_cast<long long>(speculation.mtp_rounds), speculation.mtp_prompt_ms,
                                 static_cast<double>(speculation.mtp_vram_bytes) / 1'048'576.0);
        }
    }

    append_dump_summaries(output, report.early_dumps);

    output.append("prompt  :");
    for (const std::int64_t token : report.prompt_ids) {
        STRATA_APPEND_FORMAT(output, " %lld", static_cast<long long>(token));
    }
    output.append("\noutput  :");
    for (const std::int64_t token : report.output_ids) {
        STRATA_APPEND_FORMAT(output, " %lld", static_cast<long long>(token));
    }
    output.push_back('\n');

    const std::int64_t decoded = static_cast<std::int64_t>(report.output_ids.size());
    const double decode_ms = decoded > 0 ? report.decode_total_ms / static_cast<double>(decoded) : 0.0;
    STRATA_APPEND_FORMAT(output, "%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode",
                         static_cast<long long>(decoded), report.decode_total_ms,
                         decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (report.prompt_ids.size() > 1) {
        const std::int64_t prefill_tokens = static_cast<std::int64_t>(report.prompt_ids.size() - 1);
        STRATA_APPEND_FORMAT(output,
                             "%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n",
                             "prefill", static_cast<long long>(prefill_tokens), report.prefill_ms,
                             report.prefill_ms > 0.0
                                 ? 1000.0 * static_cast<double>(prefill_tokens) / report.prefill_ms
                                 : 0.0,
                             report.time_to_first_token_ms);
    }

    append_dump_summaries(output, report.post_summary_dumps);
    if (report.include_stats) {
        output.append(format_one_shot_stats(report.stats, report.rounds, decode_ms));
    }
    append_dump_summaries(output, report.post_stats_dumps);
    return output;
}

namespace {

bool finite_nonnegative(double value) { return std::isfinite(value) && value >= 0.0; }

bool valid_times(const OneShotMetalStats& stats) {
    return finite_nonnegative(stats.encode_total_ms) && finite_nonnegative(stats.gpu_total_ms) &&
           finite_nonnegative(stats.host_wait_total_ms) && finite_nonnegative(stats.commit_draft_total_ms) &&
           finite_nonnegative(stats.miss_handshake_total_ms) && finite_nonnegative(stats.miss_handshake_max_hop_ms) &&
           finite_nonnegative(stats.adaptive_total_ms) && finite_nonnegative(stats.routing_predictor_total_ms) &&
           finite_nonnegative(stats.ssd_megabytes_read) && finite_nonnegative(stats.milliseconds_per_layer) &&
           finite_nonnegative(stats.qsa_milliseconds_per_layer) &&
           finite_nonnegative(stats.gdn_milliseconds_per_layer) &&
           finite_nonnegative(stats.experts_milliseconds_per_layer) &&
           // Counts printed by the stats lines: a negative value would print a figure with no meaning.
           stats.slot_hits >= 0 && stats.dispatches_at_t1 >= 0 && stats.dispatches_at_t4 >= 0;
}

}  // namespace

std::string validate_one_shot_report(const OneShotRequest& request, const OneShotReport& report) {
    if (report.prompt_ids != request.prompt_ids) return "backend report prompt differs from the request";
    if (static_cast<std::int64_t>(report.output_ids.size()) > request.max_new) {
        return "backend generated more tokens than max_new";
    }
    for (const std::int64_t id : report.output_ids) {
        if (id < 0 || id >= kServeVocabularySize) return "backend emitted a token outside the vocabulary";
    }
    if (!finite_nonnegative(report.decode_total_ms) || !finite_nonnegative(report.prefill_ms) ||
        !finite_nonnegative(report.time_to_first_token_ms)) {
        return "backend reported a non-finite or negative time";
    }
    if (report.rounds < 0) return "backend reported a negative round count";
    if (!valid_times(report.stats)) return "backend reported a non-finite or negative stats value";
    if (report.stats.routing_prefetch_available && report.stats.warmed_file_reads > report.stats.file_reads) {
        return "backend warmed more blob reads than it read";
    }

    const OneShotSpeculationStats& speculation = report.speculation;
    if (speculation.enabled) {
        if (speculation.rounds < 0 || speculation.rounds > report.rounds) {
            return "speculation rounds exceed the decode windows";
        }
        if (speculation.drafts_accepted < 0 || speculation.drafts_accepted > speculation.drafts_offered) {
            return "speculation accepted more drafts than it offered";
        }
        if (speculation.suffix_windows < 0 || speculation.suffix_drafts_offered < 0 ||
            speculation.suffix_drafts_accepted < 0 ||
            speculation.suffix_drafts_accepted > speculation.suffix_drafts_offered) {
            return "suffix draft counts are negative or accepted more drafts than they offered";
        }
        // Window sizes are counts. Their total is not tied to the rounds here, because the spec does not say whether
        // the counts cover every decode window or only speculative rounds.
        for (const std::int64_t count : speculation.window_counts) {
            if (count < 0) return "window size counts must not be negative";
        }
        if (!finite_nonnegative(speculation.mtp_draft_total_ms) || !finite_nonnegative(speculation.mtp_prompt_ms)) {
            return "backend reported a non-finite or negative MTP time";
        }
        // accepted_per_round[i] counts the rounds that accepted i drafts, so the weighted sum is the accepted total.
        std::int64_t counted_rounds = 0;
        std::int64_t counted_accepted = 0;
        for (std::size_t accepted = 0; accepted < speculation.accepted_per_round.size(); ++accepted) {
            const std::int64_t count = speculation.accepted_per_round[accepted];
            if (count < 0) return "accepted-per-round counts must not be negative";
            counted_rounds += count;
            counted_accepted += static_cast<std::int64_t>(accepted) * count;
        }
        if (counted_rounds > speculation.rounds || counted_accepted != speculation.drafts_accepted) {
            return "accepted-per-round histogram disagrees with the speculation totals";
        }
    }
    return {};
}

int run_one_shot(OneShotBackend& backend, const OneShotRequest& request, std::FILE* output,
                 std::FILE* diagnostics) {
    if (request.prompt_ids.empty()) {
        std::fprintf(diagnostics, "strata: the one-shot prompt has no tokens\n");
        return 2;
    }
    if (request.max_new < 1) {
        std::fprintf(diagnostics, "strata: --max-new must be at least 1\n");
        return 2;
    }
    for (const std::int64_t id : request.prompt_ids) {
        if (id < 0 || id >= kServeVocabularySize) {
            std::fprintf(diagnostics, "strata: a token id is outside the vocabulary\n");
            return 2;
        }
    }

    OneShotReport report;
    std::string error;
    bool succeeded = false;
    try {
        succeeded = backend.run(request, report, error);
    } catch (const std::exception& exception) {
        error = std::string("one-shot backend threw: ") + exception.what();
    } catch (...) {
        error = "one-shot backend threw an unknown exception";
    }
    if (!succeeded) {
        std::fprintf(diagnostics, "strata: %s\n", error.empty() ? "one-shot generation failed" : error.c_str());
        std::fflush(diagnostics);
        return 1;
    }
    // `--stats` decides the block, whatever the backend filled in.
    report.include_stats = request.include_stats;
    const std::string problem = validate_one_shot_report(request, report);
    if (!problem.empty()) {
        std::fprintf(diagnostics, "strata: %s\n", problem.c_str());
        std::fflush(diagnostics);
        return 1;
    }
    const std::string text = format_one_shot_report(report);
    if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fflush(output) != 0) {
        std::fprintf(diagnostics, "strata: could not write the one-shot report\n");
        std::fflush(diagnostics);
        return 1;
    }
    return 0;
}

}  // namespace strata::metal
