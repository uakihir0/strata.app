// src/metal/program/oneshot.cpp - upstream-compatible one-shot output formatting.
#include "strata/metal/oneshot.hpp"

#include <cstdio>
#include <stdexcept>
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

}  // namespace strata::metal
