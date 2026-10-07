#include "strata/metal/oneshot.hpp"

#include <iostream>
#include <string>

namespace {

int fail(const std::string& message) {
    std::cerr << "oneshot_output_test: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    using namespace strata::metal;

    OneShotReport report;
    report.prompt_ids = {11, 12};
    report.output_ids = {21, 22};
    report.decode_total_ms = 5.0;
    report.prefill_ms = 4.0;
    report.time_to_first_token_ms = 6.5;
    report.rounds = 2;
    report.speculation.enabled = true;
    report.speculation.window = 4;
    report.speculation.min_draft_probability = 0.25;
    report.speculation.rounds = 2;
    report.speculation.drafts_accepted = 3;
    report.speculation.drafts_offered = 6;
    report.speculation.window_counts = {0, 1, 1, 0, 0};
    report.speculation.suffix_draft = 3;
    report.speculation.suffix_windows = 1;
    report.speculation.suffix_drafts_accepted = 1;
    report.speculation.suffix_drafts_offered = 3;
    report.speculation.accepted_per_round = {0, 1, 1};
    report.speculation.mtp_used = true;
    report.speculation.mtp_rounds = 2;
    report.speculation.mtp_draft_total_ms = 3.0;
    report.speculation.mtp_prompt_ms = 4.5;
    report.speculation.mtp_vram_bytes = 2'097'152;
    report.early_dumps.push_back(
        {"layers dumped", "layers.bin", "(48 layers + the input x 1 streams x 2560 per position)"});
    report.post_summary_dumps.push_back({"mixed dumped", "mixed.bin", "(n_embd 2560, rms 0.1, mean|.| 0.2)"});
    report.post_stats_dumps.push_back({"logits dumped", "logits.bin", ""});
    report.include_stats = true;
    report.stats.gpu_window_available = true;
    report.stats.encode_total_ms = 1.0;
    report.stats.gpu_total_ms = 0.8;
    report.stats.host_wait_total_ms = 3.0;
    report.stats.commit_draft_total_ms = 2.0;
    report.stats.dispatches_at_t1 = 144;
    report.stats.dispatches_at_t4 = 160;
    report.stats.gpu_bytes_read = 80'000'000;
    report.stats.expert_tiers_available = true;
    report.stats.resident_experts = 300;
    report.stats.slot_hits = 12;
    report.stats.page_cache_hits = 20;
    report.stats.ssd_blobs = 4;
    report.stats.ssd_megabytes_read = 5.5;
    report.stats.miss_handshake_available = true;
    report.stats.miss_handshake_total_ms = 2.0;
    report.stats.miss_handshake_max_hop_ms = 0.25;
    report.stats.miss_handshake_layers = 48;
    report.stats.adaptive_tier_available = true;
    report.stats.adaptive_experts_swapped = 7;
    report.stats.adaptive_every_rounds = 10;
    report.stats.adaptive_total_ms = 4.0;
    report.stats.routing_prefetch_available = true;
    report.stats.experts_warmed = 10;
    report.stats.warmed_file_reads = 2;
    report.stats.file_reads = 4;
    report.stats.routing_predictor_total_ms = 1.0;
    report.stats.routing_layers_skipped = 3;
    report.stats.stage_timing_available = true;
    report.stats.milliseconds_per_layer = 3.5;
    report.stats.qsa_milliseconds_per_layer = 1.0;
    report.stats.gdn_milliseconds_per_layer = 2.0;
    report.stats.experts_milliseconds_per_layer = 0.5;

    const std::string expected =
        "speculation              2 rounds of 4, drafts accepted 3 of 6 (0.500), 2.50 tokens per round\n"
        "window sizes             T1:1 T2:1 T3:0 T4:0  (min draft probability 0.25)\n"
        "suffix drafts            1 windows, drafts accepted 1 of 3\n"
        "accepted per round       0:0 1:1 2:1\n"
        "mtp                      1.500 ms/round drafting (2 rounds), MTP prompt 4.5 ms, 2 MiB of VRAM\n"
        "layers dumped            layers.bin (48 layers + the input x 1 streams x 2560 per position)\n"
        "prompt  : 11 12\n"
        "output  : 21 22\n"
        "decode                   2 tokens in 5.0 ms  ->  400.00 tok/s\n"
        "prefill                  1 tokens in 4.0 ms  ->  250.00 tok/s  (time to first token 6.5 ms)\n"
        "mixed dumped             mixed.bin (n_embd 2560, rms 0.1, mean|.| 0.2)\n"
        "  per token              2.500 ms/token (WALL CLOCK: embed, layers, head, sample)\n"
        "gpu window               2.000 ms/round (encode 0.500, gpu 0.400, host wait 1.500)\n"
        "commit+draft             1.000 ms/round\n"
        "dispatches               144 per window at T=1, 160 at T=4\n"
        "gpu bandwidth            100.0 GB/s over the window\n"
        "expert tiers             resident 300, slot hits 12, page cache 20, ssd 4 blobs, 5.5 MB read\n"
        "miss handshake           1.000 ms/round over 48 layers (max hop 0.250 ms)\n"
        "adaptive tier            7 experts swapped into the resident set (every 10 rounds, 2.000 ms/round)\n"
        "routing prefetch         10 experts warmed, 2 of the file tier's 4 blob reads had been warmed (50.0%); "
        "predictor 0.5 ms/round on its thread, 3 layers skipped (still busy)\n"
        "layers                   3.500 ms/layer (QSA 1.000, GDN 2.000, experts 0.500)\n"
        "logits dumped            logits.bin\n";
    const std::string actual = format_one_shot_report(report);
    if (actual != expected) {
        std::cerr << "expected:\n" << expected << "actual:\n" << actual;
        return fail("one-shot output differs from the upstream-compatible golden lines");
    }

    OneShotReport degenerate;
    degenerate.prompt_ids = {7};
    degenerate.decode_total_ms = 0.0;
    degenerate.include_stats = true;
    const std::string no_decode = format_one_shot_report(degenerate);
    if (no_decode !=
        "prompt  : 7\n"
        "output  :\n"
        "decode                   0 tokens in 0.0 ms  ->  0.00 tok/s\n"
        "  per token              0.000 ms/token (WALL CLOCK: embed, layers, head, sample)\n") {
        return fail("empty output or one-token prompt formatting is incorrect");
    }

    OneShotReport no_mtp_draft;
    no_mtp_draft.prompt_ids = {7};
    no_mtp_draft.output_ids = {248044};
    no_mtp_draft.rounds = 1;
    no_mtp_draft.speculation.enabled = true;
    no_mtp_draft.speculation.window = 1;
    no_mtp_draft.speculation.mtp_used = true;
    const std::string no_draft_output = format_one_shot_report(no_mtp_draft);
    if (no_draft_output.find("mtp                      0.000 ms/round drafting (0 rounds)") ==
        std::string::npos) {
        return fail("MTP summary was omitted when generation ended before its first draft");
    }

    OneShotReport plain;
    plain.prompt_ids = {1};
    plain.output_ids = {2};
    plain.decode_total_ms = 2.0;
    plain.rounds = 1;
    plain.include_stats = true;
    plain.stats.gpu_window_available = true;
    const std::string plain_output = format_one_shot_report(plain);
    if (plain_output.find("speculation") != std::string::npos ||
        plain_output.find("gpu window") == std::string::npos) {
        return fail("plain T=1 stats or disabled speculation formatting is incorrect");
    }
    return 0;
}
