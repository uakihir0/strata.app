// tests/metal/oneshot_loop_test.cpp - one-shot runner over a synthetic backend (docs/mac/06, H7).
//
// The backend here is not a model. It returns fixed reports so that the runner's own rules are checked: request
// validation, the report invariants it enforces before printing, `--stats` gating, and the exit status. Nothing
// printed here is evidence of inference.
#include "strata/metal/oneshot.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace strata::metal;

namespace {

int failures = 0;

// The detail (usually the printed output) is shown only when the check fails.
void check(bool condition, std::string_view name, std::string_view detail = {}) {
    std::cout << (condition ? "ok  " : "FAIL ") << name;
    if (!condition && !detail.empty()) std::cout << ":\n" << detail;
    std::cout << '\n';
    failures += !condition;
}

class SyntheticBackend final : public OneShotBackend {
public:
    enum class Mode { Report, Fail, Throw };

    explicit SyntheticBackend(OneShotReport report, Mode mode = Mode::Report)
        : report_(std::move(report)), mode_(mode) {}

    bool run(const OneShotRequest& request, OneShotReport& report, std::string& error) override {
        ++calls_;
        last_include_stats_ = request.include_stats;
        if (mode_ == Mode::Fail) {
            error = "synthetic window failure";
            return false;
        }
        if (mode_ == Mode::Throw) throw std::runtime_error("boom");
        report = report_;
        return true;
    }

    int calls() const noexcept { return calls_; }
    bool last_include_stats() const noexcept { return last_include_stats_; }

private:
    OneShotReport report_;
    Mode mode_;
    int calls_ = 0;
    bool last_include_stats_ = false;
};

struct Outcome {
    int status = -1;
    std::string out;
    std::string diagnostics;
};

std::string read_stream(std::FILE* file) {
    std::fflush(file);
    if (std::fseek(file, 0, SEEK_END) != 0) return {};
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) return {};
    std::string result(static_cast<std::size_t>(size), '\0');
    if (!result.empty() && std::fread(result.data(), 1, result.size(), file) != result.size()) return {};
    return result;
}

Outcome run_case(OneShotBackend& backend, const OneShotRequest& request) {
    Outcome outcome;
    std::FILE* output = std::tmpfile();
    std::FILE* diagnostics = std::tmpfile();
    if (output == nullptr || diagnostics == nullptr) {
        if (output != nullptr) std::fclose(output);
        if (diagnostics != nullptr) std::fclose(diagnostics);
        return outcome;
    }
    outcome.status = run_one_shot(backend, request, output, diagnostics);
    outcome.out = read_stream(output);
    outcome.diagnostics = read_stream(diagnostics);
    std::fclose(output);
    std::fclose(diagnostics);
    return outcome;
}

OneShotRequest three_token_request(bool include_stats) {
    OneShotRequest request;
    request.prompt_ids = {11, 12, 13};
    request.max_new = 3;
    request.include_stats = include_stats;
    return request;
}

OneShotReport three_token_report() {
    OneShotReport report;
    report.prompt_ids = {11, 12, 13};
    report.output_ids = {21, 22, 23};
    report.decode_total_ms = 6.0;
    report.prefill_ms = 4.0;
    report.time_to_first_token_ms = 5.5;
    report.rounds = 3;
    return report;
}

const std::string kPlainOutput =
    "prompt  : 11 12 13\n"
    "output  : 21 22 23\n"
    "decode                   3 tokens in 6.0 ms  ->  500.00 tok/s\n"
    "prefill                  2 tokens in 4.0 ms  ->  500.00 tok/s  (time to first token 5.5 ms)\n";

void test_plain_run_prints_only_the_summary() {
    SyntheticBackend backend(three_token_report());
    const Outcome outcome = run_case(backend, three_token_request(false));
    check(outcome.status == 0 && outcome.out == kPlainOutput && outcome.diagnostics.empty(),
          "a run without --stats prints the prompt, output, decode and prefill lines", outcome.out);
    check(backend.calls() == 1 && !backend.last_include_stats(),
          "the request reaches the backend once, with --stats off");
}

void test_stats_block_is_computed_from_the_run() {
    OneShotReport report = three_token_report();
    report.stats.gpu_window_available = true;
    report.stats.encode_total_ms = 3.0;
    report.stats.host_wait_total_ms = 6.0;
    report.stats.gpu_total_ms = 3.0;
    report.stats.commit_draft_total_ms = 3.0;
    report.stats.dispatches_at_t1 = 144;
    report.stats.dispatches_at_t4 = 160;
    report.stats.gpu_bytes_read = 30'000'000;
    SyntheticBackend backend(report);
    const Outcome outcome = run_case(backend, three_token_request(true));
    const std::string expected =
        kPlainOutput +
        "per token                2.000 ms/token (WALL CLOCK: embed, layers, head, sample)\n"
        "gpu window               3.000 ms/round (encode 1.000, gpu 1.000, host wait 2.000)\n"
        "commit+draft             1.000 ms/round\n"
        "dispatches               144 per window at T=1, 160 at T=4\n"
        "gpu bandwidth            10.0 GB/s over the window\n";
    check(outcome.status == 0 && outcome.out == expected && backend.last_include_stats(),
          "--stats prints per-token, window, commit, dispatch and bandwidth lines from the run totals", outcome.out);
}

void test_stats_are_suppressed_without_the_flag() {
    OneShotReport report = three_token_report();
    report.include_stats = true;  // The backend asks for the block; the request's missing --stats must still win.
    report.stats.gpu_window_available = true;
    report.stats.gpu_total_ms = 3.0;
    report.stats.encode_total_ms = 3.0;
    SyntheticBackend backend(report);
    const Outcome outcome = run_case(backend, three_token_request(false));
    check(outcome.status == 0 && outcome.out == kPlainOutput,
          "stats the backend fills in are not printed unless --stats was given", outcome.out);
}

void test_speculation_block_and_zero_token_output() {
    OneShotReport report;
    report.prompt_ids = {5};
    report.output_ids = {7, 8, 9, 10};
    report.decode_total_ms = 8.0;
    report.rounds = 2;
    report.speculation.enabled = true;
    report.speculation.window = 2;
    report.speculation.rounds = 2;
    report.speculation.drafts_offered = 4;
    report.speculation.drafts_accepted = 3;
    report.speculation.accepted_per_round = {0, 1, 1};
    OneShotRequest request;
    request.prompt_ids = {5};
    request.max_new = 4;
    SyntheticBackend backend(report);
    Outcome outcome = run_case(backend, request);
    const std::string expected_speculative =
        "speculation              2 rounds of 2, drafts accepted 3 of 4 (0.750), 2.50 tokens per round\n"
        "accepted per round       0:0 1:1 2:1\n"
        "prompt  : 5\n"
        "output  : 7 8 9 10\n"
        "decode                   4 tokens in 8.0 ms  ->  500.00 tok/s\n";
    check(outcome.status == 0 && outcome.out == expected_speculative,
          "speculation totals are printed, and a one-token prompt has no prefill line", outcome.out);

    OneShotReport empty;
    empty.prompt_ids = {7};
    OneShotRequest one_token;
    one_token.prompt_ids = {7};
    one_token.max_new = 1;
    SyntheticBackend zero_backend(empty);
    outcome = run_case(zero_backend, one_token);
    check(outcome.status == 0 &&
              outcome.out == "prompt  : 7\noutput  :\ndecode                   0 tokens in 0.0 ms  ->  0.00 tok/s\n",
          "a run with no output tokens prints a zero decode line without dividing by zero", outcome.out);
}

void expect_rejected(const std::string& name, OneShotReport report, const std::string& reason) {
    SyntheticBackend backend(std::move(report));
    const Outcome outcome = run_case(backend, three_token_request(true));
    check(outcome.status == 1 && outcome.out.empty() && outcome.diagnostics.find(reason) != std::string::npos,
          name, outcome.diagnostics);
}

void test_inconsistent_reports_print_nothing() {
    OneShotReport prompt_mismatch = three_token_report();
    prompt_mismatch.prompt_ids = {11, 99, 13};
    expect_rejected("a report for a different prompt is rejected", prompt_mismatch, "differs from the request");

    OneShotReport too_many = three_token_report();
    too_many.output_ids = {21, 22, 23, 24};
    expect_rejected("more output tokens than max_new are rejected", too_many, "more tokens than max_new");

    OneShotReport out_of_vocabulary = three_token_report();
    out_of_vocabulary.output_ids = {21, 22, 248'320};
    expect_rejected("an output id outside the vocabulary is rejected", out_of_vocabulary,
                    "outside the vocabulary");

    OneShotReport not_a_number = three_token_report();
    not_a_number.decode_total_ms = std::nan("");
    expect_rejected("a NaN decode time is rejected", not_a_number, "non-finite or negative time");

    OneShotReport negative_prefill = three_token_report();
    negative_prefill.prefill_ms = -1.0;
    expect_rejected("a negative prefill time is rejected", negative_prefill, "non-finite or negative time");

    OneShotReport infinite_stat = three_token_report();
    infinite_stat.stats.gpu_total_ms = std::numeric_limits<double>::infinity();
    expect_rejected("an infinite GPU time is rejected before it reaches the stats line", infinite_stat,
                    "non-finite or negative stats value");

    OneShotReport over_accepted = three_token_report();
    over_accepted.speculation.enabled = true;
    over_accepted.speculation.rounds = 2;
    over_accepted.speculation.drafts_offered = 2;
    over_accepted.speculation.drafts_accepted = 3;
    expect_rejected("speculation cannot accept more drafts than it offered", over_accepted,
                    "accepted more drafts than it offered");

    OneShotReport histogram = three_token_report();
    histogram.speculation.enabled = true;
    histogram.speculation.rounds = 2;
    histogram.speculation.drafts_offered = 4;
    histogram.speculation.drafts_accepted = 2;
    histogram.speculation.accepted_per_round = {0, 1, 1};
    expect_rejected("the accepted-per-round histogram must add up to the accepted total", histogram,
                    "histogram disagrees");

    OneShotReport prefetch = three_token_report();
    prefetch.stats.routing_prefetch_available = true;
    prefetch.stats.file_reads = 4;
    prefetch.stats.warmed_file_reads = 5;
    expect_rejected("warmed blob reads cannot exceed the reads", prefetch, "warmed more blob reads");

    OneShotReport negative_output = three_token_report();
    negative_output.output_ids = {21, -1, 23};
    expect_rejected("a negative output id is rejected", negative_output, "outside the vocabulary");

    OneShotReport negative_first_token = three_token_report();
    negative_first_token.time_to_first_token_ms = -1.0;
    expect_rejected("a negative time to first token is rejected", negative_first_token,
                    "non-finite or negative time");

    OneShotReport negative_rounds = three_token_report();
    negative_rounds.rounds = -1;
    expect_rejected("a negative round count is rejected", negative_rounds, "negative round count");

    OneShotReport too_many_rounds = three_token_report();
    too_many_rounds.speculation.enabled = true;
    too_many_rounds.speculation.rounds = 4;
    expect_rejected("speculation rounds cannot exceed the decode windows", too_many_rounds,
                    "exceed the decode windows");

    OneShotReport histogram_rounds = three_token_report();
    histogram_rounds.speculation.enabled = true;
    histogram_rounds.speculation.rounds = 1;
    histogram_rounds.speculation.drafts_offered = 4;
    histogram_rounds.speculation.drafts_accepted = 3;
    histogram_rounds.speculation.accepted_per_round = {0, 1, 1};
    expect_rejected("the accepted-per-round histogram cannot count more rounds than speculation ran",
                    histogram_rounds, "histogram disagrees");

    OneShotReport suffix = three_token_report();
    suffix.speculation.enabled = true;
    suffix.speculation.suffix_draft = 2;
    suffix.speculation.suffix_windows = 1;
    suffix.speculation.suffix_drafts_offered = 1;
    suffix.speculation.suffix_drafts_accepted = 2;
    expect_rejected("suffix drafts cannot accept more drafts than they offered", suffix, "suffix draft counts");

    OneShotReport negative_window = three_token_report();
    negative_window.speculation.enabled = true;
    negative_window.speculation.min_draft_probability = 0.25;
    negative_window.speculation.window_counts = {0, -1, 2};
    expect_rejected("a window size count cannot be negative", negative_window,
                    "window size counts must not be negative");

    OneShotReport negative_slots = three_token_report();
    negative_slots.stats.slot_hits = -1;
    expect_rejected("a negative slot hit count is rejected before it reaches the stats line", negative_slots,
                    "non-finite or negative stats value");

    OneShotReport negative_dispatches = three_token_report();
    negative_dispatches.stats.dispatches_at_t1 = -1;
    expect_rejected("a negative dispatch count is rejected before it reaches the stats line", negative_dispatches,
                    "non-finite or negative stats value");
}

void test_backend_failures_and_invalid_requests() {
    SyntheticBackend failing(three_token_report(), SyntheticBackend::Mode::Fail);
    const Outcome failed = run_case(failing, three_token_request(true));
    check(failed.status == 1 && failed.out.empty() && failed.diagnostics.find("synthetic window failure") !=
                                                            std::string::npos,
          "a backend failure exits 1 with its message and no report", failed.diagnostics);

    SyntheticBackend throwing(three_token_report(), SyntheticBackend::Mode::Throw);
    const Outcome threw = run_case(throwing, three_token_request(true));
    check(threw.status == 1 && threw.out.empty() &&
              threw.diagnostics.find("one-shot backend threw: boom") != std::string::npos,
          "an exception from the backend exits 1 with no report", threw.diagnostics);

    OneShotRequest empty_prompt;
    empty_prompt.max_new = 3;
    SyntheticBackend unused_empty(three_token_report());
    const Outcome no_prompt = run_case(unused_empty, empty_prompt);
    check(no_prompt.status == 2 && unused_empty.calls() == 0 && no_prompt.out.empty(),
          "an empty prompt exits 2 before the backend runs", no_prompt.diagnostics);

    OneShotRequest zero_new = three_token_request(false);
    zero_new.max_new = 0;
    SyntheticBackend unused_zero(three_token_report());
    const Outcome no_tokens = run_case(unused_zero, zero_new);
    check(no_tokens.status == 2 && unused_zero.calls() == 0 && no_tokens.out.empty(),
          "max_new below one exits 2 before the backend runs", no_tokens.diagnostics);

    OneShotRequest bad_prompt = three_token_request(false);
    bad_prompt.prompt_ids = {11, 248'320};
    SyntheticBackend unused_vocab(three_token_report());
    const Outcome out_of_range = run_case(unused_vocab, bad_prompt);
    check(out_of_range.status == 2 && unused_vocab.calls() == 0 && out_of_range.out.empty(),
          "a prompt id outside the vocabulary exits 2 before the backend runs", out_of_range.diagnostics);

    OneShotRequest negative_prompt = three_token_request(false);
    negative_prompt.prompt_ids = {11, -1, 13};
    SyntheticBackend unused_negative(three_token_report());
    const Outcome negative_id = run_case(unused_negative, negative_prompt);
    check(negative_id.status == 2 && unused_negative.calls() == 0 && negative_id.out.empty(),
          "a negative prompt id exits 2 before the backend runs", negative_id.diagnostics);

    // Validation runs before the --stats gate, so a non-finite stat still rejects a run without --stats (D-IMPL-814).
    OneShotReport nan_stat = three_token_report();
    nan_stat.stats.gpu_total_ms = std::nan("");
    SyntheticBackend nan_backend(nan_stat);
    const Outcome nan_plain = run_case(nan_backend, three_token_request(false));
    check(nan_plain.status == 1 && nan_plain.out.empty() &&
              nan_plain.diagnostics.find("non-finite or negative stats value") != std::string::npos,
          "a NaN stat rejects the run even without --stats", nan_plain.diagnostics);
}

}  // namespace

int main() {
    test_plain_run_prints_only_the_summary();
    test_stats_block_is_computed_from_the_run();
    test_stats_are_suppressed_without_the_flag();
    test_speculation_block_and_zero_token_output();
    test_inconsistent_reports_print_nothing();
    test_backend_failures_and_invalid_requests();
    std::cout << "oneshot_loop_test: " << (failures == 0 ? "OK" : "FAILED") << '\n';
    return failures != 0;
}
