// tests/metal/metal_never_wrap_host_test.cpp - device-free never-wrap rules (05 section 6).
#include "strata/metal/window_plan.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::metal::FileWindow;
using strata::metal::FileWindowPlan;
using strata::metal::WindowPlan;
using strata::metal::WindowSlice;
using strata::metal::WindowSources;
using strata::metal::WindowTensorExclusion;
using strata::metal::WindowTensorExclusionReason;
using strata::metal::WindowTensorSource;
using strata::metal::kNoExpertIndex;
using strata::metal::kWindowPageBytes;

constexpr std::uint64_t kPage = kWindowPageBytes;
constexpr std::uint64_t kMiB = 1024ull * 1024;
constexpr std::uint64_t kMaxBuffer = 64 * kMiB;
const std::string kPath = "never-wrap.gguf";
const std::string kPle = "per_layer_token_embd.weight";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

WindowSources make_sources(std::uint64_t file_size, std::vector<WindowTensorSource> tensors,
                           std::vector<WindowTensorExclusion> excluded) {
    WindowSources sources{};
    sources.files.push_back({kPath, file_size, std::move(tensors), std::move(excluded), 0, 0});
    return sources;
}

WindowTensorSource dense(const std::string& name, std::uint64_t offset, std::uint64_t bytes) {
    return {name, offset, bytes, 0};
}

WindowTensorExclusion ple(std::uint64_t offset, std::uint64_t bytes) {
    return {kPle, offset, bytes, WindowTensorExclusionReason::ple};
}

WindowTensorExclusion converted(std::uint64_t offset, std::uint64_t bytes) {
    return {"blk.0.ffn_gate_inp.weight", offset, bytes, WindowTensorExclusionReason::converted};
}

// A slice whose window_offset is derived from the window that holds it, as the planner does.
WindowSlice slice(const std::string& name, std::uint64_t window_index, std::uint64_t window_start,
                  std::uint64_t file_offset, std::uint64_t bytes) {
    return {name, window_index, file_offset, file_offset - window_start, bytes, kNoExpertIndex};
}

WindowPlan make_plan(std::vector<FileWindow> windows, std::vector<WindowSlice> slices,
                     std::uint64_t unread) {
    WindowPlan plan;
    FileWindowPlan file;
    file.path = kPath;
    file.windows = std::move(windows);
    file.slices = std::move(slices);
    plan.files.push_back(std::move(file));
    plan.wrapped_unread_bytes = unread;
    return plan;
}

bool same_plan(const WindowPlan& left, const WindowPlan& right) {
    if (left.wrapped_unread_bytes != right.wrapped_unread_bytes ||
        left.files.size() != right.files.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.files.size(); ++i) {
        const FileWindowPlan& a = left.files[i];
        const FileWindowPlan& b = right.files[i];
        if (a.path != b.path || a.windows.size() != b.windows.size() ||
            a.slices.size() != b.slices.size()) {
            return false;
        }
        for (std::size_t w = 0; w < a.windows.size(); ++w) {
            if (a.windows[w].start != b.windows[w].start || a.windows[w].end != b.windows[w].end) {
                return false;
            }
        }
        for (std::size_t s = 0; s < a.slices.size(); ++s) {
            const WindowSlice& x = a.slices[s];
            const WindowSlice& y = b.slices[s];
            if (x.tensor_name != y.tensor_name || x.window_index != y.window_index ||
                x.file_offset != y.file_offset || x.window_offset != y.window_offset ||
                x.byte_count != y.byte_count || x.expert_index != y.expert_index) {
                return false;
            }
        }
    }
    return true;
}

WindowPlan plan_or_fail(const WindowSources& sources, std::uint64_t max_buffer) {
    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, max_buffer, plan, error),
            "planner rejected the fixture: " + error);
    return plan;
}

// Rounded-out end of a page-aligned range, computed without the planner's helpers.
std::uint64_t page_up(std::uint64_t value) {
    return (value + kPage - 1) / kPage * kPage;
}

// The window end at a file end is the mapped page end. Exact fit is accepted and one page
// beyond it, which wraps past the mapped file, is rejected.
void test_exact_fit_and_one_over_at_end_of_file() {
    const std::uint64_t file_size = 5 * kPage + 100;
    const std::uint64_t mapped_end = 6 * kPage;
    const WindowSources sources =
        make_sources(file_size, {dense("token_embd.weight", 0, file_size)}, {});

    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 1 && planned.files[0].windows[0].end == mapped_end,
            "planner must end the last window at the mapped page end");
    std::string error;
    require(strata::metal::verify_window_never_wrap(sources, planned, error),
            "exact fit at the mapped file end must pass: " + error);

    const WindowPlan exact = make_plan({{0, mapped_end}},
                                       {slice("token_embd.weight", 0, 0, 0, file_size)},
                                       mapped_end - file_size);
    require(strata::metal::verify_window_never_wrap(sources, exact, error),
            "hand-built exact fit at the mapped file end must pass: " + error);

    const WindowPlan one_over = make_plan({{0, mapped_end + kPage}},
                                          {slice("token_embd.weight", 0, 0, 0, file_size)},
                                          mapped_end + kPage - file_size);
    const WindowPlan one_over_copy = one_over;
    require(!strata::metal::verify_window_never_wrap(sources, one_over, error) &&
                error.find("extends past the mapped file") != std::string::npos,
            "a window one page past the mapped file end must be rejected: " + error);
    require(same_plan(one_over, one_over_copy), "a rejected plan must not change");

    const WindowPlan unaligned = make_plan({{0, file_size}},
                                           {slice("token_embd.weight", 0, 0, 0, file_size)}, 0);
    require(!strata::metal::verify_window_never_wrap(sources, unaligned, error) &&
                error.find("not page-aligned") != std::string::npos,
            "a window ending at the exact file size must be rejected: " + error);
}

// Outward padding: the window ends at the page end of the read tensor, so it covers PLE bytes
// only up to the first page boundary past the read end. Exact fit is accepted; one page more is
// one over and is rejected.
void test_ple_padding_exact_fit_and_one_over() {
    const std::uint64_t read_end = 2 * kPage + 100;
    const std::uint64_t ple_end = read_end + kMiB;
    const WindowSources sources = make_sources(ple_end, {dense("blk.0.attn_q.weight", 0, read_end)},
                                               {ple(read_end, kMiB)});

    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 1 &&
                planned.files[0].windows[0].end == page_up(read_end) &&
                planned.wrapped_unread_bytes == page_up(read_end) - read_end,
            "planner must round the read tensor out to its page end only");
    std::string error;
    require(strata::metal::verify_window_never_wrap(sources, planned, error),
            "PLE padding of one page at the read's page end must pass: " + error);

    const WindowPlan exact = make_plan({{0, page_up(read_end)}},
                                       {slice("blk.0.attn_q.weight", 0, 0, 0, read_end)},
                                       page_up(read_end) - read_end);
    require(strata::metal::verify_window_never_wrap(sources, exact, error),
            "hand-built exact PLE padding must pass: " + error);

    const WindowPlan one_over = make_plan({{0, page_up(read_end) + kPage}},
                                          {slice("blk.0.attn_q.weight", 0, 0, 0, read_end)},
                                          page_up(read_end) + kPage - read_end);
    require(!strata::metal::verify_window_never_wrap(sources, one_over, error) &&
                error.find("more than one PLE boundary page") != std::string::npos,
            "PLE padding of one page more than the read's page end must be rejected: " + error);
}

// A window that enters PLE with no read tensor beside the padding is PLE-only padding.
void test_ple_padding_needs_a_read_in_the_same_window() {
    const std::uint64_t ple_start = 2 * kPage;
    const WindowSources sources = make_sources(10 * kPage, {dense("blk.0.attn_q.weight", 0, kPage)},
                                               {ple(ple_start, 8 * kPage)});
    std::string error;
    require(strata::metal::verify_window_never_wrap(
                sources, make_plan({{0, kPage}}, {slice("blk.0.attn_q.weight", 0, 0, 0, kPage)}, 0),
                error),
            "the read tensor alone must pass: " + error);

    const WindowPlan padding_only = make_plan(
        {{0, kPage}, {kPage, 3 * kPage}}, {slice("blk.0.attn_q.weight", 0, 0, 0, kPage)},
        2 * kPage);
    require(!strata::metal::verify_window_never_wrap(sources, padding_only, error) &&
                error.find("does not share a page with the read tensor before it") !=
                    std::string::npos,
            "a window whose only PLE bytes are padding with no read beside them must be rejected: " +
                error);
}

// The wrap attempt: a merged window that would run from one read tensor across the PLE payload
// into a read tensor after it. The planner merges across the gap, and the verifier rejects the
// plan without changing it.
void test_wrap_across_ple_is_rejected_and_unchanged() {
    const std::uint64_t a_end = 2 * kPage + 100;
    const std::uint64_t ple_end = 5 * kPage + 100;
    const std::uint64_t file_size = 6 * kPage + 100;
    const WindowSources sources = make_sources(
        file_size, {dense("blk.0.attn_q.weight", 0, a_end), dense("blk.0.ffn_norm.weight", ple_end, kPage)},
        {ple(a_end, ple_end - a_end)});
    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 1 && planned.files[0].windows[0].end == 7 * kPage,
            "the planner must merge the two reads across the PLE gap in this fixture");

    const WindowPlan before = planned;
    std::string error;
    require(!strata::metal::verify_window_never_wrap(sources, planned, error) &&
                error.find("more than one PLE boundary page") != std::string::npos,
            "a window that wraps the PLE payload must be rejected: " + error);
    require(same_plan(before, planned), "a rejected wrap attempt must not change the plan");
}

// A tiny PLE payload inside one page shared by two read tensors is padding on both sides, and is
// accepted. Each side must be a read tensor in the same page.
void test_small_ple_inside_a_shared_page_is_padding() {
    const WindowSources sources = make_sources(
        300, {dense("blk.0.attn_q.weight", 0, 100), dense("blk.0.ffn_norm.weight", 150, 150)},
        {ple(100, 50)});
    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 1 && planned.files[0].windows[0].end == kPage &&
                planned.wrapped_unread_bytes == kPage - 250,
            "the planner must cover both reads and the PLE bytes with one page");
    std::string error;
    require(strata::metal::verify_window_never_wrap(sources, planned, error),
            "a PLE payload inside a page shared by two reads must pass: " + error);
}

// The window union counts a byte once even when windows overlap. wrapped_unread is the union
// minus the included bytes, so a count that double-counts the overlap must be rejected.
void test_union_counts_overlap_once() {
    const WindowSources sources = make_sources(
        300, {dense("blk.0.attn_q.weight", 0, 100), dense("blk.0.ffn_norm.weight", 200, 100)}, {});
    const std::vector<WindowSlice> slices{
        slice("blk.0.attn_q.weight", 0, 0, 0, 100),
        slice("blk.0.ffn_norm.weight", 0, 0, 200, 100),
    };
    std::string error;
    require(strata::metal::verify_window_never_wrap(
                sources, make_plan({{0, kPage}, {0, kPage}}, slices, kPage - 200), error),
            "identical overlapping windows must count their union once: " + error);
    require(!strata::metal::verify_window_never_wrap(
                sources, make_plan({{0, kPage}, {0, kPage}}, slices, 2 * kPage - 200), error) &&
                error.find("does not match wrapped_unread_bytes") != std::string::npos,
            "double-counting an overlapping window must be rejected: " + error);
}

// A window over bytes that the plan does not count is an unplanned wrap.
void test_unplanned_window_bytes_are_rejected() {
    const WindowSources sources =
        make_sources(3 * kPage, {dense("blk.0.attn_q.weight", 0, 100)}, {});
    std::string error;
    require(!strata::metal::verify_window_never_wrap(
                sources,
                make_plan({{0, kPage}, {2 * kPage, 3 * kPage}},
                          {slice("blk.0.attn_q.weight", 0, 0, 0, 100)}, kPage - 100),
                error) &&
                error.find("does not match wrapped_unread_bytes") != std::string::npos,
            "a window over a gap the plan does not count must be rejected: " + error);
}

// Payload ranges and slices are checked against the source inventory.
void test_payloads_and_slices_must_match_the_inventory() {
    std::string error;
    const WindowSources overlapping = make_sources(
        3 * kPage, {dense("blk.0.attn_q.weight", 0, 1000)}, {ple(500, 1500)});
    require(!strata::metal::verify_window_never_wrap(
                overlapping, make_plan({{0, kPage}}, {slice("blk.0.attn_q.weight", 0, 0, 0, 1000)}, 0),
                error) &&
                error.find("payloads overlap") != std::string::npos,
            "an excluded PLE payload overlapping a read tensor must be rejected: " + error);

    const WindowSources clean = make_sources(3 * kPage, {dense("blk.0.attn_q.weight", 0, 1000)},
                                             {ple(2000, 1000)});
    require(!strata::metal::verify_window_never_wrap(
                clean,
                make_plan({{0, kPage}}, {slice(kPle, 0, 0, 0, 1000)}, kPage - 1000), error) &&
                error.find("unknown tensor") != std::string::npos,
            "the PLE tensor must never have a window slice: " + error);
}

// A converted router exclusion is not a barrier: the windows wrap its bytes and the plan counts
// them as unread.
void test_converted_exclusion_is_wrapped_and_counted() {
    const WindowSources sources = make_sources(
        300, {dense("blk.0.attn_q.weight", 0, 100), dense("blk.0.ffn_norm.weight", 200, 100)},
        {converted(100, 100)});
    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 1 && planned.wrapped_unread_bytes == kPage - 200,
            "the planner must wrap the converted router bytes between two reads");
    std::string error;
    require(strata::metal::verify_window_never_wrap(sources, planned, error),
            "a wrapped converted exclusion must pass: " + error);
}

// A PLE payload with a gap of at least 256 MiB on each side is never inside a planned window.
void test_planner_keeps_wide_ple_gap_unwrapped() {
    const std::uint64_t ple_start = 64 * kPage;
    const std::uint64_t ple_bytes = kMiB;
    const std::uint64_t second_start = ple_start + ple_bytes + 300 * kMiB;
    const WindowSources sources =
        make_sources(second_start + kPage,
                     {dense("token_embd.weight", 0, ple_start),
                      dense("blk.0.attn_q.weight", second_start, kPage)},
                     {ple(ple_start, ple_bytes)});
    const WindowPlan planned = plan_or_fail(sources, kMaxBuffer);
    require(planned.files[0].windows.size() == 2, "a 256 MiB gap must split the plan in two");
    for (const FileWindow& window : planned.files[0].windows) {
        require(window.end <= ple_start || window.start >= ple_start + ple_bytes,
                "a planned window must not cover the PLE payload");
    }
    require(planned.wrapped_unread_bytes == 0, "no bytes are unread in this fixture");
    std::string error;
    require(strata::metal::verify_window_never_wrap(sources, planned, error),
            "the planned wide-gap layout must pass: " + error);
}

}  // namespace

int main() {
    try {
        test_exact_fit_and_one_over_at_end_of_file();
        test_ple_padding_exact_fit_and_one_over();
        test_ple_padding_needs_a_read_in_the_same_window();
        test_wrap_across_ple_is_rejected_and_unchanged();
        test_small_ple_inside_a_shared_page_is_padding();
        test_union_counts_overlap_once();
        test_unplanned_window_bytes_are_rejected();
        test_payloads_and_slices_must_match_the_inventory();
        test_converted_exclusion_is_wrapped_and_counted();
        test_planner_keeps_wide_ple_gap_unwrapped();
        std::cout << "metal_never_wrap_host_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_never_wrap_host_test: " << exception.what() << '\n';
        return 1;
    }
}
