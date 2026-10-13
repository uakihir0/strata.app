// tests/metal/kernel_env_dispositions_test.cpp - every kernel flag and kernel-scope variable of docs/mac/04, section 9
// (work item L14), checked against the code that reads it.
//
// Flags go through the production parser. STRATA_METAL_DISPATCH and STRATA_PREFILL_BF16X2 are read through the
// functions of strata/metal/kernel_env.hpp, and STRATA_METAL_GEMM through resolve_prefill_gemm_backend; the test sets
// each variable with setenv and then reads it with getenv, as the engine does. A variable that has no callable reader
// is checked against the Mac sources: an ignored name is read by nothing (section 9.3), and an honoured name must be
// read by some Mac source. Two dispositions are checked by running the strata executable: the STRATA_METAL_ICB notice
// (section 8.3) and the rejection of --check-logits, which section 9.2 keeps but Phase 1 does not read.
//
// Phase 1 amendments (docs/mac/IMPLEMENTATION-DECISIONS.md, D-IMPL-960 to D-IMPL-963): STRATA_KV_ROT,
// STRATA_PREFILL_TIMING, STRATA_PREFILL_DUMP_R and STRATA_DBG_NAN are ignored until the window or the prompt path
// reads them; --check-logits is not read until the window exists; STRATA_PREFILL_BF16X2 is read with atoi.
#include "strata/metal/kernel_env.hpp"
#include "strata/metal/options.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"

#include <sys/wait.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

namespace fs = std::filesystem;

#ifndef STRATA_MAC_SOURCE_DIR
#error "STRATA_MAC_SOURCE_DIR must name the repository root"
#endif
#ifndef STRATA_MAC_SPEC
#error "STRATA_MAC_SPEC must name docs/mac/04-metal-kernels.md"
#endif
#ifndef STRATA_CLI_BINARY
#error "STRATA_CLI_BINARY must name the strata executable"
#endif

int checks = 0;
int failures = 0;

// The detail says why a check failed; it is printed only then.
void check(bool condition, const std::string& name, const std::string& detail = {}) {
    ++checks;
    if (!condition) ++failures;
    if (condition) {
        std::printf("ok   %s\n", name.c_str());
    } else {
        std::printf("FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    }
}

// ---- section 9.2: kernel flags. The rule of section 9.1 gives the warning: a flag that selects another upstream arm
// warns; a flag whose arm is always on does not. No flag is read outside the parser in Phase 1. A KEEP flag is
// rejected by the CLI with the other inference options until its engine path exists; the test checks that for each.
struct FlagRow {
    std::string_view name;
    OptionDisposition disposition;
    bool warns;
};

constexpr FlagRow kFlags[] = {
    {"--gr-native-mmvf", OptionDisposition::Ignore, false},
    {"--gr-fp32-activations", OptionDisposition::Ignore, false},
    {"--cpu-oracle-q8-0", OptionDisposition::Ignore, true},
    {"--stream-token", OptionDisposition::Ignore, false},
    {"--check-logits", OptionDisposition::Keep, false},
    {"--shared-late", OptionDisposition::Ignore, true},
    {"--keep-canonical", OptionDisposition::Ignore, true},
    {"--no-token-graph", OptionDisposition::Ignore, true},
    {"--no-fused-gr", OptionDisposition::Ignore, true},
    {"--no-fast-attn", OptionDisposition::Ignore, true},
    {"--no-publish-kernel", OptionDisposition::Ignore, true},
    {"--no-fused-gdn", OptionDisposition::Ignore, true},
    {"--no-fast-select", OptionDisposition::Ignore, true},
};

// ---- section 9.3: kernel-scope variables. Honoured: a Mac source reads the name. Ignored: no Mac source reads it.
// Notice: the variable prints the stated line and is otherwise ignored (section 8.3).
enum class Expect : std::uint8_t { honoured, ignored, notice };

struct VariableRow {
    std::string_view name;
    Expect expect;
};

constexpr VariableRow kVariables[] = {
    // section 9.3, honoured in Phase 1: the memory plan reads it as on or off
    {"STRATA_PREFILL_BF16X2", Expect::honoured},
    // section 9.3, ignored in Phase 1 (D-IMPL-960): each is honoured once the window or the prompt path reads it
    {"STRATA_KV_ROT", Expect::ignored},
    {"STRATA_PREFILL_TIMING", Expect::ignored},
    {"STRATA_PREFILL_DUMP_R", Expect::ignored},
    {"STRATA_DBG_NAN", Expect::ignored},
    // section 9.3, ignored
    {"STRATA_PROMPT_ATTN_OLD", Expect::ignored},
    {"STRATA_SELECT_OLD", Expect::ignored},
    {"STRATA_ROPE_TABLE", Expect::ignored},
    {"STRATA_MTP_BATCH", Expect::ignored},
    {"STRATA_MTP_BATCH_F16", Expect::ignored},
    {"STRATA_PLE_BATCH", Expect::ignored},
    {"STRATA_INDEXER_PER_TOKEN", Expect::ignored},
    {"STRATA_GR_UNFUSED", Expect::ignored},
    {"STRATA_PREFILL_MMQ", Expect::ignored},
    {"STRATA_GDN_CONV_SERIAL", Expect::ignored},
    {"STRATA_GDN_REC_HEADS", Expect::ignored},
    {"STRATA_GDN_PIPELINE", Expect::ignored},
    {"STRATA_GROUP_COPY", Expect::ignored},
    {"STRATA_KV_STAGE_OWN", Expect::ignored},
    {"STRATA_PREFILL_ISSUER", Expect::ignored},
    {"STRATA_PREFILL_RING", Expect::ignored},
    {"STRATA_PREFILL_STREAM_MIN", Expect::ignored},
    {"STRATA_STAGER_RING", Expect::ignored},
    {"STRATA_STAGER_THREADS", Expect::ignored},
    {"STRATA_QSA_DUMP", Expect::ignored},
    {"STRATA_QSA_DUMP_LAST", Expect::ignored},
    {"STRATA_SEL_OVERLAP", Expect::ignored},
    {"STRATA_IDX_FP16_CHECK", Expect::ignored},
    {"STRATA_GR_V3", Expect::ignored},
    {"STRATA_HC_SPLIT", Expect::ignored},
    {"STRATA_OLD_GROUPED", Expect::ignored},
    {"STRATA_GROUPED_PAIR_MIN_HITS", Expect::ignored},
    {"STRATA_PROMPT_ATTN_V1", Expect::ignored},
    {"STRATA_QSA_WARP", Expect::ignored},
    {"STRATA_SCORES_MULTI", Expect::ignored},
    {"STRATA_SELECT_WMMA", Expect::ignored},
    {"STRATA_TOPK_OLD", Expect::ignored},
    {"STRATA_OLD_SAMPLER", Expect::ignored},
    {"STRATA_SAMPLER_ONE_BLOCK", Expect::ignored},
    {"STRATA_OLD_IQ_MMVQ", Expect::ignored},
    {"STRATA_HIP_ROUTER_OLD", Expect::ignored},
    {"STRATA_HIP_WMMA", Expect::ignored},
    {"HIPBLASLT_TUNING", Expect::ignored},
    {"HIPBLASLT_VERBOSE", Expect::ignored},
    {"HIPBLASLT_TUNING_V1", Expect::ignored},
    // the Mac's own kernel switches (section 9.3, closing paragraph; sections 6.2, 7.1, 8.2 and 8.3)
    {"STRATA_METAL_GEMM", Expect::honoured},
    {"STRATA_METAL_DISPATCH", Expect::honoured},
    {"STRATA_METAL_ICB", Expect::notice},
    {"STRATA_METAL_SMALL_TILES", Expect::ignored},
};

// The Mac sources that the engine and its tests compile: the engine, its headers, the platform layer and the n-gram
// reader. src/kernels holds upstream CUDA code, which the Mac build does not compile except for ngram.cpp.
constexpr std::string_view kMacSourceDirs[] = {"src/metal", "include/strata/metal", "src/platform", "src/ngram"};

struct SourceFile {
    std::string relative;
    std::string text;
};

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool is_source(const fs::path& path) {
    static constexpr std::string_view kSuffixes[] = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".m", ".mm"};
    const std::string extension = path.extension().string();
    for (const std::string_view suffix : kSuffixes) {
        if (extension == suffix) return true;
    }
    return false;
}

std::vector<SourceFile> mac_sources() {
    std::vector<SourceFile> files;
    const fs::path root = STRATA_MAC_SOURCE_DIR;
    for (const std::string_view dir : kMacSourceDirs) {
        for (const auto& entry : fs::recursive_directory_iterator(root / dir)) {
            if (!entry.is_regular_file() || !is_source(entry.path())) continue;
            files.push_back({fs::relative(entry.path(), root).generic_string(), read_text(entry.path())});
        }
    }
    return files;
}

// The first Mac source, other than `excluded_file`, that holds `"name"` as a string literal; empty when none does.
std::string literal_reader(const std::vector<SourceFile>& files, std::string_view name,
                           std::string_view excluded_file = {}) {
    const std::string literal = "\"" + std::string(name) + "\"";
    for (const SourceFile& file : files) {
        if (!excluded_file.empty() && file.relative.ends_with(excluded_file)) continue;
        if (file.text.find(literal) != std::string::npos) return file.relative;
    }
    return {};
}

// docs/mac/04, sections 9.2 and 9.3, which hold the names of both tables.
std::string spec_dispositions() {
    const std::string text = read_text(STRATA_MAC_SPEC);
    const auto first = text.find("### 9.2 Kernel flags");
    const auto last = text.find("### 9.4 Dumps with the fused read");
    if (first == std::string::npos || last == std::string::npos || last < first) return {};
    return text.substr(first, last - first);
}

// Sets or unsets a variable, then reads it with getenv as the engine does.
const char* set_variable(const char* name, const char* value) {
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
    return std::getenv(name);
}

// ---- process checks: the strata executable, run with the environment that the caller has set.
struct CliRun {
    int exit_code = -1;
    std::string output;  // stdout and stderr together
};

CliRun run_cli(const std::string& arguments) {
    CliRun run;
    const std::string command = std::string("\"") + STRATA_CLI_BINARY + "\" " + arguments + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) return run;
    char buffer[512];
    while (std::fgets(buffer, sizeof buffer, pipe) != nullptr) run.output += buffer;
    const int status = pclose(pipe);
    if (WIFEXITED(status)) run.exit_code = WEXITSTATUS(status);
    return run;
}

// How many lines of `text` equal `line` exactly.
int count_lines(const std::string& text, std::string_view line) {
    int count = 0;
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        if (std::string_view(text).substr(start, end - start) == line) ++count;
        start = end + 1;
    }
    return count;
}

// STRATA_METAL_ICB (section 8.3): the engine reads the variable at start and prints the notice once when it is 1.
// `--version` returns before any Metal work, so the run needs no GPU.
void check_icb_notice() {
    struct Case {
        const char* label;
        const char* setting;
        int expected_lines;
    };
    constexpr Case cases[] = {
        {"unset", nullptr, 0},
        {"0", "0", 0},
        {"1", "1", 1},
    };
    for (const Case& test : cases) {
        set_variable("STRATA_METAL_ICB", test.setting);
        const CliRun run = run_cli("--version");
        const int lines = count_lines(run.output, kMetalIcbNotice);
        check(run.exit_code == 0 && lines == test.expected_lines,
              std::string("STRATA_METAL_ICB ") + test.label + " prints the section 8.3 notice " +
                  std::to_string(test.expected_lines) + " time(s)",
              "exit " + std::to_string(run.exit_code) + ", " + std::to_string(lines) + " notice line(s); output: " +
                  run.output);
    }
    set_variable("STRATA_METAL_ICB", nullptr);
}

void check_flags(const std::vector<SourceFile>& files) {
    for (const FlagRow& row : kFlags) {
        const std::string name(row.name);
        const OptionSpec* spec = find_option(row.name);
        check(spec != nullptr && spec->disposition == row.disposition && spec->warn_when_ignored == row.warns,
              "flag " + name + " is registered with its section 9.2 disposition",
              spec == nullptr ? "not registered" : "");

        // accepted: the flag alone parses, is recorded, and warns exactly when section 9.1 says it does
        std::vector<std::string_view> arguments{row.name};
        if (row.disposition == OptionDisposition::Keep) arguments.insert(arguments.end(), {"--native", "model.gguf"});
        const OptionParseResult parsed = parse_options(arguments);
        const std::string expected_warning = "strata: " + name + " has no effect on the Metal engine";
        const bool warning_ok = row.warns ? parsed.options.warnings.size() == 1 &&
                                                parsed.options.warnings.front() == expected_warning
                                          : parsed.options.warnings.empty();
        const bool accepted = parsed.exit_code == 0 && parsed.options.has(row.name) && warning_ok;
        check(accepted, "flag " + name + " is accepted with " + (row.warns ? "its warning" : "no warning"),
              parsed.error.empty() ? "warnings " + std::to_string(parsed.options.warnings.size()) + ", expected " +
                                         std::to_string(row.warns ? 1 : 0)
                                   : parsed.error);

        // rejected: a value after a flag that takes none is an unknown argument
        std::vector<std::string_view> with_value = arguments;
        with_value.push_back("1");
        const OptionParseResult extra = parse_options(with_value);
        check(extra.exit_code == 2 && extra.error.find("unknown argument: 1") != std::string::npos,
              "flag " + name + " rejects a value", extra.error);

        // read: no flag is read outside the parser in Phase 1 (section 9.2)
        const std::string reader = literal_reader(files, row.name, "options.cpp");
        check(reader.empty(), "flag " + name + " is not read by the engine", "read by " + reader);

        // a KEEP flag that the engine does not read yet is rejected by the CLI, never accepted silently (section 9.2).
        // With a model named, the rejection is the one for inference options that are not wired (src/metal/program).
        if (row.disposition == OptionDisposition::Keep) {
            const CliRun run = run_cli(name + " --native model.gguf");
            check(run.exit_code == 2 &&
                      run.output.find("planning and inference options are not wired") != std::string::npos,
                  "flag " + name + " is rejected by the CLI until its engine path exists", run.output);
        }
    }
}

void check_dispatch_mode() {
    struct Case {
        const char* label;
        const char* setting;
        bool accepted;
        DispatchMode mode;
    };
    constexpr Case cases[] = {
        {"unset (default)", nullptr, true, DispatchMode::serial},
        {"empty (default)", "", true, DispatchMode::serial},
        {"serial", "serial", true, DispatchMode::serial},
        {"concurrent", "concurrent", true, DispatchMode::concurrent},
        {"capitalised", "Concurrent", false, DispatchMode::serial},
        {"other word", "parallel", false, DispatchMode::serial},
        {"numeric", "1", false, DispatchMode::serial},
    };
    for (const Case& test : cases) {
        const std::string name = std::string("STRATA_METAL_DISPATCH ") + test.label;
        DispatchMode mode = DispatchMode::serial;
        std::string error;
        const bool accepted = parse_dispatch_mode(set_variable("STRATA_METAL_DISPATCH", test.setting), mode, error);
        const bool ok = accepted == test.accepted &&
                        (accepted ? mode == test.mode && error.empty()
                                  : error.find("STRATA_METAL_DISPATCH must be 'serial' or 'concurrent'") !=
                                        std::string::npos);
        check(ok, name, error);
    }
    set_variable("STRATA_METAL_DISPATCH", nullptr);
}

void check_gemm() {
    struct Case {
        const char* label;
        const char* setting;
        bool apple10;
        std::uint32_t n;
        bool accepted;
        PrefillGemmBackend backend;
        std::string_view error;
    };
    constexpr Case cases[] = {
        {"unset, Apple10, n = 64 (default: MPP)", nullptr, true, 64u, true, PrefillGemmBackend::MPP, ""},
        {"unset, earlier family, n = 64 (default: SIMD)", nullptr, false, 64u, true, PrefillGemmBackend::SIMD, ""},
        {"unset, Apple10, n = 63 (SIMD below 64)", nullptr, true, 63u, true, PrefillGemmBackend::SIMD, ""},
        {"empty, Apple10, n = 64 (default: MPP)", "", true, 64u, true, PrefillGemmBackend::MPP, ""},
        {"simd forces SIMD, Apple10, n = 64", "simd", true, 64u, true, PrefillGemmBackend::SIMD, ""},
        {"mpp forces MPP, Apple10, n = 64", "mpp", true, 64u, true, PrefillGemmBackend::MPP, ""},
        {"mpp below 64 stays SIMD", "mpp", true, 63u, true, PrefillGemmBackend::SIMD, ""},
        {"mpp refused on earlier family", "mpp", false, 64u, false, PrefillGemmBackend::SIMD,
         "requires an Apple10 GPU"},
        {"auto rejected", "auto", true, 64u, false, PrefillGemmBackend::SIMD, "must be 'mpp' or 'simd'"},
        {"capitalised rejected", "MPP", true, 64u, false, PrefillGemmBackend::SIMD, "must be 'mpp' or 'simd'"},
    };
    for (const Case& test : cases) {
        const char* value = set_variable("STRATA_METAL_GEMM", test.setting);
        PrefillGemmBackend backend = PrefillGemmBackend::MPP;
        std::string error = "stale";
        const bool accepted = resolve_prefill_gemm_backend(
            test.apple10, test.n, value == nullptr ? std::string_view{} : std::string_view(value), backend, error);
        const bool ok = accepted == test.accepted &&
                        (accepted ? backend == test.backend && error.empty()
                                  : backend == PrefillGemmBackend::MPP &&
                                        error.find(test.error) != std::string::npos);
        check(ok, std::string("STRATA_METAL_GEMM ") + test.label, error);
    }
    set_variable("STRATA_METAL_GEMM", nullptr);
}

void check_prefill_bf16x2() {
    // docs/mac/06, section 4.2: the value is read with atoi, so a leading whole number counts, and 0 or an unset
    // variable is off. The memory plan needs only on or off (docs/mac/05, section 3).
    struct Case {
        const char* label;
        const char* setting;
        bool enabled;
    };
    constexpr Case cases[] = {
        {"unset (default)", nullptr, false},
        {"empty (default)", "", false},
        {"zero", "0", false},
        {"one (every BF16 projection)", "1", true},
        {"two (all but the hyper-connection's)", "2", true},
        {"other non-zero", "7", true},
        {"negative", "-1", true},
        {"not a number", "abc", false},
        {"leading space", " 1", true},
        {"leading number with trailing text", "1x", true},
    };
    for (const Case& test : cases) {
        const bool enabled = prefill_bf16x2_enabled(set_variable("STRATA_PREFILL_BF16X2", test.setting));
        check(enabled == test.enabled, std::string("STRATA_PREFILL_BF16X2 ") + test.label,
              std::string("docs/mac/06 section 4.2 reads it with atoi, so the value is ") +
                  (test.enabled ? "on" : "off") + "; the engine gives " + (enabled ? "on" : "off"));
    }
    set_variable("STRATA_PREFILL_BF16X2", nullptr);
}

}  // namespace

int main() {
    try {
        const std::vector<SourceFile> files = mac_sources();
        const std::string spec = spec_dispositions();

        // the scan must see the readers that exist today, or every "not read" result below means nothing
        check(files.size() > 50, "Mac source scan covers the engine and its headers",
              std::to_string(files.size()) + " files");
        check(literal_reader(files, "STRATA_METAL_DISPATCH") == "src/metal/host/encoder.mm",
              "Mac source scan finds the STRATA_METAL_DISPATCH reader in encoder.mm");
        check(!spec.empty(), "docs/mac/04 section 9.2 and 9.3 are readable", STRATA_MAC_SPEC);

        for (const FlagRow& row : kFlags) {
            check(spec.find("`" + std::string(row.name) + "`") != std::string::npos,
                  "flag " + std::string(row.name) + " is named in docs/mac/04 section 9.2");
        }
        check_flags(files);

        for (const VariableRow& row : kVariables) {
            const std::string name(row.name);
            check(spec.find("`" + name + "`") != std::string::npos,
                  "variable " + name + " is named in docs/mac/04 section 9.2 or 9.3");
            const std::string reader = literal_reader(files, row.name);
            switch (row.expect) {
                case Expect::honoured:
                    check(!reader.empty(),
                          reader.empty() ? "honoured variable " + name + " has no Mac reader"
                                         : "honoured variable " + name + " is read by " + reader,
                          "docs/mac/04 section 9.3 says honoured, but no Mac source reads it");
                    break;
                case Expect::ignored:
                    check(reader.empty(), "ignored variable " + name + " is read by nothing", "read by " + reader);
                    break;
                case Expect::notice:
                    check(!reader.empty(), reader.empty() ? "notice variable " + name + " has no Mac reader"
                                                          : "notice variable " + name + " is read by " + reader,
                          "docs/mac/04 section 8.3 says the variable prints a notice, but no Mac source reads it");
                    break;
            }
        }

        check_icb_notice();
        check_dispatch_mode();
        check_gemm();
        check_prefill_bf16x2();
    } catch (const fs::filesystem_error& error) {
        std::printf("FAIL cannot scan the Mac sources: %s\n", error.what());
        return 1;
    }

    std::printf("kernel_env_dispositions_test: %s (%d checks, %d failed)\n", failures == 0 ? "OK" : "FAILED", checks,
                failures);
    return failures == 0 ? 0 : 1;
}
