// tests/metal/options_dispositions_test.cpp - CLI inventory, disposition, arity and validation checks.
#include "strata/metal/options.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace strata::metal;

namespace {

struct ExpectedGroup {
    OptionDisposition disposition;
    bool warns;
    std::string_view names;
};

constexpr ExpectedGroup kGroups[] = {
    {OptionDisposition::Keep, false,
     R"(--help --pack --native-head-gguf --embd-gguf --native-dense-gguf --ple-gguf --no-ple --ple-row-cache
        --ple-inflight --ple-delay-us --ple-sync-submit --no-ple-prefetch --max-context --kv --rope-scaling
        --rope-scale --rope-freq-base --rope-freq-scale --yarn-orig-ctx --yarn-ext-factor --yarn-attn-factor
        --yarn-beta-fast --yarn-beta-slow --max-new --greedy --seed --top-k --top-p --temperature --stop-eos
        --eos-ids --tokens --tokens-file --expert-profile --prefill-until --dump-final-r --spec --mtp
        --mtp-window --mtp-max-t --spec-min-p --suffix-draft --coupled-draft --no-coupled-draft --spec-oracle
        --spec-corrupt --serve --vision --prompt-cache --prompt-cache-every --prompt-cache-root --turn-token
        --short-read --conversation-cache-mib --conversation-cache-slots --conversation-cache-min-free-mib
        --control-vector --control-vector-scaled --control-vector-layer-range --cvec-mode --cvec-dir --dump-logits
        --logits-stride --dump-residual --dump-mixed --dump-layers --dump-halves --dump-routing --check-logits)"},
    {OptionDisposition::Mac, false,
     R"(--native --ple-io --expert-cache --expert-cache-per-layer --adapt-every --adapt-swaps --vram-reserve-mib
        --resident-budget-gib --prefill --stats --stage-timing --sync-every-layer --no-capture)"},
    {OptionDisposition::Ignore, false,
     R"(--native-bf16 --native-bf16-extra --native-ple-key --native-moe-combine --native-gdn
        --native-flash-attn-short --native-qsa-indexer --native-qsa --native-rope --native-ple-postops
        --native-router --gr-fp32-activations --gr-native-mmvf --resident-experts --resident-cpu-experts
        --mmap-experts --no-pool --gpu-only-full --graph-only --gpu-stages --stream-token)"},
    {OptionDisposition::Ignore, true,
     R"(--cpu-oracle-q8-0 --kv-resident --expert-cache-device1 --expert-cache-device2 --expert-cache-device3
        --expert-cache-remote-placement --expert-cache-cpu-order --no-hit-poke --shared-expert-arena --pool-workers
        --pool-affinity --no-host-worker --pcie-frac --pcie-mode --no-split-rows --no-prefill-borrow --spec-split
        --no-spec-split --layer-split --split-device --split-skip-if-fits --shared-late --keep-canonical
        --no-token-graph --no-fused-gr --no-fast-attn --no-publish-kernel --no-fused-gdn --no-fast-select)"},
    {OptionDisposition::New, false,
     R"(--version --list-devices --self-test --metallib --metal-capture --plan-only --geometry
        --write-geometry --geometry-family --geometry-model --plan-w-mib --plan-ram-mib --gpu-budget-mib
        --expert-slots --storage-check)"},
};

constexpr std::string_view kOneValueFlags = R"(--pack --native --native-head-gguf --embd-gguf --native-dense-gguf
    --ple-gguf --ple-io --ple-row-cache --ple-inflight --ple-delay-us --max-context --kv --kv-resident
    --rope-scaling --rope-scale --rope-freq-base --rope-freq-scale --yarn-orig-ctx --yarn-ext-factor
    --yarn-attn-factor --yarn-beta-fast --yarn-beta-slow --max-new --seed --top-k --top-p --temperature
    --eos-ids --tokens --tokens-file --expert-cache --expert-profile --adapt-every --adapt-swaps
    --vram-reserve-mib --resident-budget-gib --expert-cache-device1 --expert-cache-device2 --expert-cache-device3
    --expert-cache-remote-placement --shared-expert-arena --pool-workers --pool-affinity --pcie-frac --pcie-mode
    --prefill --prefill-until --dump-final-r --spec --mtp --mtp-window --mtp-max-t --spec-min-p --suffix-draft
    --spec-oracle --spec-corrupt --layer-split --split-device --prompt-cache --prompt-cache-every
    --prompt-cache-root --turn-token --short-read --conversation-cache-mib --conversation-cache-slots
    --conversation-cache-min-free-mib --control-vector --control-vector-scaled --cvec-mode --cvec-dir
    --dump-logits --logits-stride --dump-residual --dump-mixed --dump-layers --dump-halves --dump-routing
    --metallib --metal-capture --geometry --write-geometry --geometry-family --geometry-model --plan-w-mib
    --plan-ram-mib --gpu-budget-mib --expert-slots)";

int failures = 0;

void check(bool condition, const std::string& name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", name.c_str(),
                detail.empty() ? "" : ": ", detail.c_str());
    failures += !condition;
}

std::vector<std::string> words(std::string_view text) {
    std::istringstream input{std::string(text)};
    std::vector<std::string> result;
    for (std::string item; input >> item;) result.push_back(std::move(item));
    return result;
}

std::vector<std::string_view> views(const std::vector<std::string>& owned) {
    std::vector<std::string_view> result;
    result.reserve(owned.size());
    for (const std::string& item : owned) result.emplace_back(item);
    return result;
}

bool has_error(const std::vector<std::string>& arguments, std::string_view expected) {
    const OptionParseResult result = parse_options(views(arguments));
    return result.exit_code == 2 && result.error.find(expected) != std::string::npos;
}

}  // namespace

int main() {
    std::unordered_map<std::string, std::pair<OptionDisposition, bool>> expected;
    for (const ExpectedGroup& group : kGroups) {
        for (const std::string& name : words(group.names)) {
            const bool inserted = expected.emplace(name, std::make_pair(group.disposition, group.warns)).second;
            check(inserted, "expected inventory has no duplicate", name);
        }
    }

    const std::vector<std::string> one_value = words(kOneValueFlags);
    std::unordered_map<std::string, std::uint8_t> expected_arity;
    for (const std::string& name : one_value) {
        check(expected_arity.emplace(name, 1).second, "value inventory has no duplicate", name);
    }
    expected_arity["--control-vector-layer-range"] = 2;

    const auto& specs = option_specs();
    check(specs.size() == expected.size(), "registry contains every documented option",
          std::to_string(specs.size()) + " registered / " + std::to_string(expected.size()) + " expected");
    for (const OptionSpec& spec : specs) {
        const auto found = expected.find(std::string(spec.name));
        check(found != expected.end(), "registered option appears in expected inventory", std::string(spec.name));
        if (found == expected.end()) continue;
        check(spec.disposition == found->second.first, "disposition matches docs", std::string(spec.name));
        check(spec.warn_when_ignored == found->second.second, "ignore warning policy matches docs",
              std::string(spec.name));
        const auto arity = expected_arity.find(std::string(spec.name));
        const std::uint8_t want_arity = arity == expected_arity.end() ? 0 : arity->second;
        check(spec.value_count == want_arity, "argument count matches CLI flag", std::string(spec.name));
    }
    for (const auto& [name, _] : expected) {
        check(find_option(name) != nullptr, "expected flag is registered", name);
    }

    for (const OptionSpec& spec : specs) {
        std::vector<std::string> owned{std::string(spec.name)};
        for (std::uint8_t i = 0; i < spec.value_count; ++i) {
            owned.push_back(spec.name == "--plan-w-mib" || spec.name == "--plan-ram-mib" ||
                                    spec.name == "--gpu-budget-mib" || spec.name == "--logits-stride"
                                ? "4096"
                                : "value");
        }
        if (spec.name == "--geometry") owned.emplace_back("--plan-only");
        if (spec.name == "--geometry-family" || spec.name == "--geometry-model") {
            owned.emplace_back("--write-geometry");
            owned.emplace_back("out.txt");
        }
        if (spec.name == "--plan-w-mib" || spec.name == "--plan-ram-mib") owned.emplace_back("--plan-only");
        if (spec.name == "--logits-stride") {
            owned.emplace_back("--max-new");
            owned.emplace_back("1");
        }
        const bool device_action = spec.name == "--help" || spec.name == "--version" ||
                                   spec.name == "--list-devices" || spec.name == "--self-test" ||
                                   spec.name == "--metallib" ||
                                   spec.name == "--metal-capture";
        const bool needs_native = spec.disposition == OptionDisposition::Keep ||
                                  spec.disposition == OptionDisposition::Mac ||
                                  (spec.disposition == OptionDisposition::New && !device_action &&
                                   spec.name != "--geometry");
        if (needs_native && spec.name != "--native") {
            owned.emplace_back("--native");
            owned.emplace_back("model.gguf");
        }
        const OptionParseResult parsed = parse_options(views(owned));
        check(parsed.exit_code == 0, "parse registered flag", std::string(spec.name) +
                                                           (parsed.error.empty() ? "" : " / " + parsed.error));
        check(parsed.options.has(spec.name), "record registered flag", std::string(spec.name));
        check(parsed.options.warnings.size() ==
                  static_cast<std::size_t>(spec.disposition == OptionDisposition::Ignore && spec.warn_when_ignored),
              "emit documented ignore warning count", std::string(spec.name));
        if (spec.value_count != 0) {
            check(parsed.options.values.at(std::string(spec.name)).size() == spec.value_count,
                  "record all flag values", std::string(spec.name));
        }
        if (spec.value_count != 0) {
            const OptionParseResult missing = parse_options({spec.name});
            check(missing.exit_code == 2 &&
                      missing.error == std::string(spec.name) + " needs a value",
                  "missing value exits 2", std::string(spec.name));
        }
    }

    const OptionParseResult unknown = parse_options({"--not-a-real-option"});
    check(unknown.exit_code == 2 && unknown.error.find("unknown argument: --not-a-real-option") != std::string::npos &&
              unknown.error.find("usage: strata") != std::string::npos,
          "unknown option exits 2 and includes usage", unknown.error);
    const OptionParseResult help = parse_options({"-h"});
    check(help.exit_code == 0 && help.options.has("--help"), "-h aliases --help");
    const OptionParseResult leading_dash_path = parse_options({"--native", "--model.gguf"});
    check(leading_dash_path.exit_code == 0 &&
              leading_dash_path.options.last_value("--native") == "--model.gguf",
          "a required value may begin with --", leading_dash_path.error);

    check(has_error({"--geometry", "mac.json"}, "--geometry is valid only with --plan-only"),
          "geometry requires --plan-only");
    check(parse_options({"--geometry", "mac.json", "--plan-only"}).exit_code == 0,
          "geometry and plan-only are valid together");
    for (const char* input : {"--pack", "--native", "--ple-gguf", "--mtp"}) {
        std::vector<std::string> args{"--geometry", "mac.json", "--plan-only", input, "value"};
        check(has_error(args, "--geometry cannot be combined"), "geometry rejects file inputs", input);
    }
    check(has_error({"--geometry", "mac.json", "--plan-only", "--write-geometry", "out.json"},
                    "--geometry cannot be combined with --write-geometry"),
          "geometry rejects write-geometry");
    check(has_error({"--write-geometry", "out.json", "--plan-only"},
                    "--write-geometry cannot be combined with --plan-only"),
          "write-geometry rejects plan-only");
    check(parse_options({"--storage-check", "--native", "model.gguf"}).exit_code == 0,
          "storage check accepts a native shard");
    check(has_error({"--storage-check"}, "the Metal engine needs --native"),
          "storage check requires a native shard");
    check(has_error({"--storage-check", "--native", "model.gguf", "--plan-only"},
                    "--storage-check cannot be combined"),
          "storage check rejects the light planning mode");
    check(has_error({"--write-geometry", "out.json", "--metal-capture", "trace"},
                    "--write-geometry cannot be combined with --metal-capture"),
          "write-geometry rejects metal capture");
    check(has_error({"--geometry-family", "qwen"}, "--geometry-family is valid only with --write-geometry"),
          "geometry identity rejects use outside geometry export");
    for (const char* value : {"bad model", "bad#model", "bad=model"}) {
        check(has_error({"--geometry-model", value, "--write-geometry", "out.json",
                         "--native", "model.gguf"},
                        "--geometry-model needs a non-empty valid UTF-8 value"),
              "geometry identity rejects values that cannot be serialized", value);
    }
    const std::string invalid_utf8(1, static_cast<char>(0xff));
    check(has_error({"--geometry-model", invalid_utf8, "--write-geometry", "out.json",
                     "--native", "model.gguf"},
                    "--geometry-model needs a non-empty valid UTF-8 value"),
          "geometry identity rejects malformed UTF-8 at argument parsing");
    check(has_error({"--plan-w-mib", "4096"}, "--plan-w-mib is valid only with --plan-only"),
          "plan W simulation requires plan-only");
    check(has_error({"--plan-ram-mib", "65536"}, "--plan-ram-mib is valid only with --plan-only"),
          "plan RAM simulation requires plan-only");
    check(parse_options({"--plan-only", "--native", "model.gguf", "--plan-w-mib", "4096",
                         "--plan-ram-mib", "65536"}).exit_code == 0,
          "both simulated memory values are accepted with plan-only");
    for (const char* value : {"0", "-1", "abc"}) {
        check(has_error({"--plan-only", "--plan-w-mib", value}, "--plan-w-mib needs a positive integer"),
              "plan W simulation validates positive integer", value);
        check(has_error({"--plan-only", "--plan-ram-mib", value}, "--plan-ram-mib needs a positive integer"),
              "plan RAM simulation validates positive integer", value);
    }
    check(has_error({"--gpu-budget-mib", "0"}, "--gpu-budget-mib needs a positive integer"),
          "GPU budget validates positive integer");
    check(has_error({"--plan-only", "--native", "model.gguf", "--plan-w-mib", "4096",
                     "--plan-w-mib", "0"},
                    "--plan-w-mib needs a positive integer"),
          "validate every repeated plan W value");
    check(has_error({"--plan-only", "--native", "model.gguf", "--plan-ram-mib", "65536",
                     "--plan-ram-mib", "0"},
                    "--plan-ram-mib needs a positive integer"),
          "validate every repeated plan RAM value");
    check(has_error({"--native", "model.gguf", "--gpu-budget-mib", "4096", "--gpu-budget-mib", "0"},
                    "--gpu-budget-mib needs a positive integer"),
          "validate every repeated GPU budget value");
    // --logits-stride keeps rows of a one-token run (docs/mac/03, section 9.1): it needs an explicit --max-new 1.
    check(has_error({"--native", "model.gguf", "--dump-logits", "logits.bin", "--logits-stride", "2"},
                    "--logits-stride needs --max-new 1"),
          "logits stride requires --max-new, and the default of 16 is not 1");
    check(has_error({"--native", "model.gguf", "--logits-stride", "2", "--max-new", "2"},
                    "--logits-stride needs --max-new 1"),
          "logits stride rejects a --max-new other than 1");
    check(has_error({"--native", "model.gguf", "--logits-stride", "2", "--max-new", "1", "--max-new", "4"},
                    "--logits-stride needs --max-new 1"),
          "the last --max-new value decides for logits stride");
    check(parse_options({"--native", "model.gguf", "--dump-logits", "logits.bin", "--logits-stride", "3",
                         "--max-new", "1"}).exit_code == 0,
          "logits stride is accepted with --max-new 1");
    for (const char* value : {"0", "-1", "abc"}) {
        check(has_error({"--native", "model.gguf", "--max-new", "1", "--logits-stride", value},
                        "--logits-stride needs a positive integer"),
              "logits stride validates a positive integer", value);
    }
    check(has_error({"--plan-only"}, "the Metal engine needs --native"),
          "real-file plan requires native shard");
    check(has_error({"--write-geometry", "out.json"}, "the Metal engine needs --native"),
          "geometry export requires native shard");

    std::printf("options_dispositions_test: %s (%zu flags)\n", failures ? "FAILED" : "OK", specs.size());
    return failures != 0;
}
