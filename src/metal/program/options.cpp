// src/metal/program/options.cpp - Metal command-line inventory, dispositions and argument validation.
#include "strata/metal/options.hpp"

#include <charconv>
#include <sstream>
#include <system_error>
#include <utility>

namespace strata::metal {
namespace {

using D = OptionDisposition;

const std::vector<OptionSpec> kSpecs = {
    {"--help", 0, D::Keep, false},
    {"--version", 0, D::New, false},
    {"--list-devices", 0, D::New, false},
    {"--self-test", 0, D::New, false},
    {"--metallib", 1, D::New, false},
    {"--metal-capture", 1, D::New, false},

    {"--pack", 1, D::Keep, false},
    {"--native", 1, D::Mac, false},
    {"--native-bf16", 0, D::Ignore, false},
    {"--native-bf16-extra", 0, D::Ignore, false},
    {"--native-ple-key", 0, D::Ignore, false},
    {"--native-moe-combine", 0, D::Ignore, false},
    {"--native-gdn", 0, D::Ignore, false},
    {"--native-flash-attn-short", 0, D::Ignore, false},
    {"--native-qsa-indexer", 0, D::Ignore, false},
    {"--native-qsa", 0, D::Ignore, false},
    {"--native-rope", 0, D::Ignore, false},
    {"--native-ple-postops", 0, D::Ignore, false},
    {"--native-router", 0, D::Ignore, false},
    {"--native-head-gguf", 1, D::Keep, false},
    {"--embd-gguf", 1, D::Keep, false},
    {"--native-dense-gguf", 1, D::Keep, false},
    {"--ple-gguf", 1, D::Keep, false},
    {"--no-ple", 0, D::Keep, false},
    {"--ple-io", 1, D::Mac, false},
    {"--ple-row-cache", 1, D::Keep, false},
    {"--ple-inflight", 1, D::Keep, false},
    {"--ple-delay-us", 1, D::Keep, false},
    {"--ple-sync-submit", 0, D::Keep, false},
    {"--no-ple-prefetch", 0, D::Keep, false},
    {"--cpu-oracle-q8-0", 0, D::Ignore, true},
    {"--gr-fp32-activations", 0, D::Ignore, false},
    {"--gr-native-mmvf", 0, D::Ignore, false},

    {"--max-context", 1, D::Keep, false},
    {"--kv", 1, D::Keep, false},
    {"--kv-resident", 1, D::Ignore, true},
    {"--rope-scaling", 1, D::Keep, false},
    {"--rope-scale", 1, D::Keep, false},
    {"--rope-freq-base", 1, D::Keep, false},
    {"--rope-freq-scale", 1, D::Keep, false},
    {"--yarn-orig-ctx", 1, D::Keep, false},
    {"--yarn-ext-factor", 1, D::Keep, false},
    {"--yarn-attn-factor", 1, D::Keep, false},
    {"--yarn-beta-fast", 1, D::Keep, false},
    {"--yarn-beta-slow", 1, D::Keep, false},

    {"--max-new", 1, D::Keep, false},
    {"--greedy", 0, D::Keep, false},
    {"--seed", 1, D::Keep, false},
    {"--top-k", 1, D::Keep, false},
    {"--top-p", 1, D::Keep, false},
    {"--temperature", 1, D::Keep, false},
    {"--stop-eos", 0, D::Keep, false},
    {"--eos-ids", 1, D::Keep, false},
    {"--tokens", 1, D::Keep, false},
    {"--tokens-file", 1, D::Keep, false},

    {"--expert-cache", 1, D::Mac, false},
    {"--expert-cache-per-layer", 0, D::Mac, false},
    {"--expert-profile", 1, D::Keep, false},
    {"--adapt-every", 1, D::Mac, false},
    {"--adapt-swaps", 1, D::Mac, false},
    {"--vram-reserve-mib", 1, D::Mac, false},
    {"--resident-budget-gib", 1, D::Mac, false},
    {"--resident-experts", 0, D::Ignore, false},
    {"--resident-cpu-experts", 0, D::Ignore, false},
    {"--mmap-experts", 0, D::Ignore, false},
    {"--expert-cache-device1", 1, D::Ignore, true},
    {"--expert-cache-device2", 1, D::Ignore, true},
    {"--expert-cache-device3", 1, D::Ignore, true},
    {"--expert-cache-remote-placement", 1, D::Ignore, true},
    {"--expert-cache-cpu-order", 0, D::Ignore, true},
    {"--no-hit-poke", 0, D::Ignore, true},
    {"--shared-expert-arena", 1, D::Ignore, true},
    {"--pool-workers", 1, D::Ignore, true},
    {"--pool-affinity", 1, D::Ignore, true},
    {"--no-host-worker", 0, D::Ignore, true},
    {"--pcie-frac", 1, D::Ignore, true},
    {"--pcie-mode", 1, D::Ignore, true},

    {"--prefill", 1, D::Mac, false},
    {"--prefill-until", 1, D::Keep, false},
    {"--dump-final-r", 1, D::Keep, false},
    {"--no-split-rows", 0, D::Ignore, true},
    {"--no-prefill-borrow", 0, D::Ignore, true},

    {"--spec", 1, D::Keep, false},
    {"--mtp", 1, D::Keep, false},
    {"--mtp-window", 1, D::Keep, false},
    {"--mtp-max-t", 1, D::Keep, false},
    {"--spec-min-p", 1, D::Keep, false},
    {"--suffix-draft", 1, D::Keep, false},
    {"--coupled-draft", 0, D::Keep, false},
    {"--no-coupled-draft", 0, D::Keep, false},
    {"--spec-oracle", 1, D::Keep, false},
    {"--spec-corrupt", 1, D::Keep, false},
    {"--spec-split", 0, D::Ignore, true},
    {"--no-spec-split", 0, D::Ignore, true},

    {"--layer-split", 1, D::Ignore, true},
    {"--split-device", 1, D::Ignore, true},
    {"--split-skip-if-fits", 0, D::Ignore, true},

    {"--serve", 0, D::Keep, false},
    {"--vision", 0, D::Keep, false},
    {"--prompt-cache", 1, D::Keep, false},
    {"--prompt-cache-every", 1, D::Keep, false},
    {"--prompt-cache-root", 1, D::Keep, false},
    {"--turn-token", 1, D::Keep, false},
    {"--short-read", 1, D::Keep, false},
    {"--conversation-cache-mib", 1, D::Keep, false},
    {"--conversation-cache-slots", 1, D::Keep, false},
    {"--conversation-cache-min-free-mib", 1, D::Keep, false},
    {"--control-vector", 1, D::Keep, false},
    {"--control-vector-scaled", 1, D::Keep, false},
    {"--control-vector-layer-range", 2, D::Keep, false},
    {"--cvec-mode", 1, D::Keep, false},
    {"--cvec-dir", 1, D::Keep, false},
    {"--dump-logits", 1, D::Keep, false},
    {"--logits-stride", 1, D::Keep, false},
    {"--dump-residual", 1, D::Keep, false},
    {"--dump-mixed", 1, D::Keep, false},
    {"--dump-layers", 1, D::Keep, false},
    {"--dump-halves", 1, D::Keep, false},
    {"--dump-routing", 1, D::Keep, false},
    {"--stats", 0, D::Mac, false},
    {"--stage-timing", 0, D::Mac, false},
    {"--sync-every-layer", 0, D::Mac, false},
    {"--no-capture", 0, D::Mac, false},
    {"--no-pool", 0, D::Ignore, false},
    {"--gpu-only-full", 0, D::Ignore, false},
    {"--graph-only", 0, D::Ignore, false},
    {"--gpu-stages", 0, D::Ignore, false},
    {"--shared-late", 0, D::Ignore, true},
    {"--keep-canonical", 0, D::Ignore, true},
    {"--no-token-graph", 0, D::Ignore, true},
    {"--no-fused-gr", 0, D::Ignore, true},
    {"--no-fast-attn", 0, D::Ignore, true},
    {"--no-publish-kernel", 0, D::Ignore, true},
    {"--no-fused-gdn", 0, D::Ignore, true},
    {"--no-fast-select", 0, D::Ignore, true},
    {"--stream-token", 0, D::Ignore, false},
    {"--check-logits", 0, D::Keep, false},

    {"--plan-only", 0, D::New, false},
    {"--geometry", 1, D::New, false},
    {"--write-geometry", 1, D::New, false},
    {"--geometry-family", 1, D::New, false},
    {"--geometry-model", 1, D::New, false},
    {"--plan-w-mib", 1, D::New, false},
    {"--plan-ram-mib", 1, D::New, false},
    {"--gpu-budget-mib", 1, D::New, false},
    {"--expert-slots", 1, D::New, false},
    {"--storage-check", 0, D::New, false},
};

std::string usage_text() {
    return "usage: strata [--version | --list-devices | --self-test | --storage-check | --help] "
           "[--metallib PATH] [--metal-capture FILE] [--native SHARD1] [options]\n";
}

bool positive_integer(std::string_view text) {
    if (text.empty()) return false;
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
}

bool valid_utf8(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead <= 0x7f) {
            ++i;
            continue;
        }
        std::size_t count = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            count = 2;
            codepoint = lead & 0x1f;
            minimum = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            count = 3;
            codepoint = lead & 0x0f;
            minimum = 0x800;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            count = 4;
            codepoint = lead & 0x07;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (text.size() - i < count) return false;
        for (std::size_t offset = 1; offset < count; ++offset) {
            const auto continuation = static_cast<unsigned char>(text[i + offset]);
            if ((continuation & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
        i += count;
    }
    return true;
}

bool safe_geometry_identity(std::string_view text) {
    if (text.empty() || !valid_utf8(text)) return false;
    for (const unsigned char byte : text) {
        if (byte <= 0x20 || byte == '=' || byte == '#') return false;
    }
    return true;
}

void fail(OptionParseResult& result, std::string message) {
    result.exit_code = 2;
    result.error = std::move(message);
}

}  // namespace

const std::vector<OptionSpec>& option_specs() noexcept { return kSpecs; }

const OptionSpec* find_option(std::string_view name) noexcept {
    for (const OptionSpec& spec : kSpecs) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

bool Options::has(std::string_view name) const { return present.find(std::string(name)) != present.end(); }

std::string_view Options::value(std::string_view name, std::size_t index) const {
    const auto found = values.find(std::string(name));
    if (found == values.end() || index >= found->second.size()) return {};
    return found->second[index];
}

std::string_view Options::last_value(std::string_view name) const {
    const auto found = values.find(std::string(name));
    if (found == values.end() || found->second.empty()) return {};
    return found->second.back();
}

OptionParseResult parse_options(const std::vector<std::string_view>& arguments) {
    OptionParseResult result;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        const std::string_view canonical = argument == "-h" ? "--help" : argument;
        const OptionSpec* spec = find_option(canonical);
        if (spec == nullptr) {
            fail(result, "unknown argument: " + std::string(argument) + "\n" + usage_text());
            return result;
        }

        result.options.present.emplace(spec->name);
        auto& values = result.options.values[std::string(spec->name)];
        for (std::uint8_t n = 0; n < spec->value_count; ++n) {
            if (i + 1 >= arguments.size() || arguments[i + 1].empty()) {
                fail(result, std::string(spec->name) + " needs a value");
                return result;
            }
            values.emplace_back(arguments[++i]);
        }
        if (spec->disposition == OptionDisposition::Ignore && spec->warn_when_ignored) {
            result.options.warnings.emplace_back("strata: " + std::string(spec->name) +
                                                 " has no effect on the Metal engine");
        }
    }

    const Options& options = result.options;
    if (options.has("--plan-only") && options.has("--metal-capture")) {
        fail(result, "--metal-capture cannot be combined with --plan-only");
        return result;
    }
    if (options.has("--storage-check") &&
        (options.has("--plan-only") || options.has("--geometry") ||
         options.has("--write-geometry") || options.has("--list-devices") ||
         options.has("--self-test") || options.has("--metal-capture"))) {
        fail(result, "--storage-check cannot be combined with planning, geometry export, device listing, "
                     "self-test or Metal capture");
        return result;
    }
    if (options.has("--geometry")) {
        if (!options.has("--plan-only")) {
            fail(result, "--geometry is valid only with --plan-only");
            return result;
        }
        if (options.has("--pack") || options.has("--native") || options.has("--ple-gguf") ||
            options.has("--mtp")) {
            fail(result, "--geometry cannot be combined with --pack, --native, --ple-gguf or --mtp");
            return result;
        }
        if (options.has("--write-geometry")) {
            fail(result, "--geometry cannot be combined with --write-geometry");
            return result;
        }
    }
    if (options.has("--write-geometry") && options.has("--plan-only")) {
        fail(result, "--write-geometry cannot be combined with --plan-only");
        return result;
    }
    if (options.has("--write-geometry") && options.has("--metal-capture")) {
        fail(result, "--write-geometry cannot be combined with --metal-capture");
        return result;
    }
    for (const std::string_view name : {"--geometry-family", "--geometry-model"}) {
        if (options.has(name) && !options.has("--write-geometry")) {
            fail(result, std::string(name) + " is valid only with --write-geometry");
            return result;
        }
        if (options.has(name) && !safe_geometry_identity(options.last_value(name))) {
            fail(result, std::string(name) +
                         " needs a non-empty valid UTF-8 value without spaces, '=' or '#'");
            return result;
        }
    }
    for (const std::string_view name : {"--plan-w-mib", "--plan-ram-mib"}) {
        if (options.has(name) && !options.has("--plan-only")) {
            fail(result, std::string(name) + " is valid only with --plan-only");
            return result;
        }
        if (options.has(name)) {
            const auto& values = options.values.at(std::string(name));
            for (const std::string& value : values) {
                if (!positive_integer(value)) {
                    fail(result, std::string(name) + " needs a positive integer");
                    return result;
                }
            }
        }
    }
    if (options.has("--gpu-budget-mib")) {
        for (const std::string& value : options.values.at("--gpu-budget-mib")) {
            if (!positive_integer(value)) {
                fail(result, "--gpu-budget-mib needs a positive integer");
                return result;
            }
        }
    }
    if (options.has("--logits-stride")) {
        for (const std::string& value : options.values.at("--logits-stride")) {
            if (!positive_integer(value)) {
                fail(result, "--logits-stride needs a positive integer");
                return result;
            }
        }
        // The stride keeps rows of a single-token teacher-forced run (docs/mac/03, section 9.1), so --max-new must
        // be given explicitly as 1; the default of 16 is not accepted.
        if (!options.has("--max-new") || options.last_value("--max-new") != "1") {
            fail(result, "--logits-stride needs --max-new 1");
            return result;
        }
    }

    bool model_mode = false;
    for (const std::string& name : options.present) {
        if (name == "--help" || name == "--version" || name == "--list-devices" ||
            name == "--self-test" || name == "--metallib" || name == "--metal-capture") {
            continue;
        }
        const OptionSpec* spec = find_option(name);
        if (spec != nullptr && spec->disposition != OptionDisposition::Ignore) {
            model_mode = true;
            break;
        }
    }
    const bool geometry_plan = options.has("--plan-only") && options.has("--geometry");
    if (model_mode && !geometry_plan && !options.has("--native")) {
        fail(result, "the Metal engine needs --native <shard 1 of the GGUF>");
        return result;
    }
    return result;
}

void print_usage(std::FILE* stream) {
    if (stream == nullptr) return;
    const std::string text = usage_text();
    std::fputs(text.c_str(), stream);
}

}  // namespace strata::metal
