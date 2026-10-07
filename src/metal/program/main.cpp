// src/metal/program/main.cpp - Metal engine entry point.
#include "strata/metal/context.hpp"
#include "strata/metal/expert_profile.hpp"
#include "strata/metal/geometry.hpp"
#include "strata/metal/model_storage.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/options.hpp"
#include "strata/metal/plan.hpp"
#include "strata/metal/weights.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/platform/memory_probe.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::metal {
const char* engine_version();
int metal_rev();
const char* metallib_sha256();
}  // namespace strata::metal

using namespace strata::metal;

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024;
constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;

bool parse_u64(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool option_u64(const Options& options, std::string_view name, std::uint64_t fallback,
                std::uint64_t& value, std::string& error, bool allow_zero = true) {
    value = fallback;
    if (!options.has(name)) return true;
    if (!parse_u64(options.last_value(name), value) || (!allow_zero && value == 0)) {
        error = std::string(name) + (allow_zero ? " needs an unsigned integer" : " needs a positive integer");
        return false;
    }
    return true;
}

bool mib_to_bytes(std::uint64_t mib, std::uint64_t& bytes) {
    if (mib > std::numeric_limits<std::uint64_t>::max() / kMiB) return false;
    bytes = mib * kMiB;
    return true;
}

bool parse_gib(std::string_view text, std::uint64_t& bytes) {
    if (text.empty()) return false;
    const std::string copy(text);
    char* end = nullptr;
    errno = 0;
    const long double gib = std::strtold(copy.c_str(), &end);
    if (errno != 0 || end == copy.c_str() || *end != '\0' || !std::isfinite(gib) || gib < 0 ||
        gib > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) / kGiB) {
        return false;
    }
    bytes = static_cast<std::uint64_t>(gib * kGiB);
    return true;
}

bool env_u32(const char* name, std::uint32_t fallback, std::uint32_t& value) {
    value = fallback;
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return true;
    std::uint64_t parsed = 0;
    if (!parse_u64(raw, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

int geometry_plan(const Options& cli, const Geometry& geometry,
                  PlanResult* plan_output = nullptr, Context* existing_context = nullptr) {
    std::string error;
    PlanOptions options;
    options.gpu = "simulated";
    std::uint64_t plan_w_mib = 0;
    std::uint64_t plan_ram_mib = 0;
    std::uint64_t gpu_budget_mib = 0;
    std::uint64_t reserve_mib = options.reserve_mib;
    std::uint64_t max_context = options.max_context;
    std::uint64_t mtp_window = options.mtp_window;
    std::uint64_t expert_slots = options.expert_slots;
    std::uint64_t spec = 0;
    if (!option_u64(cli, "--plan-w-mib", 0, plan_w_mib, error, false) ||
        !option_u64(cli, "--plan-ram-mib", 0, plan_ram_mib, error, false) ||
        !option_u64(cli, "--gpu-budget-mib", 0, gpu_budget_mib, error, false) ||
        !option_u64(cli, "--vram-reserve-mib", reserve_mib, reserve_mib, error) ||
        !option_u64(cli, "--max-context", max_context, max_context, error, false) ||
        !option_u64(cli, "--mtp-window", mtp_window, mtp_window, error) ||
        !option_u64(cli, "--expert-slots", expert_slots, expert_slots, error) ||
        !option_u64(cli, "--spec", 0, spec, error)) {
        std::fprintf(stderr, "strata: %s\n", error.c_str());
        return 2;
    }
    if (spec > 8) {
        std::fprintf(stderr, "strata: --spec must be between 0 and 8\n");
        return 2;
    }
    if (spec > 0 && (geometry.mtp_bytes == 0 || geometry.mtp_bind_bytes == 0)) {
        std::fprintf(stderr, "strata: --spec requires geometry that includes a validated --mtp runtime\n");
        return 2;
    }
    if (reserve_mib > std::numeric_limits<std::uint64_t>::max() / kMiB ||
        max_context > 262'144 || mtp_window > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "strata: plan option is out of range\n");
        return 2;
    }
    options.reserve_mib = reserve_mib;
    options.max_context = static_cast<std::uint32_t>(max_context);
    options.mtp_window = static_cast<std::uint32_t>(mtp_window);
    options.expert_slots = expert_slots;
    options.spec_enabled = spec > 0;

    const std::string_view kv = cli.has("--kv") ? cli.last_value("--kv") : "fp16";
    if (kv == "fp16") options.kv = KvFormat::Fp16;
    else if (kv == "int8") options.kv = KvFormat::Int8;
    else if (kv == "k8v4") options.kv = KvFormat::K8v4;
    else if (kv == "q4_0" || kv == "q4") options.kv = KvFormat::Q4_0;
    else {
        std::fprintf(stderr, "strata: --kv needs fp16, int8, q4_0 or k8v4\n");
        return 2;
    }

    std::uint32_t prefill_auto_max = options.prefill_auto_max;
    std::uint32_t configured_prefill_auto_max = prefill_auto_max;
    if (!env_u32("STRATA_PREFILL_AUTO_MAX", prefill_auto_max, configured_prefill_auto_max)) {
        std::fprintf(stderr, "strata: STRATA_PREFILL_AUTO_MAX needs an unsigned 32-bit integer\n");
        return 2;
    }
    options.prefill_auto_max = configured_prefill_auto_max;
    if (const char* bf16x2 = std::getenv("STRATA_PREFILL_BF16X2"); bf16x2 != nullptr && *bf16x2 != '\0') {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(bf16x2, &end, 10);
        if (errno == 0 && end != bf16x2 && *end == '\0') options.prefill_bf16x2 = value != 0;
    }
    if (cli.has("--prefill")) {
        const std::string_view prefill = cli.last_value("--prefill");
        if (prefill == "auto") {
            if (options.prefill_auto_max < 1'024) {
                std::fprintf(stderr, "strata: STRATA_PREFILL_AUTO_MAX must be at least 1024 for --prefill auto\n");
                return 2;
            }
            options.prefill_mode = PrefillMode::Auto;
        } else if (prefill.starts_with("auto:")) {
            std::uint64_t cap = 0;
            if (!parse_u64(prefill.substr(5), cap) || cap < 1'024 ||
                cap > std::numeric_limits<std::uint32_t>::max()) {
                std::fprintf(stderr, "strata: --prefill auto:N needs an unsigned 32-bit N of at least 1024\n");
                return 2;
            }
            options.prefill_mode = PrefillMode::Auto;
            options.prefill_auto_max = static_cast<std::uint32_t>(cap);
        } else {
            std::uint64_t chunk = 0;
            if (!parse_u64(prefill, chunk) || chunk == 0 ||
                chunk > std::numeric_limits<std::uint32_t>::max()) {
                std::fprintf(stderr, "strata: --prefill needs auto, auto:N or a positive 32-bit chunk\n");
                return 2;
            }
            options.prefill_mode = PrefillMode::Fixed;
            options.prefill_chunk = static_cast<std::uint32_t>(chunk);
        }
    }

    if (cli.has("--expert-cache")) {
        const std::string_view cache = cli.last_value("--expert-cache");
        if (cache != "auto") {
            std::uint64_t count = 0;
            if (!parse_u64(cache, count) ||
                count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                std::fprintf(stderr, "strata: --expert-cache needs auto or a non-negative integer\n");
                return 2;
            }
            options.expert_cache = static_cast<std::int64_t>(count);
        }
    }
    if (cli.has("--resident-budget-gib")) {
        if (!parse_gib(cli.last_value("--resident-budget-gib"), options.resident_budget_bytes)) {
            std::fprintf(stderr, "strata: --resident-budget-gib needs a non-negative GiB value\n");
            return 2;
        }
        options.resident_budget_is_set = true;
    }
    options.expert_cache_per_layer = cli.has("--expert-cache-per-layer");
    if (cli.has("--expert-profile")) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> ranked;
        if (!read_expert_profile(std::string(cli.last_value("--expert-profile")), geometry.n_layers,
                                 geometry.n_expert, ranked, error)) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 1;
        }
        options.expert_order = std::move(ranked);
    }

    const bool has_w_override = cli.has("--plan-w-mib");
    const bool has_ram_override = cli.has("--plan-ram-mib");
    std::uint64_t working_set = 0;
    std::uint64_t ram_bytes = 0;
    std::unique_ptr<Context> owned_context;
    Context* context = existing_context;
    if (has_w_override && has_ram_override) {
        if (!mib_to_bytes(plan_w_mib, working_set) || !mib_to_bytes(plan_ram_mib, ram_bytes)) {
            std::fprintf(stderr, "strata: plan memory override is too large\n");
            return 2;
        }
        options.available_is_override = true;
    } else {
        if (context == nullptr) {
            ContextOptions context_options;
            context_options.pipelines = false;
            owned_context = Context::open(context_options, error);
            context = owned_context.get();
        }
        if (!context) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 1;
        }
        const DeviceFacts facts = context->device_facts();
        working_set = facts.working_set_bytes;
        ram_bytes = facts.physical_memory_bytes;
        options.gpu = context->device_name();
        std::replace(options.gpu.begin(), options.gpu.end(), ' ', '_');
        if (has_w_override && !mib_to_bytes(plan_w_mib, working_set)) {
            std::fprintf(stderr, "strata: --plan-w-mib is too large\n");
            return 2;
        }
        if (has_ram_override) {
            if (!mib_to_bytes(plan_ram_mib, ram_bytes)) {
                std::fprintf(stderr, "strata: --plan-ram-mib is too large\n");
                return 2;
            }
            options.available_is_override = true;
        } else if (const auto available = strata::platform::available_memory()) {
            options.available_bytes = *available;
        } else {
            options.available_is_override = true;
        }
    }
    if (ram_bytes == 0) ram_bytes = strata::platform::total_memory();
    if (working_set == 0 || ram_bytes == 0) {
        std::fprintf(stderr, "strata: could not read a non-zero GPU working set and physical RAM size\n");
        return 1;
    }
    if (gpu_budget_mib != 0) {
        std::uint64_t gpu_budget = 0;
        if (!mib_to_bytes(gpu_budget_mib, gpu_budget)) {
            std::fprintf(stderr, "strata: --gpu-budget-mib is too large\n");
            return 2;
        }
        working_set = std::min(working_set, gpu_budget);
    }
    options.working_set_bytes = working_set;
    options.ram_bytes = ram_bytes;

    PlanResult plan;
    if (!make_plan(geometry, options, plan, error)) {
        std::fprintf(stderr, "strata: %s\n", error.c_str());
        return 1;
    }
    for (const std::string& warning : plan.warnings) {
        std::fprintf(stderr, "strata: %s\n", warning.c_str());
    }
    if (plan.kv_warning) {
        const std::uint64_t kv_bytes = plan.kv_bytes + plan.mtp_kv_bytes;
        const std::uint64_t kv_mib = kv_bytes / kMiB + (kv_bytes % kMiB != 0);
        std::fprintf(stderr, "strata: KV takes %llu MiB; lower --max-context or use --kv k8v4\n",
                     static_cast<unsigned long long>(kv_mib));
    }
    std::printf("PLAN %s\n", plan_line(plan).c_str());
    if (plan_output != nullptr) *plan_output = plan;
    return plan.fits ? 0 : 3;
}

bool inspect_real_geometry(const Options& cli, Geometry& geometry, std::string& error,
                           WindowSources* window_sources = nullptr,
                           MtpRuntimeSources* mtp_sources = nullptr) {
    WeightStoreOptions inputs;
    inputs.native_shard = std::string(cli.last_value("--native"));
    if (cli.has("--pack")) inputs.pack_dir = std::string(cli.last_value("--pack"));
    if (cli.has("--ple-gguf")) inputs.ple_gguf = std::string(cli.last_value("--ple-gguf"));
    if (cli.has("--mtp")) inputs.mtp_dir = std::string(cli.last_value("--mtp"));
    if (cli.has("--geometry-family")) inputs.family = std::string(cli.last_value("--geometry-family"));
    if (cli.has("--geometry-model")) inputs.model = std::string(cli.last_value("--geometry-model"));
    return WeightStore::inspect(inputs, geometry, error, window_sources, mtp_sources);
}

int storage_check(const Options& cli) {
    const auto supported = [](std::string_view name) {
        constexpr std::string_view kSupported[] = {
            "--storage-check", "--native", "--pack", "--ple-gguf", "--mtp", "--metallib",
            "--max-context", "--kv", "--vram-reserve-mib", "--resident-budget-gib",
            "--expert-cache", "--expert-cache-per-layer", "--expert-profile", "--prefill",
            "--spec", "--mtp-window", "--expert-slots", "--gpu-budget-mib",
        };
        for (const std::string_view candidate : kSupported) {
            if (name == candidate) return true;
        }
        const OptionSpec* spec = find_option(name);
        return spec != nullptr && spec->disposition == OptionDisposition::Ignore;
    };
    for (const std::string& name : cli.present) {
        if (!supported(name)) {
            std::fprintf(stderr, "strata: --storage-check cannot be combined with %s\n",
                         name.c_str());
            return 2;
        }
    }

    ContextOptions context_options;
    context_options.metallib_sha256 = metallib_sha256();
    if (cli.has("--metallib")) {
        context_options.metallib = cli.last_value("--metallib");
        context_options.metallib_sha256.clear();
        std::fprintf(stderr, "strata: --metallib: %s is not checked against this engine\n",
                     context_options.metallib.c_str());
    }
    if (const char* capture = std::getenv("STRATA_METAL_CAPTURE");
        capture != nullptr && *capture != '\0') {
        context_options.capture = capture;
    }
    if (!context_options.capture.empty()) {
        if (context_options.capture.size() < 9 ||
            context_options.capture.compare(context_options.capture.size() - 9, 9, ".gputrace") != 0) {
            context_options.capture += ".gputrace";
        }
        if (access(context_options.capture.c_str(), F_OK) == 0) {
            std::fprintf(stderr, "strata: %s exists; Metal does not overwrite a trace\n",
                         context_options.capture.c_str());
            return 2;
        }
    }

    std::string error;
    auto context = Context::open(context_options, error);
    if (!context) {
        std::fprintf(stderr, "strata: %s\n", error.c_str());
        return 1;
    }
    const PipelineStats& stats = context->pipeline_stats();
    std::fprintf(stderr, "strata: %u pipelines (%u from the cache, %u compiled) in %.1f ms; cache: %s\n",
                 stats.from_archive + stats.compiled, stats.from_archive, stats.compiled, stats.ms,
                 stats.archive.c_str());

    Geometry geometry;
    WindowSources sources;
    MtpRuntimeSources mtp_sources;
    if (!inspect_real_geometry(cli, geometry, error, &sources, &mtp_sources)) {
        std::fprintf(stderr, "strata: %s\n", error.c_str());
        return 1;
    }
    PlanResult memory_plan;
    const int plan_status = geometry_plan(cli, geometry, &memory_plan, context.get());
    if (plan_status != 0) return plan_status;

    auto storage = ModelStorage::open(*context, std::string(cli.last_value("--native")),
                                      geometry, sources, memory_plan, error,
                                      &mtp_sources);
    if (!storage) {
        std::fprintf(stderr, "strata: %s\n", error.c_str());
        return 1;
    }
    std::printf("STORAGE_CHECK ok mode=%s resident_experts=%llu windows=%zu warmed=%s "
                "main_kv=%s ple_table=%s window_scratch=%s "
                "inference=unavailable mtp_runtime=%s\n",
                storage->mode(),
                static_cast<unsigned long long>(storage->resident_experts()),
                storage->window_count(), storage->full_mode_warmed() ? "yes" : "no",
                storage->kv_store() != nullptr ? "yes" : "no",
                storage->ple_reader() != nullptr ? "yes" : "no",
                storage->window_scratch() != nullptr ? "yes" : "no",
                storage->mtp_runtime() != nullptr ? "yes" : "no");
    return 0;
}

bool geometry_output_is_safe(const Options& cli, const std::string& output_path,
                             std::string& error) {
    namespace fs = std::filesystem;
    const fs::path target(output_path);
    std::vector<fs::path> inputs;
    if (cli.has("--native")) {
        const fs::path native(cli.last_value("--native"));
        try {
            for (const std::string& shard : strata::gguf_split_paths(native.string())) {
                inputs.emplace_back(shard);
            }
        } catch (const std::exception&) {
            inputs.push_back(native);
        }
    }
    if (cli.has("--ple-gguf")) inputs.emplace_back(cli.last_value("--ple-gguf"));
    const fs::path pack = cli.has("--pack") ? fs::path(cli.last_value("--pack")) : fs::path("pack/full");
    for (const char* name : {"index.txt", "native_experts.txt", "conversions.json", "dense.bin"}) {
        inputs.push_back(pack / name);
    }
    if (cli.has("--mtp")) {
        const fs::path mtp(cli.last_value("--mtp"));
        for (const char* name : {"experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin"}) {
            inputs.push_back(mtp / name);
        }
    }

    std::error_code ec;
    const bool target_exists = fs::exists(target, ec);
    if (ec) {
        error = "cannot inspect geometry output " + target.string() + ": " + ec.message();
        return false;
    }
    if (!target_exists) return true;
    for (const fs::path& input : inputs) {
        ec.clear();
        const bool input_exists = fs::exists(input, ec);
        if (ec) {
            error = "cannot inspect input " + input.string() + ": " + ec.message();
            return false;
        }
        if (!input_exists) continue;
        ec.clear();
        if (fs::equivalent(target, input, ec)) {
            error = "--write-geometry output aliases input " + input.string();
            return false;
        }
        if (ec) {
            error = "cannot compare geometry output with input " + input.string() + ": " + ec.message();
            return false;
        }
    }
    return true;
}

bool write_geometry_atomic(const std::string& path, const Geometry& geometry, std::string& error) {
    namespace fs = std::filesystem;
    const fs::path target(path);
    std::ostringstream output;
    if (!write_geometry(output, geometry, error)) {
        return false;
    }
    const std::string contents = output.str();
    std::string temporary_template = target.string() + ".tmp.XXXXXX";
    std::vector<char> temporary_name(temporary_template.begin(), temporary_template.end());
    temporary_name.push_back('\0');
    const int descriptor = ::mkstemp(temporary_name.data());
    if (descriptor < 0) {
        error = "cannot create temporary geometry beside " + target.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    const fs::path temporary(temporary_name.data());
    std::size_t written = 0;
    while (written < contents.size()) {
        const ssize_t count = ::write(descriptor, contents.data() + written, contents.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            error = "failed writing " + temporary.string() + ": " +
                    std::error_code(errno, std::generic_category()).message();
            break;
        }
        written += static_cast<std::size_t>(count);
    }
    if (written == contents.size() && ::fsync(descriptor) != 0) {
        error = "failed syncing " + temporary.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
    }
    if (::close(descriptor) != 0 && error.empty()) {
        error = "failed closing " + temporary.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
    }
    if (!error.empty()) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        return false;
    }
    std::error_code ec;
    fs::rename(temporary, target, ec);
    if (ec) {
        error = "cannot replace " + target.string() + ": " + ec.message();
        std::error_code ignored;
        fs::remove(temporary, ignored);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> arguments;
    arguments.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);

    OptionParseResult parsed = parse_options(arguments);
    if (parsed.exit_code != 0) {
        std::fprintf(stderr, "%s\n", parsed.error.c_str());
        return parsed.exit_code;
    }
    for (const std::string& warning : parsed.options.warnings) {
        std::fprintf(stderr, "%s\n", warning.c_str());
    }
    const Options& options = parsed.options;

    if (options.has("--help")) {
        print_usage(stdout);
        return 0;
    }
    if (options.has("--version")) {
        std::printf("strata %s metal %d\n", engine_version(), metal_rev());
        return 0;
    }
    const bool list = options.has("--list-devices");
    const bool self_test = options.has("--self-test");
    if (list && self_test) {
        std::fprintf(stderr, "strata: --list-devices cannot be combined with --self-test\n");
        return 2;
    }
    if (list && options.has("--metal-capture")) {
        std::fprintf(stderr, "strata: --metal-capture cannot be combined with --list-devices\n");
        return 2;
    }
    if (options.has("--plan-only") && options.has("--geometry")) {
        if (list || self_test) {
            std::fprintf(stderr, "strata: --plan-only --geometry cannot be combined with --list-devices or --self-test\n");
            return 2;
        }
        Geometry geometry;
        std::string error;
        const std::string geometry_path(options.last_value("--geometry"));
        if (!read_geometry_file(geometry_path, geometry, error)) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return error.starts_with("unsupported geometry file version ") ? 2 : 1;
        }
        return geometry_plan(options, std::move(geometry));
    }
    if (options.has("--write-geometry")) {
        if (list || self_test || options.has("--metallib") || options.has("--metal-capture")) {
            std::fprintf(stderr, "strata: --write-geometry cannot be combined with device options\n");
            return 2;
        }
        Geometry geometry;
        std::string error;
        const std::string output_path(options.last_value("--write-geometry"));
        if (output_path.empty() || !geometry_output_is_safe(options, output_path, error)) {
            if (error.empty()) error = "--write-geometry needs a non-empty output path";
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 2;
        }
        if (!inspect_real_geometry(options, geometry, error)) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 1;
        }
        if (!write_geometry_atomic(output_path, geometry, error)) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }
    if (options.has("--plan-only")) {
        if (list || self_test) {
            std::fprintf(stderr, "strata: --plan-only cannot be combined with --list-devices or --self-test\n");
            return 2;
        }
        Geometry geometry;
        std::string error;
        if (!inspect_real_geometry(options, geometry, error)) {
            std::fprintf(stderr, "strata: %s\n", error.c_str());
            return 1;
        }
        return geometry_plan(options, std::move(geometry));
    }
    if (options.has("--storage-check")) {
        return storage_check(options);
    }
    for (const std::string& name : options.present) {
        const OptionSpec* spec = find_option(name);
        const bool driver_option =
            name == "--metallib" || name == "--metal-capture" || name == "--list-devices" ||
            name == "--self-test";
        if (driver_option || (spec != nullptr && spec->disposition == OptionDisposition::Ignore)) continue;
        std::fprintf(stderr, "strata: planning and inference options are not wired into the Metal engine yet\n");
        return 2;
    }
    if (!list && !self_test) {
        std::fprintf(stderr, "strata: choose --self-test until model planning and inference are available\n");
        print_usage(stderr);
        return 2;
    }

    ContextOptions context_options;
    context_options.metallib_sha256 = metallib_sha256();
    if (options.has("--metallib")) {
        context_options.metallib = options.last_value("--metallib");
        context_options.metallib_sha256.clear();
        std::fprintf(stderr, "strata: --metallib: %s is not checked against this engine\n",
                     context_options.metallib.c_str());
    }
    if (!list) {
        if (const char* capture = std::getenv("STRATA_METAL_CAPTURE");
            capture != nullptr && *capture != '\0') {
            context_options.capture = capture;
        }
        if (options.has("--metal-capture")) {
            context_options.capture = options.last_value("--metal-capture");
        }
    }
    if (!list && !context_options.capture.empty()) {
        if (context_options.capture.size() < 9 ||
            context_options.capture.compare(context_options.capture.size() - 9, 9, ".gputrace") != 0) {
            context_options.capture += ".gputrace";
        }
        if (access(context_options.capture.c_str(), F_OK) == 0) {
            std::fprintf(stderr, "strata: %s exists; Metal does not overwrite a trace\n",
                         context_options.capture.c_str());
            return 2;
        }
    }

    context_options.pipelines = !list;
    std::string err;
    auto context = Context::open(context_options, err);
    if (!context) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    if (list) {
        const DeviceFacts facts = context->device_facts();
        std::printf("device_name=%s\n", context->device_name().c_str());
        if (facts.gpu_core_count != 0) {
            std::printf("gpu_core_count=%llu\n", static_cast<unsigned long long>(facts.gpu_core_count));
        } else {
            std::printf("gpu_core_count=unknown\n");
        }
        std::printf("working_set_bytes=%llu\n", static_cast<unsigned long long>(facts.working_set_bytes));
        std::printf("max_buffer_length=%llu\n", static_cast<unsigned long long>(facts.max_buffer_length));
        std::printf("max_threadgroup_memory_length=%llu\n",
                    static_cast<unsigned long long>(facts.max_threadgroup_memory_length));
        std::printf("families=");
        for (std::size_t i = 0; i < facts.families.size(); ++i) {
            std::printf("%s%s", i == 0 ? "" : ",", facts.families[i].c_str());
        }
        std::printf("\nphysical_memory_bytes=%llu\n",
                    static_cast<unsigned long long>(facts.physical_memory_bytes));
        std::printf("macos_version=%s\n", facts.macos_version.empty() ? "unknown" : facts.macos_version.c_str());
        return 0;
    }

    const PipelineStats& stats = context->pipeline_stats();
    std::fprintf(stderr, "strata: %u pipelines (%u from the cache, %u compiled) in %.1f ms; cache: %s\n",
                 stats.from_archive + stats.compiled, stats.from_archive, stats.compiled, stats.ms,
                 stats.archive.c_str());
    if (!context->begin_capture(err)) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    const bool ok = context->run_selftest(err);  // Stand-in until H6/H7 connect the request loops.
    context->end_capture();
    if (!ok) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
