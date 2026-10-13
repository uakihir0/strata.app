// include/strata/metal/serve.hpp - host-only line protocol for the Metal engine.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {

inline constexpr std::int32_t kServeVocabularySize = 248'320;
inline constexpr std::int32_t kServeImagePadToken = 248'056;
inline constexpr std::int32_t kServeImageEmbeddingWidth = 2'560;

struct ServeImageRecord {
    std::int32_t n_tokens = 0;
    std::int32_t nx = 0;
    std::int32_t ny = 0;
    std::vector<float> embeddings;
};

struct SamplingOptions {
    std::int32_t cvec = 1;
    float temperature = 0.0f;
    float top_p = 1.0f;
    std::int32_t top_k = 20;
    float min_p = 0.0f;
    std::int32_t penalty_last_n = 0;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    std::uint64_t seed = 0;
    double pcie_frac = 0.0;
    double spec_min_p = 0.0;
};

struct ServeRequest {
    bool image = false;
    std::int64_t max_new = 0;
    std::string embeddings_path;
    std::vector<std::int32_t> token_ids;
    std::vector<ServeImageRecord> images;
    std::vector<std::array<std::int32_t, 3>> mrope_positions;
    SamplingOptions sampling;
    bool pcie_frac_is_set = false;
    bool spec_min_p_is_set = false;
};

enum class ServeParseKind : std::uint8_t { Request, Quit, InvalidCommand, BadRequest };

struct ServeParseResult {
    ServeParseKind kind = ServeParseKind::InvalidCommand;
    ServeRequest request;
    std::string error;
};

/// Parse one GEN or GENI line. STOP is consumed by the asynchronous input reader.
ServeParseResult parse_serve_request(std::string_view line);
/// Load SVE1 records, match image-pad rows and build text/image M-RoPE positions.
bool load_serve_image_embeddings(ServeRequest& request, std::int64_t max_context, std::string& error);

struct ServeInfo {
    std::int64_t context = 0;
    std::string kv = "fp16";
    std::int64_t expert_slots = 0;
    std::int64_t expert_cache_mib = 0;
    std::int32_t spec = 0;
    std::int32_t mtp_max = 0;
    std::int32_t lookup = 3;
    std::int64_t vram_free_mib = 0;
    std::string cvec = "none";
    std::int64_t arena_mib = 0;
    double spec_min_p = 0.0;
    std::int64_t conversation_cache_mib = 0;
    std::int32_t conversation_cache_slots = 0;
    std::int64_t conversation_cache_min_free_mib = 0;
    std::string engine;
    std::int32_t metal_rev = 1;
    bool stop_eos = false;
    std::vector<std::int32_t> eos_ids {248'044, 248'046};
    std::string gpu;
    std::int64_t gpu_budget_mib = 0;
    std::string residency = "full";
    std::int64_t ssd_experts = 0;
    std::int64_t plan_mib = 0;
    std::int64_t vram_reserve_mib = 700;
};

/// Serialize INFO in upstream key order, followed by the additive Metal keys.
std::string serve_info_line(const ServeInfo& info);

struct ServeOutcome {
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    std::string finish = "length";  // stop, cancel or length
    std::int64_t drafts_accepted = 0;
    std::int64_t drafts_offered = 0;
    std::int64_t checkpoints = 0;
    std::int64_t hits = 0;
    std::int64_t lookups = 0;
    std::int64_t ram_blobs = 0;
    std::int64_t file_blobs = 0;
    double file_mb = 0.0;
    bool report_resident_ram = false;
    double resident_ram_gib = 0.0;
    std::int64_t exchanges = 0;
    std::int64_t cumulative_file_reads = 0;
    bool report_expert_tiers = false;
    std::int64_t cumulative_ram_blobs = 0;
    std::int64_t cumulative_expert_file_blobs = 0;
    double cumulative_expert_file_mb = 0.0;
};

enum class ServeFailureKind : std::uint8_t { Request, Gpu, Fatal };

struct ServeFailure {
    ServeFailureKind kind = ServeFailureKind::Request;
    std::string label;
    std::string message;
};

struct ServeCallbacks {
    std::function<bool(std::int64_t)> resume;
    std::function<bool(std::int64_t, std::int64_t, double, double)> prompt_progress;
    std::function<bool(std::int64_t)> prompt_reused;
    std::function<bool(std::int32_t)> token;
    std::function<bool()> should_stop;
};

/// A real engine and the transcript fake share this boundary. Implementations do
/// prompt restoration/prefill and invoke callbacks in RESUME, PP*, REUSED, T*
/// order while they execute their windows.
class ServeBackend {
public:
    virtual ~ServeBackend() = default;
    virtual const ServeInfo& serve_info() const noexcept = 0;
    virtual bool run_request(const ServeRequest& request, const ServeCallbacks& callbacks,
                             ServeOutcome& outcome, ServeFailure& failure) = 0;
};

struct ServeLoopOptions {
    bool vision_enabled = false;
    int input_fd = 0;
    std::FILE* output = nullptr;
    std::FILE* diagnostics = nullptr;
    /// Replaces the watchdog's std::abort (docs/mac/06, section 14.2). Unset, the watchdog aborts. Tests use it to end a
    /// stalled child process without a crash report.
    std::function<void()> watchdog_abort;
};

/// Run the protocol until QUIT or stdin EOF. Returns 0 on a clean shutdown,
/// 1 after a fatal engine failure, and 2 when the protocol stream cannot start.
int run_serve_loop(ServeBackend& backend, const ServeLoopOptions& options = {});

}  // namespace strata::metal
