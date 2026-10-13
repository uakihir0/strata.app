// tests/metal/metal_window_offset_test.cpp - full-mode file offsets, address and resident tables
// checked against a sparse GGUF (05 sections 4.1, 4.2 and 5.3). Device-free: the GPU window base
// addresses are fixed numbers, and the fixture is sparse, so only the sampled slabs hold bytes.
#include "strata/metal/window.hpp"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using strata::metal::FileWindow;
using strata::metal::FullModeLayout;
using strata::metal::WindowFileSource;
using strata::metal::WindowPlan;
using strata::metal::WindowSources;
using strata::metal::WindowTensorSource;
using strata::metal::kFullModeExpertRoles;
using strata::metal::kFullModeMaxExperts;
using strata::metal::kFullModeMaxLayers;
using strata::metal::kWindowPageBytes;

namespace {

constexpr std::uint64_t kSlab = kWindowPageBytes;
constexpr std::uint64_t kMiB = 1024ull * 1024;
constexpr std::uint64_t kMaxBuffer = 64 * kMiB;
constexpr std::uint64_t kSlabsPerWindow = kMaxBuffer / kSlab;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class TempDir {
public:
    TempDir() {
        std::string pattern = (fs::temp_directory_path() / "strata-window-offset.XXXXXX").string();
        require(::mkdtemp(pattern.data()) != nullptr, "create temporary fixture directory");
        path_ = pattern;
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

// A sparse file. Only the markers written through it occupy blocks.
class SparseFile {
public:
    SparseFile(const fs::path& path, std::uint64_t size) {
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        require(fd_ >= 0, "create sparse fixture " + path.string());
        require(::ftruncate(fd_, static_cast<off_t>(size)) == 0, "size sparse fixture");
    }
    ~SparseFile() { ::close(fd_); }
    SparseFile(const SparseFile&) = delete;
    SparseFile& operator=(const SparseFile&) = delete;

    void write_marker(std::uint64_t offset, std::uint32_t marker) {
        const std::uint8_t bytes[4] = {
            static_cast<std::uint8_t>(marker), static_cast<std::uint8_t>(marker >> 8),
            static_cast<std::uint8_t>(marker >> 16), static_cast<std::uint8_t>(marker >> 24)};
        require(::pwrite(fd_, bytes, sizeof bytes, static_cast<off_t>(offset)) ==
                    static_cast<ssize_t>(sizeof bytes),
                "write sparse marker");
    }

    std::uint32_t read_marker(std::uint64_t offset) const {
        std::uint8_t bytes[4] = {};
        require(::pread(fd_, bytes, sizeof bytes, static_cast<off_t>(offset)) ==
                    static_cast<ssize_t>(sizeof bytes),
                "read sparse marker");
        return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) |
               (static_cast<std::uint32_t>(bytes[3]) << 24);
    }

private:
    int fd_ = -1;
};

// 48 layers, 512 experts, three roles, one page per slab: the full-mode table shape. The file is
// 1,207,959,552 B, and 64 MiB windows split it on slab boundaries into 18 windows of 4096 slabs.
void test_full_shape_offsets_and_tables() {
    constexpr std::uint64_t tensor_bytes = static_cast<std::uint64_t>(kFullModeMaxExperts) * kSlab;
    constexpr std::uint64_t slabs = static_cast<std::uint64_t>(kFullModeMaxLayers) *
                                    kFullModeExpertRoles * kFullModeMaxExperts;
    constexpr std::uint64_t file_size = slabs * kSlab;
    static_assert(file_size == 1'207'959'552ull, "full-shape fixture must be 1,207,959,552 B");

    const char* roles[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    TempDir dir;
    const std::string path = (dir.path() / "full-shape.gguf").string();
    WindowFileSource file{path, file_size, {}, {}, 0, 0};
    for (std::uint32_t layer = 0; layer < kFullModeMaxLayers; ++layer) {
        for (std::uint32_t role = 0; role < kFullModeExpertRoles; ++role) {
            const std::uint64_t offset =
                (static_cast<std::uint64_t>(layer) * kFullModeExpertRoles + role) * tensor_bytes;
            file.tensors.push_back({"blk." + std::to_string(layer) + "." + roles[role], offset,
                                    tensor_bytes, kSlab});
        }
    }
    WindowSources sources{};
    sources.files.push_back(std::move(file));
    SparseFile sparse(path, file_size);

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, kMaxBuffer, plan, error),
            "plan full-shape fixture: " + error);
    const std::vector<FileWindow>& windows = plan.files[0].windows;
    require(windows.size() == 18, "1.125 GiB in 64 MiB windows must be 18 windows");
    for (std::size_t w = 0; w < windows.size(); ++w) {
        require(windows[w].start == w * kMaxBuffer && windows[w].end - windows[w].start == kMaxBuffer,
                "every full-shape window must be exactly 64 MiB, starting on a slab boundary");
    }
    require(strata::metal::verify_window_never_wrap(sources, plan, error),
            "full-shape plan must satisfy the never-wrap rules: " + error);

    std::vector<std::uint64_t> bases(windows.size());
    for (std::size_t w = 0; w < bases.size(); ++w) {
        bases[w] = 0x100000000ull + static_cast<std::uint64_t>(w) * 0x10000000ull;
    }
    const std::vector<std::vector<std::uint64_t>> gpu_bases{bases};
    FullModeLayout layout;
    require(strata::metal::build_full_mode_layout(sources, plan, gpu_bases, kMaxBuffer,
                                                  kFullModeMaxLayers, kFullModeMaxExperts, layout,
                                                  error),
            "build full-shape layout: " + error);

    // 05 section 4.1 and 5.3: the expert address table holds 48 x 512 x 3 uint64 and the resident
    // table holds 48 x 512 int32. Both are fixed-capacity buffers.
    require(layout.expert_addresses.size() * sizeof(std::uint64_t) == 589'824 &&
                layout.resident.size() * sizeof(std::int32_t) == 98'304,
            "address table must be 590 KB and resident table 96 KiB at full capacity");

    // Hand-computed from entry (l * 512 + e) * 3 + role, role gate = 0, up = 1, down = 2.
    require(layout.expert_addresses[0] == bases[0] &&
                layout.expert_addresses[1] == bases[0] + 8 * kMiB &&
                layout.expert_addresses[2] == bases[0] + 16 * kMiB &&
                layout.expert_addresses[3] == bases[0] + kSlab &&
                layout.expert_addresses[1536] == bases[0] + 24 * kMiB,
            "expert address entries near the table start must follow (l * 512 + e) * 3 + role");
    // Slab 73727 is (layer 47, expert 511, down): file offset 73727 * 16 KiB, in window 17.
    require(layout.expert_addresses[73727] == bases[17] + 67'092'480ull,
            "the last table entry must be window 17's base plus its 67,092,480 B offset");
    require(layout.resident[24575] == 1 && layout.resident[0] == 1,
            "the resident table must mark the last and first active entries");

    const auto sampled = [](std::uint64_t slab) {
        return slab % kSlabsPerWindow == 0 || slab % kSlabsPerWindow == kSlabsPerWindow - 1 ||
               slab % 9973 == 0;
    };
    for (std::uint64_t slab = 0; slab < slabs; ++slab) {
        if (sampled(slab)) sparse.write_marker(slab * kSlab, static_cast<std::uint32_t>(slab));
    }

    const std::uint64_t slabs_per_role = kFullModeMaxExperts;
    for (std::uint64_t slab = 0; slab < slabs; ++slab) {
        const std::uint64_t layer = slab / (kFullModeExpertRoles * slabs_per_role);
        const std::uint64_t role = (slab / slabs_per_role) % kFullModeExpertRoles;
        const std::uint64_t expert = slab % slabs_per_role;
        const std::uint64_t entry = layer * kFullModeMaxExperts + expert;
        const std::uint64_t table = entry * kFullModeExpertRoles + role;
        const std::uint64_t offset = slab * kSlab;

        std::size_t window = windows.size();
        for (std::size_t w = 0; w < windows.size(); ++w) {
            if (offset >= windows[w].start && offset < windows[w].end) window = w;
        }
        require(window < windows.size(), "every slab must lie inside a window");
        require(layout.expert_addresses[table] == bases[window] + (offset - windows[window].start),
                "each slab's table entry must be its window's base plus the slab's window offset");
        require(layout.resident[entry] == 1, "every active resident entry must be 1 in full mode");
        if (sampled(slab)) {
            // The address the GPU would use, turned back into a file position, must hold the slab.
            const std::uint64_t file_position =
                windows[window].start + (layout.expert_addresses[table] - bases[window]);
            require(sparse.read_marker(file_position) == slab,
                    "a window offset must reach the bytes of its slab in the file");
        }
    }

    // 05 section 3.3 and 4.3: experts warm in profile order, then all the rest in layer order.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ordered;
    require(strata::metal::order_full_mode_warm_experts(kFullModeMaxLayers, kFullModeMaxExperts,
                                                        {{47, 511}, {0, 0}}, ordered, error),
            "order full-shape warm-up: " + error);
    require(ordered.size() == 24'576 && ordered[0] == std::make_pair(47u, 511u) &&
                ordered[1] == std::make_pair(0u, 0u) && ordered[2] == std::make_pair(0u, 1u) &&
                ordered.back() == std::make_pair(47u, 510u),
            "warm-up order must start with the profile and end with the last unranked expert");
}

// A file whose size is not a page multiple. The last window ends at the mapped page end, and the
// dense tensor at the file end reads through its window offset. A window one page further wraps
// past the mapped file and is rejected.
void test_mapped_page_end_offsets() {
    const std::uint64_t dense_offset = 3 * kSlab;
    const std::uint64_t dense_bytes = 100;
    const std::uint64_t file_size = dense_offset + dense_bytes;
    TempDir dir;
    const std::string path = (dir.path() / "eof.gguf").string();
    WindowFileSource file{path,
                          file_size,
                          {{"blk.0.ffn_gate_exps.weight", 0, kSlab, kSlab},
                           {"blk.0.ffn_up_exps.weight", kSlab, kSlab, kSlab},
                           {"blk.0.ffn_down_exps.weight", 2 * kSlab, kSlab, kSlab},
                           {"token_embd.weight", dense_offset, dense_bytes, 0}},
                          {},
                          0,
                          0};
    WindowSources sources{};
    sources.files.push_back(std::move(file));
    SparseFile sparse(path, file_size);
    sparse.write_marker(dense_offset, 0x5eed);

    WindowPlan plan;
    std::string error;
    require(strata::metal::plan_windows(sources, kMaxBuffer, plan, error),
            "plan page-end fixture: " + error);
    require(plan.files[0].windows.size() == 1 && plan.files[0].windows[0].end == 4 * kSlab,
            "the last window must end at the mapped page end, 64 KiB");
    require(strata::metal::verify_window_never_wrap(sources, plan, error),
            "page-end plan must satisfy the never-wrap rules: " + error);

    const std::uint64_t base = 0x200000000ull;
    const std::vector<std::vector<std::uint64_t>> gpu_bases{{base}};
    FullModeLayout layout;
    require(strata::metal::build_full_mode_layout(sources, plan, gpu_bases, kMaxBuffer, 1, 1,
                                                  layout, error),
            "build page-end layout: " + error);
    const auto dense = [&]() {
        for (const auto& tensor : layout.tensors) {
            if (tensor.tensor_name == "token_embd.weight") return tensor;
        }
        throw std::runtime_error("dense tensor binding is missing");
    }();
    require(dense.gpu_address == base + dense_offset && dense.byte_count == dense_bytes,
            "the dense tensor must bind at its window offset, 48 KiB past the base");
    require(layout.expert_addresses[2] == base + 2 * kSlab,
            "the single down slab must bind at 32 KiB past the base");
    const std::uint64_t file_position = plan.files[0].windows[0].start + (dense.gpu_address - base);
    require(sparse.read_marker(file_position) == 0x5eed,
            "the dense tensor's window offset must reach its bytes in the sparse file");

    FileWindow past_end = plan.files[0].windows[0];
    past_end.end = 8 * kSlab;
    WindowPlan wrapped = plan;
    wrapped.files[0].windows[0] = past_end;
    require(!strata::metal::build_full_mode_layout(sources, wrapped, gpu_bases, kMaxBuffer, 1, 1,
                                                   layout, error) &&
                error.find("oversized GPU window") != std::string::npos,
            "a window past the mapped page end must be rejected by the layout: " + error);
}

}  // namespace

int main() {
    try {
        test_full_shape_offsets_and_tables();
        test_mapped_page_end_offsets();
        std::cout << "metal_window_offset_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_window_offset_test: " << exception.what() << '\n';
        return 1;
    }
}
