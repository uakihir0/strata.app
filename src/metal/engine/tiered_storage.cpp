// src/metal/engine/tiered_storage.cpp - tiered arenas, bulk/miss descriptors and startup copies.
#include "strata/metal/tiered.hpp"
#include "window_source_lock.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <pthread/qos.h>
#include <set>
#include <string_view>
#include <system_error>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kExpertRoles = 3;
constexpr std::uint64_t kDenseArenaAlignment = 16 * 1024;
constexpr std::uint64_t kSlotAlignment = 16 * 1024;

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0 ? (result = value, true)
                          : add(value, alignment - remainder, result);
}

std::string system_error(const char* operation, const std::string& path, int code) {
    return std::string(operation) + " " + path + ": " +
           std::error_code(code, std::generic_category()).message();
}

class FileDescriptor {
public:
    FileDescriptor() = default;
    explicit FileDescriptor(int value) : value_(value) {}
    ~FileDescriptor() { reset(); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }

    int get() const noexcept { return value_; }

private:
    void reset() noexcept {
        if (value_ >= 0) {
            const int saved_errno = errno;
            (void)::close(value_);
            errno = saved_errno;
            value_ = -1;
        }
    }

    int value_ = -1;
};

struct ShardDescriptors {
    FileDescriptor bulk;
    FileDescriptor miss;
};

bool open_read_descriptor(const std::string& path, FileDescriptor& descriptor,
                          std::string& error) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = system_error("cannot open", path, errno);
        return false;
    }
    descriptor = FileDescriptor(fd);
    return true;
}

bool lock_source_file(int descriptor, const std::string& path, std::string& error) {
    if (::flock(descriptor, LOCK_SH | LOCK_NB) == 0) return true;
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
        error = path + ": source is locked for writing";
        return false;
    }
    error = system_error("cannot lock source file", path, errno);
    return false;
}

bool verify_open_file(int descriptor, const std::string& path, std::uint64_t expected_size,
                      std::uint64_t expected_device, std::uint64_t expected_object,
                      std::string& error) {
    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        error = system_error("cannot stat", path, errno);
        return false;
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != expected_size) {
        error = path + ": file size changed after WeightStore inspection";
        return false;
    }
    if (static_cast<std::uint64_t>(status.st_dev) != expected_device ||
        static_cast<std::uint64_t>(status.st_ino) != expected_object) {
        error = path + ": file identity changed after WeightStore inspection";
        return false;
    }
    return true;
}

bool artifact_set_lock_path(const std::filesystem::path& directory,
                            std::filesystem::path& result,
                            std::string& error) {
    std::error_code ec;
    const std::filesystem::path resolved =
        std::filesystem::weakly_canonical(directory, ec);
    if (ec || resolved.filename().empty()) {
        error = "cannot resolve artifact-set directory " + directory.string() +
                (ec ? ": " + ec.message() : "");
        return false;
    }
    result = resolved.parent_path() /
             ("." + resolved.filename().string() + ".strata-artifact-set.lock");
    return true;
}

bool set_bulk_advice(int descriptor, const std::string& path, std::string& error) {
#if defined(F_NOCACHE)
    if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
        error = system_error("cannot set F_NOCACHE on", path, errno);
        return false;
    }
    return true;
#else
    (void)descriptor;
    error = path + ": this platform does not provide F_NOCACHE";
    return false;
#endif
}

bool set_miss_advice(int descriptor, const std::string& path, std::string& error) {
#if defined(F_RDAHEAD)
    if (::fcntl(descriptor, F_RDAHEAD, 0) != 0) {
        error = system_error("cannot disable F_RDAHEAD on", path, errno);
        return false;
    }
    return true;
#else
    (void)descriptor;
    error = path + ": this platform does not provide F_RDAHEAD";
    return false;
#endif
}

bool has_inspection_lock(const WindowSources& sources, const std::string& path,
                         std::uint64_t file_size, std::uint64_t device_id,
                         std::uint64_t object_id) {
    return std::any_of(
        sources.inspected_file_locks.begin(), sources.inspected_file_locks.end(),
        [&](const WindowSourceFileLock& lock) {
            return lock.path == path && lock.lock_lease() != nullptr &&
                   lock.file_size == file_size && lock.file_device_id == device_id &&
                   lock.file_object_id == object_id;
        });
}

const WindowSourceFileLock* find_inspection_lock(const WindowSources& sources,
                                                  const std::string& path) {
    const auto found = std::find_if(
        sources.inspected_file_locks.begin(), sources.inspected_file_locks.end(),
        [&](const WindowSourceFileLock& lock) {
            return lock.path == path && lock.lock_lease() != nullptr;
        });
    return found == sources.inspected_file_locks.end() ? nullptr : &*found;
}

bool is_router_name(std::string_view name) {
    return name.ends_with(".ffn_gate_inp.weight") ||
           name.ends_with(".ffn_gate_inp_shexp.weight");
}

bool valid_converted_router(const strata::TensorInfo& tensor, const Geometry& geometry) {
    if (std::string_view(tensor.type_name()) != "F32") return false;
    if (!tensor.name.starts_with("blk.")) return false;
    const std::size_t layer_end = tensor.name.find('.', 4);
    if (layer_end == std::string::npos || layer_end == 4) return false;
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(
        tensor.name.data() + 4, tensor.name.data() + layer_end, layer);
    if (parsed.ec != std::errc{} || parsed.ptr != tensor.name.data() + layer_end ||
        layer >= geometry.n_layers ||
        tensor.name.compare(4, layer_end - 4, std::to_string(layer)) != 0) {
        return false;
    }
    if (tensor.name.ends_with(".ffn_gate_inp.weight")) {
        return tensor.shape.size() == 2 && tensor.shape[0] == 2'560 &&
               tensor.shape[1] == geometry.n_expert;
    }
    if (tensor.name.ends_with(".ffn_gate_inp_shexp.weight")) {
        return tensor.shape.size() == 1 && tensor.shape[0] == 2'560;
    }
    return false;
}

bool validate_model_sources(const strata::GgufModel& model, const Geometry& geometry,
                            const WindowSources& sources, std::string& error) {
    if (model.size() != sources.files.size() || model.size() != geometry.shards.size()) {
        error = "tiered GGUF shard count does not match the inspected source inventory";
        return false;
    }
    std::map<std::string, const DensePackTensorSource*> packed;
    for (const DensePackTensorSource& tensor : sources.dense_pack_tensors) {
        if (!packed.emplace(tensor.tensor_name, &tensor).second) {
            error = "duplicate converted dense-pack tensor " + tensor.tensor_name;
            return false;
        }
    }

    std::set<std::string> all_names;
    std::uint64_t converted_bytes = 0;
    std::uint64_t nonexpert_bytes = 0;
    std::size_t ple_count = 0;
    for (std::size_t shard = 0; shard < model.size(); ++shard) {
        const strata::GgufFile& file = model.shard(shard);
        const WindowFileSource& source = sources.files[shard];
        const ShardGeometry& planned = geometry.shards[shard];
        if (file.path() != source.path || file.file_size() != source.file_size ||
            file.file_device_id() != source.file_device_id ||
            file.file_object_id() != source.file_object_id ||
            file.mapping_data() == nullptr ||
            std::filesystem::path(source.path).filename().string() != planned.name ||
            source.file_size != planned.bytes) {
            error = source.path +
                    ": tiered GGUF shard differs from the memory plan or inspected file identity";
            return false;
        }

        std::map<std::string, const WindowTensorSource*> included;
        std::map<std::string, const WindowTensorExclusion*> excluded;
        for (const WindowTensorSource& tensor : source.tensors) {
            if (!included.emplace(tensor.tensor_name, &tensor).second) {
                error = source.path + ": duplicate tiered included tensor " + tensor.tensor_name;
                return false;
            }
        }
        for (const WindowTensorExclusion& tensor : source.excluded_tensors) {
            if (!excluded.emplace(tensor.tensor_name, &tensor).second) {
                error = source.path + ": duplicate tiered excluded tensor " + tensor.tensor_name;
                return false;
            }
        }

        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        ranges.reserve(file.tensors().size());
        for (const strata::TensorInfo& tensor : file.tensors()) {
            if (!all_names.insert(tensor.name).second) {
                error = "tiered GGUF contains a duplicate tensor name " + tensor.name;
                return false;
            }
            const auto in = included.find(tensor.name);
            const auto out = excluded.find(tensor.name);
            if ((in == included.end()) == (out == excluded.end())) {
                error = source.path + ": tensor must be included or explicitly excluded once: " +
                        tensor.name;
                return false;
            }
            const std::uint64_t bytes = strata::tensor_payload_bytes(tensor);
            std::uint64_t offset = 0;
            std::uint64_t end = 0;
            if (bytes == 0 || !add(file.data_start(), tensor.offset, offset) ||
                !add(offset, bytes, end) || end > file.file_size() ||
                (in != included.end() &&
                 (in->second->file_offset != offset || in->second->byte_count != bytes)) ||
                (out != excluded.end() &&
                 (out->second->file_offset != offset || out->second->byte_count != bytes))) {
                error = source.path + ": inspected tensor range changed for " + tensor.name;
                return false;
            }
            ranges.emplace_back(offset, end);

            const bool expert = tensor.name.find(".ffn_") != std::string::npos &&
                                tensor.name.ends_with("_exps.weight");
            if (expert) {
                if (in == included.end() || out != excluded.end() ||
                    in->second->expert_slab_bytes == 0 || tensor.shape.size() != 3 ||
                    tensor.shape[2] != geometry.n_expert ||
                    bytes % geometry.n_expert != 0 ||
                    in->second->expert_slab_bytes != bytes / geometry.n_expert) {
                    error = source.path + ": invalid expert tensor source " + tensor.name;
                    return false;
                }
            } else if (in != included.end()) {
                if (in->second->expert_slab_bytes != 0 ||
                    !add(nonexpert_bytes, bytes, nonexpert_bytes)) {
                    error = source.path + ": malformed non-expert source " + tensor.name;
                    return false;
                }
            }

            if (out != excluded.end()) {
                if (tensor.name == "per_layer_token_embd.weight") {
                    if (out->second->reason != WindowTensorExclusionReason::ple) {
                        error = source.path + ": PLE tensor has the wrong exclusion reason";
                        return false;
                    }
                    ++ple_count;
                } else {
                    const auto pack = packed.find(tensor.name);
                    if (out->second->reason != WindowTensorExclusionReason::converted ||
                        !valid_converted_router(tensor, geometry) || pack == packed.end() ||
                        pack->second->kind != 4 ||
                        pack->second->source_byte_count != pack->second->byte_count ||
                        pack->second->byte_count != bytes / 2 ||
                        bytes % 2 != 0 ||
                        !add(converted_bytes, pack->second->byte_count, converted_bytes)) {
                        error = source.path + ": invalid converted router source " + tensor.name;
                        return false;
                    }
                    packed.erase(pack);
                }
            } else if (is_router_name(tensor.name) &&
                       std::string_view(tensor.type_name()) == "F32") {
                error = source.path + ": an F32 router must use its converted dense-pack row";
                return false;
            }
        }
        if (included.size() + excluded.size() != file.tensors().size()) {
            error = source.path + ": tiered source inventory contains an unknown tensor";
            return false;
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                error = source.path + ": GGUF tensor payload ranges overlap";
                return false;
            }
        }
    }

    std::size_t ple_shard = 0;
    if (model.find("per_layer_token_embd.weight", &ple_shard) == nullptr ||
        ple_count != 1 || !packed.empty() ||
        nonexpert_bytes != geometry.nonexpert_bytes ||
        converted_bytes != geometry.dense_pack_bytes) {
        error = "tiered PLE, dense or converted-router inventory does not match geometry";
        return false;
    }
    const WindowSourceFileLock* dense_lock =
        find_inspection_lock(sources, sources.dense_pack_path);
    if (dense_lock == nullptr ||
        !dense_lock->lock_lease()->matches_dense_pack_catalog(
            sources.dense_pack_tensors)) {
        error = sources.dense_pack_path +
                ": converted row inventory differs from the WeightStore index inspection";
        return false;
    }
    return true;
}

bool open_descriptors(const WindowSources& sources, std::vector<ShardDescriptors>& shards,
                      std::vector<FileDescriptor>& artifact_set_locks,
                      FileDescriptor& dense_pack, std::string& error) {
    shards.clear();
    shards.reserve(sources.files.size());
    artifact_set_locks.clear();
    if (sources.files.empty() || sources.dense_pack_path.empty()) {
        error = "tiered artifact-set source paths are missing";
        return false;
    }
    const std::filesystem::path source_path(sources.files.front().path);
    const std::filesystem::path dense_path(sources.dense_pack_path);
    const std::filesystem::path source_directory = source_path.has_parent_path()
        ? source_path.parent_path()
        : std::filesystem::current_path();
    std::filesystem::path source_lock_path;
    std::filesystem::path pack_lock_path;
    if (!artifact_set_lock_path(source_directory, source_lock_path, error) ||
        !artifact_set_lock_path(dense_path.parent_path(), pack_lock_path, error)) {
        return false;
    }
    std::vector<std::filesystem::path> lock_paths{source_lock_path};
    if (pack_lock_path != source_lock_path) lock_paths.push_back(pack_lock_path);
    for (const std::filesystem::path& lock_path : lock_paths) {
        const WindowSourceFileLock* inspected =
            find_inspection_lock(sources, lock_path.string());
        if (inspected == nullptr) {
            error = lock_path.string() +
                    ": tiered source is missing its WeightStore artifact-set lock";
            return false;
        }
        FileDescriptor descriptor;
        if (!open_read_descriptor(lock_path.string(), descriptor, error) ||
            !lock_source_file(descriptor.get(), lock_path.string(), error) ||
            !verify_open_file(descriptor.get(), lock_path.string(), inspected->file_size,
                              inspected->file_device_id, inspected->file_object_id, error)) {
            return false;
        }
        if (!inspected->lock_lease()->matches(lock_path.string(), inspected->file_size,
                                              inspected->file_device_id,
                                              inspected->file_object_id)) {
            error = lock_path.string() +
                    ": artifact-set inspection lease does not own the inspected file";
            return false;
        }
        artifact_set_locks.push_back(std::move(descriptor));
    }
    const std::filesystem::path marker =
        dense_path.parent_path() / ".strata-generation-pending";
    std::error_code marker_error;
    if (std::filesystem::exists(marker, marker_error)) {
        error = "artifact generation is incomplete: " + marker.string();
        return false;
    }
    if (marker_error) {
        error = "cannot inspect artifact generation marker " + marker.string() +
                ": " + marker_error.message();
        return false;
    }
    for (const WindowFileSource& source : sources.files) {
        if (!has_inspection_lock(sources, source.path, source.file_size,
                                 source.file_device_id, source.file_object_id)) {
            error = source.path + ": tiered source is missing its WeightStore inspection lock";
            return false;
        }
        ShardDescriptors descriptors;
        if (!open_read_descriptor(source.path, descriptors.bulk, error) ||
            !lock_source_file(descriptors.bulk.get(), source.path, error) ||
            !verify_open_file(descriptors.bulk.get(), source.path, source.file_size,
                              source.file_device_id, source.file_object_id, error) ||
            !set_bulk_advice(descriptors.bulk.get(), source.path, error) ||
            !open_read_descriptor(source.path, descriptors.miss, error) ||
            !lock_source_file(descriptors.miss.get(), source.path, error) ||
            !verify_open_file(descriptors.miss.get(), source.path, source.file_size,
                              source.file_device_id, source.file_object_id, error) ||
            !set_miss_advice(descriptors.miss.get(), source.path, error)) {
            return false;
        }
        const WindowSourceFileLock* inspected = find_inspection_lock(sources, source.path);
        if (inspected == nullptr ||
            !inspected->lock_lease()->matches(source.path, source.file_size,
                                              source.file_device_id, source.file_object_id)) {
            error = source.path +
                    ": tiered source inspection lease does not own the inspected file";
            return false;
        }
        shards.push_back(std::move(descriptors));
    }
    if (!sources.dense_pack_tensors.empty()) {
        if (sources.dense_pack_path.empty()) {
            error = "tiered dense-pack path is missing";
            return false;
        }
        if (!has_inspection_lock(sources, sources.dense_pack_path,
                                 sources.dense_pack_file_size,
                                 sources.dense_pack_device_id,
                                 sources.dense_pack_object_id)) {
            error = sources.dense_pack_path +
                    ": tiered source is missing its WeightStore inspection lock";
            return false;
        }
        if (!open_read_descriptor(sources.dense_pack_path, dense_pack, error) ||
            !lock_source_file(dense_pack.get(), sources.dense_pack_path, error) ||
            !verify_open_file(dense_pack.get(), sources.dense_pack_path,
                              sources.dense_pack_file_size,
                              sources.dense_pack_device_id,
                              sources.dense_pack_object_id, error) ||
            !set_bulk_advice(dense_pack.get(), sources.dense_pack_path, error)) {
            return false;
        }
        const WindowSourceFileLock* inspected =
            find_inspection_lock(sources, sources.dense_pack_path);
        if (inspected == nullptr ||
            !inspected->lock_lease()->matches(sources.dense_pack_path,
                                              sources.dense_pack_file_size,
                                              sources.dense_pack_device_id,
                                              sources.dense_pack_object_id)) {
            error = sources.dense_pack_path +
                    ": tiered dense-pack inspection lease does not own the inspected file";
            return false;
        }
    }
    return true;
}

bool allocate_aligned_mapping(std::uint64_t bytes, void*& mapping, std::size_t& mapping_bytes,
                              void*& aligned, std::string& error) {
    mapping = nullptr;
    aligned = nullptr;
    mapping_bytes = 0;
    if (bytes == 0) return true;
    if (bytes > std::numeric_limits<std::size_t>::max() ||
        bytes > std::numeric_limits<std::size_t>::max() - (kDenseArenaAlignment - 1)) {
        error = "tiered dense arena does not fit in this process address space";
        return false;
    }
    mapping_bytes = static_cast<std::size_t>(bytes + kDenseArenaAlignment - 1);
    mapping = ::mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping == MAP_FAILED) {
        mapping = nullptr;
        error = system_error("cannot mmap tiered dense arena", "anonymous memory", errno);
        return false;
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(mapping);
    if (raw > std::numeric_limits<std::uintptr_t>::max() -
                  (kDenseArenaAlignment - 1)) {
        (void)::munmap(mapping, mapping_bytes);
        mapping = nullptr;
        error = "tiered dense arena address alignment overflows";
        return false;
    }
    const auto aligned_value =
        (raw + kDenseArenaAlignment - 1) &
        ~static_cast<std::uintptr_t>(kDenseArenaAlignment - 1);
    if (aligned_value - raw > mapping_bytes ||
        bytes > mapping_bytes - (aligned_value - raw)) {
        (void)::munmap(mapping, mapping_bytes);
        mapping = nullptr;
        error = "tiered dense arena alignment exceeds its anonymous mapping";
        return false;
    }
    aligned = reinterpret_cast<void*>(aligned_value);
    return true;
}

struct CopyTask {
    int descriptor = -1;
    std::uint64_t source_offset = 0;
    std::uint8_t* destination = nullptr;
    std::uint64_t byte_count = 0;
    std::string label;
};

bool pread_exact(const CopyTask& task, std::string& error) {
    if (task.descriptor < 0 || task.destination == nullptr || task.byte_count == 0 ||
        task.source_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        task.byte_count > std::numeric_limits<std::size_t>::max() ||
        task.byte_count > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) -
                              task.source_offset) {
        error = task.label + ": read range cannot be represented";
        return false;
    }
    std::uint64_t completed = 0;
    while (completed < task.byte_count) {
        const std::uint64_t remaining = task.byte_count - completed;
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining,
                                    static_cast<std::uint64_t>(std::numeric_limits<ssize_t>::max())));
        const ssize_t count = ::pread(
            task.descriptor, task.destination + static_cast<std::size_t>(completed), chunk,
            static_cast<off_t>(task.source_offset + completed));
        if (count < 0) {
            if (errno == EINTR) continue;
            error = system_error("pread failed for", task.label, errno);
            return false;
        }
        if (count == 0) {
            error = task.label + ": short read at byte " + std::to_string(completed);
            return false;
        }
        completed += static_cast<std::uint64_t>(count);
    }
    return true;
}

bool copy_startup_ranges(const std::vector<CopyTask>& tasks, std::string& error) {
    if (tasks.empty()) return true;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string first_error;
    std::vector<std::thread> workers;
    workers.reserve(kTieredCopyWorkers);
    try {
        for (std::size_t worker = 0; worker < kTieredCopyWorkers; ++worker) {
            workers.emplace_back([&]() {
                (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
                while (!failed.load(std::memory_order_acquire)) {
                    const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= tasks.size()) return;
                    std::string task_error;
                    if (!pread_exact(tasks[index], task_error)) {
                        {
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (first_error.empty()) first_error = std::move(task_error);
                        }
                        failed.store(true, std::memory_order_release);
                        return;
                    }
                }
            });
        }
    } catch (...) {
        failed.store(true, std::memory_order_release);
        for (std::thread& worker : workers) if (worker.joinable()) worker.join();
        error = "cannot create the eight tiered startup-copy workers";
        return false;
    }
    for (std::thread& worker : workers) worker.join();
    if (failed.load(std::memory_order_acquire)) {
        error = first_error.empty() ? "tiered startup copy failed" : std::move(first_error);
        return false;
    }
    return true;
}

}  // namespace

struct TieredStorage::Impl {
    Context* context = nullptr;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    TieredLayout layout;
    BufferRef dense_arena;
    BufferRef resident_arena;
    BufferRef slot_arena;
    std::unique_ptr<SlotPool> slot_pool;
    BufferRef expert_addresses;
    BufferRef resident;
    BufferRef plans;
    std::vector<ShardDescriptors> shard_descriptors;
    std::vector<std::string> shard_paths;
    std::vector<std::uint64_t> shard_file_sizes;
    std::vector<FileDescriptor> artifact_set_lock_descriptors;
    FileDescriptor dense_pack_descriptor;
};

TieredStorage::TieredStorage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TieredStorage::~TieredStorage() = default;

std::unique_ptr<TieredStorage> TieredStorage::open(Context& context,
                                                   const std::string& native_shard,
                                                   const Geometry& geometry,
                                                   const WindowSources& sources,
                                                   const PlanResult& memory_plan,
                                                   std::string& error) {
    error.clear();
    if (native_shard.empty() || !memory_plan.fits || memory_plan.mode != "tiered") {
        error = "tiered storage needs a GGUF shard and a fitting tiered memory plan";
        return nullptr;
    }
    auto candidate = std::make_unique<Impl>();
    candidate->context = &context;
    candidate->n_layers = geometry.n_layers;
    candidate->n_expert = geometry.n_expert;

    std::shared_ptr<strata::GgufModel> model;
    try {
        model = std::make_shared<strata::GgufModel>(
            strata::gguf_split_paths(native_shard));
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
    if (!validate_model_sources(*model, geometry, sources, error)) return nullptr;

    TieredLayout preflight;
    constexpr std::uint64_t kPreflightDenseBase = 0x100000000ull;
    constexpr std::uint64_t kPreflightResidentBase = 0x200000000ull;
    if (!build_tiered_layout(geometry, sources, memory_plan,
                             kPreflightDenseBase, kPreflightResidentBase,
                             preflight, error)) {
        return nullptr;
    }
    std::uint64_t max_blob = 0;
    std::vector<std::uint64_t> resident_by_layer(geometry.n_layers, 0);
    for (const auto& [layer, expert] : memory_plan.resident_order) {
        (void)expert;
        if (layer >= geometry.n_layers) {
            error = "tiered resident order contains an invalid layer";
            return nullptr;
        }
        ++resident_by_layer[layer];
    }
    for (const LayerGeometry& layer : geometry.layers) {
        max_blob = std::max(max_blob, layer.blob_bytes);
    }
    std::uint64_t slot_stride = 0;
    std::uint64_t expected_slot_bytes = 0;
    if (memory_plan.slots != 0 &&
        (!align_up(max_blob, kSlotAlignment, slot_stride) ||
         !multiply(memory_plan.slots, slot_stride, expected_slot_bytes))) {
        error = "tiered slot-pool dimensions overflow";
        return nullptr;
    }
    std::uint64_t max_layer_misses = 0;
    for (const std::uint64_t resident_count : resident_by_layer) {
        max_layer_misses = std::max<std::uint64_t>(
            max_layer_misses, geometry.n_expert - resident_count);
    }
    if (memory_plan.slot_bytes != expected_slot_bytes ||
        memory_plan.slots < max_layer_misses * 2) {
        error = "tiered slot pool does not match its planned size or per-layer miss capacity";
        return nullptr;
    }
    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0 ||
        preflight.dense_arena_bytes > facts.max_buffer_length ||
        preflight.resident_arena_bytes > facts.max_buffer_length ||
        memory_plan.slot_bytes > facts.max_buffer_length) {
        error = "tiered arena exceeds the device's maxBufferLength";
        return nullptr;
    }
    if (!open_descriptors(sources, candidate->shard_descriptors,
                          candidate->artifact_set_lock_descriptors,
                          candidate->dense_pack_descriptor, error)) {
        return nullptr;
    }
    candidate->shard_paths.reserve(sources.files.size());
    candidate->shard_file_sizes.reserve(sources.files.size());
    for (const WindowFileSource& source : sources.files) {
        candidate->shard_paths.push_back(source.path);
        candidate->shard_file_sizes.push_back(source.file_size);
    }

    auto buffer_transaction = context.buffers().transaction();
    void* mapping = nullptr;
    void* dense_address = nullptr;
    std::size_t mapping_bytes = 0;
    if (!allocate_aligned_mapping(preflight.dense_arena_bytes, mapping,
                                  mapping_bytes, dense_address, error)) {
        return nullptr;
    }
    std::shared_ptr<void> mapping_lifetime;
    if (mapping != nullptr) {
        mapping_lifetime = std::shared_ptr<void>(mapping, [mapping_bytes](void* address) {
            (void)::munmap(address, mapping_bytes);
        });
        candidate->dense_arena = context.buffers().wrap_no_copy(
            "tiered dense arena", dense_address,
            static_cast<std::size_t>(preflight.dense_arena_bytes),
            std::move(mapping_lifetime), error);
        if (!candidate->dense_arena) return nullptr;
    }
    if (preflight.resident_arena_bytes != 0) {
        if (preflight.resident_arena_bytes > std::numeric_limits<std::size_t>::max()) {
            error = "tiered resident arena does not fit in this process address space";
            return nullptr;
        }
        candidate->resident_arena = context.buffers().allocate(
            "tiered resident arena",
            static_cast<std::size_t>(preflight.resident_arena_bytes), error);
        if (!candidate->resident_arena) return nullptr;
    }
    if (memory_plan.slot_bytes != 0) {
        if (memory_plan.slot_bytes > std::numeric_limits<std::size_t>::max()) {
            error = "tiered slot arena does not fit in this process address space";
            return nullptr;
        }
        candidate->slot_arena = context.buffers().allocate(
            "tiered slot arena", static_cast<std::size_t>(memory_plan.slot_bytes), error);
        if (!candidate->slot_arena) return nullptr;
    }
    candidate->slot_pool = SlotPool::create(
        candidate->slot_arena, geometry.n_layers, geometry.n_expert,
        memory_plan.slots, slot_stride, error);
    if (!candidate->slot_pool) return nullptr;

    const std::uint64_t dense_gpu_base =
        candidate->dense_arena ? candidate->dense_arena.gpu_address : 0;
    const std::uint64_t resident_gpu_base =
        candidate->resident_arena ? candidate->resident_arena.gpu_address : 0;
    if (!build_tiered_layout(geometry, sources, memory_plan,
                             dense_gpu_base, resident_gpu_base,
                             candidate->layout, error)) {
        return nullptr;
    }

    if (candidate->layout.expert_addresses.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) ||
        candidate->layout.resident.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) ||
        candidate->layout.plans.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(TieredMoePlan)) {
        error = "tiered address, resident or plan table size overflows";
        return nullptr;
    }
    candidate->expert_addresses = context.buffers().allocate(
        "tiered expert address table",
        candidate->layout.expert_addresses.size() * sizeof(std::uint64_t), error);
    if (!candidate->expert_addresses) return nullptr;
    candidate->resident = context.buffers().allocate(
        "tiered resident table",
        candidate->layout.resident.size() * sizeof(std::int32_t), error);
    if (!candidate->resident) return nullptr;
    candidate->plans = context.buffers().allocate(
        "tiered MoE plan table",
        candidate->layout.plans.size() * sizeof(TieredMoePlan), error);
    if (!candidate->plans) return nullptr;

    std::vector<CopyTask> tasks;
    tasks.reserve(candidate->layout.dense_copies.size() +
                  memory_plan.resident_order.size() * kExpertRoles);
    auto* dense_bytes = static_cast<std::uint8_t*>(candidate->dense_arena.contents);
    for (const TieredDenseCopy& copy : candidate->layout.dense_copies) {
        if (copy.source_byte_count != copy.byte_count ||
            copy.source_byte_count > std::numeric_limits<std::size_t>::max() ||
            copy.arena_offset > candidate->dense_arena.length ||
            copy.byte_count > candidate->dense_arena.length - copy.arena_offset) {
            error = "tiered dense copy range cannot be represented for " + copy.tensor_name;
            return nullptr;
        }
        const int descriptor = copy.from_dense_pack
            ? candidate->dense_pack_descriptor.get()
            : candidate->shard_descriptors[copy.shard_index].bulk.get();
        tasks.push_back({descriptor, copy.source_offset,
                         dense_bytes + static_cast<std::size_t>(copy.arena_offset),
                         copy.source_byte_count, copy.tensor_name});
    }

    auto* resident_bytes = static_cast<std::uint8_t*>(candidate->resident_arena.contents);
    for (const auto& [layer, expert_index] : memory_plan.resident_order) {
        const TieredExpert& expert = candidate->layout.experts[
            static_cast<std::size_t>(layer) * geometry.n_expert + expert_index];
        std::uint64_t role_offset = 0;
        for (std::uint32_t role = 0; role < kExpertRoles; ++role) {
            const TieredSourceRange& source = expert.roles[role];
            if (source.byte_count > std::numeric_limits<std::size_t>::max() ||
                expert.resident_arena_offset > candidate->resident_arena.length ||
                role_offset > candidate->resident_arena.length - expert.resident_arena_offset ||
                source.byte_count >
                    candidate->resident_arena.length - expert.resident_arena_offset - role_offset) {
                error = "tiered resident expert copy range cannot be represented";
                return nullptr;
            }
            tasks.push_back({
                candidate->shard_descriptors[source.shard_index].bulk.get(),
                source.file_offset,
                resident_bytes + static_cast<std::size_t>(expert.resident_arena_offset + role_offset),
                source.byte_count,
                "layer " + std::to_string(layer) + " expert " +
                    std::to_string(expert_index) + " role " + std::to_string(role)});
            if (!add(role_offset, source.byte_count, role_offset)) {
                error = "tiered resident role destination offset overflows";
                return nullptr;
            }
        }
    }
    if (!copy_startup_ranges(tasks, error)) return nullptr;

    std::memcpy(candidate->expert_addresses.contents,
                candidate->layout.expert_addresses.data(),
                candidate->layout.expert_addresses.size() * sizeof(std::uint64_t));
    std::memcpy(candidate->resident.contents, candidate->layout.resident.data(),
                candidate->layout.resident.size() * sizeof(std::int32_t));
    std::memcpy(candidate->plans.contents, candidate->layout.plans.data(),
                candidate->layout.plans.size() * sizeof(TieredMoePlan));
    if (!context.buffers().commit_residency(error)) return nullptr;

    auto storage = std::unique_ptr<TieredStorage>(new TieredStorage(std::move(candidate)));
    buffer_transaction.commit();
    return storage;
}

const TieredLayout& TieredStorage::layout() const noexcept { return impl_->layout; }
BufferRef TieredStorage::dense_arena() const noexcept { return impl_->dense_arena; }
BufferRef TieredStorage::resident_arena() const noexcept { return impl_->resident_arena; }
BufferRef TieredStorage::slot_arena() const noexcept { return impl_->slot_arena; }
SlotPool* TieredStorage::slot_pool() noexcept { return impl_->slot_pool.get(); }
bool TieredStorage::read_expert_role(std::uint32_t layer, std::uint32_t expert_index,
                                     std::uint32_t role, void* destination,
                                     std::size_t destination_bytes,
                                     std::string& error) const {
    error.clear();
    if (layer >= impl_->n_layers || impl_->n_expert == 0) {
        error = "tiered expert layer is out of range";
        return false;
    }
    if (expert_index >= impl_->n_expert || role >= kExpertRoles) {
        error = "tiered expert id or role is out of range";
        return false;
    }
    const std::uint64_t flat_index =
        static_cast<std::uint64_t>(layer) * impl_->n_expert + expert_index;
    if (flat_index >= impl_->layout.experts.size()) {
        error = "tiered expert index is out of range";
        return false;
    }
    const TieredSourceRange& source =
        impl_->layout.experts[static_cast<std::size_t>(flat_index)].roles[role];
    if (source.shard_index >= impl_->shard_descriptors.size() ||
        source.byte_count == 0 ||
        source.byte_count != destination_bytes || destination == nullptr ||
        source.byte_count > std::numeric_limits<std::size_t>::max() ||
        source.file_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        source.byte_count >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) -
                source.file_offset) {
        error = "tiered expert slab destination or source range is invalid";
        return false;
    }
    const int descriptor = impl_->shard_descriptors[source.shard_index].miss.get();
    std::size_t completed = 0;
    while (completed < destination_bytes) {
        const std::size_t amount = std::min<std::size_t>(
            destination_bytes - completed,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const std::uint64_t offset =
            source.file_offset + static_cast<std::uint64_t>(completed);
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            error = "tiered expert slab offset exceeds off_t";
            return false;
        }
        const ssize_t count = ::pread(
            descriptor, static_cast<std::uint8_t*>(destination) + completed,
            amount, static_cast<off_t>(offset));
        if (count < 0) {
            if (errno == EINTR) continue;
            error = system_error("cannot read", impl_->shard_paths[source.shard_index],
                                 errno);
            return false;
        }
        if (count == 0) {
            error = impl_->shard_paths[source.shard_index] +
                    ": unexpected end of file while reading tiered expert slab";
            return false;
        }
        completed += static_cast<std::size_t>(count);
    }
    return true;
}

bool TieredStorage::expert_pages_resident(std::uint32_t layer,
                                          std::uint32_t expert_index) const noexcept {
    if (layer >= impl_->n_layers || impl_->n_expert == 0 ||
        expert_index >= impl_->n_expert) {
        return false;
    }
    const std::uint64_t flat_index =
        static_cast<std::uint64_t>(layer) * impl_->n_expert + expert_index;
    if (flat_index >= impl_->layout.experts.size()) return false;
    const long page_size_value = ::sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) return false;
    const std::uint64_t page_size = static_cast<std::uint64_t>(page_size_value);
    const TieredExpert& expert =
        impl_->layout.experts[static_cast<std::size_t>(flat_index)];
    for (const TieredSourceRange& source : expert.roles) {
        if (source.shard_index >= impl_->shard_descriptors.size() ||
            source.shard_index >= impl_->shard_file_sizes.size() ||
            source.byte_count == 0) {
            return false;
        }
        std::uint64_t end = 0;
        if (!add(source.file_offset, source.byte_count, end) ||
            end > impl_->shard_file_sizes[source.shard_index]) {
            return false;
        }
        const std::uint64_t aligned_start = source.file_offset / page_size * page_size;
        std::uint64_t aligned_end = 0;
        if (!align_up(end, page_size, aligned_end) || aligned_end < aligned_start) return false;
        const std::uint64_t map_bytes = aligned_end - aligned_start;
        const std::uint64_t page_count = map_bytes / page_size;
        if (map_bytes == 0 ||
            map_bytes > std::numeric_limits<std::size_t>::max() ||
            page_count > std::numeric_limits<std::size_t>::max() ||
            aligned_start > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            return false;
        }
        std::vector<char> residency;
        try {
            residency.resize(static_cast<std::size_t>(page_count));
        } catch (...) {
            return false;
        }
        const int descriptor = impl_->shard_descriptors[source.shard_index].miss.get();
        void* mapping = ::mmap(nullptr, static_cast<std::size_t>(map_bytes), PROT_READ,
                               MAP_SHARED, descriptor, static_cast<off_t>(aligned_start));
        if (mapping == MAP_FAILED) return false;
        const bool queried = ::mincore(mapping, static_cast<std::size_t>(map_bytes),
                                       residency.data()) == 0;
        (void)::munmap(mapping, static_cast<std::size_t>(map_bytes));
        if (!queried ||
            !std::all_of(residency.begin(), residency.end(),
                         [](char page) { return (page & 1) != 0; })) {
            return false;
        }
    }
    return true;
}
BufferRef TieredStorage::expert_address_table() const noexcept { return impl_->expert_addresses; }
BufferRef TieredStorage::resident_table() const noexcept { return impl_->resident; }
BufferRef TieredStorage::plan_buffer() const noexcept { return impl_->plans; }

}  // namespace strata::metal
