// src/metal/engine/mtp_storage.cpp - bounded read-only MTP mmap and Metal ownership.
#include "strata/metal/mtp_storage.hpp"
#include "window_source_lock.hpp"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::metal {
namespace {

struct MappedFile {
    int descriptor = -1;
    void* address = MAP_FAILED;
    std::size_t bytes = 0;

    MappedFile() = default;
    ~MappedFile() { reset(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : descriptor(std::exchange(other.descriptor, -1)),
          address(std::exchange(other.address, MAP_FAILED)),
          bytes(std::exchange(other.bytes, 0)) {}

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this == &other) return *this;
        reset();
        descriptor = std::exchange(other.descriptor, -1);
        address = std::exchange(other.address, MAP_FAILED);
        bytes = std::exchange(other.bytes, 0);
        return *this;
    }

    void reset() noexcept {
        if (address != MAP_FAILED) {
            (void)::munmap(address, bytes);
            address = MAP_FAILED;
            bytes = 0;
        }
        if (descriptor >= 0) {
            (void)::close(descriptor);
            descriptor = -1;
        }
    }
};

/// Declared with the lock first so mappings are unmapped before the source
/// generation lease is released.
struct MappedGeneration {
    MtpRuntimeSources sources;
    MappedFile experts;
    MappedFile dense;
};

bool round_mapping_size(std::uint64_t file_size, std::uint64_t& mapped_size) {
    if (file_size == 0) return false;
    const std::uint64_t remainder = file_size % kWindowPageBytes;
    if (remainder == 0) {
        mapped_size = file_size;
        return true;
    }
    const std::uint64_t padding = kWindowPageBytes - remainder;
    if (padding > std::numeric_limits<std::uint64_t>::max() - file_size) return false;
    mapped_size = file_size + padding;
    return true;
}

bool map_file(const std::string& path, std::uint64_t expected_size,
              std::uint64_t expected_device_id, std::uint64_t expected_object_id,
              MappedFile& output, std::string& error) {
    output.reset();
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        error = "cannot open MTP runtime file " + path + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = "cannot stat MTP runtime file " + path + ": " +
                std::error_code(saved_errno, std::generic_category()).message();
        return false;
    }
    if (!S_ISREG(status.st_mode)) {
        (void)::close(descriptor);
        error = "MTP runtime file is not a regular file: " + path;
        return false;
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != expected_size ||
        static_cast<std::uint64_t>(status.st_dev) != expected_device_id ||
        static_cast<std::uint64_t>(status.st_ino) != expected_object_id) {
        (void)::close(descriptor);
        error = "MTP runtime file identity or size changed after inspection: " + path;
        return false;
    }

    std::uint64_t mapped_size = 0;
    if (!round_mapping_size(expected_size, mapped_size) ||
        mapped_size > std::numeric_limits<std::size_t>::max() ||
        mapped_size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        (void)::close(descriptor);
        error = "MTP runtime mapping size cannot be represented: " + path;
        return false;
    }
    void* address = ::mmap(nullptr, static_cast<std::size_t>(mapped_size), PROT_READ,
                           MAP_SHARED, descriptor, 0);
    if (address == MAP_FAILED) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = "cannot map MTP runtime file " + path + ": " +
                std::error_code(saved_errno, std::generic_category()).message();
        return false;
    }

    output.descriptor = descriptor;
    output.address = address;
    output.bytes = static_cast<std::size_t>(mapped_size);
    error.clear();
    return true;
}

bool validate_generation_lock_path(const WindowSourceFileLock& lock,
                                   std::string& error) {
    const auto& lease = lock.lock_lease();
    struct stat status {};
    if (lease == nullptr ||
        !lease->matches(lock.path, lock.file_size,
                        lock.file_device_id, lock.file_object_id) ||
        ::lstat(lock.path.c_str(), &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != lock.file_size ||
        static_cast<std::uint64_t>(status.st_dev) != lock.file_device_id ||
        static_cast<std::uint64_t>(status.st_ino) != lock.file_object_id) {
        error = lock.path + ": artifact-set identity changed after WeightStore inspection";
        return false;
    }
    error.clear();
    return true;
}

BufferRef slice(BufferRef window, std::uint64_t offset, std::uint64_t bytes) noexcept {
    if (!window || window.contents == nullptr || bytes == 0 ||
        offset > window.length || bytes > window.length - offset ||
        offset > std::numeric_limits<std::size_t>::max() ||
        bytes > std::numeric_limits<std::size_t>::max() ||
        offset > std::numeric_limits<std::uint64_t>::max() - window.gpu_address ||
        offset > std::numeric_limits<std::size_t>::max() - window.offset) {
        return {};
    }
    return window.slice(static_cast<std::size_t>(offset),
                        static_cast<std::size_t>(bytes));
}

}  // namespace

struct MtpStorage::Impl {
    Buffers* buffers = nullptr;
    std::shared_ptr<const MtpRuntimeCatalog> catalog;
    MtpRuntimeLayout layout;
    std::vector<BufferRef> expert_windows;
    std::vector<std::uint64_t> expert_window_identities;
    std::vector<BufferRef> dense_windows;
    std::vector<std::uint64_t> dense_window_identities;
};

MtpStorage::MtpStorage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MtpStorage::~MtpStorage() = default;

std::unique_ptr<MtpStorage> MtpStorage::open(Context& context,
                                             const MtpRuntimeSources& sources,
                                             std::string& error) {
    error.clear();
    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0) {
        error = "Metal device reported an invalid maxBufferLength for MTP storage";
        return nullptr;
    }
    MtpRuntimeLayout layout;
    if (!build_mtp_runtime_layout(sources, facts.max_buffer_length, layout, error)) {
        return nullptr;
    }
    if (!validate_generation_lock_path(sources.generation_lock, error)) return nullptr;
    const MtpRuntimeCatalog& catalog = *sources.catalog;
    const std::filesystem::path directory(catalog.directory);
    if (catalog.experts_path != (directory / "experts.bin").string() ||
        catalog.dense_path != (directory / "dense.bin").string()) {
        error = "MTP storage paths do not match the inspected canonical directory";
        return nullptr;
    }
    const long host_page_size = ::getpagesize();
    if (host_page_size <= 0 ||
        kWindowPageBytes % static_cast<std::uint64_t>(host_page_size) != 0) {
        error = "MTP storage windows are incompatible with the host page size";
        return nullptr;
    }

    auto generation = std::make_shared<MappedGeneration>();
    generation->sources = sources;
    if (!map_file(catalog.experts_path, catalog.experts_file_size,
                  catalog.experts_file_device_id, catalog.experts_file_object_id,
                  generation->experts, error)) {
        return nullptr;
    }
    auto candidate = std::make_unique<Impl>();
    candidate->buffers = &context.buffers();
    candidate->catalog = sources.catalog;
    candidate->layout = std::move(layout);

    auto buffer_transaction = candidate->buffers->transaction();
    candidate->expert_windows.reserve(candidate->layout.expert_windows.size());
    candidate->expert_window_identities.reserve(candidate->layout.expert_windows.size());
    std::vector<BufferRef> warmup_windows;
    warmup_windows.reserve(candidate->layout.expert_windows.size() +
                           candidate->layout.dense_windows.size());
    for (std::size_t index = 0; index < candidate->layout.expert_windows.size(); ++index) {
        const FileWindow& range = candidate->layout.expert_windows[index];
        if (range.start % kWindowPageBytes != 0 || range.end <= range.start ||
            range.end > generation->experts.bytes ||
            range.start > std::numeric_limits<std::size_t>::max() ||
            range.end - range.start > std::numeric_limits<std::size_t>::max() ||
            range.end - range.start > facts.max_buffer_length) {
            error = "MTP expert window is outside its read-only mapping or device limit";
            return nullptr;
        }
        auto* address = static_cast<std::uint8_t*>(generation->experts.address) +
                        static_cast<std::size_t>(range.start);
        BufferRef buffer = candidate->buffers->wrap_no_copy(
            "MTP experts window " + std::to_string(index), address,
            static_cast<std::size_t>(range.end - range.start),
            std::static_pointer_cast<void>(generation), error);
        if (!buffer) return nullptr;
        const std::uint64_t identity = candidate->buffers->allocation_identity(buffer);
        if (identity == 0 ||
            !candidate->buffers->owns(buffer, identity)) {
            error = "Metal did not retain an owned MTP expert window";
            return nullptr;
        }
        candidate->expert_windows.push_back(buffer);
        candidate->expert_window_identities.push_back(identity);
        warmup_windows.push_back(buffer);
    }

    if (!map_file(catalog.dense_path, catalog.dense_file_size,
                  catalog.dense_file_device_id, catalog.dense_file_object_id,
                  generation->dense, error)) {
        return nullptr;
    }
    candidate->dense_windows.reserve(candidate->layout.dense_windows.size());
    candidate->dense_window_identities.reserve(candidate->layout.dense_windows.size());
    for (std::size_t index = 0; index < candidate->layout.dense_windows.size(); ++index) {
        const FileWindow& range = candidate->layout.dense_windows[index];
        if (range.start % kWindowPageBytes != 0 || range.end <= range.start ||
            range.end > generation->dense.bytes ||
            range.start > std::numeric_limits<std::size_t>::max() ||
            range.end - range.start > std::numeric_limits<std::size_t>::max() ||
            range.end - range.start > facts.max_buffer_length) {
            error = "MTP dense window is outside its read-only mapping or device limit";
            return nullptr;
        }
        auto* address = static_cast<std::uint8_t*>(generation->dense.address) +
                        static_cast<std::size_t>(range.start);
        BufferRef buffer = candidate->buffers->wrap_no_copy(
            "MTP dense window " + std::to_string(index), address,
            static_cast<std::size_t>(range.end - range.start),
            std::static_pointer_cast<void>(generation), error);
        if (!buffer) return nullptr;
        const std::uint64_t identity = candidate->buffers->allocation_identity(buffer);
        if (identity == 0 ||
            !candidate->buffers->owns(buffer, identity)) {
            error = "Metal did not retain an owned MTP dense window";
            return nullptr;
        }
        candidate->dense_windows.push_back(buffer);
        candidate->dense_window_identities.push_back(identity);
        warmup_windows.push_back(buffer);
    }
    if (!candidate->buffers->commit_residency(error) ||
        !context.warm_no_copy_windows(warmup_windows, error)) {
        return nullptr;
    }
    auto result = std::unique_ptr<MtpStorage>(
        new MtpStorage(std::move(candidate)));
    buffer_transaction.commit();
    error.clear();
    return result;
}

const MtpRuntimeLayout& MtpStorage::layout() const noexcept {
    return impl_->layout;
}

const std::vector<std::uint32_t>& MtpStorage::draft_vocab_ids() const noexcept {
    return impl_->catalog->draft_vocab_ids;
}

BufferRef MtpStorage::expert(std::uint32_t expert_index) const noexcept {
    if (expert_index >= impl_->layout.experts.size()) return {};
    const MtpExpertBinding& binding = impl_->layout.experts[expert_index];
    if (binding.expert_index != expert_index ||
        binding.window_index >= impl_->expert_windows.size() ||
        binding.window_index >= impl_->expert_window_identities.size()) {
        return {};
    }
    const std::size_t window_index = static_cast<std::size_t>(binding.window_index);
    if (!impl_->buffers->owns(impl_->expert_windows[window_index],
                              impl_->expert_window_identities[window_index])) {
        return {};
    }
    return slice(impl_->expert_windows[window_index],
                 binding.window_offset, binding.byte_count);
}

BufferRef MtpStorage::dense_weight(const std::string& tensor_name) const noexcept {
    const auto binding = std::find_if(
        impl_->layout.dense_weights.begin(), impl_->layout.dense_weights.end(),
        [&](const MtpDenseWeightBinding& item) { return item.tensor_name == tensor_name; });
    if (binding == impl_->layout.dense_weights.end() ||
        binding->window_index >= impl_->dense_windows.size() ||
        binding->window_index >= impl_->dense_window_identities.size()) {
        return {};
    }
    const std::size_t window_index = static_cast<std::size_t>(binding->window_index);
    if (!impl_->buffers->owns(impl_->dense_windows[window_index],
                              impl_->dense_window_identities[window_index])) {
        return {};
    }
    return slice(impl_->dense_windows[window_index],
                 binding->window_offset, binding->byte_count);
}

BufferRef MtpStorage::expert_window(std::size_t window_index) const noexcept {
    return window_index < impl_->expert_windows.size() &&
                   window_index < impl_->expert_window_identities.size() &&
                   impl_->buffers->owns(impl_->expert_windows[window_index],
                                        impl_->expert_window_identities[window_index])
        ? impl_->expert_windows[window_index] : BufferRef{};
}

BufferRef MtpStorage::dense_window(std::size_t window_index) const noexcept {
    return window_index < impl_->dense_windows.size() &&
                   window_index < impl_->dense_window_identities.size() &&
                   impl_->buffers->owns(impl_->dense_windows[window_index],
                                        impl_->dense_window_identities[window_index])
        ? impl_->dense_windows[window_index] : BufferRef{};
}

}  // namespace strata::metal
