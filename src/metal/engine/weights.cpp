// src/metal/engine/weights.cpp - strict, payload-free inspection for planning and geometry export.
#include "strata/metal/weights.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/mtp_bind_layout.hpp"
#include "window_source_lock.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::metal {

WindowSourceLockLease::WindowSourceLockLease(
    int descriptor, std::string path, std::uint64_t file_size,
    std::uint64_t file_device_id, std::uint64_t file_object_id)
    : descriptor_(descriptor), path_(std::move(path)), file_size_(file_size),
      file_device_id_(file_device_id), file_object_id_(file_object_id) {}

WindowSourceLockLease::~WindowSourceLockLease() {
    if (descriptor_ >= 0) {
        const int saved_errno = errno;
        (void)::close(descriptor_);
        errno = saved_errno;
    }
}

std::shared_ptr<const WindowSourceLockLease> WindowSourceLockLease::acquire_shared(
    const std::string& path, bool create, std::string& error) {
    const int flags = create
        ? O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK
        : O_RDONLY | O_CLOEXEC | O_NONBLOCK;
    const int descriptor = create
        ? ::open(path.c_str(), flags, 0600)
        : ::open(path.c_str(), flags);
    if (descriptor < 0) {
        error = "cannot open source for inspection lock " + path + ": " +
                std::error_code(errno, std::generic_category()).message();
        return nullptr;
    }
    if (::flock(descriptor, LOCK_SH | LOCK_NB) != 0) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = saved_errno == EWOULDBLOCK || saved_errno == EAGAIN
            ? path + ": source is locked for writing"
            : "cannot lock source file " + path + ": " +
                  std::error_code(saved_errno, std::generic_category()).message();
        return nullptr;
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = "cannot stat locked source file " + path + ": " +
                std::error_code(saved_errno, std::generic_category()).message();
        return nullptr;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        (void)::close(descriptor);
        error = "locked source is not a regular file: " + path;
        return nullptr;
    }

    WindowSourceLockLease* raw_lease = nullptr;
    try {
        raw_lease = new WindowSourceLockLease(
            descriptor, path, static_cast<std::uint64_t>(status.st_size),
            static_cast<std::uint64_t>(status.st_dev),
            static_cast<std::uint64_t>(status.st_ino));
    } catch (...) {
        (void)::close(descriptor);
        throw;
    }
    auto lease = std::shared_ptr<const WindowSourceLockLease>(raw_lease);
    error.clear();
    return lease;
}

bool WindowSourceLockLease::matches(
    const std::string& path, std::uint64_t file_size,
    std::uint64_t file_device_id, std::uint64_t file_object_id) const noexcept {
    if (descriptor_ < 0 || path_ != path || file_size_ != file_size ||
        file_device_id_ != file_device_id || file_object_id_ != file_object_id) {
        return false;
    }
    struct stat status {};
    return ::fstat(descriptor_, &status) == 0 && status.st_size >= 0 &&
           static_cast<std::uint64_t>(status.st_size) == file_size &&
           static_cast<std::uint64_t>(status.st_dev) == file_device_id &&
           static_cast<std::uint64_t>(status.st_ino) == file_object_id;
}

bool WindowSourceLockLease::matches_dense_pack_catalog(
    const std::vector<DensePackTensorSource>& rows) const noexcept {
    if (!dense_pack_catalog_bound_ || rows.size() != dense_pack_catalog_.size()) return false;
    for (const DensePackTensorSource& expected : dense_pack_catalog_) {
        const auto found = std::find_if(
            rows.begin(), rows.end(), [&](const DensePackTensorSource& row) {
                return row.tensor_name == expected.tensor_name;
            });
        if (found == rows.end() || found->file_offset != expected.file_offset ||
            found->source_byte_count != expected.source_byte_count ||
            found->byte_count != expected.byte_count || found->kind != expected.kind) {
            return false;
        }
    }
    return true;
}

bool WindowSourceLockLease::matches_mtp_catalog(
    const MtpRuntimeSources& sources) const noexcept {
    return mtp_catalog_ != nullptr && sources.catalog != nullptr &&
           mtp_catalog_.get() == sources.catalog.get();
}

void WindowSourceLockLease::bind_dense_pack_catalog(
    const std::vector<DensePackTensorSource>& rows) {
    dense_pack_catalog_ = rows;
    dense_pack_catalog_bound_ = true;
}

void WindowSourceLockLease::bind_mtp_catalog(
    std::shared_ptr<const MtpRuntimeCatalog> catalog) {
    mtp_catalog_ = std::move(catalog);
}

bool WindowSourceLockLease::read_exact(std::uint64_t offset, void* destination,
                                       std::uint64_t bytes, std::string& error) const {
    error.clear();
    const std::uint64_t max_offset =
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (descriptor_ < 0 || destination == nullptr || bytes == 0 ||
        offset > file_size_ || bytes > file_size_ - offset ||
        offset > max_offset || bytes > max_offset - offset ||
        bytes > std::numeric_limits<std::size_t>::max()) {
        error = path_ + ": dense-pack read range cannot be represented or is out of bounds";
        return false;
    }

    auto* output = static_cast<std::uint8_t*>(destination);
    std::uint64_t completed = 0;
    while (completed < bytes) {
        const std::uint64_t remaining = bytes - completed;
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                remaining, static_cast<std::uint64_t>(std::numeric_limits<ssize_t>::max())));
        const ssize_t count = ::pread(
            descriptor_, output + static_cast<std::size_t>(completed), chunk,
            static_cast<off_t>(offset + completed));
        if (count < 0) {
            if (errno == EINTR) continue;
            error = path_ + ": dense-pack pread failed: " +
                    std::error_code(errno, std::generic_category()).message();
            return false;
        }
        if (count == 0) {
            error = path_ + ": dense-pack short read at byte " +
                    std::to_string(completed);
            return false;
        }
        completed += static_cast<std::uint64_t>(count);
    }
    return true;
}

namespace {

namespace fs = std::filesystem;
constexpr std::uint64_t kMtpExpertBytes = kMtpExpertsFileBytes;
constexpr std::uint64_t kMtpDenseBytes = kMtpDenseFileBytes;
constexpr std::uint32_t kHidden = 2'560;
constexpr std::uint32_t kFeedForward = 640;
constexpr std::uint32_t kLayers = 48;

bool artifact_set_lock_path(const fs::path& directory, fs::path& result,
                            std::string& error) {
    std::error_code ec;
    const fs::path resolved = fs::weakly_canonical(directory, ec);
    if (ec || resolved.filename().empty()) {
        error = "cannot resolve artifact-set directory " + directory.string() +
                (ec ? ": " + ec.message() : "");
        return false;
    }
    result = resolved.parent_path() /
             ("." + resolved.filename().string() + ".strata-artifact-set.lock");
    return true;
}

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    return add(value, alignment - remainder, result);
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool read_text(const fs::path& path, std::string& text, std::string& error,
               std::uint64_t limit = 16 * 1024 * 1024) {
    text.clear();
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        error = "cannot open " + path.string();
        return false;
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = "cannot stat " + path.string() + ": " +
                std::error_code(saved_errno, std::generic_category()).message();
        return false;
    }
    if (!S_ISREG(status.st_mode)) {
        (void)::close(descriptor);
        error = "cannot read " + path.string() + " (not a regular file)";
        return false;
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) > limit ||
        static_cast<std::uint64_t>(status.st_size) >
            std::numeric_limits<std::size_t>::max()) {
        (void)::close(descriptor);
        error = "cannot read " + path.string() + " (missing or larger than the metadata limit)";
        return false;
    }
    const std::size_t size = static_cast<std::size_t>(status.st_size);
    text.resize(static_cast<std::size_t>(size));
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t count =
            ::read(descriptor, text.data() + offset, size - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            const int saved_errno = count == 0 ? EIO : errno;
            (void)::close(descriptor);
            error = "failed reading " + path.string() + ": " +
                    std::error_code(saved_errno, std::generic_category()).message();
            text.clear();
            return false;
        }
        offset += static_cast<std::size_t>(count);
    }
    struct stat after {};
    if (::fstat(descriptor, &after) != 0 || after.st_size != status.st_size ||
        after.st_dev != status.st_dev || after.st_ino != status.st_ino) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        error = "file changed while reading " + path.string() + ": " +
                std::error_code(saved_errno, std::generic_category()).message();
        text.clear();
        return false;
    }
    if (::close(descriptor) != 0) {
        error = "cannot close " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        text.clear();
        return false;
    }
    error.clear();
    return true;
}

bool file_size(const fs::path& path, std::uint64_t& size, std::string& error) {
    std::error_code ec;
    size = fs::file_size(path, ec);
    if (ec) {
        error = "cannot stat " + path.string() + ": " + ec.message();
        return false;
    }
    return true;
}

struct MtpFileIdentity {
    std::uint64_t size = 0;
    std::uint64_t device_id = 0;
    std::uint64_t object_id = 0;
};

class ScopedFileDescriptor {
public:
    explicit ScopedFileDescriptor(int descriptor) : descriptor_(descriptor) {}
    ~ScopedFileDescriptor() {
        if (descriptor_ >= 0) (void)::close(descriptor_);
    }
    ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
    ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

    int get() const noexcept { return descriptor_; }
    int close() noexcept {
        if (descriptor_ < 0) return 0;
        const int descriptor = descriptor_;
        descriptor_ = -1;
        return ::close(descriptor);
    }

private:
    int descriptor_ = -1;
};

bool read_mtp_draft_vocabulary(const fs::path& path, std::uint64_t& file_size,
                               std::vector<std::uint32_t>& ids, std::string& error) {
    file_size = 0;
    ids.clear();
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        if (errno == ENOENT) {
            error.clear();
            return true;
        }
        error = "cannot open " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    ScopedFileDescriptor file(descriptor);

    struct stat before {};
    if (::fstat(file.get(), &before) != 0) {
        error = "cannot stat " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    if (!S_ISREG(before.st_mode) || before.st_size < 0) {
        error = path.string() + " is not a regular file";
        return false;
    }
    file_size = static_cast<std::uint64_t>(before.st_size);
    if (file_size == 0 || file_size % sizeof(std::uint32_t) != 0 ||
        file_size > 248'320ull * sizeof(std::uint32_t)) {
        error = path.string() + " must contain between 1 and 248320 int32 token ids";
        return false;
    }

    const std::size_t id_count =
        static_cast<std::size_t>(file_size / sizeof(std::uint32_t));
    ids.reserve(id_count);
    for (std::size_t i = 0; i < id_count; ++i) {
        unsigned char raw[sizeof(std::uint32_t)]{};
        std::size_t offset = 0;
        while (offset < sizeof(raw)) {
            const ssize_t count =
                ::read(file.get(), raw + offset, sizeof(raw) - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                const int saved_errno = count == 0 ? EIO : errno;
                error = path.string() + " changed or ended while reading token ids: " +
                        std::error_code(saved_errno, std::generic_category()).message();
                return false;
            }
            offset += static_cast<std::size_t>(count);
        }
        const std::uint32_t id = static_cast<std::uint32_t>(raw[0]) |
                                 (static_cast<std::uint32_t>(raw[1]) << 8) |
                                 (static_cast<std::uint32_t>(raw[2]) << 16) |
                                 (static_cast<std::uint32_t>(raw[3]) << 24);
        if (id >= 248'320) {
            error = path.string() +
                    " contains a token id outside the 248320-token vocabulary";
            return false;
        }
        ids.push_back(id);
    }

    char trailing = 0;
    ssize_t trailing_bytes = 0;
    do {
        trailing_bytes = ::read(file.get(), &trailing, 1);
    } while (trailing_bytes < 0 && errno == EINTR);
    if (trailing_bytes > 0) {
        error = path.string() + " changed while reading token ids";
        return false;
    }
    if (trailing_bytes < 0) {
        error = "failed reading " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }

    struct stat after {};
    if (::fstat(file.get(), &after) != 0 || after.st_size != before.st_size ||
        after.st_dev != before.st_dev || after.st_ino != before.st_ino) {
        error = path.string() + " changed while reading token ids";
        return false;
    }
    if (file.close() != 0) {
        error = "cannot close " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }

    std::vector<std::uint32_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        error = path.string() + " contains a duplicate token id";
        ids.clear();
        return false;
    }
    error.clear();
    return true;
}

bool inspect_mtp_file(const fs::path& path, MtpFileIdentity& identity,
                      std::string& error) {
    identity = {};
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        error = "cannot open MTP runtime file " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    struct stat status {};
    const int stat_result = ::fstat(descriptor, &status);
    const int stat_errno = errno;
    const int close_result = ::close(descriptor);
    if (stat_result != 0) {
        error = "cannot stat MTP runtime file " + path.string() + ": " +
                std::error_code(stat_errno, std::generic_category()).message();
        return false;
    }
    if (close_result != 0) {
        error = "cannot close MTP runtime file " + path.string() + ": " +
                std::error_code(errno, std::generic_category()).message();
        return false;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0) {
        error = "MTP runtime file is not a regular file: " + path.string();
        return false;
    }
    identity.size = static_cast<std::uint64_t>(status.st_size);
    identity.device_id = static_cast<std::uint64_t>(status.st_dev);
    identity.object_id = static_cast<std::uint64_t>(status.st_ino);
    return true;
}

bool meta_u64(const strata::GgufFile& file, const char* key, std::uint64_t& out, std::string& error) {
    const strata::MetaValue* value = file.get(key);
    if (value == nullptr) {
        error = std::string("missing ") + key;
        return false;
    }
    switch (value->type) {
        case strata::MetaType::U8:
        case strata::MetaType::U16:
        case strata::MetaType::U32:
        case strata::MetaType::U64:
            out = value->u;
            return true;
        case strata::MetaType::I8:
        case strata::MetaType::I16:
        case strata::MetaType::I32:
        case strata::MetaType::I64:
            if (static_cast<std::int64_t>(value->u) >= 0) {
                out = value->u;
                return true;
            }
            break;
        default:
            break;
    }
    error = std::string("invalid integer metadata ") + key;
    return false;
}

bool safe_atom(std::string_view text) {
    if (text.empty()) return false;
    return std::none_of(text.begin(), text.end(), [](unsigned char c) {
        return c <= 0x20 || c == '=' || c == '#';
    });
}

std::string geometry_atom(std::string text) {
    for (char& c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte <= 0x20 || c == '=' || c == '#') c = '_';
    }
    while (!text.empty() && text.front() == '_') text.erase(text.begin());
    while (!text.empty() && text.back() == '_') text.pop_back();
    return text;
}

std::string type_name(std::uint32_t type) {
    return geometry_type_name(type);
}

std::string canonical_type_name(std::string type) {
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) {
        return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a')
                                    : static_cast<char>(c);
    });
    return type;
}

bool is_expert_tensor(std::string_view name) {
    if (!name.starts_with("blk.")) return false;
    const std::size_t dot = name.find('.', 4);
    if (dot == std::string_view::npos) return false;
    const std::string_view role = name.substr(dot + 1);
    return role == "ffn_gate_exps.weight" || role == "ffn_up_exps.weight" ||
           role == "ffn_down_exps.weight";
}

const std::string* required_converted_type(std::string_view tensor_name) {
    static const std::map<std::string, std::string> forms = {
        {"ffn_gate_inp.weight", "BF16"}, {"ffn_gate_inp_shexp.weight", "BF16"},
        {"hc_attn_down.weight", "BF16"}, {"hc_attn_up.weight", "BF16"},
        {"hc_attn_inject.weight", "BF16"}, {"hc_ffn_down.weight", "BF16"},
        {"hc_ffn_up.weight", "BF16"}, {"hc_ffn_inject.weight", "BF16"},
        {"output_hc_down.weight", "BF16"}, {"output_hc_up.weight", "BF16"},
        {"indexer.k_proj.weight", "BF16"}, {"indexer.q_proj.weight", "BF16"},
        {"ssm_alpha.weight", "BF16"}, {"ssm_beta.weight", "BF16"},
        {"ple_key.weight", "BF16"}, {"ple_value.weight", "BF16"}, {"ple_conv1d.weight", "F16"},
        {"attn_q_norm.weight", "F32"}, {"attn_k_norm.weight", "F32"},
        {"hc_attn_norm.weight", "F32"}, {"hc_ffn_norm.weight", "F32"},
        {"output_hc_norm.weight", "F32"}, {"indexer.q_norm.weight", "F32"},
        {"indexer.k_norm.weight", "F32"}, {"ple_norm_conv.weight", "F32"},
        {"ple_norm_key.weight", "F32"}, {"ple_norm_query.weight", "F32"},
        {"ssm_a", "F32"}, {"ssm_conv1d.weight", "F32"}, {"ssm_dt.bias", "F32"},
        {"ssm_norm.weight", "F32"},
    };
    std::string_view short_name = tensor_name;
    if (short_name.starts_with("blk.")) {
        const std::size_t layer_end = short_name.find('.', 4);
        if (layer_end == std::string_view::npos || layer_end == 4) return nullptr;
        std::uint32_t layer = 0;
        const auto parsed = std::from_chars(
            short_name.data() + 4, short_name.data() + layer_end, layer);
        if (parsed.ec != std::errc{} || parsed.ptr != short_name.data() + layer_end ||
            layer >= kLayers ||
            short_name.substr(4, layer_end - 4) != std::to_string(layer)) {
            return nullptr;
        }
        short_name.remove_prefix(layer_end + 1);
    }
    const auto found = forms.find(std::string(short_name));
    return found == forms.end() ? nullptr : &found->second;
}

bool tensor_shape(const strata::TensorInfo& tensor, std::uint64_t ne0, std::uint64_t ne1,
                  std::uint64_t ne2) {
    return tensor.shape.size() == 3 && tensor.shape[0] == ne0 &&
           tensor.shape[1] == ne1 && tensor.shape[2] == ne2;
}

struct Conversion {
    std::string shard;
    std::string source_type;
    std::string destination_type;
};

enum class JsonKind { Null, Boolean, Number, String, Array, Object };

struct JsonValue {
    JsonKind kind = JsonKind::Null;
    bool boolean = false;
    std::string scalar;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    const JsonValue* get(std::string_view key) const {
        if (kind != JsonKind::Object) return nullptr;
        const auto found = object.find(std::string(key));
        return found == object.end() ? nullptr : &found->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    bool parse(JsonValue& value, std::string& error) {
        error_.clear();
        skip_space();
        if (!parse_value(value, 0)) {
            error = error_;
            return false;
        }
        skip_space();
        if (position_ != input_.size()) {
            error = "unexpected trailing data in conversions.json";
            return false;
        }
        return true;
    }

private:
    static bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

    void skip_space() {
        while (position_ < input_.size() && is_space(input_[position_])) ++position_;
    }

    bool fail(std::string message) {
        error_ = std::move(message) + " at byte " + std::to_string(position_);
        return false;
    }

    bool hex4(std::uint32_t& value) {
        if (position_ + 4 > input_.size()) return false;
        value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else return false;
        }
        return true;
    }

    static void append_utf8(std::string& out, std::uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    bool parse_string(std::string& value) {
        if (position_ >= input_.size() || input_[position_++] != '"') return fail("expected JSON string");
        value.clear();
        while (position_ < input_.size()) {
            const unsigned char c = static_cast<unsigned char>(input_[position_++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("control byte in JSON string");
            if (c != '\\') {
                value.push_back(static_cast<char>(c));
                continue;
            }
            if (position_ == input_.size()) return fail("truncated JSON escape");
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = 0;
                    if (!hex4(codepoint)) return fail("invalid JSON unicode escape");
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                            input_[position_ + 1] != 'u') {
                            return fail("unpaired JSON high surrogate");
                        }
                        position_ += 2;
                        std::uint32_t low = 0;
                        if (!hex4(low) || low < 0xdc00 || low > 0xdfff) {
                            return fail("invalid JSON low surrogate");
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        return fail("unpaired JSON low surrogate");
                    }
                    append_utf8(value, codepoint);
                    break;
                }
                default: return fail("invalid JSON escape");
            }
        }
        return fail("unterminated JSON string");
    }

    bool parse_number(JsonValue& value) {
        const std::size_t start = position_;
        if (position_ < input_.size() && input_[position_] == '-') ++position_;
        if (position_ == input_.size()) return fail("truncated JSON number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
                return fail("leading zero in JSON number");
            }
        } else {
            if (input_[position_] < '1' || input_[position_] > '9') return fail("invalid JSON number");
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const std::size_t fraction = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (fraction == position_) return fail("empty JSON number fraction");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (exponent == position_) return fail("empty JSON number exponent");
        }
        value.kind = JsonKind::Number;
        value.scalar.assign(input_.substr(start, position_ - start));
        return true;
    }

    bool parse_value(JsonValue& value, unsigned depth) {
        if (depth > 64) return fail("JSON nesting exceeds 64 levels");
        skip_space();
        if (position_ == input_.size()) return fail("unexpected end of JSON");
        const char c = input_[position_];
        if (c == '"') {
            value.kind = JsonKind::String;
            return parse_string(value.scalar);
        }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(value);
        if (input_.substr(position_, 4) == "true") {
            position_ += 4;
            value.kind = JsonKind::Boolean;
            value.boolean = true;
            return true;
        }
        if (input_.substr(position_, 5) == "false") {
            position_ += 5;
            value.kind = JsonKind::Boolean;
            value.boolean = false;
            return true;
        }
        if (input_.substr(position_, 4) == "null") {
            position_ += 4;
            value.kind = JsonKind::Null;
            return true;
        }
        if (c == '[') {
            ++position_;
            value.kind = JsonKind::Array;
            skip_space();
            if (position_ < input_.size() && input_[position_] == ']') {
                ++position_;
                return true;
            }
            while (true) {
                JsonValue item;
                if (!parse_value(item, depth + 1)) return false;
                value.array.push_back(std::move(item));
                skip_space();
                if (position_ == input_.size()) return fail("unterminated JSON array");
                if (input_[position_] == ']') {
                    ++position_;
                    return true;
                }
                if (input_[position_++] != ',') return fail("expected comma in JSON array");
            }
        }
        if (c == '{') {
            ++position_;
            value.kind = JsonKind::Object;
            skip_space();
            if (position_ < input_.size() && input_[position_] == '}') {
                ++position_;
                return true;
            }
            while (true) {
                skip_space();
                std::string key;
                if (!parse_string(key)) return false;
                skip_space();
                if (position_ == input_.size() || input_[position_++] != ':') {
                    return fail("expected colon in JSON object");
                }
                JsonValue item;
                if (!parse_value(item, depth + 1)) return false;
                if (!value.object.emplace(std::move(key), std::move(item)).second) {
                    return fail("duplicate key in conversions.json");
                }
                skip_space();
                if (position_ == input_.size()) return fail("unterminated JSON object");
                if (input_[position_] == '}') {
                    ++position_;
                    return true;
                }
                if (input_[position_++] != ',') return fail("expected comma in JSON object");
            }
        }
        return fail("invalid JSON value");
    }

    std::string_view input_;
    std::size_t position_ = 0;
    std::string error_;
};

bool json_string(const JsonValue* value, std::string& out) {
    if (value == nullptr || value->kind != JsonKind::String) return false;
    out = value->scalar;
    return true;
}

bool json_u64(const JsonValue* value, std::uint64_t& out) {
    if (value == nullptr || value->kind != JsonKind::Number ||
        value->scalar.find_first_of(".-+eE") != std::string::npos) return false;
    return parse_u64(value->scalar, out);
}

bool json_bool(const JsonValue* value, bool& out) {
    if (value == nullptr || value->kind != JsonKind::Boolean) return false;
    out = value->boolean;
    return true;
}

bool json_finite_number(const JsonValue* value, double& out) {
    if (value == nullptr || value->kind != JsonKind::Number) return false;
    char* end = nullptr;
    const double parsed = std::strtod(value->scalar.c_str(), &end);
    if (end == value->scalar.c_str() || *end != '\0' || !std::isfinite(parsed)) return false;
    out = parsed;
    return true;
}

bool read_conversions(const fs::path& path, const strata::GgufModel& model,
                      std::map<std::string, Conversion>& conversions, std::string& error) {
    std::string text;
    if (!read_text(path, text, error)) return false;
    JsonValue root;
    if (!JsonParser(text).parse(root, error)) {
        error = path.string() + ": " + error;
        return false;
    }
    if (root.kind != JsonKind::Object) {
        error = path.string() + " must contain a JSON object";
        return false;
    }
    std::uint64_t schema = 0;
    bool compat = true;
    std::string tool;
    if (!json_u64(root.get("schema"), schema) || schema != 1 ||
        !json_string(root.get("tool"), tool) || tool != "tools/iq_pack.py" ||
        !json_bool(root.get("compat_bf16"), compat) || compat) {
        error = path.string() + " has an unsupported schema, tool or compat_bf16 setting";
        return false;
    }
    const JsonValue* sources = root.get("source_shards");
    if (sources == nullptr || sources->kind != JsonKind::Array || sources->array.size() != model.size()) {
        error = path.string() + " source_shards do not match the GGUF shard count";
        return false;
    }
    for (std::size_t i = 0; i < model.size(); ++i) {
        const JsonValue& source = sources->array[i];
        std::string name;
        std::uint64_t size = 0;
        if (source.kind != JsonKind::Object ||
            !json_string(source.get("name"), name) ||
            !json_u64(source.get("size"), size) ||
            name != fs::path(model.shard(i).path()).filename().string() ||
            size != model.shard(i).file_size()) {
            error = path.string() + " source_shards do not match the GGUF file names and sizes";
            return false;
        }
    }

    const JsonValue* rows = root.get("tensors");
    if (rows == nullptr || rows->kind != JsonKind::Array) {
        error = path.string() + " tensors must be an array";
        return false;
    }
    for (const JsonValue& row : rows->array) {
        std::string name, shard, src_type, dst_type, method, digest;
        bool exact = false;
        double max_error = 0;
        if (row.kind != JsonKind::Object ||
            !json_string(row.get("name"), name) ||
            !json_string(row.get("shard"), shard) ||
            !json_string(row.get("src_type"), src_type) ||
            !json_string(row.get("dst_type"), dst_type) ||
            !json_string(row.get("method"), method) || method.empty() ||
            !json_string(row.get("source_sha256"), digest) ||
            !json_bool(row.get("exact"), exact) ||
            !json_finite_number(row.get("max_abs_err"), max_error) || max_error < 0) {
            error = path.string() + " contains a malformed conversion record";
            return false;
        }
        if (digest.size() != 64 ||
            !std::all_of(digest.begin(), digest.end(), [](unsigned char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            })) {
            error = path.string() + " contains a malformed source_sha256";
            return false;
        }
        std::size_t shard_index = 0;
        const strata::TensorInfo* tensor = model.find(name, &shard_index);
        const std::string* required_type = required_converted_type(name);
        const bool router_conversion = name.ends_with(".ffn_gate_inp.weight") ||
                                       name.ends_with(".ffn_gate_inp_shexp.weight");
        if (tensor == nullptr || is_expert_tensor(name) || name == "per_layer_token_embd.weight" ||
            shard != fs::path(model.shard(shard_index).path()).filename().string() ||
            src_type != tensor->type_name() ||
            (dst_type != "BF16" && dst_type != "F16" && dst_type != "F32") ||
            required_type == nullptr || dst_type != *required_type || dst_type == src_type ||
            !router_conversion || src_type != "F32" || dst_type != "BF16" || !exact || max_error != 0 ||
            method != "f32->bf16: the values are BF16's (low 16 bits zero)") {
            error = path.string() + " conversion record does not match tensor " + name;
            return false;
        }
        if (!conversions.emplace(name, Conversion{shard, src_type, dst_type}).second) {
            error = path.string() + " contains duplicate conversion tensor " + name;
            return false;
        }
    }
    return true;
}

struct IndexRow {
    std::string name;
    // Numeric columns after name, in order: file, kind, src_off, src_bytes, dst_off, dst_bytes,
    // ne0, ne1, code_bits, code_bias, group_elems, codebook, has_offset, codes_bytes,
    // scales_bytes, offset_bytes, scales_fp16, act_kind.
    std::array<std::uint64_t, 18> values{};
};

struct IndexData {
    std::map<std::string, IndexRow> rows;
    std::uint64_t dense_pack_bytes = 0;
    std::uint64_t dense_file_size = 0;
    std::uint64_t dense_file_device_id = 0;
    std::uint64_t dense_file_object_id = 0;
};

bool split_fields(const std::string& line, std::vector<std::string>& fields) {
    fields.clear();
    std::istringstream input(line);
    std::string field;
    while (input >> field) fields.push_back(std::move(field));
    return !fields.empty();
}

bool read_index(const fs::path& pack, const strata::GgufModel& model,
                const std::map<std::string, Conversion>& conversions, IndexData& result,
                std::string& error) {
    const fs::path path = pack / "index.txt";
    std::ifstream input(path);
    if (!input) {
        error = "cannot open " + path.string();
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        error = path.string() + " is empty";
        return false;
    }
    const std::string expected_header =
        "# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from " +
        fs::path(model.shard(0).path()).filename().string();
    if (line != expected_header) {
        error = path.string() + " has an unsupported or mismatched v3 header";
        return false;
    }
    if (!std::getline(input, line)) {
        error = path.string() + " is missing its pool header";
        return false;
    }
    std::istringstream header(line);
    std::string hash, align_word, pool_word, tensors_word, trailing;
    std::uint64_t alignment = 0, declared_pool = 0, declared_tensors = 0;
    if (!(header >> hash >> align_word >> alignment >> pool_word >> declared_pool >> tensors_word >>
          declared_tensors) ||
        (header >> trailing) || hash != "#" || align_word != "align" || alignment != 64 ||
        pool_word != "pool" || tensors_word != "tensors") {
        error = path.string() + " has an invalid pool header";
        return false;
    }

    std::uint64_t destination_offset = 0;
    std::uint64_t source_offset = 0;
    std::size_t row_count = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::vector<std::string> fields;
        split_fields(line, fields);
        if (fields.size() != 19) {
            error = path.string() + ": an index row does not have 19 fields";
            return false;
        }
        IndexRow row;
        row.name = fields[0];
        for (std::size_t i = 0; i < row.values.size(); ++i) {
            if (!parse_u64(fields[i + 1], row.values[i])) {
                error = path.string() + ": invalid integer in row " + row.name;
                return false;
            }
        }
        const auto& v = row.values;
        if (v[0] != 0 || v[4] != destination_offset) {
            error = path.string() + ": unsupported file id or non-contiguous destination offset for " + row.name;
            return false;
        }
        std::size_t shard_index = 0;
        const strata::TensorInfo* tensor = model.find(row.name, &shard_index);
        if (tensor == nullptr || is_expert_tensor(row.name) || row.name == "per_layer_token_embd.weight" ||
            tensor->shape.empty() || tensor->shape.size() > 2 ||
            v[6] != tensor->shape[0] ||
            v[7] != (tensor->shape.size() > 1 ? tensor->shape[1] : 0)) {
            error = path.string() + ": row does not match a non-expert GGUF tensor: " + row.name;
            return false;
        }
        const bool source_float = tensor->type == 0 || tensor->type == 1 || tensor->type == 30;
        const auto conversion = conversions.find(row.name);
        const std::string* required_form = required_converted_type(row.name);
        const bool native = v[8] == 8;
        if (native) {
            const bool native_ple_key = row.name == "blk.1.ple_key.weight" &&
                                        (tensor->type == 42 || tensor->type == 8);
            if (v[1] != 0 || v[2] != 0 || v[3] != 0 || v[5] != 0 ||
                v[9] != 0 || v[10] != 32 || v[11] != 0 || v[12] != 0 ||
                v[13] != 0 || v[14] != 0 || v[15] != 0 || v[16] != 0 || v[17] != 0 ||
                source_float || conversion != conversions.end() ||
                (required_form != nullptr && !native_ple_key)) {
                error = path.string() + ": malformed native row for " + row.name;
                return false;
            }
        } else {
            const std::uint64_t elements = tensor->elements();
            std::uint64_t f32_bytes = 0, f16_bytes = 0;
            const std::uint64_t expected_float_kind =
                tensor->type == 0 ? 2 : tensor->type == 30 ? 4 : tensor->type == 1 ? 5 : 0;
            const std::string effective_type = conversion == conversions.end()
                ? tensor->type_name()
                : conversion->second.destination_type;
            if (!multiply(elements, 4, f32_bytes) || !multiply(elements, 2, f16_bytes) ||
                (v[1] < 2 || v[1] > 5) || v[8] != 0 || v[9] != 0 || v[10] != 1 ||
                v[11] != 0 || v[12] != 0 || v[13] != 0 || v[14] != 0 ||
                v[15] != 0 || v[16] != 0 || v[17] != 0 ||
                (conversion == conversions.end() && (!source_float || v[1] != expected_float_kind)) ||
                (conversion != conversions.end() && conversion->second.source_type != tensor->type_name()) ||
                (required_form != nullptr && *required_form != effective_type) ||
                (v[1] == 2 && (v[3] != f32_bytes || v[5] != f32_bytes)) ||
                (v[1] == 3 && (v[3] != f32_bytes || v[5] != f16_bytes)) ||
                (v[1] == 4 && (v[3] != f16_bytes || v[5] != f16_bytes)) ||
                (v[1] == 5 && (v[3] != f16_bytes || v[5] != f16_bytes)) ||
                v[2] != source_offset) {
                error = path.string() + ": malformed float row for " + row.name;
                return false;
            }
            std::uint64_t next_source = 0;
            if (!add(source_offset, v[3], next_source) || !align_up(next_source, 64, source_offset)) {
                error = path.string() + ": dense.bin source offset overflows";
                return false;
            }
        }
        std::uint64_t next_destination = 0;
        if (!add(destination_offset, v[5], next_destination) ||
            !align_up(next_destination, 64, destination_offset)) {
            error = path.string() + ": destination pool offset overflows";
            return false;
        }
        if (!result.rows.emplace(row.name, std::move(row)).second) {
            error = path.string() + ": duplicate index row";
            return false;
        }
        ++row_count;
    }
    if (!input.eof() || row_count != declared_tensors || destination_offset != declared_pool) {
        error = path.string() + " row count or pool size does not match its header";
        return false;
    }

    const fs::path dense_path = pack / "dense.bin";
    std::uint64_t dense_size = 0;
    if (!file_size(dense_path, dense_size, error)) return false;
    if (dense_size != source_offset) {
        error = dense_path.string() + " size does not match the indexed tensor spans";
        return false;
    }
    struct stat dense_status {};
    if (::stat(dense_path.c_str(), &dense_status) != 0 || dense_status.st_size < 0 ||
        static_cast<std::uint64_t>(dense_status.st_size) != dense_size) {
        error = dense_path.string() + " identity or size changed during inspection";
        return false;
    }
    result.dense_file_size = dense_size;
    result.dense_file_device_id = static_cast<std::uint64_t>(dense_status.st_dev);
    result.dense_file_object_id = static_cast<std::uint64_t>(dense_status.st_ino);
    for (const auto& [name, conversion] : conversions) {
        const auto found = result.rows.find(name);
        if (found == result.rows.end() || found->second.values[8] == 8) {
            error = path.string() + ": conversion has no float index row for " + name;
            return false;
        }
        const std::uint64_t kind = found->second.values[1];
        const std::string expected_type =
            kind == 4 ? "BF16" : kind == 5 || kind == 3 ? "F16" : kind == 2 ? "F32" : "";
        if (conversion.destination_type != expected_type ||
            found->second.values[5] == 0) {
            error = path.string() + ": conversion destination does not match index kind for " + name;
            return false;
        }
        if (!add(result.dense_pack_bytes, found->second.values[5], result.dense_pack_bytes)) {
            error = path.string() + ": converted tensor byte count overflows";
            return false;
        }
    }
    return true;
}

struct ExpertLayer {
    std::uint32_t gate_type = 0;
    std::uint32_t down_type = 0;
    std::uint64_t source_offset = 0;
    std::uint64_t blob_bytes = 0;
    std::array<std::uint64_t, 3> offsets{};
    std::array<std::string, 3> shard_names{};
};

struct ExpertTable {
    std::uint32_t version = 0;
    std::uint32_t n_expert = 0;
    std::uint64_t total_bytes = 0;
    std::vector<ExpertLayer> layers;
};

bool parse_header_integer(std::string_view line, std::string_view key, std::uint64_t& value) {
    const std::size_t found = line.find(key);
    if (found == std::string_view::npos) return false;
    std::size_t start = found + key.size();
    std::size_t end = start;
    while (end < line.size() && line[end] >= '0' && line[end] <= '9') ++end;
    if (end == start || !parse_u64(line.substr(start, end - start), value)) return false;
    return end == line.size() || line[end] == ',' || line[end] == ')' || line[end] == ';';
}

bool find_expert_role(const strata::GgufModel& model, std::uint32_t layer, std::string_view role,
                      const strata::TensorInfo*& tensor, std::size_t& shard_index,
                      std::string& error) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_" + std::string(role) + "_exps.weight";
    tensor = model.find(name, &shard_index);
    if (tensor == nullptr) {
        error = "native_experts.txt: missing GGUF tensor " + name;
        return false;
    }
    return true;
}

bool read_native_experts(const fs::path& path, const fs::path& model_source,
                         const strata::GgufModel& model, std::uint32_t n_layers,
                         std::uint32_t hidden, std::uint32_t feed_forward, ExpertTable& table,
                         std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open " + path.string();
        return false;
    }
    table = {};
    std::string line;
    bool saw_version = false;
    bool saw_expert_header = false;
    std::uint64_t declared_total = 0;
    std::map<std::uint32_t, ExpertLayer> rows;
    std::string source_name = model_source.filename().string();
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.front() == '#') {
            constexpr std::string_view tag = "# strata native experts v";
            if (line.starts_with(tag)) {
                if (saw_version) {
                    error = path.string() + " contains more than one version header";
                    return false;
                }
                std::size_t end = tag.size();
                while (end < line.size() && line[end] >= '0' && line[end] <= '9') ++end;
                std::uint64_t version = 0;
                if (end == tag.size() || !parse_u64(std::string_view(line).substr(tag.size(), end - tag.size()),
                                                     version) ||
                    version > std::numeric_limits<std::uint32_t>::max()) {
                    error = path.string() + " has an invalid version header";
                    return false;
                }
                table.version = static_cast<std::uint32_t>(version);
                saw_version = true;
                if (version > 4) {
                    error = "native_experts.txt is v" + std::to_string(version) +
                            "; this engine reads up to v4 (the pack was written by a newer tools/iq_pack.py: "
                            "update the engine, or repack with this one)";
                    return false;
                }
                if (version != 3 && version != 4) {
                    error = "native_experts.txt must be v3 or v4";
                    return false;
                }
                if (line.find("absolute offsets in " + source_name + ",") == std::string::npos) {
                    error = path.string() + " header does not name the selected source shard";
                    return false;
                }
            }
            if (line.find("(n_expert ") != std::string::npos) {
                if (saw_expert_header) {
                    error = path.string() + " contains more than one expert-count header";
                    return false;
                }
                std::uint64_t experts = 0;
                if (!parse_header_integer(line, "(n_expert ", experts) ||
                    !parse_header_integer(line, "total ", declared_total) ||
                    experts == 0 || experts > std::numeric_limits<std::uint32_t>::max()) {
                    error = path.string() + " has an invalid expert-count or total header";
                    return false;
                }
                table.n_expert = static_cast<std::uint32_t>(experts);
                saw_expert_header = true;
            }
            continue;
        }

        if (!saw_version || !saw_expert_header) {
            error = path.string() + " has layer rows before its required header";
            return false;
        }
        std::vector<std::string> fields;
        split_fields(line, fields);
        if (fields.size() < 8 || fields.size() > 9) {
            error = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        std::array<std::uint64_t, 8> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!parse_u64(fields[i], values[i])) {
                error = "native_experts.txt: a malformed line: " + line;
                return false;
            }
        }
        if (values[0] >= n_layers || values[1] > std::numeric_limits<std::uint32_t>::max() ||
            values[2] > std::numeric_limits<std::uint32_t>::max()) {
            error = "native_experts.txt: a malformed layer or format: " + line;
            return false;
        }
        const std::uint32_t layer = static_cast<std::uint32_t>(values[0]);
        const std::uint32_t gate_type = static_cast<std::uint32_t>(values[1]);
        const std::uint32_t down_type = static_cast<std::uint32_t>(values[2]);

        std::array<std::string, 3> shard_names{source_name, source_name, source_name};
        if (fields.size() == 9) {
            const std::string& column = fields[8];
            const std::size_t first_comma = column.find(',');
            if (first_comma == std::string::npos) {
                if (!safe_atom(column) || fs::path(column).filename() != column) {
                    error = "native_experts.txt: invalid shard name in layer " + std::to_string(layer);
                    return false;
                }
                shard_names.fill(column);
            } else {
                if (table.version != 4) {
                    error = "native_experts.txt v3 cannot name per-role shards";
                    return false;
                }
                std::size_t start = 0;
                for (std::size_t role = 0; role < 3; ++role) {
                    const std::size_t comma = column.find(',', start);
                    if (role == 2 && comma != std::string::npos) {
                        error = "native_experts.txt: layer " + std::to_string(layer) +
                                " has more than three role shard names";
                        return false;
                    }
                    const std::size_t end = comma == std::string::npos ? column.size() : comma;
                    const std::string part = column.substr(start, end - start);
                    if (!part.empty()) {
                        if (!safe_atom(part) || fs::path(part).filename() != part) {
                            error = "native_experts.txt: invalid per-role shard in layer " +
                                    std::to_string(layer);
                            return false;
                        }
                        shard_names[role] = part;
                    }
                    if (role < 2 && comma == std::string::npos) {
                        error = "native_experts.txt: layer " + std::to_string(layer) +
                                " needs three per-role shard names";
                        return false;
                    }
                    start = comma == std::string::npos ? column.size() : comma + 1;
                }
            }
        }

        ExpertLayer row;
        row.gate_type = gate_type;
        row.down_type = down_type;
        row.source_offset = values[3];
        row.blob_bytes = values[4];
        row.offsets = {values[5], values[6], values[7]};
        row.shard_names = shard_names;
        if (!rows.emplace(layer, std::move(row)).second) {
            error = "native_experts.txt: duplicate layer " + std::to_string(layer);
            return false;
        }
    }
    if (!input.eof() || !saw_version || !saw_expert_header || rows.size() != n_layers ||
        table.n_expert == 0 || table.n_expert > 512) {
        error = path.string() + " is truncated or its dimensions do not match the model";
        return false;
    }

    std::uint64_t expected_offset = 0;
    table.layers.resize(n_layers);
    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        const auto row_it = rows.find(layer);
        if (row_it == rows.end()) {
            error = "native_experts.txt: layer " + std::to_string(layer) + " is missing or not contiguous";
            return false;
        }
        const ExpertLayer& row = row_it->second;

        const std::array<std::string_view, 3> roles = {"gate", "up", "down"};
        std::array<std::uint64_t, 3> slabs{};
        std::array<std::uint32_t, 3> actual_types{};
        for (std::size_t role = 0; role < roles.size(); ++role) {
            const strata::TensorInfo* tensor = nullptr;
            std::size_t shard_index = 0;
            if (!find_expert_role(model, layer, roles[role], tensor, shard_index, error)) return false;
            const std::uint64_t dim0 = role == 2 ? feed_forward : hidden;
            const std::uint64_t dim1 = role == 2 ? hidden : feed_forward;
            if (!tensor_shape(*tensor, dim0, dim1, table.n_expert)) {
                error = "native_experts.txt: invalid shape for blk." + std::to_string(layer) +
                        ".ffn_" + std::string(roles[role]) + "_exps.weight";
                return false;
            }
            const std::uint64_t bytes = strata::tensor_payload_bytes(*tensor);
            if (bytes == 0 || bytes % table.n_expert != 0 ||
                row.shard_names[role] != fs::path(model.shard(shard_index).path()).filename().string()) {
                error = "native_experts.txt: invalid shard or byte size for layer " + std::to_string(layer);
                return false;
            }
            std::uint64_t absolute = 0;
            if (!add(model.shard(shard_index).data_start(), tensor->offset, absolute) ||
                absolute != row.offsets[role] || !model.in_bounds(*tensor, shard_index)) {
                error = "native_experts.txt: layer " + std::to_string(layer) +
                        " role offset does not match its GGUF tensor";
                return false;
            }
            slabs[role] = bytes / table.n_expert;
            actual_types[role] = tensor->type;
        }
        std::uint64_t two_gate = 0;
        std::uint64_t blob = 0;
        std::uint64_t layer_bytes = 0;
        if (actual_types[0] != actual_types[1] || actual_types[0] != row.gate_type ||
            actual_types[2] != row.down_type ||
            !multiply(slabs[0], 2, two_gate) || !add(two_gate, slabs[2], blob) ||
            blob != row.blob_bytes || !multiply(blob, table.n_expert, layer_bytes)) {
            error = "native_experts.txt: layer " + std::to_string(layer) +
                    " blob or role types do not match the GGUF";
            return false;
        }

        if (row.source_offset != expected_offset) {
            error = "native_experts.txt: layer " + std::to_string(layer) + " is missing or not contiguous";
            return false;
        }
        if (!add(expected_offset, layer_bytes, expected_offset)) {
            error = "native_experts.txt expert offsets overflow";
            return false;
        }
        table.layers[layer] = row;
    }
    if (expected_offset != declared_total) {
        error = path.string() + " total byte count does not match the layer table";
        return false;
    }
    table.total_bytes = expected_offset;
    return true;
}

struct ByteRange {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

bool ranges_overlap(const std::vector<ByteRange>& ranges, std::string& error) {
    auto sorted = ranges;
    std::sort(sorted.begin(), sorted.end(), [](const ByteRange& left, const ByteRange& right) {
        return left.start < right.start;
    });
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i].start < sorted[i - 1].end) {
            error = "GGUF tensor payload ranges overlap";
            return true;
        }
    }
    return false;
}

bool collect_window_sources(const strata::GgufModel& model,
                           const std::map<std::string, Conversion>& conversions,
                           const IndexData& index, const fs::path& pack,
                           std::uint32_t n_expert, WindowSources& sources,
                           std::string& error) {
    sources = {};
    sources.dense_pack_path = (pack / "dense.bin").string();
    sources.dense_pack_file_size = index.dense_file_size;
    sources.dense_pack_device_id = index.dense_file_device_id;
    sources.dense_pack_object_id = index.dense_file_object_id;
    for (const auto& [name, conversion] : conversions) {
        (void)conversion;
        const auto found = index.rows.find(name);
        if (found == index.rows.end()) {
            error = "index.txt is missing converted tensor " + name;
            return false;
        }
        const auto& values = found->second.values;
        sources.dense_pack_tensors.push_back(
            {name, values[2], values[3], values[5],
             static_cast<std::uint32_t>(values[1])});
    }
    for (std::size_t shard = 0; shard < model.size(); ++shard) {
        const strata::GgufFile& file = model.shard(shard);
        WindowFileSource source;
        source.path = file.path();
        source.file_size = file.file_size();
        source.file_device_id = file.file_device_id();
        source.file_object_id = file.file_object_id();
        for (const strata::TensorInfo& tensor : file.tensors()) {
            const std::uint64_t payload = strata::tensor_payload_bytes(tensor);
            std::uint64_t start = 0;
            std::uint64_t end = 0;
            if (payload == 0 || !add(file.data_start(), tensor.offset, start) ||
                !add(start, payload, end) || end > file.file_size()) {
                error = file.path() + ": invalid or truncated tensor " + tensor.name;
                return false;
            }
            if (tensor.name == "per_layer_token_embd.weight") {
                source.excluded_tensors.push_back(
                    {tensor.name, start, payload, WindowTensorExclusionReason::ple});
                continue;
            }
            if (conversions.contains(tensor.name)) {
                source.excluded_tensors.push_back(
                    {tensor.name, start, payload, WindowTensorExclusionReason::converted});
                continue;
            }
            std::uint64_t expert_slab_bytes = 0;
            if (is_expert_tensor(tensor.name)) {
                if (tensor.shape.size() != 3 || tensor.shape[2] != n_expert ||
                    n_expert == 0 || payload % n_expert != 0) {
                    error = file.path() + ": invalid expert slab layout for " + tensor.name;
                    return false;
                }
                expert_slab_bytes = payload / n_expert;
            }
            source.tensors.push_back({tensor.name, start, payload, expert_slab_bytes});
        }
        sources.files.push_back(std::move(source));
    }
    return true;
}

bool read_mtp_runtime(const fs::path& directory, std::uint64_t& mtp_bytes,
                      std::uint64_t& bind_bytes, std::string& error,
                      MtpRuntimeSources* output_sources) {
    if (output_sources != nullptr) *output_sources = {};
    if (directory.empty()) {
        mtp_bytes = 0;
        bind_bytes = 0;
        return true;
    }
    std::error_code ec;
    if (!fs::is_directory(directory, ec) || ec) {
        error = "cannot open MTP runtime directory " + directory.string();
        return false;
    }
    const fs::path experts_path = directory / "experts.bin";
    const fs::path dense_path = directory / "dense.bin";
    const fs::path vocab_path = directory / "draft_vocab.bin";
    const fs::path index_path = directory / "dense.txt";
    MtpFileIdentity experts_identity;
    MtpFileIdentity dense_identity;
    if (!inspect_mtp_file(experts_path, experts_identity, error) ||
        !inspect_mtp_file(dense_path, dense_identity, error)) return false;
    const std::uint64_t expert_size = experts_identity.size;
    const std::uint64_t dense_size = dense_identity.size;
    if (expert_size != kMtpExpertBytes || dense_size != kMtpDenseBytes ||
        dense_size == 0) {
        error = "MTP runtime file sizes do not match the supported runtime format";
        return false;
    }
    MtpRuntimeCatalog candidate;
    candidate.directory = directory.string();
    candidate.experts_path = experts_path.string();
    candidate.experts_file_size = expert_size;
    candidate.experts_file_device_id = experts_identity.device_id;
    candidate.experts_file_object_id = experts_identity.object_id;
    candidate.dense_path = dense_path.string();
    candidate.dense_file_size = dense_size;
    candidate.dense_file_device_id = dense_identity.device_id;
    candidate.dense_file_object_id = dense_identity.object_id;
    candidate.draft_vocab_path = vocab_path.string();
    std::uint64_t vocab_size = 0;
    if (!read_mtp_draft_vocabulary(vocab_path, vocab_size,
                                   candidate.draft_vocab_ids, error)) return false;
    candidate.draft_vocab_file_size = vocab_size;
    std::string text;
    if (!read_text(index_path, text, error, 1024 * 1024)) return false;
    struct ExpectedTensor {
        const char* kind;
        std::uint64_t rows;
        std::uint64_t columns;
    };
    static const std::map<std::string, ExpectedTensor> expected_tensors = {
        {"fc_embedding.weight", {"q8_0", 2560, 2560}},
        {"fc_hidden.weight", {"q8_0", 2560, 2560}},
        {"self_attn.q_proj.weight", {"q8_0", 12'288, 2560}},
        {"self_attn.k_proj.weight", {"q8_0", 512, 2560}},
        {"self_attn.v_proj.weight", {"q8_0", 512, 2560}},
        {"self_attn.o_proj.weight", {"q8_0", 2560, 6144}},
        {"self_attn.indexer.index_qk_proj.weight", {"q8_0", 640, 2560}},
        {"mlp.shared_expert.gate_proj.weight", {"q8_0", 640, 2560}},
        {"mlp.shared_expert.up_proj.weight", {"q8_0", 640, 2560}},
        {"mlp.shared_expert.down_proj.weight", {"q8_0", 2560, 640}},
        {"hyper_connection_mixer.hc_norm.weight", {"f32", 1, 10'240}},
        {"hyper_connection_mixer.input_mix_weight_down.weight", {"bf16", 320, 10'240}},
        {"hyper_connection_mixer.input_mix_weight_up.weight", {"bf16", 10'240, 320}},
        {"attn_hyper_connection.hc_norm.weight", {"f32", 1, 10'240}},
        {"attn_hyper_connection.input_mix_weight_down.weight", {"bf16", 320, 10'240}},
        {"attn_hyper_connection.input_mix_weight_up.weight", {"bf16", 10'240, 320}},
        {"attn_hyper_connection.block_inject_weight.weight", {"bf16", 4, 10'240}},
        {"mlp_hyper_connection.hc_norm.weight", {"f32", 1, 10'240}},
        {"mlp_hyper_connection.input_mix_weight_down.weight", {"bf16", 320, 10'240}},
        {"mlp_hyper_connection.input_mix_weight_up.weight", {"bf16", 10'240, 320}},
        {"mlp_hyper_connection.block_inject_weight.weight", {"bf16", 4, 10'240}},
        {"mlp.gate.weight", {"bf16", 512, 2560}},
        {"mlp.shared_expert_gate.weight", {"bf16", 1, 2560}},
        {"pre_fc_norm_embedding.weight", {"f32", 1, 2560}},
        {"pre_fc_norm_hidden.weight", {"f32", 1, 10'240}},
        {"self_attn.q_norm.weight", {"f32", 1, 256}},
        {"self_attn.k_norm.weight", {"f32", 1, 256}},
        {"self_attn.indexer.q_layernorm.weight", {"f32", 1, 128}},
        {"self_attn.indexer.k_layernorm.weight", {"f32", 1, 128}},
    };
    std::set<std::string> seen;
    std::vector<ByteRange> ranges;
    std::istringstream lines(text);
    std::string line;
    std::uint64_t expected_offset = 0;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        split_fields(line, fields);
        if (fields.size() != 6) {
            error = index_path.string() + " has a malformed dense tensor row";
            return false;
        }
        std::uint64_t rows = 0, columns = 0, offset = 0, bytes = 0;
        const auto expected_tensor = expected_tensors.find(fields[0]);
        if (!parse_u64(fields[2], rows) || !parse_u64(fields[3], columns) ||
            !parse_u64(fields[4], offset) || !parse_u64(fields[5], bytes) ||
            rows == 0 || columns == 0 || bytes == 0 || offset % 256 != 0 ||
            offset < expected_offset || !seen.insert(fields[0]).second ||
            expected_tensor == expected_tensors.end() || fields[1] != expected_tensor->second.kind ||
            rows != expected_tensor->second.rows || columns != expected_tensor->second.columns) {
            error = index_path.string() + " has an invalid or duplicate tensor row";
            return false;
        }
        std::uint64_t elements = 0, expected_bytes = 0;
        if (!multiply(rows, columns, elements)) {
            error = index_path.string() + " tensor element count overflows";
            return false;
        }
        if (fields[1] == "q8_0") {
            std::uint64_t row_bytes = 0;
            if (columns % 32 != 0 || !multiply(columns / 32, 34, row_bytes) ||
                !multiply(rows, row_bytes, expected_bytes)) {
                error = index_path.string() + " has an invalid q8_0 tensor shape";
                return false;
            }
        } else if (fields[1] == "bf16" || fields[1] == "f16") {
            if (!multiply(elements, 2, expected_bytes)) {
                error = index_path.string() + " tensor byte count overflows";
                return false;
            }
        } else if (fields[1] == "f32") {
            if (!multiply(elements, 4, expected_bytes)) {
                error = index_path.string() + " tensor byte count overflows";
                return false;
            }
        } else {
            error = index_path.string() + " has an unsupported tensor kind";
            return false;
        }
        std::uint64_t end = 0;
        if (bytes != expected_bytes || !add(offset, bytes, end) || end > dense_size) {
            error = index_path.string() + " tensor span does not match dense.bin";
            return false;
        }
        ranges.push_back({offset, end});
        MtpDenseWeightKind kind = MtpDenseWeightKind::bf16;
        if (fields[1] == "q8_0") kind = MtpDenseWeightKind::q8_0;
        else if (fields[1] == "f32") kind = MtpDenseWeightKind::f32;
        candidate.dense_tensors.push_back(
            {fields[0], kind, rows, columns, offset, bytes});
        if (!align_up(end, 256, expected_offset)) {
            error = index_path.string() + " tensor offset overflows";
            return false;
        }
    }
    std::set<std::string> expected_names;
    for (const auto& [name, shape] : expected_tensors) {
        (void)shape;
        expected_names.insert(name);
    }
    if (!lines.eof() || seen != expected_names || expected_offset != dense_size) {
        error = index_path.string() + " is missing required tensors or does not match dense.bin";
        return false;
    }
    std::sort(ranges.begin(), ranges.end(), [](const ByteRange& left, const ByteRange& right) {
        return left.start < right.start;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].start < ranges[i - 1].end) {
            error = index_path.string() + " tensor spans overlap";
            return false;
        }
    }
    if (!add(expert_size, dense_size, mtp_bytes)) {
        error = "MTP runtime total size overflows";
        return false;
    }
    MtpBindLayout bind_layout;
    if (!make_mtp_bind_layout(
            kMtpDefaultBindCapacity,
            static_cast<std::uint32_t>(candidate.draft_vocab_ids.size()),
            bind_layout, error)) {
        error = "MTP bind layout is invalid: " + error;
        return false;
    }
    bind_bytes = bind_layout.allocation_bytes;
    if (output_sources != nullptr) {
        output_sources->catalog =
            std::make_shared<const MtpRuntimeCatalog>(std::move(candidate));
    }
    return true;
}

std::map<std::string, std::vector<std::uint64_t>> qwen_tensor_inventory(std::uint64_t n_expert) {
    std::map<std::string, std::vector<std::uint64_t>> expected;
    auto add = [&expected](const std::string& name, std::initializer_list<std::uint64_t> shape) {
        expected.emplace(name, std::vector<std::uint64_t>(shape));
    };

    add("token_embd.weight", {kHidden, 248'320});
    add("output.weight", {kHidden, 248'320});
    add("output_hc_norm.weight", {10'240});
    add("output_hc_down.weight", {10'240, 320});
    add("output_hc_up.weight", {320, 10'240});
    add("per_layer_token_embd.weight", {160, 320'001'536});

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add(prefix + "hc_attn_norm.weight", {10'240});
        add(prefix + "hc_ffn_norm.weight", {10'240});
        add(prefix + "hc_attn_down.weight", {10'240, 320});
        add(prefix + "hc_ffn_down.weight", {10'240, 320});
        add(prefix + "hc_attn_up.weight", {320, 10'240});
        add(prefix + "hc_ffn_up.weight", {320, 10'240});
        add(prefix + "hc_attn_inject.weight", {10'240, 4});
        add(prefix + "hc_ffn_inject.weight", {10'240, 4});
        add(prefix + "ffn_gate_inp.weight", {kHidden, n_expert});
        add(prefix + "ffn_gate_inp_shexp.weight", {kHidden});
        add(prefix + "ffn_gate_shexp.weight", {kHidden, kFeedForward});
        add(prefix + "ffn_up_shexp.weight", {kHidden, kFeedForward});
        add(prefix + "ffn_down_shexp.weight", {kFeedForward, kHidden});
        add(prefix + "ffn_gate_exps.weight", {kHidden, kFeedForward, n_expert});
        add(prefix + "ffn_up_exps.weight", {kHidden, kFeedForward, n_expert});
        add(prefix + "ffn_down_exps.weight", {kFeedForward, kHidden, n_expert});

        if (layer % 4 == 3) {
            add(prefix + "attn_q.weight", {kHidden, 12'288});
            add(prefix + "attn_k.weight", {kHidden, 512});
            add(prefix + "attn_v.weight", {kHidden, 512});
            add(prefix + "attn_output.weight", {6'144, kHidden});
            add(prefix + "attn_q_norm.weight", {256});
            add(prefix + "attn_k_norm.weight", {256});
            add(prefix + "indexer.q_proj.weight", {kHidden, 512});
            add(prefix + "indexer.k_proj.weight", {kHidden, 128});
            add(prefix + "indexer.q_norm.weight", {128});
            add(prefix + "indexer.k_norm.weight", {128});
        } else {
            add(prefix + "attn_qkv.weight", {kHidden, 10'240});
            add(prefix + "attn_gate.weight", {kHidden, 6'144});
            add(prefix + "ssm_out.weight", {6'144, kHidden});
            add(prefix + "ssm_conv1d.weight", {4, 10'240});
            add(prefix + "ssm_alpha.weight", {kHidden, 48});
            add(prefix + "ssm_beta.weight", {kHidden, 48});
            add(prefix + "ssm_a", {48});
            add(prefix + "ssm_dt.bias", {48});
            add(prefix + "ssm_norm.weight", {128});
        }

        if (layer == 1) {
            add(prefix + "ple_key.weight", {kHidden, 10'240});
            add(prefix + "ple_value.weight", {kHidden, kHidden});
            add(prefix + "ple_norm_key.weight", {10'240});
            add(prefix + "ple_norm_query.weight", {10'240});
            add(prefix + "ple_norm_conv.weight", {10'240});
            add(prefix + "ple_conv1d.weight", {4, 10'240});
        }
    }
    return expected;
}

bool validate_qwen_tensor_type(const strata::TensorInfo& tensor, std::string& error) {
    using Types = std::set<std::uint32_t>;
    static const std::map<std::string, Types> allowed = {
        {"token_embd.weight", {11, 21, 23}},
        {"output.weight", {13, 14, 23}},
        {"output_hc_norm.weight", {0}},
        {"output_hc_down.weight", {30}},
        {"output_hc_up.weight", {30}},
        {"per_layer_token_embd.weight", {20}},
        {"hc_attn_norm.weight", {0}},
        {"hc_ffn_norm.weight", {0}},
        {"hc_attn_down.weight", {30}},
        {"hc_ffn_down.weight", {30}},
        {"hc_attn_up.weight", {30}},
        {"hc_ffn_up.weight", {30}},
        {"hc_attn_inject.weight", {30}},
        {"hc_ffn_inject.weight", {30}},
        {"ffn_gate_inp.weight", {0, 30}},
        {"ffn_gate_inp_shexp.weight", {0, 30}},
        {"ffn_gate_shexp.weight", {11, 12, 13, 14, 21, 23, 42}},
        {"ffn_up_shexp.weight", {11, 12, 13, 14, 21, 23, 42}},
        {"ffn_down_shexp.weight", {2, 6, 8, 20, 42}},
        {"ffn_gate_exps.weight", {16, 17, 18, 21, 22, 23, 29, 42}},
        {"ffn_up_exps.weight", {16, 17, 18, 21, 22, 23, 29, 42}},
        {"ffn_down_exps.weight", {20, 42}},
        {"attn_qkv.weight", {11, 12, 13, 14, 21, 23, 42}},
        {"attn_gate.weight", {11, 12, 13, 14, 21, 23, 42}},
        {"ssm_out.weight", {11, 12, 13, 14, 21, 23}},
        {"ssm_conv1d.weight", {0}},
        {"ssm_alpha.weight", {30}},
        {"ssm_beta.weight", {30}},
        {"ssm_a", {0}},
        {"ssm_dt.bias", {0}},
        {"ssm_norm.weight", {0}},
        {"attn_q.weight", {11, 12, 13, 14, 21, 23, 42}},
        {"attn_k.weight", {11, 12, 13, 14, 21, 23}},
        {"attn_v.weight", {11, 12, 13, 14, 23}},
        {"attn_output.weight", {12, 13, 14, 21, 23}},
        {"attn_q_norm.weight", {0}},
        {"attn_k_norm.weight", {0}},
        {"indexer.q_proj.weight", {30}},
        {"indexer.k_proj.weight", {30}},
        {"indexer.q_norm.weight", {0}},
        {"indexer.k_norm.weight", {0}},
        {"ple_key.weight", {8, 30, 42}},
        {"ple_value.weight", {30}},
        {"ple_norm_key.weight", {0}},
        {"ple_norm_query.weight", {0}},
        {"ple_norm_conv.weight", {0}},
        {"ple_conv1d.weight", {1}},
    };
    std::string_view role = tensor.name;
    if (role.starts_with("blk.")) {
        const std::size_t layer_end = role.find('.', 4);
        if (layer_end == std::string_view::npos) {
            error = "invalid Qwen tensor name " + tensor.name;
            return false;
        }
        role.remove_prefix(layer_end + 1);
    }
    const auto found = allowed.find(std::string(role));
    if (found == allowed.end() || !found->second.contains(tensor.type)) {
        error = "GGUF tensor " + tensor.name + " uses disallowed type " + type_name(tensor.type);
        return false;
    }
    return true;
}

bool validate_qwen_tensor_inventory(const strata::GgufModel& model, std::uint64_t n_expert,
                                    std::string& error) {
    const auto expected = qwen_tensor_inventory(n_expert);
    if (expected.size() != 1'224) {
        error = "internal Qwen tensor inventory does not contain 1,224 entries";
        return false;
    }

    std::set<std::string> seen;
    std::size_t tensor_count = 0;
    for (std::size_t shard = 0; shard < model.size(); ++shard) {
        for (const strata::TensorInfo& tensor : model.shard(shard).tensors()) {
            ++tensor_count;
            const auto found = expected.find(tensor.name);
            if (found == expected.end()) {
                error = "GGUF contains an unexpected Qwen tensor " + tensor.name;
                return false;
            }
            if (!seen.insert(tensor.name).second) {
                error = "GGUF contains a duplicate tensor " + tensor.name;
                return false;
            }
            if (tensor.name.ends_with("ffn_gate_inp.weight") && tensor.shape.size() == 2 &&
                tensor.shape[1] != n_expert) {
                error = "qwen4exp.expert_count does not match " + tensor.name + " ne1";
                return false;
            }
            if (!validate_qwen_tensor_type(tensor, error)) return false;
            int block_elements = 0;
            int block_bytes = 0;
            if (tensor.shape.empty() ||
                !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                tensor.shape[0] % static_cast<std::uint64_t>(block_elements) != 0 ||
                strata::tensor_payload_bytes(tensor) == 0) {
                error = "GGUF tensor " + tensor.name + " has invalid quantization block geometry";
                return false;
            }
            if (tensor.shape != found->second) {
                error = "GGUF tensor has an unexpected shape: " + tensor.name;
                return false;
            }
        }
    }
    if (tensor_count != expected.size() || seen.size() != expected.size()) {
        for (const auto& [name, shape] : expected) {
            (void)shape;
            if (!seen.contains(name)) {
                error = "GGUF is missing required Qwen tensor " + name;
                return false;
            }
        }
        error = "GGUF tensor count does not match the 1,224-tensor Qwen architecture";
        return false;
    }
    return true;
}

bool metadata_facts(const strata::GgufFile& metadata, const ExpertTable& experts,
                    Geometry& geometry, std::string& error) {
    const std::string architecture_error = strata::check_architecture(metadata);
    if (!architecture_error.empty()) {
        error = architecture_error;
        return false;
    }
    std::uint64_t n_layers = 0, n_expert_metadata = 0, experts_used = 0, embedding = 0, kv_heads = 0;
    std::uint64_t head_dim = 0, vocab = 0, expert_ff = 0;
    if (!meta_u64(metadata, "qwen4exp.block_count", n_layers, error) ||
        !meta_u64(metadata, "qwen4exp.expert_count", n_expert_metadata, error) ||
        !meta_u64(metadata, "qwen4exp.expert_used_count", experts_used, error) ||
        !meta_u64(metadata, "qwen4exp.embedding_length", embedding, error) ||
        !meta_u64(metadata, "qwen4exp.attention.head_count_kv", kv_heads, error) ||
        !meta_u64(metadata, "qwen4exp.attention.key_length", head_dim, error) ||
        !meta_u64(metadata, "qwen4exp.expert_feed_forward_length", expert_ff, error)) {
        return false;
    }
    const strata::MetaValue* tokens = metadata.get("tokenizer.ggml.tokens");
    if (tokens == nullptr || tokens->type != strata::MetaType::ARRAY ||
        tokens->elem != strata::MetaType::STRING) {
        error = "missing or invalid tokenizer.ggml.tokens";
        return false;
    }
    vocab = tokens->count;
    std::uint64_t shared_expert_ff = 0;
    if (!meta_u64(metadata, "qwen4exp.expert_shared_feed_forward_length", shared_expert_ff, error)) {
        return false;
    }
    if (n_layers != kLayers || embedding != kHidden || expert_ff != kFeedForward ||
        shared_expert_ff != kFeedForward || experts_used != 10 ||
        n_layers > std::numeric_limits<std::uint32_t>::max() ||
        n_expert_metadata != experts.n_expert || experts.n_expert == 0 || experts.n_expert > 512 ||
        embedding > std::numeric_limits<std::uint32_t>::max() ||
        kv_heads > std::numeric_limits<std::uint32_t>::max() ||
        head_dim > std::numeric_limits<std::uint32_t>::max() ||
        kv_heads != 2 || head_dim != 256 || vocab != 248'320) {
        error = "GGUF model dimensions do not match the native expert table or supported architecture";
        return false;
    }
    geometry.n_layers = static_cast<std::uint32_t>(n_layers);
    geometry.n_expert = experts.n_expert;
    geometry.n_kv_head = static_cast<std::uint32_t>(kv_heads);
    geometry.head_dim = static_cast<std::uint32_t>(head_dim);
    geometry.n_embd = static_cast<std::uint32_t>(embedding);
    geometry.vocab = static_cast<std::uint32_t>(vocab);
    for (std::uint32_t layer = 0; layer < geometry.n_layers; ++layer) {
        if (layer % 4 == 3) ++geometry.n_qsa_layers;
        geometry.layers.push_back({experts.layers[layer].blob_bytes,
                                   type_name(experts.layers[layer].gate_type),
                                   type_name(experts.layers[layer].down_type)});
    }
    return true;
}

}  // namespace

bool validate_model_tensor_type(std::string_view tensor_name, std::uint32_t tensor_type,
                                std::string& error) {
    strata::TensorInfo tensor;
    tensor.name = std::string(tensor_name);
    tensor.type = tensor_type;
    return validate_qwen_tensor_type(tensor, error);
}

bool WeightStore::acquire_inspection_lock(const std::string& path,
                                          WindowSourceFileLock& result,
                                          std::string& error, bool create) {
    result = {};
    auto lease = WindowSourceLockLease::acquire_shared(path, create, error);
    if (!lease) return false;
    result.path = lease->path();
    result.file_size = lease->file_size();
    result.file_device_id = lease->file_device_id();
    result.file_object_id = lease->file_object_id();
    result.lease_ = std::move(lease);
    error.clear();
    return true;
}

bool WeightStore::inspect(const WeightStoreOptions& options, Geometry& geometry, std::string& error,
                          WindowSources* window_sources, MtpRuntimeSources* mtp_sources,
                          std::string* warning) {
    error.clear();
    geometry = {};
    if (window_sources != nullptr) *window_sources = {};
    if (mtp_sources != nullptr) *mtp_sources = {};
    if (warning != nullptr) warning->clear();
    try {
        MtpRuntimeSources inspected_mtp_sources;
        if (options.native_shard.empty()) {
            error = "the Metal engine needs --native <shard 1 of the GGUF>";
            return false;
        }
        const fs::path source_path(options.native_shard);
        const fs::path pack_path(options.pack_dir);
        const std::vector<std::string> shard_paths =
            strata::gguf_split_paths(source_path.string());
        std::vector<WindowSourceFileLock> inspection_locks;
        inspection_locks.reserve(shard_paths.size() + 4);
        fs::path source_set_lock_path;
        fs::path pack_set_lock_path;
        const fs::path source_directory = source_path.has_parent_path()
            ? source_path.parent_path()
            : fs::current_path();
        if (!artifact_set_lock_path(source_directory, source_set_lock_path, error) ||
            !artifact_set_lock_path(pack_path, pack_set_lock_path, error)) {
            return false;
        }
        WindowSourceFileLock source_set_lock;
        if (!acquire_inspection_lock(source_set_lock_path.string(), source_set_lock,
                                     error, true)) {
            return false;
        }
        inspection_locks.push_back(std::move(source_set_lock));
        if (pack_set_lock_path != source_set_lock_path) {
            WindowSourceFileLock pack_set_lock;
            if (!acquire_inspection_lock(pack_set_lock_path.string(), pack_set_lock,
                                         error, true)) {
                return false;
            }
            inspection_locks.push_back(std::move(pack_set_lock));
        }
        const fs::path pending_marker = pack_path / ".strata-generation-pending";
        std::error_code pending_error;
        const bool pending = fs::exists(pending_marker, pending_error);
        if (pending_error) {
            error = "cannot inspect artifact generation marker " + pending_marker.string() +
                    ": " + pending_error.message();
            return false;
        }
        if (pending) {
            error = "artifact generation is incomplete: " + pending_marker.string();
            return false;
        }
        if (window_sources != nullptr) {
            for (const std::string& shard_path : shard_paths) {
                WindowSourceFileLock lock;
                if (!acquire_inspection_lock(shard_path, lock, error, false)) return false;
                inspection_locks.push_back(std::move(lock));
            }
            WindowSourceFileLock dense_lock;
            if (!acquire_inspection_lock((pack_path / "dense.bin").string(),
                                         dense_lock, error, false)) return false;
            inspection_locks.push_back(std::move(dense_lock));
        }
        strata::GgufModel model(shard_paths);
        if (window_sources != nullptr) {
            for (std::size_t shard = 0; shard < model.size(); ++shard) {
                const strata::GgufFile& file = model.shard(shard);
                const auto lock = std::find_if(
                    inspection_locks.begin(), inspection_locks.end(),
                    [&](const WindowSourceFileLock& candidate) {
                        return candidate.path == file.path();
                    });
                if (lock == inspection_locks.end() || !lock->lock_lease() ||
                    lock->file_size != file.file_size() ||
                    lock->file_device_id != file.file_device_id() ||
                    lock->file_object_id != file.file_object_id() ||
                    !lock->lock_lease()->matches(file.path(), file.file_size(),
                                                 file.file_device_id(),
                                                 file.file_object_id())) {
                    error = file.path() + ": GGUF identity changed between inspection lock and mmap";
                    return false;
                }
            }
        }
        const strata::GgufFile& metadata = model.meta();

        const std::string architecture_error = strata::check_architecture(metadata);
        if (!architecture_error.empty()) {
            error = architecture_error;
            return false;
        }
        std::uint64_t metadata_experts = 0;
        if (!meta_u64(metadata, "qwen4exp.expert_count", metadata_experts, error)) return false;
        if (metadata_experts != 256 && metadata_experts != 512) {
            error = "qwen4exp.expert_count must be 256 (Coder) or 512";
            return false;
        }
        if (model.size() > 1) {
            std::uint64_t split_count = 0, split_no = 0, declared_tensors = 0;
            if (!meta_u64(metadata, "split.count", split_count, error) ||
                !meta_u64(metadata, "split.no", split_no, error) ||
                !meta_u64(metadata, "split.tensors.count", declared_tensors, error)) {
                return false;
            }
            std::uint64_t actual_tensors = 0;
            for (std::size_t shard = 0; shard < model.size(); ++shard) {
                if (!add(actual_tensors, model.shard(shard).tensors().size(), actual_tensors)) {
                    error = "GGUF split tensor count overflows";
                    return false;
                }
            }
            if (split_count != model.size() || split_no != 0 || declared_tensors != actual_tensors) {
                error = "GGUF split.count, split.no or split.tensors.count does not match the opened shards";
                return false;
            }
        }
        if (!validate_qwen_tensor_inventory(model, metadata_experts, error)) return false;

        std::size_t ple_shard = 0;
        const strata::TensorInfo* ple = model.find("per_layer_token_embd.weight", &ple_shard);
        if (ple == nullptr || ple->shape.size() != 2 || ple->shape[0] != 160 ||
            ple->shape[1] != 320'001'536 || ple->type != 20 ||
            !model.in_bounds(*ple, ple_shard)) {
            error = "per_layer_token_embd.weight is missing or does not match the supported IQ4_NL table";
            return false;
        }
        if (!options.ple_gguf.empty()) {
            std::error_code ec;
            if (!fs::equivalent(fs::path(options.ple_gguf), fs::path(model.shard(ple_shard).path()), ec) ||
                ec) {
                error = "--ple-gguf does not name the shard containing per_layer_token_embd.weight";
                return false;
            }
        }

        ExpertTable expert_table;
        if (!read_native_experts(pack_path / "native_experts.txt", source_path, model,
                                 kLayers, kHidden, kFeedForward, expert_table, error)) {
            return false;
        }
        if (warning != nullptr && fs::exists(pack_path / "experts.bin")) {
            *warning = "ignoring pack experts.bin on macOS; routed experts are read from the GGUF shards";
        }
        if (expert_table.n_expert != metadata_experts) {
            error = "native_experts.txt n_expert does not match qwen4exp.expert_count";
            return false;
        }

        if (expert_table.layers.size() != kLayers) {
            error = "native_experts.txt layer count does not match qwen4exp.block_count";
            return false;
        }

        std::map<std::string, Conversion> conversions;
        if (!read_conversions(pack_path / "conversions.json", model, conversions, error)) return false;
        IndexData index;
        if (!read_index(pack_path, model, conversions, index, error)) return false;
        if (window_sources != nullptr) {
            const fs::path dense_path = pack_path / "dense.bin";
            const auto dense_lock = std::find_if(
                inspection_locks.begin(), inspection_locks.end(),
                [&](const WindowSourceFileLock& candidate) {
                    return candidate.path == dense_path.string();
                });
            if (dense_lock == inspection_locks.end() || !dense_lock->lock_lease() ||
                dense_lock->file_size != index.dense_file_size ||
                dense_lock->file_device_id != index.dense_file_device_id ||
                dense_lock->file_object_id != index.dense_file_object_id ||
                !dense_lock->lock_lease()->matches(dense_path.string(), index.dense_file_size,
                                                    index.dense_file_device_id,
                                                    index.dense_file_object_id)) {
                error = dense_path.string() + ": dense-pack identity changed during locked inspection";
                return false;
            }
        }

        std::set<std::string> expected_index_names;
        std::set<std::string> types;
        std::uint64_t nonexpert_bytes = 0;
        for (std::size_t shard = 0; shard < model.size(); ++shard) {
            const strata::GgufFile& file = model.shard(shard);
            std::vector<ByteRange> all_ranges;
            for (const strata::TensorInfo& tensor : file.tensors()) {
                const std::uint64_t payload = strata::tensor_payload_bytes(tensor);
                const std::uint64_t absolute = file.data_start() + tensor.offset;
                std::uint64_t end = 0;
                if (payload == 0 || !add(absolute, payload, end) || end > file.file_size() ||
                    !model.in_bounds(tensor, shard)) {
                    error = file.path() + ": invalid or truncated tensor " + tensor.name;
                    return false;
                }
                all_ranges.push_back({absolute, end});
                const auto conversion = conversions.find(tensor.name);
                if (conversion != conversions.end()) {
                    types.insert(canonical_type_name(conversion->second.destination_type));
                    if (is_expert_tensor(tensor.name) || tensor.name == "per_layer_token_embd.weight") {
                        error = "conversions.json must not convert expert or PLE tensors";
                        return false;
                    }
                } else {
                    types.insert(type_name(tensor.type));
                }
                if (!is_expert_tensor(tensor.name) && tensor.name != "per_layer_token_embd.weight") {
                    expected_index_names.insert(tensor.name);
                    if (conversion == conversions.end() &&
                        !add(nonexpert_bytes, payload, nonexpert_bytes)) {
                        error = "non-expert byte count overflows";
                        return false;
                    }
                }
            }
            if (ranges_overlap(all_ranges, error)) {
                error = file.path() + ": " + error;
                return false;
            }
        }
        for (const auto& [name, row] : index.rows) {
            (void)row;
            if (!expected_index_names.contains(name)) {
                error = "index.txt contains an unknown or excluded tensor " + name;
                return false;
            }
        }
        if (index.rows.size() != expected_index_names.size()) {
            for (const std::string& name : expected_index_names) {
                if (!index.rows.contains(name)) {
                    error = "index.txt is missing GGUF tensor " + name;
                    return false;
                }
            }
        }
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const std::string router_name = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight";
            std::size_t shard_index = 0;
            const strata::TensorInfo* router = model.find(router_name, &shard_index);
            if (router == nullptr || router->shape.size() != 2 ||
                router->shape[0] != kHidden || router->shape[1] != metadata_experts) {
                error = router_name + " is missing or its expert dimension is invalid";
                return false;
            }
            const auto converted = conversions.find(router_name);
            const std::string router_type = converted == conversions.end()
                ? type_name(router->type)
                : canonical_type_name(converted->second.destination_type);
            if (router_type != "bf16") {
                error = router_name + " must be served as BF16";
                return false;
            }
            (void)shard_index;
        }

        for (std::size_t shard = 0; shard < model.size(); ++shard) {
            const auto& file = model.shard(shard);
            geometry.shards.push_back({fs::path(file.path()).filename().string(), file.file_size()});
        }
        if (!metadata_facts(metadata, expert_table, geometry, error)) return false;
        if (!safe_atom(options.family) && !options.family.empty()) {
            error = "geometry family must be a non-empty atom";
            return false;
        }
        geometry.family = options.family.empty() ? "qwen4exp" : options.family;
        std::string inferred_model = options.model;
        if (inferred_model.empty()) {
            inferred_model = geometry_atom(source_path.stem().string());
            const std::size_t shard_tag = inferred_model.rfind("-00001-of-");
            if (shard_tag != std::string::npos) inferred_model.erase(shard_tag);
        }
        geometry.model = inferred_model.empty() ? "qwen4exp" : inferred_model;
        geometry.nonexpert_bytes = nonexpert_bytes;
        geometry.dense_pack_bytes = index.dense_pack_bytes;
        geometry.types.assign(types.begin(), types.end());

        WindowSources inspected_window_sources;
        if (!collect_window_sources(model, conversions, index, pack_path, geometry.n_expert,
                                    inspected_window_sources, error) ||
            !compute_wrapped_unread(inspected_window_sources,
                                    geometry.wrapped_unread_bytes, error)) {
            return false;
        }
        if (window_sources != nullptr) {
            const auto dense_lock = std::find_if(
                inspection_locks.begin(), inspection_locks.end(),
                [&](const WindowSourceFileLock& candidate) {
                    return candidate.path == inspected_window_sources.dense_pack_path;
                });
            if (dense_lock == inspection_locks.end() || !dense_lock->lock_lease()) {
                error = inspected_window_sources.dense_pack_path +
                        ": dense-pack inspection lease is missing its index catalog";
                return false;
            }
            std::const_pointer_cast<WindowSourceLockLease>(dense_lock->lock_lease())
                ->bind_dense_pack_catalog(inspected_window_sources.dense_pack_tensors);
        }
        if (!options.mtp_dir.empty()) {
            std::error_code mtp_resolve_error;
            const fs::path mtp_directory =
                fs::canonical(fs::path(options.mtp_dir), mtp_resolve_error);
            if (mtp_resolve_error || mtp_directory.filename().empty()) {
                error = "cannot resolve MTP runtime directory " + options.mtp_dir +
                        (mtp_resolve_error ? ": " + mtp_resolve_error.message() : "");
                return false;
            }
            fs::path mtp_set_lock_path;
            if (!artifact_set_lock_path(mtp_directory, mtp_set_lock_path, error)) {
                return false;
            }
            WindowSourceFileLock mtp_set_lock;
            if (!acquire_inspection_lock(mtp_set_lock_path.string(), mtp_set_lock,
                                         error, true)) {
                return false;
            }
            inspection_locks.push_back(std::move(mtp_set_lock));
            const fs::path mtp_pending_marker =
                mtp_directory / ".strata-generation-pending";
            std::error_code mtp_pending_error;
            const bool mtp_pending = fs::exists(mtp_pending_marker, mtp_pending_error);
            if (mtp_pending_error) {
                error = "cannot inspect artifact generation marker " +
                        mtp_pending_marker.string() + ": " + mtp_pending_error.message();
                return false;
            }
            if (mtp_pending) {
                error = "artifact generation is incomplete: " + mtp_pending_marker.string();
                return false;
            }
            if (!read_mtp_runtime(mtp_directory, geometry.mtp_bytes,
                                  geometry.mtp_bind_bytes, error,
                                  mtp_sources != nullptr ? &inspected_mtp_sources : nullptr)) {
                return false;
            }
            if (mtp_sources != nullptr) {
                const auto mtp_lock = std::find_if(
                    inspection_locks.begin(), inspection_locks.end(),
                    [&](const WindowSourceFileLock& candidate) {
                        return candidate.path == mtp_set_lock_path.string();
                    });
                if (mtp_lock == inspection_locks.end() || !mtp_lock->lock_lease()) {
                    error = mtp_set_lock_path.string() +
                            ": MTP runtime catalog is missing its generation lock";
                    return false;
                }
                inspected_mtp_sources.generation_lock = *mtp_lock;
                std::const_pointer_cast<WindowSourceLockLease>(mtp_lock->lock_lease())
                    ->bind_mtp_catalog(inspected_mtp_sources.catalog);
            }
        }
        inspected_window_sources.inspected_file_locks = std::move(inspection_locks);
        if (!validate_geometry(geometry, error)) return false;
        if (window_sources != nullptr) *window_sources = std::move(inspected_window_sources);
        if (mtp_sources != nullptr) *mtp_sources = std::move(inspected_mtp_sources);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        geometry = {};
        return false;
    } catch (...) {
        error = "unknown error while inspecting GGUF and native pack";
        geometry = {};
        return false;
    }
}

}  // namespace strata::metal
