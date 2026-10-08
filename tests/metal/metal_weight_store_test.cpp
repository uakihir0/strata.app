// Host-only adversarial fixtures for GGUF/native-pack inspection. The GGUF and runtime blobs are sparse files:
// their tables are real, while WeightStore must not read tensor payloads.
#include "strata/metal/weights.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "gguf_fixture.hpp"
#include "../../src/metal/engine/window_source_lock.hpp"
#include "../../src/metal/kernels/common/params.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/file.h>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kLayers = 48;
constexpr std::uint64_t kExpertBlob = 1'382'400;
constexpr std::uint64_t kMtpExpertsBytes = 707'788'800;
constexpr std::uint64_t kMtpDenseBytes = 116'099'072;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

int open_pinned_fixture_directory(const fs::path& path, const struct stat& expected) {
    const int descriptor =
        ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        throw std::runtime_error(
            "could not pin MTP fixture directory: " +
            std::error_code(errno, std::generic_category()).message());
    }
    struct stat opened {};
    if (::fstat(descriptor, &opened) != 0 || !S_ISDIR(opened.st_mode) ||
        opened.st_dev != expected.st_dev || opened.st_ino != expected.st_ino) {
        const int saved_errno = errno;
        (void)::close(descriptor);
        throw std::runtime_error(
            "MTP fixture directory changed while opening: " +
            std::error_code(saved_errno, std::generic_category()).message());
    }
    return descriptor;
}

void verify_fixture_owner_marker() {
    constexpr std::string_view expected =
        "strata MTP storage CTest fixture v1\n";
    const int descriptor = ::open(
        ".strata-owned-fixture",
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (descriptor < 0) {
        throw std::runtime_error(
            "MTP fixture destination must be an empty directory");
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size != static_cast<off_t>(expected.size())) {
        (void)::close(descriptor);
        throw std::runtime_error("MTP fixture ownership marker is not a valid regular file");
    }
    std::string content(expected.size(), '\0');
    std::size_t offset = 0;
    while (offset < content.size()) {
        const ssize_t count =
            ::read(descriptor, content.data() + offset, content.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            const int saved_errno = count == 0 ? EIO : errno;
            (void)::close(descriptor);
            throw std::runtime_error(
                "could not read MTP fixture ownership marker: " +
                std::error_code(saved_errno, std::generic_category()).message());
        }
        offset += static_cast<std::size_t>(count);
    }
    char trailing = 0;
    const ssize_t trailing_bytes = ::read(descriptor, &trailing, 1);
    const int saved_errno = errno;
    (void)::close(descriptor);
    if (trailing_bytes < 0) {
        throw std::runtime_error(
            "could not finish reading MTP fixture ownership marker: " +
            std::error_code(saved_errno, std::generic_category()).message());
    }
    if (content != expected || trailing_bytes != 0) {
        throw std::runtime_error("MTP fixture ownership marker is invalid");
    }
}

struct TempDir {
    fs::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-weight-store-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void put_u32(std::ofstream& output, std::uint32_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void put_u64(std::ofstream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void put_string(std::ofstream& output, const std::string& value) {
    put_u64(output, value.size());
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_kv(std::ofstream& output, const fixture::Kv& value) {
    put_string(output, value.key);
    put_u32(output, value.type);
    if (value.type == 8) {
        put_string(output, value.s);
    } else if (value.type == 2) {
        const auto narrowed = static_cast<std::uint16_t>(value.u);
        output.write(reinterpret_cast<const char*>(&narrowed), sizeof(narrowed));
    } else if (value.type == 4 || value.type == 5) {
        put_u32(output, static_cast<std::uint32_t>(value.u));
    } else if (value.type == 12) {
        put_u64(output, value.u);
    } else if (value.type == 9) {
        put_u32(output, 8);
        put_u64(output, value.items.size());
        for (const std::string& item : value.items) put_string(output, item);
    } else {
        throw std::runtime_error("unsupported synthetic GGUF metadata type");
    }
}

fixture::Kv f64(std::string key, double value) {
    return {std::move(key), 12, {},
            std::bit_cast<std::uint64_t>(value), {}};
}

fixture::Kv i32(std::string key, std::int32_t value) {
    return {std::move(key), 5, {},
            static_cast<std::uint64_t>(static_cast<std::int64_t>(value)), {}};
}

struct SyntheticModel {
    fs::path native;
    fs::path pack;
    fs::path mtp;
    fs::path ple_gguf;
    std::uint64_t data_start = 0;
    std::uint64_t source_bytes = 0;
    std::map<std::string, std::uint64_t> absolute_offsets;
    std::map<std::string, std::string> tensor_shards;
    std::vector<fs::path> shard_paths;
    std::vector<std::string> shard_names;
    std::vector<std::uint64_t> shard_sizes;
    std::vector<fixture::Tensor> tensors;
    std::uint32_t n_expert = kExperts;
};

void write_native_experts(const SyntheticModel& model, bool v4, bool corrupt_offset,
                          bool omit_offsets = false, bool too_many_shards = false,
                          bool wrong_slab_size = false, bool invalid_shard_name = false);
void write_index(const SyntheticModel& model, bool valid);
void write_conversions(const SyntheticModel& model, bool valid = true, bool wrong_form = false,
                       bool omit_router_record = false, bool lossy_router = false,
                       bool wrong_method = false);

std::vector<fixture::Tensor> model_tensors(std::string_view omit_tensor = {},
                                           std::string_view wrong_shape_tensor = {},
                                           std::string_view wrong_type_tensor = {},
                                           std::uint32_t n_expert = kExperts,
                                           std::uint32_t wrong_type_id = 16,
                                           std::string_view wrong_geometry_tensor = {}) {
    std::vector<fixture::Tensor> tensors;
    tensors.reserve(1'224);
    auto add = [&tensors](std::string name, std::vector<std::uint64_t> shape, std::uint32_t type) {
        tensors.push_back({std::move(name), std::move(shape), type, 1});
    };

    add("token_embd.weight", {2'560, 248'320}, 11);
    add("output.weight", {2'560, 248'320}, 13);
    add("output_hc_norm.weight", {10'240}, 0);
    add("output_hc_down.weight", {10'240, 320}, 30);
    add("output_hc_up.weight", {320, 10'240}, 30);

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add(prefix + "hc_attn_norm.weight", {10'240}, 0);
        add(prefix + "hc_ffn_norm.weight", {10'240}, 0);
        add(prefix + "hc_attn_down.weight", {10'240, 320}, 30);
        add(prefix + "hc_ffn_down.weight", {10'240, 320}, 30);
        add(prefix + "hc_attn_up.weight", {320, 10'240}, 30);
        add(prefix + "hc_ffn_up.weight", {320, 10'240}, 30);
        add(prefix + "hc_attn_inject.weight", {10'240, 4}, 30);
        add(prefix + "hc_ffn_inject.weight", {10'240, 4}, 30);
        add(prefix + "ffn_gate_inp.weight", {2'560, n_expert}, layer == 0 ? 0u : 30u);
        add(prefix + "ffn_gate_inp_shexp.weight", {2'560}, 30);
        add(prefix + "ffn_gate_shexp.weight", {2'560, 640}, 42);
        add(prefix + "ffn_up_shexp.weight", {2'560, 640}, 42);
        add(prefix + "ffn_down_shexp.weight", {640, 2'560}, 42);
        add(prefix + "ffn_gate_exps.weight", {2'560, 640, n_expert}, 42);
        add(prefix + "ffn_up_exps.weight", {2'560, 640, n_expert}, 42);
        add(prefix + "ffn_down_exps.weight", {640, 2'560, n_expert}, 42);

        if (layer % 4 == 3) {
            add(prefix + "attn_q.weight", {2'560, 12'288}, 42);
            add(prefix + "attn_k.weight", {2'560, 512}, 13);
            add(prefix + "attn_v.weight", {2'560, 512}, 13);
            add(prefix + "attn_output.weight", {6'144, 2'560}, 12);
            add(prefix + "attn_q_norm.weight", {256}, 0);
            add(prefix + "attn_k_norm.weight", {256}, 0);
            add(prefix + "indexer.q_proj.weight", {2'560, 512}, 30);
            add(prefix + "indexer.k_proj.weight", {2'560, 128}, 30);
            add(prefix + "indexer.q_norm.weight", {128}, 0);
            add(prefix + "indexer.k_norm.weight", {128}, 0);
        } else {
            add(prefix + "attn_qkv.weight", {2'560, 10'240}, 42);
            add(prefix + "attn_gate.weight", {2'560, 6'144}, 42);
            add(prefix + "ssm_out.weight", {6'144, 2'560}, 13);
            add(prefix + "ssm_conv1d.weight", {4, 10'240}, 0);
            add(prefix + "ssm_alpha.weight", {2'560, 48}, 30);
            add(prefix + "ssm_beta.weight", {2'560, 48}, 30);
            add(prefix + "ssm_a", {48}, 0);
            add(prefix + "ssm_dt.bias", {48}, 0);
            add(prefix + "ssm_norm.weight", {128}, 0);
        }

        if (layer == 1) {
            add(prefix + "ple_key.weight", {2'560, 10'240}, 42);
            add(prefix + "ple_value.weight", {2'560, 2'560}, 30);
            add(prefix + "ple_norm_key.weight", {10'240}, 0);
            add(prefix + "ple_norm_query.weight", {10'240}, 0);
            add(prefix + "ple_norm_conv.weight", {10'240}, 0);
            add(prefix + "ple_conv1d.weight", {4, 10'240}, 1);
        }
    }
    add("per_layer_token_embd.weight", {160, 320'001'536}, 20);
    if (!omit_tensor.empty()) {
        tensors.erase(std::remove_if(tensors.begin(), tensors.end(), [omit_tensor](const auto& tensor) {
            return tensor.name == omit_tensor;
        }), tensors.end());
    }
    if (!wrong_shape_tensor.empty()) {
        auto found = std::find_if(tensors.begin(), tensors.end(), [wrong_shape_tensor](const auto& tensor) {
            return tensor.name == wrong_shape_tensor;
        });
        require(found != tensors.end() && found->shape.size() == 2, "wrong-shape fixture tensor is unavailable");
        --found->shape[1];
    }
    if (!wrong_type_tensor.empty()) {
        auto found = std::find_if(tensors.begin(), tensors.end(), [wrong_type_tensor](const auto& tensor) {
            return tensor.name == wrong_type_tensor;
        });
        require(found != tensors.end(), "wrong-type fixture tensor is unavailable");
        found->type = wrong_type_id;
    }
    if (!wrong_geometry_tensor.empty()) {
        auto found = std::find_if(tensors.begin(), tensors.end(), [wrong_geometry_tensor](const auto& tensor) {
            return tensor.name == wrong_geometry_tensor;
        });
        require(found != tensors.end() && !found->shape.empty(),
                "invalid-geometry fixture tensor is unavailable");
        ++found->shape[0];
    }
    return tensors;
}

std::vector<fixture::Kv> model_metadata(std::uint32_t expert_count = kExperts) {
    std::vector<std::string> tokens;
    tokens.reserve(248'320);
    for (std::uint32_t i = 0; i < 248'320; ++i) tokens.push_back("tok" + std::to_string(i));
    return {
        fixture::str("general.architecture", "qwen4exp"),
        fixture::u32("qwen4exp.block_count", 48),
        fixture::u32("qwen4exp.expert_count", expert_count),
        fixture::u32("qwen4exp.expert_used_count", 10),
        fixture::u32("qwen4exp.embedding_length", 2560),
        fixture::u32("qwen4exp.attention.head_count", 24),
        fixture::u32("qwen4exp.attention.head_count_kv", 2),
        fixture::u32("qwen4exp.attention.key_length", 256),
        fixture::u32("qwen4exp.expert_feed_forward_length", 640),
        fixture::u32("qwen4exp.expert_shared_feed_forward_length", 640),
        fixture::string_array("tokenizer.ggml.tokens", std::move(tokens)),
    };
}

bool mutate_metadata_value(std::vector<fixture::Kv>& metadata, std::string_view key) {
    for (fixture::Kv& value : metadata) {
        if (value.key != key) continue;
        if (key == "general.architecture") {
            value.s = "not-qwen4exp";
        } else if (key == "qwen4exp.expert_count") {
            value.u = 256;
        } else if (key == "split.count") {
            value.u = 3;
        } else if (key == "split.no") {
            value.u = 1;
        } else if (key == "split.tensors.count") {
            value.u = 1;
        } else {
            value.u = 0;
        }
        return true;
    }
    return false;
}

struct WrittenShard {
    std::uint64_t data_start = 0;
    std::uint64_t size = 0;
    std::map<std::string, std::uint64_t> absolute_offsets;
};

WrittenShard write_gguf_shard(const fs::path& path, const std::vector<fixture::Kv>& metadata,
                              const std::vector<fixture::Tensor>& tensors,
                              std::string_view invalid_geometry_tensor = {}) {
    require(!tensors.empty(), "synthetic shard cannot be empty");
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> sizes;
    offsets.reserve(tensors.size());
    sizes.reserve(tensors.size());
    std::uint64_t cursor = 0;
    for (const fixture::Tensor& tensor : tensors) {
        strata::TensorInfo info;
        info.shape = tensor.shape;
        info.type = tensor.type;
        std::uint64_t bytes = strata::tensor_payload_bytes(info);
        if (bytes == 0 && tensor.name == invalid_geometry_tensor) bytes = 1;
        require(bytes != 0, "synthetic tensor has no payload size");
        offsets.push_back(cursor);
        sizes.push_back(bytes);
        cursor = (cursor + bytes + 31) / 32 * 32;
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "could not create synthetic GGUF shard");
    put_u32(output, 0x46554747u);
    put_u32(output, 3);
    put_u64(output, tensors.size());
    put_u64(output, metadata.size());
    for (const fixture::Kv& value : metadata) write_kv(output, value);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const fixture::Tensor& tensor = tensors[i];
        put_string(output, tensor.name);
        put_u32(output, static_cast<std::uint32_t>(tensor.shape.size()));
        for (const std::uint64_t dimension : tensor.shape) put_u64(output, dimension);
        put_u32(output, tensor.type);
        put_u64(output, offsets[i]);
    }
    WrittenShard result;
    const auto table_end = static_cast<std::uint64_t>(output.tellp());
    result.data_start = (table_end + 31) / 32 * 32;
    while (static_cast<std::uint64_t>(output.tellp()) < result.data_start) output.put('\0');
    result.size = result.data_start + offsets.back() + sizes.back();
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        result.absolute_offsets[tensors[i].name] = result.data_start + offsets[i];
    }
    output.seekp(static_cast<std::streamoff>(result.size - 1));
    output.put('\0');
    output.close();
    require(!output.fail(), "could not size sparse synthetic GGUF shard");
    return result;
}

SyntheticModel make_model(const fs::path& root, std::string_view omit_tensor = {},
                          std::string_view wrong_shape_tensor = {},
                          std::string_view wrong_type_tensor = {}, bool swift_split = false,
                          std::uint32_t n_expert = kExperts,
                          bool per_role_split = false,
                          std::string_view wrong_metadata_key = {},
                          std::uint32_t wrong_type_id = 16,
                          std::string_view wrong_geometry_tensor = {},
                          bool single_shard_split_metadata = false,
                          std::uint32_t swift_split_layer = 13) {
    static_assert(std::endian::native == std::endian::little);
    SyntheticModel result;
    result.pack = root / "pack";
    result.n_expert = n_expert;
    fs::create_directories(result.pack);

    result.tensors = model_tensors(omit_tensor, wrong_shape_tensor, wrong_type_tensor,
                                   n_expert, wrong_type_id, wrong_geometry_tensor);
    const std::string first_name = swift_split ? "tiny-00001-of-00002.gguf" : "tiny.gguf";
    const std::string second_name = "tiny-00002-of-00002.gguf";
    result.shard_names.push_back(first_name);
    if (swift_split) result.shard_names.push_back(second_name);
    result.shard_paths.push_back(root / first_name);
    if (swift_split) result.shard_paths.push_back(root / second_name);

    std::vector<std::vector<fixture::Tensor>> shard_tensors(result.shard_names.size());
    for (const fixture::Tensor& tensor : result.tensors) {
        std::size_t shard = 0;
        if (swift_split) {
            if (tensor.name == "per_layer_token_embd.weight" ||
                tensor.name == "token_embd.weight" || tensor.name == "output.weight" ||
                tensor.name == "output_hc_norm.weight" || tensor.name == "output_hc_down.weight" ||
                tensor.name == "output_hc_up.weight") {
                shard = 0;
            } else if (tensor.name.starts_with("blk.")) {
                const std::size_t layer_end = tensor.name.find('.', 4);
                require(layer_end != std::string::npos, "could not parse synthetic tensor layer");
                const std::uint32_t layer = static_cast<std::uint32_t>(
                    std::stoul(tensor.name.substr(4, layer_end - 4)));
                const bool straddling_layer_tensor =
                    layer == swift_split_layer &&
                    (tensor.name.ends_with(".attn_gate.weight") ||
                     tensor.name.ends_with(".attn_qkv.weight"));
                shard = layer < swift_split_layer || straddling_layer_tensor ? 0 : 1;
                if (per_role_split &&
                    tensor.name == "blk." + std::to_string(swift_split_layer) +
                                        ".attn_gate.weight") {
                    shard = 1;
                }
                if (per_role_split &&
                    tensor.name == "blk." + std::to_string(swift_split_layer) +
                                        ".ffn_gate_exps.weight") {
                    shard = 0;
                }
            } else {
                throw std::runtime_error("unexpected synthetic tensor name " + tensor.name);
            }
        }
        shard_tensors[shard].push_back(tensor);
        result.tensor_shards.emplace(tensor.name, result.shard_names[shard]);
    }

    const std::vector<fixture::Kv> full_metadata = model_metadata(n_expert);
    bool metadata_key_mutated = false;
    for (std::size_t shard = 0; shard < shard_tensors.size(); ++shard) {
        std::vector<fixture::Kv> metadata;
        if (shard == 0) {
            metadata = full_metadata;
        } else {
            metadata = fixture::split_keys(shard, shard_tensors.size(), result.tensors.size());
        }
        if (swift_split || single_shard_split_metadata) {
            const auto split = fixture::split_keys(shard, shard_tensors.size(), result.tensors.size());
            metadata.insert(metadata.end(), split.begin(), split.end());
        }
        metadata_key_mutated = mutate_metadata_value(metadata, wrong_metadata_key) || metadata_key_mutated;
        const WrittenShard written = write_gguf_shard(result.shard_paths[shard], metadata,
                                                       shard_tensors[shard], wrong_geometry_tensor);
        result.shard_sizes.push_back(written.size);
        if (shard == 0) {
            result.data_start = written.data_start;
            result.source_bytes = written.size;
            result.native = result.shard_paths[shard];
        }
        for (const auto& [name, absolute] : written.absolute_offsets) {
            result.absolute_offsets.emplace(name, absolute);
        }
    }
    require(wrong_metadata_key.empty() || metadata_key_mutated,
            "metadata mutation key is unavailable in the synthetic shard set");
    const auto ple = result.tensor_shards.find("per_layer_token_embd.weight");
    require(ple != result.tensor_shards.end(), "synthetic PLE table has no shard assignment");
    const auto ple_shard =
        std::find(result.shard_names.begin(), result.shard_names.end(), ple->second);
    require(ple_shard != result.shard_names.end(), "synthetic PLE shard name is unknown");
    result.ple_gguf = result.shard_paths[static_cast<std::size_t>(ple_shard - result.shard_names.begin())];

    write_native_experts(result, false, false);
    write_index(result, true);
    write_conversions(result);
    return result;
}

void write_native_experts(const SyntheticModel& model, bool v4, bool corrupt_offset,
                          bool omit_offsets, bool too_many_shards,
                          bool wrong_slab_size, bool invalid_shard_name) {
    std::ofstream output(model.pack / "native_experts.txt", std::ios::trunc);
    require(static_cast<bool>(output), "could not write native_experts.txt");
    const std::uint64_t total = static_cast<std::uint64_t>(kLayers) * kExpertBlob * model.n_expert;
    const std::string& first_shard = model.shard_names.front();
    if (v4) {
        output << "# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                  "[shard | gate,up,down] (n_expert "
               << model.n_expert << ", total " << total << "; absolute offsets in " << first_shard
               << ", or in the named shard beside it - per role where the column is gate,up,down)\n";
    } else {
        output << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                  "[shard] (n_expert "
               << model.n_expert << ", total " << total << "; absolute offsets in " << first_shard
               << ", or in the named shard beside it)\n";
    }
    std::uint64_t source_offset = 0;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
        std::uint64_t gate = model.absolute_offsets.at(prefix + "gate_exps.weight");
        if (corrupt_offset && layer == 0) gate += 32;
        const std::uint64_t up = model.absolute_offsets.at(prefix + "up_exps.weight");
        const std::uint64_t down = model.absolute_offsets.at(prefix + "down_exps.weight");
        output << layer << " 42 42 " << source_offset;
        const std::uint64_t slab_bytes = wrong_slab_size && layer == 0 ? kExpertBlob + 32 : kExpertBlob;
        if (!omit_offsets) output << ' ' << slab_bytes << ' ' << gate << ' ' << up << ' ' << down;
        if (v4) {
            output << ' ';
            if (invalid_shard_name && layer == 0) {
                output << "shard-999.gguf," << first_shard << ',' << first_shard;
            } else if (too_many_shards && layer == 0) {
                output << first_shard << ',' << first_shard << ',' << first_shard << ",extra.gguf";
            } else {
                output << model.tensor_shards.at(prefix + "gate_exps.weight") << ','
                       << model.tensor_shards.at(prefix + "up_exps.weight") << ','
                       << model.tensor_shards.at(prefix + "down_exps.weight");
            }
        } else if (model.tensor_shards.at(prefix + "gate_exps.weight") != first_shard) {
            output << ' ' << model.tensor_shards.at(prefix + "gate_exps.weight");
        }
        output << '\n';
        source_offset += kExpertBlob * model.n_expert;
    }
    output.close();
    require(!output.fail(), "failed writing native_experts.txt");
}

void write_index(const SyntheticModel& model, bool valid) {
    std::vector<std::string> rows;
    std::uint64_t destination_offset = 0;
    std::uint64_t source_offset = 0;
    for (const fixture::Tensor& tensor : model.tensors) {
        if (tensor.name == "per_layer_token_embd.weight" ||
            tensor.name.ends_with("ffn_gate_exps.weight") ||
            tensor.name.ends_with("ffn_up_exps.weight") ||
            tensor.name.ends_with("ffn_down_exps.weight")) {
            continue;
        }
        if (!valid && tensor.name == "output_hc_up.weight") continue;

        const std::uint64_t ne0 = tensor.shape[0];
        const std::uint64_t ne1 = tensor.shape.size() > 1 ? tensor.shape[1] : 0;
        const bool converted_router = tensor.name == "blk.0.ffn_gate_inp.weight";
        const bool source_float = tensor.type == 0 || tensor.type == 1 || tensor.type == 30;
        std::ostringstream row;
        row << tensor.name << " 0 ";
        if (!source_float && !converted_router) {
            row << "0 0 0 " << destination_offset << " 0 " << ne0 << ' ' << ne1
                << " 8 0 32 0 0 0 0 0 0 0";
        } else {
            std::uint64_t elements = 1;
            for (const std::uint64_t dimension : tensor.shape) elements *= dimension;
            const std::uint64_t source_bytes = elements * (converted_router ? 2 : tensor.type == 0 ? 4 : 2);
            const std::uint64_t kind =
                converted_router ? 4 : tensor.type == 0 ? 2 : tensor.type == 30 ? 4 : 5;
            const std::uint64_t destination_bytes = elements * (kind == 2 ? 4 : 2);
            row << kind << ' ' << source_offset << ' ' << source_bytes << ' ' << destination_offset << ' '
                << destination_bytes << ' ' << ne0 << ' ' << ne1 << " 0 0 1 0 0 0 0 0 0 0";
            source_offset = (source_offset + source_bytes + 63) / 64 * 64;
            destination_offset = (destination_offset + destination_bytes + 63) / 64 * 64;
        }
        rows.push_back(row.str());
    }

    std::ofstream output(model.pack / "index.txt", std::ios::trunc);
    require(static_cast<bool>(output), "could not write index.txt");
    output << "# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from "
           << model.shard_names.front() << '\n';
    output << "# align 64 pool " << destination_offset << " tensors " << rows.size() << '\n';
    for (const std::string& row : rows) output << row << '\n';
    output.close();
    require(!output.fail(), "failed writing index.txt");

    std::ofstream dense(model.pack / "dense.bin", std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(dense), "could not create sparse dense.bin");
    dense.close();
    fs::resize_file(model.pack / "dense.bin", source_offset);
}

void write_conversions(const SyntheticModel& model, bool valid, bool wrong_form, bool omit_router_record,
                       bool lossy_router, bool wrong_method) {
    std::ofstream output(model.pack / "conversions.json", std::ios::trunc);
    require(static_cast<bool>(output), "could not write conversions.json");
    if (!valid) {
        output << "{\"schema\":1";
        return;
    }
    output << "{\"schema\":1,\"tool\":\"tools/iq_pack.py\",\"compat_bf16\":false,\"source_shards\":[";
    for (std::size_t i = 0; i < model.shard_names.size(); ++i) {
        if (i != 0) output << ',';
        output << "{\"name\":\"" << model.shard_names[i] << "\",\"size\":" << model.shard_sizes[i] << '}';
    }
    output << "],\"tensors\":[";
    if (!omit_router_record) {
        output << "{\"name\":\"blk.0.ffn_gate_inp.weight\","
                  "\"shard\":\"" << model.tensor_shards.at("blk.0.ffn_gate_inp.weight")
               << "\",\"src_type\":\"F32\",\"dst_type\":\""
           << (wrong_form ? "F16" : "BF16")
           << "\",\"method\":\""
           << (wrong_method ? "unrecognized conversion" :
               "f32->bf16: the values are BF16's (low 16 bits zero)")
           << "\",\"exact\":" << (lossy_router ? "false" : "true")
           << ",\"max_abs_err\":" << (lossy_router ? "0.001" : "0")
           << ",\"source_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"}";
    }
    output << "]}\n";
    output.close();
    require(!output.fail(), "failed writing conversions.json");
}

void corrupt_index_float_row_as_native(const SyntheticModel& model, const std::string& tensor_name) {
    std::ifstream input(model.pack / "index.txt");
    require(static_cast<bool>(input), "could not read index.txt for corruption");
    std::vector<std::string> lines;
    std::string line;
    bool changed = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.front() != '#' && line.starts_with(tensor_name + " ")) {
            std::istringstream row(line);
            std::vector<std::string> fields;
            std::string field;
            while (row >> field) fields.push_back(field);
            require(fields.size() == 19, "target index row does not have 19 fields");
            fields[2] = "0";
            fields[3] = "0";
            fields[4] = "0";
            fields[6] = "0";
            fields[9] = "8";
            fields[10] = "0";
            fields[11] = "32";
            for (std::size_t i = 12; i < fields.size(); ++i) fields[i] = "0";
            std::ostringstream rewritten;
            for (std::size_t i = 0; i < fields.size(); ++i) {
                if (i != 0) rewritten << ' ';
                rewritten << fields[i];
            }
            line = rewritten.str();
            changed = true;
        }
        lines.push_back(line);
    }
    require(changed, "target float index row was not found");
    std::ofstream output(model.pack / "index.txt", std::ios::trunc);
    require(static_cast<bool>(output), "could not rewrite index.txt");
    for (const std::string& row : lines) output << row << '\n';
}

void rewrite_index_row(const SyntheticModel& model, const std::string& tensor_name,
                       std::size_t field_index, const std::string& replacement,
                       bool remove_last_field = false) {
    std::ifstream input(model.pack / "index.txt");
    require(static_cast<bool>(input), "could not read index.txt for mutation");
    std::vector<std::string> lines;
    std::string line;
    bool changed = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.front() != '#') {
            std::istringstream row(line);
            std::vector<std::string> fields;
            std::string field;
            while (row >> field) fields.push_back(field);
            if (!fields.empty() && fields.front() == tensor_name) {
                require(fields.size() == 19, "target index row does not have 19 fields");
                if (remove_last_field) {
                    fields.pop_back();
                } else {
                    require(field_index < fields.size(), "target index field is out of range");
                    fields[field_index] = replacement;
                }
                std::ostringstream rewritten;
                for (std::size_t i = 0; i < fields.size(); ++i) {
                    if (i != 0) rewritten << ' ';
                    rewritten << fields[i];
                }
                line = rewritten.str();
                changed = true;
            }
        }
        lines.push_back(line);
    }
    require(changed, "target index row was not found: " + tensor_name);
    std::ofstream output(model.pack / "index.txt", std::ios::trunc);
    require(static_cast<bool>(output), "could not rewrite index.txt");
    for (const std::string& row : lines) output << row << '\n';
    output.close();
    require(!output.fail(), "failed rewriting index.txt");
}

void set_compat_bf16(const SyntheticModel& model, bool compat) {
    const fs::path path = model.pack / "conversions.json";
    std::ifstream input(path);
    require(static_cast<bool>(input), "could not read conversions.json for mutation");
    std::ostringstream contents;
    contents << input.rdbuf();
    std::string text = contents.str();
    const std::string from = compat ? "\"compat_bf16\":false" : "\"compat_bf16\":true";
    const std::string to = compat ? "\"compat_bf16\":true" : "\"compat_bf16\":false";
    const std::size_t position = text.find(from);
    require(position != std::string::npos, "compat_bf16 field was not found");
    text.replace(position, from.size(), to);
    std::ofstream output(path, std::ios::trunc);
    require(static_cast<bool>(output), "could not rewrite conversions.json");
    output << text;
    output.close();
    require(!output.fail(), "failed rewriting conversions.json");
}

strata::metal::WeightStoreOptions options_for(const SyntheticModel& model) {
    strata::metal::WeightStoreOptions options;
    options.native_shard = model.native.string();
    options.pack_dir = model.pack.string();
    options.ple_gguf = model.ple_gguf.string();
    options.family = "qwen";
    options.model = "tiny";
    return options;
}

bool inspect(const SyntheticModel& model, strata::metal::Geometry& geometry, std::string& error,
             bool with_mtp = false, strata::metal::WindowSources* window_sources = nullptr,
             strata::metal::MtpRuntimeSources* mtp_sources = nullptr,
             std::string* warning = nullptr) {
    auto options = options_for(model);
    if (with_mtp) options.mtp_dir = model.mtp.string();
    return strata::metal::WeightStore::inspect(options, geometry, error, window_sources,
                                                mtp_sources, warning);
}

struct MtpTensor {
    std::string name;
    std::string kind;
    std::uint64_t rows;
    std::uint64_t columns;
};

std::vector<MtpTensor> mtp_tensors() {
    return {
        {"fc_embedding.weight", "q8_0", 2560, 2560},
        {"fc_hidden.weight", "q8_0", 2560, 2560},
        {"self_attn.q_proj.weight", "q8_0", 12'288, 2560},
        {"self_attn.k_proj.weight", "q8_0", 512, 2560},
        {"self_attn.v_proj.weight", "q8_0", 512, 2560},
        {"self_attn.o_proj.weight", "q8_0", 2560, 6144},
        {"self_attn.indexer.index_qk_proj.weight", "q8_0", 640, 2560},
        {"mlp.shared_expert.gate_proj.weight", "q8_0", 640, 2560},
        {"mlp.shared_expert.up_proj.weight", "q8_0", 640, 2560},
        {"mlp.shared_expert.down_proj.weight", "q8_0", 2560, 640},
        {"hyper_connection_mixer.hc_norm.weight", "f32", 1, 10'240},
        {"hyper_connection_mixer.input_mix_weight_down.weight", "bf16", 320, 10'240},
        {"hyper_connection_mixer.input_mix_weight_up.weight", "bf16", 10'240, 320},
        {"attn_hyper_connection.hc_norm.weight", "f32", 1, 10'240},
        {"attn_hyper_connection.input_mix_weight_down.weight", "bf16", 320, 10'240},
        {"attn_hyper_connection.input_mix_weight_up.weight", "bf16", 10'240, 320},
        {"attn_hyper_connection.block_inject_weight.weight", "bf16", 4, 10'240},
        {"mlp_hyper_connection.hc_norm.weight", "f32", 1, 10'240},
        {"mlp_hyper_connection.input_mix_weight_down.weight", "bf16", 320, 10'240},
        {"mlp_hyper_connection.input_mix_weight_up.weight", "bf16", 10'240, 320},
        {"mlp_hyper_connection.block_inject_weight.weight", "bf16", 4, 10'240},
        {"mlp.gate.weight", "bf16", 512, 2560},
        {"mlp.shared_expert_gate.weight", "bf16", 1, 2560},
        {"pre_fc_norm_embedding.weight", "f32", 1, 2560},
        {"pre_fc_norm_hidden.weight", "f32", 1, 10'240},
        {"self_attn.q_norm.weight", "f32", 1, 256},
        {"self_attn.k_norm.weight", "f32", 1, 256},
        {"self_attn.indexer.q_layernorm.weight", "f32", 1, 128},
        {"self_attn.indexer.k_layernorm.weight", "f32", 1, 128},
    };
}

bool mtp_catalog_matches_expected(const strata::metal::MtpRuntimeCatalog& catalog) {
    const std::vector<MtpTensor> expected = mtp_tensors();
    if (catalog.dense_tensors.size() != expected.size()) return false;
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const MtpTensor& tensor = expected[i];
        const strata::metal::MtpDenseWeightSource& actual = catalog.dense_tensors[i];
        const std::uint64_t bytes = tensor.kind == "q8_0"
            ? tensor.rows * (tensor.columns / 32) * 34
            : tensor.rows * tensor.columns * (tensor.kind == "f32" ? 4 : 2);
        const strata::metal::MtpDenseWeightKind kind = tensor.kind == "q8_0"
            ? strata::metal::MtpDenseWeightKind::q8_0
            : tensor.kind == "f32" ? strata::metal::MtpDenseWeightKind::f32
                                     : strata::metal::MtpDenseWeightKind::bf16;
        if (actual.tensor_name != tensor.name || actual.kind != kind ||
            actual.rows != tensor.rows || actual.columns != tensor.columns ||
            actual.file_offset != offset || actual.byte_count != bytes) {
            return false;
        }
        offset = (offset + bytes + 255) / 256 * 256;
    }
    return offset == kMtpDenseBytes;
}

void make_mtp_runtime(SyntheticModel& model, bool swap_q_projection_shape = false) {
    model.mtp = model.native.parent_path() / "mtp-rt";
    fs::create_directories(model.mtp);
    {
        std::ofstream experts(model.mtp / "experts.bin", std::ios::binary | std::ios::trunc);
        std::ofstream dense(model.mtp / "dense.bin", std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(experts) && static_cast<bool>(dense), "could not create MTP runtime blobs");
    }
    fs::resize_file(model.mtp / "experts.bin", kMtpExpertsBytes);
    fs::resize_file(model.mtp / "dense.bin", kMtpDenseBytes);
    std::fstream experts(model.mtp / "experts.bin",
                         std::ios::in | std::ios::out | std::ios::binary);
    std::fstream dense(model.mtp / "dense.bin",
                       std::ios::in | std::ios::out | std::ios::binary);
    std::ofstream index(model.mtp / "dense.txt", std::ios::trunc);
    require(static_cast<bool>(experts) && static_cast<bool>(dense) &&
                static_cast<bool>(index),
            "could not open MTP runtime fixture files");
    auto write_marker = [](std::fstream& file, std::uint64_t offset,
                           std::uint32_t marker) {
        require(offset <= static_cast<std::uint64_t>(
                              std::numeric_limits<std::streamoff>::max()),
                "MTP fixture marker offset cannot be represented");
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(marker),
            static_cast<unsigned char>(marker >> 8),
            static_cast<unsigned char>(marker >> 16),
            static_cast<unsigned char>(marker >> 24),
        };
        file.seekp(static_cast<std::streamoff>(offset));
        file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        require(static_cast<bool>(file), "could not write MTP fixture marker");
    };
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
        write_marker(experts, static_cast<std::uint64_t>(expert) * kExpertBlob,
                     0x4d540000u | expert);
    }
    std::uint64_t offset = 0;
    const std::vector<MtpTensor> tensors = mtp_tensors();
    for (std::size_t row = 0; row < tensors.size(); ++row) {
        const MtpTensor& tensor = tensors[row];
        std::uint64_t bytes = 0;
        if (tensor.kind == "q8_0") {
            bytes = tensor.rows * (tensor.columns / 32) * 34;
        } else {
            bytes = tensor.rows * tensor.columns * (tensor.kind == "f32" ? 4 : 2);
        }
        const bool swapped = swap_q_projection_shape && tensor.name == "self_attn.q_proj.weight";
        index << tensor.name << ' ' << tensor.kind << ' ' << (swapped ? tensor.columns : tensor.rows) << ' '
              << (swapped ? tensor.rows : tensor.columns) << ' ' << offset << ' ' << bytes << '\n';
        write_marker(dense, offset, 0x444e0000u | static_cast<std::uint32_t>(row));
        offset = (offset + bytes + 255) / 256 * 256;
    }
    index.close();
    experts.close();
    dense.close();
    require(!index.fail() && !experts.fail() && !dense.fail() &&
                offset == kMtpDenseBytes,
            "synthetic MTP tensor sizes do not total dense.bin");
}

void make_default_draft_vocab(const fs::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "could not create synthetic draft vocabulary");
    for (std::int32_t id = 0; id < 106'299; ++id) {
        const std::int32_t rotated_id = (id + 17) % 106'299;
        output.write(reinterpret_cast<const char*>(&rotated_id), sizeof(rotated_id));
    }
    output.close();
    require(!output.fail(), "failed writing synthetic draft vocabulary");
}

void check(bool ok, const char* label) {
    std::cout << "  " << label << ": " << (ok ? "ok" : "FAIL") << '\n';
    require(ok, label);
}

bool inspection_rejects_writer_lock(const SyntheticModel& model, const fs::path& path,
                                    std::string& error, bool with_mtp = false) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        error = "cannot open test file for exclusive lock";
        return false;
    }
    if (::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
        (void)::close(descriptor);
        error = "cannot take test exclusive lock";
        return false;
    }
    strata::metal::Geometry geometry;
    strata::metal::WindowSources sources;
    const bool inspected = inspect(model, geometry, error, with_mtp, &sources);
    (void)::flock(descriptor, LOCK_UN);
    (void)::close(descriptor);
    return !inspected && error.find("source is locked for writing") != std::string::npos;
}

bool shared_inspection_lock_blocks_writer(const fs::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    errno = 0;
    const int result = ::flock(descriptor, LOCK_EX | LOCK_NB);
    const int lock_error = errno;
    if (result == 0) (void)::flock(descriptor, LOCK_UN);
    (void)::close(descriptor);
    return result != 0 && (lock_error == EWOULDBLOCK || lock_error == EAGAIN);
}

fs::path artifact_lock_path(const fs::path& directory) {
    const fs::path resolved = fs::weakly_canonical(directory);
    return resolved.parent_path() /
           ("." + resolved.filename().string() + ".strata-artifact-set.lock");
}

void ensure_artifact_lock_exists(const fs::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    require(descriptor >= 0, "create artifact-set lock fixture");
    (void)::close(descriptor);
}

void check_shared_mapped_ple_table(const SyntheticModel& synthetic) {
    std::vector<std::string> shard_paths;
    shard_paths.reserve(synthetic.shard_paths.size());
    for (const fs::path& path : synthetic.shard_paths) {
        shard_paths.push_back(path.string());
    }
    auto model =
        std::make_shared<strata::GgufModel>(std::move(shard_paths));
    std::size_t ple_shard = 0;
    require(model->find("per_layer_token_embd.weight", &ple_shard) != nullptr,
            "synthetic mapped model contains its inspected PLE shard");

    strata::kernels::PleTable table;
    std::string error = "stale error";
    require(!table.open_mapped(model, model->size(), error) &&
                !table.is_open() &&
                error == "PleTable::open_mapped shard index is out of range",
            "mapped PLE open rejects an out-of-range shard without publishing a table");
    require(table.open_mapped(model, ple_shard, error),
            "mapped PLE open reuses the model shard: " + error);
    require(error.empty() && table.is_open() &&
                table.mode() == strata::kernels::PleIo::Mmap &&
                table.rows() == strata::kernels::PLE_TABLE_ROWS,
            "shared-mapping PLE table reports its validated mode and row count");

    std::weak_ptr<const strata::GgufModel> retained_model = model;
    model.reset();
    require(!retained_model.expired(),
            "PleTable retains the GGUF mmap owner after the caller releases it");

    std::array<std::uint32_t, strata::kernels::PLE_N_HEADS> row_ids{};
    std::array<float, strata::kernels::NG_N_EMBD> gathered{};
    table.gather(row_ids.data(), gathered.data());
    require(std::all_of(gathered.begin(), gathered.end(),
                        [](float value) { return value == 0.0f; }),
            "mapped PLE gather reads sparse fixture rows from the retained mapping");
    table.close();
    require(!table.is_open() && retained_model.expired(),
            "closing the PLE table releases its shared GGUF mmap owner");
}

std::shared_ptr<strata::GgufModel> make_split_mapped_fp8_model(
    const fs::path& directory, fixture::Kv scale_metadata) {
    fs::create_directories(directory);
    const fs::path first_path = directory / "fp8-00001-of-00002.gguf";
    const fs::path second_path = directory / "fp8-00002-of-00002.gguf";

    std::vector<fixture::Kv> first_metadata{
        fixture::str("general.architecture", "qwen4exp"),
        fixture::str("strata.ple.format", "f8_e4m3"),
        std::move(scale_metadata),
    };
    const auto first_split = fixture::split_keys(0, 2, 2);
    first_metadata.insert(first_metadata.end(), first_split.begin(),
                          first_split.end());
    const std::vector<fixture::Tensor> first_tensors{
        {"token_embd.weight", {4}, 0, 1},
    };
    const std::vector<fixture::Kv> second_metadata =
        fixture::split_keys(1, 2, 2);
    const std::vector<fixture::Tensor> second_tensors{
        {"per_layer_token_embd.weight", {160, 4}, 24, 1},
    };
    (void)write_gguf_shard(first_path, first_metadata, first_tensors);
    (void)write_gguf_shard(second_path, second_metadata, second_tensors);

    return std::make_shared<strata::GgufModel>(
        std::vector<std::string>{first_path.string(), second_path.string()});
}

void check_split_mapped_fp8_ple_table(const fs::path& directory) {
    auto model = make_split_mapped_fp8_model(
        directory, f64("strata.ple.scale", 2.0));
    std::size_t ple_shard = 0;
    require(model->find("per_layer_token_embd.weight", &ple_shard) != nullptr &&
                ple_shard == 1 &&
                model->meta().get("strata.ple.format") != nullptr &&
                model->shard(ple_shard).get("strata.ple.format") == nullptr,
            "split FP8 fixture keeps PLE format metadata only in the first shard");

    strata::kernels::PleTable table;
    std::string error;
    require(table.open_mapped(model, ple_shard, error) &&
                table.mode() == strata::kernels::PleIo::Mmap &&
                table.rows() == 4 && error.empty(),
            "mapped FP8 PLE reads format metadata from the model's first shard: " +
                error);
    std::array<std::uint32_t, strata::kernels::PLE_N_HEADS> row_ids{};
    std::array<float, strata::kernels::NG_N_EMBD> gathered{};
    table.gather(row_ids.data(), gathered.data());
    require(std::all_of(gathered.begin(), gathered.end(),
                        [](float value) { return value == 0.0f; }),
            "split mapped FP8 PLE gathers zero rows from the later payload shard");

    const auto rejects_scale = [&](const fs::path& invalid_directory,
                                   fixture::Kv invalid_scale,
                                   const std::string& label) {
        auto invalid_model =
            make_split_mapped_fp8_model(invalid_directory,
                                        std::move(invalid_scale));
        strata::kernels::PleTable invalid_table;
        std::string invalid_error;
        require(!invalid_table.open_mapped(invalid_model, 1, invalid_error) &&
                    !invalid_table.is_open() &&
                    invalid_error.find("representable as float") !=
                        std::string::npos,
                label + ": " + invalid_error);
    };
    rejects_scale(directory / "overflow",
                  f64("strata.ple.scale",
                      static_cast<double>(
                          std::numeric_limits<float>::max()) * 2.0),
                  "mapped FP8 rejects a scale that overflows float");
    rejects_scale(directory / "underflow",
                  f64("strata.ple.scale",
                      std::numeric_limits<double>::denorm_min()),
                  "mapped FP8 rejects a positive scale that underflows to zero");
    rejects_scale(directory / "negative-signed",
                  i32("strata.ple.scale", -1),
                  "mapped FP8 rejects a negative signed-integer scale");
}

void run_model_metadata_test() {
    TempDir temp;
    strata::metal::Geometry geometry;
    std::string error;

    for (const std::uint32_t experts : {512u, 256u}) {
        const std::string family = experts == 256 ? "coder" : "qwen";
        SyntheticModel valid = make_model(temp.path / family, {}, {}, {}, false, experts);
        require(inspect(valid, geometry, error),
                "catalogue metadata accepts the " + family + " expert count: " + error);
    }

    constexpr std::array<std::string_view, 10> keys = {
        "general.architecture",
        "qwen4exp.block_count",
        "qwen4exp.embedding_length",
        "qwen4exp.attention.head_count",
        "qwen4exp.attention.head_count_kv",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
        "split.count",
        "split.tensors.count",
        "split.no",
    };
    std::size_t case_number = 0;
    for (const std::string_view key : keys) {
        const bool split = key.starts_with("split.");
        for (const bool single_shard : {false, true}) {
            if (!split && single_shard) continue;
            SyntheticModel changed = make_model(
                temp.path / ("changed-" + std::to_string(case_number++)),
                {}, {}, {}, split && !single_shard, 512, false, key, 16, {}, single_shard);
            const bool accepted = inspect(changed, geometry, error);
            require(!accepted && error.find(key) != std::string::npos,
                    "loader must name changed metadata key " + std::string(key) +
                        (single_shard ? " on one shard: " : " on split shards: ") + error);
        }
    }
}

void run_model_types_test() {
    TempDir temp;
    strata::metal::Geometry geometry;
    std::string error;
    const std::string expert_down = "blk.0.ffn_down_exps.weight";

    SyntheticModel valid = make_model(temp.path / "valid");
    check(inspect(valid, geometry, error), "all catalogue tensor formats pass inspection");

    SyntheticModel q5_down = make_model(temp.path / "q5-down", {}, {}, expert_down,
                                        false, kExperts, false, {}, 6);
    const bool q5_rejected = !inspect(q5_down, geometry, error);
    require(q5_rejected && error.find(expert_down) != std::string::npos &&
                error.find("q5_0") != std::string::npos,
            "loader names and rejects a synthetic Q5_0 expert-down tensor: " + error);

    SyntheticModel partial_block = make_model(temp.path / "partial-block", {}, {}, {},
                                               false, kExperts, false, {}, 16, expert_down);
    const bool block_rejected = !inspect(partial_block, geometry, error);
    require(block_rejected && error.find(expert_down) != std::string::npos &&
                error.find("quantization block geometry") != std::string::npos,
            "loader names and rejects a quantized tensor with a partial block: " + error);
}

void run_native_experts_reader_test() {
    TempDir temp;
    strata::metal::Geometry geometry;
    std::string error;
    SyntheticModel model = make_model(temp.path / "native-experts");

    write_native_experts(model, false, false);
    require(inspect(model, geometry, error), "native_experts.txt v3 is accepted: " + error);
    write_native_experts(model, true, false);
    require(inspect(model, geometry, error), "native_experts.txt v4 is accepted: " + error);

    const fs::path table_path = model.pack / "native_experts.txt";
    {
        std::fstream table(table_path, std::ios::in | std::ios::out);
        constexpr std::string_view prefix = "# strata native experts v";
        table.seekp(static_cast<std::streamoff>(prefix.size()));
        table.put('5');
        table.close();
        require(!table.fail(), "could not write unsupported native-expert version fixture");
    }
    require(!inspect(model, geometry, error) && error.find("native_experts.txt is v5") != std::string::npos,
            "native_experts.txt v5 is refused with an upgrade diagnostic: " + error);

    write_native_experts(model, false, false, true);
    require(!inspect(model, geometry, error) && error.find("malformed line") != std::string::npos,
            "native_experts.txt without role offsets is refused: " + error);

    write_native_experts(model, false, false, false, false, true);
    require(!inspect(model, geometry, error) &&
                error.find("blob or role types do not match the GGUF") != std::string::npos,
            "native_experts.txt with the wrong slab size is refused: " + error);

    write_native_experts(model, true, false, false, false, false, true);
    require(!inspect(model, geometry, error) &&
                error.find("invalid shard or byte size for layer 0") != std::string::npos,
            "native_experts.txt with an out-of-range shard reference is refused: " + error);

    write_native_experts(model, false, false);
    {
        std::ifstream original(table_path);
        std::string header;
        require(static_cast<bool>(std::getline(original, header)),
                "could not read native-expert header for truncated fixture");
        std::ofstream table(table_path, std::ios::trunc);
        table << header << '\n';
    }
    require(!inspect(model, geometry, error) &&
                error.find("is truncated or its dimensions do not match the model") != std::string::npos,
            "truncated native_experts.txt is refused: " + error);

    write_native_experts(model, false, false);
    {
        std::ofstream experts(model.pack / "experts.bin", std::ios::binary | std::ios::trunc);
        experts << "legacy PC expert pack\n";
    }
    std::string warning;
    require(inspect(model, geometry, error, false, nullptr, nullptr, &warning),
            "a legacy experts.bin beside the pack does not prevent GGUF loading: " + error);
    require(warning.find("experts.bin") != std::string::npos &&
                warning.find("ignoring") != std::string::npos,
            "ignoring experts.bin is reported to the caller: " + warning);
}

void run_pack_index_test() {
    TempDir temp;
    strata::metal::Geometry geometry;
    std::string error;

    SyntheticModel valid = make_model(temp.path / "valid-index");
    require(inspect(valid, geometry, error),
            "the supported 19-field native pack index is accepted: " + error);

    SyntheticModel canonical = make_model(temp.path / "canonical-code-bits");
    rewrite_index_row(canonical, "output_hc_norm.weight", 9, "4");
    require(!inspect(canonical, geometry, error) &&
                error.find("malformed float row for output_hc_norm.weight") != std::string::npos,
            "a canonical row with code_bits other than 8 is refused: " + error);

    SyntheticModel compat = make_model(temp.path / "compat-bf16");
    set_compat_bf16(compat, true);
    require(!inspect(compat, geometry, error) &&
                error.find("unsupported schema, tool or compat_bf16 setting") != std::string::npos,
            "conversions.json with compat_bf16 true is refused: " + error);

    SyntheticModel short_row = make_model(temp.path / "short-row");
    rewrite_index_row(short_row, "output_hc_norm.weight", 0, {}, true);
    require(!inspect(short_row, geometry, error) &&
                error.find("an index row does not have 19 fields") != std::string::npos,
            "an 18-field index row is refused: " + error);
}

void verify_expert_addresses(const SyntheticModel& fixture,
                             std::uint32_t expected_swift_layer,
                             bool expect_role_split) {
    strata::metal::Geometry geometry;
    strata::metal::WindowSources sources;
    std::string error;
    require(inspect(fixture, geometry, error, false, &sources),
            "WeightStore accepts the expert-address fixture: " + error);

    const strata::GgufModel model = strata::GgufModel::open(fixture.native.string());
    strata::metal::WindowPlan plan;
    require(strata::metal::plan_windows(sources, 1024ull * 1024 * 1024, plan, error),
            "window planner accepts the expert-address fixture: " + error);

    std::map<std::string, const strata::metal::WindowFileSource*> source_files;
    std::map<std::string,
             std::pair<const strata::metal::WindowFileSource*,
                       const strata::metal::WindowTensorSource*>> source_tensors;
    for (const strata::metal::WindowFileSource& source_file : sources.files) {
        require(source_files.emplace(source_file.path, &source_file).second,
                "WeightStore returned a duplicate expert source file");
        for (const strata::metal::WindowTensorSource& tensor : source_file.tensors) {
            require(source_tensors.emplace(tensor.tensor_name,
                                           std::make_pair(&source_file, &tensor)).second,
                    "WeightStore returned a duplicate tensor source " + tensor.tensor_name);
        }
    }

    using SliceKey = std::pair<std::string, std::uint64_t>;
    using PlannedSlice = std::pair<const strata::metal::FileWindowPlan*,
                                   const strata::metal::WindowSlice*>;
    std::map<std::string, const strata::metal::FileWindowPlan*> plan_files;
    std::map<SliceKey, PlannedSlice> expert_slices;
    for (const strata::metal::FileWindowPlan& plan_file : plan.files) {
        require(plan_files.emplace(plan_file.path, &plan_file).second,
                "window planner returned a duplicate expert source file");
        for (const strata::metal::WindowSlice& slice : plan_file.slices) {
            if (slice.expert_index == strata::metal::kNoExpertIndex) continue;
            require(expert_slices.emplace(SliceKey{slice.tensor_name, slice.expert_index},
                                          PlannedSlice{&plan_file, &slice}).second,
                    "window planner returned a duplicate expert slice for " + slice.tensor_name);
        }
    }
    require(source_files.size() == model.size() && plan_files.size() == model.size(),
            "WeightStore and planner must retain every GGUF shard");

    constexpr std::array<std::string_view, 3> roles = {"gate", "up", "down"};
    std::uint64_t verified_slices = 0;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (const std::string_view role : roles) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" +
                                     std::string(role) + "_exps.weight";
            std::size_t shard_index = 0;
            const strata::TensorInfo* tensor = model.find(name, &shard_index);
            require(tensor != nullptr && shard_index < model.size(),
                    "GGUF model is missing expert tensor " + name);
            const strata::GgufFile& shard = model.shard(shard_index);
            require(model.in_bounds(*tensor, shard_index),
                    "GGUF expert tensor is outside its shard: " + name);
            require(tensor->shape.size() == 3 && tensor->shape[2] == fixture.n_expert,
                    "GGUF expert tensor has the wrong expert dimension: " + name);

            const auto source = source_tensors.find(name);
            require(source != source_tensors.end(),
                    "WeightStore omitted expert tensor " + name);
            const strata::metal::WindowFileSource& source_file = *source->second.first;
            const strata::metal::WindowTensorSource& source_tensor = *source->second.second;
            require(source_file.path == shard.path() &&
                        source_file.file_size == shard.file_size() &&
                        fs::path(source_file.path).filename() == fixture.tensor_shards.at(name),
                    "WeightStore resolved expert tensor to the wrong GGUF shard: " + name);
            const auto planned_file = plan_files.find(shard.path());
            require(planned_file != plan_files.end() &&
                        planned_file->second->path == source_file.path,
                    "window planner omitted the expert tensor's source shard: " + name);

            const std::uint64_t payload = strata::tensor_payload_bytes(*tensor);
            require(payload > 0 &&
                        tensor->offset <= std::numeric_limits<std::uint64_t>::max() - shard.data_start(),
                    "GGUF expert tensor has an invalid source address: " + name);
            const std::uint64_t expected_base = shard.data_start() + tensor->offset;
            require(payload <= std::numeric_limits<std::uint64_t>::max() - expected_base,
                    "GGUF expert tensor end address overflows: " + name);
            require(source_tensor.file_offset == expected_base &&
                        source_tensor.byte_count == payload &&
                        payload % fixture.n_expert == 0 &&
                        source_tensor.expert_slab_bytes == payload / fixture.n_expert,
                    "WeightStore expert base or slab differs from GGUF: " + name);

            for (std::uint64_t expert = 0; expert < fixture.n_expert; ++expert) {
                const auto found = expert_slices.find({name, expert});
                require(found != expert_slices.end(),
                        "window planner omitted expert " + std::to_string(expert) + " of " + name);
                const strata::metal::FileWindowPlan& file_plan = *found->second.first;
                const strata::metal::WindowSlice& slice = *found->second.second;
                require(file_plan.path == shard.path() &&
                            expert <= (std::numeric_limits<std::uint64_t>::max() - expected_base) /
                                          source_tensor.expert_slab_bytes,
                        "window planner assigned an invalid shard or expert address: " + name);
                const std::uint64_t expected_address =
                    expected_base + expert * source_tensor.expert_slab_bytes;
                require(slice.file_offset == expected_address &&
                            slice.byte_count == source_tensor.expert_slab_bytes,
                        "expert slice address differs from data_start + tensor offset + expert * slab: " +
                            name + " expert " + std::to_string(expert));
                require(slice.window_index < file_plan.windows.size(),
                        "expert slice refers to a missing file window: " + name);
                const strata::metal::FileWindow& window =
                    file_plan.windows[static_cast<std::size_t>(slice.window_index)];
                require(slice.file_offset >= window.start && slice.file_offset <= window.end &&
                            slice.window_offset == slice.file_offset - window.start &&
                            slice.byte_count <= window.end - slice.file_offset,
                        "expert slice does not map inside its selected file window: " + name);
                ++verified_slices;
            }
        }
    }

    const std::uint64_t expected_slices = static_cast<std::uint64_t>(kLayers) * roles.size() *
                                          fixture.n_expert;
    require(verified_slices == expected_slices && expert_slices.size() == expected_slices,
            "expert-address fixture did not cover each expected layer, role and expert");

    if (expected_swift_layer < kLayers) {
        for (const std::string_view role : roles) {
            const std::string name = "blk." + std::to_string(expected_swift_layer) +
                                     ".ffn_" + std::string(role) + "_exps.weight";
            require(fixture.tensor_shards.at(name) == fixture.shard_names[1] &&
                        source_tensors.at(name).first->path ==
                            fixture.shard_paths[1].string(),
                    "Swift's straddling layer resolves " + std::string(role) +
                        " experts to shard 2");
        }
    }

    if (expect_role_split) {
        const auto shard_for = [&](std::string_view role) -> const std::string& {
            const std::string name = "blk.13.ffn_" + std::string(role) + "_exps.weight";
            return source_tensors.at(name).first->path;
        };
        require(shard_for("gate") != shard_for("up") &&
                    shard_for("up") == shard_for("down") &&
                    fs::path(shard_for("gate")).filename() == fixture.tensor_shards.at(
                        "blk.13.ffn_gate_exps.weight") &&
                    fs::path(shard_for("up")).filename() == fixture.tensor_shards.at(
                        "blk.13.ffn_up_exps.weight") &&
                    fs::path(shard_for("down")).filename() == fixture.tensor_shards.at(
                        "blk.13.ffn_down_exps.weight"),
                "v4 per-role fixture resolves each expert role to its recorded shard");
    }
}

void run_expert_address_test() {
    TempDir temp;

    SyntheticModel single = make_model(temp.path / "single-512");
    verify_expert_addresses(single, kLayers, false);

    SyntheticModel swift_iq2 = make_model(temp.path / "swift-iq2-xs-like-512", {}, {}, {}, true);
    verify_expert_addresses(swift_iq2, 13, false);

    SyntheticModel swift_iq3 =
        make_model(temp.path / "swift-iq3-xxs-like-512", {}, {}, {}, true, kExperts,
                   false, {}, 16, {}, false, 12);
    verify_expert_addresses(swift_iq3, 12, false);

    SyntheticModel v4_role_split =
        make_model(temp.path / "v4-per-role-split-512", {}, {}, {}, true, kExperts, true);
    write_native_experts(v4_role_split, true, false);
    verify_expert_addresses(v4_role_split, kLayers, true);

    SyntheticModel coder = make_model(temp.path / "coder-256", {}, {}, {}, false, 256);
    verify_expert_addresses(coder, kLayers, false);
}

void write_json_array(const std::uint64_t* values, std::size_t count) {
    std::cout << '[';
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_ple_constants() {
    const strata::kernels::PleConsts constants = strata::kernels::ple_artifact_consts();
    std::cout << '{'
              << "\"embedding_length\":" << strata::kernels::NG_N_EMBD
              << ",\"metal_embedding_length\":" << N_EMBD
              << ",\"embedding_length_per_layer_input\":" << strata::kernels::PLE_HEAD_DIM
              << ",\"metal_embedding_length_per_layer_input\":" << PLE_HEAD_DIM
              << ",\"ngram_size\":" << strata::kernels::NGRAM_SIZE
              << ",\"metal_ngram_size\":" << PLE_NGRAM_SIZE
              << ",\"heads_per_ngram\":" << strata::kernels::HEADS_PER_NGRAM
              << ",\"metal_heads_per_ngram\":" << PLE_HEADS_PER_NGRAM
              << ",\"ple_heads\":" << strata::kernels::PLE_N_HEADS
              << ",\"metal_ple_heads\":" << PLE_HEADS
              << ",\"conv_kernel\":" << strata::kernels::PLE_CONV_KERNEL
              << ",\"metal_conv_kernel\":" << PLE_CONV_KERNEL
              << ",\"ple_layers\":[" << PLE_LAYER << ']'
              << ",\"metal_ple_layers\":[" << PLE_LAYER << ']'
              << ",\"eos_token_id\":" << strata::kernels::PLE_EOS_TOKEN_ID
              << ",\"metal_eos_token_id\":" << PLE_EOS
              << ",\"image_token_id\":" << IMAGE_TOKEN_ID
              << ",\"table_rows\":" << strata::kernels::PLE_TABLE_ROWS
              << ",\"metal_table_rows\":" << PLE_TABLE_ROWS
              << ",\"metal_vocab_size\":" << VOCAB_SIZE
              << ",\"metal_history_rows\":" << PLE_HISTORY_ROWS
              << ",\"layer_multipliers\":";
    write_json_array(constants.mult, strata::kernels::NGRAM_SIZE);
    std::cout << ",\"head_vocab_sizes\":";
    write_json_array(constants.vocab, strata::kernels::PLE_N_HEADS);
    std::cout << ",\"head_offsets\":";
    write_json_array(constants.offset, strata::kernels::PLE_N_HEADS);
    std::cout << "}\n";
}

void run() {
    TempDir temp;
    SyntheticModel model = make_model(temp.path);
    strata::metal::Geometry geometry;
    strata::metal::WindowSources window_sources;
    std::string error;

    check(inspection_rejects_writer_lock(model, model.shard_paths[0], error),
          "inspection refuses a cooperating writer holding an exclusive GGUF lock");
    check(inspection_rejects_writer_lock(model, model.pack / "dense.bin", error),
          "inspection refuses a cooperating writer holding an exclusive dense-pack lock");
    const fs::path source_set_lock = artifact_lock_path(model.native.parent_path());
    const fs::path pack_set_lock = artifact_lock_path(model.pack);
    ensure_artifact_lock_exists(source_set_lock);
    ensure_artifact_lock_exists(pack_set_lock);
    check(inspection_rejects_writer_lock(model, source_set_lock, error),
          "inspection refuses a writer holding the source-set generation lock");
    check(inspection_rejects_writer_lock(model, pack_set_lock, error),
          "inspection refuses a writer holding the pack generation lock");
    const fs::path pending_marker = model.pack / ".strata-generation-pending";
    {
        std::ofstream marker(pending_marker);
        require(static_cast<bool>(marker), "create interrupted-generation marker fixture");
    }
    strata::metal::Geometry interrupted_geometry;
    strata::metal::WindowSources interrupted_sources;
    require(!inspect(model, interrupted_geometry, error, false, &interrupted_sources) &&
                error.find("artifact generation is incomplete") != std::string::npos,
            "WeightStore must refuse an interrupted pack generation before reading metadata");
    fs::remove(pending_marker);

    const bool valid_v3 = inspect(model, geometry, error, false, &window_sources);
    if (!valid_v3) std::cerr << "  valid v3 rejected: " << error << '\n';
    check(valid_v3, "complete 1,224-tensor v3 model inspection succeeds without reading payloads");
    check_shared_mapped_ple_table(model);
    check_split_mapped_fp8_ple_table(temp.path / "split-fp8");
    check(window_sources.files.size() == 1 &&
              window_sources.inspected_file_locks.size() == 4 &&
              std::all_of(window_sources.inspected_file_locks.begin(),
                          window_sources.inspected_file_locks.end(),
                          [](const auto& lock) {
                              return static_cast<bool>(lock.lock_lease());
                          }) &&
              std::none_of(window_sources.files[0].tensors.begin(),
                           window_sources.files[0].tensors.end(), [](const auto& tensor) {
                               return tensor.tensor_name == "per_layer_token_embd.weight" ||
                                      tensor.tensor_name == "blk.0.ffn_gate_inp.weight";
                           }),
          "window sources retain inspection locks and omit PLE and converted router payloads");
    check(shared_inspection_lock_blocks_writer(model.shard_paths[0]) &&
              shared_inspection_lock_blocks_writer(model.pack / "dense.bin") &&
              shared_inspection_lock_blocks_writer(source_set_lock) &&
              shared_inspection_lock_blocks_writer(pack_set_lock),
          "inspection file and generation locks remain held in WindowSources until storage opens");
    struct stat shard_status {};
    struct stat dense_status {};
    check(::stat(model.shard_paths[0].c_str(), &shard_status) == 0 &&
              ::stat(window_sources.dense_pack_path.c_str(), &dense_status) == 0 &&
              window_sources.files[0].file_device_id ==
                  static_cast<std::uint64_t>(shard_status.st_dev) &&
              window_sources.files[0].file_object_id ==
                  static_cast<std::uint64_t>(shard_status.st_ino) &&
              window_sources.dense_pack_device_id ==
                  static_cast<std::uint64_t>(dense_status.st_dev) &&
              window_sources.dense_pack_object_id ==
                  static_cast<std::uint64_t>(dense_status.st_ino),
          "WeightStore pins the inspected GGUF and dense-pack file identities");
    const auto has_exclusion = [&window_sources](const std::string& name,
                                                  strata::metal::WindowTensorExclusionReason reason) {
        const auto& excluded = window_sources.files[0].excluded_tensors;
        return std::any_of(excluded.begin(), excluded.end(), [&](const auto& tensor) {
            return tensor.tensor_name == name && tensor.reason == reason;
        });
    };
    check(window_sources.files[0].excluded_tensors.size() == 2 &&
              has_exclusion("per_layer_token_embd.weight",
                            strata::metal::WindowTensorExclusionReason::ple) &&
              has_exclusion("blk.0.ffn_gate_inp.weight",
                            strata::metal::WindowTensorExclusionReason::converted),
          "WeightStore accounts for every tensor omitted from no-copy windows");
    std::uint64_t expected_router_pack_offset = 0;
    std::uint64_t expected_router_pack_bytes = 0;
    bool found_router_pack_row = false;
    {
        std::ifstream index(model.pack / "index.txt");
        std::string line;
        while (std::getline(index, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::istringstream row(line);
            std::string name;
            std::uint64_t file = 0, kind = 0, offset = 0, bytes = 0;
            if (!(row >> name >> file >> kind >> offset >> bytes)) break;
            if (name == "blk.0.ffn_gate_inp.weight") {
                expected_router_pack_offset = offset;
                expected_router_pack_bytes = bytes;
                found_router_pack_row = kind == 4 && file == 0;
                break;
            }
        }
    }
    check(window_sources.dense_pack_path == (model.pack / "dense.bin").string() &&
              fs::file_size(window_sources.dense_pack_path) ==
                  window_sources.dense_pack_file_size &&
              window_sources.dense_pack_tensors.size() == 1 &&
              found_router_pack_row &&
              window_sources.dense_pack_tensors[0].tensor_name == "blk.0.ffn_gate_inp.weight" &&
              window_sources.dense_pack_tensors[0].file_offset == expected_router_pack_offset &&
              window_sources.dense_pack_tensors[0].file_offset > 0 &&
              window_sources.dense_pack_tensors[0].source_byte_count ==
                  expected_router_pack_bytes &&
              window_sources.dense_pack_tensors[0].byte_count ==
                  geometry.dense_pack_bytes &&
              window_sources.dense_pack_tensors[0].file_offset <=
                  window_sources.dense_pack_file_size &&
              window_sources.dense_pack_tensors[0].source_byte_count <=
                  window_sources.dense_pack_file_size -
                      window_sources.dense_pack_tensors[0].file_offset &&
              window_sources.dense_pack_tensors[0].kind == 4,
          "WeightStore exposes validated converted-router dense.bin ranges");
    const auto dense_lock = std::find_if(
        window_sources.inspected_file_locks.begin(),
        window_sources.inspected_file_locks.end(),
        [&](const strata::metal::WindowSourceFileLock& lock) {
            return lock.path == window_sources.dense_pack_path;
        });
    auto remapped_rows = window_sources.dense_pack_tensors;
    ++remapped_rows[0].file_offset;
    check(dense_lock != window_sources.inspected_file_locks.end() &&
              dense_lock->lock_lease()->matches_dense_pack_catalog(
                  window_sources.dense_pack_tensors) &&
              !dense_lock->lock_lease()->matches_dense_pack_catalog(remapped_rows),
          "WeightStore's dense-pack lease retains an immutable index-row catalog");
    std::size_t expert_sources = 0;
    bool expert_slab_sizes_match = true;
    for (const auto& tensor : window_sources.files[0].tensors) {
        if (tensor.expert_slab_bytes != 0) {
            ++expert_sources;
            expert_slab_sizes_match &=
                tensor.byte_count == tensor.expert_slab_bytes * kExperts;
        }
    }
    check(expert_slab_sizes_match, "expert sources carry the per-expert slab size");
    check(expert_sources == 144, "all three expert roles from 48 layers reach the window planner");
    strata::metal::WindowPlan window_plan;
    check(strata::metal::plan_windows(window_sources, 1024ull * 1024 * 1024,
                                     window_plan, error) &&
              window_plan.files.size() == 1 && !window_plan.files[0].windows.empty() &&
              window_plan.files[0].slices.size() ==
                  window_sources.files[0].tensors.size() - expert_sources +
                      expert_sources * kExperts,
          "WeightStore tensor and expert-slab mappings feed the bounded planner");
    check(geometry.family == "qwen" && geometry.model == "tiny" && geometry.n_expert == kExperts &&
              geometry.n_layers == kLayers && geometry.n_qsa_layers == 12,
          "geometry identity and Qwen dimensions are derived");
    const auto has_type = [&geometry](const std::string& type) {
        return std::find(geometry.types.begin(), geometry.types.end(), type) != geometry.types.end();
    };
    check(geometry.shards.size() == 1 && geometry.shards[0].bytes == model.source_bytes &&
              geometry.layers.size() == kLayers && geometry.layers[0].blob_bytes == kExpertBlob &&
              geometry.dense_pack_bytes == 2'560ull * kExperts * 2 &&
              has_type("bf16") && has_type("f32") &&
              has_type("f16") && has_type("q2_0") && has_type("q3_k") && has_type("q5_k") &&
              has_type("q4_k") && has_type("iq4_nl"),
          "geometry contains shard, layer and tensor facts");

    SyntheticModel split = make_model(temp.path / "split", {}, {}, {}, true);
    const strata::GgufModel split_model = strata::GgufModel::open(split.native.string());
    check(inspect(split, geometry, error) && geometry.shards.size() == 2 &&
              geometry.shards[0].name == "tiny-00001-of-00002.gguf" &&
              geometry.shards[1].name == "tiny-00002-of-00002.gguf" &&
              split_model.shard(0).tensors().size() == 342 &&
              split_model.shard(1).tensors().size() == 882,
          "Swift-style 342/882 split resolves tensors and v3 expert rows across both shards");
    check(split.ple_gguf == split.shard_paths.front(),
          "Swift-style PLE metadata is resolved from the first shard");
    write_native_experts(split, true, false);
    check(inspect(split, geometry, error) && geometry.shards.size() == 2,
          "v4 per-role expert rows resolve roles across a split model");
    SyntheticModel split_roles = make_model(temp.path / "split-roles", {}, {}, {}, true, kExperts, true);
    const strata::GgufModel split_role_model = strata::GgufModel::open(split_roles.native.string());
    check(split_role_model.shard(0).tensors().size() == 342 &&
              split_role_model.shard(1).tensors().size() == 882,
          "per-role fixture preserves Swift shard tensor counts");
    check(!inspect(split_roles, geometry, error) && error.find("invalid shard or byte size") != std::string::npos,
          "v3 rejects expert roles distributed across two shards in one layer");
    write_native_experts(split_roles, true, false);
    check(inspect(split_roles, geometry, error),
          "v4 resolves a layer whose gate is in a different shard from up and down");

    SyntheticModel missing = make_model(temp.path / "missing", "output_hc_up.weight");
    check(!inspect(missing, geometry, error) &&
              error.find("missing required Qwen tensor output_hc_up.weight") != std::string::npos,
          "inventory rejects a missing non-expert tensor before planning");
    SyntheticModel wrong_shape =
        make_model(temp.path / "wrong-shape", {}, "output_hc_up.weight");
    check(!inspect(wrong_shape, geometry, error) &&
              error.find("unexpected shape: output_hc_up.weight") != std::string::npos,
          "inventory rejects a tensor with the wrong architecture shape");
    SyntheticModel wrong_type = make_model(temp.path / "wrong-type", {}, {}, "blk.0.attn_qkv.weight");
    const bool wrong_type_rejected = !inspect(wrong_type, geometry, error);
    if (wrong_type_rejected && error.find("disallowed type") == std::string::npos)
        std::cerr << "  wrong-type diagnostic: " << error << '\n';
    check(wrong_type_rejected && error.find("disallowed type") != std::string::npos,
          "inventory rejects a valid GGML type assigned to the wrong tensor role");
    SyntheticModel unsupported_experts =
        make_model(temp.path / "unsupported-experts", {}, {}, {}, false, 10);
    check(!inspect(unsupported_experts, geometry, error) &&
              error.find("expert_count must be 256 (Coder) or 512") != std::string::npos,
          "loader rejects expert counts outside the frozen catalogue");
    SyntheticModel coder = make_model(temp.path / "coder", {}, {}, {}, false, 256);
    check(inspect(coder, geometry, error) && geometry.n_expert == 256,
          "loader accepts the frozen Coder expert count");

    write_native_experts(model, true, false);
    check(inspect(model, geometry, error), "v4 per-role shard table succeeds");
    write_native_experts(model, true, false, false, true);
    check(!inspect(model, geometry, error) && error.find("more than three") != std::string::npos,
          "v4 table rejects an extra role shard");

    write_native_experts(model, false, true);
    check(!inspect(model, geometry, error) && error.find("role offset") != std::string::npos,
          "native table rejects a mismatched absolute tensor offset");
    write_native_experts(model, false, false, true);
    check(!inspect(model, geometry, error) && error.find("malformed line") != std::string::npos,
          "native table requires all three role offsets");
    write_native_experts(model, false, false);

    write_index(model, false);
    check(!inspect(model, geometry, error) && error.find("missing GGUF tensor") != std::string::npos,
          "index rejects a missing native tensor row");
    write_index(model, true);
    write_conversions(model, false);
    check(!inspect(model, geometry, error) && error.find("conversions.json") != std::string::npos,
          "conversions metadata rejects truncated JSON");
    write_conversions(model);
    corrupt_index_float_row_as_native(model, "output_hc_down.weight");
    check(!inspect(model, geometry, error) && error.find("malformed native row") != std::string::npos,
          "index rejects a float tensor falsely marked as native");
    write_index(model, true);
    write_conversions(model, true, false, true);
    check(!inspect(model, geometry, error) && error.find("malformed float row") != std::string::npos,
          "index requires the conversion record for a converted router");
    write_conversions(model);
    write_conversions(model, true, true);
    check(!inspect(model, geometry, error) && error.find("unsupported schema") == std::string::npos &&
              error.find("conversion record") != std::string::npos,
          "conversion metadata rejects a destination form the engine cannot read");
    write_conversions(model);
    write_conversions(model, true, false, false, true);
    check(!inspect(model, geometry, error) &&
              error.find("conversion record does not match tensor blk.0.ffn_gate_inp.weight") !=
                  std::string::npos,
          "conversion metadata rejects a lossy router conversion");
    write_conversions(model);
    write_conversions(model, true, false, false, false, true);
    check(!inspect(model, geometry, error) &&
              error.find("conversion record does not match tensor blk.0.ffn_gate_inp.weight") !=
                  std::string::npos,
          "conversion metadata rejects a method not emitted for exact router conversion");
    write_conversions(model);

    make_mtp_runtime(model);
    fs::resize_file(model.mtp / "experts.bin", kMtpExpertsBytes - 1u);
    check(!inspect(model, geometry, error, true) &&
              error.find("MTP runtime file sizes do not match") != std::string::npos,
          "MTP inspection rejects an experts.bin truncated by one byte");
    make_mtp_runtime(model);
    const fs::path mtp_set_lock = artifact_lock_path(model.mtp);
    ensure_artifact_lock_exists(mtp_set_lock);
    const fs::path fifo_experts_path = model.mtp / "experts.bin";
    const fs::path fifo_experts_backup = model.mtp / "experts.fixture-backup";
    fs::rename(fifo_experts_path, fifo_experts_backup);
    require(::mkfifo(fifo_experts_path.c_str(), 0600) == 0,
            "create MTP experts FIFO fixture");
    ::alarm(2);
    const bool fifo_experts_accepted = inspect(model, geometry, error, true);
    ::alarm(0);
    const std::string fifo_experts_error = error;
    fs::remove(fifo_experts_path);
    fs::rename(fifo_experts_backup, fifo_experts_path);
    check(!fifo_experts_accepted &&
              fifo_experts_error.find("MTP runtime file is not a regular file") !=
                  std::string::npos,
          "MTP inspection rejects an experts FIFO without blocking");
    const fs::path fifo_dense_index_path = model.mtp / "dense.txt";
    const fs::path fifo_dense_index_backup = model.mtp / "dense-index.fixture-backup";
    fs::rename(fifo_dense_index_path, fifo_dense_index_backup);
    require(::mkfifo(fifo_dense_index_path.c_str(), 0600) == 0,
            "create MTP dense-index FIFO fixture");
    ::alarm(2);
    const bool fifo_dense_index_accepted = inspect(model, geometry, error, true);
    ::alarm(0);
    const std::string fifo_dense_index_error = error;
    fs::remove(fifo_dense_index_path);
    fs::rename(fifo_dense_index_backup, fifo_dense_index_path);
    check(!fifo_dense_index_accepted &&
              fifo_dense_index_error.find("not a regular file") != std::string::npos,
          "MTP inspection rejects a dense-index FIFO without blocking");
    const fs::path fifo_draft_vocab_path = model.mtp / "draft_vocab.bin";
    require(::mkfifo(fifo_draft_vocab_path.c_str(), 0600) == 0,
            "create optional MTP draft-vocabulary FIFO fixture");
    ::alarm(2);
    const bool fifo_draft_vocab_accepted = inspect(model, geometry, error, true);
    ::alarm(0);
    const std::string fifo_draft_vocab_error = error;
    fs::remove(fifo_draft_vocab_path);
    check(!fifo_draft_vocab_accepted &&
              fifo_draft_vocab_error.find("not a regular file") != std::string::npos,
          "MTP inspection rejects an optional draft-vocabulary FIFO without blocking");
    check(inspection_rejects_writer_lock(model, mtp_set_lock, error, true),
          "inspection refuses a writer holding the MTP generation lock");
    check(inspect(model, geometry, error, true) && geometry.mtp_bytes == 823'887'872 &&
              geometry.mtp_bind_bytes == 16'442'888,
          "MTP runtime validates with the optional draft vocabulary absent");
    strata::metal::MtpRuntimeSources mtp_sources;
    const fs::path canonical_mtp = fs::canonical(model.mtp);
    check(inspect(model, geometry, error, true, nullptr, &mtp_sources),
          "MTP source inventory inspection succeeds");
    require(mtp_sources.catalog != nullptr, "MTP source inventory is present after inspection");
    check(mtp_sources.catalog->directory == canonical_mtp.string() &&
              mtp_sources.catalog->experts_path ==
                  (canonical_mtp / "experts.bin").string() &&
              mtp_sources.catalog->experts_file_size == kMtpExpertsBytes &&
              mtp_sources.catalog->dense_path == (canonical_mtp / "dense.bin").string() &&
              mtp_sources.catalog->dense_file_size == kMtpDenseBytes &&
              mtp_sources.catalog->draft_vocab_path ==
                  (canonical_mtp / "draft_vocab.bin").string() &&
              mtp_sources.catalog->draft_vocab_file_size == 0 &&
              mtp_sources.catalog->draft_vocab_ids.empty(),
          "MTP source inventory records the expected files and optional vocab state");
    check(mtp_catalog_matches_expected(*mtp_sources.catalog) &&
              mtp_sources.catalog->dense_tensors.front().tensor_name ==
                  "fc_embedding.weight" &&
              mtp_sources.catalog->dense_tensors.front().kind ==
                  strata::metal::MtpDenseWeightKind::q8_0 &&
              mtp_sources.catalog->dense_tensors.front().rows == 2560 &&
              mtp_sources.catalog->dense_tensors.front().columns == 2560 &&
              mtp_sources.catalog->dense_tensors.back().tensor_name ==
                  "self_attn.indexer.k_layernorm.weight" &&
              mtp_sources.catalog->dense_tensors.back().kind ==
                  strata::metal::MtpDenseWeightKind::f32 &&
              mtp_sources.catalog->dense_tensors.back().file_offset +
                      mtp_sources.catalog->dense_tensors.back().byte_count ==
                  kMtpDenseBytes,
          "MTP inventory matches every field of all 29 expected dense rows");
    check(mtp_sources.generation_lock.path == mtp_set_lock.string() &&
              mtp_sources.generation_lock.lock_lease() != nullptr &&
              mtp_sources.generation_lock.lock_lease()->matches_mtp_catalog(mtp_sources) &&
              shared_inspection_lock_blocks_writer(mtp_set_lock),
          "MTP inventory retains its bound generation-lock lease");
    strata::metal::MtpRuntimeLayout mtp_layout;
    constexpr std::uint64_t kMtpTestMaxBuffer = 64ull * 1024 * 1024;
    check(strata::metal::build_mtp_runtime_layout(
              mtp_sources, kMtpTestMaxBuffer, mtp_layout, error),
          "MTP layout fits the synthetic runtime into bounded Metal windows");
    require(mtp_layout.mapped_file_bytes ==
                kMtpExpertsBytes + kMtpDenseBytes &&
                mtp_layout.experts.size() == strata::metal::kMtpExpertCount &&
                mtp_layout.dense_weights.size() == mtp_sources.catalog->dense_tensors.size(),
            "MTP layout reports exact file bytes and complete tensor bindings");
    for (std::size_t window_index = 0;
         window_index < mtp_layout.expert_windows.size(); ++window_index) {
        const auto& window = mtp_layout.expert_windows[window_index];
        require(window.start % strata::metal::kWindowPageBytes == 0 &&
                    window.end % strata::metal::kWindowPageBytes == 0 &&
                    window.end > window.start &&
                    window.end - window.start <= kMtpTestMaxBuffer &&
                    window.start %
                            (8 * strata::metal::kMtpExpertBlobBytes) ==
                        0,
                "MTP expert windows are page- and blob-boundary aligned");
    }
    for (std::size_t expert_index = 0; expert_index < mtp_layout.experts.size();
         ++expert_index) {
        const auto& expert = mtp_layout.experts[expert_index];
        require(expert.expert_index == expert_index &&
                    expert.byte_count == strata::metal::kMtpExpertBlobBytes &&
                    expert.window_index < mtp_layout.expert_windows.size() &&
                    mtp_layout.expert_windows[expert.window_index].start +
                            expert.window_offset ==
                        expert_index * strata::metal::kMtpExpertBlobBytes &&
                    expert.window_offset + expert.byte_count <=
                        mtp_layout.expert_windows[expert.window_index].end -
                            mtp_layout.expert_windows[expert.window_index].start,
                "each MTP expert maps wholly into one bounded window");
    }
    for (const auto& window : mtp_layout.dense_windows) {
        require(window.start % strata::metal::kWindowPageBytes == 0 &&
                    window.end % strata::metal::kWindowPageBytes == 0 &&
                    window.end > window.start &&
                    window.end - window.start <= kMtpTestMaxBuffer,
                "MTP dense windows are page aligned and maxBufferLength bounded");
    }
    for (std::size_t tensor_index = 0;
         tensor_index < mtp_layout.dense_weights.size(); ++tensor_index) {
        const auto& binding = mtp_layout.dense_weights[tensor_index];
        const auto& source = mtp_sources.catalog->dense_tensors[tensor_index];
        require(binding.tensor_name == source.tensor_name &&
                    binding.kind == source.kind && binding.rows == source.rows &&
                    binding.columns == source.columns &&
                    binding.file_offset == source.file_offset &&
                    binding.byte_count == source.byte_count &&
                    binding.window_index < mtp_layout.dense_windows.size() &&
                    mtp_layout.dense_windows[binding.window_index].start +
                            binding.window_offset ==
                        binding.file_offset &&
                    binding.window_offset + binding.byte_count <=
                        mtp_layout.dense_windows[binding.window_index].end -
                            mtp_layout.dense_windows[binding.window_index].start,
                "each MTP dense tensor maps wholly into one bounded window");
    }
    strata::metal::MtpRuntimeLayout rejected_mtp_layout;
    rejected_mtp_layout = mtp_layout;
    check(!strata::metal::build_mtp_runtime_layout(
              mtp_sources, 24ull * 1024 * 1024, rejected_mtp_layout, error) &&
              error.find("cannot hold dense tensor") != std::string::npos &&
              rejected_mtp_layout.expert_windows.empty() &&
              rejected_mtp_layout.experts.empty() &&
              rejected_mtp_layout.dense_windows.empty() &&
              rejected_mtp_layout.dense_weights.empty() &&
              rejected_mtp_layout.mapped_file_bytes == 0,
          "MTP layout rejects a maxBufferLength smaller than the largest dense tensor");
    strata::metal::MtpRuntimeSources forged_mtp_sources = mtp_sources;
    strata::metal::MtpRuntimeCatalog forged_catalog = *mtp_sources.catalog;
    forged_catalog.dense_tensors.front().file_offset += 256;
    forged_mtp_sources.catalog =
        std::make_shared<const strata::metal::MtpRuntimeCatalog>(std::move(forged_catalog));
    check(!mtp_sources.generation_lock.lock_lease()->matches_mtp_catalog(forged_mtp_sources),
          "MTP generation lease rejects a substituted dense-row catalog");
    check(!strata::metal::build_mtp_runtime_layout(
              forged_mtp_sources, kMtpTestMaxBuffer, rejected_mtp_layout, error) &&
              error.find("does not match its live generation-lock lease") !=
                  std::string::npos,
          "MTP layout refuses a substituted catalog before planning");
    mtp_sources = {};
    const fs::path mtp_alias = model.native.parent_path() / "mtp-runtime-alias";
    const fs::path mtp_retarget = model.native.parent_path() / "mtp-runtime-retarget";
    fs::create_directory(mtp_retarget);
    fs::create_directory_symlink(model.mtp, mtp_alias);
    auto aliased_options = options_for(model);
    aliased_options.mtp_dir = mtp_alias.string();
    strata::metal::Geometry aliased_geometry;
    strata::metal::MtpRuntimeSources aliased_mtp_sources;
    const fs::path canonical_mtp_lock = artifact_lock_path(model.mtp);
    check(strata::metal::WeightStore::inspect(
              aliased_options, aliased_geometry, error, nullptr, &aliased_mtp_sources) &&
              aliased_mtp_sources.catalog->directory == canonical_mtp.string() &&
              aliased_mtp_sources.catalog->experts_path ==
                  (canonical_mtp / "experts.bin").string() &&
              aliased_mtp_sources.generation_lock.path == canonical_mtp_lock.string(),
          "MTP inspection resolves aliases before reading and recording source paths");
    fs::remove(mtp_alias);
    fs::create_directory_symlink(mtp_retarget, mtp_alias);
    check(aliased_mtp_sources.catalog->directory == canonical_mtp.string() &&
              aliased_mtp_sources.catalog->dense_path ==
                  (canonical_mtp / "dense.bin").string() &&
              aliased_mtp_sources.generation_lock.path == canonical_mtp_lock.string() &&
              shared_inspection_lock_blocks_writer(canonical_mtp_lock),
          "retargeting an MTP directory alias cannot redirect a retained catalog");
    aliased_mtp_sources = {};
    fs::remove(mtp_alias);
    fs::remove(mtp_retarget);
    auto invalid_identity_options = options_for(model);
    invalid_identity_options.mtp_dir = model.mtp.string();
    invalid_identity_options.model = "invalid model";
    strata::metal::Geometry invalid_identity_geometry;
    strata::metal::MtpRuntimeSources rejected_mtp_sources;
    check(!strata::metal::WeightStore::inspect(
              invalid_identity_options, invalid_identity_geometry, error, nullptr,
              &rejected_mtp_sources) &&
              rejected_mtp_sources.catalog == nullptr &&
              rejected_mtp_sources.generation_lock.lock_lease() == nullptr,
          "failed final model-identity validation does not publish an MTP catalog");
    check(inspect(model, geometry, error, false, nullptr, &mtp_sources) &&
              mtp_sources.catalog == nullptr &&
              mtp_sources.generation_lock.lock_lease() == nullptr,
          "MTP inspection output is empty when the runtime is not requested");
    fs::path mtp_pending_marker = model.mtp / ".strata-generation-pending";
    {
        std::ofstream marker(mtp_pending_marker);
        require(static_cast<bool>(marker), "create interrupted MTP generation marker fixture");
    }
    strata::metal::Geometry interrupted_mtp_geometry;
    require(!inspect(model, interrupted_mtp_geometry, error, true) &&
                error.find("artifact generation is incomplete") != std::string::npos,
            "WeightStore refuses an interrupted MTP generation");
    fs::remove(mtp_pending_marker);
    strata::metal::WindowSources mtp_window_sources;
    check(inspect(model, geometry, error, true, &mtp_window_sources, &mtp_sources) &&
              mtp_window_sources.inspected_file_locks.size() == 5 &&
              mtp_sources.catalog->dense_tensors.size() == 29 &&
              mtp_sources.catalog->draft_vocab_ids.empty() &&
              std::any_of(mtp_window_sources.inspected_file_locks.begin(),
                          mtp_window_sources.inspected_file_locks.end(),
                          [&](const auto& lock) { return lock.path == mtp_set_lock.string(); }) &&
              shared_inspection_lock_blocks_writer(mtp_set_lock),
          "window-source inspection retains the shared MTP generation lock");
    {
        std::ofstream vocab(model.mtp / "draft_vocab.bin", std::ios::binary | std::ios::trunc);
        const std::int32_t ids[] = {240'000, 17};
        vocab.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    check(inspect(model, geometry, error, true, nullptr, &mtp_sources) &&
              geometry.mtp_bind_bytes == 15'449'624 &&
              mtp_sources.catalog->draft_vocab_file_size == 2 * sizeof(std::int32_t) &&
              mtp_sources.catalog->draft_vocab_ids ==
                  std::vector<std::uint32_t>{240'000, 17},
          "MTP runtime accepts unique in-range little-endian int32 vocabulary ids");
    fs::resize_file(model.mtp / "draft_vocab.bin", 6);
    check(!inspect(model, geometry, error, true, nullptr, &mtp_sources) &&
              error.find("int32 token ids") != std::string::npos &&
              mtp_sources.catalog == nullptr &&
              mtp_sources.generation_lock.lock_lease() == nullptr,
          "MTP runtime rejects a partial draft vocabulary id");
    {
        std::ofstream vocab(model.mtp / "draft_vocab.bin", std::ios::binary | std::ios::trunc);
        const std::int32_t ids[] = {248'320};
        vocab.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    check(!inspect(model, geometry, error, true) && error.find("outside the 248320-token vocabulary") !=
              std::string::npos,
          "MTP runtime rejects an out-of-range draft vocabulary id");
    {
        std::ofstream vocab(model.mtp / "draft_vocab.bin", std::ios::binary | std::ios::trunc);
        const std::int32_t ids[] = {17, 17};
        vocab.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    check(!inspect(model, geometry, error, true) && error.find("duplicate token id") != std::string::npos,
          "MTP runtime rejects duplicate draft vocabulary ids");
    fs::remove(model.mtp / "draft_vocab.bin");
    make_mtp_runtime(model, true);
    check(!inspect(model, geometry, error, true) && error.find("invalid or duplicate tensor row") != std::string::npos,
          "MTP runtime rejects a dimensionally transposed projection");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--test-native-experts-reader") {
            run_native_experts_reader_test();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--test-pack-index") {
            run_pack_index_test();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--test-expert-address") {
            run_expert_address_test();
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--test-mtp-fixture-pin") {
            const std::string_view descriptor_text(argv[2]);
            int parent_descriptor = -1;
            const auto [descriptor_end, descriptor_error] = std::from_chars(
                descriptor_text.data(),
                descriptor_text.data() + descriptor_text.size(),
                parent_descriptor);
            require(descriptor_error == std::errc{} &&
                        descriptor_end == descriptor_text.data() + descriptor_text.size() &&
                        parent_descriptor >= 0,
                    "MTP fixture parent descriptor is invalid");
            if (::fchdir(parent_descriptor) != 0) {
                throw std::runtime_error(
                    "could not enter the pinned MTP fixture parent: " +
                    std::error_code(errno, std::generic_category()).message());
            }
            const fs::path probe_name(argv[3]);
            require(probe_name == probe_name.filename() &&
                        probe_name != "." && probe_name != "..",
                    "MTP fixture pin probe must be one directory name");
            require(::mkdir(probe_name.c_str(), 0700) == 0,
                    "could not create MTP fixture pin probe directory");
            struct stat created_status {};
            require(::lstat(probe_name.c_str(), &created_status) == 0 &&
                        S_ISDIR(created_status.st_mode),
                    "MTP fixture pin probe changed during creation");
            const int fixture_descriptor =
                open_pinned_fixture_directory(probe_name, created_status);
            const fs::path held_name = probe_name.string() + ".held";
            require(::rename(probe_name.c_str(), held_name.c_str()) == 0,
                    "could not rename pinned MTP fixture probe");
            require(::symlink(argv[4], probe_name.c_str()) == 0,
                    "could not replace MTP fixture probe with symlink");
            if (::fchdir(fixture_descriptor) != 0) {
                const int saved_errno = errno;
                (void)::close(fixture_descriptor);
                throw std::runtime_error(
                    "could not enter the pinned fixture directory: " +
                    std::error_code(saved_errno, std::generic_category()).message());
            }
            const int proof_descriptor =
                ::open("pinned-write", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                       0600);
            if (proof_descriptor < 0) {
                const int saved_errno = errno;
                (void)::close(fixture_descriptor);
                throw std::runtime_error(
                    "could not write through the pinned fixture directory: " +
                    std::error_code(saved_errno, std::generic_category()).message());
            }
            constexpr std::string_view proof = "pinned\n";
            const ssize_t proof_bytes =
                ::write(proof_descriptor, proof.data(), proof.size());
            const int saved_errno = errno;
            (void)::close(proof_descriptor);
            (void)::close(fixture_descriptor);
            require(proof_bytes == static_cast<ssize_t>(proof.size()),
                    "could not complete the pinned fixture directory write: " +
                        std::error_code(saved_errno, std::generic_category()).message());
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--create-fixture") {
            const fs::path root(argv[2]);
            fs::create_directories(root);
            (void) make_model(root);
            return 0;
        }
        if (argc == 5 &&
            std::string(argv[1]) == "--create-fixture-with-mtp-in-dir-fd") {
            const std::string_view descriptor_text(argv[2]);
            const std::string_view device_text(argv[3]);
            const std::string_view inode_text(argv[4]);
            int directory_descriptor = -1;
            std::uint64_t expected_device = 0;
            std::uint64_t expected_inode = 0;
            const auto [descriptor_end, descriptor_error] = std::from_chars(
                descriptor_text.data(),
                descriptor_text.data() + descriptor_text.size(),
                directory_descriptor);
            const auto [device_end, device_error] = std::from_chars(
                device_text.data(), device_text.data() + device_text.size(), expected_device);
            const auto [inode_end, inode_error] = std::from_chars(
                inode_text.data(), inode_text.data() + inode_text.size(), expected_inode);
            require(descriptor_error == std::errc{} &&
                        descriptor_end == descriptor_text.data() + descriptor_text.size() &&
                        directory_descriptor >= 0 &&
                        device_error == std::errc{} &&
                        device_end == device_text.data() + device_text.size() &&
                        inode_error == std::errc{} &&
                        inode_end == inode_text.data() + inode_text.size(),
                    "pinned MTP fixture directory identity is invalid");
            struct stat directory_status {};
            require(::fstat(directory_descriptor, &directory_status) == 0 &&
                        S_ISDIR(directory_status.st_mode) &&
                        static_cast<std::uint64_t>(directory_status.st_dev) ==
                            expected_device &&
                        static_cast<std::uint64_t>(directory_status.st_ino) ==
                            expected_inode,
                    "pinned MTP fixture directory identity changed");
            if (::fchdir(directory_descriptor) != 0) {
                throw std::runtime_error(
                    "could not enter the pinned MTP fixture directory: " +
                    std::error_code(errno, std::generic_category()).message());
            }
            std::error_code directory_error;
            std::size_t directory_entries = 0;
            for (const fs::directory_entry& entry :
                 fs::directory_iterator(".", directory_error)) {
                if (entry.path().filename() != ".strata-owned-fixture") {
                    throw std::runtime_error(
                        "MTP fixture destination must contain only its ownership marker");
                }
                ++directory_entries;
            }
            if (directory_error || directory_entries != 1) {
                throw std::runtime_error(
                    "MTP fixture destination must contain only its ownership marker");
            }
            verify_fixture_owner_marker();

            SyntheticModel model = make_model(".");
            make_mtp_runtime(model);
            make_default_draft_vocab(model.mtp / "draft_vocab.bin");
            return 0;
        }
        if (argc == 4 &&
            std::string(argv[1]) == "--create-fixture-with-mtp-at-fd") {
            const std::string_view descriptor_text(argv[2]);
            int parent_descriptor = -1;
            const auto [end, parse_error] = std::from_chars(
                descriptor_text.data(),
                descriptor_text.data() + descriptor_text.size(),
                parent_descriptor);
            require(parse_error == std::errc{} &&
                        end == descriptor_text.data() + descriptor_text.size() &&
                        parent_descriptor >= 0,
                    "MTP fixture parent descriptor is invalid");
            if (::fchdir(parent_descriptor) != 0) {
                throw std::runtime_error(
                    "could not enter the pinned MTP fixture parent: " +
                    std::error_code(errno, std::generic_category()).message());
            }
            const fs::path root(argv[3]);
            require(root == root.filename() && root != "." && root != "..",
                    "descriptor-relative MTP fixture path must be one directory name");
            struct stat created_status {};
            if (::lstat(root.c_str(), &created_status) == 0) {
                if (S_ISLNK(created_status.st_mode)) {
                    throw std::runtime_error(
                        "MTP fixture destination must not be a symlink");
                }
                require(S_ISDIR(created_status.st_mode),
                        "MTP fixture destination must be an empty directory");
            } else {
                const int saved_errno = errno;
                if (saved_errno == ENOENT) {
                    throw std::runtime_error(
                        "MTP fixture destination must be an empty directory");
                }
                throw std::runtime_error(
                    "could not inspect MTP fixture destination: " +
                    std::error_code(saved_errno, std::generic_category()).message());
            }
            const int fixture_descriptor =
                open_pinned_fixture_directory(root, created_status);
            if (::fchdir(fixture_descriptor) != 0) {
                const int saved_errno = errno;
                (void)::close(fixture_descriptor);
                throw std::runtime_error(
                    "could not enter the pinned MTP fixture directory: " +
                    std::error_code(saved_errno, std::generic_category()).message());
            }
            (void)::close(fixture_descriptor);
            std::error_code directory_error;
            std::size_t directory_entries = 0;
            for (const fs::directory_entry& entry :
                 fs::directory_iterator(".", directory_error)) {
                if (entry.path().filename() != ".strata-owned-fixture") {
                    throw std::runtime_error(
                        "MTP fixture destination must be an empty directory");
                }
                ++directory_entries;
            }
            if (directory_error || directory_entries != 1) {
                throw std::runtime_error(
                    "MTP fixture destination must contain only its ownership marker");
            }
            verify_fixture_owner_marker();

            SyntheticModel model = make_model(".");
            make_mtp_runtime(model);
            make_default_draft_vocab(model.mtp / "draft_vocab.bin");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--test-model-metadata") {
            run_model_metadata_test();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--check-model-snapshot") {
            const std::string_view expert_text(argv[3]);
            std::uint32_t expected_experts = 0;
            const auto [end, parse_error] = std::from_chars(
                expert_text.data(), expert_text.data() + expert_text.size(), expected_experts);
            require(parse_error == std::errc{} && end == expert_text.data() + expert_text.size() &&
                        (expected_experts == 256 || expected_experts == 512),
                    "expected expert count must be 256 or 512");
            const strata::GgufModel model = strata::GgufModel::open(argv[2]);
            strata::Qwen4ExpGuard guard;
            guard.experts = expected_experts;
            const std::string metadata_error = strata::check_architecture(model.meta(), guard);
            require(metadata_error.empty(), metadata_error);
            for (std::size_t shard = 0; shard < model.size(); ++shard) {
                for (const strata::TensorInfo& tensor : model.shard(shard).tensors()) {
                    std::string type_error;
                    require(strata::metal::validate_model_tensor_type(tensor.name, tensor.type, type_error),
                            type_error);
                    int block_elements = 0;
                    int block_bytes = 0;
                    require(!tensor.shape.empty() &&
                                strata::block_geometry(tensor.type, block_elements, block_bytes) &&
                                tensor.shape[0] % static_cast<std::uint64_t>(block_elements) == 0 &&
                                strata::tensor_payload_bytes(tensor) != 0,
                            "GGUF tensor " + tensor.name + " has invalid quantization block geometry");
                }
            }
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--test-model-types") {
            run_model_types_test();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--print-ple-constants") {
            print_ple_constants();
            return 0;
        }
        if (argc != 1) {
            std::cerr << "usage: metal_weight_store_test [--test-model-metadata | "
                         "--check-model-snapshot SHARD EXPERT_COUNT | --test-model-types | "
                         "--test-native-experts-reader | --test-pack-index | "
                         "--test-expert-address | "
                         "--print-ple-constants | "
                         "--test-mtp-fixture-pin FD NAME EXTERNAL | "
                         "--create-fixture DIRECTORY | "
                         "--create-fixture-with-mtp-at-fd FD NAME | "
                         "--create-fixture-with-mtp-in-dir-fd FD DEV INO]\n";
            return 2;
        }
        run();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_weight_store_test: " << exception.what() << '\n';
        return 1;
    }
}
