// Validate each catalogue model's MTP runtime through WeightStore's production inspector.

#include "strata/artifact/gguf_reader.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/weights.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace metal = strata::metal;

namespace {

struct ModelTag {
    const char* tag;
    const char* family;
    std::uint64_t file_type;
    std::uint64_t expert_count;
    bool swift;
};

constexpr std::array<ModelTag, 7> kModelTags{{
    {"qwen-Q2_0", "qwen", 41, 512, false},
    {"qwen-IQ2_XS", "qwen", 20, 512, false},
    {"qwen-IQ3_XXS", "qwen", 23, 512, false},
    {"qwen-IQ3_S", "qwen", 26, 512, false},
    {"swift-IQ2_XS", "swift", 20, 512, true},
    {"swift-IQ3_XXS", "swift", 23, 512, true},
    {"coder-IQ1_M", "coder", 31, 256, false},
}};

struct ShippedVocabulary {
    const char* filename;
    std::size_t ids;
};

constexpr std::array<ShippedVocabulary, 3> kShippedVocabularies{{
    {"draft_vocab.bin", 106'299u},
    {"draft_vocab_en.bin", 40'525u},
    {"draft_vocab_cyrillic.bin", 58'963u},
}};
using ShippedVocabularySet =
    std::array<std::vector<std::uint32_t>, kShippedVocabularies.size()>;

struct ExpectedTensor {
    std::string_view name;
    metal::MtpDenseWeightKind kind;
    std::uint64_t rows;
    std::uint64_t columns;
};

constexpr std::array<ExpectedTensor, 29> kExpectedTensors{{
    {"fc_embedding.weight", metal::MtpDenseWeightKind::q8_0, 2560, 2560},
    {"fc_hidden.weight", metal::MtpDenseWeightKind::q8_0, 2560, 2560},
    {"self_attn.q_proj.weight", metal::MtpDenseWeightKind::q8_0, 12'288, 2560},
    {"self_attn.k_proj.weight", metal::MtpDenseWeightKind::q8_0, 512, 2560},
    {"self_attn.v_proj.weight", metal::MtpDenseWeightKind::q8_0, 512, 2560},
    {"self_attn.o_proj.weight", metal::MtpDenseWeightKind::q8_0, 2560, 6144},
    {"self_attn.indexer.index_qk_proj.weight", metal::MtpDenseWeightKind::q8_0, 640, 2560},
    {"mlp.shared_expert.gate_proj.weight", metal::MtpDenseWeightKind::q8_0, 640, 2560},
    {"mlp.shared_expert.up_proj.weight", metal::MtpDenseWeightKind::q8_0, 640, 2560},
    {"mlp.shared_expert.down_proj.weight", metal::MtpDenseWeightKind::q8_0, 2560, 640},
    {"hyper_connection_mixer.hc_norm.weight", metal::MtpDenseWeightKind::f32, 1, 10'240},
    {"hyper_connection_mixer.input_mix_weight_down.weight", metal::MtpDenseWeightKind::bf16, 320, 10'240},
    {"hyper_connection_mixer.input_mix_weight_up.weight", metal::MtpDenseWeightKind::bf16, 10'240, 320},
    {"attn_hyper_connection.hc_norm.weight", metal::MtpDenseWeightKind::f32, 1, 10'240},
    {"attn_hyper_connection.input_mix_weight_down.weight", metal::MtpDenseWeightKind::bf16, 320, 10'240},
    {"attn_hyper_connection.input_mix_weight_up.weight", metal::MtpDenseWeightKind::bf16, 10'240, 320},
    {"attn_hyper_connection.block_inject_weight.weight", metal::MtpDenseWeightKind::bf16, 4, 10'240},
    {"mlp_hyper_connection.hc_norm.weight", metal::MtpDenseWeightKind::f32, 1, 10'240},
    {"mlp_hyper_connection.input_mix_weight_down.weight", metal::MtpDenseWeightKind::bf16, 320, 10'240},
    {"mlp_hyper_connection.input_mix_weight_up.weight", metal::MtpDenseWeightKind::bf16, 10'240, 320},
    {"mlp_hyper_connection.block_inject_weight.weight", metal::MtpDenseWeightKind::bf16, 4, 10'240},
    {"mlp.gate.weight", metal::MtpDenseWeightKind::bf16, 512, 2560},
    {"mlp.shared_expert_gate.weight", metal::MtpDenseWeightKind::bf16, 1, 2560},
    {"pre_fc_norm_embedding.weight", metal::MtpDenseWeightKind::f32, 1, 2560},
    {"pre_fc_norm_hidden.weight", metal::MtpDenseWeightKind::f32, 1, 10'240},
    {"self_attn.q_norm.weight", metal::MtpDenseWeightKind::f32, 1, 256},
    {"self_attn.k_norm.weight", metal::MtpDenseWeightKind::f32, 1, 256},
    {"self_attn.indexer.q_layernorm.weight", metal::MtpDenseWeightKind::f32, 1, 128},
    {"self_attn.indexer.k_layernorm.weight", metal::MtpDenseWeightKind::f32, 1, 128},
}};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool unsigned_metadata(const strata::MetaValue* value, std::uint64_t& result) {
    if (value == nullptr) return false;
    switch (value->type) {
        case strata::MetaType::U8:
        case strata::MetaType::U16:
        case strata::MetaType::U32:
        case strata::MetaType::U64:
            result = value->u;
            return true;
        default:
            return false;
    }
}

fs::path find_model_shard_one(const ModelTag& tag, const fs::path& model_dir) {
    std::optional<fs::path> result;
    std::error_code error;
    for (fs::directory_iterator it(model_dir, error), end; it != end; it.increment(error)) {
        require(!error, "cannot enumerate model directory " + model_dir.string() + ": " + error.message());
        const fs::path path = it->path();
        const bool regular_file = it->is_regular_file(error);
        require(!error, "cannot inspect model path " + path.string() + ": " + error.message());
        if (!regular_file || path.extension() != ".gguf") continue;

        strata::GgufFile file(path.string());
        const strata::MetaValue* architecture = file.get("general.architecture");
        std::uint64_t split_count = 0;
        std::uint64_t shard_no = 0;
        std::uint64_t split_tensor_count = 0;
        std::uint64_t file_type = 0;
        std::uint64_t expert_count = 0;
        const strata::MetaValue* recurrent_layers =
            file.get("qwen4exp.attention.recurrent_layers");
        if (architecture == nullptr || architecture->type != strata::MetaType::STRING ||
            architecture->s != "qwen4exp" ||
            !unsigned_metadata(file.get("split.count"), split_count) || split_count != 2u ||
            !unsigned_metadata(file.get("split.no"), shard_no) || shard_no != 0u ||
            !unsigned_metadata(file.get("split.tensors.count"), split_tensor_count) ||
            split_tensor_count != 1'224u ||
            !unsigned_metadata(file.get("general.file_type"), file_type) ||
            file_type != tag.file_type ||
            !unsigned_metadata(file.get("qwen4exp.expert_count"), expert_count) ||
            expert_count != tag.expert_count || (recurrent_layers != nullptr) != tag.swift) {
            continue;
        }
        require(!result.has_value(), "multiple qwen4exp shard-1 GGUFs under " + model_dir.string());
        result = path;
    }
    require(!error, "cannot enumerate model directory " + model_dir.string() + ": " + error.message());
    require(result.has_value(), "no shard-1 GGUF matching catalogue tag " +
                                std::string(tag.tag) + " found under " + model_dir.string());
    return *result;
}

fs::path find_native_pack(const fs::path& model_dir) {
    const fs::path pack_root = model_dir / "pack";
    std::vector<fs::path> candidates;
    std::error_code error;
    for (const fs::path& direct : {pack_root, pack_root / "full"}) {
        const bool has_index = fs::is_regular_file(direct / "index.txt", error);
        require(!error, "cannot inspect native pack index " + (direct / "index.txt").string() +
                            ": " + error.message());
        if (has_index) candidates.push_back(direct);
    }
    if (fs::is_directory(pack_root, error)) {
        require(!error, "cannot inspect native pack directory " + pack_root.string() + ": " + error.message());
        for (fs::directory_iterator it(pack_root, error), end; it != end; it.increment(error)) {
            require(!error, "cannot enumerate native pack directory " + pack_root.string() +
                                ": " + error.message());
            const fs::path path = it->path();
            const bool is_dir = it->is_directory(error);
            require(!error, "cannot inspect native pack path " + path.string() + ": " + error.message());
            if (!is_dir || path == pack_root / "full") continue;
            const bool has_index = fs::is_regular_file(path / "index.txt", error);
            require(!error, "cannot inspect native pack index " + (path / "index.txt").string() +
                                ": " + error.message());
            if (has_index) candidates.push_back(path);
        }
        require(!error, "cannot enumerate native pack directory " + pack_root.string() +
                            ": " + error.message());
    } else {
        require(!error, "cannot inspect native pack directory " + pack_root.string() + ": " + error.message());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    require(candidates.size() == 1u,
            model_dir.string() + ": expected one native pack index under pack/, found " +
                std::to_string(candidates.size()));
    return candidates.front();
}

std::uint64_t expected_bytes(const ExpectedTensor& tensor) {
    const std::uint64_t elements = tensor.rows * tensor.columns;
    if (tensor.kind == metal::MtpDenseWeightKind::q8_0) {
        require(tensor.columns % 32u == 0u, "test MTP Q8_0 fixture has a partial block");
        return tensor.rows * (tensor.columns / 32u) * 34u;
    }
    return elements * (tensor.kind == metal::MtpDenseWeightKind::f32 ? 4u : 2u);
}

std::vector<std::uint32_t> read_vocabulary_ids(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open shipped draft vocabulary " + path.string());
    const std::vector<char> bytes{std::istreambuf_iterator<char>(input),
                                  std::istreambuf_iterator<char>()};
    require(!input.bad() && bytes.size() % sizeof(std::uint32_t) == 0u,
            "shipped draft vocabulary has an invalid byte count: " + path.string());
    std::vector<std::uint32_t> ids;
    ids.reserve(bytes.size() / sizeof(std::uint32_t));
    for (std::size_t index = 0; index < bytes.size(); index += sizeof(std::uint32_t)) {
        const auto byte = [&bytes](std::size_t offset) {
            return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[offset]));
        };
        ids.push_back(byte(index) | (byte(index + 1u) << 8u) |
                      (byte(index + 2u) << 16u) | (byte(index + 3u) << 24u));
    }
    return ids;
}

ShippedVocabularySet load_shipped_vocabularies() {
    ShippedVocabularySet vocabularies;
    for (std::size_t index = 0; index < kShippedVocabularies.size(); ++index) {
        const ShippedVocabulary& expected = kShippedVocabularies[index];
        vocabularies[index] = read_vocabulary_ids(fs::path(STRATA_TEST_DATA_DIR) /
                                                  expected.filename);
        require(vocabularies[index].size() == expected.ids,
                std::string("shipped draft vocabulary has an unexpected ID count: ") +
                    expected.filename);
    }
    return vocabularies;
}

bool matches_shipped_vocabulary(const std::vector<std::uint32_t>& ids,
                                const ShippedVocabularySet& vocabularies) {
    return std::find(vocabularies.begin(), vocabularies.end(), ids) != vocabularies.end();
}

void check_shipped_vocabulary_data() {
    const ShippedVocabularySet vocabularies = load_shipped_vocabularies();
    std::vector<bool> present(248'320u, false);
    for (const auto& ids : vocabularies) {
        std::vector<bool> seen(248'320u, false);
        for (const std::uint32_t id : ids) {
            require(id < seen.size(), "shipped draft vocabulary has an out-of-range token ID");
            require(!seen[id], "shipped draft vocabulary has a duplicate token ID");
            seen[id] = true;
            present[id] = true;
        }
        require(matches_shipped_vocabulary(ids, vocabularies),
                "exact shipped draft vocabulary was not recognized");
        std::vector<std::uint32_t> reordered = ids;
        std::reverse(reordered.begin(), reordered.end());
        require(!matches_shipped_vocabulary(reordered, vocabularies),
                "reordered draft vocabulary was accepted");
    }
    const auto unused = std::find(present.begin(), present.end(), false);
    require(unused != present.end(), "shipped vocabularies cover the whole token ID range");
    const std::uint32_t replacement = static_cast<std::uint32_t>(unused - present.begin());
    for (const auto& ids : vocabularies) {
        std::vector<std::uint32_t> changed = ids;
        changed.front() = replacement;
        require(!matches_shipped_vocabulary(changed, vocabularies),
                "changed draft-vocabulary token was accepted");
    }
}

void check_catalog(const metal::MtpRuntimeCatalog& catalog, const fs::path& runtime_dir,
                   const ShippedVocabularySet& vocabularies) {
    require(fs::path(catalog.directory) == runtime_dir,
            runtime_dir.string() + ": reader catalog points to a different runtime directory");
    require(fs::path(catalog.experts_path) == runtime_dir / "experts.bin" &&
                catalog.experts_file_size == metal::kMtpExpertsFileBytes,
            runtime_dir.string() + ": experts.bin name or size differs from the runtime contract");
    require(fs::path(catalog.dense_path) == runtime_dir / "dense.bin" &&
                catalog.dense_file_size == metal::kMtpDenseFileBytes,
            runtime_dir.string() + ": dense.bin name or size differs from the runtime contract");
    require(fs::path(catalog.draft_vocab_path) == runtime_dir / "draft_vocab.bin",
            runtime_dir.string() + ": draft_vocab.bin path differs from the runtime contract");

    constexpr std::array<std::uint64_t, 3> kSupportedVocabularyBytes{
        425'196u, 162'100u, 235'852u};
    require(std::find(kSupportedVocabularyBytes.begin(), kSupportedVocabularyBytes.end(),
                      catalog.draft_vocab_file_size) != kSupportedVocabularyBytes.end(),
            runtime_dir.string() + ": draft_vocab.bin is not one of the three shipped vocabularies");
    require(catalog.draft_vocab_ids.size() * sizeof(std::uint32_t) == catalog.draft_vocab_file_size,
            runtime_dir.string() + ": draft_vocab.bin byte count disagrees with its parsed IDs");
    std::vector<bool> vocabulary_seen(248'320u, false);
    for (const std::uint32_t token_id : catalog.draft_vocab_ids) {
        require(token_id < vocabulary_seen.size(),
                runtime_dir.string() + ": draft vocabulary contains an out-of-range token ID");
        require(!vocabulary_seen[token_id],
                runtime_dir.string() + ": draft vocabulary contains a duplicate token ID");
        vocabulary_seen[token_id] = true;
    }
    require(matches_shipped_vocabulary(catalog.draft_vocab_ids, vocabularies),
            runtime_dir.string() + ": draft vocabulary IDs/order do not match any shipped vocabulary");

    require(catalog.dense_tensors.size() == kExpectedTensors.size(),
            runtime_dir.string() + ": dense.txt must contain exactly 29 tensors");
    std::set<std::string> seen;
    std::uint64_t expected_offset = 0;
    for (const metal::MtpDenseWeightSource& actual : catalog.dense_tensors) {
        const auto expected = std::find_if(kExpectedTensors.begin(), kExpectedTensors.end(),
                                           [&actual](const ExpectedTensor& tensor) {
                                               return tensor.name == actual.tensor_name;
                                           });
        require(expected != kExpectedTensors.end() && seen.insert(actual.tensor_name).second,
                runtime_dir.string() + ": dense.txt contains an unknown or duplicate tensor " +
                    actual.tensor_name);
        const std::uint64_t bytes = expected_bytes(*expected);
        require(actual.kind == expected->kind && actual.rows == expected->rows &&
                    actual.columns == expected->columns && actual.byte_count == bytes,
                runtime_dir.string() + ": dense tensor contract differs for " + actual.tensor_name);
        require(actual.file_offset % 256u == 0u && actual.file_offset >= expected_offset &&
                    actual.file_offset <= catalog.dense_file_size &&
                    bytes <= catalog.dense_file_size - actual.file_offset,
                runtime_dir.string() + ": dense tensor extent is invalid for " + actual.tensor_name);
        const std::uint64_t end = actual.file_offset + bytes;
        expected_offset = (end + 255u) & ~std::uint64_t{255u};
    }
    require(seen.size() == kExpectedTensors.size() &&
                expected_offset == metal::kMtpDenseFileBytes,
            runtime_dir.string() + ": dense.txt is missing a required tensor or leaves an invalid extent");
}

metal::WeightStoreOptions make_options(const ModelTag& tag, const fs::path& model_dir,
                                       const fs::path& shard_one) {
    metal::WeightStoreOptions options;
    options.native_shard = shard_one.string();
    options.pack_dir = find_native_pack(model_dir).string();
    options.mtp_dir = (model_dir / "mtp" / "rt").string();
    options.family = tag.family;
    const std::string_view tag_name(tag.tag);
    const std::size_t separator = tag_name.find('-');
    require(separator != std::string_view::npos,
            "catalogue tag is missing its family separator: " + std::string(tag.tag));
    options.model = std::string(tag_name.substr(separator + 1u));
    return options;
}

int check_models() {
    const char* models_env = std::getenv("STRATA_TEST_MODELS");
    if (models_env == nullptr || *models_env == '\0') {
        std::cerr << "mtp_reader_test: STRATA_TEST_MODELS must point to the catalogue model root\n";
        return 2;
    }
    const fs::path root = fs::absolute(models_env);
    require(fs::is_directory(root), "STRATA_TEST_MODELS is not a directory: " + root.string());

    const ShippedVocabularySet vocabularies = load_shipped_vocabularies();
    for (const ModelTag& tag : kModelTags) {
        const fs::path model_dir = root / tag.tag;
        require(fs::is_directory(model_dir),
                "catalogue model directory is missing: " + model_dir.string());
        const fs::path shard_one = find_model_shard_one(tag, model_dir);
        const metal::WeightStoreOptions options = make_options(tag, model_dir, shard_one);
        std::error_code runtime_error;
        const bool runtime_exists = fs::is_directory(options.mtp_dir, runtime_error);
        require(!runtime_error && runtime_exists,
                "catalogue MTP runtime directory is missing: " + options.mtp_dir +
                    (runtime_error ? ": " + runtime_error.message() : ""));
        const fs::path runtime_dir = fs::canonical(options.mtp_dir, runtime_error);
        require(!runtime_error,
                "cannot resolve catalogue MTP runtime directory " + options.mtp_dir + ": " +
                    runtime_error.message());
        metal::Geometry geometry;
        metal::MtpRuntimeSources sources;
        std::string error;
        require(metal::WeightStore::inspect(options, geometry, error, nullptr, &sources),
                model_dir.string() + ": production MTP inspection failed: " + error);
        require(sources.catalog != nullptr,
                model_dir.string() + ": production MTP inspection returned no catalog");
        require(geometry.mtp_bytes == metal::kMtpExpertsFileBytes + metal::kMtpDenseFileBytes,
                model_dir.string() + ": total MTP runtime bytes differ from the format contract");
        check_catalog(*sources.catalog, runtime_dir, vocabularies);
        std::cout << "  " << tag.tag << ": experts.bin, dense.bin, dense.txt and draft_vocab.bin: ok\n";
    }
    std::cout << "mtp_reader_test: all seven catalogue MTP runtimes passed\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mtp_reader_test --models|--vocabulary-self-test\n";
        return 2;
    }
    try {
        if (std::string_view(argv[1]) == "--models") return check_models();
        if (std::string_view(argv[1]) == "--vocabulary-self-test") {
            check_shipped_vocabulary_data();
            std::cout << "mtp_reader_vocabulary_test: all shipped vocabularies passed\n";
            return 0;
        }
        std::cerr << "usage: mtp_reader_test --models|--vocabulary-self-test\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "mtp_reader_test: " << error.what() << '\n';
        return 1;
    }
}
