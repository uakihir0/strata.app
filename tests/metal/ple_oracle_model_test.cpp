// Check the PLE table reader against the frozen numpy oracle and both GGUF layouts:
// a standalone table shard and a table embedded in a model shard.

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "src/kernels/ple_oracle_vectors.inc"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace kernels = strata::kernels;
namespace oracle = strata::kernels::ple_oracle;

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;
constexpr std::uint32_t kGgufVersion = 3u;
constexpr std::uint32_t kMetaString = 8u;
constexpr std::uint32_t kMetaU64 = 10u;
constexpr std::uint32_t kTypeF32 = 0u;
constexpr std::uint32_t kTypeIq4Nl = 20u;
constexpr std::uint64_t kExpectedTableBytes =
    oracle::kTableRows * static_cast<std::uint64_t>(kernels::PLE_ROW_BYTES);

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class UInt>
void append_le(std::vector<std::uint8_t>& bytes, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        bytes.push_back(static_cast<std::uint8_t>(value & static_cast<UInt>(0xffu)));
        value >>= 8u;
    }
}

void append_string(std::vector<std::uint8_t>& bytes, std::string_view value) {
    append_le(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

struct Kv {
    std::string key;
    std::uint32_t type = 0;
    std::string string_value;
    std::uint64_t integer_value = 0;
};

Kv string_kv(std::string key, std::string value) {
    return {std::move(key), kMetaString, std::move(value), 0};
}

Kv u64_kv(std::string key, std::uint64_t value) {
    return {std::move(key), kMetaU64, {}, value};
}

struct TensorFixture {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
    std::vector<std::uint8_t> payload;
};

std::uint64_t write_gguf(const fs::path& path, const std::vector<Kv>& metadata,
                         const std::vector<TensorFixture>& tensors) {
    std::vector<std::uint8_t> bytes;
    append_le(bytes, kGgufMagic);
    append_le(bytes, kGgufVersion);
    append_le(bytes, static_cast<std::uint64_t>(tensors.size()));
    append_le(bytes, static_cast<std::uint64_t>(metadata.size()));
    for (const Kv& kv : metadata) {
        append_string(bytes, kv.key);
        append_le(bytes, kv.type);
        if (kv.type == kMetaString) {
            append_string(bytes, kv.string_value);
        } else if (kv.type == kMetaU64) {
            append_le(bytes, kv.integer_value);
        } else {
            throw std::runtime_error("synthetic GGUF writer received an unsupported metadata type");
        }
    }
    std::uint64_t payload_end = 0;
    for (const TensorFixture& tensor : tensors) {
        append_string(bytes, tensor.name);
        append_le(bytes, static_cast<std::uint32_t>(tensor.shape.size()));
        for (const std::uint64_t dimension : tensor.shape) append_le(bytes, dimension);
        append_le(bytes, tensor.type);
        append_le(bytes, tensor.offset);
        require(tensor.offset <= std::numeric_limits<std::uint64_t>::max() - tensor.payload.size(),
                "synthetic tensor payload extent overflows");
        payload_end = std::max(payload_end, tensor.offset + tensor.payload.size());
    }

    const std::uint64_t data_start = (bytes.size() + 31u) & ~std::uint64_t{31u};
    require(data_start <= std::numeric_limits<std::size_t>::max() - payload_end,
            "synthetic GGUF file extent overflows address space");
    bytes.resize(static_cast<std::size_t>(data_start + payload_end), 0u);
    for (const TensorFixture& tensor : tensors) {
        std::copy(tensor.payload.begin(), tensor.payload.end(),
                  bytes.begin() + static_cast<std::ptrdiff_t>(data_start + tensor.offset));
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "create synthetic GGUF " + path.string());
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    require(!output.fail(), "write synthetic GGUF " + path.string());
    return data_start;
}

class TempDir {
public:
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("strata-ple-oracle-" + std::to_string(stamp));
        require(fs::create_directory(path_), "create temporary PLE test directory");
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::uint64_t required_unsigned_metadata(const strata::GgufFile& file, const char* key,
                                         const fs::path& path) {
    const strata::MetaValue* value = file.get(key);
    require(value != nullptr, path.string() + ": missing " + key);
    switch (value->type) {
        case strata::MetaType::U8:
        case strata::MetaType::U16:
        case strata::MetaType::U32:
        case strata::MetaType::U64:
            return value->u;
        default:
            throw std::runtime_error(path.string() + ": " + key + " must be an unsigned integer");
    }
}

void require_split_metadata(const strata::GgufFile& file, const fs::path& path,
                            std::uint64_t expected_shard, std::uint64_t expected_count,
                            std::uint64_t expected_tensors) {
    require(required_unsigned_metadata(file, "split.no", path) == expected_shard,
            path.string() + ": split.no does not match the PLE shard position");
    require(required_unsigned_metadata(file, "split.count", path) == expected_count,
            path.string() + ": split.count does not match the catalogue model");
    require(required_unsigned_metadata(file, "split.tensors.count", path) == expected_tensors,
            path.string() + ": split.tensors.count does not match the opened model");
}

std::array<std::uint8_t, kernels::PLE_ROW_BYTES> fixture_row() {
    std::array<std::uint8_t, kernels::PLE_ROW_BYTES> row{};
    for (std::size_t block = 0; block < kernels::PLE_HEAD_DIM / 32; ++block) {
        const std::size_t start = block * 18u;
        row[start] = 0x00u;
        row[start + 1u] = 0x38u;  // binary16 0.5
        std::fill(row.begin() + static_cast<std::ptrdiff_t>(start + 2u),
                  row.begin() + static_cast<std::ptrdiff_t>(start + 18u), 0x88u);
        row[start + 2u] = 0x89u;  // low code 9 => 13, high code 8 => 1
    }
    return row;
}

void check_synthetic_reader_layouts() {
    TempDir temp;

    const auto payload = fixture_row();
    std::vector<std::uint8_t> small_table(kernels::PLE_ROW_BYTES * 3u, 0u);
    std::copy(payload.begin(), payload.end(), small_table.begin() + kernels::PLE_ROW_BYTES);
    const fs::path standalone_path = temp.path() / "standalone-ple.gguf";
    const std::vector<Kv> standalone_metadata{
        u64_kv("split.no", 1u), u64_kv("split.count", 2u), u64_kv("split.tensors.count", 1224u)};
    const std::uint64_t standalone_data_start = write_gguf(
        standalone_path, standalone_metadata,
        {{"per_layer_token_embd.weight", {kernels::PLE_HEAD_DIM, 3u}, kTypeIq4Nl, 0u,
          small_table}});

    kernels::PleTable direct;
    std::string error;
    require(direct.open(standalone_path.string(), error),
            "open synthetic standalone table in Direct mode: " + error);
    require(direct.mode() == kernels::PleIo::Direct && direct.rows() == 3u,
            "standalone synthetic table selects Direct mode and reports three rows");
    std::array<float, kernels::PLE_HEAD_DIM> values{};
    direct.read_row(1u, values.data());
    require(values[0] == 6.5f && values[1] == 0.5f && values[16] == 0.5f &&
                values[kernels::PLE_HEAD_DIM - 1] == 0.5f && direct.bytes_read() == kernels::PLE_ROW_BYTES,
            "Direct reader decodes the nonzero row from the standalone table payload");
    {
        strata::GgufFile source(standalone_path.string());
        const strata::TensorInfo* tensor = source.find("per_layer_token_embd.weight");
        require(tensor != nullptr && tensor->offset == 0u && standalone_data_start == source.data_start(),
                "standalone fixture puts the table at tensor offset zero");
        require(source.file_size() - source.data_start() == small_table.size(),
                "standalone fixture tensor exactly fills its data section");
        require_split_metadata(source, standalone_path, 1u, 2u, 1224u);
    }
    direct.close();

    const fs::path first_path = temp.path() / "embedded-00001-of-00002.gguf";
    const fs::path second_path = temp.path() / "embedded-00002-of-00002.gguf";
    std::vector<std::uint8_t> prefix(32u, 0u);
    const auto split_keys = [](std::uint64_t shard) {
        return std::vector<Kv>{u64_kv("split.count", 2u), u64_kv("split.no", shard),
                               u64_kv("split.tensors.count", 3u)};
    };
    std::vector<Kv> first_metadata{string_kv("general.architecture", "qwen4exp")};
    const auto first_keys = split_keys(0u);
    first_metadata.insert(first_metadata.end(), first_keys.begin(), first_keys.end());
    write_gguf(first_path, first_metadata,
               {{"prefix.weight", {8u}, kTypeF32, 0u, prefix},
                {"per_layer_token_embd.weight", {kernels::PLE_HEAD_DIM, 3u}, kTypeIq4Nl,
                 32u, small_table}});
    write_gguf(second_path, split_keys(1u),
               {{"token_embd.weight", {1u}, kTypeF32, 0u, std::vector<std::uint8_t>(4u, 0u)}});

    auto model = std::make_shared<strata::GgufModel>(
        std::vector<std::string>{first_path.string(), second_path.string()});
    std::size_t ple_shard = std::numeric_limits<std::size_t>::max();
    const strata::TensorInfo* tensor = model->find("per_layer_token_embd.weight", &ple_shard);
    require(tensor != nullptr && ple_shard == 0u && tensor->offset == 32u,
            "split fixture places PLE at a nonzero offset in shard one");
    require_split_metadata(model->shard(ple_shard), first_path, 0u, 2u, 3u);

    kernels::PleTable mapped;
    require(mapped.open_mapped(model, ple_shard, error),
            "open synthetic embedded table through the parsed tensor table: " + error);
    require(mapped.mode() == kernels::PleIo::Mmap && mapped.rows() == 3u,
            "embedded synthetic table selects Mmap mode and reports three rows");
    mapped.read_row(1u, values.data());
    require(values[0] == 6.5f && values[1] == 0.5f && values[16] == 0.5f &&
                values[kernels::PLE_HEAD_DIM - 1] == 0.5f && mapped.bytes_read() == kernels::PLE_ROW_BYTES,
            "mapped reader follows the nonzero GGUF tensor offset and decodes the expected row");
    std::cout << "  synthetic standalone Direct and embedded split-shard Mmap layouts: ok\n";
}

using ProbeRows = std::array<std::array<std::uint8_t, kernels::PLE_ROW_BYTES>,
                              oracle::kProbeCount>;

template <class ReadRaw>
ProbeRows check_oracle_rows(kernels::PleTable& table, ReadRaw&& read_raw,
                           const std::string& description) {
    require(table.is_open(), description + ": PLE table did not open");
    require(table.rows() == oracle::kTableRows,
            description + ": expected " + std::to_string(oracle::kTableRows) + " rows, found " +
                std::to_string(table.rows()));
    require(std::string_view(table.format()) == "IQ4_NL",
            description + ": expected IQ4_NL, found " + table.format());

    ProbeRows raw_rows{};
    for (int probe = 0; probe < oracle::kProbeCount; ++probe) {
        const std::uint32_t row_id = oracle::kProbeRows[probe];
        require(row_id < table.rows(), description + ": oracle row exceeds the PLE table");
        std::array<float, kernels::PLE_HEAD_DIM> values{};
        table.read_row(row_id, values.data());
        const auto raw = read_raw(row_id);
        raw_rows[probe] = raw;
        std::array<float, kernels::PLE_HEAD_DIM> raw_decoded{};
        kernels::iq4nl_dequant_row(raw.data(), raw_decoded.data());

        bool finite = true;
        double head_error = 0.0;
        double tail_error = 0.0;
        double l1 = 0.0;
        for (int i = 0; i < kernels::PLE_HEAD_DIM; ++i) {
            finite = finite && std::isfinite(values[i]);
            l1 += std::fabs(static_cast<double>(values[i]));
            if (i < 8) {
                head_error = std::max(head_error,
                                      std::fabs(static_cast<double>(values[i]) -
                                                oracle::kProbeHead[probe][i]));
            }
            if (i >= kernels::PLE_HEAD_DIM - 8) {
                const int tail_index = i - (kernels::PLE_HEAD_DIM - 8);
                tail_error = std::max(tail_error,
                                      std::fabs(static_cast<double>(values[i]) -
                                                oracle::kProbeTail[probe][tail_index]));
            }
            require(values[i] == raw_decoded[i],
                    description + ": mapped/direct bytes disagree with the tensor's raw-byte decode at row " +
                        std::to_string(row_id) + ", element " + std::to_string(i));
        }
        require(finite, description + ": row " + std::to_string(row_id) + " contains a non-finite value");
        require(head_error == 0.0,
                description + ": row " + std::to_string(row_id) + " does not match oracle head values");
        require(tail_error == 0.0,
                description + ": row " + std::to_string(row_id) + " does not match oracle tail values");
        require(std::fabs(l1 - oracle::kProbeSum[probe]) <= 1e-9,
                description + ": row " + std::to_string(row_id) + " does not match oracle L1 sum");
        if (row_id == oracle::kTableRows - 1u) {
            for (float value : values) {
                require(value == 0.0f,
                        description + ": final oracle row is not entirely zero after decoding");
            }
        }
    }
    require(table.bytes_read() == static_cast<std::uint64_t>(oracle::kProbeCount) * kernels::PLE_ROW_BYTES,
            description + ": reader did not account for exactly the three oracle rows");
    return raw_rows;
}

struct Candidate {
    fs::path path;
    std::string tag;
    bool standalone = false;
};

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::optional<Candidate> inspect_candidate(const fs::path& path, const fs::path& root) {
    strata::GgufFile file(path.string());
    const strata::TensorInfo* tensor = file.find("per_layer_token_embd.weight");
    if (tensor == nullptr) return std::nullopt;

    require(tensor->shape.size() == 2u && tensor->shape[0] == kernels::PLE_HEAD_DIM,
            path.string() + ": PLE tensor must have shape [160, rows]");
    require(tensor->type == kTypeIq4Nl,
            path.string() + ": PLE oracle source must use IQ4_NL");
    require(tensor->shape[1] == oracle::kTableRows,
            path.string() + ": unexpected PLE row count " + std::to_string(tensor->shape[1]));
    require(tensor->shape[1] <= std::numeric_limits<std::uint64_t>::max() / kernels::PLE_ROW_BYTES,
            path.string() + ": PLE tensor byte size overflows");
    const std::uint64_t table_bytes = tensor->shape[1] * kernels::PLE_ROW_BYTES;
    const fs::path relative = path.lexically_relative(root);
    require(!relative.empty() && relative.begin() != relative.end(),
            "cannot identify model tag for " + path.string());
    const std::string tag = lowercase(relative.begin()->string());
    require(file.data_start() <= file.file_size(), path.string() + ": data section starts past EOF");
    const std::uint64_t payload_bytes = file.file_size() - file.data_start();
    require(tensor->offset <= payload_bytes && table_bytes <= payload_bytes - tensor->offset,
            path.string() + ": PLE tensor range exceeds its GGUF shard");

    const bool standalone = file.tensors().size() == 1u;
    if (standalone) {
        require_split_metadata(file, path, 1u, 2u, 1224u);
        require(file.data_start() == oracle::kTableDataStart,
                path.string() + ": standalone PLE data_start differs from the oracle source");
        require(tensor->offset == 0u && table_bytes == payload_bytes &&
                    file.file_size() == oracle::kTableFileSize,
                path.string() + ": standalone PLE tensor does not exactly fill the expected shard");
    } else {
        require(tensor->offset != 0u,
                path.string() + ": embedded PLE tensor unexpectedly starts at offset zero");
        require(table_bytes == kExpectedTableBytes,
                path.string() + ": embedded PLE tensor byte size differs from the oracle source");
    }

    return Candidate{path, tag, standalone};
}

ProbeRows check_standalone_model(const Candidate& candidate) {
    strata::GgufFile source(candidate.path.string());
    const strata::TensorInfo* tensor = source.find("per_layer_token_embd.weight");
    require(tensor != nullptr, candidate.path.string() + ": standalone PLE tensor disappeared");
    const std::uint8_t* raw_table = source.tensor_data(*tensor);

    kernels::PleTable table;
    std::string error;
    require(table.open(candidate.path.string(), error),
            candidate.path.string() + ": open standalone PLE reader: " + error);
    ProbeRows raw_rows = check_oracle_rows(table, [raw_table](std::uint32_t row) {
        std::array<std::uint8_t, kernels::PLE_ROW_BYTES> raw{};
        std::copy_n(raw_table + static_cast<std::size_t>(row) * kernels::PLE_ROW_BYTES,
                    raw.size(), raw.begin());
        return raw;
    }, candidate.path.string());
    std::cout << "  " << candidate.path << ": standalone PLE oracle probes: ok\n";
    return raw_rows;
}

ProbeRows check_embedded_model(const Candidate& candidate) {
    auto model = std::make_shared<strata::GgufModel>(
        strata::GgufModel::open(candidate.path.string()));
    std::size_t shard_index = std::numeric_limits<std::size_t>::max();
    const strata::TensorInfo* tensor = model->find("per_layer_token_embd.weight", &shard_index);
    require(tensor != nullptr, candidate.path.string() + ": embedded PLE tensor disappeared");
    require(shard_index < model->size(), candidate.path.string() + ": PLE shard index is invalid");
    const strata::GgufFile& shard = model->shard(shard_index);
    require_split_metadata(shard, candidate.path, shard_index, model->size(), 1224u);
    if (candidate.tag == "swift-iq2_xs") {
        require(shard_index == 0u,
                candidate.path.string() + ": Swift IQ2_XS PLE tensor must be in shard 1");
    }
    require(shard.find("per_layer_token_embd.weight") == tensor,
            candidate.path.string() + ": model index does not point into the PLE shard tensor table");
    const std::uint64_t table_bytes = tensor->shape[1] * kernels::PLE_ROW_BYTES;
    require(tensor->offset > 0u && table_bytes == kExpectedTableBytes,
            candidate.path.string() + ": embedded PLE offset or tensor size is invalid");
    const std::uint8_t* raw_table = shard.tensor_data(*tensor);

    kernels::PleTable table;
    std::string error;
    require(table.open_mapped(model, shard_index, error),
            candidate.path.string() + ": open embedded PLE reader: " + error);
    ProbeRows raw_rows = check_oracle_rows(table, [raw_table](std::uint32_t row) {
        std::array<std::uint8_t, kernels::PLE_ROW_BYTES> raw{};
        std::copy_n(raw_table + static_cast<std::size_t>(row) * kernels::PLE_ROW_BYTES,
                    raw.size(), raw.begin());
        return raw;
    }, candidate.path.string());
    std::cout << "  " << candidate.path << ": embedded PLE tensor oracle probes: ok (shard "
              << shard_index + 1u << ", tensor offset " << tensor->offset << ")\n";
    return raw_rows;
}

int check_models() {
    const char* models_env = std::getenv("STRATA_TEST_MODELS");
    if (models_env == nullptr || *models_env == '\0') {
        std::cerr << "ple_oracle_model_test: STRATA_TEST_MODELS must point to the catalogue model root\n";
        return 2;
    }
    const fs::path root = fs::absolute(models_env);
    require(fs::is_directory(root), "STRATA_TEST_MODELS is not a directory: " + root.string());

    std::vector<fs::path> gguf_paths;
    std::error_code error;
    for (fs::recursive_directory_iterator it(root, error), end; it != end; it.increment(error)) {
        require(!error, "cannot enumerate model root " + root.string() + ": " + error.message());
        const fs::path path = it->path();
        if (path.extension() != ".gguf") continue;
        const bool regular_file = it->is_regular_file(error);
        require(!error, "cannot inspect model file " + path.string() + ": " + error.message());
        if (regular_file) gguf_paths.push_back(path);
    }
    require(!error, "cannot enumerate model root " + root.string() + ": " + error.message());
    std::sort(gguf_paths.begin(), gguf_paths.end());
    require(!gguf_paths.empty(), "no .gguf files found under " + root.string());

    std::vector<Candidate> candidates;
    for (const fs::path& path : gguf_paths) {
        try {
            std::optional<Candidate> candidate = inspect_candidate(path, root);
            if (candidate) candidates.push_back(std::move(*candidate));
        } catch (const std::exception& exc) {
            throw std::runtime_error(path.string() + ": " + exc.what());
        }
    }

    require(!candidates.empty(), "no PLE tensor was found under STRATA_TEST_MODELS");

    std::optional<ProbeRows> q2_0_probe_rows;
    std::optional<ProbeRows> swift_iq2_xs_probe_rows;
    for (const Candidate& candidate : candidates) {
        if (candidate.tag == "qwen-q2_0") {
            require(candidate.standalone,
                    candidate.path.string() + ": the Q2_0 ISTA PLE source must be a standalone shard");
            require(!q2_0_probe_rows.has_value(),
                    "multiple qwen-Q2_0 PLE shards were found under STRATA_TEST_MODELS");
        }
        if (candidate.tag == "swift-iq2_xs") {
            require(!candidate.standalone,
                    candidate.path.string() + ": the Swift IQ2_XS PLE source must be embedded in shard 1");
            require(!swift_iq2_xs_probe_rows.has_value(),
                    "multiple Swift IQ2_XS PLE shards were found under STRATA_TEST_MODELS");
        }

        ProbeRows raw_rows = candidate.standalone ? check_standalone_model(candidate)
                                                  : check_embedded_model(candidate);
        if (candidate.tag == "qwen-q2_0") q2_0_probe_rows = raw_rows;
        if (candidate.tag == "swift-iq2_xs") swift_iq2_xs_probe_rows = raw_rows;
    }
    require(q2_0_probe_rows.has_value(),
            "the canonical qwen-Q2_0 ISTA shard-2 PLE source is missing from STRATA_TEST_MODELS");
    require(swift_iq2_xs_probe_rows.has_value(),
            "the canonical Swift IQ2_XS shard-1 PLE source is missing from STRATA_TEST_MODELS");
    require(*q2_0_probe_rows == *swift_iq2_xs_probe_rows,
            "the frozen PLE probe rows are not byte-identical between ISTA Q2_0 shard 2 and Swift IQ2_XS shard 1");
    std::cout << "ple_oracle_model_test: " << candidates.size()
              << " PLE tables passed frozen head/tail and L1 probes; packed probe rows are byte-identical "
                 "between ISTA Q2_0 shard 2 and Swift IQ2_XS shard 1\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
            check_synthetic_reader_layouts();
            std::cout << "ple_oracle_layout_test: all passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--models") return check_models();
        std::cerr << "usage: ple_oracle_model_test --selftest|--models\n";
        return 2;
    } catch (const std::exception& exc) {
        std::cerr << "ple_oracle_model_test: " << exc.what() << '\n';
        return 1;
    }
}
