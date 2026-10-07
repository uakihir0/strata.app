// src/metal/engine/geometry.cpp - strict geometry v1 reader/writer and type naming.
#include "strata/metal/geometry.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace strata::metal {
namespace {

std::string_view trim(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t' ||
                              value.front() == '\r' || value.front() == '\n')) {
        value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                              value.back() == '\r' || value.back() == '\n')) {
        value.remove_suffix(1);
    }
    return value;
}

bool valid_utf8(std::string_view value) {
    std::size_t i = 0;
    while (i < value.size()) {
        const auto lead = static_cast<unsigned char>(value[i]);
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
        if (value.size() - i < count) return false;
        for (std::size_t offset = 1; offset < count; ++offset) {
            const auto continuation = static_cast<unsigned char>(value[i + offset]);
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

bool parse_u64(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    std::uint64_t wide = 0;
    if (!parse_u64(text, wide) || wide > std::numeric_limits<std::uint32_t>::max()) return false;
    value = static_cast<std::uint32_t>(wide);
    return true;
}

bool valid_atom(std::string_view text) {
    if (text.empty() || !valid_utf8(text)) return false;
    return std::none_of(text.begin(), text.end(), [](unsigned char c) {
        return c <= 0x20 || c == '=' || c == '#';
    });
}

bool set_once(std::map<std::string, std::string>& values, std::string key, std::string value,
              std::string& error, std::size_t line_number) {
    if (!values.emplace(key, std::move(value)).second) {
        error = "duplicate geometry key " + key + " at line " + std::to_string(line_number);
        return false;
    }
    return true;
}

bool take_u64(const std::map<std::string, std::string>& values, const char* key, std::uint64_t& out,
              std::string& error) {
    const auto it = values.find(key);
    if (it == values.end() || !parse_u64(it->second, out)) {
        error = std::string("missing or invalid geometry key ") + key;
        return false;
    }
    return true;
}

bool take_u32(const std::map<std::string, std::string>& values, const char* key, std::uint32_t& out,
              std::string& error) {
    const auto it = values.find(key);
    if (it == values.end() || !parse_u32(it->second, out)) {
        error = std::string("missing or invalid geometry key ") + key;
        return false;
    }
    return true;
}

bool check_geometry(const Geometry& geometry, std::string& error) {
    if (geometry.family.empty() || !valid_atom(geometry.family) ||
        geometry.model.empty() || !valid_atom(geometry.model)) {
        error = "geometry family and model must be non-empty atoms";
        return false;
    }
    if (geometry.shards.empty() || geometry.shards.size() > 99) {
        error = "geometry must describe between 1 and 99 shards";
        return false;
    }
    std::set<std::string> shard_names;
    for (const auto& shard : geometry.shards) {
        if (!valid_atom(shard.name) || shard.bytes == 0) {
            error = "geometry shard names and sizes must be non-empty and positive";
            return false;
        }
        if (!shard_names.insert(shard.name).second) {
            error = "geometry shard names must be unique";
            return false;
        }
    }
    if (geometry.n_layers == 0 || geometry.n_layers > 1024 ||
        geometry.n_expert == 0 || geometry.n_expert > 512 ||
        geometry.n_qsa_layers == 0 || geometry.n_qsa_layers > geometry.n_layers ||
        geometry.n_kv_head == 0 ||
        geometry.head_dim == 0 || geometry.n_embd == 0 || geometry.vocab == 0) {
        error = "geometry dimensions must be positive";
        return false;
    }
    if (geometry.layers.size() != geometry.n_layers) {
        error = "geometry layer count does not match n_layers";
        return false;
    }
    for (const auto& layer : geometry.layers) {
        if (layer.blob_bytes == 0 || !valid_atom(layer.gate_up_type) || !valid_atom(layer.down_type)) {
            error = "geometry layer records must have a positive blob and two tensor types";
            return false;
        }
    }
    if (geometry.types.empty()) {
        error = "geometry types list is empty";
        return false;
    }
    std::set<std::string> unique_types;
    for (const auto& type : geometry.types) {
        if (!valid_atom(type) || type.find(',') != std::string::npos ||
            !unique_types.insert(type).second) {
            error = "geometry types contains an empty or duplicate type";
            return false;
        }
    }
    for (const auto& layer : geometry.layers) {
        if (!unique_types.contains(layer.gate_up_type) || !unique_types.contains(layer.down_type)) {
            error = "geometry layer type is missing from the types list";
            return false;
        }
    }
    return true;
}

}  // namespace

bool validate_geometry(const Geometry& geometry, std::string& error) {
    error.clear();
    return check_geometry(geometry, error);
}

std::string geometry_type_name(std::uint32_t ggml_type) {
    std::string name = strata::ggml_type_name(ggml_type);
    if (name == "?") return "ggml_type_" + std::to_string(ggml_type);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
        return static_cast<char>(c);
    });
    return name;
}

bool read_geometry(std::istream& input, Geometry& geometry, std::string& error) {
    error.clear();
    Geometry parsed;
    std::map<std::string, std::string> values;
    std::map<std::uint32_t, LayerGeometry> layers;
    std::set<std::uint32_t> shard_ids;
    bool saw_header = false;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!valid_utf8(line)) {
            error = "geometry line " + std::to_string(line_number) + " is not valid UTF-8";
            return false;
        }
        std::string_view text = trim(line);
        if (text.empty() || text.front() == '#') continue;
        if (!saw_header) {
            saw_header = true;
            if (text == "strata-geometry 1") continue;
            if (text.starts_with("strata-geometry ")) {
                const auto version = trim(text.substr(std::string_view("strata-geometry ").size()));
                error = "unsupported geometry file version " + std::string(version);
            } else {
                error = "geometry line " + std::to_string(line_number) +
                        " must be 'strata-geometry 1'";
            }
            return false;
        }
        if (text.starts_with("layer ")) {
            std::istringstream row{std::string(text)};
            std::string tag, blob_token, gu_token, down_token, extra;
            std::uint32_t layer_id = 0;
            LayerGeometry layer;
            if (!(row >> tag >> layer_id >> blob_token >> gu_token >> down_token) || tag != "layer" ||
                (row >> extra) || !blob_token.starts_with("blob=") || !gu_token.starts_with("gu=") ||
                !down_token.starts_with("d=") ||
                !parse_u64(std::string_view(blob_token).substr(5), layer.blob_bytes)) {
                error = "invalid geometry layer record at line " + std::to_string(line_number);
                return false;
            }
            layer.gate_up_type = gu_token.substr(3);
            layer.down_type = down_token.substr(2);
            if (!layers.emplace(layer_id, std::move(layer)).second) {
                error = "duplicate geometry layer " + std::to_string(layer_id);
                return false;
            }
            continue;
        }
        const std::size_t equals = text.find('=');
        if (equals == std::string_view::npos) {
            error = "invalid geometry key/value at line " + std::to_string(line_number);
            return false;
        }
        const std::string key(trim(text.substr(0, equals)));
        const std::string value(trim(text.substr(equals + 1)));
        if (key.starts_with("shard") && key.ends_with("_name")) {
            const std::string_view number(key.data() + 5, key.size() - 10);
            std::uint32_t id = 0;
            if (!parse_u32(number, id) || id == 0 || !valid_atom(value)) {
                error = "invalid geometry shard name at line " + std::to_string(line_number);
                return false;
            }
            if (key != "shard" + std::to_string(id) + "_name") {
                error = "non-canonical geometry shard name key at line " +
                        std::to_string(line_number);
                return false;
            }
            if (!set_once(values, key, value, error, line_number)) return false;
            shard_ids.insert(id);
            continue;
        }
        if (key.starts_with("shard") && key.ends_with("_bytes")) {
            const std::string_view number(key.data() + 5, key.size() - 11);
            std::uint32_t id = 0;
            std::uint64_t bytes = 0;
            if (!parse_u32(number, id) || id == 0 || !parse_u64(value, bytes) || bytes == 0) {
                error = "invalid geometry shard size at line " + std::to_string(line_number);
                return false;
            }
            if (key != "shard" + std::to_string(id) + "_bytes") {
                error = "non-canonical geometry shard size key at line " +
                        std::to_string(line_number);
                return false;
            }
            if (!set_once(values, key, value, error, line_number)) return false;
            shard_ids.insert(id);
            continue;
        }
        if (key.starts_with("shard")) {
            error = "unknown geometry shard key " + key;
            return false;
        }
        if (!set_once(values, key, value, error, line_number)) {
            return false;
        }
    }
    if (!input.eof() && input.fail()) {
        error = "failed while reading geometry file";
        return false;
    }
    if (!saw_header) {
        error = "empty geometry file";
        return false;
    }

    auto string_value = [&](const char* key, std::string& out) {
        const auto it = values.find(key);
        if (it == values.end() || !valid_atom(it->second)) {
            error = std::string("missing or invalid geometry key ") + key;
            return false;
        }
        out = it->second;
        return true;
    };
    if (!string_value("family", parsed.family) || !string_value("model", parsed.model) ||
        !take_u32(values, "n_layers", parsed.n_layers, error) ||
        !take_u32(values, "n_expert", parsed.n_expert, error) ||
        !take_u64(values, "nonexpert_bytes", parsed.nonexpert_bytes, error) ||
        !take_u64(values, "wrapped_unread_bytes", parsed.wrapped_unread_bytes, error) ||
        !take_u64(values, "dense_pack_bytes", parsed.dense_pack_bytes, error) ||
        !take_u64(values, "mtp_bytes", parsed.mtp_bytes, error) ||
        !take_u64(values, "mtp_bind_bytes", parsed.mtp_bind_bytes, error) ||
        !take_u32(values, "n_qsa_layers", parsed.n_qsa_layers, error) ||
        !take_u32(values, "n_kv_head", parsed.n_kv_head, error) ||
        !take_u32(values, "head_dim", parsed.head_dim, error) ||
        !take_u32(values, "n_embd", parsed.n_embd, error) ||
        !take_u32(values, "vocab", parsed.vocab, error)) {
        return false;
    }

    const auto types = values.find("types");
    if (types == values.end()) {
        error = "missing geometry key types";
        return false;
    }
    if (types->second.empty() || types->second.back() == ',') {
        error = "geometry types contains an empty type";
        return false;
    }
    std::string_view remaining(types->second);
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(',');
        const std::string_view item = trim(remaining.substr(0, comma));
        if (item.empty()) {
            error = "geometry types contains an empty type";
            return false;
        }
        parsed.types.emplace_back(item);
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1);
    }
    if (shard_ids.empty()) {
        error = "geometry has no shard records";
        return false;
    }
    const std::uint32_t max_shard = *shard_ids.rbegin();
    if (shard_ids.size() != max_shard) {
        error = "geometry shard records must be consecutive from shard1";
        return false;
    }
    for (std::uint32_t id = 1; id <= max_shard; ++id) {
        const std::string prefix = "shard" + std::to_string(id);
        const auto name = values.find(prefix + "_name");
        const auto size = values.find(prefix + "_bytes");
        std::uint64_t bytes = 0;
        if (name == values.end() || size == values.end() || !parse_u64(size->second, bytes)) {
            error = "geometry shard" + std::to_string(id) + " requires both name and bytes";
            return false;
        }
        parsed.shards.push_back({name->second, bytes});
    }
    if (layers.size() != parsed.n_layers) {
        error = "geometry layer records do not match n_layers";
        return false;
    }
    parsed.layers.resize(parsed.n_layers);
    for (std::uint32_t id = 0; id < parsed.n_layers; ++id) {
        const auto it = layers.find(id);
        if (it == layers.end()) {
            error = "missing geometry layer " + std::to_string(id);
            return false;
        }
        parsed.layers[id] = it->second;
    }

    static const std::set<std::string> known_keys = {
        "family", "model", "n_layers", "n_expert", "nonexpert_bytes", "wrapped_unread_bytes",
        "dense_pack_bytes", "mtp_bytes", "mtp_bind_bytes", "n_qsa_layers", "n_kv_head", "head_dim",
        "n_embd", "vocab", "types"};
    for (const auto& [key, value] : values) {
        (void)value;
        const bool shard_record = key.starts_with("shard") &&
                                  (key.ends_with("_name") || key.ends_with("_bytes"));
        if (!known_keys.contains(key) && !shard_record) {
            error = "unknown geometry key " + key;
            return false;
        }
    }
    if (!validate_geometry(parsed, error)) return false;
    geometry = std::move(parsed);
    error.clear();
    return true;
}

bool read_geometry_file(const std::string& path, Geometry& geometry, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open geometry file " + path;
        return false;
    }
    return read_geometry(input, geometry, error);
}

bool write_geometry(std::ostream& output, const Geometry& geometry, std::string& error) {
    if (!validate_geometry(geometry, error)) return false;
    output << "strata-geometry 1\n";
    output << "family=" << geometry.family << '\n';
    output << "model=" << geometry.model << '\n';
    for (std::size_t i = 0; i < geometry.shards.size(); ++i) {
        output << "shard" << i + 1 << "_name=" << geometry.shards[i].name << '\n';
        output << "shard" << i + 1 << "_bytes=" << geometry.shards[i].bytes << '\n';
    }
    output << "n_layers=" << geometry.n_layers << '\n';
    output << "n_expert=" << geometry.n_expert << '\n';
    output << "nonexpert_bytes=" << geometry.nonexpert_bytes << '\n';
    output << "wrapped_unread_bytes=" << geometry.wrapped_unread_bytes << '\n';
    output << "dense_pack_bytes=" << geometry.dense_pack_bytes << '\n';
    output << "mtp_bytes=" << geometry.mtp_bytes << '\n';
    output << "mtp_bind_bytes=" << geometry.mtp_bind_bytes << '\n';
    output << "n_qsa_layers=" << geometry.n_qsa_layers << '\n';
    output << "n_kv_head=" << geometry.n_kv_head << '\n';
    output << "head_dim=" << geometry.head_dim << '\n';
    output << "n_embd=" << geometry.n_embd << '\n';
    output << "vocab=" << geometry.vocab << '\n';
    output << "types=";
    for (std::size_t i = 0; i < geometry.types.size(); ++i) {
        if (i != 0) output << ',';
        output << geometry.types[i];
    }
    output << '\n';
    for (std::size_t i = 0; i < geometry.layers.size(); ++i) {
        const auto& layer = geometry.layers[i];
        output << "layer " << i << " blob=" << layer.blob_bytes << " gu=" << layer.gate_up_type
               << " d=" << layer.down_type << '\n';
    }
    if (!output) {
        error = "failed while writing geometry";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace strata::metal
