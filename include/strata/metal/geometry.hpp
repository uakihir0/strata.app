// include/strata/metal/geometry.hpp - the device-free model facts consumed by the Metal planner.
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {

struct ShardGeometry {
    std::string name;
    std::uint64_t bytes = 0;
};

struct LayerGeometry {
    std::uint64_t blob_bytes = 0;
    std::string gate_up_type;
    std::string down_type;
};

struct Geometry {
    std::string family;
    std::string model;
    std::vector<ShardGeometry> shards;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    std::uint64_t nonexpert_bytes = 0;
    std::uint64_t wrapped_unread_bytes = 0;
    std::uint64_t dense_pack_bytes = 0;
    std::uint64_t mtp_bytes = 0;
    std::uint64_t mtp_bind_bytes = 0;
    std::uint32_t n_qsa_layers = 0;
    std::uint32_t n_kv_head = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t n_embd = 0;
    std::uint32_t vocab = 0;
    std::vector<std::string> types;
    std::vector<LayerGeometry> layers;
};

/// True for the supported Qwen3.8-Flash-Next package families and the
/// `qwen4exp` default label used when geometry is inspected directly.
inline bool is_qwen4exp_catalog_family(std::string_view family) noexcept {
    return family == "qwen4exp" || family == "qwen" ||
           family == "swift" || family == "coder";
}

/// Check the fixed architecture dimensions shared by the Mac catalogue.
inline bool is_qwen4exp_runtime_geometry(const Geometry& geometry) noexcept {
    return is_qwen4exp_catalog_family(geometry.family) &&
           geometry.n_layers == 48u &&
           geometry.n_qsa_layers == 12u &&
           geometry.n_kv_head == 2u &&
           geometry.head_dim == 256u &&
           geometry.n_embd == 2'560u &&
           geometry.vocab == 248'320u;
}

/// Validate the full format contract for callers that construct geometry without parsing a file.
bool validate_geometry(const Geometry& geometry, std::string& error);

/// Parse format v1. Unknown records and duplicate keys are rejected to keep the plan deterministic.
bool read_geometry(std::istream& input, Geometry& geometry, std::string& error);
bool read_geometry_file(const std::string& path, Geometry& geometry, std::string& error);
bool write_geometry(std::ostream& output, const Geometry& geometry, std::string& error);

/// Canonical GGML spelling used by the geometry format, e.g. "iq3_xxs" or "q4_k".
std::string geometry_type_name(std::uint32_t ggml_type);

}  // namespace strata::metal
