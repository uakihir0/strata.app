// src/metal/program/dump_format.cpp - writers for the --dump-* byte layouts (docs/mac/03, sections 9.1 and 9.2).
#include "strata/metal/dump_format.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata::metal {
namespace {

static_assert(std::endian::native == std::endian::little, "dump files are little-endian");

constexpr std::size_t kWidth = static_cast<std::size_t>(kDumpEmbeddingWidth);
constexpr std::size_t kStreams = static_cast<std::size_t>(kDumpStreams);
constexpr std::size_t kHalfAttentionOffset = 0;
constexpr std::size_t kHalfFfnOffset = 2560;
constexpr std::size_t kHalfAttentionInjectOffset = 5120;
constexpr std::size_t kHalfFfnInjectOffset = 5124;
constexpr std::size_t kHalfQsaPreGateOffset = 5128;
constexpr std::size_t kHalfQsaVcurOffset = 11272;
constexpr std::size_t kHalfQsaGatherVOffset = 12296;
constexpr std::size_t kHalfQsaIdsOffset = 12552;
constexpr std::size_t kHalfQsaStepsOffset = 12556;
constexpr std::size_t kHalfQsaVPoolOffset = 12560;
constexpr std::size_t kHalfQsaVcurAppendedOffset = 12816;
constexpr std::size_t kHalfQsaKcurOffset = 13328;

using Bytes = std::vector<unsigned char>;

void append_raw(Bytes& bytes, const void* data, std::size_t size) {
    const auto* first = static_cast<const unsigned char*>(data);
    bytes.insert(bytes.end(), first, first + size);
}

void append_int32(Bytes& bytes, std::int32_t value) { append_raw(bytes, &value, sizeof(value)); }

void append_floats(Bytes& bytes, std::span<const float> values) {
    append_raw(bytes, values.data(), values.size_bytes());
}

bool write_file(const std::string& path, const Bytes& bytes, std::string& error) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        error = "cannot open " + path + " for writing";
        return false;
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    const bool closed = std::fclose(file) == 0;
    if (written != bytes.size() || !closed) {
        error = "cannot write all of " + path;
        return false;
    }
    return true;
}

bool fits_int32(std::size_t value) { return value <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()); }

// int32 values sit in float slots as their bit patterns (section 9.2).
void copy_int32s(std::span<const std::int32_t> values, std::span<float> out, std::size_t offset) {
    std::memcpy(out.data() + offset, values.data(), values.size_bytes());
}

}  // namespace

std::vector<std::int64_t> logits_stride_rows(std::int64_t rows, std::int64_t stride, std::string& error) {
    std::vector<std::int64_t> kept;
    if (rows < 1 || stride < 1) {
        error = "logits rows and the stride must be positive";
        return kept;
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        if (row % stride == 0 || row == rows - 1) kept.push_back(row);
    }
    return kept;
}

bool write_logits_dump(const std::string& path, std::int32_t n_vocab, std::span<const float> logits,
                       std::string& error) {
    if (n_vocab < 1) {
        error = "the vocabulary size must be positive";
        return false;
    }
    const auto width = static_cast<std::size_t>(n_vocab);
    if (logits.empty() || logits.size() % width != 0 || !fits_int32(logits.size() / width)) {
        error = "logits must hold one or more whole rows of n_vocab values";
        return false;
    }
    Bytes bytes;
    append_int32(bytes, n_vocab);
    append_int32(bytes, static_cast<std::int32_t>(logits.size() / width));
    append_floats(bytes, logits);
    return write_file(path, bytes, error);
}

bool write_first_logits_dump(const std::string& path, std::int32_t n_vocab, std::span<const float> logits,
                             std::string& error) {
    if (n_vocab < 1 || logits.size() != static_cast<std::size_t>(n_vocab)) {
        error = "the first-logits dump must hold exactly n_vocab values";
        return false;
    }
    Bytes bytes;
    append_floats(bytes, logits);
    return write_file(path, bytes, error);
}

bool write_residual_dump(const std::string& path, std::span<const float> residual, std::string& error) {
    if (residual.size() != kStreams * kWidth) {
        error = "the residual dump must hold 4 x 2560 values";
        return false;
    }
    Bytes bytes;
    append_int32(bytes, kDumpStreams);
    append_int32(bytes, kDumpEmbeddingWidth);
    append_floats(bytes, residual);
    return write_file(path, bytes, error);
}

bool write_layers_dump(const std::string& path, std::span<const float> ladder, std::string& error) {
    if (ladder.empty() || ladder.size() % kDumpLadderPositionFloats != 0) {
        error = "the residual ladder must hold one or more whole positions of 49 x 4 x 2560 values";
        return false;
    }
    Bytes bytes;
    append_floats(bytes, ladder);
    return write_file(path, bytes, error);
}

bool pack_half_layer(const HalfLayer& layer, std::span<float> out, std::string& error) {
    if (out.size() != kDumpHalfLayerFloats) {
        error = "a half layer is 13,840 floats";
        return false;
    }
    const bool qsa_present = !layer.qsa_pre_gate.empty() || !layer.qsa_vcur.empty() ||
                             !layer.qsa_gather_v.empty() || !layer.qsa_ids.empty() ||
                             !layer.qsa_steps.empty() || !layer.qsa_vcur_appended.empty() ||
                             !layer.qsa_kcur.empty();
    const bool qsa_complete = !layer.qsa_pre_gate.empty() && !layer.qsa_vcur.empty() &&
                              !layer.qsa_gather_v.empty() && !layer.qsa_ids.empty() && !layer.qsa_steps.empty() &&
                              !layer.qsa_vcur_appended.empty() && !layer.qsa_kcur.empty();
    if (layer.attention.size() != kWidth || layer.ffn.size() != kWidth || layer.attention_inject.size() != kStreams ||
        layer.ffn_inject.size() != kStreams) {
        error = "attention and FFN blocks are 2560 floats and their injects are 4 floats each";
        return false;
    }
    if (qsa_present && !qsa_complete) {
        error = "a QSA layer needs every QSA slice; a GDN layer needs none";
        return false;
    }
    if (qsa_complete && (layer.qsa_pre_gate.size() != 6144 || layer.qsa_vcur.size() != 512 ||
                         layer.qsa_gather_v.size() != 256 || layer.qsa_ids.size() != 4 ||
                         layer.qsa_steps.size() != 4 || layer.qsa_vcur_appended.size() != 512 ||
                         layer.qsa_kcur.size() != 512 ||
                         (!layer.qsa_v_pool.empty() && layer.qsa_v_pool.size() != 256))) {
        error = "a QSA slice has the wrong length";
        return false;
    }

    std::fill(out.begin(), out.end(), 0.0f);
    const auto place = [&out](std::span<const float> values, std::size_t offset) {
        std::memcpy(out.data() + offset, values.data(), values.size_bytes());
    };
    place(layer.attention, kHalfAttentionOffset);
    place(layer.ffn, kHalfFfnOffset);
    place(layer.attention_inject, kHalfAttentionInjectOffset);
    place(layer.ffn_inject, kHalfFfnInjectOffset);
    if (qsa_complete) {
        place(layer.qsa_pre_gate, kHalfQsaPreGateOffset);
        place(layer.qsa_vcur, kHalfQsaVcurOffset);
        place(layer.qsa_gather_v, kHalfQsaGatherVOffset);
        copy_int32s(layer.qsa_ids, out, kHalfQsaIdsOffset);
        copy_int32s(layer.qsa_steps, out, kHalfQsaStepsOffset);
        if (!layer.qsa_v_pool.empty()) place(layer.qsa_v_pool, kHalfQsaVPoolOffset);
        place(layer.qsa_vcur_appended, kHalfQsaVcurAppendedOffset);
        place(layer.qsa_kcur, kHalfQsaKcurOffset);
    }
    return true;
}

bool write_halves_dump(const std::string& path, std::span<const float> halves, std::string& error) {
    const std::size_t per_position = static_cast<std::size_t>(kDumpLayerCount) * kDumpHalfLayerFloats;
    if (halves.empty() || halves.size() % per_position != 0) {
        error = "the halves dump must hold one or more whole positions of 48 x 13,840 values";
        return false;
    }
    Bytes bytes;
    append_floats(bytes, halves);
    return write_file(path, bytes, error);
}

bool append_routing_record(std::FILE* file, std::int32_t layer, std::span<const std::int32_t> ids,
                           std::span<const float> weights, std::string& error) {
    if (file == nullptr || ids.empty() || ids.size() != weights.size() || !fits_int32(ids.size())) {
        error = "a routing record needs one weight for each of its one or more ids";
        return false;
    }
    // The rules that tools/make_profile.py applies when it reads a trace: distinct, non-negative ids and finite,
    // non-negative weights. Its bounds on the layer and on k depend on the model, so they are left to the reader.
    std::vector<std::int32_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        error = "routing ids must be distinct and non-negative";
        return false;
    }
    for (const float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            error = "routing weights must be finite and non-negative";
            return false;
        }
    }
    Bytes bytes;
    append_int32(bytes, layer);
    append_int32(bytes, static_cast<std::int32_t>(ids.size()));
    append_raw(bytes, ids.data(), ids.size_bytes());
    append_floats(bytes, weights);
    if (std::fwrite(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
        error = "cannot write a routing record";
        return false;
    }
    return true;
}

}  // namespace strata::metal
