// tests/metal/dump_format_host_test.cpp - the --dump-* byte layouts (docs/mac/03, sections 9.1 and 9.2), host side.
//
// Every file is written by the writers and then decoded here from its bytes, with the offsets and sizes of the spec
// tables typed in directly. The decoder does not use the writer's constants.
#include "strata/metal/dump_format.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using namespace strata::metal;

namespace {

int failures = 0;

void check(bool condition, const std::string& name, const std::string& detail = {}) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s%s%s\n", name.c_str(), detail.empty() ? "" : " - ", detail.c_str());
}

using Bytes = std::vector<unsigned char>;

Bytes read_file(const std::string& path) {
    Bytes bytes;
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return bytes;
    unsigned char chunk[4096];
    std::size_t count = 0;
    while ((count = std::fread(chunk, 1, sizeof(chunk), file)) > 0) bytes.insert(bytes.end(), chunk, chunk + count);
    std::fclose(file);
    return bytes;
}

// Little-endian readers at a byte offset, decoded independently of the writer.
std::int32_t int32_at(const Bytes& bytes, std::size_t offset) {
    const std::uint32_t value = static_cast<std::uint32_t>(bytes.at(offset)) |
                                static_cast<std::uint32_t>(bytes.at(offset + 1)) << 8 |
                                static_cast<std::uint32_t>(bytes.at(offset + 2)) << 16 |
                                static_cast<std::uint32_t>(bytes.at(offset + 3)) << 24;
    return static_cast<std::int32_t>(value);
}

float float_at(const Bytes& bytes, std::size_t offset) {
    const auto bits = static_cast<std::uint32_t>(int32_at(bytes, offset));
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// The F32 at float index `index` of a file whose floats start at byte `base`.
float float_index(const Bytes& bytes, std::size_t base, std::size_t index) { return float_at(bytes, base + 4 * index); }

bool all_zero(const Bytes& bytes, std::size_t base, std::size_t first, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (float_index(bytes, base, first + i) != 0.0f) return false;
    }
    return true;
}

std::string temp_dir() {
    std::string pattern = "/tmp/strata-dump-format-XXXXXX";
    char* made = ::mkdtemp(pattern.data());
    return made == nullptr ? std::string() : std::string(made);
}

void test_logits_stride_rows() {
    std::string error;
    check(logits_stride_rows(10, 3, error) == std::vector<std::int64_t> {0, 3, 6, 9}, "stride 3 keeps 0, 3, 6, 9");
    check(logits_stride_rows(10, 4, error) == std::vector<std::int64_t> {0, 4, 8, 9},
          "the final input row is kept even off the stride");
    check(logits_stride_rows(4, 1, error) == std::vector<std::int64_t> {0, 1, 2, 3}, "stride 1 keeps every row");
    check(logits_stride_rows(1, 5, error) == std::vector<std::int64_t> {0}, "a one-row dump keeps its only row");
    check(logits_stride_rows(0, 2, error).empty() && !error.empty(), "an empty dump is rejected");
    error.clear();
    check(logits_stride_rows(5, 0, error).empty() && !error.empty(), "a zero stride is rejected");
}

void test_logits_dump_layout(const std::string& dir) {
    const std::string path = dir + "/logits.bin";
    std::string error;
    const std::vector<float> logits = {0.5f, -1.0f, 2.0f, 3.5f, 4.0f,   //
                                       0.25f, 1.25f, -2.0f, 6.0f, 7.0f,  //
                                       8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    check(write_logits_dump(path, 5, 3, logits, error), "logits dump of 3 rows of 5 is written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 8 + 3 * 5 * 4, "logits dump is 8 + rows x n_vocab x 4 bytes", std::to_string(bytes.size()));
    check(int32_at(bytes, 0) == 5, "logits header holds n_vocab first");
    check(int32_at(bytes, 4) == 3, "logits header holds the declared row count second");
    bool values_match = bytes.size() == 8 + logits.size() * 4;
    for (std::size_t i = 0; values_match && i < logits.size(); ++i) values_match = float_index(bytes, 8, i) == logits[i];
    check(values_match, "logits values follow the header as F32, row after row");

    error.clear();
    check(!write_logits_dump(path, 0, 3, logits, error) && !error.empty(), "a zero vocabulary is rejected");
    error.clear();
    // Three one-value rows fit the declared count, so only the vocabulary rule can reject them.
    check(!write_logits_dump(path, 1, 3, std::vector<float> {1.0f, 2.0f, 3.0f}, error) && !error.empty(),
          "a vocabulary of one is rejected");
    error.clear();
    check(!write_logits_dump(path, 4, 3, logits, error) && !error.empty(), "a partial row is rejected");
    error.clear();
    check(!write_logits_dump(path, 5, 3, std::span<const float> {}, error) && !error.empty(), "no rows is rejected");
    error.clear();
    check(!write_logits_dump(path, 5, 0, logits, error) && !error.empty(), "a declared row count of zero is rejected");
    error.clear();
    check(!write_logits_dump(path, 5, 2, logits, error) && !error.empty(), "more rows than the declared count is rejected");
    // tf_compare.py rejects NaN and infinity in the logits, so the writer does too.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const auto& [index, value] : {std::pair<std::size_t, float> {7, nan}, {14, inf}, {0, -inf}}) {
        std::vector<float> bad = logits;
        bad[index] = value;
        error.clear();
        check(!write_logits_dump(path, 5, 3, bad, error) && !error.empty(), "non-finite logits are rejected",
              std::to_string(value));
    }
}

// A prompt-path run writes fewer rows than the header declares (docs/mac/09, section 3.1). The header keeps the declared
// count, and tf_compare.py takes the rows from the file size and the first position from declared minus rows.
void test_short_logits_dump(const std::string& dir) {
    const std::string path = dir + "/short.bin";
    std::string error;
    const std::vector<float> rows(36 * 5, 0.5f);
    check(write_logits_dump(path, 5, 100, rows, error), "36 rows of a 100-row run are written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 8 + 36 * 5 * 4, "a short dump holds only the rows written", std::to_string(bytes.size()));
    check(int32_at(bytes, 0) == 5, "a short dump's header holds n_vocab");
    check(int32_at(bytes, 4) == 100, "a short dump's header holds the declared count, not the rows present");
    const std::int64_t rows_on_disk = static_cast<std::int64_t>((bytes.size() - 8) / (4 * 5));
    check(rows_on_disk == 36 && (bytes.size() - 8) % (4 * 5) == 0, "tf_compare reads 36 whole rows from the payload");
    check(int32_at(bytes, 4) - rows_on_disk == 64, "the first row of a short dump is at position 64",
          std::to_string(int32_at(bytes, 4) - rows_on_disk));
}

void test_first_logits_layout(const std::string& dir) {
    const std::string path = dir + "/first.bin";
    std::string error;
    const std::vector<float> row = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    check(write_first_logits_dump(path, 6, row, error), "first-logits dump is written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 6 * 4, "first-logits dump is n_vocab F32 with no header", std::to_string(bytes.size()));
    check(float_index(bytes, 0, 0) == 1.0f && float_index(bytes, 0, 5) == 6.0f, "first-logits values start at byte 0");
    error.clear();
    check(!write_first_logits_dump(path, 5, row, error), "a row of the wrong length is rejected");
    // Empty input is rejected even with n_vocab 0, where the length check alone would pass.
    error.clear();
    check(!write_first_logits_dump(path, 0, std::span<const float> {}, error) && !error.empty(),
          "an empty first-logits dump with n_vocab 0 is rejected");
    error.clear();
    check(!write_first_logits_dump(path, 1, std::vector<float> {1.0f}, error) && !error.empty(),
          "a first-logits vocabulary of one is rejected");
    error.clear();
    check(!write_first_logits_dump(path, 2, std::vector<float> {1.0f, std::numeric_limits<float>::quiet_NaN()}, error),
          "non-finite first logits are rejected");
}

void test_residual_layout(const std::string& dir) {
    const std::string path = dir + "/residual.bin";
    std::string error;
    std::vector<float> residual(4 * 2560);
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = static_cast<float>(i) * 0.5f;
    check(write_residual_dump(path, residual, error), "residual dump is written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 8 + 4 * 2560 * 4, "residual dump is 8 + 4 x 2560 x 4 bytes", std::to_string(bytes.size()));
    check(int32_at(bytes, 0) == 4 && int32_at(bytes, 4) == 2560, "residual header is int32 4 then int32 2560");
    check(float_index(bytes, 8, 0) == 0.0f && float_index(bytes, 8, 10239) == 10239 * 0.5f,
          "residual values follow the header in order");
    error.clear();
    check(!write_residual_dump(path, std::span<const float>(residual.data(), 2559), error),
          "a residual of the wrong width is rejected");
}

void test_layers_layout(const std::string& dir) {
    const std::string path = dir + "/layers.bin";
    std::string error;
    const std::size_t position_floats = 49 * 4 * 2560;  // (48 + 1) slots of 4 x 2560
    std::vector<float> ladder(2 * position_floats);
    for (std::size_t i = 0; i < ladder.size(); ++i) ladder[i] = static_cast<float>(i);
    check(write_layers_dump(path, ladder, error), "two positions of the residual ladder are written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 2 * 49 * 4 * 2560 * 4, "layers dump has no header and 49 x 4 x 2560 F32 per position",
          std::to_string(bytes.size()));
    check(float_index(bytes, 0, 0) == 0.0f, "slot 0 of position 0 is the first float");
    check(float_index(bytes, 0, 10240) == 10240.0f, "slot 1 (after layer 0) starts 10,240 floats in");
    check(float_index(bytes, 0, position_floats) == static_cast<float>(position_floats),
          "position 1 starts after one whole position");
    error.clear();
    check(!write_layers_dump(path, std::span<const float>(ladder.data(), position_floats - 1), error),
          "a truncated position is rejected");
    error.clear();
    check(!write_layers_dump(path, std::span<const float> {}, error) && !error.empty(), "an empty ladder is rejected");
}

// Section 9.2: where each slice of a layer sits in the 13,840 floats.
void test_half_layer_offsets(const std::string& dir) {
    std::string error;
    std::vector<float> attention(2560), ffn(2560), attention_inject(4), ffn_inject(4);
    for (std::size_t i = 0; i < 2560; ++i) {
        attention[i] = 1000.0f + static_cast<float>(i);
        ffn[i] = 2000.0f + static_cast<float>(i);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        attention_inject[i] = 3000.0f + static_cast<float>(i);
        ffn_inject[i] = 4000.0f + static_cast<float>(i);
    }

    // A GDN layer: the QSA slots stay zero.
    std::vector<float> gdn(kDumpHalfLayerFloats, 99.0f);
    HalfLayer gdn_layer;
    gdn_layer.attention = attention;
    gdn_layer.ffn = ffn;
    gdn_layer.attention_inject = attention_inject;
    gdn_layer.ffn_inject = ffn_inject;
    check(pack_half_layer(gdn_layer, gdn, error), "a GDN layer packs", error);
    check(gdn.size() == 13840, "a half layer is 13,840 floats");
    check(gdn[0] == 1000.0f && gdn[2559] == 3559.0f, "attention block at float 0, 2560 values");
    check(gdn[2560] == 2000.0f && gdn[5119] == 4559.0f, "FFN block at float 2560, 2560 values");
    check(gdn[5120] == 3000.0f && gdn[5123] == 3003.0f, "attention inject at float 5120");
    check(gdn[5124] == 4000.0f && gdn[5127] == 4003.0f, "FFN inject at float 5124");
    bool qsa_zero = true;
    for (std::size_t i = 5128; i < 13840; ++i) qsa_zero = qsa_zero && gdn[i] == 0.0f;
    check(qsa_zero, "a GDN layer leaves every QSA slot and the gap at 11784 zero");

    // A QSA layer: every slice lands at its documented offset.
    std::vector<float> pre_gate(6144), vcur(512), gather_v(256), v_pool(256), vcur_appended(512), kcur(512);
    for (std::size_t i = 0; i < pre_gate.size(); ++i) pre_gate[i] = 5000.0f + static_cast<float>(i);
    for (std::size_t i = 0; i < vcur.size(); ++i) vcur[i] = 6000.0f + static_cast<float>(i);
    for (std::size_t i = 0; i < gather_v.size(); ++i) gather_v[i] = 7000.0f + static_cast<float>(i);
    for (std::size_t i = 0; i < v_pool.size(); ++i) v_pool[i] = 8000.0f + static_cast<float>(i);
    for (std::size_t i = 0; i < vcur_appended.size(); ++i) vcur_appended[i] = 9000.0f + static_cast<float>(i);
    for (std::size_t i = 0; i < kcur.size(); ++i) kcur[i] = 9500.0f + static_cast<float>(i);
    const std::vector<std::int32_t> ids {10, 20, 30, 40};
    const std::vector<std::int32_t> steps {1, 2, 3, 4};
    std::vector<float> qsa(kDumpHalfLayerFloats, 99.0f);
    HalfLayer qsa_layer = gdn_layer;
    qsa_layer.qsa_pre_gate = pre_gate;
    qsa_layer.qsa_vcur = vcur;
    qsa_layer.qsa_gather_v = gather_v;
    qsa_layer.qsa_ids = ids;
    qsa_layer.qsa_steps = steps;
    qsa_layer.qsa_v_pool = v_pool;
    qsa_layer.qsa_vcur_appended = vcur_appended;
    qsa_layer.qsa_kcur = kcur;
    check(pack_half_layer(qsa_layer, qsa, error), "a QSA layer packs", error);
    const Bytes as_bytes = [&qsa] {
        Bytes bytes(qsa.size() * 4);
        std::memcpy(bytes.data(), qsa.data(), bytes.size());
        return bytes;
    }();
    check(float_index(as_bytes, 0, 5128) == 5000.0f && float_index(as_bytes, 0, 5128 + 6143) == 11143.0f,
          "pre-gate attention output at float 5128, 6144 values");
    check(float_index(as_bytes, 0, 11272) == 6000.0f && float_index(as_bytes, 0, 11783) == 6511.0f,
          "vcur as projected at float 11272, 512 values");
    check(all_zero(as_bytes, 0, 11784, 512), "the 512 floats at 11784 are always zero");
    check(float_index(as_bytes, 0, 12296) == 7000.0f && float_index(as_bytes, 0, 12551) == 7255.0f,
          "gathered V scratch at float 12296, 256 values");
    check(int32_at(as_bytes, 4 * 12552) == 10 && int32_at(as_bytes, 4 * 12555) == 40,
          "selection ids at float 12552 as int32 bits");
    check(int32_at(as_bytes, 4 * 12556) == 1 && int32_at(as_bytes, 4 * 12559) == 4,
          "selection steps at float 12556 as int32 bits");
    check(float_index(as_bytes, 0, 12560) == 8000.0f && float_index(as_bytes, 0, 12815) == 8255.0f,
          "V pool at float 12560, 256 values");
    check(float_index(as_bytes, 0, 12816) == 9000.0f && float_index(as_bytes, 0, 13327) == 9511.0f,
          "vcur after the KV append at float 12816, 512 values");
    check(float_index(as_bytes, 0, 13328) == 9500.0f && float_index(as_bytes, 0, 13839) == 10011.0f,
          "kcur after norm and RoPE at float 13328, 512 values");

    // A QSA layer on the FP32 KV path has no V pool: its slots are zero.
    qsa_layer.qsa_v_pool = {};
    check(pack_half_layer(qsa_layer, qsa, error), "a QSA layer without the FP16 V pool packs", error);
    check(qsa[12560] == 0.0f && qsa[12815] == 0.0f && qsa[12816] == 9000.0f,
          "without the FP16 V pool, its slots stay zero and the rest is written");

    // The wrong shapes are rejected.
    HalfLayer partial = gdn_layer;
    partial.qsa_pre_gate = pre_gate;
    error.clear();
    check(!pack_half_layer(partial, qsa, error) && !error.empty(), "a QSA layer with a missing slice is rejected");
    HalfLayer short_attention = gdn_layer;
    short_attention.attention = std::span<const float>(attention.data(), 2559);
    error.clear();
    check(!pack_half_layer(short_attention, qsa, error), "an attention block of the wrong width is rejected");

    // A V pool with no other QSA slice is a QSA layer that cannot be written, so it is rejected rather than dropped.
    HalfLayer pool_only = gdn_layer;
    pool_only.qsa_v_pool = v_pool;
    error.clear();
    check(!pack_half_layer(pool_only, qsa, error) && !error.empty(), "a V pool alone is rejected, not dropped");

    // The QSA slice lengths are checked, including the 511-float kcur and the 255-float V pool.
    const std::vector<float> kcur_511(511, 1.0f);
    HalfLayer short_kcur = qsa_layer;
    short_kcur.qsa_kcur = kcur_511;
    error.clear();
    check(!pack_half_layer(short_kcur, qsa, error) && !error.empty(), "a 511-float kcur slice is rejected");
    const std::vector<float> pool_255(255, 1.0f);
    HalfLayer short_pool = qsa_layer;
    short_pool.qsa_v_pool = pool_255;
    error.clear();
    check(!pack_half_layer(short_pool, qsa, error) && !error.empty(), "a 255-float V pool slice is rejected");
    HalfLayer short_gather = qsa_layer;
    short_gather.qsa_gather_v = std::span<const float>(gather_v.data(), 255);
    error.clear();
    check(!pack_half_layer(short_gather, qsa, error) && !error.empty(), "a 255-float gathered V slice is rejected");
    error.clear();
    check(!write_halves_dump(dir + "/halves_empty.bin", std::span<const float> {}, error) && !error.empty(),
          "an empty halves dump is rejected");

    // The file: 48 packed layers per position, no header, each layer 55,360 bytes.
    std::vector<float> position;
    for (int layer = 0; layer < 48; ++layer) {
        const HalfLayer& source = layer == 47 ? qsa_layer : gdn_layer;
        std::vector<float> packed(kDumpHalfLayerFloats);
        check(pack_half_layer(source, packed, error), "a layer of the position packs", error);
        position.insert(position.end(), packed.begin(), packed.end());
    }
    const std::string path = dir + "/halves.bin";
    check(write_halves_dump(path, position, error), "one position of halves is written", error);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 48 * 13840 * 4, "halves dump is 48 x 13,840 F32 per position with no header",
          std::to_string(bytes.size()));
    check(float_index(bytes, 0, 2560) == 2000.0f, "layer 0 FFN block is at byte 0 + 2560 floats");
    check(float_index(bytes, 47 * 13840 * 4, 12296) == 7000.0f,
          "layer 47's gathered V sits at its own 13,840-float block");
    error.clear();
    check(!write_halves_dump(path, std::span<const float>(position.data(), 13840), error),
          "a halves dump with fewer than 48 layers is rejected");
}

void test_routing_records(const std::string& dir) {
    const std::string path = dir + "/routing.bin";
    std::FILE* file = std::fopen(path.c_str(), "wb");
    check(file != nullptr, "routing trace opens");
    if (file == nullptr) return;
    std::string error;
    const std::vector<std::int32_t> first_ids {5, 9};
    const std::vector<float> first_weights {0.5f, 0.25f};
    const std::vector<std::int32_t> second_ids {7};
    const std::vector<float> second_weights {0.0f};  // A verify-window record carries zero weights.
    check(append_routing_record(file, 3, first_ids, first_weights, error), "first routing record", error);
    check(append_routing_record(file, 4, second_ids, second_weights, error), "second routing record", error);
    check(!append_routing_record(file, 5, second_ids, first_weights, error), "ids and weights must pair up");
    // The rules tools/make_profile.py applies on read: a record has at least one id, the ids are distinct and
    // non-negative, and the weights are finite and non-negative. A rejected record writes no bytes.
    const std::vector<std::int32_t> none;
    const std::vector<std::int32_t> duplicated {5, 5};
    const std::vector<std::int32_t> negative_id {-1};
    const std::vector<float> one_weight {1.0f};
    const std::vector<float> two_weights {1.0f, 1.0f};
    const std::vector<float> negative_weight {-0.5f};
    const std::vector<float> not_finite {std::numeric_limits<float>::quiet_NaN()};
    check(!append_routing_record(file, 5, none, std::span<const float> {}, error), "a record needs at least one id");
    check(!append_routing_record(file, 5, duplicated, two_weights, error), "ids must be distinct");
    check(!append_routing_record(file, 5, negative_id, one_weight, error), "ids must be non-negative");
    check(!append_routing_record(file, 5, second_ids, negative_weight, error), "weights must be non-negative");
    check(!append_routing_record(file, 5, second_ids, not_finite, error), "weights must be finite");
    std::fclose(file);
    const Bytes bytes = read_file(path);
    check(bytes.size() == (8 + 2 * 4 + 2 * 4) + (8 + 1 * 4 + 1 * 4), "routing records are 8 + 8k bytes each",
          std::to_string(bytes.size()));
    check(int32_at(bytes, 0) == 3 && int32_at(bytes, 4) == 2, "first record: layer 3, k 2");
    check(int32_at(bytes, 8) == 5 && int32_at(bytes, 12) == 9, "first record: ids follow k");
    check(float_index(bytes, 16, 0) == 0.5f && float_index(bytes, 16, 1) == 0.25f, "first record: weights follow ids");
    const std::size_t second = 24;
    check(int32_at(bytes, second) == 4 && int32_at(bytes, second + 4) == 1, "second record: layer 4, k 1");
    check(int32_at(bytes, second + 8) == 7 && float_index(bytes, second + 12, 0) == 0.0f,
          "second record: id 7 and a zero weight");
}

// The bounds tools/make_profile.py reads: layers 0 to 47 and k from 1 to 64. A rejected record writes no bytes.
void test_routing_bounds(const std::string& dir) {
    const std::string path = dir + "/routing_bounds.bin";
    std::FILE* file = std::fopen(path.c_str(), "wb");
    check(file != nullptr, "routing bounds trace opens");
    if (file == nullptr) return;
    std::string error;
    const std::vector<std::int32_t> one_id {5};
    const std::vector<float> one_weight {1.0f};
    check(!append_routing_record(file, 48, one_id, one_weight, error) && !error.empty(), "layer 48 is rejected");
    check(!append_routing_record(file, -1, one_id, one_weight, error), "a negative layer is rejected");
    std::vector<std::int32_t> ids65(65);
    std::vector<float> weights65(65, 1.0f);
    for (std::size_t i = 0; i < ids65.size(); ++i) ids65[i] = static_cast<std::int32_t>(i);
    check(!append_routing_record(file, 3, ids65, weights65, error) && !error.empty(), "k of 65 is rejected");
    const std::vector<std::int32_t> ids64(ids65.begin(), ids65.begin() + 64);
    const std::vector<float> weights64(weights65.begin(), weights65.begin() + 64);
    check(append_routing_record(file, 47, ids64, weights64, error), "layer 47 with k of 64 is accepted", error);
    std::fclose(file);
    const Bytes bytes = read_file(path);
    check(bytes.size() == 8 + 64 * 8, "only the accepted record is written", std::to_string(bytes.size()));
    check(int32_at(bytes, 0) == 47 && int32_at(bytes, 4) == 64, "the accepted record is layer 47, k 64");
}

}  // namespace

int main() {
    test_logits_stride_rows();
    const std::string dir = temp_dir();
    if (dir.empty()) {
        check(false, "a temporary directory for the dump files");
    } else {
        test_logits_dump_layout(dir);
        test_short_logits_dump(dir);
        test_first_logits_layout(dir);
        test_residual_layout(dir);
        test_layers_layout(dir);
        test_half_layer_offsets(dir);
        test_routing_records(dir);
        test_routing_bounds(dir);
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }
    std::printf("dump_format_host_test: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
